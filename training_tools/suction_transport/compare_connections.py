"""Isolated FC01-only A/B test; restore the production telemetry service on exit."""
import argparse
import fcntl
import json
import os
import signal
import socket
import statistics
import struct
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool,String


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase-seconds',type=float,default=60)
    a=p.parse_args()
    rclpy.init();n=Node('m050_connection_comparison');state={}
    def policy(m):state['policy']=(json.loads(m.data),time.monotonic())
    subs=[n.create_subscription(String,'/robot/policy/status',policy,qos_profile_sensor_data)]
    for side in ('left','right'):
        subs.append(n.create_subscription(Bool,f'/robot/suction/{side}/state',
            lambda m,s=side:state.update({s:(bool(m.data),time.monotonic())}),qos_profile_sensor_data))
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=min(.02,end-time.monotonic()))
    def idle():
        v,t=state.get('policy',({},0))
        return v.get('phase')=='IDLE' and time.monotonic()-t<.5
    discovery_deadline=time.monotonic()+10
    while True:
        spin(.1)
        if idle() and all(s in state and not state[s][0] and
                          time.monotonic()-state[s][1]<.5 for s in ('left','right')):
            break
        if time.monotonic()>discovery_deadline:
            raise RuntimeError('Fresh IDLE and both OFF required; received='+str(list(state)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def interrupt(*_):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupt)
    fd=os.open(f'/run/user/{os.getuid()}/robot-m050.lock',os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW)
    stopped=False;sock=None;tid=65529
    with a.output.open('x',buffering=1) as log:
        def event(**x):
            log.write(json.dumps(dict(time_ns=time.time_ns(),**x),separators=(',',':'))+'\n')
        try:
            stopped=True
            subprocess.run(['systemctl','--user','stop','robot-hardware-state.service'],check=True,timeout=15)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            event(event='isolated',outputs_modified=False,motion_requested=False)
            for phase,mode in enumerate(('short','persistent','short','persistent')):
                if sock:sock.close();sock=None
                end=time.monotonic()+a.phase_seconds
                rows=[];event(event='phase_start',phase=phase,mode=mode)
                print(json.dumps(dict(event='phase_start',phase=phase,mode=mode)),flush=True)
                while time.monotonic()<end:
                    spin(.001)
                    if not idle():raise RuntimeError('Executor is no longer fresh IDLE')
                    started=time.monotonic();deadline=started+1.0
                    row=dict(event='read',phase=phase,mode=mode,start_ns=time.time_ns())
                    try:
                        if sock is None:
                            t=time.monotonic();sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                            sock.settimeout(max(.001,deadline-time.monotonic()))
                            sock.connect(('172.16.13.167',20108))
                            row['connect_ms']=(time.monotonic()-t)*1000
                        tid=(tid+1)&65535;row['tid']=tid
                        request=struct.pack('>HHHBBHH',tid,0,6,1,1,0,2)
                        row['request_ns']=time.time_ns();t=time.monotonic()
                        sock.settimeout(max(.001,deadline-time.monotonic()));sock.sendall(request)
                        data=b''
                        while len(data)<10:
                            sock.settimeout(max(.001,deadline-time.monotonic()))
                            part=sock.recv(10-len(data))
                            if not part:raise RuntimeError('Relay closed socket')
                            data+=part
                        if data[:2]!=struct.pack('>H',tid) or data[2:9]!=bytes([0,0,0,4,1,1,1]):
                            raise RuntimeError('Invalid FC01 response: '+data.hex())
                        row.update(ok=True,response_ms=(time.monotonic()-t)*1000,
                                   outputs=[bool(data[9]&1),bool(data[9]&2)])
                        if any(row['outputs']):raise RuntimeError('External suction activation observed')
                    except (OSError,RuntimeError) as error:
                        row.update(ok=False,error=str(error))
                        if sock:sock.close();sock=None
                        if 'External suction' in str(error):raise
                    finally:
                        row.update(finish_ns=time.time_ns(),total_ms=(time.monotonic()-started)*1000)
                        event(**row);rows.append(row)
                        if mode=='short' and sock:sock.close();sock=None
                    spin(max(0,.1-(time.monotonic()-started)) if row.get('ok') else .5)
                values=[r['total_ms'] for r in rows]
                summary=dict(event='phase_finished',phase=phase,mode=mode,reads=len(rows),
                             failures=sum(not r.get('ok') for r in rows),median_ms=statistics.median(values),
                             max_ms=max(values),over_300ms=sum(v>300 for v in values))
                event(**summary);print(json.dumps(summary),flush=True)
        finally:
            if sock:sock.close()
            fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
            if stopped:
                restore=subprocess.run([str(Path.home()/'.local/share/robot-state-streams/start.sh')],timeout=20)
                event(event='production_telemetry_restore',returncode=restore.returncode)
            n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
