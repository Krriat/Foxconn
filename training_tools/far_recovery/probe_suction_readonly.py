"""Read-only M050 timing/TID probes, serialized with the deployed IO lock."""
import argparse
import fcntl
import json
import os
import socket
import struct
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, String


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=120)
    a = p.parse_args()
    rclpy.init()
    node = Node('suction_readonly_probe')
    current = {}

    def on_status(m):
        current['policy'] = (json.loads(m.data), time.monotonic())

    subs = [node.create_subscription(String, '/robot/policy/status',
                                    on_status, qos_profile_sensor_data)]
    for side in ('left', 'right'):
        subs.append(node.create_subscription(Bool, f'/robot/suction/{side}/state',
            lambda m, s=side: current.update({s:(m.data,time.monotonic())}),
            qos_profile_sensor_data))

    def idle(poll=True):
        if poll:
            until = time.monotonic()+.06
            while time.monotonic()<until:
                rclpy.spin_once(node, timeout_sec=.005)
        v, t = current.get('policy', ({}, 0))
        return (v.get('phase') == 'IDLE' and time.monotonic()-t < .5 and
                all(s in current and not current[s][0] and
                    time.monotonic()-current[s][1] < .5 for s in ('left','right')))

    deadline=time.monotonic()+5
    while not idle():
        if time.monotonic()>deadline:
            raise RuntimeError('Idle robot and two fresh OFF states required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(f'/run/user/{os.getuid()}/robot-m050.lock',os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW)
    tids=[65534,65535,0,1,2]*3+list(range(100,100+a.count))
    with a.output.open('x',buffering=1) as log:
        for index,tid in enumerate(tids):
            if not idle():
                log.write(json.dumps(dict(event='aborted_not_idle_or_stale',time_ns=time.time_ns()))+'\n')
                break
            row=dict(event='FC01_read_only',tid=tid,index=index,start_ns=time.time_ns())
            started=time.monotonic()
            locked=False
            try:
                while not locked:
                    try:
                        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
                    except BlockingIOError:
                        if time.monotonic()-started>.5:raise TimeoutError('IO lock busy >500 ms')
                        time.sleep(.002)
                row['lock_ms']=(time.monotonic()-started)*1000
                if not idle(poll=False):raise RuntimeError('Robot no longer idle; no hardware query sent')
                with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    t=time.monotonic();sock.connect(('172.16.13.167',20108))
                    row['connect_ms']=(time.monotonic()-t)*1000
                    # FC01 reads two output-coil states. No write function is sent.
                    request=struct.pack('>HHHBBHH',tid,0,6,1,1,0,2)
                    t=time.monotonic();row['request_ns']=time.time_ns();sock.sendall(request)
                    data=b''
                    while len(data)<10:
                        part=sock.recv(10-len(data))
                        if not part:raise RuntimeError('EOF before complete read response')
                        if not data:row['first_response_ms']=(time.monotonic()-t)*1000
                        data+=part
                    row['response_ms']=(time.monotonic()-t)*1000
                    row['response_hex']=data.hex()
                    if data[:2]!=struct.pack('>H',tid) or data[2:9]!=bytes([0,0,0,4,1,1,1]):
                        raise RuntimeError('Unexpected read response header')
                    row['state']=[bool(data[9]&1),bool(data[9]&2)]
                    row['ok']=True
            except Exception as error:
                row.update(ok=False,error=str(error))
            finally:
                row['finish_ns']=time.time_ns()
                row['total_ms']=(time.monotonic()-started)*1000
                if locked:fcntl.flock(fd,fcntl.LOCK_UN)
                log.write(json.dumps(row,separators=(',',':'))+'\n')
            # At most 2 extra reads per second, only while idle and both OFF.
            end=time.monotonic()+.5
            while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.02)
    os.close(fd)
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
