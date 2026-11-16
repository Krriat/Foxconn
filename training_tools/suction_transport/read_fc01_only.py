"""Bounded FC01-only relay telemetry measurement; never publishes ROS or writes coils."""
import argparse
import fcntl
import json
import os
import signal
import socket
import statistics
import struct
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--port',type=int,default=20108)
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    lock=os.open(f'/run/user/{os.getuid()}/robot-m050.lock',os.O_CREAT|os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupt(*_):raise KeyboardInterrupt
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,interrupt)
    sock=None;tid=30000;results=[]
    with a.output.open('x',buffering=1) as log:
        def event(**row):log.write(json.dumps(dict(time_ns=time.time_ns(),**row),separators=(',',':'))+'\n')
        event(event='started',host=a.host,source=a.source,port=a.port,seconds=a.seconds,
              function=1,write_commands=False,publishes=False,hz=10,connection_reuse=True)
        end=time.monotonic()+a.seconds
        try:
            while time.monotonic()<end:
                start=time.monotonic();deadline=start+1.0
                row=dict(event='read',start_ns=time.time_ns())
                try:
                    if sock is None:
                        connect_start=time.monotonic()
                        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                        sock.bind((a.source,0));sock.settimeout(max(.001,deadline-time.monotonic()))
                        sock.connect((a.host,a.port))
                        row['connect_ms']=(time.monotonic()-connect_start)*1000
                    tid=(tid+1)&65535
                    # Unit 1, FC01, first coil 0, two coils. No write function exists here.
                    request=struct.pack('>HHHBBHH',tid,0,6,1,1,0,2)
                    row.update(tid=tid,request_ns=time.time_ns())
                    sent=time.monotonic();sock.settimeout(max(.001,deadline-time.monotonic()))
                    sock.sendall(request)
                    data=b''
                    while len(data)<10:
                        sock.settimeout(max(.001,deadline-time.monotonic()))
                        part=sock.recv(10-len(data))
                        if not part:raise RuntimeError('Remote closed connection')
                        data+=part
                    if data[:2]!=struct.pack('>H',tid) or data[2:9]!=bytes([0,0,0,4,1,1,1]):
                        raise RuntimeError('Invalid FC01 reply: '+data.hex())
                    row.update(ok=True,response_ms=(time.monotonic()-sent)*1000,
                               outputs=[bool(data[9]&1),bool(data[9]&2)])
                except (OSError,RuntimeError) as error:
                    row.update(ok=False,error=str(error))
                    if sock:sock.close();sock=None
                row.update(total_ms=(time.monotonic()-start)*1000)
                event(**row);results.append(row)
                time.sleep(max(0,.1-(time.monotonic()-start)) if row['ok'] else .5)
        finally:
            if sock:sock.close()
            values=[r['total_ms'] for r in results if r.get('ok')]
            summary=dict(event='finished',reads=len(results),replies=len(values),
                         failures=len(results)-len(values),median_ms=statistics.median(values) if values else None,
                         max_ms=max(values,default=None),over300ms=sum(v>300 for v in values))
            event(**summary);print(json.dumps(summary),flush=True)
            fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)


if __name__=='__main__':main()
