"""Passive TCP metadata capture restricted to the authorized relay endpoint."""
import argparse
import json
import select
import socket
import struct
import time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--interface',default='wlP1p1s0')
p.add_argument('--host',default='172.16.13.167')
p.add_argument('--port',type=int,default=20108)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=1200)
a=p.parse_args()
relay=socket.inet_aton(a.host)
s=socket.socket(socket.AF_PACKET,socket.SOCK_RAW,socket.htons(0x0003))
s.bind((a.interface,0))
s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)
# SO_TIMESTAMPNS_OLD=35 on this Linux/aarch64 host (asm-generic/socket.h).
s.setsockopt(socket.SOL_SOCKET,35,1)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x',buffering=1) as f:
    f.write(json.dumps(dict(event='capture_started',time_ns=time.time_ns(),
                           passive=True,relay=f'{a.host}:{a.port}',interface=a.interface))+'\n')
    print('Capture started; passive relay TCP only',flush=True)
    end=time.monotonic()+a.seconds
    while time.monotonic()<end:
        if not select.select([s],[],[],.5)[0]:continue
        data,ancillary,_,_=s.recvmsg(65535,socket.CMSG_SPACE(16))
        received=time.time_ns()
        kernel_ns=None
        for level,kind,value in ancillary:
            if level==socket.SOL_SOCKET and kind==35 and len(value)>=16:
                seconds,nanoseconds=struct.unpack('=qq',value[:16])
                kernel_ns=seconds*10**9+nanoseconds
        if len(data)<54 or data[12:14]!=b'\x08\x00':continue
        ip=data[14:]
        ihl=(ip[0]&15)*4
        if ip[9]!=6 or relay not in (ip[12:16],ip[16:20]):continue
        tcp=ip[ihl:]
        if len(tcp)<20:continue
        src_port,dst_port,seq,ack=struct.unpack('!HHII',tcp[:12])
        if a.port not in (src_port,dst_port):continue
        thl=(tcp[12]>>4)*4
        length=struct.unpack('!H',ip[2:4])[0]-ihl-thl
        payload=tcp[thl:thl+max(0,length)]
        row=dict(time_ns=kernel_ns or received,userspace_time_ns=received,
                 capture_scheduling_ms=None if kernel_ns is None else (received-kernel_ns)/1e6,
                 src=socket.inet_ntoa(ip[12:16]),
                 dst=socket.inet_ntoa(ip[16:20]),src_port=src_port,dst_port=dst_port,
                 seq=seq,ack=ack,flags=tcp[13],payload_len=length,
                 payload_hex=payload[:64].hex())
        f.write(json.dumps(row,separators=(',',':'))+'\n')
    f.write(json.dumps(dict(event='capture_finished',time_ns=time.time_ns()))+'\n')
s.close()
