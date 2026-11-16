"""Send ARP queries for the known right-arm IP on the two attached wired interfaces."""
import json
import select
import socket
import struct
import time
from pathlib import Path

target=socket.inet_aton('192.168.2.160')
for interface,source in [('enP8p1s0','192.168.2.10'),('enx00e04c2b7928','192.168.50.10')]:
    mac=bytes.fromhex((Path('/sys/class/net')/interface/'address').read_text().strip().replace(':',''))
    sock=socket.socket(socket.AF_PACKET,socket.SOCK_RAW,socket.htons(0x0806))
    sock.bind((interface,0))
    arp=struct.pack('!HHBBH6s4s6s4s',1,0x0800,6,4,1,mac,socket.inet_aton(source),b'\x00'*6,target)
    packet=b'\xff'*6+mac+b'\x08\x06'+arp
    replies=[]
    for attempt in range(3):
        sock.send(packet)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            if not select.select([sock],[],[],max(0,deadline-time.monotonic()))[0]:break
            data=sock.recv(4096)
            if len(data)>=42 and data[12:14]==b'\x08\x06' and data[20:22]==b'\x00\x02' and data[28:32]==target:
                replies.append(dict(attempt=attempt,mac=':'.join(f'{b:02x}' for b in data[22:28])))
    sock.close()
    print(json.dumps(dict(interface=interface,source=source,target='192.168.2.160',replies=replies)),flush=True)
