"""Bounded local ARP inventory to locate the previously known relay MAC."""
import argparse
import ipaddress
import json
import select
import socket
import struct
import time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--interface',required=True)
p.add_argument('--source',required=True)
p.add_argument('--network',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--match-mac',default='38:f4:06:1d:66:25')
a=p.parse_args()
network=ipaddress.ip_network(a.network)
if network.num_addresses>512:
    p.error('Maximum 512 addresses per local inventory')
mac=bytes.fromhex((Path('/sys/class/net')/a.interface/'address').read_text().strip().replace(':',''))
sock=socket.socket(socket.AF_PACKET,socket.SOCK_RAW,socket.htons(0x0806))
sock.bind((a.interface,0))
hosts=iter(str(h) for h in network.hosts() if str(h)!=a.source)
next_send=time.monotonic()
finished=None
results={}
sent=0
while finished is None or time.monotonic()-finished<3:
    now=time.monotonic()
    if finished is None and now>=next_send:
        address=next(hosts,None)
        if address is None:
            finished=now
        else:
            arp=struct.pack('!HHBBH6s4s6s4s',1,0x0800,6,4,1,mac,socket.inet_aton(a.source),b'\0'*6,socket.inet_aton(address))
            sock.send(b'\xff'*6+mac+b'\x08\x06'+arp)
            sent+=1
            next_send=now+.05
    if select.select([sock],[],[],.01)[0]:
        data=sock.recv(4096)
        if len(data)>=42 and data[20:22]==b'\0\x02':
            ip=socket.inet_ntoa(data[28:32])
            if ipaddress.ip_address(ip) in network:
                results[ip]=':'.join(f'{b:02x}' for b in data[22:28])
sock.close()
r=dict(interface=a.interface,source=a.source,network=a.network,arp_requests=sent,
       replies=results,matches={k:v for k,v in results.items() if v==a.match_mac.lower()})
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(interface=a.interface,replies=len(results),matches=r['matches'])),flush=True)
