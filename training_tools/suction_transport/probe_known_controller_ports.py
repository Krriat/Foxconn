"""Read-only ARP probes of known robot endpoints, without assigning IP addresses."""
import argparse
import json
import select
import socket
import struct
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--interfaces', nargs='+', required=True)
p.add_argument('--targets', nargs='+', default=['192.168.1.160', '192.168.2.160', '192.168.1.7'])
p.add_argument('--sender-ip', default='0.0.0.0', help='An IP already assigned to the selected interface')
p.add_argument('--target-mac', help='Optional previously verified MAC for unicast ARP')
a = p.parse_args()
for interface in a.interfaces:
    mac = bytes.fromhex((Path('/sys/class/net')/interface/'address').read_text().strip().replace(':', ''))
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0806))
    sock.bind((interface, 0))
    for address in a.targets:
        target = socket.inet_aton(address)
        # Sender 0.0.0.0 avoids advertising an IP binding on the wrong interface.
        dest = bytes.fromhex(a.target_mac.replace(':', '')) if a.target_mac else b'\xff'*6
        arp = struct.pack('!HHBBH6s4s6s4s', 1, 0x0800, 6, 4, 1, mac, socket.inet_aton(a.sender_ip),
                          dest if a.target_mac else b'\0'*6, target)
        frame = dest + mac + b'\x08\x06' + arp
        replies = []
        for attempt in range(3):
            sock.send(frame)
            deadline = time.monotonic()+.7
            while time.monotonic() < deadline:
                if not select.select([sock], [], [], max(0, deadline-time.monotonic()))[0]:
                    break
                data = sock.recv(4096)
                if len(data) >= 42 and data[20:22] == b'\0\x02' and data[28:32] == target:
                    replies.append(dict(attempt=attempt, mac=':'.join(f'{b:02x}' for b in data[22:28])))
        print(json.dumps(dict(interface=interface, target=address, replies=replies)), flush=True)
    sock.close()
