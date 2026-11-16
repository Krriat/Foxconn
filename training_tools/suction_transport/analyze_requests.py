"""Match serialized Modbus requests/replies and TCP retransmissions, including keepalive."""
import argparse
import collections
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('capture',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
live={};connections=[];scheduling=[]
for line in a.capture.open():
    try:x=json.loads(line)
    except ValueError:continue
    if 'src' not in x:continue
    if x.get('capture_scheduling_ms') is not None:scheduling.append(x['capture_scheduling_ms'])
    out=x['dst_port']==20108;port=x['src_port'] if out else x['dst_port']
    if out and x['flags']&2:
        if port not in live or live[port][0]['seq']!=x['seq']:
            live[port]=[];connections.append(live[port])
    if port not in live:
        # A reused connection can predate capture; still match its new requests.
        live[port]=[];connections.append(live[port])
    live[port].append(x)
rows=[]
for c in connections:
    requests=collections.OrderedDict()
    incoming=[x for x in c if x['src_port']==20108]
    synack=next((x for x in incoming if x['flags']&18==18),None)
    syn_seen=c[0]['dst_port']==20108 and bool(c[0]['flags']&2)
    for x in c:
        if x['dst_port']!=20108 or x['payload_len']<=0:continue
        data=bytes.fromhex(x['payload_hex'])
        # These captures use serialized FC01 requests (12 bytes) and replies (10 bytes).
        # Flag unexpected segmentation/coalescing rather than silently calling it packet loss.
        if len(data)!=12 or data[2:7]!=bytes([0,0,0,6,1]):continue
        key=(x['seq'],x['payload_hex'])
        requests.setdefault(key,[]).append(x)
    for index,(key,packets) in enumerate(requests.items()):
        first=packets[0];data=bytes.fromhex(first['payload_hex']);tid=int.from_bytes(data[:2],'big')
        following=[x for x in incoming if x['time_ns']>=first['time_ns']]
        responses=[x for x in following if x['payload_len']==10 and
                   x['payload_hex'].startswith(data[:2].hex())]
        expected_ack=(first['seq']+12)%(2**32)
        acknowledged=next((x for x in following if x['ack']==expected_ack),None)
        row=dict(connection_start_ns=c[0]['time_ns'] if syn_seen else None,
                 capture_began_mid_connection=not syn_seen,client_port=first['src_port'],
                 request_ns=first['time_ns'],tid=tid,function=data[7],tcp_seq=first['seq'],
                 retransmissions=len(packets)-1,
                 retransmit_after_ms=[(x['time_ns']-first['time_ns'])/1e6 for x in packets[1:]])
        if syn_seen and synack:row['connection_ms']=(synack['time_ns']-c[0]['time_ns'])/1e6
        if acknowledged:row['ack_ms']=(acknowledged['time_ns']-first['time_ns'])/1e6
        if responses:
            row['response_ms']=(responses[0]['time_ns']-first['time_ns'])/1e6
            row['response_hex']=responses[0]['payload_hex']
            row['read_total_ms']=(responses[0]['time_ns']-(c[0]['time_ns'] if index==0 and syn_seen else first['time_ns']))/1e6
        else:row['reply_not_seen']=True
        rows.append(row)
values=[x['response_ms'] for x in rows if 'response_ms' in x]
result=dict(connections=len(connections),requests=len(rows),responses=len(values),
            request_transactions_with_retransmission=sum(x['retransmissions']>0 for x in rows),
            retransmitted_request_packets=sum(x['retransmissions'] for x in rows),
            retransmission_cases=[x for x in rows if x['retransmissions']],
            slow_over_300ms=[x for x in rows if x.get('read_total_ms',0)>300],
            median_response_ms=statistics.median(values) if values else None,
            max_response_ms=max(values,default=0),
            capture_scheduling_max_ms=max(scheduling,default=None),
            scope='Sender-side TCP observation. Retransmission does not distinguish lost request, lost ACK, or excessive delay; this is not an 802.11 frame-loss measurement.',
            all_requests=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_requests'},indent=2))
