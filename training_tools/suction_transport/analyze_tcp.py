"""Analyze the filtered passive TCP log; no network or hardware access."""
import argparse
import collections
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('capture',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
connections=[]; live={}
for line in a.capture.open():
    try:r=json.loads(line)
    except ValueError:continue
    if 'src' not in r:continue
    outbound=r['dst_port']==20108
    port=r['src_port'] if outbound else r['dst_port']
    if outbound and r['flags']&2:
        if port not in live or live[port][0]['seq']!=r['seq']:
            live[port]=[];connections.append(live[port])
    if port in live:live[port].append(r)
rows=[]
for packets in connections:
    start=packets[0]['time_ns']
    out=[x for x in packets if x['dst_port']==20108]
    incoming=[x for x in packets if x['src_port']==20108]
    syn=[x for x in out if x['flags']&2]
    synack=[x for x in incoming if x['flags']&18==18]
    requests=[x for x in out if x['payload_len']>0]
    responses=[x for x in incoming if x['payload_len']>0]
    row=dict(start_ns=start,client_port=packets[0]['src_port'],syn_count=len(syn),
             request_packets=len(requests),response_packets=len(responses))
    if synack:row['connect_ms']=(synack[0]['time_ns']-start)/1e6
    if requests:
        req=requests[0]
        data=bytes.fromhex(req['payload_hex'])
        row.update(request_ns=req['time_ns'],tid=int.from_bytes(data[:2],'big'),
                   function=data[7] if len(data)>7 else None)
        acked=[x for x in incoming if x['time_ns']>=req['time_ns'] and
               x['ack']==(req['seq']+req['payload_len'])%(2**32)]
        if acked:row['request_ack_ms']=(acked[0]['time_ns']-req['time_ns'])/1e6
        if responses:
            row['response_ms']=(responses[0]['time_ns']-req['time_ns'])/1e6
            row['operation_ms']=(responses[-1]['time_ns']-start)/1e6
        counts=collections.Counter(x['seq'] for x in requests)
        row['request_retransmissions']=sum(n-1 for n in counts.values())
    row['last_packet_ns']=packets[-1]['time_ns']
    rows.append(row)
stats={}
for key in ('connect_ms','request_ack_ms','response_ms','operation_ms'):
    values=sorted(x[key] for x in rows if key in x)
    if values:stats[key]=dict(n=len(values),median=statistics.median(values),
                             p99=values[min(len(values)-1,int(len(values)*.99))],max=max(values))
slow=[x for x in rows if x.get('operation_ms',0)>150 or x.get('connect_ms',0)>150 or
      x.get('request_retransmissions',0)>0 or x['syn_count']>1 or
      ('request_ns' in x and 'response_ms' not in x and x['last_packet_ns']-x['start_ns']>500_000_000)]
result=dict(connections=len(rows),stats=stats,slow=slow,all_connections=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='all_connections'},indent=2))
