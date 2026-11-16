"""Extract a verified successful live demonstration for recovery training.

The 19:01 recording contains RGBs; the later 21:05 recording does not.
Targets are executed future joint commands and measured suction, not the
model's guessed future chunks. This is training data, not a held-out test.
"""
import argparse,json,sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
ROOT=Path('/home/jj/code/ACT')
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,default=ROOT/'training_verification/restore_far_20261112/camera_fixed_live')
p.add_argument('--output',type=Path,default=ROOT/'prepared_data/far_success_recovery_20261113')
a=p.parse_args();SRC=a.source.resolve();OUT=a.output.resolve()
OUT.mkdir(parents=True,exist_ok=False)
event_path=SRC/('executor_events.jsonl' if (SRC/'executor_events.jsonl').exists() else 'run_events.jsonl')
events=[json.loads(line) for line in event_path.open()]
chunks=[e for e in events if e['event']=='chunk'][::2]
assert len({c['run_id'] for c in chunks})==1
assert all(c.get('policy_key','far') in ('far','default') for c in chunks)
traces=[e['trace'] for e in events if e['event']=='servo_trace' and e['trace'].get('active') and e['trace'].get('hardware_commands_sent')]
t0=chunks[0]['observation_time_ns']
times=(np.array([t['selection_time_ns'] for t in traces],dtype=np.int64)-t0)/1e9
commands=np.array([t['command'] for t in traces])
db=sqlite3.connect('file:'+str(next((SRC/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
cameras=('head','left_wrist','right_wrist')
found={c:set() for c in cameras}
for c in cameras:
    dest=OUT/c;dest.mkdir()
    needed={x['source_stamps_ns'][c] for x in chunks}
    for data, in db.execute('select m.data from messages m join topics t on t.id=m.topic_id where t.name=?',(f'/robot/sensors/cameras/{c}/color/image_raw/compressed',)):
        m=deserialize_message(data,CompressedImage)
        stamp=int((m.header.stamp.sec+m.header.stamp.nanosec/1e9)*1e9)
        if stamp in needed:
            (dest/f'{stamp}.jpg').write_bytes(bytes(m.data));found[c].add(stamp)
suction=[]
for side in ('left','right'):
    entries=[((stamp-t0)/1e9,float(deserialize_message(data,Bool).data)) for stamp,data in db.execute(
        'select m.timestamp,m.data from messages m join topics t on t.id=m.topic_id where t.name=? order by m.timestamp',(f'/robot/suction/{side}/state',))]
    suction.append(np.array(entries))
states=[];actions=[];pads=[];paths=[];sequences=[]
for c in chunks:
    if not all(c['source_stamps_ns'][k] in found[k] for k in cameras):continue
    target_times=(c['observation_time_ns']-t0)/1e9+np.arange(50)*.05
    joints=np.stack([np.interp(target_times,times,commands[:,j]) for j in range(14)],axis=1)
    io=np.stack([entries[np.clip(np.searchsorted(entries[:,0],target_times,side='right')-1,0,len(entries)-1),1] for entries in suction],axis=1)
    states.append(c['observation_state']);actions.append(np.concatenate([joints,io],axis=1));pads.append(target_times>times[-1])
    paths.append([f'{k}/{c["source_stamps_ns"][k]}.jpg' for k in cameras]);sequences.append(c['sequence'])
np.savez_compressed(OUT/'data.npz',state=np.array(states,np.float32),action=np.array(actions,np.float32),action_is_pad=np.array(pads,bool),images=np.array(paths),sequence=np.array(sequences))
manifest=dict(source=str(SRC),run_id=chunks[0]['run_id'],samples=len(states),duration_seconds=float(times[-1]),
    source_date=datetime.fromtimestamp(t0/1e9,ZoneInfo('Asia/Shanghai')).isoformat(),
    source_diagnostic_mode=chunks[0].get('diagnostic_mode'),
    role='Training-only additional successful demonstration; not an independent validation set',
    target='Executed joint commands interpolated at image timestamp + 20 Hz horizons; suction readback zero-order hold using bag receipt time (Bool lacks source timestamps)',
    normalization='Unchanged original ACT train-set normalizers',cameras=cameras)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
