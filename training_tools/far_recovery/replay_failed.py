"""Teacher-forced replay of recorded failed far inputs; no hardware access."""
import copy, hashlib, json, sqlite3, sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
ROOT = Path('/home/jj/code/ACT')
OUT = ROOT / 'training_verification/far_recovery_20261113'
BASE = ROOT / 'training_verification/manual_disconnect_20261113/live_gc_fixed_02'
APP = Path.home() / '.local/share/robot-policy-streams'
sys.path.insert(0,str(APP))
from inference_node import load_runtime
from protocol import joint_bounds
chunks = [x for line in (ROOT/'training_verification/success_2105_comparison_20261113/current_far.jsonl').open()
          if (x:=json.loads(line)).get('event')=='chunk'][::20]
cameras = ('head','left_wrist','right_wrist')
required = {key:{c['source_stamps_ns'][key] for c in chunks} for key in cameras}
images = {key:{} for key in cameras}
db = sqlite3.connect('file:'+str(next((BASE/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
for camera in cameras:
    topic=f'/robot/sensors/cameras/{camera}/color/image_raw/compressed'
    for raw, in db.execute('select m.data from messages m join topics t on m.topic_id=t.id where t.name=?',(topic,)):
        msg=deserialize_message(raw,CompressedImage)
        stamp=int((msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)*1e9)
        if stamp in required[camera]:
            images[camera][stamp]=cv2.cvtColor(cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
db.close()
cfg=yaml.safe_load((APP/'config.yaml').read_text())
path=ROOT/'model_outputs/act_far_recovery_20261113_01/checkpoints/000500/pretrained_model'
cfg['checkpoint_sha256']['act']=hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()
_,predict,reset,_=load_runtime(cfg,path)
lo,hi=map(np.array,joint_bounds(cfg['joint_limit_margin_rad'],cfg.get('joint_upper_limits_rad')))
rows=[]
for c in chunks:
    if not all(c['source_stamps_ns'][key] in images[key] for key in cameras):continue
    state=np.array(c['observation_state'],np.float32)
    actual=predict(dict(images=[images[key][c['source_stamps_ns'][key]] for key in cameras],state=state))
    reference=np.array(c['actions'],np.float32)
    rows.append(dict(seconds=(c['observation_time_ns']-chunks[0]['observation_time_ns'])/1e9,
        original_suction=float(reference[2,14]),candidate_suction=float(actual[2,14]),
        original_target_deg=np.rad2deg(reference[2,:14]).tolist(),
        candidate_target_deg=np.rad2deg(actual[2,:14]).tolist(),
        observed_deg=np.rad2deg(state[:14]).tolist(),
        violations=int(((actual[:,:14]<lo)|(actual[:,:14]>hi)).sum())))
result=dict(scope='Recorded inputs fixed to failed trajectory; this does not predict a new closed-loop rollout.',
    samples=len(rows),total_violations=sum(r['violations'] for r in rows),rows=rows)
(OUT/'failed_input_candidate_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
for r in rows[:30]:print(r['seconds'],r['original_suction'],r['candidate_suction'])
