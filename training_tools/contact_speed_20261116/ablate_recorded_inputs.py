"""Replay exact recorded RGB/state, then swap inputs to locate sensitivity.

No ROS initialization or hardware calls. Swapped inputs can be inconsistent;
this diagnoses model sensitivity, not a physical causal proof.
"""
import itertools
import json
from pathlib import Path
import sqlite3
import sys

import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

ROOT=Path('/home/jj/code/ACT');APP=Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0,str(APP))
from inference_node import load_runtime

out=ROOT/'training_verification/contact_speed_20261116/input_ablation';out.mkdir(exist_ok=True)
cameras=('head','left_wrist','right_wrist');inputs={};chunks={}
for label,name in [('success','both_act_1p5'),('failure','both_act_1p5_repeat01')]:
    root=ROOT/'training_verification/speed_control_20261116'/name
    ev=[json.loads(l) for l in (root/'executor_events.jsonl').open()]
    cs=[r for r in ev if r['event']=='chunk' and r.get('policy_key')=='far']
    start=next(r['time_ns'] for r in ev if r['event']=='phase' and r['phase']=='RUNNING' and r['run_id']==cs[0]['run_id'])
    c=min(cs,key=lambda r:abs(r['observation_time_ns']-start-25e9));chunks[label]=c;images={}
    db=sqlite3.connect('file:'+str(next((root/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
    for camera in cameras:
        needed=c['source_stamps_ns'][camera]
        for data, in db.execute('select m.data from messages m join topics t on t.id=m.topic_id where t.name=?',
                               (f'/robot/sensors/cameras/{camera}/color/image_raw/compressed',)):
            m=deserialize_message(data,CompressedImage);stamp=int((m.header.stamp.sec+m.header.stamp.nanosec/1e9)*1e9)
            if stamp==needed:
                bgr=cv2.imdecode(np.frombuffer(m.data,np.uint8),cv2.IMREAD_COLOR)
                images[camera]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(out/f'{label}_{camera}.jpg'),bgr);break
        if camera not in images:raise RuntimeError('Exact source image missing')
    db.close();inputs[label]={'images':[images[k] for k in cameras],'state':np.array(c['observation_state'],np.float32)}
cfg=yaml.safe_load((APP/'config.yaml').read_text());checkpoint=Path(cfg['act_stage_checkpoints']['far']['path'])
_,predict,reset,_=load_runtime(cfg,checkpoint)
rows=[]
for state_source in ('success','failure'):
    for mask in itertools.product(('success','failure'),repeat=3):
        observation={'state':inputs[state_source]['state'],'images':[inputs[source]['images'][i] for i,source in enumerate(mask)]}
        reset();pred=predict(observation);state=observation['state']
        row={'state_source':state_source,'image_sources':dict(zip(cameras,mask)),
             'right_J2_state_deg':float(np.rad2deg(state[8])),
             'right_J2_next_2s_target_deg':float(np.rad2deg(pred[40,8])),
             'right_J2_next_2s_delta_deg':float(np.rad2deg(pred[40,8]-state[8])),
             'left_J5_next_2s_delta_deg':float(np.rad2deg(pred[40,4]-state[4])),
             'suction_prediction':pred[[2,40],14:].tolist()}
        if all(source==state_source for source in mask):
            expected=np.array(chunks[state_source]['actions'])
            difference=float(abs(pred-expected).max())
            row['replay_max_absolute_difference']=difference
            row['replay_max_joint_difference_deg']=float(np.rad2deg(abs(pred[:,:14]-expected[:,:14])).max())
            row['replay_max_suction_difference']=float(abs(pred[:,14:]-expected[:,14:]).max())
            # TF32/cuDNN kernel selection can cause small numeric differences.
            # Preserve and report them; require a much smaller error than the
            # ~14-degree behavior difference being investigated.
            assert row['replay_max_joint_difference_deg']<.05,row
            assert row['replay_max_suction_difference']<.001,row
        rows.append(row)
report={'checkpoint_sha256':chunks['success']['checkpoint_sha256'],'source_seconds':25,
        'scope':'Teacher-forced recorded-input ablation; swaps may create inconsistent image/state combinations. NOT physical causal proof or new task success.',
        'robot_motion_requested':False,'rows':rows}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(json.dumps(row),flush=True)
