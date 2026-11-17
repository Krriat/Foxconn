"""Offline ACT comparison on a successful raw bag and failed video/state.

Failed video has been JPEG-encoded again and recorded at 10Hz: it is explicitly
approximate input replay, not an exact source-image or physical counterfactual.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'training_verification/jetson_dual_act_20261116_171114/bundle'
CAMERAS=('head','left_wrist','right_wrist')


def prepare(output,failed):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    rows=[json.loads(l) for l in (failed/'live01/ros_events.jsonl').open()]
    chunks=[e['value'] for e in rows if e['topic']=='action_chunk']
    running=next(e['received_ns'] for e in rows if e['topic']=='status' and e['value']['phase']=='RUNNING')
    metadata=[json.loads(l) for l in (failed/'live01/video_frames.jsonl').open()]
    video=cv2.VideoCapture(str(failed/'live01/cameras.avi'))
    cases=[];images=[];states=[];recorded=[]
    for seconds in (0.,2.,4.,6.,8.,10.,12.,14.,16.,18.):
        c=min(chunks,key=lambda x:abs(x['observation_time_ns']-running-seconds*1e9))
        def error(frame):
            return max(abs(frame['image_stamps_ns'][cam]-c['source_stamps_ns'][cam]) for cam in CAMERAS)
        m=min(metadata,key=error);video.set(cv2.CAP_PROP_POS_FRAMES,m['frame']);ok,bgr=video.read()
        if not ok:raise RuntimeError('Cannot decode recorded video frame')
        rgb=[cv2.cvtColor(bgr[:,i*640:(i+1)*640],cv2.COLOR_BGR2RGB) for i in range(3)]
        images.append(rgb);states.append(c['observation_state']);recorded.append(c['actions'])
        cases.append(dict(source='failed_approximate_video',seconds=seconds,
                          max_image_stamp_error_ms=error(m)/1e6,exact_pixels=False))
    video.release()
    good=ROOT/'training_verification/speed_control_20261116/both_act_1p5'
    source=[json.loads(l) for l in (good/'executor_events.jsonl').open()]
    cs=[r for r in source if r['event']=='chunk' and r.get('policy_key')=='far']
    first=cs[0]['observation_time_ns']
    db=sqlite3.connect('file:'+str(next((good/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
    for seconds in (0.,2.,4.,6.,8.,10.,12.,14.,16.,18.):
        c=min(cs,key=lambda x:abs(x['observation_time_ns']-first-seconds*1e9));rgb=[]
        for camera in CAMERAS:
            wanted=c['source_stamps_ns'][camera];found=None
            for data, in db.execute('select m.data from messages m join topics t on m.topic_id=t.id '
                'where t.name=? and m.timestamp between ? and ?',
                (f'/robot/sensors/cameras/{camera}/color/image_raw/compressed',int(wanted-1e9),int(wanted+1e9))):
                msg=deserialize_message(data,CompressedImage)
                stamp=int((msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)*1e9)
                if abs(stamp-wanted)<=512:
                    found=cv2.cvtColor(cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB);break
            if found is None:raise RuntimeError('Exact successful source image not found: '+camera)
            rgb.append(found)
        images.append(rgb);states.append(c['observation_state']);recorded.append(c['actions'])
        cases.append(dict(source='successful_raw_bag',seconds=seconds,max_image_stamp_error_ms=0,exact_pixels=True))
    db.close()
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'inputs.npz',images=np.array(images),states=np.array(states,np.float32),recorded=np.array(recorded))
    (output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')


def evaluate(output):
    import torch
    sys.path.insert(0,str(ROOT/'training_tools/jetson_dual_act_20261116'))
    from jetson_policy_runtime import ActEngine
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    data=np.load(output/'inputs.npz');cases=json.loads((output/'cases.json').read_text());predictions={};reports=[]
    for name,original in (('current',ROOT/'model_outputs/act_far_current_recovery_20261113_03/checkpoints/001000/pretrained_model'),
                          ('original_success',ROOT/'model_outputs/act_suction_20261111_213453/checkpoints/014000/pretrained_model')):
        target=output/'models'/name;target.mkdir(parents=True,exist_ok=True);checks={}
        for filename in ('model.safetensors','config.json','policy_preprocessor_step_3_normalizer_processor.safetensors',
                         'policy_postprocessor_step_0_unnormalizer_processor.safetensors'):
            source=original/filename;link=target/filename
            if not link.exists():link.symlink_to(source)
            checks[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
        (target/'artifact_checksums.json').write_text(json.dumps(checks))
        engine=ActEngine(target,BUNDLE);predictions[name]=[]
        for i,(images,state) in enumerate(zip(data['images'],data['states'])):
            pred=engine.predict(dict(images=list(images),state=state));predictions[name].append(pred)
            reference=data['recorded'][i]
            reports.append(dict(model=name,case=cases[i],state_left_deg=np.rad2deg(state[:7]).tolist(),
                left_target_1s_deg=np.rad2deg(pred[20,:7]).tolist(),left_target_2s_deg=np.rad2deg(pred[40,:7]).tolist(),
                suction_0_1_2s=pred[[0,20,40],14].tolist(),
                can_reach_left_lift_milestone=bool(((np.rad2deg(pred[:,1])>=47)&(np.rad2deg(pred[:,0])<=28)).any()),
                replay_max_joint_difference_deg=float(np.rad2deg(abs(pred[:,:14]-reference[:,:14])).max()),
                replay_max_suction_difference=float(abs(pred[:,14:]-reference[:,14:]).max())))
        del engine;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(output/'predictions.npz',**predictions)
    report=dict(robot_motion_commands=0,input_limits='Failed video pixels approximate; successful bag pixels exact. '
        'Offline model sensitivity comparison, not proof of physical causality or success.',rows=reports)
    (output/'model_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in reports:
        print(json.dumps(dict(model=row['model'],source=row['case']['source'],s=row['case']['seconds'],
            next2s_left_J1_J2_J3_J6=[round(row['left_target_2s_deg'][j],2) for j in (0,1,2,5)],
            vacuum=[round(x,3) for x in row['suction_0_1_2s']],lift=row['can_reach_left_lift_milestone'],
            replay_joint_error_deg=round(row['replay_max_joint_difference_deg'],4))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--failed',type=Path);p.add_argument('--prepare',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare(a.output,a.failed)
    else:evaluate(a.output)


if __name__=='__main__':main()
