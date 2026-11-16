#!/usr/bin/env python3
"""Offline checkpoint screening on a fixed contact audit set; no ROS node."""
import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training_tools'))
import test_ros_models as runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    torch=runtime.torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    out=ROOT/'training_verification/restore_far_20261112/near_contact_audit'
    reference=json.loads((out/'contact_prediction_audit.json').read_text())
    samples=[]
    for row in reference['rows']:
        ep=ROOT/'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz'/row['episode']
        data=np.load(ep/'trajectory.npz');index=row['index']
        images=[cv2.cvtColor(cv2.imread(str(ep/f'images/{camera}/{index:06d}.png')),cv2.COLOR_BGR2RGB)
                for camera in ('head','left_wrist','right_wrist')]
        samples.append(dict(meta=row,state=data['state'][index],images=images,
                            targets=data['action'][index:index+50]))
    meta=runtime.LeRobotDatasetMetadata('local/dual_arm_suction_train',root=ROOT/'prepared_data/suction_20hz_v1/train')
    base=ROOT/'model_outputs/act_suction_20261111_213453/checkpoints'
    cases=[('act_'+step,base/step/'pretrained_model','act') for step in
           ('005000','010000','014000','020000','030000','040000','050000','060000')]
    cases.append(('interact_best',ROOT/'model_outputs/interact_suction_20261111_234920/checkpoints/best/pretrained_model','interact'))
    if args.candidate:
        cases=[('act_014000',base/'014000/pretrained_model','act'),
               ('candidate',args.candidate.resolve(),'act')]
    prefix=args.output_prefix or out/('candidate_screen' if args.candidate else 'checkpoint_screen')
    reports=[];predictions={}
    for name,path,model in cases:
        if model=='act':policy,pre,post,_=runtime.load_policy(path,meta,'cuda')
        else:
            policy=runtime.InterACT3DPolicy.from_pretrained(path).cuda().eval()
            assert not policy.config.use_point_cloud
            stats=runtime.NormalizationStats.load(path/'normalization.json')
        errors=[];outputs=[]
        for sample in samples:
            with torch.inference_mode():
                if model=='act':
                    policy.reset();pre.reset();post.reset()
                    raw={'observation.state':sample['state']}
                    raw.update({f'observation.images.{key}':value for key,value in
                                zip(('head','left_wrist','right_wrist'),sample['images'])})
                    batch=pre(runtime.prepare_observation_for_inference(raw,torch.device('cuda'),task='',robot_type=''))
                    p=post(policy.predict_action_chunk(batch)).float().cpu().numpy()[0]
                else:
                    images,points,state=runtime.preprocess_observation(*sample['images'],state=sample['state'],stats=stats,
                                             image_size=policy.config.image_size,device='cuda')
                    p=runtime.denormalize_actions(policy.predict_action_chunk(images,points,state)[0],stats).float().cpu().numpy()
            outputs.append(p)
            err={str(h):float(np.rad2deg(abs(p[h,:7]-sample['targets'][h,:7])).mean()) for h in (0,1,5,10,20)}
            errors.append(dict(episode=sample['meta']['episode'],index=sample['meta']['index'],split=sample['meta']['split'],
                mae_left_deg=err,suction_wrong_h10=bool((p[10,14]>=.5)!=(sample['targets'][10,14]>=.5))))
        predictions[name]=np.stack(outputs)
        row=dict(name=name,checkpoint=str(path),cases=errors)
        for group in ('train','val'):
            selected=[x for x in errors if x['split']==group]
            row[group]=dict(median_left_mae_deg={str(h):float(np.median([x['mae_left_deg'][str(h)] for x in selected])) for h in (0,1,5,10,20)},
                           suction_h10_wrong=sum(x['suction_wrong_h10'] for x in selected),count=len(selected))
        reports.append(row);print(json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)
        del policy
        if model=='act':del pre,post,batch
        gc.collect();torch.cuda.empty_cache()
    prefix.with_suffix('.json').write_text(json.dumps(dict(
        scope='Small fixed contact set: 20 held-out and 10 training timestamps. Screening only, not full validation or closed-loop success.',
        candidates=reports),indent=2)+'\n')
    np.savez_compressed(prefix.with_suffix('.npz'),**predictions)


if __name__=='__main__':main()
