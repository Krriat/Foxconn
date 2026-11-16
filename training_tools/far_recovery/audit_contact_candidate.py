"""Compare contact adaptation candidates on saved demonstrations, without ROS.

These data are training demonstrations. This audit measures fit and joint-bound
violations; it is explicitly not held-out accuracy or physical success.
"""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

APP=Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0,str(APP))
from inference_node import load_runtime
from protocol import joint_bounds

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,action='append',required=True)
p.add_argument('--data',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--stride',type=int,default=3)
a=p.parse_args()
if a.stride<1:raise ValueError('stride must be positive')
cfg=yaml.safe_load((APP/'config.yaml').read_text())
lo,hi=map(np.asarray,joint_bounds(cfg['joint_limit_margin_rad'],cfg.get('joint_upper_limits_rad')))
result={'scope':'Training-demonstration fit, NOT independent accuracy or live task success','checkpoints':[]}
for checkpoint in a.checkpoint:
    sha=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()
    # Local audit configuration only; never changes the deployed model identity.
    cfg['checkpoint_sha256']['act']=sha
    runtime,predict,reset,_=load_runtime(cfg,checkpoint)
    report={'checkpoint':str(checkpoint.resolve()),'sha256':sha,'datasets':[]}
    for root in a.data:
        d=np.load(root/'data.npz');errors=[];ioerrors=[];contact=[];violations=0;count=0;nonfinite=0
        for i in range(0,len(d['state']),a.stride):
            images=[cv2.cvtColor(cv2.imread(str(root/path)),cv2.COLOR_BGR2RGB) for path in d['images'][i]]
            reset();pred=predict(dict(images=images,state=d['state'][i]));target=d['action'][i];valid=~d['action_is_pad'][i]
            nonfinite+=int((~np.isfinite(pred)).sum())
            violations+=int(((pred[:,:14]<lo)|(pred[:,:14]>hi)).sum())
            error=np.rad2deg(abs(pred[:,:14]-target[:,:14]));errors.append(error[valid])
            ioerrors.append(((pred[:,14:]>.5)!=(target[:,14:]>.5))[valid])
            if d['state'][i,14]>.5:contact.append(error[valid])
            count+=1
        error=np.concatenate(errors);io=np.concatenate(ioerrors);ce=np.concatenate(contact)
        report['datasets'].append(dict(source=str(root.resolve()),samples=count,valid_targets=len(error),
            left_joint_mae_deg=float(error[:,:7].mean()),right_joint_mae_deg=float(error[:,7:].mean()),
            joint_p95_deg=float(np.percentile(error,95)),suction_error_fraction=io.mean(0).tolist(),
            suction_on_left_joint_mae_deg=float(ce[:,:7].mean()),joint_bound_violations=violations,
            nonfinite_predictions=nonfinite))
    result['checkpoints'].append(report)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    del predict,reset,runtime
    gc.collect()
    import torch
    torch.cuda.empty_cache()
