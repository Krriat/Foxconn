#!/usr/bin/env python3
"""Offline next-stage policy check; no publishers or robot service clients."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
sys.path.insert(0,'/home/jj/.local/share/robot-policy-streams')
from inference_node import load_runtime

ROOT=Path('/home/jj/code/ACT')
config=yaml.safe_load(Path('/home/jj/.local/share/robot-policy-streams/config.yaml').read_text())
runtime,predict,reset,sha=load_runtime(config)
home=np.array(config['home_left']+config['home_right']+[0.,0.],np.float32)
rows=[]
for ep in (18,20,21,35,39,47):
    p=ROOT/f'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz/episode_{ep:06d}'
    states=np.load(p/'trajectory.npz')['state']
    images=[cv2.cvtColor(cv2.imread(str(p/f'images/{camera}/000000.png')),cv2.COLOR_BGR2RGB)
            for camera in ('head','left_wrist','right_wrist')]
    for kind,state in (('demonstrated_start',states[0]),('home_reset',home)):
        reset();a=predict({'state':state,'images':images})
        desired=states[49,:14]-states[0,:14]
        predicted=a[49,:14]-a[0,:14]
        mask=np.abs(desired)>.01
        rows.append({'episode':ep,'stage':'far' if ep%2==0 else 'near','input':kind,
                     'finite':bool(np.isfinite(a).all()),
                     'start_target_max_offset_rad':float(np.max(np.abs(a[0,:14]-state[:14]))),
                     'chunk_mae_to_demonstration_rad':float(np.mean(np.abs(a[:,:14]-states[:50,:14]))),
                     'moving_joints':int(mask.sum()),
                     'direction_match_fraction':float(np.mean(predicted[mask]*desired[mask]>0)) if mask.any() else None,
                     'predicted_displacement_rad':predicted.tolist(),
                     'demo_displacement_rad':desired.tolist()})
report={'weights_sha256':sha,'cases':rows,'scope':'Offline first 50-step chunk with recorded images; home_reset is a counterfactual state, not a closed-loop rollout or task success guarantee.'}
(ROOT/'training_verification/two_stage_20261112/near_policy_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
