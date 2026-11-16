#!/usr/bin/env python3
"""Offline input ablations. Does not initialize ROS or command any hardware."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, '/home/jj/.local/share/robot-policy-streams')
from inference_node import load_runtime

ROOT = Path('/home/jj/code/ACT')
OUT = ROOT/'training_verification/restore_far_20261112'
config = yaml.safe_load(Path('/home/jj/.local/share/robot-policy-streams/config.yaml').read_text())
runtime, predict, reset, sha = load_runtime(config)
data = np.load(OUT/'exact_inputs.npz')
cams = ('head', 'left_wrist', 'right_wrist')
rows, predictions = [], {}
def check(name, state, images, reference):
    reset()
    result = predict(dict(state=state, images=images))
    predictions[name] = result
    row = dict(name=name, max_difference_from_recorded_rad=float(np.max(abs(result-reference))),
               mean_joint_difference_deg=float(np.rad2deg(abs(result[:,:14]-reference[:,:14])).mean()),
               left_J4_start_end_deg=np.rad2deg(result[[0,-1],3]).tolist(),
               right_J2_start_end_deg=np.rad2deg(result[[0,-1],8]).tolist(),
               left_suction_start_end=result[[0,-1],14].tolist(),
               mean_RGB=[np.mean(image,axis=(0,1)).tolist() for image in images])
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='mean_RGB'}),flush=True)
    return result

for sample in range(len(data['state'])):
    state, reference = data['state'][sample], data['recorded'][sample]
    images = [data[cam][sample] for cam in cams]
    for gain in (1.,1.2,1.4,1.6):
        transformed = [image if i==0 else np.clip(image.astype(np.float32)*gain,0,255).astype(np.uint8)
                       for i,image in enumerate(images)]
        check(f'sample{sample}_wrist_gain{gain}',state,transformed,reference)
    if sample==0:
        for ep in (0,18,32):
            base=ROOT/f'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz/episode_{ep:06d}'
            recorded_images=[cv2.cvtColor(cv2.imread(str(base/f'images/{cam}/000000.png')),cv2.COLOR_BGR2RGB) for cam in cams]
            for which in ('head','wrists','all'):
                mix=[recorded_images[i] if which=='all' or (which=='head' and i==0) or (which=='wrists' and i>0) else images[i] for i in range(3)]
                check(f'sample0_dataset{ep}_{which}',state,mix,reference)
        check('sample0_repeat',state,images,reference)

(OUT/'input_audit.json').write_text(json.dumps(dict(weights_sha256=sha,cases=rows,
    scope='Offline counterfactual image inputs only. Does not establish closed-loop improvement.'),indent=2)+'\n')
np.savez_compressed(OUT/'input_audit_predictions.npz',**predictions)
