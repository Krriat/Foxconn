#!/usr/bin/env python3
"""Offline near-contact audit on train/held-out demonstrations; no robot calls."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path.home()/'.local/share/robot-policy-streams'))
from inference_node import load_runtime


def main():
    config=yaml.safe_load((Path.home()/'.local/share/robot-policy-streams/config.yaml').read_text())
    runtime,predict,reset,sha=load_runtime(config)
    base=ROOT/'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz'
    out=ROOT/'training_verification/restore_far_20261112/near_contact_audit'
    split=json.loads((ROOT/'prepared_data/suction_20hz_v1/split_manifest.json').read_text())
    validation=set(split['splits']['val']['raw_episodes'])
    rows=[]
    for epnum in (21,35,39,47,31,37):
        ep=base/f'episode_{epnum:06d}'
        d=np.load(ep/'trajectory.npz');a=d['action'];state=d['state']
        on=state[:,14]>.5;off=np.flatnonzero(np.diff(on.astype(int))<0)+1
        release=int(off[0])
        for index in sorted(set((0,60,max(0,release-80),max(0,release-40),max(0,release-20)))):
            images=[cv2.cvtColor(cv2.imread(str(ep/f'images/{cam}/{index:06d}.png')),cv2.COLOR_BGR2RGB)
                    for cam in ('head','left_wrist','right_wrist')]
            reset();prediction=predict(dict(state=state[index],images=images))
            metrics={}
            for horizon in (0,1,5,10,20):
                if index+horizon>=len(a):continue
                error=np.rad2deg(prediction[horizon,:7]-a[index+horizon,:7])
                demo_motion=np.rad2deg(a[index+horizon,:7]-state[index,:7])
                metrics[str(horizon)]=dict(left_joint_mae_deg=float(abs(error).mean()),
                    left_joint_max_error_deg=float(abs(error).max()),
                    demonstrated_max_motion_deg=float(abs(demo_motion).max()),
                    suction_prediction=float(prediction[horizon,14]),suction_target=float(a[index+horizon,14]))
            rows.append(dict(episode=ep.name,split='val' if ep.name in validation else 'train',
                             index=index,release=release,metrics=metrics))
    summary={}
    for group in ('train','val'):
        summary[group]={}
        for horizon in (0,1,5,10,20):
            selected=[r['metrics'][str(horizon)] for r in rows if r['split']==group]
            summary[group][str(horizon)]={k:float(np.median([r[k] for r in selected]))
                  for k in ('left_joint_mae_deg','left_joint_max_error_deg','demonstrated_max_motion_deg')}
    report=dict(weights_sha256=sha,scope='Six selected near-flap episodes, five timestamps each. Four held-out and two training episodes. Offline teacher-forced inputs; not a closed-loop success estimate.',
                rows=rows,median_metrics=summary)
    (out/'contact_prediction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
