"""Image/state sensitivity only; swapped pairs are not physical counterfactuals."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training_tools/jetson_dual_act_20261116'))
from jetson_policy_runtime import ActEngine


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);a=p.parse_args()
    data=np.load(a.comparison/'inputs.npz');engine=ActEngine(a.comparison/'models/current',ROOT/'training_verification/jetson_dual_act_20261116_171114/bundle')
    torch.set_num_threads(3);rows=[]
    for bad in (3,5,7):
        good=10+int(np.argmin(np.linalg.norm(data['states'][10:,:7]-data['states'][bad,:7],axis=1)))
        for state_from in (bad,good):
            for mask in itertools.product((False,True),repeat=3):
                images=[data['images'][good if flag else bad,i] for i,flag in enumerate(mask)]
                pred=engine.predict(dict(images=images,state=data['states'][state_from]))
                rows.append(dict(bad_case=bad,good_case=good,state='success' if state_from==good else 'failure',
                    image_sources=['success' if f else 'failure' for f in mask],
                    left_2s_deg=np.rad2deg(pred[40,:7]).tolist(),suction_2s=float(pred[40,14])))
    for bad in (3,5,7):
        state=data['states'][bad].copy();state[7:14]=data['states'][10,7:14]
        images=list(data['images'][bad]);images[2]=data['images'][10,2]
        pred=engine.predict(dict(images=images,state=state))
        rows.append(dict(bad_case=bad,good_case=10,state='left current; right verified-home surrogate',
            image_sources=['failure','failure','success-home'],left_2s_deg=np.rad2deg(pred[40,:7]).tolist(),
            suction_2s=float(pred[40,14])))
    (a.comparison/'ablation.json').write_text(json.dumps({'motion_commands':0,
        'limitations':__doc__,'rows':rows},indent=2)+'\n')
    for r in rows:
        if len(set(r['image_sources']))==1:print(json.dumps(r),flush=True)


if __name__=='__main__':main()
