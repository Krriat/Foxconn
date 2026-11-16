"""Training-fit and failed-input diagnostics; never a closed-loop success score."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,yaml
ROOT=Path('/home/jj/code/ACT');APP=Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0,str(APP))
from inference_node import load_runtime
from protocol import joint_bounds
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--recovery-data',type=Path,default=ROOT/'prepared_data/far_success_recovery_20261113');a=p.parse_args()
cfg=yaml.safe_load((APP/'config.yaml').read_text());sha=hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest();cfg['checkpoint_sha256']['act']=sha
_,predict,reset,_=load_runtime(cfg,a.checkpoint)
lo,hi=map(np.array,joint_bounds(cfg['joint_limit_margin_rad'],cfg.get('joint_upper_limits_rad')))
root=a.recovery_data;data=np.load(root/'data.npz');errors=[];ioerror=[];violations=0;total=0
for i in range(0,len(data['state']),3):
    images=[cv2.cvtColor(cv2.imread(str(root/path)),cv2.COLOR_BGR2RGB) for path in data['images'][i]]
    actual=predict(dict(images=images,state=data['state'][i]));target=data['action'][i];mask=~data['action_is_pad'][i]
    errors.append(np.rad2deg(abs(actual[mask,:14]-target[mask,:14])));ioerror.append((actual[mask,14:]>.5)!=(target[mask,14:]>.5))
    violations+=int(((actual[:,:14]<lo)|(actual[:,:14]>hi)).sum());total+=1
result=dict(checkpoint=str(a.checkpoint.resolve()),sha256=sha,training_fit_source=str(root.resolve()),training_fit_scope='Subsample of additional training demonstration; NOT independent accuracy',
    samples=total,training_joint_mae_deg=np.concatenate(errors).mean(0).tolist(),training_suction_error=np.concatenate(ioerror).mean(0).tolist(),training_bound_violations=violations)
d=np.load(ROOT/'training_verification/far_recovery_20261113/live_500_wifi_off/selected_inputs.npz');rows=[]
for i in range(len(d['state'])):
    actual=predict(dict(images=list(d['images'][i]),state=d['state'][i]));state=d['state'][i]
    rows.append(dict(seconds=int(d['seconds'][i]),bound_violations=int(((actual[:,:14]<lo)|(actual[:,:14]>hi)).sum()),
        first_offset_deg=np.rad2deg(actual[2,:14]-state[:14]).tolist(),
        next_2s_offset_deg=np.rad2deg(actual[42,:14]-state[:14]).tolist(),suction=actual[[2,42],14:].tolist()))
result['failed_observation_predictions']=rows
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='failed_observation_predictions'}))
