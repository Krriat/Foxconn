"""Install current dual ACT on Jetson from IDLE/OFF; never starts motion."""
import argparse,hashlib,json,shlex,subprocess,sys,datetime
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
APP=Path.home()/'.local/share/robot-policy-streams';REMOTE='/home/jetson/.local/share/robot-policy-streams'
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--ssh-control-path',required=True);a=p.parse_args()
SSH=['ssh','-S',a.ssh_control_path,'-o','BatchMode=yes','-o','ConnectTimeout=5','jetson@172.16.12.95']
def run(argv,**kw):return subprocess.run(argv,check=True,capture_output=True,text=True,timeout=45,**kw).stdout
def remote(cmd,**kw):return run(SSH+[cmd],**kw)
def sha(data):return hashlib.sha256(data).hexdigest()
def stop_remote():return remote('for unit in robot-policy-inference.service robot-policy-executor.service; do if systemctl --user is-active --quiet \"$unit\"; then systemctl --user stop \"$unit\"; fi; done')
sys.path.insert(0,str(ROOT/'training_tools/near_limit_fix'));from run_candidate_once import status
before=status();assert before['phase']=='IDLE' and before['readback_suction']==[False,False]
plan=json.loads((a.evidence/'plan.json').read_text());bundle=plan['remote_bundle']
validation=json.loads(remote('cat '+bundle+'/tensorrt_validation.json'));assert validation['passed'] and validation['both_models_resident']
assert all(row['passed'] for row in json.loads((a.evidence/'desktop_equivalence.json').read_text()).values())
for stage,spec in plan['stages'].items():
 meta=json.loads(remote('cat '+bundle+'/'+stage+'.engine.json'))
 assert meta['weights_sha256']==spec['sha256'] and meta['precision']=='IEEE FP32, FP16/TF32 disabled'
 assert validation['stages'][stage]['hysteresis_threshold_crossings']==0
brain_sha=sha((ROOT/'model_outputs/task_brain_lora_20261112/adapter/adapter_model.safetensors').read_bytes())
assert brain_sha==plan['brain_adapter_sha256']
local_names=['config.yaml','run.sh','start.sh','inference_remote.txt']
remote_names=['config.yaml','inference_node.py','jetson_policy_runtime.py','jetson_inference_bootstrap.py','run_inference_jetson.sh','fastdds_act_transport.xml','inference_remote.txt']
backup=a.evidence/'deployment_backup';(backup/'pc').mkdir(parents=True);(backup/'jetson').mkdir()
local_old={n:(APP/n).read_text() if (APP/n).exists() else None for n in local_names};remote_old={}
for n,data in local_old.items():
 if data is not None:(backup/'pc'/n).write_text(data)
for n in remote_names:
 exists=remote('test ! -f '+REMOTE+'/'+n+' || echo exists').strip();data=remote('cat '+REMOTE+'/'+n) if exists else None;remote_old[n]=data
 if data is not None:(backup/'jetson'/n).write_text(data)
for n in ('executor_node.py','protocol.py'):
 assert remote('sha256sum '+REMOTE+'/'+n).split()[0]==sha((HERE/n).read_bytes()),'Live controller changed: '+n
cfg=yaml.safe_load(remote_old['config.yaml'])
assert cfg['execution_speed_scale']==1.0 and cfg['execution_speed_profile']=='uniform' and cfg['max_joint_acceleration_rad_s2']==1.0
assert {k:v['sha256'] for k,v in cfg['act_stage_checkpoints'].items()}=={k:v['sha256'] for k,v in plan['stages'].items()}
cfg.update(act_stage_checkpoints=plan['stages'],inference_runtime='jetson_act',inference_bundle=bundle,act_root=bundle,inference_cpu_threads=2,act_backend='tensorrt')
assert cfg['io_host']=='192.168.50.20' and cfg['io_port']==20108
sys.path.insert(0,str(HERE))
from protocol import validate_config
validate_config(cfg)
new_cfg=yaml.safe_dump(cfg,sort_keys=False);(a.evidence/'jetson_target_config.yaml').write_text(new_cfg)
marker='ACT inference runs on jetson@172.16.12.95. This computer runs Qwen LoRA only.\nStart Jetson: ~/.local/share/robot-policy-streams/start.sh inference\n'
run_sh=local_old['run.sh'];needle='  inference)\n';assert run_sh.count(needle)==1
run_sh=run_sh.replace(needle,needle+'    if [[ -f "$policy_dir/inference_remote.txt" ]]; then cat "$policy_dir/inference_remote.txt" >&2; exit 2; fi\n')
changed=False
try:
 changed=True
 subprocess.run(['systemctl','--user','stop','robot-policy-inference.service'],capture_output=True,text=True,timeout=30)
 stop_remote()
 for n in ('inference_node.py','jetson_policy_runtime.py','jetson_inference_bootstrap.py','run_inference_jetson.sh','fastdds_act_transport.xml'):
  remote('cat > '+REMOTE+'/'+n,input=(HERE/n).read_text())
 remote('cat > '+REMOTE+'/config.yaml',input=new_cfg)
 remote('rm -f '+REMOTE+'/inference_remote.txt')
 (APP/'run.sh').write_text(run_sh);(APP/'inference_remote.txt').write_text(marker)
 remote(REMOTE+'/start.sh executor');remote(REMOTE+'/start.sh inference')
except BaseException:
 if changed:
  stop_remote()
  for n,data in remote_old.items():
   if data is None:remote('rm -f '+REMOTE+'/'+n)
   else:remote('cat > '+REMOTE+'/'+n,input=data)
  for n,data in local_old.items():
   if data is None:(APP/n).unlink(missing_ok=True)
   else:(APP/n).write_text(data)
  remote(REMOTE+'/start.sh executor')
 raise
report=dict(deployed=True,hardware_motion_started=False,inference_host='tegra-ubuntu',brain_host='jj-MS-7E71',backend='tensorrt',precision='IEEE FP32',weights=plan['stages'],brain_adapter_sha256=brain_sha,execution_speed_scale=1.0,execution_speed_profile='uniform',max_joint_acceleration_rad_s2=1.0,io_endpoint_preserved=cfg['io_host']+':'+str(cfg['io_port']),native_driver_and_executor_code_preserved=True,desktop_ACT_disabled=True,backup=str(backup),timestamp=datetime.datetime.now().astimezone().isoformat())
(a.evidence/'deployment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
