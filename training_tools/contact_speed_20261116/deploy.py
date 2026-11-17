"""Deploy only from fresh IDLE/OFF, with byte backups and automatic rollback.

Never invokes a motion-start, home, or suction service. Restarts the existing
idle inference/executor/driver services after installing the staged build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];APP=Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0,str(ROOT))
from training_tools.near_limit_fix.run_candidate_once import status
from protocol import validate_config

p=argparse.ArgumentParser();p.add_argument('--ssh-control-path',required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
SSH=['ssh','-S',a.ssh_control_path,'-o','BatchMode=yes','-o','ConnectTimeout=5','jetson@172.16.12.95']
REMOTE='/home/jetson/.local/share/robot-policy-streams';STAGED='/tmp/contact_speed_20261116'
BACKUP=REMOTE+'/backup_before_contact_speed_20261116'
def run(argv,**kw):return subprocess.run(argv,check=True,text=True,capture_output=True,timeout=120,**kw).stdout
def remote(command,**kw):return run(SSH+[command],**kw)
def sha(data):return hashlib.sha256(data).hexdigest()
before=status();assert before['phase']=='IDLE' and before['readback_suction']==[False,False],before
remote("python3 -c "+shlex.quote("from pathlib import Path; b=Path('/tmp/contact_speed_20261116/native_build/robot_timed_policy_driver').read_bytes(); assert b'continuous_action_clock_v1' in b, 'Staged binary lacks speed-scaling capability; rebuild from clean sources'"))
pc_files=['config.yaml','protocol.py','inference_node.py']
pc_original={name:(APP/name).read_bytes() for name in pc_files}
for name,data in pc_original.items():(a.output/('pc_before_'+name)).write_bytes(data)
pc_cfg=yaml.safe_load(pc_original['config.yaml']);remote_original=remote('cat '+REMOTE+'/config.yaml')
remote_cfg=yaml.safe_load(remote_original)
for cfg in (pc_cfg,remote_cfg):
    cfg.update(execution_speed_scale=1.5,max_joint_acceleration_rad_s2=2.25,execution_speed_profile="contact_slow_v1")
    validate_config(cfg)
remote_after=yaml.safe_dump(remote_cfg,sort_keys=False)
(a.output/'jetson_before_config.yaml').write_text(remote_original)
(a.output/'jetson_after_config.yaml').write_text(remote_after)
(a.output/'pc_after_config.yaml').write_text(yaml.safe_dump(pc_cfg,sort_keys=False))

prepare=r'''
from pathlib import Path
import hashlib,json,shutil
root=Path.home()/'.local/share/robot-policy-streams'
staged=Path('/tmp/contact_speed_20261116');backup=root/'backup_before_contact_speed_20261116'
if backup.exists():raise RuntimeError('Deployment backup already exists; do not overwrite it')
files=['config.yaml','protocol.py','executor_node.py','native/timed_policy.hpp','native/test_timed_policy.cpp',
       'native/vendor/src/ros_servoj_subscriber.cpp','native/build/robot_timed_policy_driver']
backup.mkdir()
manifest={}
for name in files:
    source=root/name;dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest);manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
(backup/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
'''
manifest=json.loads(remote('python3 -',input=prepare))
reference=ROOT/'training_verification/contact_speed_20261116/backup/jetson'
for name in ['config.yaml','protocol.py','executor_node.py','native/timed_policy.hpp','native/test_timed_policy.cpp','native/vendor/src/ros_servoj_subscriber.cpp']:
    assert manifest[name]==sha((reference/Path(name).name).read_bytes()),'Live file changed since review: '+name
(a.output/'jetson_before_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')

install=r'''
from pathlib import Path
import os,shutil,sys
root=Path.home()/'.local/share/robot-policy-streams';stage=Path('/tmp/contact_speed_20261116')
mapping={'protocol.py':'protocol.py','executor_node.py':'executor_node.py',
         'native_build_source/timed_policy.hpp':'native/timed_policy.hpp',
         'native_build_source/test_timed_policy.cpp':'native/test_timed_policy.cpp',
         'native_build_source/vendor/src/ros_servoj_subscriber.cpp':'native/vendor/src/ros_servoj_subscriber.cpp',
         'native_build/robot_timed_policy_driver':'native/build/robot_timed_policy_driver'}
for source,target in mapping.items():
    dest=root/target;tmp=dest.with_name(dest.name+'.speed15_tmp');shutil.copy2(stage/source,tmp);os.replace(tmp,dest)
config=root/'config.yaml';tmp=config.with_suffix('.speed15_tmp');tmp.write_text(sys.stdin.read());os.replace(tmp,config)
'''
# Pass configuration as a file, preserving real newlines and avoiding shell expansion.
cfg_stage=STAGED+'/deployment_config.yaml'
remote('cat > '+cfg_stage,input=remote_after)
stop_inference=lambda:subprocess.run(['systemctl','--user','stop','robot-policy-inference.service'],capture_output=True,text=True,timeout=30)
try:
    stop_inference()
    remote('systemctl --user stop robot-policy-executor.service')
    remote('systemctl --user stop robot-policy-driver.service')
    remote('python3 -c '+shlex.quote(install)+' < '+cfg_stage)
    for name in pc_files:
        data=(a.output/'pc_after_config.yaml').read_bytes() if name=='config.yaml' else (HERE/name).read_bytes()
        tmp=APP/(name+'.speed15_tmp');tmp.write_bytes(data);tmp.replace(APP/name)
    remote(REMOTE+'/start.sh driver');remote(REMOTE+'/start.sh executor')
    run([str(APP/'start.sh'),'inference'])
except BaseException:
    stop_inference()
    remote('systemctl --user stop robot-policy-executor.service')
    remote('systemctl --user stop robot-policy-driver.service')
    restore=r'''
from pathlib import Path
import json,shutil
root=Path.home()/'.local/share/robot-policy-streams';backup=root/'backup_before_contact_speed_20261116'
for name in json.loads((backup/'sha256.json').read_text()):shutil.copy2(backup/name,root/name)
'''
    remote('python3 -',input=restore)
    for name,data in pc_original.items():(APP/name).write_bytes(data)
    remote(REMOTE+'/start.sh driver');remote(REMOTE+'/start.sh executor');run([str(APP/'start.sh'),'inference'])
    raise
report=dict(deployed=True,execution_speed_scale=1.5,max_joint_acceleration_rad_s2=2.25,execution_speed_profile="contact_slow_v1",
            max_joint_speed_rad_s=pc_cfg['max_joint_speed_rad_s'],robot_motion_requested=False,
            home_parameters_changed=False,clock_and_watchdog_parameters_changed=False,
            remote_backup=BACKUP,prior_status=before,pc_after_sha256={k:sha((APP/k).read_bytes()) for k in pc_files})
(a.output/'deployment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='prior_status'},indent=2))
