#!/usr/bin/env python3
"""Install reviewed dual-ACT standby services, only from IDLE with suction off.

Does not call any robot motion/start service. Prior local/remote files are backed up.
"""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
APP=Path.home()/'.local/share/robot-policy-streams'
REMOTE='/home/jetson/.local/share/robot-policy-streams'
EVIDENCE=ROOT/'training_verification/manual_stage_20261112'
SSH=['ssh','-o','ControlPath=/tmp/act_restore_far.sock','-o','BatchMode=yes','-o','ConnectTimeout=5','jetson@172.16.12.95']
sys.path.insert(0,str(ROOT/'training_tools/near_limit_fix'))
from run_candidate_once import status

def run(argv,**kwargs):
    return subprocess.run(argv,text=True,capture_output=True,check=True,timeout=30,**kwargs).stdout

def remote(command,**kwargs):return run(SSH+[command],**kwargs)

def main():
    assert len(sys.argv)==2 and sys.argv[1]=='--install', 'Use --install to update standby services'
    before=status()
    assert before['phase']=='IDLE' and before['readback_suction']==[False,False],before
    stage=yaml.safe_load((SOURCE/'config.yaml').read_text())['act_stage_checkpoints']
    for entry in stage.values():
        assert hashlib.sha256((Path(entry['path'])/'model.safetensors').read_bytes()).hexdigest()==entry['sha256']
    tag='backup_before_manual_stages_'+time.strftime('%Y%m%d_%H%M%S')
    backup=APP/tag;backup.mkdir()
    remote('mkdir '+shlex.quote(REMOTE+'/'+tag))
    local_names=['config.yaml','inference_node.py','executor_node.py','protocol.py','stage_policy.py','brain_protocol.py','joint_anchor.py','TWO_STAGE.md','start.sh']
    remote_names=['config.yaml','executor_node.py','protocol.py','stage_policy.py','brain_protocol.py','start.sh','TWO_STAGE.md']
    local_old={n:(APP/n).read_bytes() if (APP/n).exists() else None for n in local_names}
    remote_old={}
    for name,data in local_old.items():
        if data is not None:(backup/name).write_bytes(data)
    for name in remote_names:
        path=REMOTE+'/'+name
        exists=remote('test ! -f '+shlex.quote(path)+' || echo exists').strip()
        data=remote('cat '+shlex.quote(path)) if exists else None
        remote_old[name]=data
        if data is not None:
            remote('cp '+shlex.quote(path)+' '+shlex.quote(REMOTE+'/'+tag+'/'+name))
    config_local=yaml.safe_load(local_old['config.yaml'])
    config_remote=yaml.safe_load(remote_old['config.yaml'])
    for c in (config_local,config_remote):
        assert c['model']=='act'
        c['act_stage_checkpoints']=stage
    (EVIDENCE/'before_install.json').write_text(json.dumps(before,indent=2)+'\n')
    try:
        run(['systemctl','--user','stop','robot-policy-inference.service'])
        run(['systemctl','--user','stop','robot-policy-task-monitor.service'])
        remote('systemctl --user stop robot-policy-executor.service')
        for n in ('inference_node.py','executor_node.py','protocol.py','stage_policy.py','brain_protocol.py','joint_anchor.py','start.sh'):
            shutil.copyfile(SOURCE/n,APP/n)
        shutil.copyfile(SOURCE/'README.md',APP/'TWO_STAGE.md')
        (APP/'config.yaml').write_text(yaml.safe_dump(config_local,sort_keys=False))
        for n in ('executor_node.py','protocol.py','stage_policy.py','brain_protocol.py','start.sh'):
            remote('cat > '+shlex.quote(REMOTE+'/'+n),input=(SOURCE/n).read_text())
        remote('cat > '+shlex.quote(REMOTE+'/config.yaml'),input=yaml.safe_dump(config_remote,sort_keys=False))
        remote('cat > '+shlex.quote(REMOTE+'/TWO_STAGE.md'),input=(SOURCE/'README.md').read_text())
        remote('python3 -m py_compile '+ ' '.join(shlex.quote(REMOTE+'/'+n) for n in ('executor_node.py','protocol.py','stage_policy.py')))
        remote('~/.local/share/robot-policy-streams/start.sh executor')
        run([str(APP/'start.sh'),'inference'])
    except BaseException:
        run(['systemctl','--user','stop','robot-policy-inference.service'])
        remote('systemctl --user stop robot-policy-executor.service')
        for n,data in local_old.items():
            if data is None:(APP/n).unlink(missing_ok=True)
            else:(APP/n).write_bytes(data)
        for n,data in remote_old.items():
            if data is None:remote('rm -f '+shlex.quote(REMOTE+'/'+n))
            else:remote('cat > '+shlex.quote(REMOTE+'/'+n),input=data)
        remote('~/.local/share/robot-policy-streams/start.sh executor')
        run([str(APP/'start.sh'),'inference'])
        raise
    result=dict(local_backup=str(backup),remote_backup=REMOTE+'/'+tag,stage_checkpoints=stage,
                motion_started=False,box_scene_check_enabled=False,completion_mode='manual',
                installed_local={},installed_remote={})
    for n in local_names:
        result['installed_local'][n]=hashlib.sha256((APP/n).read_bytes()).hexdigest()
    for n in remote_names:
        result['installed_remote'][n]=remote('sha256sum '+shlex.quote(REMOTE+'/'+n)).split()[0]
        if n!='config.yaml':assert result['installed_remote'][n]==result['installed_local'][n]
    (EVIDENCE/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
