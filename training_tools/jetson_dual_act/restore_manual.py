"""Pause Jetson/Qwen experiments and restore PC dual ACT, without motion start."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT=Path(__file__).resolve().parents[2]
APP=Path.home()/'.local/share/robot-policy-streams'
REMOTE='/home/jetson/.local/share/robot-policy-streams'
ORIGINAL='backup_before_jetson_act_20261113_094435'
SSH=['ssh','-o','ControlPath=/tmp/act_jetson_local_20261113.sock','-o','BatchMode=yes','jetson@172.16.12.95']


def run(command,**kwargs):
    return subprocess.run(command,check=True,capture_output=True,text=True,timeout=45,**kwargs).stdout


def remote(command,**kwargs):return run(SSH+[command],**kwargs)


def main():
    sys.path.insert(0,str(ROOT/'training_tools/near_limit_fix'))
    from run_candidate_once import status
    before=status()
    assert before['phase']=='IDLE' and before['readback_suction']==[False,False],before
    cfg=yaml.safe_load((APP/'config.yaml').read_text())
    assert not cfg.get('inference_runtime')
    for spec in cfg['act_stage_checkpoints'].values():
        assert hashlib.sha256((Path(spec['path'])/'model.safetensors').read_bytes()).hexdigest()==spec['sha256']
    tag='backup_before_restore_manual_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    local_backup=APP/tag;local_backup.mkdir()
    for name in ('run.sh','start.sh','config.yaml','inference_remote.txt'):
        if (APP/name).exists():shutil.copy2(APP/name,local_backup/name)
    remote('mkdir '+shlex.quote(REMOTE+'/'+tag))
    for name in ('config.yaml','executor_node.py','inference_remote.txt'):
        remote('if test -f '+shlex.quote(REMOTE+'/'+name)+'; then cp '+shlex.quote(REMOTE+'/'+name)+' '+shlex.quote(REMOTE+'/'+tag+'/'+name)+'; fi')
    remote('for unit in robot-policy-inference.service robot-policy-executor.service; do '
           'if systemctl --user is-active --quiet "$unit"; then systemctl --user stop "$unit"; fi; done')
    for name in ('run.sh','start.sh'):
        shutil.copy2(APP/ORIGINAL/name,APP/name)
        (APP/name).chmod(0o755)
    (APP/'inference_remote.txt').unlink(missing_ok=True)
    remote('cp '+REMOTE+'/'+ORIGINAL+'/config.yaml '+REMOTE+'/config.yaml')
    remote('cat > '+REMOTE+'/executor_node.py',input=(ROOT/'training_tools/two_stage_policy/executor_node.py').read_text())
    remote('cat > '+REMOTE+'/inference_remote.txt',input='Jetson ACT inference is paused. ACT inference now runs on jj@172.16.12.16.\nStart ~/.local/share/robot-policy-streams/start.sh inference on that computer.\n')
    remote(REMOTE+'/start.sh executor')
    run([str(APP/'start.sh'),'inference'])
    result=dict(task_state='paused',active_workflow='PC dual ACT + Jetson manual stages',
        local_backup=str(local_backup),remote_backup=REMOTE+'/'+tag,
        model_checkpoints=cfg['act_stage_checkpoints'],robot_task_started=False,
        brain_enabled=False,jetson_inference_enabled=False,pc_inference_enabled=True,
        manual_stage_timeout_seconds=None,automatic_stage_timeout_seconds=90)
    (ROOT/'training_verification/jetson_dual_act_20261113/paused_restore_manual.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
