"""Install Jetson dual-ACT inference from IDLE; never starts a robot task."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
APP = Path.home()/'.local/share/robot-policy-streams'
REMOTE = '/home/jetson/.local/share/robot-policy-streams'
BUNDLE = REMOTE+'/jetson_act_runtime_20261113'
EVIDENCE = ROOT/'training_verification/jetson_dual_act_20261113'
SSH = ['ssh','-o','ControlPath=/tmp/act_jetson_local_20261113.sock','-o','BatchMode=yes','jetson@172.16.12.95']


def run(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=True, timeout=60, **kwargs).stdout


def remote(command, **kwargs): return run(SSH+[command], **kwargs)


def stop_remote():
    remote('for unit in robot-policy-executor.service robot-policy-inference.service; do '
           'if systemctl --user is-active --quiet "$unit"; then systemctl --user stop "$unit"; fi; done')


def main():
    assert sys.argv[1:] == ['--install']
    sys.path.insert(0,str(ROOT/'training_tools/near_limit_fix'))
    from run_candidate_once import status
    before = status()
    assert before['phase']=='IDLE' and before['readback_suction']==[False,False],before
    benchmark = json.loads(remote('cat '+BUNDLE+'/benchmark_report.json'))
    assert benchmark['passed'] and benchmark['both_models_resident']
    acceleration = json.loads(remote('cat '+BUNDLE+'/tensorrt_validation.json'))
    assert acceleration['passed'] and acceleration['both_models_resident']
    equivalence = json.loads((EVIDENCE/'local_equivalence.json').read_text())
    assert all(r['passed'] for r in equivalence.values())
    tag = 'backup_before_jetson_act_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    backup = APP/tag;backup.mkdir()
    remote('mkdir '+shlex.quote(REMOTE+'/'+tag))
    local_names = ['run.sh','start.sh','inference_remote.txt']
    remote_names = ['config.yaml','run.sh','start.sh','inference_node.py','inference_remote.txt',
                    'executor_node.py','protocol.py','stage_policy.py','brain_protocol.py',
                    'jetson_policy_runtime.py','jetson_inference_bootstrap.py','joint_anchor.py','run_inference_jetson.sh']
    old_local = {n:(APP/n).read_text() if (APP/n).exists() else None for n in local_names}
    old_remote = {}
    for name,value in old_local.items():
        if value is not None:(backup/name).write_text(value)
    for name in remote_names:
        exists = remote('test ! -f '+shlex.quote(REMOTE+'/'+name)+' || echo exists').strip()
        old_remote[name] = remote('cat '+shlex.quote(REMOTE+'/'+name)) if exists else None
        if exists:remote('cp '+shlex.quote(REMOTE+'/'+name)+' '+shlex.quote(REMOTE+'/'+tag+'/'+name))
    cfg = yaml.safe_load(old_remote['config.yaml'])
    specs = yaml.safe_load(remote('cat '+BUNDLE+'/config.yaml'))['act_stage_checkpoints']
    assert {s:v['sha256'] for s,v in specs.items()} == {s:v['sha256'] for s,v in cfg['act_stage_checkpoints'].items()}
    cfg.update(act_stage_checkpoints=specs,inference_runtime='jetson_act',inference_bundle=BUNDLE,
               act_root=BUNDLE,inference_cpu_threads=2,act_backend='tensorrt')
    changed = []
    try:
        if subprocess.run(['systemctl','--user','is-active','--quiet','robot-policy-inference.service']).returncode == 0:
            run(['systemctl','--user','stop','robot-policy-inference.service'])
        stop_remote()
        for name in ('run.sh','start.sh'):
            text = (ROOT/'training_tools/two_stage_policy'/name).read_text()
            (APP/name).write_text(text)
            remote('cat > '+shlex.quote(REMOTE+'/'+name),input=text)
        (APP/'inference_remote.txt').write_text('ACT inference has moved to jetson@172.16.12.95.\nRun ~/.local/share/robot-policy-streams/start.sh inference on Jetson.\nThis computer runs only the Qwen LoRA brain.\n')
        for name in ('inference_node.py','executor_node.py','protocol.py','stage_policy.py','brain_protocol.py',
                     'jetson_policy_runtime.py','jetson_inference_bootstrap.py','joint_anchor.py'):
            source = HERE/name if (HERE/name).exists() else ROOT/'training_tools/two_stage_policy'/name
            remote('cat > '+shlex.quote(REMOTE+'/'+name),input=source.read_text())
        remote('cat > '+REMOTE+'/run_inference_jetson.sh',input=(HERE/'run_inference_jetson.sh').read_text())
        remote('cat > '+REMOTE+'/config.yaml',input=yaml.safe_dump(cfg,sort_keys=False))
        remote('rm -f '+REMOTE+'/inference_remote.txt')
        remote(REMOTE+'/start.sh executor')
        remote(REMOTE+'/start.sh inference')
        changed = remote_names
    except BaseException:
        stop_remote()
        for name,value in old_local.items():
            if value is None:(APP/name).unlink(missing_ok=True)
            else:(APP/name).write_text(value)
        for name,value in old_remote.items():
            if value is None:remote('rm -f '+shlex.quote(REMOTE+'/'+name))
            else:remote('cat > '+shlex.quote(REMOTE+'/'+name),input=value)
        remote(REMOTE+'/start.sh executor')
        run([str(APP/'start.sh'),'inference'])
        raise
    result = dict(local_backup=str(backup),remote_backup=REMOTE+'/'+tag,robot_task_started=False,
        desktop_act_disabled=True,jetson_models=specs,changed_remote=changed,
        control_parameters_preserved=True,before=before)
    (EVIDENCE/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='before'},indent=2))


if __name__ == '__main__': main()
