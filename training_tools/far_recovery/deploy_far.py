"""Swap only far ACT identity while both arms are IDLE; never starts motion."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

ROOT = Path('/home/jj/code/ACT')
APP = Path.home() / '.local/share/robot-policy-streams'
REMOTE = '/home/jetson/.local/share/robot-policy-streams'
SSH = ['ssh', '-o', 'ControlPath=/tmp/act_jetson_local_20261113.sock', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=5', 'jetson@172.16.12.95']
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
from training_tools.near_limit_fix.run_candidate_once import status
from protocol import validate_config

def run(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30, **kwargs).stdout

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--ssh-control-path', default='/tmp/act_jetson_local_20261113.sock')
parser.add_argument('--leave-inference-stopped', action='store_true')
args = parser.parse_args()
SSH[2] = 'ControlPath=' + args.ssh_control_path
checkpoint = args.checkpoint.resolve()
assert checkpoint.is_relative_to(ROOT / 'model_outputs')
sha = hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest()
args.output.mkdir(parents=True, exist_ok=False)
before = status()
assert before['phase'] == 'IDLE' and before['readback_suction'] == [False, False], before
originals = [(APP / 'config.yaml').read_text(), run(SSH + ['cat ' + REMOTE + '/config.yaml'])]
replacements = []
for label, original in zip(('pc', 'jetson'), originals):
    cfg = yaml.safe_load(original)
    cfg['act_stage_checkpoints']['far'] = dict(path=str(checkpoint), sha256=sha)
    cfg['checkpoint_sha256']['act'] = sha
    validate_config(cfg)
    replacement = yaml.safe_dump(cfg, sort_keys=False)
    replacements.append(replacement)
    (args.output / (label + '_before.yaml')).write_text(original)
    (args.output / (label + '_after.yaml')).write_text(replacement)

unit = subprocess.run(['systemctl', '--user', 'show', 'robot-policy-inference.service',
                      '--property=LoadState', '--value'], capture_output=True, text=True, timeout=10)
if unit.stdout.strip() != 'not-found':
    unit.check_returncode()
    run(['systemctl', '--user', 'stop', 'robot-policy-inference.service'])
run(SSH + ['systemctl --user stop robot-policy-executor.service'])
try:
    tmp = APP / 'config.yaml.tmp'
    tmp.write_text(replacements[0]); tmp.replace(APP / 'config.yaml')
    run(SSH + ['cat > ' + REMOTE + '/config.yaml.tmp && mv ' + REMOTE + '/config.yaml.tmp ' + REMOTE + '/config.yaml'], input=replacements[1])
    run(SSH + [REMOTE + '/start.sh executor'])
    if not args.leave_inference_stopped:
        run([str(APP / 'start.sh'), 'inference'])
except BaseException:
    (APP / 'config.yaml').write_text(originals[0])
    run(SSH + ['cat > ' + REMOTE + '/config.yaml'], input=originals[1])
    run(SSH + [REMOTE + '/start.sh executor'])
    if not args.leave_inference_stopped:
        run([str(APP / 'start.sh'), 'inference'])
    raise
manifest = dict(checkpoint=str(checkpoint), sha256=sha, changed='far checkpoint only',
    robot_motion_requested=False, inference_started=not args.leave_inference_stopped, prior_status=before)
(args.output / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(checkpoint=str(checkpoint), sha256=sha, deployed=True, robot_motion_requested=False)))
