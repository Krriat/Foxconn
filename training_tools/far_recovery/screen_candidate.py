"""Screen far ACT against archived and current RGB/state; never creates ROS nodes."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path('/home/jj/code/ACT')
APP = Path.home() / '.local/share/robot-policy-streams'
OUT = ROOT / 'training_verification/far_recovery_20261113'
sys.path.insert(0, str(APP))
from inference_node import load_runtime
from protocol import joint_bounds

cfg = yaml.safe_load((APP / 'config.yaml').read_text())
source = cfg['act_stage_checkpoints']['far']['path']
candidate = ROOT / 'model_outputs/act_far_recovery_20261113_01/checkpoints/000500/pretrained_model'
base = ROOT / 'training_verification/manual_disconnect_20261113'
historical = np.load(base / 'historical_image_pairs.npz')
current = np.load(base / 'live_gc_fixed_02/first_input.npz')
samples = [(f'archive_{i}', images, state) for i, (stage, images, state) in enumerate(
    zip(historical['stage'], historical['images'], historical['state'])) if stage == 'far']
samples.append(('current_home', current['images'], current['state']))
lo, hi = map(np.asarray, joint_bounds(cfg['joint_limit_margin_rad'], cfg.get('joint_upper_limits_rad')))
report = {}
arrays = {}
for name, path in [('original', Path(source)), ('candidate', candidate)]:
    sha = hashlib.sha256((path / 'model.safetensors').read_bytes()).hexdigest()
    effective = copy.deepcopy(cfg)
    effective['checkpoint_sha256']['act'] = sha
    _, predict, reset, _ = load_runtime(effective, path)
    rows = []
    for label, images, state in samples:
        reset()
        action = predict(dict(images=list(images), state=state))
        arrays[name + '_' + label] = action
        violations = (action[:, :14] < lo) | (action[:, :14] > hi)
        rows.append(dict(sample=label, finite=bool(np.isfinite(action).all()),
            violating_joint_targets=int(violations.sum()),
            first_target_offset_deg=np.rad2deg(action[0,:14] - state[:14]).tolist(),
            left_suction_range=[float(action[:,14].min()),float(action[:,14].max())]))
    report[name] = dict(checkpoint=str(path), sha256=sha, samples=rows)
    del predict, reset
(OUT / 'candidate_screen.json').write_text(json.dumps(report, indent=2) + '\n')
np.savez_compressed(OUT / 'candidate_screen_outputs.npz', **arrays)
print(json.dumps(report, indent=2))
