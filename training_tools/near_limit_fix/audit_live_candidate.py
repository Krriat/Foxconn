#!/usr/bin/env python3
"""Compare a candidate with six frozen real near-flap inputs, without ROS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'AIDE_ACT_dataset_20261111_195100/act/xcore_act/scripts/auditing'))
sys.path.insert(0, str(ROOT / 'training_tools/two_stage_policy'))
from audit_xcore_act_checkpoints import load_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.utils import prepare_observation_for_inference
from protocol import joint_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    frozen = ROOT / 'training_verification/restore_far_20261112/near_contact_audit/live_inputs.npz'
    data = np.load(frozen)
    config = yaml.safe_load((ROOT / 'training_tools/near_limit_fix/config.yaml').read_text())
    lower, upper = map(np.asarray, joint_bounds(config['joint_limit_margin_rad'], config['joint_upper_limits_rad']))
    meta = LeRobotDatasetMetadata('local/dual_arm_suction_train', root=ROOT / 'prepared_data/suction_20hz_v1/train')
    output = dict(input_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  scope='Six frozen real input timestamps; no physical success labels or robot connection.', cases=[])
    predictions = {}
    for name, path in (('baseline', ROOT / 'model_outputs/act_suction_20261111_213453/checkpoints/best/pretrained_model'),
                       ('candidate', args.candidate)):
        policy, pre, post, cameras = load_policy(path, meta, 'cuda')
        rows = []
        for index in range(len(data['state'])):
            raw = {'observation.state': data['state'][index].astype(np.float32)}
            raw.update({key: data[key.rsplit('.', 1)[1]][index] for key in cameras})
            with torch.inference_mode():
                batch = pre(prepare_observation_for_inference(raw, torch.device('cuda'), task='', robot_type=''))
                pred = post(policy.predict_action_chunk(batch)).float().cpu().numpy()[0]
            assert pred.shape == (50, 16) and np.isfinite(pred).all()
            rows.append(pred)
            invalid = (pred[:, :14] < lower) | (pred[:, :14] > upper)
            record = dict(model=name, input_index=index,
                violating_joint_points=int(invalid.sum()),
                suction_left=[float(pred[h, 14]) for h in (0, 10, 49)],
                max_delta_from_observed_deg=[float(np.rad2deg(abs(pred[h, :14]-data['state'][index, :14])).max()) for h in (0, 10, 49)])
            if name == 'baseline':
                record['max_difference_from_recorded_chunk'] = float(abs(pred-data['chunk'][index]).max())
            else:
                record['max_joint_difference_from_baseline_deg'] = float(np.rad2deg(abs(pred[:, :14]-predictions['baseline'][index, :, :14])).max())
            output['cases'].append(record)
        predictions[name] = np.stack(rows)
        del policy, pre, post
        torch.cuda.empty_cache()
    args.output.with_suffix('.json').write_text(json.dumps(output, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'), **predictions)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
