#!/usr/bin/env python3
"""Save real RGB/state fixtures and validate the isolated ACT network loader."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'training_verification/agx_orin_20261113/bundle'
sys.path[:0] = [str(BUNDLE), str(ROOT / 'AIDE_ACT_dataset_20261111_195100/act/xcore_act/scripts/auditing')]
from jetson_inference_runtime import Engine
from audit_xcore_act_checkpoints import load_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.utils import prepare_observation_for_inference


def main():
    raw = ROOT / 'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz'
    images, states, rows = [], [], []
    for number in (18, 20, 21, 35):
        episode = raw / f'episode_{number:06d}'
        with np.load(episode / 'trajectory.npz') as data:
            for frame in (0, len(data['state']) // 2):
                views = [cv2.imread(str(episode / f'images/{camera}/{frame:06d}.png'))
                         for camera in ('head', 'left_wrist', 'right_wrist')]
                assert all(v is not None and v.shape == (480, 640, 3) for v in views)
                images.append(np.stack([cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in views]))
                states.append(data['state'][frame].copy())
                rows.append(dict(raw_episode=number, frame=frame,
                    source_stage='far' if number % 2 == 0 else 'near'))
    images, states = np.stack(images), np.stack(states)
    np.savez_compressed(BUNDLE / 'samples.npz', images=images, states=states)
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    manifest.update(fixtures=rows, image_shape=list(images.shape),
        samples_sha256=hashlib.sha256((BUNDLE / 'samples.npz').read_bytes()).hexdigest(),
        safety='Offline inference only; no ROS, robot driver, suction or network-control modules bundled')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = LeRobotDatasetMetadata('local/dual_arm_suction_train', root=ROOT / 'prepared_data/suction_20hz_v1/train')
    checks = {}
    with torch.inference_mode():
        for stage in ('far', 'near'):
            engine = Engine(BUNDLE / stage, 'act')
            policy, pre, post, keys = load_policy(BUNDLE / stage / 'act', meta, 'cuda')
            differences, reference = [], []
            for views, state in zip(images, states):
                actual = engine.postprocess(engine.forward(engine.prepare(views, state)))
                raw_observation = {'observation.state': state, **dict(zip(keys, views))}
                batch = pre(prepare_observation_for_inference(raw_observation,
                    torch.device('cuda'), task='', robot_type=''))
                expected = post(policy.predict_action_chunk(batch)).float().cpu().numpy()
                assert actual.shape == expected.shape == (1, 50, 16)
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
                differences.append(float(np.abs(actual - expected).max()))
                reference.append(expected[0])
            np.save(BUNDLE / f'{stage}_reference_fp32.npy', np.stack(reference))
            checks[stage] = dict(fixtures=len(reference), original_policy_matches=True,
                max_abs_difference=max(differences))
            del engine, policy, pre, post, batch
            torch.cuda.empty_cache()
    manifest['local_equivalence'] = checks
    manifest['bundle_files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BUNDLE.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
