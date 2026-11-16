#!/usr/bin/env python3
"""Isolated near-flap ACT fine-tune and full held-out evaluation. No ROS calls.

Keeps the original normalization and train/validation split. Writes new model
directories only; never edits the deployed configuration or original weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'AIDE_ACT_dataset_20261111_195100/act/xcore_act/scripts/auditing'))
from audit_xcore_act_checkpoints import load_policy
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_IMAGES


def provenance(split):
    return json.loads((ROOT / f'prepared_data/suction_20hz_v1/{split}/meta/keyboard_teach_source.json').read_text())['episodes']


def is_near(row):
    return int(row['raw_episode'].rsplit('_', 1)[1]) % 2 == 1


def make_loader(split, policy, batch_size, workers, near_only=False):
    root = ROOT / f'prepared_data/suction_20hz_v1/{split}'
    repo = f'local/dual_arm_suction_{split}'
    meta = LeRobotDatasetMetadata(repo, root=root)
    episodes = [r['episode_index'] for r in provenance(split) if is_near(r)] if near_only else None
    dataset = LeRobotDataset(repo, root=root, episodes=episodes,
                            delta_timestamps=resolve_delta_timestamps(policy.config, meta), video_backend='pyav')
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      prefetch_factor=2 if workers else None)


@torch.inference_mode()
def evaluate(policy, pre, post, loader):
    policy.eval()
    near_ids = {r['episode_index'] for r in provenance('val') if is_near(r)}
    accum = {}
    for group in ('near', 'far'):
        accum[group] = dict(frames=0, targets=0, normalized_sum=0., joint_sum=np.zeros(14),
                            suction_wrong=np.zeros(2), horizon10_count=0,
                            horizon10_joint_sum=np.zeros(14), horizon10_suction_wrong=np.zeros(2))
    for raw in loader:
        target = raw['action'].float().numpy().copy()
        groups = np.array([int(ep) in near_ids for ep in raw['episode_index']])
        batch = pre(raw)
        # FP32, like the deployed inference node; prediction uses zero VAE latent.
        normalized = policy.predict_action_chunk(batch).float()
        if not torch.isfinite(normalized).all():
            raise FloatingPointError('non-finite validation prediction')
        norm_error = (normalized - batch['action']).abs().cpu().numpy()
        pred = post(normalized).float().cpu().numpy()
        valid = (~batch['action_is_pad']).cpu().numpy()
        for group, selection in (('near', groups), ('far', ~groups)):
            a = accum[group]
            mask = valid & selection[:, None]
            a['frames'] += int(selection.sum())
            a['targets'] += int(mask.sum())
            a['normalized_sum'] += float(norm_error[mask].sum())
            error = np.rad2deg(np.abs(pred[..., :14] - target[..., :14]))
            a['joint_sum'] += error[mask].sum(axis=0)
            wrong = (pred[..., 14:] >= .5) != (target[..., 14:] >= .5)
            a['suction_wrong'] += wrong[mask].sum(axis=0)
            hmask = selection & valid[:, 10]
            a['horizon10_count'] += int(hmask.sum())
            a['horizon10_joint_sum'] += error[hmask, 10].sum(axis=0)
            a['horizon10_suction_wrong'] += wrong[hmask, 10].sum(axis=0)
    result = {}
    for group, a in accum.items():
        n, h = a['targets'], a['horizon10_count']
        if n == 0 or h == 0:
            raise ValueError(f'empty validation group: {group}')
        result[group] = dict(frames=a['frames'], valid_targets=n,
            normalized_l1=a['normalized_sum'] / (n * 16),
            left_joint_mae_deg=float(a['joint_sum'][:7].mean() / n),
            right_joint_mae_deg=float(a['joint_sum'][7:].mean() / n),
            per_joint_mae_deg=(a['joint_sum'] / n).tolist(),
            suction_error_fraction=(a['suction_wrong'] / n).tolist(),
            horizon10_left_mae_deg=float(a['horizon10_joint_sum'][:7].mean() / h),
            horizon10_right_mae_deg=float(a['horizon10_joint_sum'][7:].mean() / h),
            horizon10_suction_error_fraction=(a['horizon10_suction_wrong'] / h).tolist())
    return result


def append(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'model_outputs/act_suction_20261111_213453/checkpoints/best/pretrained_model')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr-backbone', type=float, default=2e-6)
    parser.add_argument('--objective', choices=('cvae', 'zero-latent'), default='cvae')
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.batch_size, args.lr, args.lr_backbone) <= 0:
        raise ValueError('steps, interval, batch size and learning rates must be positive')
    if args.workers < 0:
        raise ValueError('workers must be non-negative')
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    seed = 20261112
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_rows = [r for r in provenance('train') if is_near(r)]
    val_rows = provenance('val')
    train_raw = {r['raw_episode'] for r in train_rows}
    val_raw = {r['raw_episode'] for r in val_rows}
    assert train_raw.isdisjoint(val_raw), 'train/validation leakage'
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        seed=seed, source_sha256=hashlib.sha256((args.source / 'model.safetensors').read_bytes()).hexdigest(),
        objective=('Original ACT CVAE L1 + KL' if args.objective == 'cvae' else
                   'Masked L1 on deployed zero-latent inference path; dropout disabled; architecture unchanged'),
        selection_metric='full held-out near-flap mean absolute left-joint error in degrees across valid chunk targets',
        training_raw_episodes=sorted(train_raw), validation_raw_episodes=sorted(val_raw),
        training_prepared_episodes=[r['episode_index'] for r in train_rows],
        limitation='Validation is teacher-forced offline prediction, not autonomous contact success.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(__file__, args.output / 'finetune_near.py')
    meta = LeRobotDatasetMetadata('local/dual_arm_suction_train', root=ROOT / 'prepared_data/suction_20hz_v1/train')
    policy, pre, post, _ = load_policy(args.source, meta, 'cuda')
    train = make_loader('train', policy, args.batch_size, args.workers, near_only=True)
    validation = make_loader('val', policy, args.batch_size, args.workers)
    groups = [dict(params=[p for name, p in policy.named_parameters() if 'backbone' not in name], lr=args.lr),
              dict(params=[p for name, p in policy.named_parameters() if 'backbone' in name], lr=args.lr_backbone)]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    start = time.monotonic()
    baseline = evaluate(policy, pre, post, validation)
    append(args.output / 'validation.jsonl', dict(step=0, elapsed_seconds=time.monotonic()-start, **baseline))
    best = dict(step=0, checkpoint=str(args.source), score=baseline['near']['left_joint_mae_deg'])
    iterator = iter(train)
    for step in range(1, args.steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train)
            raw = next(iterator)
        # Eval mode still permits gradients. It selects the identical zero-latent
        # branch used by predict_action_chunk and disables dropout for this ablation.
        policy.train(args.objective == 'cvae')
        batch = pre(raw)
        if step == 1 and args.objective == 'zero-latent':
            with torch.no_grad():
                check_batch = dict(batch)
                check_batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
                direct = policy.model(check_batch)[0]
                wrapped = policy.predict_action_chunk(batch)
                torch.testing.assert_close(direct, wrapped, rtol=1e-5, atol=1e-5)
            print(json.dumps(dict(event='zero_latent_matches_deployed_forward', max_error=float((direct-wrapped).abs().max()))), flush=True)
            del direct, wrapped, check_batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if args.objective == 'cvae':
                loss, detail = policy(batch)
            else:
                model_batch = dict(batch)
                model_batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
                prediction = policy.model(model_batch)[0]
                valid = (~batch['action_is_pad']).unsqueeze(-1)
                loss = ((prediction.float() - batch['action']).abs() * valid).sum() / (valid.sum() * 16)
                detail = dict(l1_loss=float(loss.detach()), kld_loss=None)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step}')
        loss.backward()
        if step == 1 and args.objective == 'zero-latent':
            assert policy.model.action_head.weight.grad is not None
            assert all(p.grad is None for name, p in policy.named_parameters() if 'vae_encoder' in name)
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            append(args.output / 'training.jsonl', dict(step=step, elapsed_seconds=time.monotonic()-start,
                loss=float(loss.detach()), grad_norm=float(norm), **detail))
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(policy, pre, post, validation)
            append(args.output / 'validation.jsonl', dict(step=step, elapsed_seconds=time.monotonic()-start, **metrics))
            dest = args.output / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
            policy.save_pretrained(dest)
            pre.save_pretrained(dest)
            post.save_pretrained(dest)
            score = metrics['near']['left_joint_mae_deg']
            if score < best['score']:
                best = dict(step=step, checkpoint=str(dest), score=score)
            (args.output / 'best_validation.json').write_text(json.dumps(best, indent=2) + '\n')
    print(json.dumps(dict(event='completed', best=best, deployed=False)), flush=True)


if __name__ == '__main__':
    main()
