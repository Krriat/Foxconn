#!/usr/bin/env python3
"""Train task-specific RGB interACT policies offline. No ROS/hardware imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'AIDE_ACT_dataset_20261111_195100/act/xcroe_interACT'
sys.path.insert(0, str(PROJECT / 'src'))
from xcore_interact.data import InterACTWindowDataset, NormalizationStats
from xcore_interact.policy import InterACT3DConfig, InterACT3DPolicy

DATA = ROOT / 'prepared_data/suction_20hz_v1'
SOURCE = ROOT / 'model_outputs/interact_suction_20261111_234920/checkpoints/best/pretrained_model'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def log(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def task_rows(stage, split):
    mapping = json.loads((DATA / split / 'meta/keyboard_teach_source.json').read_text())
    parity = {'far': 0, 'near': 1}[stage]
    return [r for r in mapping['episodes'] if int(r['raw_episode'].rsplit('_', 1)[1]) % 2 == parity]


def validate_provenance(stage):
    manifest = json.loads((DATA / 'split_manifest.json').read_text())
    if not manifest['complete'] or manifest['fps'] != 20:
        raise ValueError('Expected complete, audited 20 Hz dataset')
    result = {}
    for split in ('train', 'val'):
        rows = task_rows(stage, split)
        if not rows or len({r['episode_index'] for r in rows}) != len(rows):
            raise ValueError('Empty or duplicate episode selection')
        provenance = json.loads((DATA / split / 'meta/keyboard_teach_source.json').read_text())
        for name in sorted({r['raw_episode'] for r in rows}):
            # Preserve the audited raw->prepared mapping, including pause boundaries.
            for file, digest in provenance['raw_episode_hashes'][name].items():
                if sha256(Path(manifest['raw_root']) / name / file) != digest:
                    raise ValueError(f'Raw source changed: {name}/{file}')
        result[split] = dict(raw_episodes=sorted({r['raw_episode'] for r in rows}),
            prepared_episodes=[r['episode_index'] for r in rows],
            frames=sum(r['end'] - r['start'] for r in rows), segments=rows)
    if set(result['train']['raw_episodes']) & set(result['val']['raw_episodes']):
        raise ValueError('Raw demonstration leakage')
    return result


def make_loader(stage, split, config, stats, batch_size, workers):
    rows = task_rows(stage, split)
    dataset = InterACTWindowDataset(DATA / split, f'local/dual_arm_suction_{split}',
        episodes=[r['episode_index'] for r in rows], chunk_size=config.chunk_size,
        image_size=config.image_size, stats=stats, use_point_cloud=False,
        state_layout='dual_arm_suction')
    expected = sum(r['end'] - r['start'] for r in rows)
    if len(dataset) != expected:
        raise ValueError(f'Task selection has {len(dataset)} frames, expected {expected}')
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
        num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
        prefetch_factor=2 if workers else None)


def task_statistics(stage):
    raw_root = Path(json.loads((DATA / 'split_manifest.json').read_text())['raw_root'])
    collected = {k: [] for k in ('state', 'action')}
    for row in task_rows(stage, 'train'):
        with np.load(raw_root / row['raw_episode'] / 'trajectory.npz') as arrays:
            for key in collected:
                collected[key].append(arrays[key][row['start']:row['end']].astype(np.float64))
    values = {}
    for key, segments in collected.items():
        data = np.concatenate(segments)
        if not np.isfinite(data).all():
            raise ValueError('Non-finite task normalization data')
        values[key+'_mean'] = data.mean(axis=0).astype(np.float32).tolist()
        values[key+'_std'] = np.sqrt(np.maximum(data.var(axis=0), 1e-8)).astype(np.float32).tolist()
    return NormalizationStats(**values)


class ErrorTotals:
    """All metrics exclude padding; dimensions 0:14 are radians, 14:16 suction."""
    def __init__(self):
        self.count = 0
        self.normalized = np.zeros(16, np.float64)
        self.native = np.zeros(16, np.float64)
        self.wrong = np.zeros(2, np.int64)

    def add(self, prediction, target, valid, std, mean):
        p, t = prediction[valid], target[valid]
        error = np.abs(p - t)
        self.count += len(p)
        self.normalized += error.sum(axis=0, dtype=np.float64)
        self.native += (error * std).sum(axis=0, dtype=np.float64)
        self.wrong += (((p[:, 14:] * std[14:] + mean[14:]) >= .5) !=
                       ((t[:, 14:] * std[14:] + mean[14:]) >= .5)).sum(axis=0)

    def report(self):
        if not self.count:
            raise ValueError('Empty validation targets')
        joint = np.rad2deg(self.native[:14] / self.count)
        return dict(valid_targets=self.count, normalized_l1=float(self.normalized.mean() / self.count),
            joint_mae_deg=float(joint.mean()), left_joint_mae_deg=float(joint[:7].mean()),
            right_joint_mae_deg=float(joint[7:].mean()), per_joint_mae_deg=joint.tolist(),
            native_mae_per_dimension=(self.native / self.count).tolist(),
            suction_error_fraction=(self.wrong / self.count).tolist())


@torch.inference_mode()
def evaluate(model, loader, stats):
    model.eval()
    totals = {name: ErrorTotals() for name in ('all', 'first10', 'horizon0', 'horizon10', 'horizon49', 'suction_on')}
    frames = 0
    std, mean = np.array(stats.action_std), np.array(stats.action_mean)
    start = time.monotonic()
    for raw in loader:
        # FP32 evaluation follows predict_action_chunk; training alone uses BF16.
        batch = {k: v.cuda(non_blocking=True) for k, v in raw.items()}
        pred = model.predict_action_chunk(batch['images'], state=batch['state']).float()
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Non-finite validation prediction')
        p, t = pred.cpu().numpy(), raw['action'].numpy()
        valid = ~raw['action_is_pad'].numpy()
        frames += len(p)
        totals['all'].add(p, t, valid, std, mean)
        totals['first10'].add(p[:, :10], t[:, :10], valid[:, :10], std, mean)
        for h in (0, 10, 49):
            if h < p.shape[1]:
                totals[f'horizon{h}'].add(p[:, h], t[:, h], valid[:, h], std, mean)
        on = valid & (t[..., 14] * std[14] + mean[14] >= .5)
        totals['suction_on'].add(p, t, on, std, mean)
    return dict(frames=frames, seconds=time.monotonic() - start,
        **{k: v.report() if v.count else None for k, v in totals.items()})


def save_checkpoint(output, step, model, stats, optimizer):
    parent = output / 'checkpoints'
    parent.mkdir(exist_ok=True)
    final = parent / f'{step:06d}'
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{step:06d}-', dir=parent))
    try:
        model.save_pretrained(temporary / 'pretrained_model')
        stats.save(temporary / 'pretrained_model/normalization.json')
        torch.save(dict(step=step, optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state()), temporary / 'training_state.pt')
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    update_link(parent / 'last', final.name)
    return final


def update_link(path, target):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, path)


@torch.inference_mode()
def benchmark_and_verify(model, checkpoint, loader, stats):
    model.eval()
    raw = next(iter(loader))
    images, state = raw['images'][:1].cuda(), raw['state'][:1].cuda()
    reference = model.predict_action_chunk(images, state=state)
    loaded = InterACT3DPolicy.from_pretrained(checkpoint / 'pretrained_model').cuda().eval()
    restored = loaded.predict_action_chunk(images, state=state)
    torch.testing.assert_close(reference, restored, atol=1e-6, rtol=1e-6)
    assert not loaded.config.use_point_cloud and loaded.geometry is None
    assert reference.shape == (1, 50, 16)
    del loaded, restored
    for _ in range(20):
        model.predict_action_chunk(images, state=state)
    durations = []
    for _ in range(100):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model.predict_action_chunk(images, state=state)
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end))
    return dict(checkpoint_reload_matches=True, shape=list(reference.shape),
        point_cloud_required=False, gpu=torch.cuda.get_device_name(0), precision='fp32',
        input='batch 1, 3xRGB224, 16 state; preprocessed tensors already on GPU',
        chunk_steps=50, warmups=20, iterations=100, mean_ms=float(np.mean(durations)),
        median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations, 95)),
        max_ms=float(np.max(durations)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('far', 'near'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--initialization', choices=('imagenet', 'mixed'), default='imagenet')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--min-steps', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.save_every, args.min_steps, args.patience,
           args.batch_size, args.lr, args.backbone_lr) <= 0 or args.workers < 0:
        parser.error('Invalid numeric training parameter')
    if args.min_steps > args.steps:
        parser.error('min-steps must be <= steps')
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    provenance = validate_provenance(args.stage)
    args.output.mkdir(parents=True, exist_ok=False)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    config = InterACT3DConfig.from_pretrained(args.source)
    if config.use_point_cloud or config.state_layout != 'dual_arm_suction':
        raise ValueError('Expected RGB-only, dual-suction pretrained policy')
    started = time.monotonic()
    write_json(args.output / 'status.json', dict(state='baseline', stage=args.stage, pid=os.getpid()))
    original_stats = NormalizationStats.load(args.source / 'normalization.json')
    baseline_model = InterACT3DPolicy.from_pretrained(args.source).cuda()
    baseline_loader = make_loader(args.stage, 'val', config, original_stats, args.batch_size, args.workers)
    baseline = evaluate(baseline_model, baseline_loader, original_stats)
    write_json(args.output / 'baseline.json', baseline)
    if args.initialization == 'mixed':
        model, stats = baseline_model, original_stats
    else:
        # Reset after loading the baseline so its constructor does not change initialization.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        stats = task_statistics(args.stage)
        model = InterACT3DPolicy(config).cuda()
    del baseline_model, baseline_loader
    loaders = {s: make_loader(args.stage, s, model.config, stats, args.batch_size, args.workers)
               for s in ('train', 'val')}
    backbone_ids = {id(p) for p in model.rgb.features.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in backbone_ids], lr=args.lr),
        dict(params=list(model.rgb.features.parameters()), lr=args.backbone_lr)], weight_decay=1e-4)
    manifest = dict(stage=args.stage, method=('Independent task-only training; ImageNet ResNet18 initialization, random interACT layers'
        if args.initialization == 'imagenet' else 'Task-specific full-parameter fine-tune of completed mixed-task interACT'),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source=str(args.source), source_sha256=sha256(args.source / 'model.safetensors'),
        normalization=('Computed from this task training segments only; no other-task or validation statistics'
            if args.initialization == 'imagenet' else 'Preserved original training-split normalization'),
        baseline_normalization_sha256=sha256(args.source / 'normalization.json'),
        fps=20, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        architecture=model.config.__dict__, data=provenance,
        selection_metric='Full held-out valid-target mean absolute joint error over all 14 joints (degrees), FP32',
        validation_limits='Whole demonstrations held out from gradients; used for selection, not an independent test. Offline teacher-forced predictions cannot establish autonomous contact success.',
        right_suction_limit='All source right-suction labels are off; opening right suction is not learned.')
    source_dir = args.output / 'source_snapshot'
    shutil.copytree(PROJECT / 'src', source_dir / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, source_dir / 'train.py')
    write_json(args.output / 'manifest.json', manifest)
    initial = evaluate(model, loaders['val'], stats) if args.initialization == 'imagenet' else baseline
    log(args.output / 'validation.jsonl', dict(step=0, elapsed_seconds=time.monotonic()-started, **initial))
    checkpoint = save_checkpoint(args.output, 0, model, stats, optimizer)
    update_link(args.output / 'checkpoints/best', checkpoint.name)
    best = dict(step=0, checkpoint=str(checkpoint), score=initial['all']['joint_mae_deg'])
    write_json(args.output / 'best_validation.json', best)
    iterator = iter(loaders['train'])
    stagnant = 0
    loss_sum = 0.
    log_start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    first_weight = model.decoder.left_head.weight.detach().clone()
    stop_reason = 'maximum_steps'
    for step in range(1, args.steps + 1):
        model.train()
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loaders['train'])
            raw = next(iterator)
        batch = {k: v.cuda(non_blocking=True) for k, v in raw.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(images=batch['images'], state=batch['state'], actions=batch['action'],
                         action_is_pad=batch['action_is_pad'])['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite training loss at {step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(loss.detach())
        if step == 1:
            difference = float((first_weight-model.decoder.left_head.weight).abs().max())
            if difference <= 0:
                raise ValueError('Optimizer did not change weights')
            log(args.output / 'training.jsonl', dict(event='first_optimizer_update', step=1,
                head_weight_max_delta=difference, loss=float(loss.detach()), grad_norm=float(norm)))
            del first_weight
        if step % 50 == 0:
            record = dict(step=step, loss=loss_sum / 50, grad_norm=float(norm),
                elapsed_seconds=time.monotonic()-started, steps_per_second=50/(time.monotonic()-log_start),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            log(args.output / 'training.jsonl', record)
            write_json(args.output / 'status.json', dict(state='training', stage=args.stage, pid=os.getpid(),
                best_step=best['step'], best_score=best['score'], **record))
            loss_sum = 0.
            log_start = time.monotonic()
        is_best = False
        evaluated = step % args.eval_every == 0 or step == args.steps
        if evaluated:
            validation = evaluate(model, loaders['val'], stats)
            log(args.output / 'validation.jsonl', dict(step=step, elapsed_seconds=time.monotonic()-started, **validation))
            score = validation['all']['joint_mae_deg']
            is_best = score < best['score']
            stagnant = 0 if is_best else stagnant + 1
        should_stop = evaluated and step >= args.min_steps and stagnant >= args.patience
        if step % args.save_every == 0 or step == args.steps or is_best or should_stop:
            checkpoint = save_checkpoint(args.output, step, model, stats, optimizer)
            if is_best:
                best = dict(step=step, checkpoint=str(checkpoint), score=score)
                update_link(args.output / 'checkpoints/best', checkpoint.name)
                write_json(args.output / 'best_validation.json', best)
        if should_stop:
            stop_reason = 'validation_patience'
            break
    del optimizer, iterator, batch, loss
    selected = Path(best['checkpoint'])
    del model
    torch.cuda.empty_cache()
    model = InterACT3DPolicy.from_pretrained(selected / 'pretrained_model').cuda().eval()
    metrics = evaluate(model, loaders['val'], stats)
    np.testing.assert_allclose(metrics['all']['joint_mae_deg'], best['score'], rtol=1e-6, atol=1e-6)
    timing = benchmark_and_verify(model, selected, loaders['val'], stats)
    update_link(args.output / 'pretrained_model', f'checkpoints/{selected.name}/pretrained_model')
    result = dict(stage=args.stage, state='complete', executed_steps=step, best_step=best['step'],
        stop_reason=stop_reason, elapsed_seconds=time.monotonic()-started,
        baseline=baseline, selected=metrics, inference=timing,
        pretrained_model=str(args.output / 'pretrained_model'),
        model_sha256=sha256(selected / 'pretrained_model/model.safetensors'),
        normalization_sha256=sha256(selected / 'pretrained_model/normalization.json'),
        baseline_joint_mae_deg=baseline['all']['joint_mae_deg'],
        selected_joint_mae_deg=metrics['all']['joint_mae_deg'],
        reduction_percent=100*(1-metrics['all']['joint_mae_deg']/baseline['all']['joint_mae_deg']))
    write_json(args.output / 'result.json', result)
    write_json(args.output / 'status.json', dict(state='complete', stage=args.stage,
        executed_steps=step, best_step=best['step'], result=str(args.output / 'result.json')))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
