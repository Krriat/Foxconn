#!/usr/bin/env python3
"""Compute held-out action errors without connecting to robot hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xcore_interact.data import InterACTWindowDataset, NormalizationStats
from xcore_interact.policy import InterACT3DPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="local/xcore_bimanual_interact3d")
    parser.add_argument("--episodes", required=True, help="comma-separated held-out episode indices")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes = sorted({int(value) for value in args.episodes.split(",")})
    policy_path = args.policy_path.expanduser().resolve()
    model = InterACT3DPolicy.from_pretrained(policy_path).cuda().eval()
    stats = NormalizationStats.load(policy_path / "normalization.json")
    dataset = InterACTWindowDataset(
        args.dataset_root,
        args.dataset_repo_id,
        chunk_size=model.config.chunk_size,
        image_size=model.config.image_size,
        stats=stats,
        episodes=episodes,
        use_point_cloud=model.config.use_point_cloud,
        state_layout=model.config.state_layout,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)
    absolute_sum = torch.zeros(16, device="cuda")
    valid_count = torch.zeros(1, device="cuda")
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            prediction = model.predict_action_chunk(
                batch["images"], points=batch.get("points"), state=batch["state"]
            )
            valid = (~batch["action_is_pad"]).unsqueeze(-1)
            absolute_sum += ((prediction - batch["action"]).abs() * valid).sum(dim=(0, 1))
            valid_count += valid.sum()
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
    normalized_mae = absolute_sum / valid_count.clamp_min(1)
    action_std = torch.tensor(stats.action_std, device="cuda")
    report = {
        "episodes": episodes,
        "batches": batches,
        "normalized_mae_per_dimension": normalized_mae.cpu().tolist(),
        "native_unit_mae_per_dimension": (normalized_mae * action_std).cpu().tolist(),
        "normalized_mae": float(normalized_mae.mean()),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
