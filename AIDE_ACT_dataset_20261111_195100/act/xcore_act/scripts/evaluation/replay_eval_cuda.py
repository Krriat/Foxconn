#!/usr/bin/env python3
"""Replay a recorded LeRobot eval through one policy on CUDA.

This is an offline diagnostic: it reads the recorded joint state and decoded
camera frames, runs the policy once for every frame, and never connects to or
commands a robot.  The resulting predictions can be compared across machines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action


KNOWN_IMAGE_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def resolve_policy_image_keys(config, dataset) -> tuple[str, ...]:
    keys = tuple(
        key for key in config.input_features if key.startswith("observation.images.")
    )
    if not keys:
        keys = tuple(key for key in KNOWN_IMAGE_KEYS if key in dataset.meta.camera_keys)
    available = set(dataset.meta.camera_keys)
    missing = set(keys).difference(available)
    if missing:
        raise RuntimeError(
            "Policy/dataset camera mismatch: policy requires "
            f"{sorted(keys)}, dataset provides {sorted(available)}; "
            f"missing {sorted(missing)}"
        )
    ignored = available.difference(keys)
    if ignored:
        print(
            "Dataset has additional cameras not used by this checkpoint: "
            + ", ".join(sorted(ignored)),
            flush=True,
        )
    return keys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_image_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.rint(image * 255.0).clip(0, 255).astype(np.uint8)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay an xCore eval dataset through an ACT checkpoint on CUDA."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument(
        "--output-prefix",
        help="Output path without extension (default: <dataset-root>/cuda_replay).",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--task",
        default="pick up the green cube and put it into the blue box",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    policy_path = Path(args.policy_path).expanduser().resolve()
    model_path = policy_path / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else dataset_root / "cuda_replay"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    repo_id = f"local/{dataset_root.name}"
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        video_backend="pyav",
    )
    frame_count = len(dataset)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        frame_count = min(frame_count, args.max_frames)

    config = PreTrainedConfig.from_pretrained(policy_path)
    config.device = "cuda"
    config.use_amp = False
    config.n_action_steps = 1
    config.pretrained_path = str(policy_path)
    config.pretrained_backbone_weights = None
    image_keys = resolve_policy_image_keys(config, dataset)

    policy = make_policy(config, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    states = np.empty((frame_count, 8), dtype=np.float32)
    recorded_actions = np.empty((frame_count, 8), dtype=np.float32)
    replay_actions = np.empty((frame_count, 8), dtype=np.float32)
    input_digest = hashlib.sha256()
    inference_seconds: list[float] = []
    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model SHA256: {sha256_file(model_path)}", flush=True)
    print(f"Replaying {frame_count} frames without connecting to the robot", flush=True)

    for index in range(frame_count):
        item = dataset[index]
        state = item["observation.state"].detach().cpu().numpy().astype(np.float32)
        recorded_action = item["action"].detach().cpu().numpy().astype(np.float32)
        observation: dict[str, np.ndarray] = {"observation.state": state.copy()}
        input_digest.update(state.astype("<f4", copy=False).tobytes())
        for key in image_keys:
            image = tensor_image_to_uint8_hwc(item[key])
            observation[key] = image
            input_digest.update(image.tobytes())

        torch.cuda.synchronize()
        start = time.perf_counter()
        action = predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=args.task,
            robot_type="xcore_right_arm",
        )
        torch.cuda.synchronize()
        inference_seconds.append(time.perf_counter() - start)

        states[index] = state
        recorded_actions[index] = recorded_action
        replay_actions[index] = action.detach().cpu().numpy().astype(np.float32)
        if (index + 1) % 10 == 0 or index + 1 == frame_count:
            print(f"Inference {index + 1}/{frame_count}", flush=True)

    replay_delta = replay_actions[:, :7] - states[:, :7]
    recorded_delta = recorded_actions[:, :7] - states[:, :7]
    observed_net = states[-1, :7] - states[0, :7]
    late_count = min(30, frame_count)
    late = slice(frame_count - late_count, frame_count)
    lag = min(2, frame_count - 1)
    future_motion = states[lag:, :7] - states[:-lag, :7]

    report = {
        "dataset_root": str(dataset_root),
        "policy_path": str(policy_path),
        "model_sha256": sha256_file(model_path),
        "input_sha256": input_digest.hexdigest(),
        "prediction_float32_sha256": hashlib.sha256(
            replay_actions.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "use_amp": False,
        "n_action_steps": 1,
        "image_keys": list(image_keys),
        "frames": frame_count,
        "mean_inference_ms": float(np.mean(inference_seconds) * 1000.0),
        "observed_final_minus_initial_rad": observed_net.tolist(),
        "recorded_action_minus_state_mean_rad": recorded_delta.mean(axis=0).tolist(),
        "replay_action_minus_state_mean_rad": replay_delta.mean(axis=0).tolist(),
        "replay_last_30_action_minus_state_mean_rad": replay_delta[late].mean(axis=0).tolist(),
        "replay_last_30_j1_positive_ratio": float((replay_delta[late, 0] > 0).mean()),
        "replay_last_30_j4_negative_ratio": float((replay_delta[late, 3] < 0).mean()),
        "replay_vs_recorded_action_mae": np.abs(replay_actions - recorded_actions).mean(axis=0).tolist(),
        "replay_vs_recorded_action_max_abs": np.abs(replay_actions - recorded_actions).max(axis=0).tolist(),
        "replay_error_vs_measured_motion_lag_2_correlation": [
            correlation(replay_delta[:-lag, joint], future_motion[:, joint])
            for joint in range(7)
        ],
    }

    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    np.savez_compressed(
        npz_path,
        state=states,
        recorded_action=recorded_actions,
        replay_action=replay_actions,
    )
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\nMeasured state change (last - first):")
    for joint, value in enumerate(observed_net, 1):
        print(f"  J{joint}: {value:+.6f} rad")
    print("Replay model target - state, last 30 frames:")
    for joint, value in enumerate(replay_delta[late].mean(axis=0), 1):
        print(f"  J{joint}: {value:+.6f} rad")
    print(
        "Replay late direction: "
        f"J1 positive {report['replay_last_30_j1_positive_ratio']:.1%}, "
        f"J4 negative {report['replay_last_30_j4_negative_ratio']:.1%}"
    )
    print(f"Report: {json_path}")
    print(f"Arrays: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
