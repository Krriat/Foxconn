#!/usr/bin/env python3
"""Offline trajectory audit. Reads saved data/weights; has no ROS or robot code."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("head", "left_wrist", "right_wrist")
HOME = np.array([.795451, .412276, -.585123, 1.905242, .530428, .246453, .198464,
                 -.671164, .310045, .326427, 1.877737, -.441627, .312674, -.453450])


def load_split(path):
    tables = [pq.read_table(p) for p in sorted((path / "data").rglob("*.parquet"))]
    table = pa.concat_tables(tables).sort_by([("index", "ascending")])
    arrays = {key: np.asarray(table[key].to_pylist(), dtype=np.float32 if key in
              ("observation.state", "action") else np.int64) for key in
              ("observation.state", "action", "episode_index", "frame_index", "index")}
    episodes = pa.concat_tables([pq.read_table(p) for p in
                    sorted((path / "meta/episodes").rglob("*.parquet"))]).to_pylist()
    source = json.loads((path / "meta/keyboard_teach_source.json").read_text())
    return arrays, episodes, source


def frames(path, episode, stride):
    """Decode the same H264 RGB frames consumed in training, at exact frame times."""
    containers, streams, iterators = [], [], []
    try:
        for camera in CAMERAS:
            key = f"videos/observation.images.{camera}"
            p = path / f"videos/observation.images.{camera}" / (
                f"chunk-{episode[key + '/chunk_index']:03d}/file-{episode[key + '/file_index']:03d}.mp4")
            container = av.open(str(p))
            containers.append(container)
            stream = container.streams.video[0]
            # A single decoder thread avoids FFmpeg thread-pool stalls when
            # repeatedly seeking the three long concatenated camera videos.
            stream.codec_context.thread_count = 1
            start = episode[key + "/from_timestamp"]
            container.seek(int(round(start / stream.time_base)), stream=stream, backward=True)
            streams.append((stream, start))
            iterators.append(iter(container.decode(stream)))
        for i in range(episode["length"]):
            selected = []
            for iterator, (stream, start) in zip(iterators, streams):
                target = start + i / 20
                while True:
                    frame = next(iterator)
                    timestamp = float(frame.pts * stream.time_base)
                    if timestamp >= target - .001:
                        break
                if abs(timestamp - target) > .001:
                    raise ValueError(f"Video alignment error: {timestamp} != {target}")
                if i % stride == 0:
                    selected.append(frame.to_ndarray(format="rgb24"))
            if i % stride == 0:
                yield i, np.stack(selected)
    finally:
        for container in containers:
            container.close()


def predict_batch(engines, images, states):
    # InterACT training resizes on CPU; keep the identical interpolation here.
    rgb = torch.from_numpy(np.stack(images)).permute(0, 1, 4, 2, 3).contiguous().float().div_(255)
    results = {}
    for name, engine in engines.items():
        state = torch.from_numpy(np.stack(states)).to(engine.device)
        if name == "act":
            gpu_rgb = rgb.to(engine.device)
            batch = {key: (gpu_rgb[:, i] - engine.stats[key + ".mean"]) /
                     (engine.stats[key + ".std"] + 1e-8) for i, key in enumerate(engine.keys)}
            batch["observation.state"] = (state - engine.stats["observation.state.mean"]) / (
                engine.stats["observation.state.std"] + 1e-8)
            batch["observation.images"] = [batch[key] for key in engine.keys]
            prepared = batch
        else:
            size = engine.config["image_size"]
            resized = F.interpolate(rgb.flatten(0, 1), size=(size, size), mode="bilinear",
                                    align_corners=False).unflatten(0, (len(states), 3)).to(engine.device)
            prepared = (resized, None, (state - engine.stats["state_mean"]) / engine.stats["state_std"])
        output = engine.postprocess(engine.forward(prepared))
        assert output.shape == (len(states), 50, 16) and np.isfinite(output).all()
        results[name] = output
    return results


def number_summary(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def dataset_audit(arrays, episodes, source):
    state, action = arrays["observation.state"], arrays["action"]
    rows = []
    for ep, provenance in zip(episodes, source["episodes"]):
        a, b = ep["dataset_from_index"], ep["dataset_to_index"]
        q, act = state[a:b], action[a:b]
        home_dist = np.abs(q[:, :14] - HOME).max(axis=1)
        transitions = np.flatnonzero(np.diff(act[:, 14]) != 0) + 1
        rows.append({"episode_index": ep["episode_index"], "length": len(q), **provenance,
                     "start_suction": act[0, 14:].tolist(), "end_suction": act[-1, 14:].tolist(),
                     "left_suction_transitions": [{"frame": int(t), "seconds": float(t / 20),
                        "from": float(act[t-1, 14]), "to": float(act[t, 14])} for t in transitions],
                     "start_max_joint_error_from_home_rad": float(home_dist[0]),
                     "frames_within_home_0_02_rad": int(np.sum(home_dist <= .02)),
                     "joint_range_rad": np.ptp(q[:, :14], axis=0).tolist(),
                     "start_joints_rad": q[0, :14].tolist()})
    return {"frames": len(state), "segment_count": len(episodes), "raw_episodes": len(source["raw_episode_hashes"]),
            "joint_action_equals_same_time_measured_state_fraction": float(
                (state[:, :14] == action[:, :14]).all(axis=1).mean()),
            "left_suction_on_at_segment_start": sum(r["start_suction"][0] >= .5 for r in rows),
            "right_suction_on_frames": int(np.sum(action[:, 15] >= .5)),
            "left_suction_on_fraction": float(action[:, 14].mean()),
            "near_home_definition": "All 14 measured joints within 0.02 rad of deployed home",
            "near_home_frames": int(np.sum(np.abs(state[:, :14] - HOME).max(axis=1) <= .02)),
            "near_home_left_suction_off_frames": int(np.sum(
                (np.abs(state[:, :14] - HOME).max(axis=1) <= .02) & (state[:, 14] < .5))),
            "segments": rows}


def model_metrics(arrays, episodes, predictions, indices):
    state, action = arrays["observation.state"], arrays["action"]
    ends = np.empty(len(state), dtype=np.int64)
    for ep in episodes:
        ends[ep["dataset_from_index"]:ep["dataset_to_index"]] = ep["dataset_to_index"]
    output = {"observations": len(indices), "horizons": {}}
    for horizon in (0, 1, 5, 9, 19, 49):
        valid = indices + horizon < ends[indices]
        ix = indices[valid]
        predicted, target, current = predictions[valid, horizon], action[ix + horizon], state[ix]
        err, base = np.abs(predicted[:, :14] - target[:, :14]), np.abs(current[:, :14] - target[:, :14])
        delta, wanted = predicted[:, :14] - current[:, :14], target[:, :14] - current[:, :14]
        # Per-joint direction is assessed only when the demonstration moves >0.01 rad.
        moving = np.abs(wanted) > .01
        centered_delta = predicted[:, :14] - predictions[valid, 0, :14]
        entry = {"samples": len(ix), "seconds_ahead": horizon / 20,
                 "joint_mae_rad": float(err.mean()), "left_joint_mae_rad": float(err[:, :7].mean()),
                 "right_joint_mae_rad": float(err[:, 7:].mean()), "joint_abs_error_rad": number_summary(err),
                 "per_joint_mae_rad": err.mean(axis=0).tolist(),
                 "hold_current_joint_mae_rad": float(base.mean()),
                 "mae_ratio_to_hold_current": float(err.mean() / base.mean()) if base.mean() > 0 else None,
                 "diagnostic_start_anchored_joint_mae_rad": float(np.abs(current[:, :14] + centered_delta - target[:, :14]).mean()),
                 "diagnostic_note": "Anchoring removes the model's first-target offset offline; it is not a deployed correction or a closed-loop test.",
                 "moving_joint_targets": int(moving.sum()),
                 "moving_joint_mae_rad": float(err[moving].mean()) if moving.any() else None,
                 "moving_hold_mae_rad": float(base[moving].mean()) if moving.any() else None,
                 "moving_direction_accuracy_from_measured": float(
                     (delta[moving] * wanted[moving] > 0).mean()) if moving.any() else None,
                 "moving_direction_accuracy_within_chunk": float(
                     (centered_delta[moving] * wanted[moving] > 0).mean()) if moving.any() else None,
                 "left_suction_accuracy_threshold_0_6": float(((predicted[:, 14] >= .6) ==
                                                          (target[:, 14] >= .5)).mean()),
                 "left_suction_mean_absolute_error": float(np.abs(predicted[:, 14] - target[:, 14]).mean()),
                 "right_suction_predicted_on_fraction": float((predicted[:, 15] >= .6).mean())}
        output["horizons"][str(horizon)] = entry
    lookup = {int(i): k for k, i in enumerate(indices)}
    replay_rows, all_err, all_deltas, all_gt_deltas, all_boundaries, all_hold_err = [], [], [], [], [], []
    for ep in episodes:
        a, b = ep["dataset_from_index"], ep["dataset_to_index"]
        stitched = np.empty((b-a, 16), dtype=np.float32)
        held = np.empty((b-a, 14), dtype=np.float32)
        used = []
        for t in range(0, b-a, 10):
            if a+t not in lookup:
                break
            n = min(10, b-a-t)
            stitched[t:t+n] = predictions[lookup[a+t], :n]
            held[t:t+n] = state[a+t, :14]
            used.append(t)
        if len(used) != (b-a+9)//10:
            continue
        truth = action[a:b]
        errors = np.abs(stitched[:, :14] - truth[:, :14])
        changes, gt_changes = np.diff(stitched[:, :14], axis=0), np.diff(truth[:, :14], axis=0)
        boundaries = np.array([t-1 for t in used[1:]], dtype=np.int64)
        suction = []
        previous = bool(state[a, 14] >= .5)
        for x in stitched[:, 14]:
            previous = True if x >= .6 else False if x <= .4 else previous
            suction.append(previous)
        suction = np.array(suction)
        predicted_switches = np.flatnonzero(np.diff(suction.astype(int)) != 0) + 1
        row = {"episode_index": ep["episode_index"], "frames": b-a,
               "joint_mae_rad": float(errors.mean()), "joint_max_error_rad": float(errors.max()),
               "hold_current_joint_mae_rad": float(np.abs(held-truth[:, :14]).mean()),
               "per_joint_mae_rad": errors.mean(axis=0).tolist(),
               "model_adjacent_step_abs_rad": number_summary(np.abs(changes)),
               "demo_adjacent_step_abs_rad": number_summary(np.abs(gt_changes)),
               "chunk_boundary_max_joint_jump_rad": number_summary(np.abs(changes[boundaries]).max(axis=1)),
               "demo_same_boundary_max_joint_jump_rad": number_summary(np.abs(gt_changes[boundaries]).max(axis=1)),
               "left_suction_hysteresis_accuracy": float((suction == (truth[:, 14] >= .5)).mean()),
               "left_suction_hysteresis_switch_frames": predicted_switches.tolist()}
        replay_rows.append(row)
        all_err.append(errors); all_deltas.append(changes); all_gt_deltas.append(gt_changes)
        all_hold_err.append(np.abs(held-truth[:, :14]))
        all_boundaries.append(np.abs(changes[boundaries]).max(axis=1))
    output["teacher_forced_first10_replay"] = {"episodes": replay_rows,
        "joint_mae_rad": float(np.concatenate(all_err).mean()),
        "hold_current_joint_mae_rad": float(np.concatenate(all_hold_err).mean()),
        "model_adjacent_step_abs_rad": number_summary(np.abs(np.concatenate(all_deltas))),
        "demo_adjacent_step_abs_rad": number_summary(np.abs(np.concatenate(all_gt_deltas))),
        "chunk_boundary_max_joint_jump_rad": number_summary(np.concatenate(all_boundaries)),
        "interpretation": "Every 10 frames uses true demonstration images AND state, then stitches predicted actions 0..9. This is not a closed-loop simulation or task success test."}
    return output


def plot_validation(path, arrays, episodes, predictions, indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {int(i): k for k, i in enumerate(indices)}
    for ep in episodes:
        a, b = ep["dataset_from_index"], ep["dataset_to_index"]
        t = np.arange(b-a) / 20
        fig, axs = plt.subplots(5, 3, figsize=(17, 13))
        for name, pred in predictions.items():
            stitched = np.concatenate([pred[lookup[a+i], :min(10,b-a-i)] for i in range(0,b-a,10)])
            for j in range(14):
                axs.flat[j].plot(t, stitched[:, j], label=name, alpha=.8, lw=.9)
            axs.flat[14].plot(t, stitched[:, 14], label=name, alpha=.8, lw=.9)
        for j in range(14):
            ax = axs.flat[j]
            ax.plot(t, arrays["action"][a:b,j], "k--", label="demonstration", lw=1)
            ax.set_title(f"{'Left' if j<7 else 'Right'} joint {j%7+1} (rad)")
            ax.grid(alpha=.2)
        axs.flat[14].plot(t, arrays["action"][a:b,14], "k--", label="demo suction", lw=1.4)
        axs.flat[14].axhline(.6, color="r", alpha=.3)
        axs.flat[14].set_title("Left suction output (on threshold=0.6)")
        axs.flat[14].legend()
        fig.suptitle(f"Validation segment {ep['episode_index']}: true observation replay, replan every 10 frames (0.5 s)")
        fig.supxlabel("Demonstration time (s)")
        fig.tight_layout()
        fig.savefig(path / f"val_episode_{ep['episode_index']:02d}_trajectories.png", dpi=130)
        plt.close(fig)


def main():
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", nargs="+", choices=("val", "train"), default=("val", "train"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = ROOT / "training_verification/jetson_bundle_20261112"
    sys.path.insert(0, str(bundle))
    from jetson_inference_runtime import Engine
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    engines = {name: Engine(bundle, name) for name in ("act", "interact")}
    started = time.monotonic()
    report = {"started_at": datetime.now().astimezone().isoformat(), "mode": "offline_no_ROS_no_robot",
              "precision": "FP32, TF32 allowed, eval mode, zero ACT VAE latent as deployment",
              "image_source": "exact timestamp matched training H264 videos, PyAV RGB24",
              "bundle_equivalence": json.loads((bundle / "local_equivalence.json").read_text()),
              "checkpoint_sha256": {name: hashlib.sha256((bundle / name / "model.safetensors").read_bytes()).hexdigest()
                                     for name in engines}, "splits": {}}
    if (args.output / "report.json").exists():
        previous = json.loads((args.output / "report.json").read_text())
        if previous["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("Output directory contains results for different weights")
        report["splits"] = previous["splits"]
    with torch.inference_mode():
        for split in args.splits:
            stride = 1 if split == "val" else args.train_stride
            dataset = ROOT / "prepared_data/suction_20hz_v1" / split
            arrays, episodes, source = load_split(dataset)
            audit = dataset_audit(arrays, episodes, source)
            predictions = {name: [] for name in engines}
            indices, image_batch, state_batch, batch_indices = [], [], [], []
            def flush():
                if not image_batch:
                    return
                result = predict_batch(engines, image_batch, state_batch)
                for name in engines:
                    predictions[name].append(result[name])
                indices.extend(batch_indices)
                image_batch.clear(); state_batch.clear(); batch_indices.clear()
            for ep in episodes:
                for frame, rgb in frames(dataset, ep, stride):
                    index = ep["dataset_from_index"] + frame
                    image_batch.append(rgb); state_batch.append(arrays["observation.state"][index]); batch_indices.append(index)
                    if len(image_batch) == args.batch_size:
                        flush()
                flush()
                print(f"{split} episode {ep['episode_index']} done, observations={len(indices)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
            indices = np.asarray(indices)
            predictions = {name: np.concatenate(rows) for name, rows in predictions.items()}
            np.savez_compressed(args.output / f"{split}_predictions.npz", indices=indices,
                                state=arrays["observation.state"], action=arrays["action"],
                                episode_index=arrays["episode_index"], **predictions)
            report["splits"][split] = {"stride": stride, "dataset": audit,
                "models": {name: model_metrics(arrays, episodes, pred, indices) for name, pred in predictions.items()}}
            (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
            if split == "val":
                try:
                    plot_validation(args.output, arrays, episodes, predictions, indices)
                except ModuleNotFoundError as error:
                    if error.name != "matplotlib":
                        raise
                    print("Plotting deferred: use system Python with plot_box_opening.py", flush=True)
            print(f"{split} metrics saved", flush=True)
    report["duration_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
