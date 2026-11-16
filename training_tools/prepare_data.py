#!/usr/bin/env python3
"""Export group-disjoint train/validation data without changing raw captures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
BASE = WORKSPACE / "AIDE_ACT_dataset_20261111_195100"
sys.path.insert(0, str(BASE / "act/xcore_act/scripts/recording"))
import record_xcore_teach_suction as recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=BASE / "act_teach_keyboard_suction_30hz")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "prepared_data/suction_20hz_v1")
    parser.add_argument("--validation-count", type=int, default=6)
    parser.add_argument("--min-segment-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--limit-raw-episodes", type=int, default=0, help="Small export for pipeline checks only")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class LinkedImageDataset(LeRobotDataset):
        # LeRobot 0.4.3 first stages PNGs and then encodes video. Reuse existing
        # lossless PNGs through hard links; its cleanup only removes those links.
        def _save_image(self, image, fpath, compress_level=1):
            source = self.raw_images[id(image)]
            try:
                os.link(source, fpath)
            except OSError:
                shutil.copyfile(source, fpath)

    raw = args.raw_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output root: {output}")
    settings = json.loads((raw / "schema.json").read_text())
    recorder.validate_schema(settings)
    paths = sorted(raw.glob("episode_*"))
    if args.limit_raw_episodes:
        paths = paths[:args.limit_raw_episodes]
    if not 0 < args.validation_count < len(paths) or args.min_segment_frames < 1:
        raise ValueError("Need nonempty, disjoint train and validation groups and positive segment length")
    shuffled = paths.copy()
    random.Random(args.seed).shuffle(shuffled)
    validation = {p.name for p in shuffled[:args.validation_count]}
    groups = {"train": [p for p in paths if p.name not in validation],
              "val": [p for p in paths if p.name in validation]}
    features = {key: dict(dtype="float32", shape=(16,), names=settings["state_names"])
                for key in ("observation.state", "action")}
    features.update({f"observation.images.{camera}": dict(
        dtype="video", shape=(settings["height"], settings["width"], 3),
        names=["height", "width", "channels"]) for camera in settings["cameras"]})
    output.mkdir(parents=True)
    manifest = dict(raw_root=str(raw), seed=args.seed, fps=settings["fps"],
                    min_segment_frames=args.min_segment_frames, splits={}, complete=False)
    for split, selected in groups.items():
        destination = output / split
        dataset = LinkedImageDataset.create(
            repo_id=f"local/dual_arm_suction_{split}", root=destination,
            fps=int(settings["fps"]), features=features,
            robot_type="xcore_dual_arm_suction_keyboard", use_videos=True,
            video_backend="pyav", vcodec="h264", image_writer_processes=0, image_writer_threads=0,
        )
        mapping, omitted, hashes = [], [], {}
        started = time.monotonic()
        try:
            for episode in selected:
                meta, arrays = recorder.validate_episode(episode, settings, check_images=False)
                hashes[episode.name] = {
                    name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
                    for name in ("episode.json", "trajectory.npz")
                }
                for segment in meta["segments"]:
                    length = segment["end"] - segment["start"]
                    if length < args.min_segment_frames:
                        omitted.append(dict(raw_episode=episode.name, **segment))
                        continue
                    for index in range(segment["start"], segment["end"]):
                        frame = {"observation.state": arrays["state"][index].copy(),
                                 "action": arrays["action"][index].copy(), "task": meta["task"]}
                        dataset.raw_images = {}
                        for camera in settings["cameras"]:
                            source = episode / "images" / camera / f"{index:06d}.png"
                            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                            if image is None or image.shape != (settings["height"], settings["width"], 3) or image.dtype != np.uint8:
                                raise ValueError(f"Invalid RGB frame: {source}")
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            frame[f"observation.images.{camera}"] = image
                            dataset.raw_images[id(image)] = source
                        dataset.add_frame(frame)
                    dataset.save_episode(parallel_encoding=True)
                    mapping.append(dict(episode_index=len(mapping), raw_episode=episode.name, **segment))
                    print(f"{split}: {episode.name} [{segment['start']}:{segment['end']}] "
                          f"saved, segments={len(mapping)}, elapsed={time.monotonic()-started:.0f}s", flush=True)
            dataset.finalize()
        finally:
            dataset.stop_image_writer()
        provenance = dict(schema=settings, episodes=mapping, omitted_segments=omitted,
                          raw_episode_hashes=hashes)
        (destination / "meta/keyboard_teach_source.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        manifest["splits"][split] = dict(
            raw_episodes=[p.name for p in selected], root=str(destination),
            repo_id=f"local/dual_arm_suction_{split}", segments=len(mapping),
            frames=sum(s["end"]-s["start"] for s in mapping),
            omitted_frames=sum(s["end"]-s["start"] for s in omitted),
        )
        (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    assert not (set(manifest["splits"]["train"]["raw_episodes"]) &
                set(manifest["splits"]["val"]["raw_episodes"]))
    manifest["complete"] = True
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
