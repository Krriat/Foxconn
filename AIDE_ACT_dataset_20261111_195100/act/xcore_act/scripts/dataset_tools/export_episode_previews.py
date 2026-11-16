#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".hf_datasets_cache"
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE))

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class RawEpisodeDataset:
    """Read saved PNG episodes without exporting or modifying their data."""

    def __init__(self, root: Path):
        self.root = root
        schema = json.loads((root / "schema.json").read_text())
        self.episodes = {}
        for path in sorted(root.glob("episode_*")):
            if not path.is_dir() or not path.name.removeprefix("episode_").isdigit():
                continue
            metadata = path / "episode.json"
            if not metadata.is_file():
                continue
            meta = json.loads(metadata.read_text())
            if not meta.get("complete") or meta.get("frames", 0) < 1:
                continue
            episode = int(path.name.removeprefix("episode_"))
            self.episodes[episode] = (path, meta)
        if not self.episodes:
            raise ValueError(f"No completed raw episodes found at {root}")
        self.meta = SimpleNamespace(camera_keys=list(schema["cameras"]),
                                    total_episodes=max(self.episodes) + 1)
        self._trajectory_episode = None
        self._trajectory = {}

    def episode_indices(self, episode: int):
        if episode not in self.episodes:
            return []
        return [(episode, frame) for frame in range(self.episodes[episode][1]["frames"])]

    def __getitem__(self, reference):
        episode, frame = reference
        path, _ = self.episodes[episode]
        item = {}
        for camera in self.meta.camera_keys:
            filename = path / "images" / camera / f"{frame:06d}.png"
            image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Missing or unreadable image: {filename}")
            item[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return item

    def frame_label(self, reference, name: str):
        episode, frame = reference
        path, meta = self.episodes[episode]
        if self._trajectory_episode != episode:
            trajectory = path / "trajectory.npz"
            self._trajectory = {}
            if trajectory.is_file():
                with np.load(trajectory, allow_pickle=False) as arrays:
                    self._trajectory = {key: arrays[key] for key in
                                        ("ros_timestamp", "state", "action") if key in arrays}
            self._trajectory_episode = episode
        stamps = self._trajectory.get("ros_timestamp")
        elapsed = float(stamps[frame] - stamps[0]) if stamps is not None else frame / meta["fps"]
        label = f"ep {episode:03d} {name} f={frame} t={elapsed:.2f}s"
        if meta.get("schema_id") == "keyboard_dual_arm_suction_v1":
            action, state = self._trajectory.get("action"), self._trajectory.get("state")
            if action is not None and state is not None:
                label += (f" L={int(action[frame, 14])}/{int(state[frame, 14])}"
                          f" R={int(action[frame, 15])}/{int(state[frame, 15])}")
        return label


def parse_episode_list(raw: str, total: int) -> list[int]:
    if not raw:
        return list(range(total))
    episodes: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.update(range(int(start), int(end) + 1))
        else:
            episodes.add(int(part))
    return sorted(ep for ep in episodes if 0 <= ep < total)


def image_to_uint8_hwc(value):
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def make_preview_frame(item, camera_keys: list[str], width: int, label: str):
    images = []
    for key in camera_keys:
        img = image_to_uint8_hwc(item[key])
        scale = width / img.shape[1]
        height = int(img.shape[0] * scale)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, key, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        images.append(img)

    # Cameras may have different aspect ratios (for example, the head camera
    # is 16:9 while the wrist camera is 4:3).  Resizing both to the same width
    # therefore produces different heights.  Pad the shorter images instead
    # of stretching or cropping them so they can be concatenated safely.
    target_height = max(img.shape[0] for img in images)
    padded_images = []
    for img in images:
        pad = target_height - img.shape[0]
        top = pad // 2
        bottom = pad - top
        padded_images.append(
            cv2.copyMakeBorder(
                img,
                top,
                bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        )
    frame = (
        np.concatenate(padded_images, axis=1)
        if len(padded_images) > 1
        else padded_images[0]
    )
    cv2.putText(frame, label, (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "smolvla":
        return REPO_ROOT / path
    if path.exists():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def resolve_dataset_root(raw: str) -> Path:
    path = Path(raw).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    elif path.parts and path.parts[0] == "smolvla":
        candidates.append(REPO_ROOT / path)
        candidates.append(path)
    else:
        candidates.append(path)
        candidates.append(REPO_ROOT / path)
    for candidate in candidates:
        if ((candidate / "meta" / "info.json").exists() or
                (candidate / "schema.json").is_file()):
            return candidate
    return path


def make_contact_sheet(frames: list[np.ndarray], columns: int) -> np.ndarray:
    if not frames:
        raise ValueError("no frames for contact sheet")
    columns = max(1, columns)
    rows = int(np.ceil(len(frames) / columns))
    h, w = frames[0].shape[:2]
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        row = i // columns
        col = i % columns
        sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame
    return sheet


def export_episode(ds, episode: int, out_dir: Path, args) -> None:
    camera_keys = args.cameras or list(ds.meta.camera_keys)
    indices = (ds.episode_indices(episode) if isinstance(ds, RawEpisodeDataset) else
               [i for i, ep in enumerate(ds.hf_dataset["episode_index"]) if int(ep) == episode])
    if not indices:
        print(f"episode {episode}: no frames")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    keyframe_dir = out_dir / "keyframes"
    keyframe_dir.mkdir(exist_ok=True)
    for name, src_idx in {
        "start": indices[0],
        "middle": indices[len(indices) // 2],
        "end": indices[-1],
    }.items():
        item = ds[src_idx]
        frame = make_preview_frame(
            item,
            camera_keys,
            args.camera_width,
            ds.frame_label(src_idx, name) if isinstance(ds, RawEpisodeDataset)
            else f"episode {episode:03d} {name}",
        )
        cv2.imwrite(str(keyframe_dir / f"episode_{episode:03d}_{name}.jpg"), frame)

    sample_count = min(args.samples, len(indices))
    sample_positions = np.linspace(0, len(indices) - 1, sample_count, dtype=int)
    sample_frames = []
    for sample_i, pos in enumerate(sample_positions):
        src_idx = indices[int(pos)]
        item = ds[src_idx]
        sample_frames.append(
            make_preview_frame(
                item,
                camera_keys,
                args.camera_width,
                ds.frame_label(src_idx, f"{sample_i + 1}/{sample_count}")
                if isinstance(ds, RawEpisodeDataset)
                else f"episode {episode:03d} sample {sample_i + 1}/{sample_count}",
            )
        )
    sheet = make_contact_sheet(sample_frames, args.columns)
    sheet_path = out_dir / f"episode_{episode:03d}_contact.jpg"
    cv2.imwrite(str(sheet_path), sheet)

    print(f"episode {episode:03d}: {len(indices)} frames -> {sheet_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export preview images for raw recordings or LeRobot episodes.")
    parser.add_argument("--root", default="smolvla/datasets/right_arm_vr")
    parser.add_argument("--repo-id", default="YOUR_HF_USER/right_arm_vr_test")
    parser.add_argument("--out-dir", default="smolvla/previews/right_arm_vr")
    parser.add_argument("--episodes", default="", help="Episode list, e.g. 0,3,5-8. Default: all")
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()
    if min(args.camera_width, args.samples, args.columns) < 1:
        parser.error("--camera-width, --samples and --columns must be positive")

    root = resolve_dataset_root(args.root)
    if (root / "meta" / "info.json").is_file():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(args.repo_id, root=root, video_backend=args.video_backend)
        episodes = parse_episode_list(args.episodes, ds.meta.total_episodes)
    elif (root / "schema.json").is_file():
        ds = RawEpisodeDataset(root)
        episodes = (parse_episode_list(args.episodes, ds.meta.total_episodes)
                    if args.episodes else sorted(ds.episodes))
    else:
        raise FileNotFoundError(
            f"Dataset not found at {root}; expected raw schema.json or LeRobot meta/info.json."
        )
    print(f"Exporting previews for episodes: {episodes}")
    for episode in episodes:
        export_episode(ds, episode, resolve_path(args.out_dir), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
