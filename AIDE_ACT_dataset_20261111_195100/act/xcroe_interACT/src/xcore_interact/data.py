from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .schema import (
    ACTION_KEY,
    HEAD_IMAGE_KEY,
    HEAD_POINT_CLOUD_KEY,
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    infer_state_layout,
)


@dataclass
class NormalizationStats:
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        return cls(**json.loads(Path(path).read_text()))

    def tensors(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {
            name: torch.tensor(getattr(self, name), dtype=torch.float32, device=device)
            for name in ("state_mean", "state_std", "action_mean", "action_std")
        }


class _RunningMoments:
    def __init__(self, width: int):
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.square = np.zeros(width, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.size)
        if not np.isfinite(values).all():
            raise ValueError("dataset normalization encountered NaN or infinity")
        self.count += values.shape[0]
        self.sum += values.sum(axis=0)
        self.square += np.square(values).sum(axis=0)

    def result(self) -> tuple[list[float], list[float]]:
        if self.count == 0:
            raise ValueError("cannot normalize an empty dataset")
        mean = self.sum / self.count
        variance = np.maximum(self.square / self.count - np.square(mean), 1e-8)
        return mean.astype(np.float32).tolist(), np.sqrt(variance).astype(np.float32).tolist()


def compute_normalization_stats(root: str | Path) -> NormalizationStats:
    """Stream state/action columns directly from Parquet without decoding video."""
    import pyarrow.parquet as pq

    state = _RunningMoments(16)
    action = _RunningMoments(16)
    files = sorted((Path(root) / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {Path(root) / 'data'}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=[STATE_KEY, ACTION_KEY]):
            state.add(np.asarray(batch.column(0).to_pylist(), dtype=np.float32))
            action.add(np.asarray(batch.column(1).to_pylist(), dtype=np.float32))
    state_mean, state_std = state.result()
    action_mean, action_std = action.result()
    return NormalizationStats(state_mean, state_std, action_mean, action_std)


def _chw_rgb(image: torch.Tensor, image_size: int) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 3:
        raise ValueError(f"image must have three dimensions, got {tuple(image.shape)}")
    if image.shape[0] != 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.shape[0] != 3:
        raise ValueError(f"image must have three channels, got {tuple(image.shape)}")
    if image.max() > 1.5:
        image = image / 255.0
    return F.interpolate(
        image.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(0)


class InterACTWindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        *,
        chunk_size: int,
        image_size: int,
        stats: NormalizationStats,
        video_backend: str = "pyav",
        episodes: list[int] | None = None,
        use_point_cloud: bool = False,
        state_layout: str | None = None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(root).expanduser().resolve()
        info = json.loads((root / "meta" / "info.json").read_text())
        actual_layout = infer_state_layout(info)
        if state_layout is not None and state_layout != actual_layout:
            raise ValueError(f"dataset layout {actual_layout} != checkpoint layout {state_layout}")
        self.use_point_cloud = use_point_cloud
        if use_point_cloud and HEAD_POINT_CLOUD_KEY not in info["features"]:
            raise ValueError("head point-cloud feature is required by this checkpoint")
        fps = float(info["fps"])
        self.dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps={ACTION_KEY: [index / fps for index in range(chunk_size)]},
            video_backend=video_backend,
        )
        self.image_size = image_size
        self.stats = stats.tensors()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        state = item[STATE_KEY].to(torch.float32)
        action = item[ACTION_KEY].to(torch.float32)
        images = torch.stack(
            [
                _chw_rgb(item[HEAD_IMAGE_KEY], self.image_size),
                _chw_rgb(item[LEFT_WRIST_IMAGE_KEY], self.image_size),
                _chw_rgb(item[RIGHT_WRIST_IMAGE_KEY], self.image_size),
            ]
        )
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        action = (action - self.stats["action_mean"]) / self.stats["action_std"]
        result = {
            "images": images,
            "state": state,
            "action": action,
            "action_is_pad": item[f"{ACTION_KEY}_is_pad"].to(torch.bool),
        }
        if self.use_point_cloud:
            result["points"] = item[HEAD_POINT_CLOUD_KEY].to(torch.float32)
        return result


def preprocess_observation(
    head_rgb: np.ndarray,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    points: np.ndarray | None = None,
    state: np.ndarray | None = None,
    *,
    stats: NormalizationStats,
    image_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if state is None:
        raise ValueError("state is required")
    tensors = stats.tensors(device)
    images = torch.stack(
        [
            _chw_rgb(torch.from_numpy(head_rgb), image_size),
            _chw_rgb(torch.from_numpy(left_rgb), image_size),
            _chw_rgb(torch.from_numpy(right_rgb), image_size),
        ]
    ).unsqueeze(0).to(device)
    point_tensor = (torch.from_numpy(points).to(device=device, dtype=torch.float32).unsqueeze(0)
                    if points is not None else None)
    state_tensor = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
    state_tensor = (state_tensor - tensors["state_mean"]) / tensors["state_std"]
    return images, point_tensor, state_tensor


def denormalize_actions(actions: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    tensors = stats.tensors(actions.device)
    return actions * tensors["action_std"] + tensors["action_mean"]

