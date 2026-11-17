from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .model import load_roi_act_checkpoint


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    model: Any
    metadata: dict[str, Any]
    runtime: dict[str, Any]
    qpos_mean: Tensor
    qpos_std: Tensor
    action_mean: Tensor
    action_std: Tensor


@dataclass(frozen=True)
class PreparedObservation:
    qpos: Tensor
    images: Tensor
    boxes: Tensor | None
    letterboxed_rgb: dict[str, np.ndarray]
    box_values: dict[str, np.ndarray]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_model_bundle(
    model_dir: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> ModelBundle:
    root = Path(model_dir).expanduser().resolve()
    metadata = _read_json(root / "metadata.json")
    runtime = _read_json(root / "runtime_config.json")
    runtime_type = runtime.get("runtime_type")
    if runtime_type not in {"roi_act_v1", "full_frame_act_v1"}:
        raise ValueError(
            "runtime_config.json runtime_type must be 'roi_act_v1' or "
            "'full_frame_act_v1'"
        )
    if metadata.get("format_version") != 1:
        raise ValueError(f"unsupported metadata format: {metadata.get('format_version')}")
    camera_names = metadata.get("camera_names")
    if camera_names != runtime.get("camera_names"):
        raise ValueError("metadata and runtime camera_names do not match")

    if runtime_type == "full_frame_act_v1":
        source_name = str(runtime.get("model_source") or "roi_act.py")
        source_path = (root / source_name).resolve()
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise FileNotFoundError(source_path)
        module_name = f"_xcore_full_frame_act_{abs(hash(str(source_path)))}"
        specification = importlib.util.spec_from_file_location(module_name, source_path)
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load full-frame ACT source: {source_path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("full-frame checkpoint root must be a mapping")
        checkpoint_config_value = checkpoint.get("model_config")
        state_dict = checkpoint.get("model")
        if not isinstance(checkpoint_config_value, dict) or not isinstance(state_dict, dict):
            raise ValueError("checkpoint must contain model and model_config mappings")
        config_class = getattr(module, "RoiActConfig", None)
        model_class = getattr(module, "RoiActionChunkTransformer", None)
        if config_class is None or model_class is None:
            raise ValueError(
                f"{source_path} must define RoiActConfig and RoiActionChunkTransformer"
            )
        config = config_class.from_mapping(checkpoint_config_value)
        if getattr(config, "vision_mode", None) != "full_frame":
            raise ValueError("full-frame runtime requires model_config vision_mode='full_frame'")
        model = model_class(config)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
    else:
        model, checkpoint = load_roi_act_checkpoint(root / "best.pt", device=device)
    checkpoint_config = checkpoint["model_config"]
    expected = {
        "num_cameras": len(camera_names),
        "state_dim": int(metadata.get("state_dim", -1)),
        "action_dim": int(metadata.get("action_dim", -1)),
    }
    for key, value in expected.items():
        if int(checkpoint_config.get(key, -1)) != value:
            raise ValueError(
                f"checkpoint {key}={checkpoint_config.get(key)!r} does not match {value}"
            )
    if int(runtime.get("chunk_size", -1)) != model.config.chunk_size:
        raise ValueError("runtime chunk_size does not match checkpoint model_config")

    try:
        statistics = np.load(root / "normalization_stats.npz")
        arrays = {
            key: np.asarray(statistics[key], dtype=np.float32)
            for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")
        }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid normalization stats in {root}") from exc
    for key, array in arrays.items():
        if array.shape != (expected["state_dim"],):
            raise ValueError(f"{key} must have shape {(expected['state_dim'],)}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key} contains NaN or infinity")
    if np.any(arrays["qpos_std"] <= 0) or np.any(arrays["action_std"] <= 0):
        raise ValueError("normalization standard deviations must be positive")

    target = torch.device(device)
    return ModelBundle(
        root=root,
        model=model,
        metadata=metadata,
        runtime=runtime,
        qpos_mean=torch.from_numpy(arrays["qpos_mean"]).to(target),
        qpos_std=torch.from_numpy(arrays["qpos_std"]).to(target),
        action_mean=torch.from_numpy(arrays["action_mean"]).to(target),
        action_std=torch.from_numpy(arrays["action_std"]).to(target),
    )


def letterbox_rgb(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got {image.shape}")
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


class GreenHSVDetector:
    """Dependency-free ROI provider for the green-cube task."""

    def __init__(
        self,
        *,
        lower_hsv: tuple[int, int, int] = (35, 70, 35),
        upper_hsv: tuple[int, int, int] = (95, 255, 255),
        minimum_area_ratio: float = 0.00025,
    ) -> None:
        self.lower = np.asarray(lower_hsv, dtype=np.uint8)
        self.upper = np.asarray(upper_hsv, dtype=np.uint8)
        self.minimum_area_ratio = minimum_area_ratio

    def detect(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = rgb.shape[:2]
        minimum_area = height * width * self.minimum_area_ratio
        eligible = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
        if not eligible:
            return np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        contour = max(eligible, key=cv2.contourArea)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        return np.asarray(
            [
                x / width,
                y / height,
                min(width, x + box_width) / width,
                min(height, y + box_height) / height,
                1.0,
            ],
            dtype=np.float32,
        )


def _full_frame_box(_image: np.ndarray) -> np.ndarray:
    return np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)


def prepare_observation(
    bundle: ModelBundle,
    observation: dict[str, Any],
    *,
    roi_provider: GreenHSVDetector | None,
    device: torch.device | str,
) -> PreparedObservation:
    camera_names = list(bundle.metadata["camera_names"])
    image_shape = bundle.metadata.get("image_shape")
    if not isinstance(image_shape, list) or len(image_shape) != 3:
        raise ValueError("metadata image_shape must be [height,width,channels]")
    target_height, target_width, channels = (int(value) for value in image_shape)
    if channels != 3:
        raise ValueError("only RGB model images are supported")

    joint_names = [f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"]
    try:
        state_values = np.asarray(
            [float(observation[name]) for name in joint_names], dtype=np.float32
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("observation is missing a valid 8-D xCore state") from exc
    if not np.all(np.isfinite(state_values)):
        raise ValueError("observation state contains NaN or infinity")

    frames: dict[str, np.ndarray] = {}
    box_values: dict[str, np.ndarray] = {}
    tensors: list[Tensor] = []
    for camera in camera_names:
        frame = observation.get(camera)
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"observation camera {camera!r} is missing")
        letterboxed = letterbox_rgb(frame, target_height, target_width)
        frames[camera] = letterboxed
        if bundle.runtime["runtime_type"] == "roi_act_v1":
            box_values[camera] = (
                roi_provider.detect(letterboxed)
                if roi_provider
                else _full_frame_box(letterboxed)
            )
        tensors.append(torch.from_numpy(letterboxed).permute(2, 0, 1))

    target = torch.device(device)
    images = torch.stack(tensors).unsqueeze(0).to(target, dtype=torch.float32) / 255.0
    images = F.interpolate(
        images.flatten(0, 1),
        size=(bundle.model.config.image_size, bundle.model.config.image_size),
        mode="bilinear",
        align_corners=False,
    ).unflatten(0, (1, len(camera_names)))
    boxes = None
    if bundle.runtime["runtime_type"] == "roi_act_v1":
        boxes = torch.from_numpy(np.stack([box_values[name] for name in camera_names]))
        boxes = boxes.unsqueeze(0).to(target)
    qpos = torch.from_numpy(state_values).unsqueeze(0).to(target)
    qpos = (qpos - bundle.qpos_mean) / bundle.qpos_std
    return PreparedObservation(
        qpos=qpos,
        images=images,
        boxes=boxes,
        letterboxed_rgb=frames,
        box_values=box_values,
    )


@torch.inference_mode()
def predict_denormalized_chunk(
    bundle: ModelBundle,
    prepared: PreparedObservation,
    *,
    use_amp: bool,
) -> Tensor:
    device = prepared.qpos.device
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp and device.type == "cuda",
    ):
        if bundle.runtime["runtime_type"] == "full_frame_act_v1":
            outputs = bundle.model(images=prepared.images, qpos=prepared.qpos)
            normalized = outputs.get("actions")
            if not isinstance(normalized, Tensor):
                raise RuntimeError("full-frame model output is missing actions")
        else:
            if prepared.boxes is None:
                raise RuntimeError("ROI model observation is missing boxes")
            normalized = bundle.model.predict_action_chunk(
                prepared.qpos, prepared.images, prepared.boxes
            )
    actions = normalized.float() * bundle.action_std + bundle.action_mean
    if actions.shape != (
        prepared.qpos.shape[0],
        bundle.model.config.chunk_size,
        bundle.model.config.action_dim,
    ):
        raise RuntimeError(f"model returned an unexpected action shape: {actions.shape}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("model returned NaN or infinite actions")
    return actions
