"""Loading and validation helpers for the shared camera registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_camera_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate a camera registry from disk."""
    registry_path = Path(path).expanduser()
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Camera configuration not found: {registry_path}"
        )

    with registry_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError("The YAML top level must be a mapping")

    validate_camera_registry(config)
    return config


def _require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"'{field_name}' must be a mapping")
    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def validate_camera_registry(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate registry structure and cross-camera invariants."""
    required_top_level = {
        "schema_version",
        "api_version",
        "camera_namespace",
        "cameras",
    }
    missing = required_top_level - set(config)
    if missing:
        raise ValueError(f"Missing top-level fields: {sorted(missing)}")

    _require_non_empty_string(config["schema_version"], "schema_version")
    _require_non_empty_string(config["api_version"], "api_version")
    namespace = _require_non_empty_string(
        config["camera_namespace"], "camera_namespace"
    ).strip("/")
    if not namespace or "//" in namespace:
        raise ValueError("'camera_namespace' is not a valid ROS namespace")

    cameras = _require_mapping(config["cameras"], "cameras")
    if not cameras:
        raise ValueError("'cameras' must not be empty")

    required_fields = {
        "enabled",
        "camera_name",
        "semantic_location",
        "model",
        "serial_no",
        "streams",
        "interface",
    }
    enabled_names: dict[str, str] = {}
    enabled_serials: dict[str, str] = {}

    for camera_id, camera_value in cameras.items():
        camera_id = _require_non_empty_string(camera_id, "camera_id")
        camera = _require_mapping(camera_value, f"cameras.{camera_id}")
        missing = required_fields - set(camera)
        if missing:
            raise ValueError(
                f"Camera '{camera_id}' is missing: {sorted(missing)}"
            )

        if not isinstance(camera["enabled"], bool):
            raise TypeError(f"Camera '{camera_id}' enabled must be boolean")

        camera_name = _require_non_empty_string(
            camera["camera_name"], f"cameras.{camera_id}.camera_name"
        ).strip("/")
        if not camera_name or "/" in camera_name:
            raise ValueError(
                f"Camera '{camera_id}' camera_name must be one ROS name token"
            )
        _require_non_empty_string(
            camera["semantic_location"],
            f"cameras.{camera_id}.semantic_location",
        )
        _require_non_empty_string(
            camera["model"], f"cameras.{camera_id}.model"
        )
        serial_no = _require_non_empty_string(
            camera["serial_no"], f"cameras.{camera_id}.serial_no"
        ).lstrip("_")

        streams = _require_mapping(
            camera["streams"], f"cameras.{camera_id}.streams"
        )
        stream_defaults = {
            "enable_color": True,
            "enable_depth": True,
            "align_depth": True,
            "pointcloud": False,
            "enable_sync": True,
        }
        for stream_name, default in stream_defaults.items():
            stream_value = streams.get(stream_name, default)
            if not isinstance(stream_value, bool):
                raise TypeError(
                    f"Camera '{camera_id}' streams.{stream_name} "
                    "must be boolean"
                )

        enable_color = streams.get("enable_color", True)
        enable_depth = streams.get("enable_depth", True)
        align_depth = streams.get("align_depth", True)
        if align_depth and not (enable_color and enable_depth):
            raise ValueError(
                f"Camera '{camera_id}' align_depth requires both color "
                "and depth streams"
            )

        interface = _require_mapping(
            camera["interface"], f"cameras.{camera_id}.interface"
        )
        topics = _require_mapping(
            interface.get("topics"),
            f"cameras.{camera_id}.interface.topics",
        )
        required_topics: set[str] = set()
        if enable_color:
            required_topics.update(("color_image", "color_info"))
        if enable_depth:
            required_topics.update(("depth_image", "depth_info"))
        if align_depth:
            required_topics.update(("aligned_depth", "aligned_depth_info"))
        for topic_name in required_topics:
            _require_non_empty_string(
                topics.get(topic_name),
                f"cameras.{camera_id}.interface.topics.{topic_name}",
            )

        if not camera["enabled"]:
            continue
        if not (enable_color or enable_depth):
            raise ValueError(
                f"Enabled camera '{camera_id}' has no image streams enabled"
            )
        if serial_no.upper().startswith("REPLACE_"):
            raise ValueError(
                f"Enabled camera '{camera_id}' still uses a placeholder serial"
            )

        if camera_name in enabled_names:
            raise ValueError(
                f"Enabled cameras '{enabled_names[camera_name]}' and "
                f"'{camera_id}' share camera_name '{camera_name}'"
            )
        if serial_no in enabled_serials:
            raise ValueError(
                f"Enabled cameras '{enabled_serials[serial_no]}' and "
                f"'{camera_id}' share serial_no '{serial_no}'"
            )
        enabled_names[camera_name] = camera_id
        enabled_serials[serial_no] = camera_id

        frames = _require_mapping(
            camera.get("frames"), f"cameras.{camera_id}.frames"
        )
        if enable_color:
            _require_non_empty_string(
                frames.get("color_optical"),
                f"cameras.{camera_id}.frames.color_optical",
            )
        if enable_depth:
            _require_non_empty_string(
                frames.get("depth_optical"),
                f"cameras.{camera_id}.frames.depth_optical",
            )

        if enable_depth:
            depth = _require_mapping(
                camera.get("depth"), f"cameras.{camera_id}.depth"
            )
            scale = depth.get("scale_m_per_unit")
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or scale <= 0
            ):
                raise ValueError(
                    f"Camera '{camera_id}' depth.scale_m_per_unit "
                    "must be a positive number"
                )

    return cameras
