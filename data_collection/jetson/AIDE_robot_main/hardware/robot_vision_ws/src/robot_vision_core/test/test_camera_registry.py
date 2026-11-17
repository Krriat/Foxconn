"""Tests for camera registry validation."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from robot_vision_core.camera_registry import validate_camera_registry


REGISTRY_PATH = (
    Path(__file__).parents[2]
    / "robot_vision_bringup"
    / "config"
    / "cameras.yaml"
)


def load_registry():
    """Return an independent copy of the repository registry."""
    with REGISTRY_PATH.open("r", encoding="utf-8") as registry_file:
        return yaml.safe_load(registry_file)


def test_repository_registry_is_valid():
    """The checked-in registry should satisfy the shared schema."""
    config = load_registry()
    assert validate_camera_registry(config) is config["cameras"]


def test_duplicate_enabled_camera_names_are_rejected():
    """Two running nodes must not share a ROS node/topic root."""
    config = deepcopy(load_registry())
    config["cameras"]["left_wrist"]["enabled"] = True
    config["cameras"]["left_wrist"]["camera_name"] = "head"

    with pytest.raises(ValueError, match="share camera_name"):
        validate_camera_registry(config)


def test_placeholder_serial_is_rejected_when_enabled():
    """An enabled camera must be bound to a real device serial."""
    config = deepcopy(load_registry())
    config["cameras"]["chest"]["enabled"] = True

    with pytest.raises(ValueError, match="placeholder serial"):
        validate_camera_registry(config)


def test_align_depth_requires_color_and_depth():
    """Depth-to-color alignment cannot run without both source streams."""
    config = deepcopy(load_registry())
    config["cameras"]["head"]["streams"]["enable_color"] = False

    with pytest.raises(ValueError, match="requires both"):
        validate_camera_registry(config)
