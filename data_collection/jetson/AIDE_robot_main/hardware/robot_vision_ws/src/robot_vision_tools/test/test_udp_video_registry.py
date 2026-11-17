"""Keep VR camera selection consistent with the active driver registry."""

from pathlib import Path

import pytest
import yaml

from robot_vision_tools.udp_video_sender import load_video_cameras


CONFIG_ROOT = Path(__file__).parents[2] / "robot_vision_bringup" / "config"


def write_registry(tmp_path, edit):
    config = yaml.safe_load((CONFIG_ROOT / "head_camera.yaml").read_text())
    edit(config)
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_current_head_and_wrist_use_the_driver_namespace():
    cameras = load_video_cameras(CONFIG_ROOT / "head_camera.yaml")
    assert cameras == {
        "head": ("/robot_02/sensors/cameras/head/color/image_raw", 6000),
        "right_wrist": (
            "/robot_02/sensors/cameras/right_wrist/color/image_raw", 6002
        ),
    }


def test_three_camera_registry_preserves_port_assignments():
    cameras = load_video_cameras(CONFIG_ROOT / "cameras.yaml")
    assert {name: port for name, (_, port) in cameras.items()} == {
        "head": 6000, "left_wrist": 6001, "right_wrist": 6002,
    }
    assert all(topic.startswith("/robot/sensors/cameras/")
               for topic, _ in cameras.values())


def test_disabled_camera_is_not_required_by_the_sender(tmp_path):
    path = write_registry(
        tmp_path,
        lambda config: config["cameras"]["head"].update(enabled=False),
    )
    assert set(load_video_cameras(path)) == {"right_wrist"}


def test_absolute_color_topic_is_preserved(tmp_path):
    def set_topic(config):
        topics = config["cameras"]["right_wrist"]["interface"]["topics"]
        topics["color_image"] = "/shared/right_color"

    path = write_registry(tmp_path, set_topic)
    assert load_video_cameras(path)["right_wrist"] == (
        "/shared/right_color", 6002
    )


def test_registry_without_color_streams_fails_before_starting(tmp_path):
    def disable_color(config):
        for camera in config["cameras"].values():
            camera["streams"].update(enable_color=False, align_depth=False)

    path = write_registry(tmp_path, disable_color)
    with pytest.raises(ValueError, match="no enabled color streams"):
        load_video_cameras(path)
