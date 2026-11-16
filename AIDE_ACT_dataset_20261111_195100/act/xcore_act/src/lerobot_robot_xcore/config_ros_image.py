from __future__ import annotations

from dataclasses import dataclass

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode


@CameraConfig.register_subclass("ros_image")
@dataclass(kw_only=True)
class RosImageCameraConfig(CameraConfig):
    """LeRobot camera backed by a ``sensor_msgs/Image`` ROS 2 topic."""

    topic: str
    color_mode: ColorMode = ColorMode.RGB
    connect_timeout_s: float = 5.0
    frame_timeout_ms: float = 1_000.0

    def __post_init__(self) -> None:
        if not self.topic.startswith("/"):
            raise ValueError("ROS image topic must be absolute (start with '/')")
        if self.width is None or self.width <= 0:
            raise ValueError("ROS image camera width must be positive")
        if self.height is None or self.height <= 0:
            raise ValueError("ROS image camera height must be positive")
        if self.fps is None or self.fps <= 0:
            raise ValueError("ROS image camera fps must be positive")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")
        if self.frame_timeout_ms <= 0:
            raise ValueError("frame_timeout_ms must be positive")
