from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode

from .config_ros_image import RosImageCameraConfig


class RosImageCamera(Camera):
    """Subscribe to one ROS image topic without owning the camera hardware.

    Every instance uses an independent rclpy context.  This is deliberate: the
    xCore robot also owns a global ROS context for its gripper, while the PICO
    recorder owns another node for controller samples.  Disconnecting a camera
    therefore cannot shut down either of those consumers.
    """

    def __init__(self, config: RosImageCameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self._condition = threading.Condition()
        self._frame: NDArray[np.uint8] | None = None
        self._frame_time = 0.0
        self._last_error: BaseException | None = None
        self._context = None
        self._node = None
        self._subscription = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # ROS topics are runtime graph resources, not locally attached devices.
        return []

    def _decode(self, message) -> NDArray[np.uint8]:
        encoding = str(message.encoding).lower()
        channel_count = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
        }.get(encoding)
        if channel_count is None:
            raise ValueError(
                f"unsupported encoding {message.encoding!r} on {self.config.topic}; "
                "expected bgr8/rgb8/bgra8/rgba8/mono8"
            )
        height = int(message.height)
        width = int(message.width)
        if (width, height) != (self.width, self.height):
            raise ValueError(
                f"unexpected image size on {self.config.topic}: got {width}x{height}, "
                f"expected {self.width}x{self.height}"
            )
        row_bytes = width * channel_count
        step = int(message.step)
        if step < row_bytes:
            raise ValueError(
                f"invalid image step on {self.config.topic}: {step} < {row_bytes}"
            )
        raw = np.frombuffer(message.data, dtype=np.uint8)
        expected_bytes = height * step
        if raw.size < expected_bytes:
            raise ValueError(
                f"short image payload on {self.config.topic}: "
                f"{raw.size} < {expected_bytes}"
            )
        image = raw[:expected_bytes].reshape(height, step)[:, :row_bytes]
        image = image.reshape(height, width, channel_count)
        if encoding in {"bgra8", "rgba8"}:
            image = image[:, :, :3]
        elif encoding == "mono8":
            image = np.repeat(image, 3, axis=2)

        source_is_rgb = encoding in {"rgb8", "rgba8"}
        output_is_rgb = self.color_mode == ColorMode.RGB
        if source_is_rgb != output_is_rgb:
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)

    def _on_image(self, message) -> None:
        try:
            frame = self._decode(message)
        except BaseException as exc:
            with self._condition:
                self._last_error = exc
                self._condition.notify_all()
            return
        with self._condition:
            self._frame = frame
            self._frame_time = time.monotonic()
            self._last_error = None
            self._condition.notify_all()

    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"ROS image camera {self.config.topic} is already connected")
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self._context = Context()
        rclpy.init(
            args=None,
            context=self._context,
            domain_id=0,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        safe_name = self.config.topic.strip("/").replace("/", "_") or "image"
        self._node = rclpy.create_node(
            f"lerobot_{safe_name}_{id(self) & 0xffff:x}",
            context=self._context,
        )
        self._subscription = self._node.create_subscription(
            Image,
            self.config.topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name=f"ros-camera-{safe_name}",
            daemon=True,
        )
        self._spin_thread.start()
        try:
            # Always require a first frame.  ``warmup`` only exists for API
            # compatibility; a topic subscriber is unusable without one.
            self._wait_for_frame(self.config.connect_timeout_s * 1_000.0)
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def _wait_for_frame(self, timeout_ms: float) -> NDArray[np.uint8]:
        deadline = time.monotonic() + timeout_ms / 1_000.0
        with self._condition:
            while True:
                if self._last_error is not None:
                    error = self._last_error
                    self._last_error = None
                    raise RuntimeError(
                        f"invalid ROS image on {self.config.topic}: {error}"
                    ) from error
                now = time.monotonic()
                if (
                    self._frame is not None
                    and now - self._frame_time <= self.config.frame_timeout_ms / 1_000.0
                ):
                    return self._frame.copy()
                remaining = deadline - now
                if remaining <= 0:
                    state = "no first frame" if self._frame is None else "stale frame"
                    raise TimeoutError(
                        f"{state} on {self.config.topic} for {timeout_ms:.0f} ms; "
                        "verify all_camera is running and the topic uses sensor-data QoS"
                    )
                self._condition.wait(remaining)

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if color_mode is not None and color_mode != self.color_mode:
            raise ValueError(
                f"per-read color conversion is unsupported; configured {self.color_mode}"
            )
        return self.async_read()

    def async_read(self, timeout_ms: float | None = None) -> NDArray[Any]:
        if not self._connected:
            raise RuntimeError(f"ROS image camera {self.config.topic} is not connected")
        return self._wait_for_frame(
            self.config.frame_timeout_ms if timeout_ms is None else timeout_ms
        )

    def disconnect(self) -> None:
        self._connected = False
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            try:
                self._context.shutdown()
            except Exception:
                pass
        self._subscription = None
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._context = None
        with self._condition:
            self._frame = None
            self._frame_time = 0.0
            self._last_error = None
