from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pointcloud import depth_rgb_to_xyzrgb
from .schema import CameraConfig


@dataclass
class CameraFrame:
    head_rgb: np.ndarray
    left_wrist_rgb: np.ndarray
    right_wrist_rgb: np.ndarray
    head_points: np.ndarray | None
    timestamp_s: float


def _stamp_seconds(message: Any, fallback: float) -> float:
    stamp = message.header.stamp
    seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return seconds if seconds > 0.0 else fallback


def _decode_color(message: Any, *, width: int, height: int, topic: str) -> np.ndarray:
    encoding = str(message.encoding).lower()
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported color encoding {message.encoding!r} on {topic}")
    if (int(message.width), int(message.height)) != (width, height):
        raise ValueError(
            f"unexpected image size on {topic}: {message.width}x{message.height}, "
            f"expected {width}x{height}"
        )
    row_bytes = width * channels
    step = int(message.step)
    if step < row_bytes:
        raise ValueError(f"invalid image step on {topic}: {step} < {row_bytes}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError(f"short image payload on {topic}: {raw.size} < {expected}")
    image = raw[:expected].reshape(height, step)[:, :row_bytes]
    image = image.reshape(height, width, channels)
    if channels == 4:
        image = image[:, :, :3]
    if encoding in {"rgb8", "rgba8"}:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def _decode_depth(
    message: Any, *, width: int, height: int, topic: str
) -> tuple[np.ndarray, float]:
    encoding = str(message.encoding).lower()
    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
        depth_scale = 0.001  # ROS 16UC1 depth convention: millimetres.
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
        depth_scale = 1.0  # ROS 32FC1 depth convention: metres.
    else:
        raise ValueError(
            f"unsupported depth encoding {message.encoding!r} on {topic}; "
            "expected 16UC1 or 32FC1"
        )
    if (int(message.width), int(message.height)) != (width, height):
        raise ValueError(
            f"unexpected depth size on {topic}: {message.width}x{message.height}, "
            f"expected {width}x{height}"
        )
    row_bytes = width * dtype.itemsize
    step = int(message.step)
    if step < row_bytes or step % dtype.itemsize:
        raise ValueError(f"invalid depth step on {topic}: {step}")
    raw = np.frombuffer(message.data, dtype=dtype)
    row_values = step // dtype.itemsize
    expected = height * row_values
    if raw.size < expected:
        raise ValueError(f"short depth payload on {topic}: {raw.size} < {expected}")
    depth = raw[:expected].reshape(height, row_values)[:, :width]
    return np.ascontiguousarray(depth.astype(dtype.newbyteorder("="), copy=False)), depth_scale


class RosRgbdRig:
    """Subscribe to the shared ROS camera service; never own RealSense hardware."""

    _FRAME_KEYS = ("head", "left_wrist", "right_wrist", "depth")

    def __init__(self, config: CameraConfig):
        self.config = config
        self._frame_keys = self._FRAME_KEYS if config.use_point_cloud else self._FRAME_KEYS[:-1]
        self._condition = threading.Condition()
        self._frames: dict[str, np.ndarray] = {}
        self._receive_times: dict[str, float] = {}
        self._source_times: dict[str, float] = {}
        self._versions = {key: 0 for key in self._FRAME_KEYS}
        self._last_read_versions = {key: 0 for key in self._FRAME_KEYS}
        self._depth_scale = 0.0
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._last_error: BaseException | None = None
        self._context = None
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._subscriptions: list[Any] = []
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _store_frame(
        self, key: str, frame: np.ndarray, source_time: float, receive_time: float
    ) -> None:
        with self._condition:
            self._frames[key] = frame
            self._source_times[key] = source_time
            self._receive_times[key] = receive_time
            self._versions[key] += 1
            self._last_error = None
            self._condition.notify_all()

    def _capture_error(self, error: BaseException) -> None:
        with self._condition:
            self._last_error = error
            self._condition.notify_all()

    def _on_color(self, key: str, width: int, height: int, topic: str, message: Any) -> None:
        now = time.monotonic()
        try:
            frame = _decode_color(message, width=width, height=height, topic=topic)
        except BaseException as error:
            self._capture_error(error)
            return
        self._store_frame(key, frame, _stamp_seconds(message, now), now)

    def _on_depth(self, message: Any) -> None:
        now = time.monotonic()
        try:
            depth, scale = _decode_depth(
                message,
                width=self.config.head_width,
                height=self.config.head_height,
                topic=self.config.head_depth_topic,
            )
        except BaseException as error:
            self._capture_error(error)
            return
        with self._condition:
            self._depth_scale = scale
        self._store_frame("depth", depth, _stamp_seconds(message, now), now)

    def _on_camera_info(self, message: Any) -> None:
        try:
            if (int(message.width), int(message.height)) != (
                self.config.head_width,
                self.config.head_height,
            ):
                raise ValueError(
                    f"unexpected camera info size on {self.config.head_depth_info_topic}: "
                    f"{message.width}x{message.height}"
                )
            values = tuple(float(value) for value in message.k)
            if len(values) != 9:
                raise ValueError("head aligned-depth camera matrix must contain 9 values")
            intrinsics = (values[0], values[4], values[2], values[5])
            if intrinsics[0] <= 0 or intrinsics[1] <= 0:
                raise ValueError("invalid head aligned-depth camera matrix")
        except BaseException as error:
            self._capture_error(error)
            return
        with self._condition:
            self._intrinsics = intrinsics
            self._last_error = None
            self._condition.notify_all()

    def connect(self, warmup_frames: int = 1) -> None:
        del warmup_frames  # ROS publisher owns hardware warmup.
        if self.is_connected:
            raise RuntimeError("ROS RGB-D rig is already connected")
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.signals import SignalHandlerOptions
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 Python packages are required; source environment/activate.sh"
            ) from error

        self._context = Context()
        rclpy.init(
            args=None,
            context=self._context,
            domain_id=0,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self._node = rclpy.create_node(
            f"xcore_interact_cameras_{id(self) & 0xffff:x}", context=self._context
        )

        def subscribe_color(key: str, topic: str, width: int, height: int) -> None:
            callback = lambda message: self._on_color(key, width, height, topic, message)
            self._subscriptions.append(
                self._node.create_subscription(Image, topic, callback, qos_profile_sensor_data)
            )

        subscribe_color(
            "head",
            self.config.head_image_topic,
            self.config.head_width,
            self.config.head_height,
        )
        subscribe_color(
            "left_wrist",
            self.config.left_wrist_image_topic,
            self.config.wrist_width,
            self.config.wrist_height,
        )
        subscribe_color(
            "right_wrist",
            self.config.right_wrist_image_topic,
            self.config.wrist_width,
            self.config.wrist_height,
        )
        if self.config.use_point_cloud:
            self._subscriptions.append(
                self._node.create_subscription(
                    Image,
                    self.config.head_depth_topic,
                    self._on_depth,
                    qos_profile_sensor_data,
                )
            )
            self._subscriptions.append(
                self._node.create_subscription(
                    CameraInfo,
                    self.config.head_depth_info_topic,
                    self._on_camera_info,
                    qos_profile_sensor_data,
                )
            )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="interact-ros-cameras", daemon=True
        )
        self._spin_thread.start()
        try:
            self._wait_bundle(self.config.connect_timeout_s, require_new=False)
            with self._condition:
                self._last_read_versions = dict(self._versions)
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def _wait_bundle(self, timeout_s: float, *, require_new: bool) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._last_error is not None:
                    error = self._last_error
                    self._last_error = None
                    raise RuntimeError(f"invalid ROS camera message: {error}") from error
                now = time.monotonic()
                complete = (
                    all(key in self._frames for key in self._frame_keys)
                    and (not self.config.use_point_cloud or
                         (self._intrinsics is not None and self._depth_scale > 0))
                )
                fresh = complete and all(
                    now - self._receive_times[key] <= self.config.frame_timeout_s
                    for key in self._frame_keys
                )
                new = not require_new or all(
                    self._versions[key] > self._last_read_versions[key]
                    for key in self._frame_keys
                )
                head_depth_match = not self.config.use_point_cloud or (complete and abs(
                    self._source_times["head"] - self._source_times["depth"]
                ) <= 0.001)
                stream_skew = complete and (
                    max(self._source_times[key] for key in self._frame_keys)
                    - min(self._source_times[key] for key in self._frame_keys)
                    <= self.config.max_frame_skew_s
                )
                if complete and fresh and new and head_depth_match and stream_skew:
                    return
                remaining = deadline - now
                if remaining <= 0:
                    missing = [key for key in self._frame_keys if key not in self._frames]
                    raise TimeoutError(
                        "timed out waiting for ROS camera bundle; "
                        f"missing={missing}, camera_info={self._intrinsics is not None}. "
                        f"Verify robot_vision publishes {'RGB and aligned depth' if self.config.use_point_cloud else 'three RGB streams'} "
                        "with sensor-data QoS."
                    )
                self._condition.wait(remaining)

    def read(self, timeout_ms: int = 2000, *, point_seed: int | None = None) -> CameraFrame:
        if not self.is_connected:
            raise RuntimeError("ROS RGB-D rig is not connected")
        self._wait_bundle(timeout_ms / 1_000.0, require_new=True)
        with self._condition:
            head = self._frames["head"].copy()
            left = self._frames["left_wrist"].copy()
            right = self._frames["right_wrist"].copy()
            if self.config.use_point_cloud:
                depth = self._frames["depth"].copy()
                fx, fy, ppx, ppy = self._intrinsics
                depth_scale = self._depth_scale
            self._last_read_versions = dict(self._versions)
        points = None
        if self.config.use_point_cloud:
            points = depth_rgb_to_xyzrgb(
                depth,
                head[..., ::-1],  # published BGR -> point features use RGB
                fx=fx, fy=fy, ppx=ppx, ppy=ppy, depth_scale=depth_scale,
                point_count=self.config.point_count,
                min_depth_m=self.config.min_depth_m,
                max_depth_m=self.config.max_depth_m, seed=point_seed,
            )
        return CameraFrame(head, left, right, points, time.monotonic())

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
        self._subscriptions.clear()
        self._executor = None
        self._spin_thread = None
        self._node = None
        self._context = None
        with self._condition:
            self._frames.clear()
            self._receive_times.clear()
            self._source_times.clear()
            self._intrinsics = None
            self._depth_scale = 0.0
            self._last_error = None

    def __enter__(self) -> "RosRgbdRig":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
