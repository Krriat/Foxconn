#!/usr/bin/env python3

"""camera_manager：加载 cameras.yaml 并在线监测每台已启用相机的健康状态。

启动流程：
  1. 读取 config_file 指向的 cameras.yaml，校验字段。
  2. 打印相机注册表。
  3. 对每台 enabled=true 的相机，按 streams 配置订阅其 color / depth /
     aligned_depth / camera_info 话题。
  4. 以 publish_rate_hz 周期计算各相机健康状态，发布
     robot_camera_msgs/CameraStatus 到 'camera_status' 话题。
     状态变化时打 info 日志，避免每帧刷屏。

健康状态判定（详见 robot_camera_msgs/msg/CameraStatus.msg）：
  STATE_UNKNOWN  启动宽限期内且非全流就绪
  STATE_OFFLINE  从未收到任何流数据
  STATE_OK       所有启用流均 fresh（age <= stale_threshold_sec）
  STATE_STALE    部分流 stale
  STATE_ERROR    预留给深度比例不一致等（当前未触发）
"""

from __future__ import annotations

import json
from std_msgs.msg import String

from collections import deque
import math
from pathlib import Path
from typing import Any, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from robot_camera_msgs.msg import CameraStatus
from robot_camera_msgs.srv import GetCameraInfo
from sensor_msgs.msg import CameraInfo, Image

from robot_vision_core.camera_registry import (
    load_camera_registry,
    validate_camera_registry,
)


# CameraStatus.state 枚举（与 robot_camera_msgs/msg/CameraStatus.msg 对应）
STATE_UNKNOWN = 0
STATE_OK = 1
STATE_STALE = 2
STATE_OFFLINE = 3
STATE_ERROR = 4

_STATE_NAMES = {
    STATE_UNKNOWN: "UNKNOWN",
    STATE_OK: "OK",
    STATE_STALE: "STALE",
    STATE_OFFLINE: "OFFLINE",
    STATE_ERROR: "ERROR",
}


def join_ros_name(*parts: str) -> str:
    """Join multiple strings into one absolute ROS name."""
    clean_parts = [
        str(part).strip("/")
        for part in parts
        if str(part).strip("/")
    ]
    return "/" + "/".join(clean_parts)


class _CameraMonitor:
    """单台已启用相机的运行时状态采集器。"""

    def __init__(
        self,
        node: Node,
        camera_id: str,
        camera: dict[str, Any],
        camera_namespace: str,
        hz_window_sec: float,
    ) -> None:
        self._node = node
        self.camera_id = camera_id
        self.camera_name = str(camera["camera_name"])
        self.model = str(camera["model"])
        self.semantic_location = str(
            camera["semantic_location"]
        )
        scale = camera.get("depth", {}).get("scale_m_per_unit")
        self.expected_scale = (
            float(scale) if isinstance(scale, (int, float)) else 0.0
        )

        streams = camera.get("streams", {})
        topics = camera["interface"]["topics"]

        self.enable_color = bool(
            streams.get("enable_color", False)
        )
        self.enable_depth = bool(
            streams.get("enable_depth", False)
        )
        self.align_depth = bool(
            streams.get("align_depth", False)
        )

        self.last_color_time: Optional[Time] = None
        self.last_depth_time: Optional[Time] = None
        self.last_aligned_time: Optional[Time] = None
        self.last_camera_info_time: Optional[Time] = None
        self.last_camera_info_msg: Optional[CameraInfo] = None

        self._hz_window_sec = hz_window_sec
        self._color_times: deque = deque()
        self.last_state = STATE_UNKNOWN
        self._subscriptions = []

        camera_root = join_ros_name(
            camera_namespace,
            self.camera_name,
        )

        # 健康检查只需要到达时间，Image 使用 raw 订阅避免 Python
        # 反序列化整幅图像；CameraInfo 仍保留结构化消息供服务返回。
        if self.enable_color:
            color_topic = join_ros_name(
                camera_root,
                topics["color_image"],
            )
            self._subscriptions.append(node.create_subscription(
                Image,
                color_topic,
                self._on_color,
                qos_profile_sensor_data,
                raw=True,
            ))
        if self.enable_depth:
            depth_topic = join_ros_name(
                camera_root,
                topics["depth_image"],
            )
            self._subscriptions.append(node.create_subscription(
                Image,
                depth_topic,
                self._on_depth,
                qos_profile_sensor_data,
                raw=True,
            ))
        if self.align_depth:
            aligned_topic = join_ros_name(
                camera_root,
                topics["aligned_depth"],
            )
            self._subscriptions.append(node.create_subscription(
                Image,
                aligned_topic,
                self._on_aligned,
                qos_profile_sensor_data,
                raw=True,
            ))

        # 优先缓存与 aligned depth 同坐标系的内参；未启用对齐时退回
        # color/depth CameraInfo，避免服务永远报告内参不可用。
        if self.align_depth:
            info_key = "aligned_depth_info"
        elif self.enable_color:
            info_key = "color_info"
        elif self.enable_depth:
            info_key = "depth_info"
        else:
            info_key = ""
        if info_key:
            info_topic = join_ros_name(camera_root, topics[info_key])
            self._subscriptions.append(node.create_subscription(
                CameraInfo,
                info_topic,
                self._on_camera_info,
                qos_profile_sensor_data,
            ))

    # ----------------------------------------------------------------
    # 回调：只记录最新时间戳，Hz 统计在 _on_color 中维护滑动窗口
    # ----------------------------------------------------------------
    def _on_color(self, msg: Image) -> None:
        now = self._node.get_clock().now()
        self.last_color_time = now
        self._color_times.append(now)
        self._trim_hz_window(now)

    def _on_depth(self, msg: Image) -> None:
        self.last_depth_time = self._node.get_clock().now()

    def _on_aligned(self, msg: Image) -> None:
        self.last_aligned_time = self._node.get_clock().now()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.last_camera_info_time = self._node.get_clock().now()
        self.last_camera_info_msg = msg

    def _trim_hz_window(self, now: Time) -> None:
        cutoff = now - Duration(
            seconds=self._hz_window_sec
        )
        while (
            self._color_times
            and self._color_times[0] < cutoff
        ):
            self._color_times.popleft()

    @staticmethod
    def _age_sec(
        last_time: Optional[Time],
        now: Time,
    ) -> float:
        if last_time is None:
            return -1.0
        return (now - last_time).nanoseconds / 1e9

    @staticmethod
    def _is_fresh(
        last_time: Optional[Time],
        now: Time,
        stale_sec: float,
    ) -> bool:
        if last_time is None:
            return False
        return (
            (now - last_time).nanoseconds / 1e9
            <= stale_sec
        )

    def compute_status(
        self,
        now: Time,
        stale_sec: float,
        offline_sec: float,
        grace_sec: float,
        node_start: Time,
    ) -> CameraStatus:
        """计算本相机当前健康状态并填入 CameraStatus 消息。"""
        self._trim_hz_window(now)
        color_ok = self._is_fresh(
            self.last_color_time, now, stale_sec
        )
        depth_ok = self._is_fresh(
            self.last_depth_time, now, stale_sec
        )
        aligned_ok = self._is_fresh(
            self.last_aligned_time, now, stale_sec
        )

        # 按启用情况收集流，便于判断 stale 来源与是否收到过数据
        enabled_streams: list[
            tuple[str, bool, Optional[Time]]
        ] = []
        if self.enable_color:
            enabled_streams.append(
                ("color", color_ok, self.last_color_time)
            )
        if self.enable_depth:
            enabled_streams.append(
                ("depth", depth_ok, self.last_depth_time)
            )
        if self.align_depth:
            enabled_streams.append(
                (
                    "aligned_depth",
                    aligned_ok,
                    self.last_aligned_time,
                )
            )

        any_received = any(
            t is not None for _, _, t in enabled_streams
        )
        all_fresh = (
            all(ok for _, ok, _ in enabled_streams)
            if enabled_streams
            else False
        )
        elapsed_sec = (
            (now - node_start).nanoseconds / 1e9
        )

        received_times = [
            t for _, _, t in enabled_streams if t is not None
        ]
        newest_time = max(received_times) if received_times else None
        newest_age = self._age_sec(newest_time, now)

        if not enabled_streams:
            state = STATE_UNKNOWN
            detail = "no streams enabled in cameras.yaml"
        elif elapsed_sec < grace_sec and not all_fresh:
            state = STATE_UNKNOWN
            detail = "startup grace, waiting for streams"
        elif not any_received:
            state = STATE_OFFLINE
            detail = (
                "no stream data received from camera node"
            )
        elif not all_fresh and newest_age > offline_sec:
            state = STATE_OFFLINE
            detail = (
                f"no stream data received for {newest_age:.1f}s"
            )
        elif all_fresh:
            state = STATE_OK
            detail = ""
        else:
            state = STATE_STALE
            stale_names = [
                name
                for name, ok, _ in enabled_streams
                if not ok
            ]
            detail = "stale streams: " + ", ".join(
                stale_names
            )

        msg = CameraStatus()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = ""
        msg.camera_id = self.camera_id
        msg.camera_name = self.camera_name
        msg.model = self.model
        msg.semantic_location = self.semantic_location
        msg.state = state
        msg.detail = detail
        msg.color_stream_ok = color_ok
        msg.depth_stream_ok = depth_ok
        msg.aligned_depth_ok = aligned_ok
        if len(self._color_times) >= 2:
            sample_span = (
                self._color_times[-1] - self._color_times[0]
            ).nanoseconds / 1e9
            msg.last_hz = (
                (len(self._color_times) - 1) / sample_span
                if sample_span > 0.0
                else 0.0
            )
        else:
            msg.last_hz = 0.0
        msg.last_seen_age_sec = self._age_sec(
            newest_time, now
        )
        msg.expected_scale_m_per_unit = self.expected_scale
        # 标准 CameraInfo 不包含 RealSense 深度比例。0 表示尚未实测，避免
        # 把配置值伪装成设备测量值。
        msg.measured_scale_m_per_unit = 0.0

        return msg


class CameraManager(Node):
    """Load and validate the camera registry, then monitor runtime health."""

    def __init__(self) -> None:
        super().__init__("camera_manager", namespace="/robot/vision")

        # ---- 参数 ----
        self.declare_parameter("config_file", "")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("stale_threshold_sec", 2.0)
        self.declare_parameter("offline_threshold_sec", 10.0)
        self.declare_parameter("startup_grace_sec", 5.0)
        self.declare_parameter("hz_window_sec", 5.0)

        config_file = str(
            self.get_parameter("config_file").value
        ).strip()

        if not config_file:
            raise RuntimeError(
                "Parameter 'config_file' is required"
            )

        self.config_path = Path(config_file).expanduser()

        self.config = load_camera_registry(self.config_path)
        self.cameras = self.config["cameras"]

        self.print_camera_registry()

        # ---- 监测参数 ----
        self._publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        self._stale_sec = float(
            self.get_parameter("stale_threshold_sec").value
        )
        self._offline_sec = float(
            self.get_parameter("offline_threshold_sec").value
        )
        self._grace_sec = float(
            self.get_parameter("startup_grace_sec").value
        )
        self._hz_window_sec = float(
            self.get_parameter("hz_window_sec").value
        )
        self._validate_monitor_parameters()

        camera_namespace = str(
            self.config["camera_namespace"]
        )

        # ---- 每台已启用相机建一个监测器 ----
        self._monitors: dict[str, _CameraMonitor] = {}
        for camera_id, camera in self.cameras.items():
            if not camera["enabled"]:
                continue
            self._monitors[camera_id] = _CameraMonitor(
                self,
                camera_id,
                camera,
                camera_namespace,
                self._hz_window_sec,
            )
            self.get_logger().info(
                f"Monitoring camera '{camera_id}': "
                f"{camera['model']} "
                f"serial={camera['serial_no']}"
            )

        # ---- 发布 CameraStatus ----
        self._status_pub = self.create_publisher(
            CameraStatus,
            "camera_status",
            10,
        )
        # Preserve the platform's aggregate JSON health contract under the
        # canonical vision namespace; CameraStatus remains the native API.
        self._json_status_pub = self.create_publisher(String, "camera_status_json", 10)
        self._start_time = self.get_clock().now()
        self._timer = self.create_timer(
            1.0 / self._publish_rate_hz,
            self._publish_statuses,
        )

        self.get_logger().info(
            f"Camera status publisher started at "
            f"{self._publish_rate_hz} Hz on "
            f"'/robot/vision/camera_status'"
        )

        # ---- GetCameraInfo 服务端 ----
        self._get_info_service = self.create_service(
            GetCameraInfo,
            "get_camera_info",
            self._handle_get_camera_info,
        )
        self.get_logger().info(
            "GetCameraInfo service ready at "
            "'/robot/vision/get_camera_info'"
        )

    def _publish_statuses(self) -> None:
        now = self.get_clock().now()
        aggregate = {}
        for camera_id, mon in self._monitors.items():
            msg = mon.compute_status(
                now,
                self._stale_sec,
                self._offline_sec,
                self._grace_sec,
                self._start_time,
            )
            self._status_pub.publish(msg)
            cfg = self.cameras[camera_id]
            aggregate[camera_id] = {
                "state": {STATE_OK: "streaming", STATE_UNKNOWN: "starting",
                          STATE_STALE: "reconnecting", STATE_OFFLINE: "offline",
                          STATE_ERROR: "error"}.get(msg.state, "unknown"),
                "serial": cfg["serial_no"], "model": cfg["model"],
                "last_frame_age_sec": max(0.0, msg.last_seen_age_sec),
                "actual_fps": msg.last_hz,
                "depth_enabled": bool(cfg["streams"]["enable_depth"]),
                "depth_scale_m": msg.expected_scale_m_per_unit,
                "error": "" if msg.state == STATE_OK else msg.detail,
            }

            # 状态变化时打日志，避免每帧刷屏
            if msg.state != mon.last_state:
                self.get_logger().info(
                    f"[{camera_id}] state: "
                    f"{_STATE_NAMES.get(msg.state, '?')} "
                    f"({msg.detail})"
                )
                mon.last_state = msg.state
        self._json_status_pub.publish(String(data=json.dumps({
            "node": self.get_fully_qualified_name(), "cameras": aggregate,
        }, ensure_ascii=False)))

    def _handle_get_camera_info(
        self,
        request: GetCameraInfo.Request,
        response: GetCameraInfo.Response,
    ) -> GetCameraInfo.Response:
        """按需返回某台相机的完整资料：身份/内参/外参/深度比例/状态/话题/TF。"""
        camera_id = request.camera_id
        response.camera_id = camera_id

        # 运行时默认值（未就绪时返回的占位）
        response.k = [0.0] * 9
        response.t_cam2base = [0.0] * 16
        response.d = []
        response.last_hz = 0.0
        response.last_seen_age_sec = -1.0
        response.state = STATE_OFFLINE
        response.intrinsics_available = False
        response.extrinsics_available = False

        if camera_id not in self.cameras:
            response.exists = False
            response.detail = (
                f"camera_id '{camera_id}' not in cameras.yaml"
            )
            return response

        cfg = self.cameras[camera_id]
        response.camera_name = str(cfg["camera_name"])
        response.model = str(cfg["model"])
        response.semantic_location = str(
            cfg["semantic_location"]
        )

        scale = (
            cfg.get("depth", {})
            .get("scale_m_per_unit")
        )
        response.depth_scale_m_per_unit = (
            float(scale)
            if isinstance(scale, (int, float))
            else 0.0
        )

        camera_root = join_ros_name(
            self.config["camera_namespace"],
            cfg["camera_name"],
        )
        topics = cfg["interface"]["topics"]
        response.color_topic = join_ros_name(
            camera_root, topics.get("color_image", "")
        )
        response.depth_topic = join_ros_name(
            camera_root, topics.get("depth_image", "")
        )
        response.aligned_depth_topic = join_ros_name(
            camera_root, topics.get("aligned_depth", "")
        )
        response.aligned_depth_info_topic = join_ros_name(
            camera_root,
            topics.get("aligned_depth_info", ""),
        )

        frames = cfg.get("frames", {})
        response.color_optical_frame = str(
            frames.get("color_optical", "")
        )
        response.depth_optical_frame = str(
            frames.get("depth_optical", "")
        )
        planned = (
            cfg.get("interface", {})
            .get("planned_target_frames", [])
        )
        response.base_frame = (
            str(planned[0]) if planned else ""
        )

        if not cfg["enabled"]:
            response.exists = False
            response.detail = (
                "camera configured but enabled=false "
                "in cameras.yaml"
            )
            return response

        # 已启用：填运行时内参与状态
        response.exists = True
        mon = self._monitors[camera_id]
        now = self.get_clock().now()

        if mon.last_camera_info_msg is not None:
            ci = mon.last_camera_info_msg
            response.k = list(ci.k)
            response.d = list(ci.d)
            response.width = ci.width
            response.height = ci.height
            response.intrinsics_available = True

        status = mon.compute_status(
            now,
            self._stale_sec,
            self._offline_sec,
            self._grace_sec,
            self._start_time,
        )
        response.state = status.state
        response.detail = status.detail
        response.last_hz = status.last_hz
        response.last_seen_age_sec = (
            status.last_seen_age_sec
        )

        return response

    @staticmethod
    def load_config(path: Path) -> dict[str, Any]:
        """Read a YAML configuration file."""
        return load_camera_registry(path)

    @staticmethod
    def validate_config(
        config: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Validate the minimum required camera fields."""
        return validate_camera_registry(config)

    def _validate_monitor_parameters(self) -> None:
        """Reject invalid timing parameters before creating the timer."""
        values = {
            "publish_rate_hz": self._publish_rate_hz,
            "stale_threshold_sec": self._stale_sec,
            "offline_threshold_sec": self._offline_sec,
            "startup_grace_sec": self._grace_sec,
            "hz_window_sec": self._hz_window_sec,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"Parameter '{name}' must be finite")
        for name in (
            "publish_rate_hz",
            "stale_threshold_sec",
            "offline_threshold_sec",
            "hz_window_sec",
        ):
            if values[name] <= 0.0:
                raise ValueError(f"Parameter '{name}' must be greater than zero")
        if self._grace_sec < 0.0:
            raise ValueError(
                "Parameter 'startup_grace_sec' must not be negative"
            )
        if self._offline_sec <= self._stale_sec:
            raise ValueError(
                "Parameter 'offline_threshold_sec' must be greater than "
                "'stale_threshold_sec'"
            )

    def print_camera_registry(self) -> None:
        """Print all configured and enabled cameras."""
        camera_namespace = str(
            self.config["camera_namespace"]
        )

        enabled_count = sum(
            bool(camera["enabled"])
            for camera in self.cameras.values()
        )

        self.get_logger().info(
            "\n"
            "Camera registry loaded successfully\n"
            f"  config_file:    {self.config_path}\n"
            f"  schema_version: "
            f"{self.config['schema_version']}\n"
            f"  api_version:    "
            f"{self.config['api_version']}\n"
            f"  configured:     "
            f"{len(self.cameras)}\n"
            f"  enabled:        {enabled_count}"
        )

        for camera_id, camera in self.cameras.items():
            if not camera["enabled"]:
                continue

            camera_root = join_ros_name(
                camera_namespace,
                camera["camera_name"],
            )

            topics = camera["interface"]["topics"]
            frames = camera["frames"]
            depth = camera["depth"]

            color_topic = join_ros_name(
                camera_root,
                topics["color_image"],
            )

            aligned_depth_topic = join_ros_name(
                camera_root,
                topics["aligned_depth"],
            )

            self.get_logger().info(
                "\n"
                f"Camera ID:       {camera_id}\n"
                f"  model:         {camera['model']}\n"
                f"  serial:        {camera['serial_no']}\n"
                f"  location:      "
                f"{camera['semantic_location']}\n"
                f"  usage:         "
                f"{camera['interface']['primary_usage']}\n"
                f"  camera_root:   {camera_root}\n"
                f"  color_topic:   {color_topic}\n"
                f"  aligned_depth: "
                f"{aligned_depth_topic}\n"
                f"  optical_frame: "
                f"{frames['color_optical']}\n"
                f"  depth_scale:   "
                f"{depth['scale_m_per_unit']} m/raw_unit"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = CameraManager()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"camera_manager failed: {error}")

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
