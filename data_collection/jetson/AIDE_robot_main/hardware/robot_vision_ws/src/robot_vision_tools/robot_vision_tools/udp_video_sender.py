#!/usr/bin/env python3
"""Forward shared ROS camera topics to the VR device over UDP.

The node never opens a camera device. Each ROS frame becomes one JPEG UDP
datagram, matching the legacy ACT video receiver format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import socket
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


MAX_UDP_PAYLOAD = 65_507
CAMERAS = {
    "head": ("/robot/sensors/cameras/head/color/image_raw", 6000),
    "left_wrist": ("/robot/sensors/cameras/left_wrist/color/image_raw", 6001),
    "right_wrist": ("/robot/sensors/cameras/right_wrist/color/image_raw", 6002),
}


def load_video_cameras(path: Path) -> dict[str, tuple[str, int]]:
    """Use the same enabled color topics as the camera driver registry."""
    from robot_vision_core.camera_registry import load_camera_registry

    registry = load_camera_registry(path)
    namespace = registry["camera_namespace"].strip("/")
    cameras = {}
    for camera_id, camera in registry["cameras"].items():
        if not camera["enabled"] or not camera["streams"].get("enable_color", True):
            continue
        if camera_id not in CAMERAS:
            raise ValueError(f"No VR UDP port is defined for camera '{camera_id}'")
        topic = camera["interface"]["topics"]["color_image"]
        if not topic.startswith("/"):
            topic = f"/{namespace}/{camera['camera_name']}/{topic}"
        cameras[camera_id] = (topic, CAMERAS[camera_id][1])
    if not cameras:
        raise ValueError("Camera registry has no enabled color streams")
    return cameras


def image_to_bgr(message: Image) -> np.ndarray:
    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }.get(message.encoding.lower())
    if channels is None:
        raise ValueError(f"unsupported ROS image encoding: {message.encoding}")
    if message.height <= 0 or message.width <= 0 or message.step <= 0:
        raise ValueError("invalid ROS image dimensions")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = message.height * message.step
    if raw.size < required:
        raise ValueError(f"short ROS image buffer: {raw.size} < {required}")
    rows = raw[:required].reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(
        message.height, message.width, channels
    )
    encoding = message.encoding.lower()
    if encoding == "bgr8":
        return image
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def encode_jpeg(frame_bgr: np.ndarray, initial_quality: int) -> tuple[bytes, int]:
    quality = initial_quality
    encoded_size = 0
    while quality >= 5:
        ok, buffer = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise RuntimeError("OpenCV JPEG encoding failed")
        payload = buffer.tobytes()
        encoded_size = len(payload)
        if encoded_size <= MAX_UDP_PAYLOAD:
            return payload, quality
        quality -= 5
    raise RuntimeError(
        f"JPEG is larger than one UDP datagram ({encoded_size} bytes at quality 5)"
    )


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        **payload,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class UdpVideoSenderNode(Node):
    def __init__(
        self,
        target_ip: str,
        jpeg_quality: int,
        send_every: int,
        cameras: dict[str, tuple[str, int]] | None = None,
    ) -> None:
        super().__init__("video_sender", namespace="/robot/vision")
        self.cameras = dict(CAMERAS if cameras is None else cameras)
        self.target_ip = target_ip
        self.jpeg_quality = jpeg_quality
        self.send_every = max(1, send_every)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.started_at = time.monotonic()
        self.frames_seen = {camera: 0 for camera in self.cameras}
        self.frames_sent = {camera: 0 for camera in self.cameras}
        self.bytes_sent = {camera: 0 for camera in self.cameras}
        self.last_frame_at: dict[str, float] = {}
        self.last_quality: dict[str, int] = {}
        self.errors: dict[str, str] = {}
        self._subscriptions_keepalive: list[Any] = []
        for camera, (topic, port) in self.cameras.items():
            subscription = self.create_subscription(
                Image,
                topic,
                lambda message, camera=camera: self._on_image(camera, message),
                qos_profile_sensor_data,
            )
            self._subscriptions_keepalive.append(subscription)
            self.get_logger().info(
                f"{camera}: {topic} -> {self.target_ip}:{port}"
            )

    def _on_image(self, camera: str, message: Image) -> None:
        self.frames_seen[camera] += 1
        if self.frames_seen[camera] % self.send_every:
            return
        try:
            payload, quality = encode_jpeg(
                image_to_bgr(message), self.jpeg_quality
            )
            self.socket.sendto(payload, (self.target_ip, self.cameras[camera][1]))
            self.frames_sent[camera] += 1
            self.bytes_sent[camera] += len(payload)
            self.last_frame_at[camera] = time.monotonic()
            self.last_quality[camera] = quality
            self.errors.pop(camera, None)
        except Exception as exc:
            self.errors[camera] = str(exc)

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        ready = sorted(self.last_frame_at)
        stale = sorted(
            camera
            for camera, last_frame_at in self.last_frame_at.items()
            if now - last_frame_at > 5.0
        )
        errors = dict(self.errors)
        if stale:
            errors["stale"] = "超过 5 秒未发送：" + "、".join(stale)
        elapsed = max(now - self.started_at, 1e-6)
        return {
            "status": (
                "failed" if errors
                else "running" if len(ready) == len(self.cameras)
                else "starting"
            ),
            "target_ip": self.target_ip,
            "ports": {camera: port for camera, (_topic, port) in self.cameras.items()},
            "cameras": ready,
            "frames_sent": dict(self.frames_sent),
            "bytes_sent": dict(self.bytes_sent),
            "fps": {
                camera: round(count / elapsed, 2)
                for camera, count in self.frames_sent.items()
            },
            "jpeg_quality": dict(self.last_quality),
            "error": "; ".join(f"{key}: {value}" for key, value in errors.items()) or None,
        }

    def close(self) -> None:
        self.socket.close()


def empty_status(
    target_ip: str,
    status: str,
    error: str | None = None,
    cameras: dict[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    cameras = CAMERAS if cameras is None else cameras
    return {
        "status": status,
        "target_ip": target_ip,
        "ports": {camera: port for camera, (_topic, port) in cameras.items()},
        "cameras": [],
        "frames_sent": {camera: 0 for camera in cameras},
        "bytes_sent": {camera: 0 for camera in cameras},
        "fps": {camera: 0.0 for camera in cameras},
        "jpeg_quality": {},
        "error": error,
    }


def run_self_test() -> int:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "xCore UDP self-test",
        (80, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    payload, quality = encode_jpeg(frame, 50)
    assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
    assert len(payload) <= MAX_UDP_PAYLOAD
    print(
        f"UDP video self-test passed: {len(payload)} bytes, quality={quality}, "
        "ports=6000/6001/6002"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forward robot_vision ROS images to the VR device over UDP"
    )
    parser.add_argument("--target-ip", default="172.16.13.184")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument(
        "--camera-registry",
        type=Path,
        help="Read enabled color topics from the camera driver's YAML registry",
    )
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--send-every", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if args.status_file is None:
        parser.error("--status-file is required")
    try:
        socket.inet_aton(args.target_ip)
    except OSError:
        parser.error("--target-ip must be an IPv4 address")
    if not 5 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [5, 100]")
    if args.send_every <= 0:
        parser.error("--send-every must be positive")
    try:
        cameras = (
            load_video_cameras(args.camera_registry)
            if args.camera_registry else CAMERAS
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    node: UdpVideoSenderNode | None = None
    clean_stop = False
    write_status(
        args.status_file, empty_status(args.target_ip, "starting", cameras=cameras)
    )
    try:
        rclpy.init(args=None, domain_id=0)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        node = UdpVideoSenderNode(
            args.target_ip, args.jpeg_quality, args.send_every, cameras
        )
        deadline = time.monotonic() + args.startup_timeout
        last_status_at = 0.0
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            status = node.status()
            if status["status"] == "starting" and now >= deadline:
                missing = sorted(set(cameras) - set(status["cameras"]))
                raise RuntimeError("未收到 ROS 相机话题：" + "、".join(missing))
            if now - last_status_at >= 1.0:
                write_status(args.status_file, status)
                last_status_at = now
        clean_stop = True
        return 0
    except Exception as exc:
        status = node.status() if node is not None else empty_status(
            args.target_ip, "failed", cameras=cameras
        )
        status["status"] = "failed"
        status["error"] = str(exc)
        write_status(args.status_file, status)
        print(f"UDP video sender failed: {exc}", flush=True)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
            node.close()
        if rclpy.ok():
            rclpy.shutdown()
        if clean_stop:
            status = empty_status(args.target_ip, "stopped", cameras=cameras)
            if node is not None:
                status["frames_sent"] = node.frames_sent
                status["bytes_sent"] = node.bytes_sent
                status["jpeg_quality"] = node.last_quality
            write_status(args.status_file, status)


if __name__ == "__main__":
    raise SystemExit(main())
