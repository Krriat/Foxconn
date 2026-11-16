#!/usr/bin/env python3
"""Forward the three shared ROS camera topics to VR UDP/JPEG endpoints.

The RealSense devices are owned only by ``hardware/robot_vision_ws``. This
program is a consumer and can run alongside recording, inference and Web
preview without opening ``/dev/video*`` a second time.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from queue import Empty, Full, Queue

import cv2
import numpy as np

# Keep a direct/manual launch in the same local ROS graph as startup and Edge.
# These defaults must be set before importing rclpy.
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
os.environ.setdefault("ROS_DOMAIN_ID", "89")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
os.environ.setdefault(
    "CYCLONEDDS_URI",
    "file:///home/niic/aide/startup/cyclonedds_local.xml",
)

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


DESTINATION_IP = os.environ.get("XCORE_UDP_VIDEO_IP", "172.16.13.184")
JPEG_QUALITY = 50
MAX_UDP_PAYLOAD = 65_507
CAMERAS = (
    ("head", "/robot/sensors/cameras/head/color/image_raw", 6000),
    ("left_wrist", "/robot/sensors/cameras/left_wrist/color/image_raw", 6001),
    ("right_wrist", "/robot/sensors/cameras/right_wrist/color/image_raw", 6002),
)


def encode_udp_jpeg(frame: np.ndarray) -> tuple[bytes, int]:
    """Encode one JPEG that still fits in a single UDP datagram."""
    quality = JPEG_QUALITY
    while quality >= 5:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise RuntimeError("OpenCV JPEG encoding failed")
        data = buffer.tobytes()
        if len(data) <= MAX_UDP_PAYLOAD:
            return data, quality
        quality -= 5
    raise RuntimeError(f"JPEG is larger than one UDP datagram ({len(data)} bytes)")


class CameraSender:
    """Encode only the latest frame so ROS callbacks never wait on UDP."""

    def __init__(self, name: str, port: int) -> None:
        self.name = name
        self.port = port
        self.frames_received = 0
        self.frames_sent = 0
        self.last_frame_at = 0.0
        self._queue: Queue[np.ndarray | None] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"udp-{name}", daemon=True
        )
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        self.frames_received += 1
        self.last_frame_at = time.monotonic()
        item = np.ascontiguousarray(frame).copy()
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except Full:
                pass

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        try:
            while not self._stop.is_set():
                try:
                    frame = self._queue.get(timeout=0.2)
                except Empty:
                    continue
                if frame is None:
                    break
                try:
                    data, _quality = encode_udp_jpeg(frame)
                    sock.sendto(data, (DESTINATION_IP, self.port))
                    self.frames_sent += 1
                except Exception as error:
                    print(f"{self.name} UDP warning: {error}", flush=True)
        finally:
            sock.close()

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._thread.join(timeout=2.0)


class CameraUdpBridge(Node):
    def __init__(self) -> None:
        super().__init__("xcore_camera_udp_bridge")
        self._bridge = CvBridge()
        self._senders = {
            name: CameraSender(name, port) for name, _topic, port in CAMERAS
        }
        self._subscriptions = []
        for name, topic, port in CAMERAS:
            self._subscriptions.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda message, camera=name: self._on_image(camera, message),
                    qos_profile_sensor_data,
                )
            )
            self.get_logger().info(f"{name}: {topic} -> {DESTINATION_IP}:{port}")
        self.create_timer(5.0, self._report)

    def _on_image(self, camera: str, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            self._senders[camera].submit(frame)
        except Exception as error:
            self.get_logger().warning(f"{camera} frame conversion failed: {error}")

    def _report(self) -> None:
        now = time.monotonic()
        for name, sender in self._senders.items():
            age = None if sender.last_frame_at == 0.0 else now - sender.last_frame_at
            if age is None or age > 2.0:
                self.get_logger().warning(f"{name}: waiting for shared camera topic")
            else:
                self.get_logger().info(
                    f"{name}: received={sender.frames_received}, sent={sender.frames_sent}"
                )

    def close(self) -> None:
        for sender in self._senders.values():
            sender.close()


def main() -> None:
    rclpy.init()
    node = CameraUdpBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
