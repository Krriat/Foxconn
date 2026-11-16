#!/usr/bin/env python3
"""Publish host-local cameras as ROS 2 raw/JPEG streams with camera metadata."""

import argparse
import array
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
import yaml


PREFIX = "/robot/sensors/cameras"
IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class CameraPublisher(Node):
    def __init__(self, config):
        super().__init__(config["node_name"])
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.threads = []
        self.states = {}
        self.latest_frames = {}
        self.status_pub = self.create_publisher(
            String, f"{PREFIX}/status/{self.get_name()}", 5
        )
        for camera in config["cameras"]:
            name = camera["name"]
            root = f"{PREFIX}/{name}/color"
            pubs = (
                self.create_publisher(Image, f"{root}/image_raw", IMAGE_QOS),
                self.create_publisher(CompressedImage, f"{root}/image_raw/compressed", IMAGE_QOS),
                self.create_publisher(CameraInfo, f"{root}/camera_info", IMAGE_QOS),
            )
            self.states[name] = {
                "backend": camera["backend"],
                "device": camera.get("serial", camera.get("usb_serial", camera.get("device", ""))),
                "state": "starting", "frames": 0, "reconnects": 0,
                "last_frame_monotonic": None, "fps": 0.0,
                "width": camera["width"], "height": camera["height"],
                "requested_fps": camera.get("publish_fps", camera["fps"]),
                "requested_capture_fps": camera["fps"],
                "capture_fps": 0.0, "captured_frames": 0,
                "last_capture_monotonic": None,
                "last_publish_latency_ms": None, "calibrated": False,
                "last_jpeg_bytes": 0, "error": "",
            }
            self.latest_frames[name] = None
            for target, arguments, suffix in (
                (self.worker, (camera,), "capture"),
                (self.publish_worker, (camera, pubs), "publish"),
            ):
                thread = threading.Thread(
                    target=target, args=arguments, name=f"{name}_{suffix}", daemon=True
                )
                self.threads.append(thread)
                thread.start()
        self.create_timer(1.0, self.publish_status)

    def update(self, name, **fields):
        with self.lock:
            self.states[name].update(fields)

    @staticmethod
    def v4l2_device(camera):
        if "usb_serial" not in camera:
            return camera["device"]
        matches = []
        for device in Path("/sys/class/video4linux").glob("video*"):
            if (device / "index").read_text().strip() != "0":
                continue
            parents = list(device.resolve().parents)
            serial_matches = any(
                (p / "serial").exists()
                and (p / "serial").read_text().strip() == camera["usb_serial"]
                for p in parents
            )
            interface_matches = any(
                (p / "bInterfaceNumber").exists()
                and (p / "bInterfaceNumber").read_text().strip() == camera["usb_interface"]
                for p in parents
            )
            if serial_matches and interface_matches:
                matches.append("/dev/" + device.name)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one RGB UVC device, found {matches}")
        return matches[0]

    @staticmethod
    def make_info(width, height, intrinsics=None):
        info = CameraInfo()
        info.width, info.height = width, height
        # K[0] == 0 explicitly marks an uncalibrated UVC camera in CameraInfo.
        if intrinsics is not None:
            i = intrinsics
            info.distortion_model = "plumb_bob"
            info.d = list(map(float, i.coeffs))
            info.k = [i.fx, 0.0, i.ppx, 0.0, i.fy, i.ppy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [i.fx, 0.0, i.ppx, 0.0, 0.0, i.fy, i.ppy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def worker(self, camera):
        name = camera["name"]
        while rclpy.ok() and not self.stop_event.is_set():
            pipeline = capture = None
            try:
                self.update(name, state="connecting", error="")
                intrinsics = None
                if camera["backend"] == "realsense":
                    import pyrealsense2 as rs
                    pipeline = rs.pipeline()
                    settings = rs.config()
                    settings.enable_device(str(camera["serial"]))
                    settings.enable_stream(
                        rs.stream.color, camera["width"], camera["height"],
                        rs.format.bgr8, camera["fps"],
                    )
                    profile = pipeline.start(settings)
                    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    negotiated_fps = profile.get_stream(rs.stream.color).fps()
                elif camera["backend"] == "v4l2":
                    device = self.v4l2_device(camera)
                    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera["width"])
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera["height"])
                    capture.set(cv2.CAP_PROP_FPS, camera["fps"])
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not capture.isOpened():
                        raise RuntimeError(f"Cannot open {device}")
                    negotiated_fps = capture.get(cv2.CAP_PROP_FPS)
                    self.update(name, video_device=device)
                else:
                    raise ValueError(f"Unsupported backend: {camera['backend']}")
                if abs(negotiated_fps - camera["fps"]) > 0.1:
                    raise RuntimeError(f"Capture negotiated {negotiated_fps} fps, requested {camera['fps']}")
                self.update(name, negotiated_capture_fps=negotiated_fps)
                self.get_logger().info(f"{name}: camera opened; backend={camera['backend']}")
                info = None
                fps_start, fps_frames = time.monotonic(), 0
                while rclpy.ok() and not self.stop_event.is_set():
                    if pipeline is not None:
                        frames = pipeline.wait_for_frames(timeout_ms=3000)
                        color = frames.get_color_frame()
                        if not color:
                            continue
                        frame = np.asanyarray(color.get_data())
                    else:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError("Camera returned no image")
                    # Own the pixels: the SDK may reuse its buffer after the next read.
                    stamp = self.get_clock().now().to_msg()
                    acquired = time.monotonic()
                    frame = np.array(frame, copy=True, order="C")
                    height, width = frame.shape[:2]
                    if info is None:
                        info = self.make_info(width, height, intrinsics)
                        self.update(name, width=width, height=height,
                                    calibrated=intrinsics is not None)
                    fps_frames += 1
                    with self.lock:
                        state = self.states[name]
                        state["captured_frames"] += 1
                        state["last_capture_monotonic"] = acquired
                        self.latest_frames[name] = (
                            state["captured_frames"], frame, stamp, acquired, info
                        )
                        if acquired - fps_start >= 1.0:
                            state["capture_fps"] = round(fps_frames / (acquired - fps_start), 2)
                            fps_start, fps_frames = acquired, 0
            except Exception as error:
                if self.stop_event.is_set() or not rclpy.ok():
                    break
                message = f"{type(error).__name__}: {error}"
                with self.lock:
                    self.states[name]["reconnects"] += 1
                    self.latest_frames[name] = None
                self.update(name, state="reconnecting", error=message)
                self.get_logger().warning(f"{name}: {message}; retrying in 2s")
            finally:
                if capture is not None:
                    capture.release()
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except RuntimeError:
                        pass
            self.stop_event.wait(2.0)

    def publish_worker(self, camera, pubs):
        name = camera["name"]
        raw_pub, jpeg_pub, info_pub = pubs
        frame_id = f"{name}_color_optical_frame"
        period = 1.0 / camera.get("publish_fps", camera["fps"])
        next_tick = time.monotonic()
        last_sequence = None
        fps_start, fps_frames = next_tick, 0
        while rclpy.ok() and not self.stop_event.is_set():
            if self.stop_event.wait(max(0.0, next_tick - time.monotonic())):
                break
            now = time.monotonic()
            next_tick += period
            if next_tick <= now:
                # Skip missed ticks rather than releasing a burst of queued images.
                next_tick = now + period
            with self.lock:
                sample = self.latest_frames[name]
            if sample is None or sample[0] == last_sequence:
                continue
            sequence, frame, stamp, acquired, info = sample
            if now - acquired > max(0.2, 3.0 / camera["fps"]):
                continue
            try:
                height, width = frame.shape[:2]
                info.header.stamp, info.header.frame_id = stamp, frame_id
                if raw_pub.get_subscription_count():
                    raw = Image()
                    raw.header = info.header
                    raw.height, raw.width = height, width
                    raw.encoding, raw.is_bigendian = "bgr8", 0
                    raw.step = width * 3
                    raw.data = array.array("B", frame.tobytes())
                    raw_pub.publish(raw)
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, camera.get("jpeg_quality", 80)]
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                jpeg = CompressedImage()
                jpeg.header, jpeg.format = info.header, "bgr8; jpeg compressed bgr8"
                jpeg.data = array.array("B", encoded.tobytes())
                jpeg_pub.publish(jpeg)
                info_pub.publish(info)
                last_sequence = sequence
                published = time.monotonic()
                fps_frames += 1
                with self.lock:
                    state = self.states[name]
                    state.update(state="streaming", error="", last_frame_monotonic=published,
                                 last_jpeg_bytes=len(jpeg.data),
                                 last_publish_latency_ms=round((published - acquired) * 1000, 2))
                    state["frames"] += 1
                    if published - fps_start >= 1.0:
                        state["fps"] = round(fps_frames / (published - fps_start), 2)
                        fps_start, fps_frames = published, 0
            except Exception as error:
                if self.stop_event.is_set() or not rclpy.ok():
                    break
                message = f"{type(error).__name__}: {error}"
                self.update(name, state="publish_error", error=message)
                self.get_logger().warning(f"{name}: {message}")

    def publish_status(self):
        now = time.monotonic()
        with self.lock:
            states = {}
            for name, values in self.states.items():
                state = dict(values)
                last = state.pop("last_frame_monotonic")
                state["last_frame_age_sec"] = None if last is None else round(now - last, 3)
                capture_last = state.pop("last_capture_monotonic")
                state["last_capture_age_sec"] = (
                    None if capture_last is None else round(now - capture_last, 3)
                )
                states[name] = state
        message = String()
        message.data = json.dumps({"node": self.get_name(), "cameras": states})
        self.status_pub.publish(message)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, ros_args = parser.parse_known_args()
    config = yaml.safe_load(args.config.read_text())
    cameras = config["cameras"]
    if not cameras or len({c["name"] for c in cameras}) != len(cameras):
        parser.error("cameras must be nonempty and have unique names")
    for camera in cameras:
        if camera["backend"] not in ("realsense", "v4l2"):
            parser.error("backend must be realsense or v4l2")
        if any(int(camera[k]) <= 0 for k in ("width", "height", "fps")):
            parser.error("width, height and fps must be positive")
        if not 0 < float(camera.get("publish_fps", camera["fps"])) <= camera["fps"]:
            parser.error("publish_fps must be positive and no greater than capture fps")
    rclpy.init(args=ros_args)
    node = None
    try:
        node = CameraPublisher(config)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
