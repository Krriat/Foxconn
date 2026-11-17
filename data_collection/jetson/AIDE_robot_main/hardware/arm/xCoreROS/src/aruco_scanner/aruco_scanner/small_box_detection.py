#!/usr/bin/env python3
"""Detect the rightmost stable small box and publish its grasp target."""

import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from box_detection_interfaces.msg import SmallBoxTarget
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32
from ultralytics import YOLO

from .big_box_detection import (
    clamp_uv,
    offset_quad,
    order_quad,
    transform_point,
)


DEFAULT_MODEL_PATH = os.path.join(
    get_package_share_directory("aruco_scanner"),
    "models",
    "small_box_best.pt",
)


def robust_pixel_to_3d(
    u, v, depth_image, depth_encoding, intrinsics,
    radius=4, minimum_depth=0.15, maximum_depth=2.0,
):
    """Deproject using the median of valid depth pixels around the target."""
    height, width = depth_image.shape[:2]
    u, v = clamp_uv(u, v, width, height)
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    samples = np.asarray(depth_image[y0:y1, x0:x1], dtype=np.float32)
    if depth_encoding != "32FC1":
        samples *= 0.001
    valid = samples[
        np.isfinite(samples)
        & (samples >= minimum_depth)
        & (samples <= maximum_depth)
    ]
    minimum_samples = max(5, int(samples.size * 0.2))
    if valid.size < minimum_samples:
        return None

    depth = float(np.median(valid))
    fx, fy, cx, cy = intrinsics
    if fx <= 0.0 or fy <= 0.0:
        return None
    return np.array([
        (u - cx) * depth / fx,
        (v - cy) * depth / fy,
        depth,
    ], dtype=float)


def normalized_box_angle(quad):
    """Return the top-edge image angle normalized to [-90, 90) degrees."""
    edge = quad[1] - quad[0]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    return (angle + 90.0) % 180.0 - 90.0


class SmallBoxDetectionNode(Node):
    def __init__(self):
        super().__init__("small_box_detection_node")
        self._declare_parameters()
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.depth_encoding = ""
        self.intrinsics = None
        self.enabled = False
        self.candidate_center = None
        self.candidate_since = None
        self.stable_detection = None

        model_path = self.get_parameter("model_path").value
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Small-box YOLO model does not exist: {model_path}"
            )
        self.get_logger().info(f"Loading small-box model: {model_path}")
        self.model = YOLO(model_path)
        self.transform = np.asarray(
            self.get_parameter("extrinsic_matrix").value, dtype=float
        ).reshape(4, 4)
        self.edge_offsets = np.asarray(
            self.get_parameter("edge_offsets").value, dtype=float
        )
        self.gui_enabled = bool(
            self.get_parameter("enable_gui").value
            and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        )
        if self.get_parameter("enable_gui").value and not self.gui_enabled:
            self.get_logger().warn(
                "No graphical display; small-box GUI disabled"
            )

        self.create_subscription(
            Image, self.get_parameter("image_topic").value,
            self.color_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.get_parameter("depth_topic").value,
            self.depth_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self.camera_info_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Int32, self.get_parameter("enable_topic").value,
            self.enable_callback, 10
        )
        self.target_pub = self.create_publisher(
            SmallBoxTarget, self.get_parameter("target_topic").value, 10
        )
        rate = max(float(self.get_parameter("publish_rate").value), 0.1)
        self.timer = self.create_timer(1.0 / rate, self.process_frame)
        self.get_logger().info(
            "Small-box detector ready; publish 1 to /small_box/enable"
        )

    def _declare_parameters(self):
        self.declare_parameters("", [
            ("model_path", DEFAULT_MODEL_PATH),
            ("image_topic", "/camera/color/image_raw"),
            ("depth_topic", "/camera/depth/image_rect_raw"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("enable_topic", "/small_box/enable"),
            ("target_topic", "/small_box/target"),
            ("frame_id", "base_link"),
            ("publish_rate", 10.0),
            ("confidence_threshold", 0.25),
            ("minimum_contour_area", 600.0),
            ("stable_time", 2.0),
            ("tracking_threshold_px", 25.0),
            ("depth_window_radius", 4),
            ("minimum_depth_m", 0.15),
            ("maximum_depth_m", 2.0),
            ("enable_gui", True),
            ("edge_offsets", [3.0, 3.0, 3.0, 3.0]),
            ("extrinsic_matrix", [
                -0.024283, -0.763799, 0.644997, 0.066737,
                -0.996673, 0.068711, 0.043844, 0.020982,
                -0.077807, -0.641787, -0.762926, 0.183970,
                0.0, 0.0, 0.0, 1.0,
            ]),
        ])

    def color_callback(self, message):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except CvBridgeError as error:
            self.get_logger().warn(f"Color conversion failed: {error}")

    def depth_callback(self, message):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="passthrough"
            )
            self.depth_encoding = message.encoding
        except CvBridgeError as error:
            self.get_logger().warn(f"Depth conversion failed: {error}")

    def camera_info_callback(self, message):
        self.intrinsics = (
            float(message.k[0]), float(message.k[4]),
            float(message.k[2]), float(message.k[5]),
        )

    def enable_callback(self, message):
        if message.data == 1:
            if not self.enabled:
                self.enabled = True
                self.reset_tracking()
                self.get_logger().info("Small-box detection enabled")
        elif message.data == 0:
            if self.enabled:
                self.enabled = False
                self.reset_tracking()
                self.get_logger().info("Small-box detection disabled")
        else:
            self.get_logger().warn(
                f"Invalid enable value {message.data}; use 1 or 0"
            )

    def reset_tracking(self):
        self.candidate_center = None
        self.candidate_since = None
        self.stable_detection = None

    def process_frame(self):
        if not self.enabled:
            return
        if self.color_image is None or self.depth_image is None or \
                self.intrinsics is None:
            self.get_logger().warn(
                "Waiting for color, aligned depth and CameraInfo",
                throttle_duration_sec=2.0,
            )
            return

        image = self.color_image.copy()
        prediction = self.model.predict(
            source=image,
            imgsz=640,
            conf=float(self.get_parameter("confidence_threshold").value),
            retina_masks=True,
            verbose=False,
            device="cpu",
        )[0]
        boxes = self.extract_boxes(prediction, image.shape[:2])
        selected = max(boxes, key=lambda item: item["center"][0]) \
            if boxes else None
        self.update_tracking(selected)

        display = image.copy()
        for item in boxes:
            cv2.polylines(
                display, [item["quad"].astype(int)], True, (100, 100, 100), 1
            )
        if self.stable_detection is not None:
            detection = self.resolve_target(self.stable_detection)
            if detection is not None:
                point, angle, uv, quad = detection
                self.publish_target(point, angle)
                cv2.polylines(
                    display, [quad.astype(int)], True, (0, 255, 255), 2
                )
                cv2.circle(display, uv, 7, (0, 0, 255), -1)
                cv2.putText(
                    display, f"STABLE angle={angle:.1f} deg", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            else:
                self.get_logger().warn(
                    "Stable box has insufficient valid depth",
                    throttle_duration_sec=2.0,
                )
        self.show_image(display)

    def extract_boxes(self, prediction, image_shape):
        if prediction.masks is None or prediction.boxes is None:
            return []
        height, width = image_shape
        output = []
        for index in range(len(prediction.masks.data)):
            if int(prediction.boxes.cls[index].item()) != 0:
                continue
            mask = prediction.masks.data[index].cpu().numpy()
            mask = cv2.resize(mask, (width, height))
            mask = (mask > 0.5).astype(np.uint8)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < float(
                self.get_parameter("minimum_contour_area").value
            ):
                continue
            approximation = cv2.approxPolyDP(
                contour, 0.05 * cv2.arcLength(contour, True), True
            )
            if len(approximation) != 4:
                approximation = cv2.boxPoints(cv2.minAreaRect(contour))
            quad = order_quad(
                np.asarray(approximation).reshape(4, 2).astype(np.float32)
            )
            output.append({"center": quad.mean(axis=0), "quad": quad})
        return output

    def update_tracking(self, selected):
        if selected is None:
            self.reset_tracking()
            return
        now = time.monotonic()
        center = selected["center"]
        threshold = float(
            self.get_parameter("tracking_threshold_px").value
        )
        if self.candidate_center is None or np.linalg.norm(
            center - self.candidate_center
        ) >= threshold:
            self.candidate_center = center.copy()
            self.candidate_since = now
            self.stable_detection = None
            return
        self.candidate_center = center.copy()
        if now - self.candidate_since >= float(
            self.get_parameter("stable_time").value
        ):
            self.stable_detection = selected

    def resolve_target(self, selected):
        quad = offset_quad(selected["quad"], self.edge_offsets)
        center = quad.mean(axis=0)
        height, width = self.depth_image.shape[:2]
        uv = clamp_uv(center[0], center[1], width, height)
        camera_point = robust_pixel_to_3d(
            *uv, self.depth_image, self.depth_encoding, self.intrinsics,
            radius=int(self.get_parameter("depth_window_radius").value),
            minimum_depth=float(
                self.get_parameter("minimum_depth_m").value
            ),
            maximum_depth=float(
                self.get_parameter("maximum_depth_m").value
            ),
        )
        base_point = transform_point(camera_point, self.transform)
        if base_point is None or not np.all(np.isfinite(base_point)):
            return None
        return base_point, normalized_box_angle(quad), uv, quad

    def publish_target(self, point, angle):
        message = SmallBoxTarget()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.get_parameter("frame_id").value
        message.center.x = float(point[0])
        message.center.y = float(point[1])
        message.center.z = float(point[2])
        message.angle_deg = float(angle)
        self.target_pub.publish(message)

    def show_image(self, image):
        if not self.gui_enabled:
            return
        cv2.imshow("Small Box Detection", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()

    def close(self):
        if self.gui_enabled:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SmallBoxDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
