#!/usr/bin/env python3

"""订阅 ROS 2 图像话题，识别 ArUco 码并发布检测到的 ID。

默认订阅头部相机彩色图，发布统一命名空间下的 marker_id 话题：
  订阅: /robot/sensors/cameras/head/color/image_raw
  发布: /robot/vision/aruco/marker_id   (std_msgs/Int32)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def join_topic(root: str, relative: str) -> str:
    """拼接两个 ROS 话题路径，避免重复斜杠。"""
    return f"{root.rstrip('/')}/{relative.lstrip('/')}"


class ArucoScannerNode(Node):
    """订阅彩色图像，检测 ArUco 码并去重后发布 ID。"""

    def __init__(self) -> None:
        super().__init__("id_detector", namespace="/robot/vision/aruco")
        self.declare_parameter("detection_enabled", False)
        self.declare_parameter("enable_topic", "/robot/vision/aruco/enable")
        self.enable_detection = bool(self.get_parameter("detection_enabled").value)
        self.control_sub = self.create_subscription(
            Int32, self.get_parameter("enable_topic").value,
            self.control_callback, 10,
        )

        # ---- 可配置参数 ----
        self.declare_parameter(
            "camera_root",
            "/robot/sensors/cameras/head",
        )
        self.declare_parameter(
            "image_topic",
            "color/image_raw",
        )
        self.declare_parameter(
            "publish_namespace",
            "/robot/vision/aruco",
        )
        self.declare_parameter(
            "publish_topic",
            "marker_id",
        )
        self.declare_parameter(
            "aruco_dict",
            "DICT_6X6_250",
        )
        self.declare_parameter(
            "deduplicate_interval_sec",
            3.0,
        )
        self.declare_parameter(
            "show_preview",
            False,
        )
        self.declare_parameter(
            "preview_window_name",
            "ArUco Scan",
        )

        camera_root = str(
            self.get_parameter("camera_root").value
        )
        image_topic = join_topic(
            camera_root,
            str(self.get_parameter("image_topic").value),
        )
        publish_namespace = str(
            self.get_parameter("publish_namespace").value
        )
        publish_topic = join_topic(
            publish_namespace,
            str(self.get_parameter("publish_topic").value),
        )

        self.deduplicate_interval_sec = float(
            self.get_parameter(
                "deduplicate_interval_sec"
            ).value
        )
        self.show_preview = bool(
            self.get_parameter("show_preview").value
        )
        self.preview_window_name = str(
            self.get_parameter("preview_window_name").value
        )

        # ---- ArUco 字典 ----
        dict_name = str(
            self.get_parameter("aruco_dict").value
        )
        if dict_name not in ARUCO_DICTS:
            available = ", ".join(sorted(ARUCO_DICTS.keys()))
            raise ValueError(
                f"Unknown ArUco dictionary '{dict_name}'. "
                f"Available: {available}"
            )
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICTS[dict_name]
        )
        self.aruco_params = (
            cv2.aruco.DetectorParameters()
        )
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ---- 发布 / 订阅 ----
        self.pub_marker_id = self.create_publisher(
            Int32, publish_topic, 10
        )

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # ---- 去重缓存 ----
        self.barcode_cache: dict[str, float] = {}

        # ---- 预览窗口 ----
        if self.show_preview:
            cv2.namedWindow(
                self.preview_window_name,
                cv2.WINDOW_NORMAL,
            )

        self.get_logger().info(
            "ArUco scanner started\n"
            f"  subscribe:      {image_topic}\n"
            f"  publish:        {publish_topic}\n"
            f"  aruco_dict:     {dict_name}\n"
            f"  dedup_interval: {self.deduplicate_interval_sec}s\n"
            f"  show_preview:   {self.show_preview}"
        )

    # ----------------------------------------------------------------
    # 去重
    # ----------------------------------------------------------------
    def is_duplicate(self, marker_id: int) -> bool:
        """同一 ID 在去重窗口内视为重复，返回 True。"""
        now = time.time()

        expired = [
            k
            for k, v in self.barcode_cache.items()
            if now - v > self.deduplicate_interval_sec
        ]
        for k in expired:
            del self.barcode_cache[k]

        key = str(marker_id)
        if key in self.barcode_cache:
            return True
        self.barcode_cache[key] = now
        return False

    # ----------------------------------------------------------------
    # 图像回调
    # ----------------------------------------------------------------
    def image_callback(self, msg: Image) -> None:
        if not self.enable_detection:
            return
        # 1. ROS Image -> OpenCV BGR
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except CvBridgeError as exc:
            self.get_logger().error(
                f"cv_bridge conversion failed: {exc}"
            )
            return

        if frame is None or frame.size == 0:
            return

        # 2. 转灰度 + 检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        # corners, ids, _ = cv2.aruco.detectMarkers(
        #     gray,
        #     self.aruco_dict,
        #     parameters=self.aruco_params,
        # )

        # 3. 有结果就画框 + 去重 + 发布
        if ids is not None and len(ids) > 0:
            if self.show_preview:
                cv2.aruco.drawDetectedMarkers(
                    frame, corners, ids
                )

            for mid in ids.flatten():
                if not self.is_duplicate(int(mid)):
                    self.get_logger().info(
                        f"Detected ArUco ID: {int(mid)}"
                    )
                    out = Int32()
                    out.data = int(mid)
                    self.pub_marker_id.publish(out)

        # 4. 预览（可选）
        if self.show_preview:
            cv2.imshow(self.preview_window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info(
                    "'q' pressed, shutting down"
                )
                rclpy.shutdown()

    # ----------------------------------------------------------------
    # 资源清理
    # ----------------------------------------------------------------
    def control_callback(self, msg: Int32) -> None:
        if msg.data in (0, 1):
            self.enable_detection = bool(msg.data)

    def destroy_node(self) -> None:  # type: ignore[override]
        try:
            if self.show_preview:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
        finally:
            super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ArucoScannerNode] = None

    try:
        node = ArucoScannerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"aruco_scanner failed: {error}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
