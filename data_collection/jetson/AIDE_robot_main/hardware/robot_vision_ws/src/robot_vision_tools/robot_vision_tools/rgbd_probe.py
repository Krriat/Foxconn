#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy

from message_filters import (
    ApproximateTimeSynchronizer,
    Subscriber,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def join_topic(root: str, relative: str) -> str:
    return f"{root.rstrip('/')}/{relative.lstrip('/')}"


def stamp_to_seconds(msg_stamp) -> float:
    return (
        float(msg_stamp.sec)
        + float(msg_stamp.nanosec) * 1e-9
    )


class RgbdProbe(Node):

    def __init__(self) -> None:
        super().__init__("rgbd_probe", namespace="/robot/vision")

        self.declare_parameter(
            "camera_root",
            "/robot/sensors/cameras/head",
        )
        self.declare_parameter(
            "color_topic",
            "color/image_raw",
        )
        self.declare_parameter(
            "depth_topic",
            "aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "aligned_depth_to_color/camera_info",
        )
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("log_every_n_frames", 30)

        camera_root = str(
            self.get_parameter("camera_root").value
        )

        color_topic = join_topic(
            camera_root,
            str(self.get_parameter("color_topic").value),
        )
        depth_topic = join_topic(
            camera_root,
            str(self.get_parameter("depth_topic").value),
        )
        info_topic = join_topic(
            camera_root,
            str(
                self.get_parameter(
                    "camera_info_topic"
                ).value
            ),
        )

        queue_size = int(
            self.get_parameter("sync_queue_size").value
        )
        slop_sec = float(
            self.get_parameter("sync_slop_sec").value
        )

        self.log_every_n_frames = int(
            self.get_parameter(
                "log_every_n_frames"
            ).value
        )

        self.latest_camera_info: Optional[
            CameraInfo
        ] = None
        self.frame_count = 0

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.color_sub = Subscriber(
            self,
            Image,
            color_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=queue_size,
            slop=slop_sec,
        )
        self.synchronizer.registerCallback(
            self.rgbd_callback
        )

        self.get_logger().info(
            "RGB-D probe started\n"
            f"  camera_root: {camera_root}\n"
            f"  color:       {color_topic}\n"
            f"  depth:       {depth_topic}\n"
            f"  camera_info: {info_topic}\n"
            f"  queue_size:  {queue_size}\n"
            f"  slop_sec:    {slop_sec}"
        )

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ) -> None:
        self.latest_camera_info = msg

    @staticmethod
    def read_center_depth(
        msg: Image,
    ) -> Optional[float]:
        if msg.width == 0 or msg.height == 0:
            return None

        if msg.encoding == "16UC1":
            dtype = np.dtype(
                ">u2" if msg.is_bigendian else "<u2"
            )
        elif msg.encoding == "32FC1":
            dtype = np.dtype(
                ">f4" if msg.is_bigendian else "<f4"
            )
        else:
            return None

        row_elements = msg.step // dtype.itemsize
        required_elements = msg.height * row_elements

        data = np.frombuffer(
            bytes(msg.data),
            dtype=dtype,
        )

        if data.size < required_elements:
            return None

        image = data[:required_elements].reshape(
            msg.height,
            row_elements,
        )

        center_y = msg.height // 2
        center_x = msg.width // 2

        return float(image[center_y, center_x])

    def rgbd_callback(
        self,
        color_msg: Image,
        depth_msg: Image,
    ) -> None:
        self.frame_count += 1

        if (
            self.frame_count
            % self.log_every_n_frames
            != 0
        ):
            return

        color_stamp = stamp_to_seconds(
            color_msg.header.stamp
        )
        depth_stamp = stamp_to_seconds(
            depth_msg.header.stamp
        )
        time_difference_ms = abs(
            color_stamp - depth_stamp
        ) * 1000.0

        center_depth_raw = self.read_center_depth(
            depth_msg
        )

        if self.latest_camera_info is None:
            self.get_logger().warning(
                "RGB-D data received, but CameraInfo "
                "has not arrived yet"
            )
            return

        info = self.latest_camera_info

        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])

        self.get_logger().info(
            "\n"
            f"frame_count: {self.frame_count}\n"
            f"color: "
            f"{color_msg.width}x{color_msg.height}, "
            f"encoding={color_msg.encoding}, "
            f"frame={color_msg.header.frame_id}\n"
            f"depth: "
            f"{depth_msg.width}x{depth_msg.height}, "
            f"encoding={depth_msg.encoding}, "
            f"frame={depth_msg.header.frame_id}\n"
            f"sync difference: "
            f"{time_difference_ms:.3f} ms\n"
            f"center depth raw: {center_depth_raw}\n"
            f"camera_info frame: "
            f"{info.header.frame_id}\n"
            f"fx={fx:.3f}, fy={fy:.3f}, "
            f"cx={cx:.3f}, cy={cy:.3f}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = RgbdProbe()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
