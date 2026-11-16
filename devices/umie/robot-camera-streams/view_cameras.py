#!/usr/bin/env python3
"""Display JPEG images received from ROS; never open the camera hardware."""

import argparse
import socket
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs="+", default=["head", "left_wrist", "right_wrist"])
    args = parser.parse_args()
    rclpy.init()
    node = Node("camera_preview_" + socket.gethostname().replace("-", "_"))
    frames = {}
    subscriptions = []

    def receive(name, message):
        frame = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frames[name] = (frame, time.monotonic())

    for name in args.cameras:
        subscriptions.append(node.create_subscription(
            CompressedImage, f"/robot/sensors/cameras/{name}/color/image_raw/compressed",
            lambda message, name=name: receive(name, message), qos_profile_sensor_data))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            tiles = []
            for name in args.cameras:
                if name in frames:
                    image, stamp = frames[name]
                    tile = cv2.resize(image, (480, 360))
                    label = f"{name} | age {time.monotonic()-stamp:.1f}s"
                else:
                    tile = np.zeros((360, 480, 3), dtype=np.uint8)
                    label = f"{name} | waiting for ROS images"
                cv2.rectangle(tile, (0, 0), (480, 28), (0, 0, 0), -1)
                cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            cv2.imshow("Robot cameras - ROS JPEG subscribers - Q to quit", np.hstack(tiles))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
