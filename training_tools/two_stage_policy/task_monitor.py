#!/usr/bin/env python3
"""Read-only head camera monitor. Never publishes actions or calls robot services."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from task_sequence import SceneDetector


class Monitor(Node):
    def __init__(self):
        super().__init__("robot_policy_task_monitor")
        self.detector = SceneDetector(Path(__file__).with_name("task_detector.npz"))
        self.executor_context, self.context_received = {}, 0.
        self.latest, self.last_stamp = None, 0
        self.publisher = self.create_publisher(String, "/robot/policy/task_scene", 1)
        self.subs = [self.create_subscription(CompressedImage,
            "/robot/sensors/cameras/head/color/image_raw/compressed", self.image, qos_profile_sensor_data),
            self.create_subscription(String, "/robot/policy/status", self.status, 1)]
        self.create_timer(.1, self.tick)

    def image(self, message):
        self.latest = (message, time.monotonic())

    def status(self, message):
        self.executor_context = json.loads(message.data)
        self.context_received = time.monotonic()

    def tick(self):
        if not self.latest or time.monotonic()-self.context_received > .4:
            return
        msg, received = self.latest
        stamp = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        if stamp <= self.last_stamp or time.monotonic()-received > .3:
            return
        self.last_stamp = stamp
        try:
            result = self.detector.classify(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR))
        except Exception as error:
            result = {"scene": "unknown", "error": str(error), "detector_sha256": self.detector.sha}
        result.update(schema="carton_scene_v1", run_id=self.executor_context.get("run_id"),
                      executor_heartbeat=self.executor_context.get("heartbeat"), image_time_ns=stamp,
                      send_time_ns=time.time_ns())
        self.publisher.publish(String(data=json.dumps(result, allow_nan=False)))


def main():
    rclpy.init(); node = Monitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
