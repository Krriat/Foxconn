#!/usr/bin/python3
"""Read-only ROS diagnostics. Creates subscriptions; no command or service calls."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 86400:
        parser.error("seconds must be in (0, 86400]")
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node("readonly_feedback_monitor_" + socket.gethostname().replace("-", "_"))
    subscriptions, latest, counts, stamps = [], {}, defaultdict(int), defaultdict(list)
    log = (args.output/"samples.jsonl").open("w", buffering=1)
    def status(topic, message):
        try:
            value = json.loads(message.data)
        except ValueError:
            value = {"error":"invalid JSON"}
        latest[topic] = value; counts[topic] += 1
        log.write(json.dumps({"receive_ns":time.time_ns(),"topic":topic,"data":value})+"\n")
    def camera(name, message):
        counts[name] += 1
        stamps[name].append((time.monotonic(),time.time_ns()-(message.header.stamp.sec*10**9+message.header.stamp.nanosec)))
    for topic in ("/robot/policy/status", "/robot/policy/inference_status", "/robot/policy/trajectory_status", "/robot/policy/servo_trace"):
        subscriptions.append(node.create_subscription(String,topic,lambda msg,t=topic:status(t,msg),qos_profile_sensor_data))
    for name in ("head", "left_wrist", "right_wrist"):
        topic=f"/robot/sensors/cameras/{name}/color/image_raw/compressed"
        subscriptions.append(node.create_subscription(CompressedImage,topic,lambda msg,n=name:camera(n,msg),qos_profile_sensor_data))
    start = time.monotonic()
    try:
        while time.monotonic()-start < args.seconds:
            rclpy.spin_once(node,timeout_sec=.1)
    finally:
        result={"mode":"read_only_subscriptions_only", "hostname":socket.gethostname(),
                "finished_at":datetime.now().astimezone().isoformat(),"counts":dict(counts),"latest":latest,"cameras":{}}
        for name, rows in stamps.items():
            ages=sorted(row[1]/1e6 for row in rows)
            result["cameras"][name]={"received_hz":(len(rows)-1)/(rows[-1][0]-rows[0][0]) if len(rows)>1 else 0,
                "source_to_receive_ms_median":ages[len(ages)//2],"source_to_receive_ms_max":max(ages),
                "age_scope":"Host image-read stamp to callback; includes relative clock error, not hardware exposure age"}
        (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
        log.close();node.destroy_node();rclpy.shutdown()
    print(json.dumps({"hostname":result["hostname"],"counts":result["counts"],"cameras":result["cameras"]},indent=2))


if __name__ == "__main__":
    main()
