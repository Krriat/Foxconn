#!/usr/bin/env python3
"""Verify live ROS joint feedback, suction readback and freshness diagnostics."""

import argparse
import json
import math
import socket
import time
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path, default=Path("state_validation.json"))
    args = parser.parse_args()
    rclpy.init()
    node = Node("state_validation_" + socket.gethostname().replace("-", "_"))
    data = {name: {"samples": 0, "first": None, "last": None, "errors": []}
            for name in ["left_arm", "right_arm", "left_suction", "right_suction"]}
    diagnostics = {}
    subscriptions = []

    def received(key):
        now = time.monotonic()
        record = data[key]
        if record["first"] is None:
            record["first"] = now
        record["last"] = now
        record["samples"] += 1

    def joints(side, message):
        record = data[side + "_arm"]
        expected = [f"{side}_joint_{i}" for i in range(1, 8)]
        if list(message.name) != expected or len(message.position) != 7:
            if len(record["errors"]) < 5:
                record["errors"].append("Joint names/order/position count differ from seven-joint contract")
            return
        if not all(math.isfinite(v) for v in message.position):
            if len(record["errors"]) < 5:
                record["errors"].append("Nonfinite joint angle")
            return
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp_ns <= record.get("last_stamp_ns", -1) and len(record["errors"]) < 5:
            record["errors"].append("Non-increasing sample timestamp")
        record.update(names=list(message.name), position_rad=list(message.position),
                      position_deg=[round(math.degrees(v), 3) for v in message.position],
                      last_stamp_ns=stamp_ns)
        received(side + "_arm")

    def suction(side, message):
        data[side + "_suction"]["enabled"] = bool(message.data)
        received(side + "_suction")

    def diagnostic(message):
        for status in message.status:
            diagnostics[status.name] = {
                "level": (int.from_bytes(status.level, "little")
                          if isinstance(status.level, (bytes, bytearray)) else int(status.level)),
                "message": status.message,
                "hardware_id": status.hardware_id,
                "values": {v.key: v.value for v in status.values},
            }

    for side in ["left", "right"]:
        subscriptions.append(node.create_subscription(
            JointState, f"/robot/upperlimb/joint_states/{side}_arm",
            lambda message, side=side: joints(side, message), 10))
        subscriptions.append(node.create_subscription(
            Bool, f"/robot/suction/{side}/state",
            lambda message, side=side: suction(side, message), 10))
    subscriptions.append(node.create_subscription(
        DiagnosticArray, "/robot/hardware_state/diagnostics", diagnostic, 10))
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        now = time.monotonic()
        for record in data.values():
            first, last = record.pop("first"), record.pop("last")
            elapsed = 0 if first is None or last is None else last - first
            record["received_hz"] = round((record["samples"] - 1) / elapsed, 2) if elapsed > 0 else 0.0
            record["last_sample_age_sec"] = None if last is None else round(now - last, 3)
        report = {"duration_sec": args.seconds, "streams": data,
                  "diagnostics": diagnostics,
                  "suction_semantics": "M050 DO readback: true=enabled/relay closed; not vacuum detection"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        node.destroy_node()
        rclpy.shutdown()
    streams_ok = all(
        r["samples"] >= 10 and not r["errors"]
        and r["last_sample_age_sec"] is not None and r["last_sample_age_sec"] < 1
        and r["received_hz"] >= (10 if name.endswith("arm") else 5)
        for name, r in data.items())
    diagnostics_ok = set(diagnostics) == {
        "left_arm_feedback", "right_arm_feedback", "suction_output_feedback"}
    diagnostics_ok = diagnostics_ok and all(s["level"] == 0 for s in diagnostics.values())
    return 0 if streams_ok and diagnostics_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
