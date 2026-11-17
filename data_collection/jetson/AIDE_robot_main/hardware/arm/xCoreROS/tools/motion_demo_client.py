#!/usr/bin/env python3
"""Map the referenced mode-0 motion demo to the existing ROS service names.

Default: validate/print requests only. --execute explicitly sends motion.
"""
import argparse
import json
import math
from pathlib import Path

import yaml


def quaternion_to_rpy(orientation):
    values = [float(orientation[axis]) for axis in "xyzw"]
    norm = math.hypot(*values)
    if not all(math.isfinite(v) for v in values) or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("orientation must be a finite nonzero quaternion")
    x, y, z, w = [v / norm for v in values]
    return [math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1., min(1., 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))]


def requests_from_yaml(path, action_id=None, prefix="/robot_02/upperlimb"):
    data = yaml.safe_load(Path(path).read_text())
    result = []

    def walk(value, tcp_type):
        if isinstance(value, list):
            for item in value:
                walk(item, tcp_type)
        elif isinstance(value, dict):
            tcp_type = value.get("tcp_type", tcp_type)
            if "type" in value and "motion" in value:
                if action_id is not None and value.get("id") != action_id:
                    return
                result.extend(convert(value, tcp_type))
            else:
                for key in ("tasks", "locations", "behaviors", "actions"):
                    if key in value:
                        walk(value[key], tcp_type)

    def convert(action, tcp_type):
        if action.get("mode", 0) != 0:
            raise ValueError("this client accepts resolved mode 0; use the BT dispatcher for local modes")
        motion = action["motion"]
        if motion.get("zone", 0) != 0 or motion.get("is_async", False):
            raise ValueError("the motion demo requires zone=0 and is_async=false")
        kind = action["type"]
        source = action.get("source", {})
        suffix = {"moveabsj": "move_absj", "movej": "movej_by_path", "movel": "move_l"}[kind]
        if kind != "movej" and tcp_type != "sdk_tcp":
            raise ValueError("existing MoveAbsJ/MoveL interfaces use the active SDK tool; select sdk_tcp here")
        target = source.get("right_arm", {}).get("pose", {})
        speed = float(motion["line_speed"] if kind == "movel" else motion["joint_speed"])
        if not math.isfinite(speed) or speed <= 0 or (kind != "movel" and speed > 1):
            raise ValueError("motion speed is outside its valid range")
        if kind == "moveabsj":
            points = source.get("joints", [])
            if not points:
                raise ValueError("MoveAbsJ source.joints is empty")
            rows = []
            for index, point in enumerate(points):
                if len(point) != 7 or not all(math.isfinite(float(v)) for v in point):
                    raise ValueError("MoveAbsJ needs 7 finite joints")
                rows.append({"id": action["id"], "point": index, "service": f"{prefix}/{suffix}/right_arm",
                             "type": "MoveAbsJ", "request": {"joint_positions": point, "speed_mm_s": 4000.,
                             "joint_speed_scale": speed, "zone_mm": 0.}})
            return rows
        pose = {"position": target.get("position", {}), "orientation": target.get("orientation", {})}
        if pose["position"] and not all(math.isfinite(float(pose["position"][a])) for a in "xyz"):
            raise ValueError("position must be finite")
        if pose["orientation"]:
            angles = quaternion_to_rpy(pose["orientation"])
        else:
            angles = None
        conf = target.get("conf_data", [])
        if conf and (len(conf) != 8 or any(type(v) is not int for v in conf)):
            raise ValueError("conf_data must have 8 integers")
        if "elbow" in target and not math.isfinite(float(target["elbow"])):
            raise ValueError("elbow must be finite")
        request = {"use_elbow": "elbow" in target, "elbow": target.get("elbow", 0.), "conf_data": conf}
        if kind == "movej":
            request.update(use_cartesian_target=True, use_current_pose=not bool(target),
                           use_current_orientation=angles is None, tcp_type=tcp_type,
                           pose={"position": pose["position"] or {a: 0. for a in "xyz"},
                                 "orientation": pose["orientation"] or {"x": 0., "y": 0., "z": 0., "w": 1.}},
                           joint_speed_scale=speed, zone_mm=0., is_async=False)
            request.update({key: value for key, value in data.get("feedback", {}).items()
                            if key in ("joint_tolerance_rad", "position_tolerance_m", "orientation_tolerance_rad",
                                       "stable_samples", "poll_interval_ms", "timeout_s")})
            service_type = "MoveJByPath"
        else:
            if not pose["position"] or angles is None:
                raise ValueError("this MoveL demo client needs explicit position and orientation")
            request.update(position=[pose["position"][a] for a in "xyz"], orientation_rpy=angles,
                           speed_mm_s=speed * 1000., zone_mm=0.)
            service_type = "MoveL"
        return [{"id": action["id"], "service": f"{prefix}/{suffix}/right_arm", "type": service_type, "request": request}]

    walk(data, "sdk_tcp")
    if not result:
        raise ValueError("no matching action")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=Path(__file__).resolve().parents[2] / "bo_project/open_box/运动类型演示.yaml")
    parser.add_argument("--action-id")
    parser.add_argument("--prefix", default="/robot_02/upperlimb")
    parser.add_argument("--execute", action="store_true", help="send the printed motion requests")
    args = parser.parse_args()
    requests = requests_from_yaml(args.yaml, args.action_id, args.prefix.rstrip("/"))
    print(json.dumps(requests, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Validated requests only; no ROS connection or motion command sent.")
        return

    import rclpy
    from rokae_interfaces import srv
    from rosidl_runtime_py.set_message import set_message_fields

    rclpy.init()
    node = rclpy.create_node("motion_demo_client")
    try:
        for row in requests:
            service_type = getattr(srv, row["type"])
            client = node.create_client(service_type, row["service"])
            if not client.wait_for_service(timeout_sec=5.):
                raise RuntimeError("service unavailable: " + row["service"])
            request = service_type.Request()
            set_message_fields(request, row["request"])
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=660.)
            if not future.done() or future.result() is None:
                raise RuntimeError("request did not return: " + row["id"])
            response = future.result()
            print(row["id"], response.message)
            if not response.success:
                raise RuntimeError("motion failed; remaining actions were not sent")
            node.destroy_client(client)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
