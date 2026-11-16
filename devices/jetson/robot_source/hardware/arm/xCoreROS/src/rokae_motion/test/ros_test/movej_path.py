#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""左臂多路点关节空间运动示例。

该脚本是 ``MoveJByPath`` 服务客户端，不直接连接 xCoreSDK。实际运动由
``rokae_driver`` 统一驱动进程完成，服务端会负责软限位、路点跳变和控制锁检查。

默认运行是 dry-run，只显示将要发送的路径。确认路径和现场安全条件后，增加
``--execute`` 才会调用真实机械臂。

JSON 路径文件可以是以下两种格式之一：

    [[q1, q2, q3, q4, q5, q6, q7], [q1, q2, q3, q4, q5, q6, q7]]

或：

    {"waypoints": [[q1, q2, q3, q4, q5, q6, q7], ...]}

角度单位为弧度；每个路点必须包含 7 个关节角。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import rclpy
from rclpy.node import Node

from rokae_interfaces.srv import MoveJByPath


SERVICE_NAME = "/robot_02/upperlimb/movej_by_path/left_arm"
JOINT_COUNT = 7
MIN_WAYPOINTS = 2
MAX_WAYPOINTS = 100

# 仅作为格式示例；发送前请替换为现场确认过的安全路径。
DEFAULT_WAYPOINTS_RAD = [
    [1.712167996, 1.570796327, -1.570796327, 0.0, 0.0, 0.0, 0.0],
    [1.712167996, 1.520796327, -1.520796327, 0.05, 0.0, 0.0, 0.0],
]


def validate_waypoints(raw: Any) -> list[list[float]]:
    """校验并复制路点，返回可直接写入 ROS 请求的列表。"""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("路径必须是二维数组，例如 [[q1, ..., q7], ...]")
    if not MIN_WAYPOINTS <= len(raw) <= MAX_WAYPOINTS:
        raise ValueError(
            f"路径必须包含 {MIN_WAYPOINTS} 到 {MAX_WAYPOINTS} 个路点"
        )

    waypoints: list[list[float]] = []
    for index, point in enumerate(raw, start=1):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise ValueError(f"第 {index} 个路点不是数组")
        if len(point) != JOINT_COUNT:
            raise ValueError(
                f"第 {index} 个路点需要 {JOINT_COUNT} 个关节角，实际为 {len(point)}"
            )
        try:
            joints = [float(value) for value in point]
        except (TypeError, ValueError) as error:
            raise ValueError(f"第 {index} 个路点包含非数字值") from error
        if not all(math.isfinite(value) for value in joints):
            raise ValueError(f"第 {index} 个路点包含 NaN 或无穷值")
        waypoints.append(joints)
    return waypoints


def load_waypoints(path_file: Path | None) -> list[list[float]]:
    if path_file is None:
        return validate_waypoints(DEFAULT_WAYPOINTS_RAD)

    try:
        document = json.loads(path_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取路径文件 {path_file}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"路径文件不是合法 JSON: {error}") from error

    if isinstance(document, dict):
        document = document.get("waypoints")
    return validate_waypoints(document)


def flatten_waypoints(waypoints: Sequence[Sequence[float]]) -> list[float]:
    return [value for point in waypoints for value in point]


class MoveJPathClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("movej_path_left_client")
        self.client = self.create_client(MoveJByPath, service_name)

    def send(self, waypoints: Sequence[Sequence[float]]):
        request = MoveJByPath.Request()
        request.joint_positions = flatten_waypoints(waypoints)
        return self.client.call_async(request)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用左臂 MoveJByPath 服务执行多路点关节轨迹"
    )
    parser.add_argument(
        "--path-file",
        type=Path,
        help="JSON 路径文件；未指定时使用脚本顶部的示例路点",
    )
    parser.add_argument(
        "--service",
        default=SERVICE_NAME,
        help=f"服务名（默认：{SERVICE_NAME}）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=130.0,
        help="等待服务响应的最长秒数（默认：130）",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正调用服务；不指定时仅校验并打印路径",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_argument_parser()
    options = parser.parse_args(args)
    if options.timeout <= 0 or not math.isfinite(options.timeout):
        parser.error("--timeout 必须是正数")

    try:
        waypoints = load_waypoints(options.path_file)
    except ValueError as error:
        parser.error(str(error))

    print(f"服务: {options.service}")
    print(f"路点数: {len(waypoints)}（每点 {JOINT_COUNT} 轴，单位 rad）")
    for index, point in enumerate(waypoints, start=1):
        print(f"  P{index}: {point}")

    if not options.execute:
        print("dry-run：未调用机械臂。确认安全后增加 --execute。")
        return 0

    rclpy.init()
    node = MoveJPathClient(options.service)
    try:
        node.get_logger().info("等待 MoveJByPath 服务上线...")
        if not node.client.wait_for_service(timeout_sec=10.0):
            node.get_logger().error(f"服务未上线：{options.service}")
            return 2

        future = node.send(waypoints)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=options.timeout
        )
        if not future.done():
            node.get_logger().error("等待服务响应超时")
            return 3

        response = future.result()
        if response is None:
            node.get_logger().error("服务调用没有返回响应")
            return 4
        if response.success:
            node.get_logger().info(f"左臂轨迹执行成功：{response.message}")
            return 0
        node.get_logger().error(f"左臂轨迹执行失败：{response.message}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
