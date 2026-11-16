# -*- coding: utf-8 -*-
"""依次将左臂和右臂的7个关节运动到指定位置。"""

from __future__ import annotations

import os
import platform
import sys

import setup_path

if platform.system() == "Windows":
    from Release.windows import xCoreSDK_python
elif platform.system() == "Linux":
    arch = platform.machine()
    if arch == "aarch64":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(root_dir, "Release", "linux", "arm"))
        import xCoreSDK_python
    elif arch == "x86_64":
        from Release.linux import xCoreSDK_python
    else:
        raise ImportError(f"不支持的Linux架构: {arch}")
else:
    raise ImportError("Unsupported operating system")

from log import print_log
from move_example import wait_robot


def moveabsj_target(
    robot: xCoreSDK_python.ArRobot, arm_name: str, target: list[float]
):
    ec = {}
    command = xCoreSDK_python.MoveAbsJCommand(target, 1000, 10)
    command_id = xCoreSDK_python.PyString()

    print(f"{arm_name}开始运动到目标位置: {target}")
    robot.moveAppend([command], command_id, ec)
    print(f"{arm_name} Command ID: {command_id.content()}")
    print_log("moveAppend", ec)

    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)
    print(f"{arm_name}已到达目标位置")


if __name__ == "__main__":
    try:
        left_robot = xCoreSDK_python.ArRobot("192.168.1.160", "192.168.1.10")
        right_robot = xCoreSDK_python.ArRobot("192.168.2.160", "192.168.2.10")

        left_target = [1.712168435628017, -1.570796327, 0, 0, 0, 0, 0]
        right_target = [-1.953017, -1.570796327, 0, 0, 0, 0, 0]

        moveabsj_target(left_robot, "左臂", left_target)
        moveabsj_target(right_robot, "右臂", right_target)
    except Exception as e:
        print(f"An error occurred: {e}")
