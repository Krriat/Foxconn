# -*- coding: utf-8 -*-
"""依次执行四条 MoveAbsJ 关节运动指令。"""

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


def moveabsj(robot: xCoreSDK_python.ArRobot, ec: dict):
    joint_positions = [
        # [-0.90, -1.0, 0.0, -0.7576, -1.9697, 0.3788, 0.0],
        [1.18857, -1.570796327, 1.570796327, 0.2617993878, 0, 0.7853981634, 0],


        # [-1.953017, -1.087, 0.0, 0.6109, 0.0, 0.0, 0.0],
        # [-0.9091, -1.0985, 0.0, -0.7576, -1.9697, 0.3788, 0.0],
        # [-1.3258, -1.0985, 0.0, -0.7576, -1.9697, 0.8712, 0.0],
        # [-1.3658, -1.5785, 0.0, -0.967599, -2.249700, 0.511201, 0.0],
    ]
    commands = [
        xCoreSDK_python.MoveAbsJCommand(position, 1000, 10)
        for position in joint_positions
    ]

    command_id = xCoreSDK_python.PyString()
    robot.moveAppend(commands, command_id, ec)
    print("Command ID:", command_id.content())
    print_log("moveAppend", ec)

    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


if __name__ == "__main__":
    try:
        robot = xCoreSDK_python.ArRobot("192.168.1.160", "192.168.1.10")
        moveabsj(robot, {})
    except Exception as e:
        print(f"An error occurred: {e}")
