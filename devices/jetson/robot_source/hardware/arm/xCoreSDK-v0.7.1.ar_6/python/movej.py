# -*- coding: utf-8 -*-
"""执行 MoveJ 关节空间运动指令。"""

from __future__ import annotations

import math
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


def movej(robot: xCoreSDK_python.ArRobot, ec: dict):
    target1 = xCoreSDK_python.CartesianPosition(
        [
            -0.5431547776023002, -0.3100642488048048, -0.3775364972240102,
            -1.237292844937695, -0.2899610409906598, 1.6727636640026757,
        ]
    )
    target1.elbow = math.radians(0)

    # target2 = xCoreSDK_python.CartesianPosition(
    #     [0.53466, -0.351432, 1.69627e-16, 1.5708, 2.06757e-16, 1.5708]
    # )
    # target2.elbow = math.radians(16.6993)

    commands = [
        xCoreSDK_python.MoveJCommand(target1, 100, 10),
        # xCoreSDK_python.MoveJCommand(target2, 100, 10),
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
        robot = xCoreSDK_python.ArRobot("192.168.2.160", "192.168.2.10")
        movej(robot, {})
    except Exception as e:
        print(f"An error occurred: {e}")
