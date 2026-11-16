# -*- coding: utf-8 -*-
"""执行一条 MoveL 直线运动指令。"""

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


def movel(robot: xCoreSDK_python.ArRobot, ec: dict):
    target = xCoreSDK_python.CartesianPosition(
        [
            0.528835, -0.175858, -0.339850,
            1.536734, -0.948848, 1.693041,
        ]
    )
    target.elbow = math.radians(0)

    command = xCoreSDK_python.MoveLCommand(target, 500, 10)
    command_id = xCoreSDK_python.PyString()
    robot.moveAppend([command], command_id, ec)
    print("Command ID:", command_id.content())
    print_log("moveAppend", ec)

    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


if __name__ == "__main__":
    try:
        robot = xCoreSDK_python.ArRobot("192.168.2.160", "192.168.2.10")
        movel(robot, {})
    except Exception as e:
        print(f"An error occurred: {e}")
