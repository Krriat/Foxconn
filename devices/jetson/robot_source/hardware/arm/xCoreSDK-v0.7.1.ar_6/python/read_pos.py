# -*- coding: utf-8 -*-
"""读取并输出左臂、右臂的关节位置和末端笛卡尔位置。"""

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


def read_pos(robot: xCoreSDK_python.ArRobot, arm_name: str):
    ec = {}
    print(f"\n========== {arm_name} ==========")

    joint_pos = robot.jointPos(ec)
    print_log("jointPos", ec)
    print(f"joint_pos_rad: {list(joint_pos)}")

    pos = robot.posture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print_log("posture", ec)
    print(f"pos trans_m: {list(pos[:3])}")
    print(f"pos rpy_rad: {list(pos[3:])}")

    current_pose = robot.cartPosture(
        xCoreSDK_python.CoordinateType.endInRef, ec
    )
    print_log("cartPosture", ec)
    print(f"elbow_rad: {current_pose.elbow}")


if __name__ == "__main__":
    arms = [
        ("左臂", "192.168.1.160", "192.168.1.10"),
        ("右臂", "192.168.2.160", "192.168.2.10"),
    ]

    for arm_name, robot_ip, local_ip in arms:
        try:
            robot = xCoreSDK_python.ArRobot(robot_ip, local_ip)
            read_pos(robot, arm_name)
        except Exception as e:
            print(f"{arm_name}读取失败: {e}")
