# -*- coding: utf-8 -*-
"""读取左臂关节位置和末端笛卡尔位置。"""

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


def print_cart_pose(cart_pose: xCoreSDK_python.CartesianPosition):
    print(f"elbow_rad,{cart_pose.elbow}")
    print(f"hasElbow,{cart_pose.hasElbow}")
    print(f"confData,{','.join(map(str, cart_pose.confData))}")
    print(f"external,{','.join(map(str, cart_pose.external))}")
    print(f"trans_m,{','.join(map(str, cart_pose.trans))}")
    print(f"rpy_rad,{','.join(map(str, cart_pose.rpy))}")


def get_jointpos(robot: xCoreSDK_python.ArRobot, ec: dict):
    """读取关节位置；前7位是机械臂关节，后面是附加轴。"""
    joint_pos = robot.jointPos(ec)
    print_log("jointPos", ec)
    print(f"joint_pos_rad: {list(joint_pos)}")


def get_cartpos(robot: xCoreSDK_python.ArRobot, ec: dict):
    """读取工具相对参考坐标系的末端位置。"""
    pos = robot.posture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print_log("posture", ec)
    print(f"pos trans_m: {list(pos[:3])}")
    print(f"pos rpy_rad: {list(pos[3:])}")

    # cart_posture = robot.cartPosture(xCoreSDK_python.CoordinateType.endInRef, ec)
    # print_log("cartPosture", ec)
    # print_cart_pose(cart_posture)


if __name__ == "__main__":
    try:
        robot = xCoreSDK_python.ArRobot("192.168.1.160", "192.168.1.10")
        ec = {}
        get_jointpos(robot, ec)
        get_cartpos(robot, ec)
    except Exception as e:
        print(f"An error occurred: {e}")
