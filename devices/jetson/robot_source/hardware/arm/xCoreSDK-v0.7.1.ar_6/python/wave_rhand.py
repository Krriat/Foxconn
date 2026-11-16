# -*- coding: utf-8 -*-
"""
@file: ar_example.py
@brief: AR机型示例，点位基于AR5-5_0.7L-W4C1C1,侧装，A=90°
@copyright: Copyright (C) 2025 ROKAE (Beijing) Technology Co., LTD. All Rights Reserved.
Information in this file is the intellectual property of Rokae Technology Co., Ltd,
And may contains trade secrets that must be stored and viewed confidentially.
"""
import setup_path
import platform
import os
import sys
import time
import math
import numpy as np
from datetime import timedelta

# 根据操作系统导入相应的模块
if platform.system() == "Windows":
    from Release.windows import xCoreSDK_python
    from Release.windows.xCoreSDK_python import utility
    from Release.windows.xCoreSDK_python import RtControllerMode
elif platform.system() == "Linux":
    arch = platform.machine()
    if arch == "aarch64":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        arm_lib_path = os.path.join(root_dir, "Release", "linux", "arm")
        sys.path.insert(0, arm_lib_path)
        import xCoreSDK_python
        from xCoreSDK_python import utility
        from xCoreSDK_python import RtControllerMode
        from xCoreSDK_python import RtSupportedFields
    elif arch == "x86_64":
        from Release.linux import xCoreSDK_python
        from Release.linux.xCoreSDK_python import utility
        from Release.linux.xCoreSDK_python import RtControllerMode
        from Release.linux.xCoreSDK_python import RtSupportedFields
    else:
        raise ImportError(f"不支持的Linux架构: {arch}")
else:
    raise ImportError("Unsupported operating system")
from log import print_log, print_separator
from move_example import wait_robot

M_PI = math.pi

def rad2deg(rad):
    return rad * 180 / M_PI

def m2mm(m):
    return m * 1000

def moveabsj(robot: xCoreSDK_python.ArRobot, ec: dict):
    # joint_pos1 = [
    #     1.7121993948756893, -1.5705386659594314,0.0,0.0,0.0,0.0
    # ]
    joint_pos1 = [
        -1.2879, -1.5708, 1.5909, 0.0, 1.5909, 0.0, 0.0
    ]
    joint_pos2 = [
        -0.1894, -1.5708, 1.5909, -0.9848, 1.5909, 0.0, 0.0
    ]
    joint_pos3 = [
        -0.1894, -1.5708, 1.9318, -0.9848, 1.5909, 0.0, 0.0
    ]
    joint_pos4 = [
        -0.1894, -1.5708, 1.0606, -0.9848, 1.5909, 0.0, 0.0
    ]
    joint_pos5 = [
        -0.1894, -1.5708, 1.9318, -0.9848, 1.5909, 0.0, 0.0
    ]
    joint_pos6 = [
        -0.1894, -1.5708, 1.0606, -0.9848, 1.5909, 0.0, 0.0
    ]
    joint_pos7 = [
        -0.1894, -1.5708, 1.0606, -0.9848, 1.5909, 0.0, 0.0
    ]
    joint_pos8 = [
        -1.953017, -1.5708, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    absjcmd1 = xCoreSDK_python.MoveAbsJCommand(joint_pos1, 1500, 10)
    absjcmd2 = xCoreSDK_python.MoveAbsJCommand(joint_pos2, 1500, 10)
    absjcmd3 = xCoreSDK_python.MoveAbsJCommand(joint_pos3, 1500, 10) 
    absjcmd4 = xCoreSDK_python.MoveAbsJCommand(joint_pos4, 1500, 10) 
    absjcmd5 = xCoreSDK_python.MoveAbsJCommand(joint_pos5, 1500, 10)
    absjcmd6 = xCoreSDK_python.MoveAbsJCommand(joint_pos6, 1500, 10)
    absjcmd7 = xCoreSDK_python.MoveAbsJCommand(joint_pos7, 1500, 10)
    absjcmd8 = xCoreSDK_python.MoveAbsJCommand(joint_pos8, 1500, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend(
        [absjcmd1, absjcmd2, absjcmd3, absjcmd4, absjcmd5, absjcmd6, absjcmd7, absjcmd8], cmdID, ec
    )  # [absjcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


if __name__ == "__main__":
    try:
        # 连接机器人
        ip = "192.168.2.160"
        lcoal_ip = "192.168.2.10"
        # 实例化AR机型并连接
        robot = xCoreSDK_python.ArRobot(ip, lcoal_ip)
        ec = {}
        # get_jointpos(robot, ec)
        # get_cartpos(robot, ec)
        # calcFk(robot, ec)
        # calcIk(robot, ec)
        # movej(robot, ec)
        # movel(robot, ec)
        moveabsj(robot, ec)
        # rt_cart_controller(robot, ec)
        # rt_joint_controller(robot, ec)
        # servol(robot, ec)
        # servoj(robot, ec)
    except Exception as e:
        print(f"An error occurred: {e}")
