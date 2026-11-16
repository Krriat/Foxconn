"""Minimal helpers shared by local examples."""

import time
import platform
import os
import sys

import setup_path

if platform.system() == "Windows":
    from Release.windows import xCoreSDK_python
elif platform.system() == "Linux":
    arch = platform.machine()
    if arch == "aarch64":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        arm_lib_path = os.path.join(root_dir, "Release", "linux", "arm")
        sys.path.insert(0, arm_lib_path)
        import xCoreSDK_python
    elif arch == "x86_64":
        from Release.linux import xCoreSDK_python
    else:
        raise ImportError(f"不支持的Linux架构: {arch}")
else:
    raise ImportError("Unsupported operating system")

from log import print_separator


def wait_robot(robot, ec):
    """等待运动结束 - 通过查询机械臂是否处于运动中的方式"""
    print_separator("wait_robot", length=80)
    running = True
    while running:
        time.sleep(0.1)
        st = robot.operationState(ec)
        if (
            st == xCoreSDK_python.OperationState.idle
            or st == xCoreSDK_python.OperationState.unknown
        ):
            running = False
    print("move finished")
