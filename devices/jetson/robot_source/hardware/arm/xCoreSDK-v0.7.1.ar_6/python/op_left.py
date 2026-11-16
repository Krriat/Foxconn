"""Base operation helpers for local lxy scripts.

This file keeps the power-on / operate-mode flow from example/base_example.py
in the lxy directory, with ARM64 import support.
"""

from __future__ import annotations

import os
import platform
import sys

import setup_path
from log import print_log, print_separator

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


def get_powerstate(robot, ec):
    """获取上电状态"""
    print_separator("get_powerstate", length=80)
    power_state = robot.powerState(ec)
    print_log("powerState", ec, f"powerState={power_state}")
    return power_state


def set_powerstate(robot, ec, on=True):
    """设置机器人上下电状态，True: 上电，False: 下电"""
    print_separator("set_powerstate", length=80)
    robot.setPowerState(on, ec)
    print_log("setPowerState", ec)


def get_operatemode(robot, ec):
    """获取操作模式"""
    print_separator("get_operatemode", length=80)
    operate_mode = robot.operateMode(ec)
    print_log("operateMode", ec, f"operateMode={operate_mode}")
    return operate_mode


def set_operatemode(robot, ec, mode=None):
    """设置操作模式，默认切到 automatic"""
    print_separator("set_operatemode", length=80)
    if mode is None:
        mode = xCoreSDK_python.OperateMode.automatic
    robot.setOperateMode(mode, ec)
    print_log("setOperateMode", ec)


def set_motioncontrolmode(robot, ec):
    """切换到传统非实时指令模式。"""
    print_separator("set_motioncontrolmode", length=80)
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    print_log("setMotionControlMode", ec)


def prepare_op(robot, ec):
    """上 OP 流程：传统模式 -> 自动模式 -> 上电。"""
    print_separator("prepare_op", length=110)
    get_powerstate(robot, ec)
    set_motioncontrolmode(robot, ec)
    get_operatemode(robot, ec)
    set_operatemode(robot, ec, xCoreSDK_python.OperateMode.automatic)
    set_powerstate(robot, ec, True)
    power_state = get_powerstate(robot, ec)
    if power_state != xCoreSDK_python.PowerState.on:
        raise RuntimeError(f"机器人上电失败，当前状态: {power_state}")
    set_baseframe(robot, ec)
    get_baseframe(robot, ec)


def get_baseframe(robot, ec):
    """查询基坐标系"""
    print_separator("get_baseframe", length=80)
    baseframe = robot.baseFrame(ec)
    print_log("baseFrame", ec, ",".join(map(str, baseframe)))


def set_baseframe(robot, ec):
    """将基坐标系设置为零位姿。"""
    print_separator("set_baseframe", length=80)
    frame = xCoreSDK_python.Frame([0.0, 0.0, 0.105, 0.0, 0.0, -0.141372])
    robot.setBaseFrame(frame, ec)
    print_log("setBaseFrame", ec)


def main():
    try:
        ip = "192.168.1.160"
        robot = xCoreSDK_python.ArRobot(ip)
        ec = {}
        prepare_op(robot, ec)
    except Exception as exc:
        print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
