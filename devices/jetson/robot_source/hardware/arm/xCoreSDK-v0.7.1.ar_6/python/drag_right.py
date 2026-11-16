# -*- coding: utf-8 -*-
"""右臂自由拖拽：输入1开始，输入2停止并退出。"""

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


ROBOT_IP = "192.168.2.160"
LOCAL_IP = "192.168.2.10"


def _check_ec(ec: dict, action: str):
    message = ec.get("message", "")
    print(f"{action}: {message}")
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{action} failed: {message}")


def drag_control(robot: xCoreSDK_python.ArRobot):
    ec = {}
    drag_enabled = False

    robot.setPowerState(False, ec)
    _check_ec(ec, "setPowerState(False)")
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _check_ec(ec, "setOperateMode(manual)")
    robot.moveReset(ec)
    _check_ec(ec, "moveReset")

    try:
        while True:
            command = input("输入1开始拖拽，输入2停止拖拽并退出: ").strip()

            if command == "1":
                if drag_enabled:
                    print("拖拽模式已经开启")
                    continue

                robot.enableDrag(1, 2, ec, True)
                _check_ec(ec, "enableDrag")
                drag_enabled = True
                print("拖拽模式已开启，可以直接拖动右臂")
            elif command == "2":
                robot.disableDrag(ec)
                _check_ec(ec, "disableDrag")
                drag_enabled = False
                print("拖拽模式已关闭")
                break
            else:
                print("无效输入，请输入1或2")
    finally:
        if drag_enabled:
            try:
                robot.disableDrag(ec)
                print(f"disableDrag: {ec.get('message', '')}")
            except Exception:
                pass


if __name__ == "__main__":
    try:
        right_robot = xCoreSDK_python.ArRobot(ROBOT_IP, LOCAL_IP)
        drag_control(right_robot)
    except KeyboardInterrupt:
        print("\n收到终止信号，拖拽模式已关闭")
    except Exception as e:
        print(f"An error occurred: {e}")
