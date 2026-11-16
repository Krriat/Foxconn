# -*- coding: utf-8 -*-
"""Right-hand grasp translated from armtest_demo_0.7.1/demo_can_gripper.cpp."""

from __future__ import annotations

import os
import platform
import sys
import time

import setup_path
from log import print_log, print_separator

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
RIGHT_FRAME_ID = 0x28

# 来自厂家 armtest_demo_0.7.1/config/demo_can_gripper.yaml。
OPEN_FRAMES = [
    [0x01, 0xCD, 0x00, 0x00, 0x00, 0x00, 0x00],
]
GRASP_FRAMES = [
    [0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
]


def ec_failed(ec: dict) -> bool:
    return ec.get("ec", 0) != 0


def send_frames(robot, payloads: list[list[int]]) -> bool:
    """Translate demo_can_gripper.cpp sendFrames() to Python."""
    frames = []

    for payload in payloads:
        if not 1 <= len(payload) <= 8:
            raise ValueError("每个CAN帧必须包含1到8个字节")
        if any(not 0 <= value <= 255 for value in payload):
            raise ValueError("CAN帧数据必须在0到255之间")

        frame = xCoreSDK_python.CANFrame()
        frame.frame_id = RIGHT_FRAME_ID
        frame.frame_format = int(xCoreSDK_python.CANFormat.STANDARD)
        frame.frame_type = int(xCoreSDK_python.CANType.CAN)
        frame.frame_valid_length = len(payload)
        frame.data = payload
        frames.append(frame)

    send_ec = {}
    robot.CANSendData("uint8", frames, send_ec)
    print_log("CANSendData", send_ec)
    print("send:", [frame.data for frame in frames])
    if ec_failed(send_ec):
        return False

    recv_frame = xCoreSDK_python.CANFrame()
    recv_ec = {}
    robot.CANReceiveData(50, "uint8", recv_frame, recv_ec)
    print(f"CANReceiveData: {recv_ec} {recv_frame}")
    return not ec_failed(recv_ec)


def prepare_robot(robot) -> bool:
    """Match the connection preparation used by the vendor demo."""
    ec = {}
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    print_log("setOperateMode", ec)
    if ec_failed(ec):
        return False

    ec = {}
    robot.setPowerState(True, ec)
    print_log("setPowerState", ec)
    return not ec_failed(ec)


def grasp(robot) -> bool:
    """Open the hand, wait two seconds, then close it to grasp."""
    print_separator("right hand open", length=80)
    open_ok = send_frames(robot, OPEN_FRAMES)
    print(f"open result: {'ok' if open_ok else 'fail'}")

    time.sleep(2.0)

    print_separator("right hand grasp", length=80)
    grasp_ok = send_frames(robot, GRASP_FRAMES)
    print(f"grasp result: {'ok' if grasp_ok else 'fail'}")
    return open_ok and grasp_ok


def main() -> None:
    print(f"Connecting robot: remote={ROBOT_IP}, local={LOCAL_IP}")
    robot = xCoreSDK_python.ArRobot(ROBOT_IP, LOCAL_IP)

    if not prepare_robot(robot):
        raise RuntimeError("机械臂自动模式或上电失败")

    if not grasp(robot):
        raise RuntimeError("抓取CAN帧已发送，但没有收到CAN确认")

    print("右手抓取完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"An error occurred: {exc}")
