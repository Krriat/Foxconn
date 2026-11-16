# -*- coding: utf-8 -*-
"""右臂 ServoJ：在两个7轴关节位置之间平滑往返10秒。"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from datetime import timedelta

import setup_path

if platform.system() == "Windows":
    from Release.windows import xCoreSDK_python
    from Release.windows.xCoreSDK_python import RtControllerMode, RtSupportedFields
elif platform.system() == "Linux":
    arch = platform.machine()
    if arch == "aarch64":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(root_dir, "Release", "linux", "arm"))
        import xCoreSDK_python
        from xCoreSDK_python import RtControllerMode, RtSupportedFields
    elif arch == "x86_64":
        from Release.linux import xCoreSDK_python
        from Release.linux.xCoreSDK_python import RtControllerMode, RtSupportedFields
    else:
        raise ImportError(f"不支持的Linux架构: {arch}")
else:
    raise ImportError("Unsupported operating system")


ROBOT_IP = "192.168.2.160"
LOCAL_IP = "192.168.2.10"

PLAN_PERIOD = 0.02
RUN_TIME = 10.0
ONE_WAY_TIME = 2.0
MOVEJ_SPEED = 0.3

PREPARE_POSITION = [-0.95, -1.0, 0.0, -0.7576, -1.9697, 0.3788, 0.0]
POSITION_A = [-0.95, -1.0, 0.0, -0.7576, -1.9697, 0.4788, 0.0]
POSITION_B = [-0.95, -1.0, 0.0, -0.7576, -1.9697, 0.5788, 0.0]


def _check_ec(ec: dict, action: str):
    message = ec.get("message", "")
    print(f"{action}: {message}")
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{action} failed: {message}")


def _refresh_state(robot: xCoreSDK_python.ArRobot, ec: dict) -> list[float]:
    while robot.updateRobotState(timedelta(milliseconds=0)):
        pass
    joints = list(robot.jointPos(ec))[:7]
    _check_ec(ec, "jointPos")
    return joints


def _interpolate_position(elapsed: float) -> list[float]:
    blend = 0.5 * (1.0 - math.cos(math.pi * elapsed / ONE_WAY_TIME))
    return [
        start + (end - start) * blend
        for start, end in zip(POSITION_A, POSITION_B)
    ]


def servoj(robot: xCoreSDK_python.ArRobot):
    ec = {}
    rt_controller = None
    receiving_state = False
    servo_enabled = False
    servo_motion_started = False

    try:
        robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
        _check_ec(ec, "setOperateMode")
        robot.setRtNetworkTolerance(20, ec)
        _check_ec(ec, "setRtNetworkTolerance")
        robot.setMotionControlMode(
            xCoreSDK_python.MotionControlMode.RtCommandMode, ec
        )
        _check_ec(ec, "setMotionControlMode")
        robot.setPowerState(True, ec)
        _check_ec(ec, "setPowerState")

        rt_controller = robot.getRtMotionController()
        robot.startReceiveRobotState(
            timedelta(milliseconds=1), [RtSupportedFields.jointPos_m]
        )
        receiving_state = True

        current_position = _refresh_state(robot, ec)
        print(f"运动到准备位: {PREPARE_POSITION}")
        rt_controller.MoveJ(MOVEJ_SPEED, current_position, PREPARE_POSITION)
        rt_controller.stopMove()

        current_position = _refresh_state(robot, ec)
        print(f"运动到往返起点: {POSITION_A}")
        rt_controller.MoveJ(MOVEJ_SPEED, current_position, POSITION_A)
        rt_controller.stopMove()

        rt_controller.setServoJoint(PLAN_PERIOD, PLAN_PERIOD, 0, ec)
        _check_ec(ec, "setServoJoint")
        servo_enabled = True
        rt_controller.startMove(RtControllerMode.jointPosition)
        servo_motion_started = True

        print(
            f"开始在 POSITION_A 和 POSITION_B 之间平滑往返，持续 {RUN_TIME:.1f}s"
        )
        start_time = time.perf_counter()
        next_tick = start_time

        while True:
            now = time.perf_counter()
            elapsed = now - start_time
            if elapsed >= RUN_TIME:
                break

            robot.updateRobotState(timedelta(milliseconds=1))
            if rt_controller.hasMotionError():
                raise RuntimeError("ServoJ运动错误，请检查控制器日志")

            command = xCoreSDK_python.JointPosition()
            command.joints = _interpolate_position(elapsed)
            rt_controller.sendCommand(command)

            next_tick += PLAN_PERIOD
            while time.perf_counter() < next_tick:
                pass

        final_command = xCoreSDK_python.JointPosition()
        final_command.joints = _interpolate_position(RUN_TIME)
        final_command.setFinished()
        rt_controller.sendCommand(final_command)

        rt_controller.stopMove()
        servo_motion_started = False
        rt_controller.stopServoJoint()
        servo_enabled = False
        print("ServoJ往返运动结束")
    finally:
        if rt_controller is not None and servo_motion_started:
            try:
                rt_controller.stopMove()
            except Exception:
                pass
        if rt_controller is not None and servo_enabled:
            try:
                rt_controller.stopServoJoint()
            except Exception:
                pass
        if receiving_state:
            try:
                robot.stopReceiveRobotState()
            except Exception:
                pass
        try:
            robot.setMotionControlMode(
                xCoreSDK_python.MotionControlMode.NrtCommandMode, ec
            )
            robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        right_robot = xCoreSDK_python.ArRobot(ROBOT_IP, LOCAL_IP)
        servoj(right_robot)
    except KeyboardInterrupt:
        print("收到终止信号，ServoJ已停止")
    except Exception as e:
        print(f"An error occurred: {e}")
