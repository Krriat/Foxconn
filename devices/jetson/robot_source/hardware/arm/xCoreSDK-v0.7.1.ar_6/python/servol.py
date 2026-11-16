# -*- coding: utf-8 -*-
"""右臂 ServoL：在基坐标系 Y-Z 平面画半径25 mm的圆。"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from datetime import timedelta

import numpy as np

import setup_path

if platform.system() == "Windows":
    from Release.windows import xCoreSDK_python
    from Release.windows.xCoreSDK_python import utility
    from Release.windows.xCoreSDK_python import RtControllerMode, RtSupportedFields
elif platform.system() == "Linux":
    arch = platform.machine()
    if arch == "aarch64":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(root_dir, "Release", "linux", "arm"))
        import xCoreSDK_python
        from xCoreSDK_python import utility
        from xCoreSDK_python import RtControllerMode, RtSupportedFields
    elif arch == "x86_64":
        from Release.linux import xCoreSDK_python
        from Release.linux.xCoreSDK_python import utility
        from Release.linux.xCoreSDK_python import RtControllerMode, RtSupportedFields
    else:
        raise ImportError(f"不支持的Linux架构: {arch}")
else:
    raise ImportError("Unsupported operating system")


ROBOT_IP = "192.168.2.160"
LOCAL_IP = "192.168.2.10"

PLAN_PERIOD = 0.02
RUN_TIME = 20.0
RADIUS = 0.025
MOVEJ_SPEED = 0.6
PREPARE_POSITION = [-0.95, -1.0, 0.0, -0.7576, -1.9697, 0.3788, 0.0]


def _check_ec(ec: dict, action: str):
    message = ec.get("message", "")
    print(f"{action}: {message}")
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{action} failed: {message}")


def _refresh_joints(robot: xCoreSDK_python.ArRobot, ec: dict) -> list[float]:
    while robot.updateRobotState(timedelta(milliseconds=0)):
        pass
    joints = list(robot.jointPos(ec))[:7]
    _check_ec(ec, "jointPos")
    return joints


def _fill_cart_command(
    command: xCoreSDK_python.CartesianPosition,
    pos: list[float],
    reference: xCoreSDK_python.CartesianPosition,
):
    command.pos = list(pos)
    command.confData = list(reference.confData)
    command.hasElbow = reference.hasElbow
    command.elbow = reference.elbow
    if reference.external:
        command.external = list(reference.external)


def _build_circle_trajectory(
    rotation: np.ndarray, initial_translation: np.ndarray
) -> list[list[float]]:
    """当前位置作为圆上的起点，避免ServoL第一帧产生位置跳变。"""
    frame_count = int(round(RUN_TIME / PLAN_PERIOD))
    trajectory = []
    initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(3)

    for index in range(frame_count):
        theta = 2.0 * math.pi * index / frame_count
        target_translation = initial_translation.copy()
        target_translation[1] += RADIUS * (math.cos(theta) - 1.0)
        target_translation[2] += RADIUS * math.sin(theta)
        trajectory.append(
            utility.transMatrixToArray(rotation, target_translation)
        )

    return trajectory


def _restore_traditional_mode(robot: xCoreSDK_python.ArRobot, ec: dict):
    """正常结束后恢复传统自动模式并重新上电。"""
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _check_ec(ec, "setMotionControlMode(NrtCommandMode)")
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    _check_ec(ec, "setOperateMode(automatic)")
    robot.setPowerState(True, ec)
    _check_ec(ec, "setPowerState(True)")
    power_state = robot.powerState(ec)
    _check_ec(ec, "powerState")
    if power_state != xCoreSDK_python.PowerState.on:
        raise RuntimeError(f"传统模式恢复后上电失败: {power_state}")


def servol(robot: xCoreSDK_python.ArRobot):
    ec = {}
    rt_controller = None
    receiving_state = False
    servo_enabled = False
    servo_motion_started = False
    completed = False

    try:
        robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
        _check_ec(ec, "setOperateMode")
        robot.setRtNetworkTolerance(20, ec)
        _check_ec(ec, "setRtNetworkTolerance")
        robot.setMotionControlMode(
            xCoreSDK_python.MotionControlMode.RtCommandMode, ec
        )
        _check_ec(ec, "setMotionControlMode(RtCommandMode)")
        robot.setPowerState(True, ec)
        _check_ec(ec, "setPowerState")

        rt_controller = robot.getRtMotionController()
        robot.startReceiveRobotState(
            timedelta(milliseconds=1),
            [RtSupportedFields.jointPos_m, RtSupportedFields.tcpPose_m],
        )
        receiving_state = True

        current_joints = _refresh_joints(robot, ec)
        print(f"运动到准备位: {PREPARE_POSITION}")
        rt_controller.MoveJ(MOVEJ_SPEED, current_joints, PREPARE_POSITION)
        rt_controller.stopMove()

        while robot.updateRobotState(timedelta(milliseconds=0)):
            pass

        rt_controller.setFilterLimit(True, 100)
        rt_controller.setFilterFrequency(100, 100, 100, ec)
        _check_ec(ec, "setFilterFrequency")
        rt_controller.setServoJoint(PLAN_PERIOD, PLAN_PERIOD, 0, ec)
        _check_ec(ec, "setServoJoint")
        servo_enabled = True

        cart_reference = robot.cartPosture(
            xCoreSDK_python.CoordinateType.flangeInBase, ec
        )
        _check_ec(ec, "cartPosture")
        rotation, initial_translation = utility.arrayToTransMatrix(
            list(cart_reference.pos)
        )
        initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(3)
        trajectory = _build_circle_trajectory(rotation, initial_translation)

        print(
            f"初始位置: X={initial_translation[0]:.4f}, "
            f"Y={initial_translation[1]:.4f}, Z={initial_translation[2]:.4f}"
        )
        print(f"开始画圆: 半径={RADIUS * 1000:.1f} mm，时间={RUN_TIME:.1f} s")

        rt_controller.startMove(RtControllerMode.cartesianPosition)
        servo_motion_started = True
        command = xCoreSDK_python.CartesianPosition()

        for _ in range(5):
            robot.updateRobotState(timedelta(milliseconds=1))
            _fill_cart_command(command, trajectory[0], cart_reference)
            rt_controller.sendCommand(command)
            end = time.perf_counter() + PLAN_PERIOD
            while time.perf_counter() < end:
                pass

        start_time = time.perf_counter()
        next_tick = start_time
        for pos in trajectory:
            robot.updateRobotState(timedelta(milliseconds=1))
            if rt_controller.hasMotionError():
                raise RuntimeError("ServoL运动错误，请检查控制器日志")

            _fill_cart_command(command, pos, cart_reference)
            rt_controller.sendCommand(command)

            next_tick += PLAN_PERIOD
            while time.perf_counter() < next_tick:
                pass

        final_command = xCoreSDK_python.CartesianPosition()
        _fill_cart_command(
            final_command,
            utility.transMatrixToArray(rotation, initial_translation),
            cart_reference,
        )
        final_command.setFinished()
        rt_controller.sendCommand(final_command)

        rt_controller.stopMove()
        servo_motion_started = False
        rt_controller.stopServoJoint()
        servo_enabled = False
        completed = True
        print("ServoL圆轨迹结束，末端已回到圆轨迹起点")
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

        if completed:
            _restore_traditional_mode(robot, ec)
            print("已恢复传统自动模式，机器人保持上电")
        else:
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
        servol(right_robot)
    except KeyboardInterrupt:
        print("收到终止信号，ServoL已停止并切回手动模式")
    except Exception as e:
        print(f"An error occurred: {e}")
