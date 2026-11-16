# -*- coding: utf-8 -*-
"""左臂：到达准备关节位后，沿基坐标系+Y方向直线运动100 mm。"""

from __future__ import annotations

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


ROBOT_IP = "192.168.1.160"
LOCAL_IP = "192.168.1.10"

PLAN_PERIOD = 0.02
LINE_TIME = 5.0
LINE_DISTANCE = 0.100
PREPARE_SPEED = 500
PREPARE_ZONE = 0
PREPARE_TIMEOUT = 120.0
JOINT_TOLERANCE = 0.02
LIMIT_EPSILON = 1e-6

PREPARE_POSITION = [
    1.18857,
    -1.570796327,
    1.570796327,
    0.2617993878,
    0.0,
    0.7853981634,
    0.0,
]


def _check_ec(ec: dict, action: str):
    message = ec.get("message", "")
    print(f"{action}: {message}")
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{action} failed: {message}")


def _get_joints(robot: xCoreSDK_python.ArRobot, ec: dict) -> list[float]:
    joints = list(robot.jointPos(ec))[:7]
    _check_ec(ec, "jointPos")
    return joints


def _check_joint_limits(
    robot: xCoreSDK_python.ArRobot,
    current_joints: list[float],
    target_joints: list[float],
    ec: dict,
):
    limits_out = xCoreSDK_python.PyTypeVectorArrayDouble2()
    enabled = robot.getSoftLimit(limits_out, ec)
    _check_ec(ec, "getSoftLimit")
    if not enabled:
        print("警告：机器人软限位当前未开启")
        return

    limits = limits_out.content()[:7]
    for index, (current, target, limits_for_axis) in enumerate(
        zip(current_joints, target_joints, limits), start=1
    ):
        lower, upper = limits_for_axis
        if not lower - LIMIT_EPSILON <= current <= upper + LIMIT_EPSILON:
            raise RuntimeError(
                f"第{index}轴实际角度超限: {current}, 允许范围=[{lower}, {upper}]"
            )
        if not lower - LIMIT_EPSILON <= target <= upper + LIMIT_EPSILON:
            raise RuntimeError(
                f"第{index}轴准备位超限: {target}, 允许范围=[{lower}, {upper}]"
            )


def _set_traditional_mode(robot: xCoreSDK_python.ArRobot, ec: dict):
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _check_ec(ec, "setMotionControlMode(NrtCommandMode)")
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    _check_ec(ec, "setOperateMode(automatic)")
    robot.setPowerState(True, ec)
    _check_ec(ec, "setPowerState(True)")
    power_state = robot.powerState(ec)
    _check_ec(ec, "powerState")
    if power_state != xCoreSDK_python.PowerState.on:
        raise RuntimeError(f"左臂上电失败: {power_state}")


def _move_to_prepare(robot: xCoreSDK_python.ArRobot, ec: dict):
    current_joints = _get_joints(robot, ec)
    _check_joint_limits(robot, current_joints, PREPARE_POSITION, ec)

    robot.moveReset(ec)
    _check_ec(ec, "moveReset")
    command = xCoreSDK_python.MoveAbsJCommand(
        PREPARE_POSITION, PREPARE_SPEED, PREPARE_ZONE
    )
    command_id = xCoreSDK_python.PyString()
    robot.moveAppend([command], command_id, ec)
    _check_ec(ec, "moveAppend")
    print(f"准备位 Command ID: {command_id.content()}")
    robot.moveStart(ec)
    _check_ec(ec, "moveStart")

    time.sleep(0.2)
    deadline = time.monotonic() + PREPARE_TIMEOUT
    while time.monotonic() < deadline:
        state = robot.operationState(ec)
        _check_ec(ec, "operationState")
        if state in (
            xCoreSDK_python.OperationState.idle,
            xCoreSDK_python.OperationState.unknown,
        ):
            break
        time.sleep(0.1)
    else:
        robot.stop(ec)
        raise TimeoutError("运动到准备位超时")

    actual_joints = _get_joints(robot, ec)
    max_error = max(
        abs(actual - target)
        for actual, target in zip(actual_joints, PREPARE_POSITION)
    )
    if max_error > JOINT_TOLERANCE:
        event_info = robot.queryEventInfo(xCoreSDK_python.Event.moveExecution, ec)
        _check_ec(ec, "queryEventInfo")
        raise RuntimeError(
            f"准备位实际未到达，最大关节误差={max_error:.6f} rad，"
            f"运动事件={event_info}"
        )

    print(f"准备位已到达，最大关节误差={max_error:.6f} rad")


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


def _calculate_line_target(
    reference: xCoreSDK_python.CartesianPosition,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rotation, initial_translation = utility.arrayToTransMatrix(list(reference.pos))
    initial_translation = np.asarray(initial_translation, dtype=np.float64).reshape(3)
    target_translation = initial_translation.copy()
    target_translation[1] += LINE_DISTANCE
    target_pos = utility.transMatrixToArray(rotation, target_translation)
    return rotation, initial_translation, target_pos


def _check_line_path(
    robot: xCoreSDK_python.ArRobot,
    reference: xCoreSDK_python.CartesianPosition,
    current_joints: list[float],
    target_pos: list[float],
    ec: dict,
):
    target = xCoreSDK_python.CartesianPosition()
    _fill_cart_command(target, target_pos, reference)
    target_joints = robot.checkPath(reference, current_joints, target, ec)
    _check_ec(ec, "checkPath")
    if len(target_joints) < 7:
        raise RuntimeError(f"+Y直线路径不可达，checkPath返回: {target_joints}")
    print(f"checkPath通过，预计终点关节角: {list(target_joints)[:7]}")


def line(robot: xCoreSDK_python.ArRobot):
    ec = {}
    rt_controller = None
    receiving_state = False
    servo_enabled = False
    servo_motion_started = False
    completed = False

    try:
        _set_traditional_mode(robot, ec)
        _move_to_prepare(robot, ec)

        current_joints = _get_joints(robot, ec)
        cart_reference = robot.cartPosture(
            xCoreSDK_python.CoordinateType.flangeInBase, ec
        )
        _check_ec(ec, "cartPosture")
        _, _, checked_target_pos = _calculate_line_target(cart_reference)
        _check_line_path(
            robot, cart_reference, current_joints, checked_target_pos, ec
        )

        robot.setRtNetworkTolerance(20, ec)
        _check_ec(ec, "setRtNetworkTolerance")
        robot.setMotionControlMode(
            xCoreSDK_python.MotionControlMode.RtCommandMode, ec
        )
        _check_ec(ec, "setMotionControlMode(RtCommandMode)")
        robot.setPowerState(True, ec)
        _check_ec(ec, "setPowerState(True)")

        rt_controller = robot.getRtMotionController()
        robot.startReceiveRobotState(
            timedelta(milliseconds=1),
            [RtSupportedFields.jointPos_m, RtSupportedFields.tcpPose_m],
        )
        receiving_state = True
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
        rotation, initial_translation, final_pos = _calculate_line_target(
            cart_reference
        )
        initial_pos = utility.transMatrixToArray(rotation, initial_translation)

        print(
            f"开始直线运动: Y={initial_translation[1]:.4f} m -> "
            f"Y={initial_translation[1] + LINE_DISTANCE:.4f} m，"
            f"距离={LINE_DISTANCE * 1000:.1f} mm，时间={LINE_TIME:.1f} s"
        )

        rt_controller.startMove(RtControllerMode.cartesianPosition)
        servo_motion_started = True
        command = xCoreSDK_python.CartesianPosition()

        for _ in range(5):
            robot.updateRobotState(timedelta(milliseconds=1))
            _fill_cart_command(command, initial_pos, cart_reference)
            rt_controller.sendCommand(command)
            end = time.perf_counter() + PLAN_PERIOD
            while time.perf_counter() < end:
                pass

        frame_count = int(round(LINE_TIME / PLAN_PERIOD))
        start_time = time.perf_counter()
        next_tick = start_time
        for index in range(1, frame_count + 1):
            robot.updateRobotState(timedelta(milliseconds=1))
            if rt_controller.hasMotionError():
                raise RuntimeError("ServoL直线运动错误，请检查控制器日志")

            progress = index / frame_count
            target_translation = initial_translation.copy()
            target_translation[1] += LINE_DISTANCE * progress
            target_pos = utility.transMatrixToArray(rotation, target_translation)
            _fill_cart_command(command, target_pos, cart_reference)
            rt_controller.sendCommand(command)

            next_tick += PLAN_PERIOD
            while time.perf_counter() < next_tick:
                pass

        final_command = xCoreSDK_python.CartesianPosition()
        _fill_cart_command(final_command, final_pos, cart_reference)
        final_command.setFinished()
        rt_controller.sendCommand(final_command)

        rt_controller.stopMove()
        servo_motion_started = False
        rt_controller.stopServoJoint()
        servo_enabled = False
        completed = True
        print("+Y方向100 mm直线运动完成")
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
            _set_traditional_mode(robot, ec)
            print("已恢复传统自动模式，左臂保持上电")
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
        left_robot = xCoreSDK_python.ArRobot(ROBOT_IP, LOCAL_IP)
        line(left_robot)
    except KeyboardInterrupt:
        print("收到终止信号，直线运动已停止并切回手动模式")
    except Exception as e:
        print(f"An error occurred: {e}")
