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

def print_cart_pose(cart_pose: xCoreSDK_python.CartesianPosition):
    [
        print(f"elbow_rad,{cart_pose.elbow}"),  # 臂角，单位弧度
        print(f"hasElbow,{cart_pose.hasElbow}"),  # 是否有臂角
        print(f"confData,{','.join(map(str,cart_pose.confData))}"),  # conf数据
        print(f"external,{','.join(map(str,cart_pose.external))}"),  # 外部轴数据
        print(f"trans_m,{','.join(map(str,cart_pose.trans))}"),  # xyz，单位米
        print(f"rpy_rad,{','.join(map(str,cart_pose.rpy))}"),  # abc，单位弧度
    ]

# 获取关节位置，前7位是机械臂关节位置，后面是附加轴
def get_jointpos(robot: xCoreSDK_python.ArRobot, ec: dict):
    joint_pos = robot.jointPos(ec)
    print_log("jointPos", ec)
    print(f"joint_pos_rad: {list(joint_pos)}")


# 获取笛卡尔位置
def get_cartpos(robot: xCoreSDK_python.ArRobot, ec: dict):
    pos = robot.posture(
        xCoreSDK_python.CoordinateType.endInRef, ec
    )  # 获取工具基于参考坐标系的笛卡尔位置
    print_log("posture", ec)
    print(f"pos trans_m: {list(pos[:3])}")
    print(f"pos rpy_rad: {list(pos[3:])}")

    # 获取笛卡尔位置，数据更多
    cart_posture = robot.cartPosture(xCoreSDK_python.CoordinateType.endInRef, ec)
    print_log("cartPosture", ec)
    print_cart_pose(cart_posture)


# 正解，关节角度->笛卡尔坐标
def calcFk(robot: xCoreSDK_python.ArRobot, ec: dict):
    start_angle = robot.jointPos(ec)[:7]
    robot_model: xCoreSDK_python.model.Model_1_7 = robot.model()
    cart_pose = robot_model.calcFk(start_angle, ec)  # 默认工具工件
    print_log("calcFk", ec)
    print_cart_pose(cart_pose)

    # 获取具体工具工件下的正解笛卡尔坐标
    toolset: xCoreSDK_python.Toolset = xCoreSDK_python.Toolset()
    toolset.end.trans = [0.01, 0.01, 0.01]
    toolset.end.rpy = [0.01, 0.01, 0.01]
    toolset.ref.trans = [0, 0, 0]
    toolset.ref.rpy = [0, 0, 0]
    cart_pose = robot_model.calcFk(start_angle, toolset, ec)
    print_log("calcFk", ec)
    print_cart_pose(cart_pose)


# 计算逆解，笛卡尔坐标 -> 关节角度
def calcIk(robot: xCoreSDK_python.ArRobot, ec: dict):
    # cart_pos = robot.cartPosture(xCoreSDK_python.CoordinateType.endInRef, ec)
    cart_pos = xCoreSDK_python.CartesianPosition(
        [0.53466, -0.451432, 1.69627e-16, 1.5708, 2.06757e-16, 1.5708]
    )
    cart_pos.elbow = 16.6993 * M_PI / 180;
    robot_model: xCoreSDK_python.model.Model_1_7 = robot.model()

    # 默认工具工件下的逆解
    joint_pos = robot_model.calcIk(cart_pos, ec)
    print_log("calcIk", ec, ",".join(map(str, map(rad2deg, joint_pos))))

    # 获取具体工具工件下的逆解
    toolset: xCoreSDK_python.Toolset = xCoreSDK_python.Toolset()
    toolset.end.trans = [0.01, 0.01, 0.01]
    toolset.end.rpy = [0.01, 0.01, 0.01]
    toolset.ref.trans = [0, 0, 0]
    toolset.ref.rpy = [0, 0, 0]
    cart_pos = xCoreSDK_python.CartesianPosition(
        [
            0.54466, -0.441432, 0.01, 1.5807, -0.0099995, 1.5808
        ]
    )
    joint_pos = robot_model.calcIk(cart_pos, toolset, ec)
    print_log("calcIk", ec, ",".join(map(str, map(rad2deg, joint_pos))))


def movej(robot: xCoreSDK_python.ArRobot, ec: dict):
    cart_pos1 = xCoreSDK_python.CartesianPosition(
        [0.53466, -0.451432, 1.69627e-16, 1.5708, 2.06757e-16, 1.5708]
    )
    cart_pos1.elbow = 16.6993 * M_PI / 180;
    cart_pos2 = xCoreSDK_python.CartesianPosition(
        [0.53466, -0.351432, 1.69627e-16, 1.5708, 2.06757e-16, 1.5708]
    )
    cart_pos2.elbow = 16.6993 * M_PI / 180;
    movejcmd = xCoreSDK_python.MoveJCommand(cart_pos1, 100, 10)
    movejcmd2 = xCoreSDK_python.MoveJCommand(cart_pos2, 100, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend(
        [movejcmd, movejcmd2], cmdID, ec
    )  # [movejcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def movel(robot: xCoreSDK_python.ArRobot, ec: dict):
    cart_pos1 = xCoreSDK_python.CartesianPosition(
        [-0.6096459322456094, -0.37618589954142356, 0.29888523343103146, -0.12275254217712268, 1.2915458559945348, -2.954794322448758]
    )
    cart_pos1.elbow = 16.6993 * M_PI / 180;
    # cart_pos2 = xCoreSDK_python.CartesianPosition(
    #     [0.53466, -0.351432, 1.69627e-16, 1.5708, 2.06757e-16, 1.5708]
    # )
    # cart_pos2.elbow = 16.6993 * M_PI / 180;
    movejcmd = xCoreSDK_python.MoveLCommand(cart_pos1, 500, 10)
    # movejcmd2 = xCoreSDK_python.MoveLCommand(cart_pos2, 100, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend(
        [movejcmd], cmdID, ec
    )  # [movejcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


def moveabsj(robot: xCoreSDK_python.ArRobot, ec: dict):
    joint_pos1 = [
        0.991509055169305,-1.4863856384110994,1.2833995134410652,0.7432906304544855,
        1.7082978106155013,-0.022611356182430566,-0.05560344797788099
    ]
    # joint_pos2 = [
    #     0.0,0.5235987755982988,0.0,1.0471975511965976,
    #     0.0,0,0.0,
    # ]
    absjcmd1 = xCoreSDK_python.MoveAbsJCommand(joint_pos1, 3000, 10)
    # absjcmd2 = xCoreSDK_python.MoveAbsJCommand(joint_pos2, 100, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend(
        [absjcmd1], cmdID, ec
    )  # [absjcmd]指令列表，可以添加多条指令，须为同类型指令
    print("Command ID:", cmdID.content())
    print_log("moveAppend", ec)
    robot.moveStart(ec)
    print_log("moveStart", ec)
    wait_robot(robot, ec)


# 实时模式-笛卡尔方式控制
def rt_cart_controller(robot: xCoreSDK_python.ArRobot, ec: dict):
    # 先用非实时movej到第一个点位
    start_pos = [0.53466, -0.351432, 0, M_PI / 2, 0, M_PI / 2]
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    robot.setPowerState(True, ec)
    start_tcp_point = xCoreSDK_python.CartesianPosition(start_pos)
    move_j_cmd = xCoreSDK_python.MoveJCommand(start_tcp_point, 100, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([move_j_cmd], cmdID, ec)
    robot.moveStart(ec)
    wait_robot(robot, ec)
    # 实时模式-笛卡尔方式控制
    try:
        robot.setRtNetworkTolerance(100, ec)
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.RtCommandMode, ec)
        rtCon: xCoreSDK_python.motioncontrolRT.PyRTmotioncontrol7 = robot.getRtMotionController()
        rtCon.setFilterLimit(True, 10)
        rtCon.setFilterFrequency(10, 10, 10, ec);
        def create_callback():  # 回调函数,务必创建两层，否则只能返回callback函数的地址
            start_pos_rt = [0.53466, 0, 0.351432, -M_PI, M_PI / 2, M_PI]
            cmd: xCoreSDK_python.CartesianPosition = xCoreSDK_python.CartesianPosition()
            cmd.pos = utility.postureToTransArray(start_pos_rt)
            cmd.hasElbow = True
            time = 0.0
            def callback():
                nonlocal time, cmd  # 允许修改外部变量
                time += 0.001
                list_pos = list(cmd.pos)
                list_pos[11] -= 0.0001  # z方向偏移
                cmd.pos = list_pos
                if time > 2:
                    cmd.setFinished()
                return cmd  # 返回joints和external两个参数值给回调函数callback
            return callback  # 返回闭包函数,保证回调函数传入的参数类型是函数而不是其它变量参数或结构体
        # 创建回调
        callback = create_callback()  # 返回值类型为函数
        # 设置回调函数
        rtCon.setControlLoopCar(callback)
        rtCon.startMove(RtControllerMode.cartesianPosition)
        rtCon.startLoop(False)  # False 表示非阻塞，仅单步运行.非阻塞模式下，函数只执行单次控制周期.这种方式能够插入调试代码（如 time.sleep 和打印语句）,确保控制周期被多次执行
        time.sleep(5)  # 模拟控制周期多次执行
        print("控制结束")
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)  # 关闭实时模式并下电
        robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
        robot.setPowerState(False, ec)
    except Exception as e:
        print(f"An error occurred: {e}")
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.Idle, ec)
        robot.setPowerState(False, ec)


# 实时模式-轴角度方式控制
def rt_joint_controller(robot: xCoreSDK_python.ArRobot, ec: dict):
    # 先用非实时moveabsj到第一个点位
    jntPos = [0.0, 0.5235987755982988, 0.0, 1.0471975511965976, 0.0, 0.0, 0.0]
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    robot.setPowerState(True, ec)
    moveabsj_cmd = xCoreSDK_python.MoveAbsJCommand(jntPos, 100, 10)
    cmdID = xCoreSDK_python.PyString()
    robot.moveAppend([moveabsj_cmd], cmdID, ec)
    robot.moveStart(ec)
    wait_robot(robot, ec)

    # 实时模式-轴角度方式控制
    try:
        robot.setRtNetworkTolerance(100, ec)
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.RtCommandMode, ec)
        rtCon: xCoreSDK_python.motioncontrolRT.PyRTmotioncontrol7 = robot.getRtMotionController()
        rtCon.setFilterLimit(True, 10)
        def create_callback():  # 回调函数,务必创建两层，否则只能返回callback函数的地址
            # 需要修改的状态变量
            time = 0.0
            def callback():
                nonlocal time  # 允许修改外部变量
                time += 0.001
                delta_angle = M_PI / 20.0 * (1 - math.cos(M_PI / 2.5 * time)) / 5
                cmd = xCoreSDK_python.JointPosition()
                cmd.joints = [
                    jntPos[0] + delta_angle,
                    jntPos[1] + delta_angle,
                    jntPos[2] - delta_angle,
                    jntPos[3] + delta_angle,
                    jntPos[4] - delta_angle,
                    jntPos[5] + delta_angle,
                    jntPos[6] - delta_angle,
                ]
                if time > 2:
                    cmd.setFinished()
                return cmd

            return callback

        # 创建回调
        callback = create_callback()  # 返回值类型为函数
        # 设置回调函数
        rtCon.setControlLoopJoi(callback)
        rtCon.startMove(RtControllerMode.jointPosition)
        rtCon.startLoop(False)  # False 表示非阻塞，仅单步运行.非阻塞模式下，函数只执行单次控制周期.这种方式能够插入调试代码（如 time.sleep 和打印语句）,确保控制周期被多次执行
        time.sleep(5)  # 模拟控制周期多次执行
        print("控制结束")
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)  # 关闭实时模式并下电
        robot.setPowerState(False, ec)
    except Exception as e:
        print(f"An error occurred: {e}")
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.Idle, ec)
        robot.setPowerState(False, ec)

# servoj
def servoj(robot: xCoreSDK_python.ArRobot, ec: dict):
    # 用户指令下发周期(s)
    PLAN_PERIOD = 0.02


    def busy_wait(milliseconds: int):
        """延迟发送，与指令周期对齐"""
        end = time.perf_counter() + milliseconds / 1000.0
        while time.perf_counter() < end:
            pass


    def servoj_op(robot: xCoreSDK_python.xMateRobot, ec: dict):
        """
        实时模式 servoJ 示例：周期性 sendCommand 下发关节位置。
        对应 C++ 示例 servoj_demo.cpp。
        """
        print_separator("servoj_op", length=110)
        robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
        print_log("setOperateMode", ec)
        # 必做：启用实时模式
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.RtCommandMode, ec)
        print_log("setMotionControlMode", ec)
        robot.setPowerState(True, ec)
        print_log("setPowerState", ec)

        rtCon: xCoreSDK_python.motioncontrolRT.PyRTmotioncontrol7 = (
            robot.getRtMotionController()
        )
        # 设置要接收的数据
        robot.startReceiveRobotState(
            timedelta(milliseconds=1), [RtSupportedFields.jointPos_m]
        )

        q_drag = [0, M_PI / 6, -M_PI / 2, 0, -M_PI / 3, 0,0]
        q2_drag = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0]

        while robot.updateRobotState(timedelta(milliseconds=0)):
            pass

        jntPos = list(robot.jointPos(ec))[:7]
        rtCon.MoveJ(0.3, jntPos, q_drag)
        jntPos = list(robot.jointPos(ec))[:7]

        # 必做：启用 servoj 功能
        # 周期下发不稳定时，可使用前瞻时间和增益调节，默认前瞻时间为一个指令周期，增益为0
        rtCon.setServoJoint(PLAN_PERIOD, PLAN_PERIOD, 0, ec)
        print_log("setServoJoint", ec)
        jntPos = list(robot.jointPos(ec))[:7]
        # 必做：设置运动模式，注意在启用 servoj 功能后
        rtCon.startMove(RtControllerMode.jointPosition)

        start = time.perf_counter()
        while True:
            robot.updateRobotState(timedelta(milliseconds=1))
            elapsed_seconds = time.perf_counter() - start
            delta_angle = M_PI / 30.0 * (1 - math.cos(M_PI / 2.5 * elapsed_seconds))
            cmd = xCoreSDK_python.JointPosition()
            cmd.joints = [
                jntPos[0] + delta_angle,
                jntPos[1] + delta_angle,
                jntPos[2] - delta_angle,
                jntPos[3] + delta_angle,
                jntPos[4] - delta_angle,
                jntPos[5] + delta_angle,
            ]
            # 必做：发送计算出的关节位置
            rtCon.sendCommand(cmd)
            # 必做：间隔指令周期发送
            busy_wait(int(PLAN_PERIOD * 1000))

            if elapsed_seconds > 60:
                cmd.setFinished()
                rtCon.sendCommand(cmd)
                # 必做：关闭 servoj 功能
                rtCon.stopServoJoint()
                break

        while robot.updateRobotState(timedelta(milliseconds=0)):
            pass
        rtCon.MoveJ(0.3, list(robot.jointPos(ec))[:7], q2_drag)
        print("控制结束")

        robot.setMotionControlMode(
            xCoreSDK_python.MotionControlMode.NrtCommandMode, ec
        )
        print_log("setMotionControlMode", ec)
        robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
        print_log("setOperateMode", ec)
        
    servoj_op(robot, ec)
    
# servol
def servol(robot: xCoreSDK_python.ArRobot, ec: dict):
    # 用户指令下发周期(s)
    PLAN_PERIOD = 0.02

    # 矩形轨迹参数（与 C++ servol_demo.cpp 一致）
    RECT_X = 0.05
    RECT_Y = 0.05
    EDGE_TIME = 4.0
    

    def busy_wait(milliseconds: int):
        """延迟发送，与指令周期对齐"""
        end = time.perf_counter() + milliseconds / 1000.0
        while time.perf_counter() < end:
            pass


    def _rect_offsets(t: float) -> tuple[float, float]:
        """按当前时间 t 计算矩形轨迹在 X/Y 方向的偏移"""
        if t < EDGE_TIME:
            x_offset = (t / EDGE_TIME) * RECT_X
            y_offset = 0.0
        elif t < EDGE_TIME * 2:
            x_offset = RECT_X
            y_offset = ((t - EDGE_TIME) / EDGE_TIME) * RECT_Y
        elif t < EDGE_TIME * 3:
            x_offset = RECT_X * (1.0 - (t - EDGE_TIME * 2) / EDGE_TIME)
            y_offset = RECT_Y
        else:
            x_offset = 0.0
            y_offset = RECT_Y * (1.0 - (t - EDGE_TIME * 3) / EDGE_TIME)
        return x_offset, y_offset


    def _check_ec(ec: dict, step: str):
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"{step} failed: {ec.get('message', ec)}")


    def _fill_cart_cmd(
        cmd: xCoreSDK_python.CartesianPosition,
        pos: list[float],
        ref: xCoreSDK_python.CartesianPosition,
    ):
        """下发笛卡尔指令时保持 confData 等与当前位姿一致，避免逆解失败"""
        cmd.pos = list(pos)
        cmd.confData = list(ref.confData)
        cmd.hasElbow = ref.hasElbow
        cmd.elbow = ref.elbow
        if ref.external:
            cmd.external = list(ref.external)


    def build_rect_trajectory(
        rot_initial: np.ndarray, trans_initial: np.ndarray
    ) -> list[list[float]]:
        """在 RT 循环外预计算矩形轨迹（行优先 4x4 齐次变换矩阵）"""
        total_time = EDGE_TIME * 4
        total_frames = int(math.ceil(total_time / PLAN_PERIOD))
        trajectory = []
        trans_initial = np.asarray(trans_initial, dtype=np.float64).reshape(3)

        for fi in range(total_frames):
            t = fi * PLAN_PERIOD
            x_offset, y_offset = _rect_offsets(t)
            trans_target = trans_initial.copy()
            trans_target[0] += x_offset
            trans_target[1] += y_offset
            trajectory.append(utility.transMatrixToArray(rot_initial, trans_target))

        return trajectory


    def servol_op(robot: xCoreSDK_python.xMateRobot, ec: dict, run_loops: int = 0):
        """
        实时模式 ServoL 示例：周期性 sendCommand 下发笛卡尔位姿，沿矩形轨迹循环运动。
        对应 C++ 示例 servol_demo.cpp。

        Args:
            run_loops: 矩形循环次数，0 表示无限循环（Ctrl+C 结束）
        """
        print_separator("servol_op", length=110)
        dof = 7
        robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
        print_log("setOperateMode", ec)
        _check_ec(ec, "setOperateMode")
        # 须在切换实时模式前设置网络延迟阈值
        robot.setRtNetworkTolerance(20, ec)
        print_log("setRtNetworkTolerance", ec)
        _check_ec(ec, "setRtNetworkTolerance")
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.RtCommandMode, ec)
        print_log("setMotionControlMode", ec)
        _check_ec(ec, "setMotionControlMode")
        robot.setPowerState(True, ec)
        print_log("setPowerState", ec)
        _check_ec(ec, "setPowerState")

        rtCon: xCoreSDK_python.motioncontrolRT.PyRTmotioncontrol7 = (
            robot.getRtMotionController()
        )

        robot.startReceiveRobotState(
            timedelta(milliseconds=1),
            [RtSupportedFields.jointPos_m, RtSupportedFields.tcpPose_m],
        )

        q_drag = [0, M_PI / 6, 0, M_PI / 3, 0, 0,0]
        q2_drag = [0.0] * dof

        while robot.updateRobotState(timedelta(milliseconds=0)):
            pass

        print("MoveJ 运动至起始关节位...")
        rtCon.MoveJ(0.3, list(robot.jointPos(ec))[:dof], q_drag)
        # MoveJ 内部会 startMove(jointPosition)+startLoop，结束后需显式 stopMove
        rtCon.stopMove()
        time.sleep(0.2)
        while robot.updateRobotState(timedelta(milliseconds=0)):
            pass

        # 笛卡尔伺服前配置滤波（须在 startMove 之前，参考 ar_example）
        rtCon.setFilterLimit(True, 100)
        rtCon.setFilterFrequency(100, 100, 100, ec)
        print_log("setFilterFrequency", ec)
        _check_ec(ec, "setFilterFrequency")

        # 必做：启用 ServoL（接口为 setServoJoint，须在 startMove 之前）
        rtCon.setServoJoint(PLAN_PERIOD, PLAN_PERIOD, 0, ec)
        print_log("setServoJoint", ec)
        _check_ec(ec, "setServoJoint")

        cart_ref = robot.cartPosture(
            xCoreSDK_python.CoordinateType.flangeInBase, ec
        )
        print_log("cartPosture", ec)
        _check_ec(ec, "cartPosture")

        rtCon.startMove(RtControllerMode.cartesianPosition)

        rot_initial, trans_initial = utility.arrayToTransMatrix(list(cart_ref.pos))
        trans_initial = np.asarray(trans_initial, dtype=np.float64).reshape(3)
        print(
            f"初始位姿: X={trans_initial[0]:.4f}, "
            f"Y={trans_initial[1]:.4f}, Z={trans_initial[2]:.4f}"
        )

        trajectory = build_rect_trajectory(rot_initial, trans_initial)
        print(f"矩形轨迹预计算完成，共 {len(trajectory)} 帧")

        cmd = xCoreSDK_python.CartesianPosition()
        # 首帧与当前位姿对齐，避免首条指令跳变过大
        for _ in range(5):
            robot.updateRobotState(timedelta(milliseconds=1))
            _fill_cart_cmd(cmd, trajectory[0], cart_ref)
            rtCon.sendCommand(cmd)
            busy_wait(int(PLAN_PERIOD * 1000))

        idx = 0
        loop_count = 0
        print("矩形轨迹开始循环运行，按 Ctrl+C 终止...")
        try:
            while True:
                robot.updateRobotState(timedelta(milliseconds=1))
                if rtCon.hasMotionError():
                    raise RuntimeError("实时运动发生错误，请检查控制器日志")
                _fill_cart_cmd(cmd, trajectory[idx], cart_ref)
                rtCon.sendCommand(cmd)
                busy_wait(int(PLAN_PERIOD * 1000))
                idx += 1
                if idx >= len(trajectory):
                    idx = 0
                    loop_count += 1
                    print(f"完成第 {loop_count} 圈矩形轨迹")
                    if run_loops > 0 and loop_count >= run_loops:
                        break
        except KeyboardInterrupt:
            print("收到终止信号，正在结束 ServoL...")

        rtCon.stopMove()
        final_cmd = xCoreSDK_python.CartesianPosition()
        _fill_cart_cmd(
            final_cmd, utility.transMatrixToArray(rot_initial, trans_initial), cart_ref
        )
        final_cmd.setFinished()
        rtCon.sendCommand(final_cmd)
        rtCon.stopServoJoint()

        while robot.updateRobotState(timedelta(milliseconds=0)):
            pass
        rtCon.MoveJ(0.3, list(robot.jointPos(ec))[:dof], q2_drag)
        rtCon.stopMove()
        print("ServoL控制结束")

        robot.setMotionControlMode(
            xCoreSDK_python.MotionControlMode.NrtCommandMode, ec
        )
        print_log("setMotionControlMode", ec)
        robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
        print_log("setOperateMode", ec)

    servol_op(robot, ec, run_loops=10)

if __name__ == "__main__":
    try:
        # 连接机器人
        ip = "192.168.1.160"
        lcoal_ip = "192.168.1.10"
        # 实例化AR机型并连接
        robot = xCoreSDK_python.ArRobot(ip, lcoal_ip)
        ec = {}
        # get_jointpos(robot, ec)
        # get_cartpos(robot, ec)
        # calcFk(robot, ec)
        # calcIk(robot, ec)
        movej(robot, ec)
        # movel(robot, ec)
        # moveabsj(robot, ec)
        # rt_cart_controller(robot, ec)
        # rt_joint_controller(robot, ec)
        # servol(robot, ec)
        # servoj(robot, ec)
    except Exception as e:
        print(f"An error occurred: {e}")
