#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 2 MoveJByPath 服务客户端。

这个脚本调用 ``rokae_driver`` 已提供的关节空间轨迹服务，不直接连接
xCoreSDK。服务端会负责机器人状态、控制锁、软限位、路点跳变和轨迹执行。

单臂路径 JSON 格式：

    [[q1, q2, q3, q4, q5, q6, q7],
     [q1, q2, q3, q4, q5, q6, q7]]

或者：

    {"waypoints": [[q1, q2, q3, q4, q5, q6, q7], ...]}

双臂路径 JSON 格式：

    {
      "left": [[q1, q2, q3, q4, q5, q6, q7], ...],
      "right": [[q1, q2, q3, q4, q5, q6, q7], ...]
    }

所有关节角单位均为弧度。默认只执行 dry-run；确认现场安全后再增加
``--execute``。

直接运行本脚本时，会读取文件开头的 ``ARM_SELECTION``、``PATH_FILE``、
``TOTAL_MOTION_TIME_S`` 和 ``EXECUTE`` 配置，不需要再输入命令行参数。

当 ``TOTAL_MOTION_TIME_S`` 为正数时，脚本使用已有的 ServoJ 实时接口，
默认使用有界拐角圆滑和速度平滑过渡；CORNER_BLEND_RAD=0 恢复逐段停顿。
设为 ``None`` 才使用原来的
MoveJByPath 非实时服务。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence


JOINT_COUNT = 7
MIN_WAYPOINTS = 2
MAX_WAYPOINTS = 100
DEFAULT_MAX_STEP_RAD = 0.35

# ======================== 运行配置：只需修改这里 ========================
# 1 = 左臂，2 = 右臂，3 = 双臂。
ARM_SELECTION = 1

# 单臂可以设为 None，使用下面的 DEFAULT_SINGLE_WAYPOINTS；
# 也可以填写 JSON 路径文件，例如：
# PATH_FILE = "/home/niic/aide/hardware/arm/xCoreROS/src/test/left_path.json"
# 双臂模式 ARM_SELECTION = 3 时，必须填写包含 left/right 的 JSON 文件。
PATH_FILE: str | None = None

# False = 只 dry-run；确认路径、上电状态和现场安全后改为 True 才真实运动。
EXECUTE = True

# True：运动前请求所选机械臂上电；确认现场安全后手动开启。
# 不在结束或异常后自动补上电，不调用会影响双臂的 initialize。
POWER_ON_BEFORE_MOTION = True

# 期望整段轨迹的总时间，单位秒。正数启用定时 ServoJ 轨迹模式；
# 设置为 None 使用 MoveJByPath 服务（该服务本身不支持精确时间参数）。
TOTAL_MOTION_TIME_S = 6.0

# 当前驱动 ServoJ 默认周期和单帧最大关节跳变。
SERVO_PERIOD_S = 0.01  # 100 Hz，与驱动 period_s 保持一致。
SERVO_MAX_STEP_RAD = 0.01  # 保持原来的 1 rad/s 单帧变化速率边界。
# 拐角圆滑过渡允许的每关节偏差（rad）；0 恢复逐点停顿。
# 0.01 rad 约 0.57 度。此限制是关节空间偏差，不是 TCP 毫米距离。
CORNER_BLEND_RAD = 0.01
# ========================================================================

SERVICE_NAMES = {
    "left": "/robot_02/upperlimb/movej_by_path/left_arm",
    "right": "/robot_02/upperlimb/movej_by_path/right_arm",
    "dual": "/robot_02/upperlimb/movej_by_path/dual_arm",
}

# 仅用于 dry-run 的格式示例。真实执行前必须根据当前反馈位置和现场安全
# 条件替换为经过确认的路径；服务端仍会再次检查当前反馈、软限位和跳变。
DEFAULT_SINGLE_WAYPOINTS = [
    [1.299912, 1.114625, -1.005041, 0.859679, -0.531911, 0, -0.486584],
    [1.299912, 1.114625, -1.005041, 0.859679, -0.531911, 0.329822, -0.486584],
    [1.299912, 1.114625, -1.305041, 0.859679, -0.531911, 0.329822, -0.486584],
]


def validate_waypoints(raw: Any, name: str = "路径") -> list[list[float]]:
    """校验路点并转换为可写入 ROS 消息的浮点数组。"""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{name}必须是二维数组，例如 [[q1, ..., q7], ...]")
    if not MIN_WAYPOINTS <= len(raw) <= MAX_WAYPOINTS:
        raise ValueError(
            f"{name}必须包含 {MIN_WAYPOINTS} 到 {MAX_WAYPOINTS} 个路点，"
            f"当前为 {len(raw)} 个"
        )

    waypoints: list[list[float]] = []
    for point_index, point in enumerate(raw, start=1):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise ValueError(f"{name}第 {point_index} 个路点不是数组")
        if len(point) != JOINT_COUNT:
            raise ValueError(
                f"{name}第 {point_index} 个路点必须包含 {JOINT_COUNT} 个关节角，"
                f"实际为 {len(point)} 个"
            )
        try:
            joints = [float(value) for value in point]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name}第 {point_index} 个路点包含非数字值"
            ) from error
        if not all(math.isfinite(value) for value in joints):
            raise ValueError(
                f"{name}第 {point_index} 个路点包含 NaN 或无穷值"
            )
        waypoints.append(joints)
    return waypoints


def validate_steps(
    waypoints: Sequence[Sequence[float]],
    max_step_rad: float,
    name: str,
) -> None:
    """提前检查相邻路点跳变，规则与当前驱动默认值保持一致。"""
    if max_step_rad <= 0.0 or not math.isfinite(max_step_rad):
        raise ValueError("--max-step-rad 必须是正的有限数")
    for point_index in range(1, len(waypoints)):
        for joint_index in range(JOINT_COUNT):
            delta = abs(
                waypoints[point_index][joint_index]
                - waypoints[point_index - 1][joint_index]
            )
            if delta > max_step_rad:
                raise ValueError(
                    f"{name}第 {point_index + 1} 个路点的 J{joint_index + 1} "
                    f"跳变为 {delta:.6f} rad，超过 {max_step_rad:.6f} rad"
                )


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"无法读取路径文件 {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"路径文件不是合法 JSON: {error}") from error


def load_single_waypoints(path: Path | None) -> list[list[float]]:
    if path is None:
        return validate_waypoints(DEFAULT_SINGLE_WAYPOINTS)

    document = load_json(path)
    if isinstance(document, dict):
        document = document.get("waypoints")
    return validate_waypoints(document)


def load_dual_waypoints(
    path: Path | None,
) -> tuple[list[list[float]], list[list[float]]]:
    if path is None:
        raise ValueError("双臂模式必须通过 --path-file 提供包含 left/right 的 JSON 文件")

    document = load_json(path)
    if not isinstance(document, dict):
        raise ValueError(
            "双臂路径文件必须是对象，例如 {\"left\": [...], \"right\": [...] }"
        )

    left_raw = document.get("left", document.get("left_waypoints"))
    right_raw = document.get("right", document.get("right_waypoints"))
    left = validate_waypoints(left_raw, "左臂路径")
    right = validate_waypoints(right_raw, "右臂路径")
    if len(left) != len(right):
        raise ValueError(
            f"双臂路点数量必须相同，左臂 {len(left)} 个，右臂 {len(right)} 个"
        )
    return left, right


def flatten_waypoints(waypoints: Sequence[Sequence[float]]) -> list[float]:
    """将 [[q1..q7], ...] 展平为 MoveJByPath.srv 要求的数组。"""
    return [value for point in waypoints for value in point]


def print_waypoints(
    side: str,
    waypoints: Sequence[Sequence[float]],
) -> None:
    print(f"{side}臂：{len(waypoints)} 个路点，每点 {JOINT_COUNT} 轴，单位 rad")
    for index, point in enumerate(waypoints, start=1):
        values = ", ".join(f"{value:.9f}" for value in point)
        print(f"  P{index}: [{values}]")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用 Rokae MoveJByPath 服务执行关节空间多路点轨迹"
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "dual"),
        default="left",
        help="目标机械臂（默认：left；双臂使用 dual）",
    )
    parser.add_argument(
        "--path-file",
        type=Path,
        help=(
            "JSON 路径文件；单臂为数组或 waypoints 对象，双臂为 left/right 对象。"
            "单臂未指定时使用脚本内置示例"
        ),
    )
    parser.add_argument(
        "--service",
        help="覆盖默认服务名；默认根据 --side 自动选择",
    )
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=10.0,
        help="等待服务上线的秒数（默认：10）",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=130.0,
        help="等待运动服务返回的秒数（默认：130）",
    )
    parser.add_argument(
        "--max-step-rad",
        type=float,
        default=DEFAULT_MAX_STEP_RAD,
        help=(
            "客户端提前检查相邻路点的最大关节跳变；默认 0.35 rad，"
            "应与驱动参数 max_waypoint_delta_rad 一致"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正调用服务并使机械臂运动；不指定时仅执行 dry-run",
    )
    return parser


def validate_timeout(value: float, option_name: str) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{option_name} 必须是正的有限数")


def configured_side() -> str:
    """将文件开头的 ARM_SELECTION 转换为服务目标名称。"""
    side_options = {1: "left", 2: "right", 3: "dual"}
    try:
        return side_options[int(ARM_SELECTION)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ARM_SELECTION 只能设置为 1（左臂）、2（右臂）或 3（双臂）") from error


def configured_path() -> Path | None:
    if PATH_FILE is None:
        return None
    path = Path(PATH_FILE).expanduser()
    if not path.is_file():
        raise ValueError(f"PATH_FILE 文件不存在：{path}")
    return path


def validate_total_motion_time(value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("TOTAL_MOTION_TIME_S 必须是正的有限数，或设置为 None")


def _build_stop_frames(
    waypoints: Sequence[Sequence[float]],
    total_time_s: float,
) -> list[list[float]]:
    """按路点距离分配时间，并生成五次多项式 ServoJ 帧。"""
    validate_total_motion_time(total_time_s)
    if len(waypoints) < 2:
        raise ValueError("定时轨迹至少需要两个路点")

    # 使用每段最大关节位移作为权重，使七个关节和每个路段保持同步。
    segment_weights = [
        max(
            abs(waypoints[index + 1][joint] - waypoints[index][joint])
            for joint in range(JOINT_COUNT)
        )
        for index in range(len(waypoints) - 1)
    ]
    weight_sum = sum(segment_weights)
    if weight_sum <= 1e-12:
        segment_durations = [
            total_time_s / (len(waypoints) - 1)
            for _ in segment_weights
        ]
    else:
        segment_durations = [
            total_time_s * weight / weight_sum
            for weight in segment_weights
        ]

    cumulative = [0.0]
    for duration in segment_durations:
        cumulative.append(cumulative[-1] + duration)

    frame_count = max(1, math.ceil(total_time_s / SERVO_PERIOD_S))
    frames: list[list[float]] = []
    segment_index = 0
    for frame_index in range(frame_count + 1):
        current_time = min(frame_index * SERVO_PERIOD_S, total_time_s)
        while (
            segment_index < len(segment_durations) - 1
            and (
                segment_durations[segment_index] <= 1e-12
                or current_time > cumulative[segment_index + 1]
            )
        ):
            segment_index += 1

        segment_start = waypoints[segment_index]
        segment_end = waypoints[segment_index + 1]
        segment_duration = segment_durations[segment_index]
        if segment_duration <= 1e-12:
            blend = 1.0
        else:
            local_time = current_time - cumulative[segment_index]
            blend = max(0.0, min(1.0, local_time / segment_duration))

        # 五次多项式：10s^3 - 15s^4 + 6s^5，起止速度和加速度均为零。
        smooth = (
            10.0 * blend**3
            - 15.0 * blend**4
            + 6.0 * blend**5
        )
        frames.append([
            start + smooth * (end - start)
            for start, end in zip(segment_start, segment_end)
        ])

    frames[-1] = list(waypoints[-1])
    largest_step = max(
        max(
            abs(frames[index][joint] - frames[index - 1][joint])
            for joint in range(JOINT_COUNT)
        )
        for index in range(1, len(frames))
    )
    if largest_step > SERVO_MAX_STEP_RAD + 1e-9:
        raise ValueError(
            f"TOTAL_MOTION_TIME_S={total_time_s:.3f} s 太短："
            f"ServoJ 最大单帧关节跳变为 {largest_step:.6f} rad，"
            f"超过驱动限制 {SERVO_MAX_STEP_RAD:.6f} rad；请增大总时间"
        )
    return frames


def smooth_path_model(waypoints, total_time_s, tolerance):
    """Smooth a piecewise linear path with a compact positive kernel.

    Integrate quintic smoothstep velocity transitions. Position, velocity,
    acceleration and jerk are continuous; derivatives vanish at endpoints.
    The time window bounds per-coordinate deviation from the timed polyline.
    This is a local offline smoother, not a time-optimal Ruckig implementation.
    """
    validate_total_motion_time(total_time_s)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("圆滑偏差必须为正的有限值")
    if not waypoints or not waypoints[0]:
        raise ValueError("轨迹不能为空")
    dimension = len(waypoints[0])
    points = []
    for point in waypoints:
        if len(point) != dimension or not all(math.isfinite(v) for v in point):
            raise ValueError("轨迹维度不一致或包含非有限数")
        if not points or max(abs(a-b) for a,b in zip(point, points[-1])) > 1e-12:
            points.append(list(point))
    if len(points) == 1:
        return points, [], [], total_time_s / 4
    weights = [max(abs(a-b) for a,b in zip(p,q))
               for p,q in zip(points, points[1:])]
    length = sum(weights)
    # The positive symmetric kernel has E|offset| = 5*w/16.
    # Lipschitz bound: deviation <= (5*w/16)*L/(T-2*w) <= tolerance.
    # Kernel averaging also handles clustered or almost-identical waypoints.
    window = min(total_time_s / 8,
                 tolerance * total_time_s / (5 * length / 16 + 2 * tolerance))
    travel_time = total_time_s - 2 * window
    knots = [window]
    slopes = []
    for p,q,weight in zip(points, points[1:], weights):
        duration = travel_time * weight / length
        knots.append(knots[-1] + duration)
        slopes.append([(b-a)/duration for a,b in zip(p,q)])
    return points, knots, slopes, window


def sample_smooth_path(model, t):
    points, knots, slopes, window = model
    if not slopes:
        return list(points[0])
    result = list(points[0])
    previous = [0.0] * len(result)
    for knot, slope in zip(knots, slopes + [[0.0] * len(result)]):
        x = t - knot
        if x <= -window:
            ramp = 0.0
        elif x >= window:
            ramp = x
        else:
            u = (x + window) / (2 * window)
            ramp = 2 * window * (2.5*u**4 - 3*u**5 + u**6)
        result = [q + (v-old)*ramp for q,v,old in zip(result,slope,previous)]
        previous = slope
    return result


def build_timed_frames(waypoints, total_time_s):
    if CORNER_BLEND_RAD == 0:
        return _build_stop_frames(waypoints, total_time_s)
    model = smooth_path_model(waypoints, total_time_s, CORNER_BLEND_RAD)
    count = max(1, math.ceil(total_time_s / SERVO_PERIOD_S))
    frames = [sample_smooth_path(model, min(i*SERVO_PERIOD_S, total_time_s))
              for i in range(count+1)]
    frames[0], frames[-1] = list(waypoints[0]), list(waypoints[-1])
    # Bound steps continuously as well as at sample points, including tiny
    # segments that could otherwise disappear between samples.
    speed = max((abs(v) for slope in model[2] for v in slope), default=0.0)
    if speed * SERVO_PERIOD_S > SERVO_MAX_STEP_RAD + 1e-9:
        raise ValueError("总时间太短：连续速度超过 ServoJ 单周期位移限制，请增加总时间")
    return frames


def build_arm_frames(left_path, right_path, duration):
    if right_path is None:
        return build_timed_frames(left_path, duration), None
    if len(left_path) != len(right_path):
        raise ValueError("双臂路径数量不同")
    if CORNER_BLEND_RAD == 0:
        return (_build_stop_frames(left_path, duration),
                _build_stop_frames(right_path, duration))
    # Plan both arms as one 14-dimensional path to share corner times.
    frames = build_timed_frames(
        [list(l)+list(r) for l,r in zip(left_path,right_path)], duration
    )
    return [f[:7] for f in frames], [f[7:] for f in frames]


def servo_startup_ready(arms, status, ready_since, received, now):
    """Require recent running status and feedback received after readiness."""
    return all(
        status.get(arm, ("", 0.0))[0] == "running"
        and now - status[arm][1] <= 0.5
        and arm in ready_since
        and received.get(arm, 0.0) > ready_since[arm]
        and now - received.get(arm, 0.0) <= 0.5
        for arm in arms
    )


def call_timed_servoj(
    side: str,
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]] | None,
    total_time_s: float,
    state_timeout_s: float,
) -> int:
    """通过现有 ServoJ topic 按指定总时间发送五次关节轨迹。"""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from rokae_interfaces.msg import DualArmServoJ, ServoJ
    except ImportError as error:
        print(
            "无法导入 ServoJ 所需的 ROS 2 依赖，请先 source ROS 2 和工作区环境：\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /home/niic/aide/hardware/arm/xCoreROS/install/local_setup.bash\n"
            f"详细错误：{error}"
        )
        return 2

    class TimedServoNode(Node):
        def __init__(self) -> None:
            super().__init__(f"movej_timed_{side}_client")
            self.status = {}
            self.ready_since = {}
            self.received = {}
            self.create_subscription(
                String, "/robot_02/upperlimb/servoj/status",
                self._status_callback, 10,
            )
            self.current: dict[str, list[float] | None] = {
                "left": None,
                "right": None,
            }
            for arm in ("left", "right"):
                self.create_subscription(
                    JointState,
                    f"/robot_02/upperlimb/joint_states/{arm}_arm",
                    lambda message, arm_name=arm: self._state_callback(
                        arm_name, message
                    ),
                    qos_profile_sensor_data,
                )

            if side == "dual":
                self.publisher = self.create_publisher(
                    DualArmServoJ,
                    "/robot_02/upperlimb/servoj/dual_arm",
                    qos_profile_sensor_data,
                )
            else:
                self.publisher = self.create_publisher(
                    ServoJ,
                    f"/robot_02/upperlimb/servoj/{side}_arm",
                    qos_profile_sensor_data,
                )

        def _status_callback(self, message: Any) -> None:
            arm, _, state = message.data.partition(":")
            self.status[arm] = (state, time.monotonic())
            if state == "running":
                self.ready_since.setdefault(arm, time.monotonic())

        def _state_callback(self, arm: str, message: Any) -> None:
            if len(message.position) >= JOINT_COUNT:
                if not all(math.isfinite(v) for v in message.position[:7]):
                    return
                self.received[arm] = time.monotonic()
                self.current[arm] = [
                    float(value) for value in message.position[:JOINT_COUNT]
                ]

        def publish_frame(
            self,
            enable: bool,
            left_position: Sequence[float],
            right_position: Sequence[float] | None,
        ) -> None:
            if side == "dual":
                assert right_position is not None
                message = DualArmServoJ()
                message.enable = enable
                message.left_positions = list(left_position)
                message.right_positions = list(right_position)
            else:
                message = ServoJ()
                message.enable = enable
                message.positions = list(left_position)
            self.publisher.publish(message)

    rclpy.init()
    node = TimedServoNode()
    active = False
    try:
        required_arms = ("left", "right") if side == "dual" else (side,)
        deadline = time.monotonic() + state_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if all(node.current[arm] is not None for arm in required_arms):
                break
            rclpy.spin_once(node, timeout_sec=0.05)

        if not all(node.current[arm] is not None for arm in required_arms):
            node.get_logger().error(
                "未收到当前关节状态；请确认 launch 开启了 "
                "start_state_publisher"
            )
            return 3

        # Wait for discovery before sending, including the new driver status.
        deadline = time.monotonic() + state_timeout_s
        while (node.publisher.get_subscription_count() == 0
               or node.count_publishers("/robot_02/upperlimb/servoj/status") == 0):
            if not rclpy.ok() or time.monotonic() >= deadline:
                raise ValueError("ServoJ 或状态接口未上线，请重新编译并重启驱动")
            rclpy.spin_once(node, timeout_sec=0.02)

        # Discovery may take seconds. Acquire a new start pose afterwards.
        refresh_started = time.monotonic()
        deadline = refresh_started + state_timeout_s
        while not all(node.received.get(a, 0.0) > refresh_started
                      and time.monotonic() - node.received[a] <= 0.5
                      for a in required_arms):
            if not rclpy.ok() or time.monotonic() >= deadline:
                raise ValueError("启动前未收到新鲜关节反馈，未发送运动指令")
            rclpy.spin_once(node, timeout_sec=0.02)
        initial_arm = "left" if side == "dual" else side
        left_frames, right_frames = build_arm_frames(
            [node.current[initial_arm]] + list(left),
            ([node.current["right"]] + list(right)) if side == "dual" else None,
            total_time_s,
        )

        def check_status(require_running=False):
            if not rclpy.ok():
                raise ValueError("ROS 已关闭")
            for arm in required_arms:
                state, stamp = node.status.get(arm, ("", 0.0))
                if state.startswith("error:") or state == "stopped":
                    raise ValueError(f"{arm} ServoJ: {state}")
                if require_running and (
                    state != "running" or time.monotonic() - stamp > 0.5
                ):
                    raise ValueError(f"{arm} ServoJ 执行反馈中断")
                if (require_running
                        and time.monotonic() - node.received.get(arm, 0.0) > 0.5):
                    raise ValueError(f"{arm} 关节反馈过期")

        # Hold the initial position throughout SDK startup.
        node.status.clear()
        node.ready_since.clear()
        node.get_logger().info("保持当前位置，等待 ServoJ 启动和关节反馈恢复")
        deadline = time.monotonic() + state_timeout_s
        while True:
            check_status()
            active = True
            node.publish_frame(
                True, left_frames[0], right_frames[0] if right_frames else None
            )
            until = time.monotonic() + SERVO_PERIOD_S
            while rclpy.ok() and time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.002)
            check_status()
            if servo_startup_ready(
                required_arms, node.status, node.ready_since,
                node.received, time.monotonic()
            ):
                break
            if time.monotonic() >= deadline:
                raise ValueError(
                    "等待 ServoJ 启动或关节反馈恢复超时；轨迹尚未开始"
                )

        node.get_logger().info(
            f"驱动已就绪，开始 {total_time_s:.3f} 秒 ServoJ 定时轨迹，"
            f"共 {len(left_frames)} 帧，周期 {SERVO_PERIOD_S:.3f} 秒"
        )

        start_time = time.monotonic()
        for index, left_frame in enumerate(left_frames):
            target_time = start_time + min(
                index * SERVO_PERIOD_S, total_time_s
            )
            while rclpy.ok():
                remaining = target_time - time.monotonic()
                if remaining <= 0.0:
                    break
                rclpy.spin_once(node, timeout_sec=min(remaining, 0.005))

            right_frame = right_frames[index] if right_frames else None
            check_status(require_running=True)
            if time.monotonic() - target_time > SERVO_PERIOD_S:
                raise ValueError("轨迹发送调度超时，停止运动以避免突发补发")
            node.publish_frame(True, left_frame, right_frame)
            active = True

        final_left = left_frames[-1]
        final_right = right_frames[-1] if right_frames else None
        # Keep sending the final target until fresh joint feedback confirms it.
        final_sent = time.monotonic()
        deadline = final_sent + 2.0
        targets = dict(zip(required_arms,
                           [final_left, final_right] if side == "dual"
                           else [final_left]))
        while True:
            check_status(require_running=True)
            node.publish_frame(True, final_left, final_right)
            until = time.monotonic() + SERVO_PERIOD_S
            while rclpy.ok() and time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.002)
            check_status(require_running=True)
            if all(node.received[a] > final_sent and
                   max(abs(q - t) for q, t in zip(node.current[a], targets[a]))
                   <= 0.02 for a in required_arms):
                break
            if time.monotonic() >= deadline:
                raise ValueError("轨迹发送结束，但关节未到位（误差超过 0.02 rad）")
        node.publish_frame(False, final_left, final_right)
        # enable=False ends the servo session; it is not a power-off request.
        stop_sent = time.monotonic()
        deadline = stop_sent + state_timeout_s
        while True:
            if not rclpy.ok():
                raise ValueError("ROS 已关闭，无法确认停止后的上电状态")
            rclpy.spin_once(node, timeout_sec=0.02)
            for arm in required_arms:
                state, stamp = node.status.get(arm, ("", 0.0))
                if stamp > stop_sent and state.startswith("error:"):
                    raise ValueError(f"{arm} ServoJ 结束状态：{state}")
            if all(node.status.get(a, ("", 0.0))[0] == "idle_power_on"
                   and node.status[a][1] > stop_sent for a in required_arms):
                active = False
                break
            if time.monotonic() >= deadline:
                raise ValueError("停止后的上电状态未确认，请检查驱动日志及版本")
            node.publish_frame(False, final_left, final_right)
        node.get_logger().info("轨迹完成，关节已到位；ServoJ 退出时上电检查通过（不代表后续持续上电）")
        return 0
    except (KeyboardInterrupt, ValueError) as error:
        node.get_logger().error(f"定时轨迹未执行完成：{error}")
        return 1
    finally:
        if active:
            try:
                current_left = left_frames[-1]
                current_right = right_frames[-1] if right_frames else None
                node.publish_frame(False, current_left, current_right)
                # Allow stop messages to leave the publisher before destruction.
                for _ in range(3):
                    if not rclpy.ok():
                        break
                    rclpy.spin_once(node, timeout_sec=SERVO_PERIOD_S)
                    node.publish_frame(False, current_left, current_right)
            except (NameError, UnboundLocalError):
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def call_service(
    side: str,
    service_name: str,
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]] | None,
    service_timeout: float,
    response_timeout: float,
) -> int:
    """初始化 ROS 2，发送一次服务请求并返回 shell 状态码。"""
    # 延迟导入 ROS 2，使得不 source ROS 环境时仍可运行 dry-run 和语法检查。
    try:
        import rclpy
        from rclpy.node import Node
        from rokae_interfaces.srv import MoveJByPath
    except ImportError as error:
        print(
            "无法导入 ROS 2 依赖，请先 source ROS 2 和工作区环境：\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /home/niic/aide/hardware/arm/xCoreROS/install/local_setup.bash\n"
            f"详细错误：{error}"
        )
        return 2

    class ClientNode(Node):
        def __init__(self) -> None:
            super().__init__(f"movej_by_path_{side}_client")
            self.client = self.create_client(MoveJByPath, service_name)

    rclpy.init()
    node = ClientNode()
    try:
        node.get_logger().info(f"等待服务上线：{service_name}")
        if not node.client.wait_for_service(timeout_sec=service_timeout):
            node.get_logger().error(f"服务未上线：{service_name}")
            return 3

        request = MoveJByPath.Request()
        if side == "dual":
            assert right is not None
            request.left_joint_positions = flatten_waypoints(left)
            request.right_joint_positions = flatten_waypoints(right)
        else:
            request.joint_positions = flatten_waypoints(left)

        future = node.client.call_async(request)
        rclpy.spin_until_future_complete(
            node,
            future,
            timeout_sec=response_timeout,
        )
        if not future.done():
            node.get_logger().error(
                f"服务响应超时（{response_timeout:.1f} 秒）；请检查机械臂状态"
            )
            return 4

        response = future.result()
        if response is None:
            node.get_logger().error("服务调用没有返回响应")
            return 5
        if response.success:
            node.get_logger().info(f"关节轨迹执行成功：{response.message}")
            return 0

        node.get_logger().error(f"关节轨迹执行失败：{response.message}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def request_power_on(side: str, timeout: float) -> bool:
    """显式请求所选臂上电；失败/超时不重试、不启动轨迹。"""
    import rclpy
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node("trajectory_power_on_client")
    try:
        arms = ("left", "right") if side == "dual" else (side,)
        for arm in arms:
            name = f"/robot_02/upperlimb/power_on/{arm}_arm"
            client = node.create_client(Trigger, name)
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"上电服务未上线：{name}，请编译并重启驱动")
            node.get_logger().warn(f"请求 {arm} 臂上电（不发送运动指令）")
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
            if not future.done():
                raise RuntimeError("上电响应超时，实际电源状态未知；不重试、不执行轨迹")
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(f"{arm} 上电失败：{response.message if response else '无响应'}")
            node.get_logger().info(f"{arm}: {response.message}")
        return True
    except (Exception, KeyboardInterrupt) as error:
        node.get_logger().error(f"取消轨迹：{error}；已上电的机械臂不会自动下电")
        return False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args: list[str] | None = None) -> int:
    parser = build_argument_parser()
    configured_mode = args is None and len(sys.argv) == 1
    options = parser.parse_args(args)

    try:
        if configured_mode:
            options.side = configured_side()
            options.path_file = configured_path()
            options.execute = EXECUTE

        validate_timeout(options.service_timeout, "--service-timeout")
        validate_timeout(options.response_timeout, "--response-timeout")
        validate_total_motion_time(TOTAL_MOTION_TIME_S)

        if options.side == "dual":
            left, right = load_dual_waypoints(options.path_file)
            validate_steps(left, options.max_step_rad, "左臂路径")
            validate_steps(right, options.max_step_rad, "右臂路径")
            print_waypoints("左", left)
            print_waypoints("右", right)
        else:
            left = load_single_waypoints(options.path_file)
            right = None
            validate_steps(left, options.max_step_rad, f"{options.side}臂路径")
            print_waypoints("左" if options.side == "left" else "右", left)
    except (EOFError, KeyboardInterrupt):
        print("\n已取消本次操作，未调用机械臂。")
        return 130
    except ValueError as error:
        parser.error(str(error))

    service_name = options.service or SERVICE_NAMES[options.side]
    if TOTAL_MOTION_TIME_S is None:
        print(f"服务：{service_name}")
    else:
        print(
            f"控制模式：ServoJ 定时轨迹，总时间 {TOTAL_MOTION_TIME_S:.3f} 秒"
        )
        print(f"拐角圆滑偏差上限：{CORNER_BLEND_RAD} rad/关节（0 为逐点停顿）")

    if not options.execute:
        if configured_mode:
            print("配置 EXECUTE=False：dry-run，未调用机械臂。")
        else:
            print("dry-run：未调用机械臂；确认路点和现场安全后增加 --execute。")
        return 0

    if POWER_ON_BEFORE_MOTION:
        try:
            if not request_power_on(options.side, options.service_timeout):
                return 1
        except ImportError as error:
            print(f"无法导入 ROS 上电服务依赖，请 source ROS 和工作区环境：{error}")
            return 2

    if TOTAL_MOTION_TIME_S is not None:
        return call_timed_servoj(
            options.side,
            left,
            right,
            TOTAL_MOTION_TIME_S,
            options.service_timeout,
        )

    return call_service(
        options.side,
        service_name,
        left,
        right,
        options.service_timeout,
        options.response_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
