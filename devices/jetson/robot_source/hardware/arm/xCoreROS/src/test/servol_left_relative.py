#!/usr/bin/env python3
"""左臂 ServoL 相对平移；ROS 位姿使用 TCP 相对外部参考系。"""
import copy
import math
import time

# ======================= 用户配置 =======================
DX = 0.00                    # 沿左臂外部参考系 X 方向，单位 m
DY = 0.0                     # 沿左臂外部参考系 Y 方向，单位 m
DZ = 0.050                   # 沿左臂外部参考系 Z 方向，单位 m
MOTION_TIME_S = 6.0
EXECUTE = True              # 确认现场安全和位移后改为 True
# 不自动上电；运行前由用户通过已提供的上电接口上电。
# ========================================================
PERIOD_S = 0.01               # ROS Topic 与 SDK ServoL 均为 100 Hz
MAX_STEP_M = 0.0005           # 与 Pico 一致的默认单帧位移限制
FEEDBACK_TIMEOUT_S = 0.5
START_TIMEOUT_S = 10.0
COMMAND_POSITION_TOLERANCE_M = 0.00001
COMMAND_ORIENTATION_TOLERANCE_RAD = 0.0001
COMMAND_TOPIC = "/robot_02/upperlimb/servol/left_arm"
# 驱动生成的 EndInRef TCP 命令位姿（不是机械臂实测位姿）。
COMMAND_POSE_TOPIC = "/robot_02/upperlimb/servol/command_pose/left_arm"
STATUS_TOPIC = "/robot_02/upperlimb/servol/status"


def offsets():
    delta = (DX, DY, DZ)
    if not all(math.isfinite(v) for v in (*delta, MOTION_TIME_S)) or MOTION_TIME_S <= 0:
        raise ValueError("位移必须为有限值，总时间必须为正数")
    # 五次平滑函数的最大导数为 1.875，先检查整个连续曲线的速度上界。
    if 1.875 * math.dist(delta, (0, 0, 0)) / MOTION_TIME_S * PERIOD_S > MAX_STEP_M:
        raise ValueError("运动时间过短，会超过每帧 0.5 mm 限制，请增加 MOTION_TIME_S")
    count = math.ceil(MOTION_TIME_S / PERIOD_S)
    result = []
    for i in range(count + 1):
        t = min(i * PERIOD_S, MOTION_TIME_S)
        u = t / MOTION_TIME_S
        s = 10*u**3 - 15*u**4 + 6*u**5
        result.append((t, tuple(s*v for v in delta)))
    return result


def main():
    try:
        frames = offsets()
        print(f"左臂相对平移：DX={DX} m, DY={DY} m, DZ={DZ} m")
        print(f"保持姿态；ROS Topic / SDK ServoL 100 Hz；总时间 {MOTION_TIME_S} s；{len(frames)} 帧")
        if not EXECUTE:
            print("EXECUTE=False：仅检查参数，未连接 ROS、未上电、未发送运动。")
            return 0
        if DX == DY == DZ == 0:
            print("位移为零，不启动 ServoL。")
            return 0
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from geometry_msgs.msg import Pose, PoseStamped
        from std_msgs.msg import Bool, String
    except (ImportError, ValueError) as error:
        print(f"无法执行：{error}")
        print("ROS 依赖缺失时，先 source /opt/ros/humble/setup.bash 和工作区 install/local_setup.bash")
        return 1

    class Client(Node):
        def __init__(self):
            super().__init__("servol_left_relative_client")
            self.pose = None
            self.received = 0.0
            self.frame_id = None
            self.status = ""
            self.status_time = 0.0
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.publisher = self.create_publisher(Pose, COMMAND_TOPIC, qos)
            self.stop_publisher = self.create_publisher(
                Bool, "/robot_02/upperlimb/servol/stop/left_arm", 10)
            self.create_subscription(
                PoseStamped, COMMAND_POSE_TOPIC, self.command_pose, qos)
            self.create_subscription(String, STATUS_TOPIC, self.state, 10)

        def command_pose(self, msg):
            p, q = msg.pose.position, msg.pose.orientation
            values = (p.x,p.y,p.z,q.x,q.y,q.z,q.w)
            if not all(math.isfinite(v) for v in values):
                return
            norm = math.sqrt(sum(v*v for v in values[3:]))
            if abs(norm-1) > 0.01:
                return
            if self.frame_id is not None and msg.header.frame_id != self.frame_id:
                return
            self.frame_id = msg.header.frame_id
            self.pose = copy.deepcopy(msg.pose)
            self.received = time.monotonic()

        def state(self, msg):
            arm, _, state = msg.data.partition(":")
            if arm == "left":
                self.status, self.status_time = state, time.monotonic()

    rclpy.init()
    node = Client()
    sent = False
    result = 1

    def spin_until(deadline):
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.002, max(0.0, deadline-time.monotonic())))
        if not rclpy.ok():
            raise RuntimeError("ROS 已关闭")

    def check(starting=False):
        if node.status.startswith("error:") or node.status == "stopped":
            raise RuntimeError(f"左臂 ServoL：{node.status}")
        if not starting:
            now = time.monotonic()
            if now-node.received > FEEDBACK_TIMEOUT_S:
                raise RuntimeError("左臂 SDK 命令位姿更新中断")
            if node.status != "running" or now-node.status_time > FEEDBACK_TIMEOUT_S:
                raise RuntimeError("ServoL running 状态中断")

    try:
        deadline = time.monotonic()+START_TIMEOUT_S
        while (node.publisher.get_subscription_count() == 0 or
               node.count_publishers(STATUS_TOPIC) == 0 or
               node.count_publishers(COMMAND_POSE_TOPIC) == 0):
            if time.monotonic() > deadline:
                raise RuntimeError("等待 ServoL 接口超时，请重新编译并启动驱动")
            spin_until(time.monotonic()+PERIOD_S)
        node.status, node.status_time = "", 0.0
        beginning = time.monotonic()
        trigger = Pose()
        trigger.orientation.w = 1.0
        node.get_logger().info(
            "请求启动 ServoL，读取 EndInRef TCP 锚点；"
            "驱动按 Pico 方式转换为 FlanInBase；不会自动上电")
        while True:
            check(starting=True)
            # 仅用于触发会话；驱动在取得同源反馈前不会执行此占位位姿。
            node.publisher.publish(trigger)
            sent = True
            spin_until(time.monotonic()+PERIOD_S)
            check(starting=True)
            if (node.status == "ready" and node.status_time > beginning and
                    node.received > beginning and time.monotonic()-node.received < FEEDBACK_TIMEOUT_S):
                break
            if time.monotonic()-beginning > START_TIMEOUT_S:
                raise RuntimeError("ServoL 启动超时")
        initial = copy.deepcopy(node.pose)
        while node.status != "running":
            check(starting=True)
            node.publisher.publish(initial)
            spin_until(time.monotonic()+PERIOD_S)
            if time.monotonic()-beginning > START_TIMEOUT_S:
                raise RuntimeError("ServoL 初始位姿对齐超时")
        check()
        node.get_logger().info("ServoL 已就绪，开始相对位移")
        start = time.monotonic()
        target = initial
        for t, delta in frames:
            spin_until(start+t)
            check()
            if time.monotonic()-(start+t) > PERIOD_S:
                raise RuntimeError("100 Hz 调度超时，停止发送，不补发追赶")
            target = copy.deepcopy(initial)
            target.position.x += delta[0]
            target.position.y += delta[1]
            target.position.z += delta[2]
            node.publisher.publish(target)
        final_sent = time.monotonic()
        deadline = final_sent+2.0
        stable = 0
        last_feedback = 0.0
        error = math.inf
        angle = math.inf
        while True:
            check()
            node.publisher.publish(target)
            spin_until(time.monotonic()+PERIOD_S)
            if node.received > max(final_sent, last_feedback):
                last_feedback = node.received
                actual, goal = node.pose.position, target.position
                error = math.dist((actual.x,actual.y,actual.z),(goal.x,goal.y,goal.z))
                q, g = node.pose.orientation, target.orientation
                dot = sum(a*b for a,b in zip((q.x,q.y,q.z,q.w),(g.x,g.y,g.z,g.w)))
                norm = math.sqrt(sum(v*v for v in (q.x,q.y,q.z,q.w))*sum(v*v for v in (g.x,g.y,g.z,g.w)))
                angle = 2*math.acos(min(1.0,abs(dot)/norm))
                stable = stable+1 if (error <= COMMAND_POSITION_TOLERANCE_M and
                                      angle <= COMMAND_ORIENTATION_TOLERANCE_RAD) else 0
                if stable >= 3:
                    break
            if time.monotonic() > deadline:
                actual = node.pose.position
                goal = target.position
                raise RuntimeError(
                    "发送结束但 SDK 插值命令未到终点："
                    f"命令位置误差={error*1000:.3f} mm，姿态误差={angle:.6f} rad；"
                    f"当前命令=({actual.x:.6f}, {actual.y:.6f}, {actual.z:.6f})，"
                    f"目标=({goal.x:.6f}, {goal.y:.6f}, {goal.z:.6f})")
        node.get_logger().info(
            "左臂 ServoL 命令轨迹已完成（不代表实测到位），请求驱动停止")
        result = 0
    except (Exception, KeyboardInterrupt) as error:
        node.get_logger().error(f"轨迹中止：{error}")
    finally:
        if sent and rclpy.ok():
            stop_message = Bool()
            stop_message.data = True
            node.stop_publisher.publish(stop_message)
            stop_time = time.monotonic()
            acknowledged = (node.status_time > beginning and
                            (node.status == "stopped" or node.status.startswith("error:")))
            if acknowledged:
                node.get_logger().info(f"驱动会话已结束：{node.status}；不保证持续上电")
            try:
                while not acknowledged and time.monotonic()-stop_time < 5.0 and rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.02)
                    if node.status_time > stop_time and (node.status == "stopped" or node.status.startswith("error:")):
                        acknowledged = True
                        if node.status.startswith("error:") and node.status != "error:command watchdog expired":
                            result = 1
                        node.get_logger().info(f"停止阶段驱动反馈：{node.status}；不保证持续上电")
                        break
            except (Exception, KeyboardInterrupt):
                pass
            if not acknowledged:
                result = 1
                node.get_logger().error("未确认 ServoL 停止，请检查驱动和机械臂状态")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
