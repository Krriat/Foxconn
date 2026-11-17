#!/usr/bin/env python3
"""调用左右臂绝对 MoveL ROS 2 服务。"""

import math
import sys


# ======================= 用户配置 =======================
ARM_TYPE = 1  # 1=左臂，2=右臂

# 完整绝对 TCP 目标：[x, y, z, roll, pitch, yaw]。
# XYZ 单位 m，RPY 为 XYZ Euler 角，单位 rad。
# 必须根据当前外部参考坐标系和现场标定结果填写全部 6 个值。
TARGET_TCP_RPY = [0.26712913310313624, 0.31802965378648357, -0.4009537625992056, 2.990419029348442, -1.2981718767773467, 0.1334008457516025]

# False：只打印并检查参数；填写并确认目标安全后改为 True 才调用 MoveL。
EXECUTE = True

# False：保持控制器当前七轴臂角和构型；True：使用 ELBOW_RAD。
USE_ELBOW = False
ELBOW_RAD = 0.0

SPEED_MM_S = 20.0
ZONE_MM = 0.0
SERVICE_WAIT_TIMEOUT_S = 10.0
MOTION_RESPONSE_TIMEOUT_S = 70.0
# ========================================================


def finite_vector(values, length, name):
    if len(values) != length:
        raise ValueError(f"{name} 必须包含 {length} 个数值")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 不能包含 NaN 或无穷值")
    return result


def wait_result(node, future, timeout_s, operation):
    import rclpy

    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        raise RuntimeError(f"{operation}超时（{timeout_s:.1f} s）")
    exception = future.exception()
    if exception is not None:
        raise RuntimeError(f"{operation}异常：{exception}")
    response = future.result()
    if response is None:
        raise RuntimeError(f"{operation}没有返回响应")
    if not response.success:
        raise RuntimeError(f"{operation}失败：{response.message}")
    return response


def validate_configuration():
    if ARM_TYPE not in (1, 2):
        raise ValueError("ARM_TYPE 必须是 1（左臂）或 2（右臂）")
    if not math.isfinite(SPEED_MM_S) or SPEED_MM_S <= 0.0:
        raise ValueError("SPEED_MM_S 必须是正的有限数")
    if not math.isfinite(ZONE_MM) or ZONE_MM < 0.0:
        raise ValueError("ZONE_MM 必须是非负有限数")
    if USE_ELBOW and not math.isfinite(ELBOW_RAD):
        raise ValueError("启用臂角时，ELBOW_RAD 必须是有限数")

    tcp_rpy = finite_vector(TARGET_TCP_RPY, 6, "TARGET_TCP_RPY")
    return tcp_rpy[:3], tcp_rpy[3:]


def main():
    try:
        position, orientation = validate_configuration()
        import rclpy
        from rclpy.node import Node
        from rokae_interfaces.srv import MoveL
    except (ImportError, TypeError, ValueError) as error:
        print(f"无法启动 MoveL 客户端：{error}", file=sys.stderr)
        print(
            "请先 source /opt/ros/humble/setup.bash 和工作区 "
            "install/local_setup.bash",
            file=sys.stderr,
        )
        return 1

    side = "left" if ARM_TYPE == 1 else "right"
    side_cn = "左臂" if ARM_TYPE == 1 else "右臂"
    move_service_name = f"/robot_02/upperlimb/move_l/{side}_arm"

    rclpy.init()
    node = Node(f"movel_{side}_client")
    move_client = node.create_client(MoveL, move_service_name)

    try:
        print(f"控制对象：{side_cn}")
        print("绝对位置 XYZ [m]：", position)
        print("绝对姿态 RPY [rad]：", orientation)
        print(
            f"use_elbow={USE_ELBOW}，elbow={ELBOW_RAD:.6f} rad，"
            f"速度={SPEED_MM_S:.3f} mm/s，zone={ZONE_MM:.3f} mm"
        )
        print("本程序不会自动上电；调用前请确认机械臂已上电且现场安全。")

        if not EXECUTE:
            print("当前 EXECUTE=False，仅检查参数，未调用 MoveL 服务。")
            print("确认 TARGET_TCP_RPY 安全后，将 EXECUTE 改为 True。")
            return 0

        if not move_client.wait_for_service(
            timeout_sec=SERVICE_WAIT_TIMEOUT_S
        ):
            raise RuntimeError(f"等待 MoveL 服务超时：{move_service_name}")

        request = MoveL.Request()
        request.position = position
        request.orientation_rpy = orientation
        request.use_elbow = USE_ELBOW
        request.elbow = ELBOW_RAD if USE_ELBOW else 0.0
        request.speed_mm_s = SPEED_MM_S
        request.zone_mm = ZONE_MM

        response = wait_result(
            node,
            move_client.call_async(request),
            MOTION_RESPONSE_TIMEOUT_S,
            f"{side_cn} MoveL",
        )
        node.get_logger().info(f"MoveL 执行成功：{response.message}")
        return 0
    except KeyboardInterrupt:
        node.get_logger().warning("用户中断 MoveL 客户端")
        return 130
    except RuntimeError as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
