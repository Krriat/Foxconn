#!/usr/bin/env python3
"""调用左臂或右臂 MoveAbsJ ROS 2 服务。"""

import math
import sys


# ======================= 用户配置 =======================
ARM_TYPE = 2  # 1=左臂，2=右臂

# J1..J7，单位 rad。默认值来自 xCoreSDK cxl/moveabsj.cpp 的右臂示例。
TARGET_JOINTS = [
    1.45036,
    -1.570796327,
    -1.570796327,
    0.0,
    0.0,
    0.0,
    0.0,
]

SPEED_MM_S = 1000.0
ZONE_MM = 10.0

# 默认只校验并打印。确认当前姿态、目标和现场安全后改为 True。
EXECUTE = False
SERVICE_WAIT_TIMEOUT_S = 10.0
MOTION_RESPONSE_TIMEOUT_S = 70.0
# ========================================================


def validate_configuration():
    if ARM_TYPE not in (1, 2):
        raise ValueError("ARM_TYPE 必须是 1（左臂）或 2（右臂）")
    if len(TARGET_JOINTS) != 7:
        raise ValueError("TARGET_JOINTS 必须包含 J1..J7 共 7 个关节角")
    joints = [float(value) for value in TARGET_JOINTS]
    if not all(math.isfinite(value) for value in joints):
        raise ValueError("TARGET_JOINTS 不能包含 NaN 或无穷值")
    if not math.isfinite(SPEED_MM_S) or SPEED_MM_S <= 0.0:
        raise ValueError("SPEED_MM_S 必须是正的有限数")
    if not math.isfinite(ZONE_MM) or ZONE_MM < 0.0:
        raise ValueError("ZONE_MM 必须是非负有限数")
    return joints


def main():
    try:
        joints = validate_configuration()
        import rclpy
        from rclpy.node import Node
        from rokae_interfaces.srv import MoveAbsJ
    except (ImportError, TypeError, ValueError) as error:
        print(f"无法启动 MoveAbsJ 客户端：{error}", file=sys.stderr)
        return 1

    side = "left" if ARM_TYPE == 1 else "right"
    side_cn = "左臂" if ARM_TYPE == 1 else "右臂"
    service_name = f"/robot_02/upperlimb/move_absj/{side}_arm"

    print(f"控制对象：{side_cn}")
    print(f"目标 J1..J7 [rad]：{joints}")
    print(f"speed={SPEED_MM_S:.3f} mm/s，zone={ZONE_MM:.3f} mm")
    print("本程序不会自动上电；调用前请确认机械臂已上电且现场安全。")
    if not EXECUTE:
        print("当前 EXECUTE=False，仅检查参数，未调用 MoveAbsJ 服务。")
        return 0

    rclpy.init()
    node = Node(f"moveabsj_{side}_client")
    client = node.create_client(MoveAbsJ, service_name)
    try:
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_S):
            raise RuntimeError(f"等待 MoveAbsJ 服务超时：{service_name}")

        request = MoveAbsJ.Request()
        request.joint_positions = joints
        request.speed_mm_s = SPEED_MM_S
        request.zone_mm = ZONE_MM
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=MOTION_RESPONSE_TIMEOUT_S
        )
        if not future.done():
            raise RuntimeError("等待 MoveAbsJ 执行结果超时")
        if future.exception() is not None:
            raise RuntimeError(f"MoveAbsJ 调用异常：{future.exception()}")
        response = future.result()
        if response is None or not response.success:
            message = "没有响应" if response is None else response.message
            raise RuntimeError(f"MoveAbsJ 执行失败：{message}")
        node.get_logger().info(f"MoveAbsJ 执行成功：{response.message}")
        return 0
    except (KeyboardInterrupt, RuntimeError) as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
