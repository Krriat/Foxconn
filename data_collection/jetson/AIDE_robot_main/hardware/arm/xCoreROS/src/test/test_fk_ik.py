#!/usr/bin/env python3
"""测试左右臂 FK/IK ROS 2 服务；只计算，不会控制机械臂运动。"""

import math
import sys

# ======================= 用户配置 =======================
ARM_TYPE = 1  # 1=左臂，2=右臂

# FK 输入关节角，顺序 J1...J7，单位 rad。
LEFT_JOINTS = [
1.197709349, 0.729368112, 0.000000000, 1.226708437, -1.148124111, 0.855286763, 0.541396564,
]
RIGHT_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# FK 返回有效臂角时，IK 是否使用该臂角约束七轴逆解。
USE_FK_ELBOW_FOR_IK = True
SERVICE_TIMEOUT_S = 10.0
# ========================================================


def validate_configuration():
    if ARM_TYPE not in (1, 2):
        raise ValueError("ARM_TYPE 必须是 1（左臂）或 2（右臂）")
    joints = LEFT_JOINTS if ARM_TYPE == 1 else RIGHT_JOINTS
    if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
        raise ValueError("关节角必须包含 7 个有限数值")
    if not math.isfinite(SERVICE_TIMEOUT_S) or SERVICE_TIMEOUT_S <= 0.0:
        raise ValueError("SERVICE_TIMEOUT_S 必须是正数")
    return joints


def pose_text(pose):
    p = pose.position
    q = pose.orientation
    return (
        f"位置 XYZ [m] = [{p.x:.9f}, {p.y:.9f}, {p.z:.9f}]\n"
        f"姿态 XYZW     = [{q.x:.9f}, {q.y:.9f}, {q.z:.9f}, {q.w:.9f}]"
    )


def call(node, client, request, operation):
    future = client.call_async(request)
    import rclpy

    rclpy.spin_until_future_complete(
        node, future, timeout_sec=SERVICE_TIMEOUT_S
    )
    if not future.done():
        raise RuntimeError(f"{operation} 调用超时")
    exception = future.exception()
    if exception is not None:
        raise RuntimeError(f"{operation} 调用异常：{exception}")
    response = future.result()
    if response is None:
        raise RuntimeError(f"{operation} 没有返回响应")
    if not response.success:
        raise RuntimeError(f"{operation} 失败：{response.message}")
    return response


def main():
    try:
        joints = validate_configuration()
        import rclpy
        from rclpy.node import Node
        from rokae_interfaces.srv import ForwardKinematics, InverseKinematics
    except (ImportError, ValueError) as error:
        print(f"无法启动测试：{error}", file=sys.stderr)
        print(
            "请先 source /opt/ros/humble/setup.bash 和工作区 "
            "install/local_setup.bash",
            file=sys.stderr,
        )
        return 1

    side = "left" if ARM_TYPE == 1 else "right"
    side_cn = "左臂" if ARM_TYPE == 1 else "右臂"
    fk_name = f"/robot_02/upperlimb/fk/{side}_arm"
    ik_name = f"/robot_02/upperlimb/ik/{side}_arm"

    rclpy.init()
    node = Node(f"test_fk_ik_{side}")
    fk_client = node.create_client(ForwardKinematics, fk_name)
    ik_client = node.create_client(InverseKinematics, ik_name)

    try:
        print(f"测试对象：{side_cn}")
        print("本程序只进行运动学计算，不会上电或发送运动指令。")
        print(f"等待服务：{fk_name}、{ik_name}")
        if not fk_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError(f"等待 FK 服务超时：{fk_name}")
        if not ik_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError(f"等待 IK 服务超时：{ik_name}")

        fk_request = ForwardKinematics.Request()
        fk_request.joints = joints
        fk_response = call(node, fk_client, fk_request, "FK")

        print("\nFK 输入关节角 [rad]：")
        print("[" + ", ".join(f"{value:.9f}" for value in joints) + "]")
        print("FK 输出 TCP（相对当前外部参考坐标系）：")
        print(pose_text(fk_response.pose))
        print(
            f"臂角={fk_response.elbow:.9f} rad，"
            f"has_elbow={fk_response.has_elbow}，"
            f"构型={list(fk_response.configuration)}"
        )

        ik_request = InverseKinematics.Request()
        ik_request.pose = fk_response.pose
        ik_request.use_elbow = bool(
            USE_FK_ELBOW_FOR_IK and fk_response.has_elbow
        )
        ik_request.elbow = fk_response.elbow if ik_request.use_elbow else 0.0
        ik_response = call(node, ik_client, ik_request, "IK")

        print("\nIK 输出关节角 [rad]：")
        print(
            "["
            + ", ".join(f"{value:.9f}" for value in ik_response.joints)
            + "]"
        )
        print(
            "SDK 内部 IK→FK 校验："
            f"位置误差={ik_response.position_error_m * 1000.0:.6f} mm，"
            f"姿态误差={ik_response.orientation_error_rad:.9f} rad"
        )

        verify_request = ForwardKinematics.Request()
        verify_request.joints = ik_response.joints
        verify_response = call(
            node, fk_client, verify_request, "IK 结果的 FK 回算"
        )
        print("FK 回算 TCP：")
        print(pose_text(verify_response.pose))
        print("\nFK、IK 与回算服务均调用成功。")
        print("提示：七轴存在多组逆解，IK 关节角不一定与原 FK 输入完全相同。")
        return 0
    except (RuntimeError, KeyboardInterrupt) as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
