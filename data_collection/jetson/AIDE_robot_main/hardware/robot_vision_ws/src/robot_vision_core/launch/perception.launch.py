#!/usr/bin/env python3

"""按需启停 robot_vision_core 中的感知节点。

设计要点：
  - 与 multi_realsense.launch.py 解耦：相机在另一个终端单独启动，
    本文件只起感知节点。Ctrl+C 关闭本 launch 不会动到相机进程。
  - 五个布尔开关分别控制五个节点，套 IfCondition，未启用的节点
    完全不进入 launch 状态机（不是启动后立即退出）。
  - 节点的 topic / 参数默认值沿用各自源码中的 declare_parameter
    默认值（均为头部相机）。需要切换相机或覆盖参数时，用
    `ros2 run robot_vision_core <节点程序> --ros-args -p <param>:=<value>`。

用法示例：
  # 显式启用 YOLO；所有感知节点默认关闭
  ros2 launch robot_vision_core perception.launch.py run_yolo:=true

  # 同时起 YOLO 与 ArUco ID 识别
  ros2 launch robot_vision_core perception.launch.py \\
      run_yolo:=true run_aruco_id:=true

  # 只起 ArUco 刀具位姿解算
  ros2 launch robot_vision_core perception.launch.py \\
      run_yolo:=false run_aruco_pose:=true

前置条件：
  终端 1 已先执行（相机常驻）：
    ros2 launch robot_vision_bringup multi_realsense.launch.py \\
        start_camera_manager:=false
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    run_yolo = LaunchConfiguration("run_yolo")
    run_aruco_id = LaunchConfiguration("run_aruco_id")
    run_aruco_pose = LaunchConfiguration("run_aruco_pose")
    run_big_box = LaunchConfiguration("run_big_box")
    run_small_box = LaunchConfiguration("run_small_box")

    yolo_node = Node(
        package="robot_vision_core",
        executable="position_detection",
        name="object_detector",
        namespace="/robot/vision",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(run_yolo),
    )

    aruco_id_node = Node(
        package="robot_vision_core",
        executable="aruco_scanner",
        name="id_detector",
        namespace="/robot/vision/aruco",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(run_aruco_id),
    )

    aruco_pose_node = Node(
        package="robot_vision_core",
        executable="arucode_position_detection",
        name="pose_estimator",
        namespace="/robot/vision/aruco",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(run_aruco_pose),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "run_yolo",
                default_value="false",
                description=(
                    "是否启动 YOLOv8 分割+深度反投影节点 "
                    "(position_detection)。"
                ),
            ),
            DeclareLaunchArgument(
                "run_aruco_id",
                default_value="false",
                description=(
                    "是否启动 ArUco ID 识别节点 (aruco_scanner)。"
                ),
            ),
            DeclareLaunchArgument(
                "run_aruco_pose",
                default_value="false",
                description=(
                    "是否启动 ArUco 刀具位姿解算节点 "
                    "(arucode_position_detection)。"
                ),
            ),
            DeclareLaunchArgument("run_big_box", default_value="false"),
            DeclareLaunchArgument("run_small_box", default_value="false"),
            Node(package="robot_vision_core", executable="big_box_detection",
                 name="large_detector", namespace="/robot/vision/box",
                 output="screen", condition=IfCondition(run_big_box)),
            Node(package="robot_vision_core", executable="small_box_detection",
                 name="small_detector", namespace="/robot/vision/box",
                 output="screen", condition=IfCondition(run_small_box)),
            yolo_node,
            aruco_id_node,
            aruco_pose_node,
        ]
    )
