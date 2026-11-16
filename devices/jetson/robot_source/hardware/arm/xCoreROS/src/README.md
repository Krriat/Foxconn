# ROS 源码

状态更新：2026-09-13。

`rokae_driver`、`rokae_interfaces`、`rokae_bringup` 已重新用于本机 SDK 的 ROS 2 接口。
当前启动方法和接口映射见 [SDK 节点说明](../SDK_NODES.md)。其他旧应用包仍保留供参考。

| 目录 | 状态 |
| --- | --- |
| `rokae_driver` | 当前使用：机械臂 SDK 驱动与通信封装 |
| `rokae_interfaces` | 当前使用：消息、服务与 Action 定义 |
| `rokae_bringup` | 当前使用：节点启动与配置 |
| `aruco_scanner` | 已废弃：本工作空间中的视觉 ROS 包 |
| `box_detection_interfaces` | 已废弃：本工作空间中的检测接口包 |
| `realsense_multi_camera` | 已废弃：本工作空间中的相机 ROS 包 |
| `rokae_motion` | ROS 接入已废弃；应用程序保留参考价值 |
| `test` | 旧 ROS 接口的测试与调用示例，仅作历史参考 |

`rokae_motion` 中的行为树、YAML 动作配置、任务编排和视觉运动应用逻辑仍可作为后续开发参考。复用时需适配实际使用的通信与控制接口；保留源码不表示旧 ROS 接口仍在使用。

历史废弃状态不适用于上述三个当前使用的接口包。
