# 本机 xCore SDK ROS 2 接口

节点与业务接口沿用包内的 `/robot_02/upperlimb` 命名。左右臂共享同一个驱动进程，
每臂只有一个 SDK 连接，运动接口之间使用同一把命令锁。
整机只运行一套驱动；后台 `aide-arm-sdk` 已运行时，无需再手动启动。

## 构建与启动

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
colcon build --symlink-install --packages-up-to rokae_bringup \
  --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/local_setup.bash
bash tools/start_state_publisher.sh
```

该入口统一启动双臂接口和头部相机彩色、原始深度、对齐深度发布。
头部相机使用 `hardware/robot_vision_ws` 已构建的 RealSense 驱动及 `config/head_camera.yaml`，
不需要修改行为树。相机接口为：

| 接口 | 类型 |
| --- | --- |
| `/robot_02/sensors/cameras/head/color/image_raw` | `sensor_msgs/msg/Image` |
| `/robot_02/sensors/cameras/head/depth/image_rect_raw` | `sensor_msgs/msg/Image` |
| `/robot_02/sensors/cameras/head/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` |

同时发布各流的 CameraInfo；图像分辨率、编码和单位见
[头部相机说明](../../robot_vision_ws/HEAD_CAMERA.md)。
启动器先确认三个图像话题各有一个发布者且收到有效帧，再启动双臂接口。
已有头部相机时验证并复用，不创建第二个采集进程；重复运行独立相机脚本会被拦截。
相机未运行时由统一入口启动，按 Ctrl+C 时与双臂一起退出；复用的外部相机不会被停止。
仅需机械臂时可用 `bash tools/start_state_publisher.sh start_head_camera:=false`。
此次统一启动不增加开机自启，也不启动 `box_cut_ros.py` 等检测算法。

双臂入口启用现有的状态发布、初始化、MoveAbsJ Action/Service、MoveL、MoveLRelative、
MoveJByPath、FK 和 IK 节点。启动只建立连接和开放接口；初始化、上电及运动均由后续
显式请求触发。`start_state_publisher.sh` 和 `start_sdk_nodes.sh` 保留为同一入口的兼容别名。

若以 `aide-arm-sdk` 临时后台服务运行，可通过以下命令管理；目前未配置开机自启：

```bash
systemctl --user status aide-arm-sdk
systemctl --user restart aide-arm-sdk
systemctl --user stop aide-arm-sdk
```

后台服务也应使用上述统一入口。若后台服务正在运行，需要在终端前台启动时，先执行
`systemctl --user stop aide-arm-sdk`，然后运行 `bash tools/start_upperlimb.sh`，
确保只有一个驱动实例。
若当前驱动在其他终端前台运行，请先在原终端按 Ctrl+C 结束旧入口，再运行更新后的脚本。
统一启动脚本会检查进程和启动锁，发现已有驱动时直接退出，不创建第二套 SDK 连接。
直接重复执行 `ros2 launch` 会绕过脚本检查，应统一使用此入口。
若 ROS2 出现同名节点警告或同一服务的多个提供者，先排除重复驱动，再执行行为树。

## 接口与示例 SDK 的对应关系

以下 `{arm}` 可取 `left_arm` 或 `right_arm`：

| 接口名 | 类型 | SDK 调用 |
| --- | --- | --- |
| `/robot_02/upperlimb/move_absj/{arm}` | `rokae_interfaces/srv/MoveAbsJ` | `MoveAbsJCommand → moveAppend → moveStart` |
| `/robot_02/upperlimb/movej_by_path/{arm}` | `rokae_interfaces/srv/MoveJByPath` | 原有关节路径；可选笛卡尔输入使用 `calcIk → MoveJCommand → moveAppend → moveStart` |
| `/robot_02/upperlimb/move_l/{arm}` | `rokae_interfaces/srv/MoveL` | `MoveLCommand → moveAppend → moveStart` |
| `/robot_02/upperlimb/move_l_relative/{arm}` | `rokae_interfaces/srv/MoveLRelative` | 在当前 TCP 上计算相对目标，再调用 MoveL |
| `/robot_02/upperlimb/get_cartesian_state/{arm}` | `rokae_interfaces/srv/GetCartesianState` | `cartPosture(endInRef)` |
| `/robot_02/upperlimb/fk/{arm}` | `rokae_interfaces/srv/ForwardKinematics` | 当前工具/外部参考系下的 `model.calcFk` |
| `/robot_02/upperlimb/ik/{arm}` | `rokae_interfaces/srv/InverseKinematics` | `model.calcIk`，并以 FK 验证结果 |
| `/robot_02/upperlimb/initialize/{arm}` | `std_srvs/srv/Trigger` | 切换 NRT、automatic，并上电 |
| `/robot_02/upperlimb/initialize/dual_arm` | `std_srvs/srv/Trigger` | 初始化双臂 |
| `/robot_02/upperlimb/power_on/{arm}` | `std_srvs/srv/Trigger` | 单臂上电；已上电则保持当前模式 |
| `/robot_02/upperlimb/stop/{arm}` | `std_srvs/srv/Trigger` | 单臂停止、清空 NRT 队列；独立回调组可在运动期间调用 |

`move_absj/{arm}` 同时保留原有 `control_msgs/action/FollowJointTrajectory` Action。
`movej_by_path/dual_arm` 继续接收双臂关节路径，不接受新增的单臂笛卡尔输入。

状态话题保留 `joint_states`、`flange_pose`、`tcp_gripper_pose`、
`tcp_gripper_state`、`tcp_cut_pos`、`jacobian`。夹爪和刀具偏移仍分别沿法兰局部负 Z 轴
0.11 m，以及局部 Z 轴 +15° 后沿旋转后的负 X 轴 0.11 m。

## `运动类型演示.yaml` 的请求映射

检查脚本直接读取用户提供的 YAML。默认仅打印和检查请求，不连接 ROS 或机器人：

```bash
python3 tools/motion_demo_client.py
python3 tools/motion_demo_client.py --action-id demo_j_above
```

只有显式增加 `--execute` 才发送运动请求；本次部署验证未使用该选项。
执行前应由调用端显式完成初始化，并确认本次目标与现场路径。

| YAML 字段 | ROS 请求 |
| --- | --- |
| MoveAbsJ `source.joints` | `MoveAbsJ.joint_positions`，每个 7 轴点一次请求 |
| MoveAbsJ `motion.joint_speed` | `MoveAbsJ.joint_speed_scale`；`0` 表示沿用节点参数 |
| MoveJ `source.right_arm.pose` | `MoveJByPath.pose`，并设置 `use_cartesian_target=true` |
| MoveJ `motion.joint_speed` | `MoveJByPath.joint_speed_scale` |
| MoveL `source.right_arm.pose.position` | `MoveL.position`，单位 m |
| MoveL `source.right_arm.pose.orientation` | 四元数转换为 `MoveL.orientation_rpy`，单位 rad |
| MoveL `motion.line_speed` | 乘 1000 后传入 `MoveL.speed_mm_s` |
| `elbow`、`conf_data` | 对应请求的 `use_elbow`、`elbow`、`conf_data` |

新增的笛卡尔 MoveJ 输入先做 IK/FK 和软限位检查，再发送真正的 SDK MoveJ 指令。
返回成功需要本次 command ID 的到位事件、idle 状态、TCP/关节误差及连续稳定采样都符合要求。
失败或超时会调用 stop/reset。`tcp_type` 支持 `sdk_tcp`、`tcp_gripper_pose`、`tcp_cut_pos`；
命名工具在本次单臂笛卡尔 MoveJ 或 MoveL 请求中生效，完成或失败停止后确认 idle 并恢复原工具。
原有关节路径请求默认 `use_cartesian_target=false`，行为不变。

MoveAbsJ 只接收关节目标，保持控制器工具。MoveL 新增 tcp_type、use_current_pose、
use_current_orientation；省略目标或姿态时可直接使用所选工具的当前值。
MoveL 执行前用所选工具校验 checkPath 和目标软限位，成功要求本次指令 ID、到位事件、idle、
位置/姿态容差及连续稳定采样。姿态误差使用四元数旋转角，避免等价欧拉表示被误判。
三个运动 Service 均支持请求级反馈参数；MoveAbsJ 的零容差/超时、MoveL 的零容差/超时保留节点默认值。
MoveJ/MoveL 支持 resource_valid_until_unix_s，在 moveStart 前拒绝过期资源；
check_current_pose/expected_current_pose 可校验所选 TCP 的话题观测与 SDK 是否一致（5 mm/0.035 rad）。

`bo_project/open_box/scripts/bt_run.py` 已改为 ROS2 服务后端：先解析 mode 和 TCP 继承，
逐动作收集资源，再发送全局目标。`scripts/control.py` 默认检查全部步骤后真机执行，
首次动作前调用 initialize/right_arm；中断/超时调用 stop/right_arm 并等待原动作完成清理。
旧 C++ 工作者只供离线模拟，不会由上述真机入口启动。
`motion_demo_client.py` 仍是 sdk_tcp、mode 0 的简单示例，不替代完整行为树调度。

stop 服务应答代表 SDK stop/reset 已确认；活动的行为树运动回调会检测停止代次变化并退出，
调用端还需等待原请求返回以确认清理和工具恢复完成。此接口不替代硬件急停；
持续发送的外部 servo 话题控制应由其发布端停止发送。

MoveL 的 `expected_base_frame` 已与本机控制器及 `set_rotation.cpp` 的当前配置对齐。
接口仍拒绝基坐标不匹配的运动请求，不会自行更改控制器基坐标。

本次扩展了现有 Service 的请求字段。远程调用端需要构建同版 `rokae_interfaces`，
并在执行终端加载 `install/local_setup.bash`。

本机 FK→IK 回算验证显式设置 `use_elbow=true` 并传入 FK 返回的 `elbow`。
右臂当前姿态的 FK 返回 `has_elbow=false`，但传入返回的臂角后逆解成功；
不指定臂角时 SDK 返回逆解错误。示例 MoveJ 请求本身已经包含臂角约束。

## 只读检查

```bash
ros2 service list -t | rg /robot_02/upperlimb
ros2 action list -t | rg /robot_02/upperlimb
ros2 service call /robot_02/upperlimb/get_cartesian_state/right_arm \
  rokae_interfaces/srv/GetCartesianState '{}'
ros2 topic echo /robot_02/upperlimb/tcp_cut_pos/right_arm --once
ctest --test-dir build/rokae_driver --output-on-failure
```
