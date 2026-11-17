# Rokae ROS 2 workspace

当前机械臂 SDK 接口已恢复使用。引用的 `运动类型演示.yaml` 可通过包内现有接口调用；
构建、启动、接口映射和只读检查见 [本机 SDK 节点说明](SDK_NODES.md)。
`bash tools/start_upperlimb.sh` 统一启动双臂状态、SDK 运动接口和头部 RGB-D 相机；
`start_state_publisher.sh` 和 `start_sdk_nodes.sh` 是该入口的兼容别名。

> **历史状态说明（2026-09-13）**：此前本工作空间曾停止使用 ROS 接入。
> 当前 `rokae_driver`、`rokae_interfaces`、`rokae_bringup` 已重新用于本机 SDK 接口；
> 下文其他旧应用接入与历史示例需以当前接口说明核对。
> `src/rokae_motion` 应用程序仍有参考价值，保留其行为树、任务编排和应用逻辑供参考；其依赖的旧 ROS 接口同样已废弃。详见 [源码状态说明](src/README.md)。

完整的 Topic、Service、Action、参数、单位和错误行为说明见
[`docs/ROS2_INTERFACE_REFERENCE.md`](docs/ROS2_INTERFACE_REFERENCE.md)。

## 当前机器：统一启动双臂状态与 SDK 接口

工作区通信域为 `ROS_DOMAIN_ID=0`，允许局域网通信 `ROS_LOCALHOST_ONLY=0`，
由 `rokae_bringup` 的环境 hook 和统一启动脚本设置。
机械臂节点、Topic、Service 和 Action 统一使用 `/robot_02/upperlimb` 前缀。
局域网订阅端也需设置 `ROS_DOMAIN_ID=0`、`ROS_LOCALHOST_ONLY=0`；
以下位姿话题采用标准消息类型，无需在订阅端编译机械臂工作区。

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-up-to rokae_bringup \
  --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
bash tools/start_upperlimb.sh
```

该脚本统一启动状态发布、MoveAbsJ、MoveL、MoveJByPath、FK/IK 和初始化接口，
以及头部彩色、原始深度、对齐到彩色的深度话题。完整话题名见 [SDK_NODES.md](SDK_NODES.md)。
相机已运行时复用，未运行时先启动并确认三路图像有有效帧；
Ctrl+C 会关闭本入口启动的相机，保留复用的外部相机。
仅启动双臂时可传 `start_head_camera:=false`；相机工作区需要先完成构建。
启动时不自动初始化、上电或运动。已有后台 `aide-arm-sdk` 服务运行时，
先执行 `systemctl --user stop aide-arm-sdk` 再在终端前台启动。
它优先使用已安装的
Cyclone DDS，否则使用 Fast DDS；已显式设置的 `RMW_IMPLEMENTATION` 会保留。
此构建命令包含 bringup 的工作区内依赖；上述接口不需要旧文档中的
`dbstest_ws` 或底盘接口。
按 `Ctrl+C` 停止。在另一个终端查看左右手末端位姿：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
# 当前机器已安装的中间件；两个终端使用相同设置。
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS/install/local_setup.bash
ros2 topic echo /robot_02/upperlimb/tcp_gripper_pose/left_arm --once
ros2 topic echo /robot_02/upperlimb/tcp_gripper_pose/right_arm --once
ros2 topic hz /robot_02/upperlimb/tcp_gripper_pose/left_arm
```

位姿为 `geometry_msgs/msg/PoseStamped`，位置单位 m、姿态为四元数，配置频率
为 20 Hz。当前 `tcp_gripper_pose` 定义为沿法兰局部 Z 轴负方向偏移
`0.11 m` 的工具末端位姿：`p_tcp = p_flange + R_flange * [0, 0, -0.11]`，
偏移随手腕姿态旋转，姿态与法兰相同。法兰和夹爪位姿均以外部参考系表达；
法兰位姿可查看 `/robot_02/upperlimb/flange_pose/{left_arm,right_arm}`。

刀具位姿话题为 `/robot_02/upperlimb/tcp_cut_pos/{left_arm,right_arm}`，同样使用
`geometry_msgs/msg/PoseStamped`，频率为 20 Hz。以法兰局部坐标系为起点，先绕
局部 Z 轴正向（逆时针）旋转 15°，再沿旋转后的局部 X 轴负方向平移 0.11 m。
其变换为 `T_ref_cut = T_ref_flange * Rz(15°) * Tx(-0.11 m)`，发布的位置和姿态
均以对应的外部参考系表达，时间戳与法兰位姿相同。

This workspace keeps the application-owned ROS 2 sources separate from the
vendor xCoreSDK. The SDK still supplies its headers, Eigen copy and binary
library.

Expected Linux layout:

```text
/home/niic/
|-- aide/hardware/arm/
|   |-- xCoreSDK-v0.7.1.ar_6/
|   `-- xCoreROS/
|-- dbstest_ws/                 # seer_interfaces / chassis integration
`-- rokae_ws -> aide/hardware/arm/xCoreROS/
```

原工作区已整体迁移到 `/home/niic/aide/hardware/arm/xCoreROS`，保留 Git 历史、
本地修改和编译产物。`/home/niic/rokae_ws` 是兼容软链接，供现有编译缓存、
安装脚本和旧文档引用使用；重新生成 build/install 之前请保留该链接。

## Build

```bash
cd /home/niic/aide/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
source /home/niic/dbstest_ws/install/setup.bash

# This matches the current robot computer.
export ROKAE_SDK_ROOT=/home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6

# The driver automatically migrates the former /home/niic/cxl SDK cache value.
colcon build --symlink-install
source install/local_setup.bash
```

The driver selects `lib/Linux/aarch64` or `lib/Linux/x86_64` from the target
processor reported by CMake. Check the robot computer architecture with:

```bash
uname -m
```

For a custom SDK, unsupported target or cross-compiled target, configure the
driver package explicitly before building the full workspace:

```bash
colcon build --symlink-install --packages-select rokae_driver --cmake-args \
  -DROKAE_SDK_ROOT="/home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6" \
  -DXCORESDK_LIBRARY=/absolute/path/to/libxCoreSDK.a
colcon build --symlink-install
```

## Start

The launch file starts the state publisher, MoveAbsJ action server, MoveL
service, Linker Hand end-CAN service, robot-initialization service and the
vision trigger/target cache. Starting the launch file alone does not send a
motion or hand command, trigger vision or power on either robot.

All arm SDK interfaces run inside the single `ros_dual_arm_driver` process.
That process owns exactly two `rokae::ArRobot` objects (one left and one right)
and shares them between state, Jacobian, initialization, MoveAbsJ, MoveL and
hand-CAN interfaces. Per-arm SDK access is serialized, and MoveAbsJ, MoveL and
initialization share a per-arm command lock so independent ROS interfaces
cannot control the same arm concurrently.

```bash
source /opt/ros/humble/setup.bash
source /home/niic/dbstest_ws/install/setup.bash
source /home/niic/aide/hardware/arm/xCoreROS/install/local_setup.bash
ros2 launch rokae_bringup dual_arm.launch.py
```

The workspace selects `rmw_cyclonedds_cpp` when `RMW_IMPLEMENTATION` is not
already set. This avoids a Fast DDS client-creation hang observed on the robot
computer while still allowing an explicit operator override.

The deployed arm network values, matching the SDK programs under `cxl/`, are:

| Arm | Controller IP | Robot-computer IP |
| --- | --- | --- |
| Left | `192.168.1.160` | `192.168.1.10` |
| Right | `192.168.2.160` | `192.168.2.10` |

Both robot-computer addresses must be assigned to the active Ethernet interface
before starting any Rokae driver node. The authoritative runtime copy is the
`rokae_bringup/config/dual_arm.yaml` file installed by the workspace.

Optional launch switches:

```bash
# State topics only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_move_server:=false start_movel_service:=false \
  start_hand_service:=false start_initializer_service:=false \
  start_cartesian_teach_service:=false \
  start_servoj:=false start_go_home_service:=false \
  start_movej_by_path_service:=false

# Action server only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_movel_service:=false \
  start_hand_service:=false start_initializer_service:=false \
  start_cartesian_teach_service:=false \
  start_servoj:=false start_go_home_service:=false \
  start_movej_by_path_service:=false

# MoveL services only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_move_server:=false \
  start_hand_service:=false start_initializer_service:=false \
  start_cartesian_teach_service:=false \
  start_servoj:=false start_go_home_service:=false \
  start_movej_by_path_service:=false

# ServoJ realtime topics only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_move_server:=false \
  start_movel_service:=false start_hand_service:=false \
  start_cartesian_teach_service:=false \
  start_initializer_service:=false start_go_home_service:=false \
  start_movej_by_path_service:=false

# Go-home services only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_move_server:=false \
  start_movel_service:=false start_hand_service:=false \
  start_cartesian_teach_service:=false \
  start_initializer_service:=false start_servoj:=false \
  start_movej_by_path_service:=false

# Linker Hand services only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_move_server:=false \
  start_movel_service:=false start_initializer_service:=false \
  start_cartesian_teach_service:=false \
  start_servoj:=false start_go_home_service:=false \
  start_movej_by_path_service:=false

# Robot initialization service only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_move_server:=false \
  start_movel_service:=false start_hand_service:=false \
  start_cartesian_teach_service:=false \
  start_servoj:=false start_go_home_service:=false \
  start_movej_by_path_service:=false

# Vision trigger and target cache only
ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=false start_move_server:=false \
  start_movel_service:=false start_hand_service:=false \
  start_cartesian_teach_service:=false \
  start_initializer_service:=false start_servoj:=false \
  start_go_home_service:=false start_movej_by_path_service:=false
```

The action names are:

```text
/robot_02/upperlimb/move_absj/left_arm
/robot_02/upperlimb/move_absj/right_arm
```

All arm APIs use `/robot_02/upperlimb/<function>/<target>`: `robot_02` identifies
the robot, `upperlimb` identifies the subsystem, and the target is
`left_arm`, `right_arm` or `dual_arm` where applicable.

The multi-waypoint joint-space services are:

```text
/robot_02/upperlimb/movej_by_path/left_arm
/robot_02/upperlimb/movej_by_path/right_arm
/robot_02/upperlimb/movej_by_path/dual_arm
```

They use `rokae_interfaces/srv/MoveJByPath`. A single-arm request flattens
waypoints into `joint_positions` (`N * 7` values); a dual-arm request uses
`left_joint_positions` and `right_joint_positions` with equal waypoint counts.
The driver queues the complete path in one SDK `moveAppend` call (2–100
waypoints), validates limits and adjacent-point jumps, and uses the configured
`zone_mm` to blend points. The default `zone_mm=5.0` gives continuous motion;
set it to zero for exact stops during testing.

The realtime joint-position command topics are:

```text
/robot_02/upperlimb/servoj/left_arm   rokae_interfaces/msg/ServoJ
/robot_02/upperlimb/servoj/right_arm  rokae_interfaces/msg/ServoJ
/robot_02/upperlimb/servoj/dual_arm   rokae_interfaces/msg/DualArmServoJ
```

Single-arm messages contain `enable` and seven joint positions in radians.
The dual-arm message contains `enable` plus left and right seven-joint arrays.
With `enable=true`, publish continuously at the configured 50 Hz and begin at
the current `joint_states`; `enable=false` stops the session. A 0.10 s command
watchdog, controller soft limits, a 0.02 rad per-frame step limit and the same
per-arm command locks used by MoveAbsJ/MoveL protect the interface. The driver
does not power on the arms automatically.

The linear-motion services are:

```text
/robot_02/upperlimb/move_l/left_arm
/robot_02/upperlimb/move_l/right_arm
/robot_02/upperlimb/move_l_relative/left_arm
/robot_02/upperlimb/move_l_relative/right_arm
/robot_02/upperlimb/get_cartesian_state/left_arm
/robot_02/upperlimb/get_cartesian_state/right_arm
```

`move_l` is also used by visual motion modes. It requires an absolute TCP
position and complete absolute RPY orientation, and preserves the current
seven-axis configuration unless `use_elbow=true`.

The Linker Hand services are:

```text
/robot_02/upperlimb/control_hand/left_arm
/robot_02/upperlimb/control_hand/right_arm
```

The `op.cpp`-style initialization service is:

```text
/robot_02/upperlimb/initialize/dual_arm
```

The fixed home-position services are:

```text
/robot_02/upperlimb/go_home/left_arm
/robot_02/upperlimb/go_home/right_arm
/robot_02/upperlimb/go_home/dual_arm
```

They use `std_srvs/srv/Trigger` and MoveAbsJ. The configured joint targets are
the left and right origin arrays marked in `cxl/moveabsj.cpp`. A request causes
real robot motion, does not power on the arms, and shares the same per-arm
command locks and runtime safety checks as the other motion interfaces.

```bash
ros2 service call /robot_02/upperlimb/go_home/left_arm std_srvs/srv/Trigger
ros2 service call /robot_02/upperlimb/go_home/right_arm std_srvs/srv/Trigger
ros2 service call /robot_02/upperlimb/go_home/dual_arm std_srvs/srv/Trigger
```

These are empty `Trigger` requests. Joint targets are fixed in the driver
configuration and are not supplied on the command line.

One request initializes both arms in order: connect, select
`NrtCommand`, select automatic mode, power on, then verify that the final
power state is `on`. It does not send a motion command. It can be called
independently with:

```bash
ros2 service call /robot_02/upperlimb/initialize/dual_arm std_srvs/srv/Trigger "{}"
```

The initialization node uses `192.168.1.10` and `192.168.2.10` from the
`/robot_02/upperlimb/initializer` section of `dual_arm.yaml`, matching the currently
deployed `cxl/op.cpp`.

They expose the `control_hand.cpp` commands `open`, `half`, `close`,
`position`, `motors`/`joints`, `speed` and `pressure`. Position and speed
values are integer Linker Hand protocol bytes in `[0, 255]`. For example:

```bash
ros2 service call /robot_02/upperlimb/control_hand/left_arm \
  rokae_interfaces/srv/ControlHand \
  "{command: motors, values: [255, 160, 69, 69, 69, 69]}"
```

The request position is absolute in the controller's external reference frame,
in metres. `orientation_rpy` is the required absolute XYZ Euler orientation in
radians. `use_elbow=false` preserves the current arm configuration; when
enabled, `elbow` is in radians. `speed_mm_s` is mm/s and `zone_mm` is the
blending radius in mm. For example:

```bash
ros2 service call /robot_02/upperlimb/move_l/right_arm rokae_interfaces/srv/MoveL \
  "{position: [0.280941, -0.314533, -0.489353], \
orientation_rpy: [-2.761879, 0.340152, -0.746153], \
use_elbow: false, elbow: 0.0, speed_mm_s: 50.0, zone_mm: 0.0}"
```

The driver rejects a request that exceeds the configured per-command
translation, orientation, speed or zone limits. Inspect and tune the
`/robot_02/upperlimb/cartesian_motion` section of `dual_arm.yaml` for the actual cell.

The Cartesian hand-guiding services are:

```text
/robot_02/upperlimb/cartesian_teach/left_arm
/robot_02/upperlimb/cartesian_teach/right_arm
/robot_02/upperlimb/cartesian_teach/stop
```

The start request has one `enable_drag_button` boolean. Its default `false`
value requires holding the physical end drag button; `true` enables direct
dragging without holding it. Sessions have a fixed 300-second maximum and can
be ended early through the shared Trigger stop service. The implementation
switches the selected arm to manual mode and powers it off as required by the
SDK example; it does not automatically restore automatic mode or power.

### PICO A 键拖动示教

`tools/start_upperlimb.sh` 默认启用 `start_cartesian_drag_safe_service:=true`，复用
统一驱动的 SDK 连接。PICO 右 A 键统一开始／停止双臂笛卡尔拖动示教，左 A 不触发示教，
每次默认 60 秒。任意一臂示教时，两臂的手柄位姿和臂角遥操均屏蔽，头腰继续独立可控。
恢复手臂遥操需等待示教停止确认，再松开、按住 Grip。

接口为 `/robot_02/upperlimb/cartesian_drag_safe/{start,stop}/{left_arm,right_arm}`，
类型 `std_srvs/srv/Trigger`；start 响应在示教与清理结束后返回，stop 响应只确认停止请求。
该集成显式使用 `enableDrag(..., false)`，各臂必须按住自己的实体按钮才可拖动。
SDK 要求先切到手动模式并下电再开启拖动，此准备步骤保留；不能保证切换全程保持上电。
沿用原 SDK 的 0 kg／零质心设置和监控阈值，仅适用于原示例的无附加工具条件。
完整启动步骤见 [PICO 简明文档](tools/PICO_QUICKSTART.md)。

### 直接封装原始 cartesian_drag_safe.cpp

`rokae_driver` 另提供 `cartesian_drag_safe` 和 `cartesian_drag_safe_node`。
前者直接编译同级 SDK 的 `cxl/cartesian_drag_safe.cpp`，后者通过 ROS 服务
管理该程序的启动、停止。它们与上面的 `cartesian_teach` 服务是独立入口。
原始 C++ 文件、工具负载写入、网络配置、拖动按钮要求、监测逻辑均不改动：
当前源码会写入 0 kg、零质心，TCP/关节速度阈值分别为 1 m/s、2 rad/s，
起点位移/旋转范围为 0.50 m、90°，并在终端持续输出 TCP 位姿和力/力矩。
这些值以原始 C++ 源码为准，更新源码后需要重新编译 ROS 包。

编译（只编译，不连接机械臂）：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --symlink-install \
  --packages-up-to rokae_driver
source install/local_setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

停止其他控制同一机械臂的程序后，启动 ROS 节点。此入口无需先启动统一驱动，
节点启动后等待 `start` 服务，不会自动连接或开启拖动：

```bash
ros2 run rokae_driver cartesian_drag_safe_node --ros-args \
  -p arm:=right -p duration_seconds:=60.0
```

`arm` 支持 `left`、`right`；`duration_seconds` 支持 1～600 秒。
在另一个终端加载相同环境后，开始示教：

```bash
source /opt/ros/humble/setup.bash
source /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS/install/local_setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 service call /robot_02/upperlimb/cartesian_drag_safe/start \
  std_srvs/srv/Trigger "{}"
```

启动服务立即返回是否成功创建 SDK 进程；是否成功开启拖动、故障原因和退出码
以节点终端为准。与原程序相同，等模式切换约 2 秒后按住物理拖动按钮移动。
提前结束时调用：

```bash
ros2 service call /robot_02/upperlimb/cartesian_drag_safe/stop \
  std_srvs/srv/Trigger "{}"
```

停止服务发送 SIGINT，返回表示停止请求已发送；等待节点终端确认拖动关闭。
节点 Ctrl+C 退出也会发送 SIGINT 并等待原程序完成清理，不会强杀 SDK 进程。
到时结束后节点仍可接收下一次启动请求。该封装不自动保存点位或录制轨迹。

也可以仅通过 ROS 包入口直接运行原命令行程序（立即执行，无 ROS 服务）：

```bash
ros2 run rokae_driver cartesian_drag_safe right 60
```

The state topics are:

```text
/robot_02/upperlimb/joint_states/left_arm
/robot_02/upperlimb/flange_pose/left_arm
/robot_02/upperlimb/flange_state/left_arm
/robot_02/upperlimb/tcp_gripper_pose/left_arm
/robot_02/upperlimb/tcp_gripper_state/left_arm
/robot_02/upperlimb/tcp_cut_pos/left_arm
/robot_02/upperlimb/jacobian/left_arm
/robot_02/upperlimb/joint_states/right_arm
/robot_02/upperlimb/flange_pose/right_arm
/robot_02/upperlimb/flange_state/right_arm
/robot_02/upperlimb/tcp_gripper_pose/right_arm
/robot_02/upperlimb/tcp_gripper_state/right_arm
/robot_02/upperlimb/tcp_cut_pos/right_arm
/robot_02/upperlimb/jacobian/right_arm
```

Each `flange_pose`, `tcp_gripper_pose` and `tcp_cut_pos` topic uses
`geometry_msgs/msg/PoseStamped`. The matching `flange_state` and
`tcp_gripper_state` topics use
`rokae_interfaces/msg/TcpState`: it contains the same stamped position and
quaternion plus `orientation_rpy=[roll,pitch,yaw]` in radians.

These replace the old `tcp_pose` / `tcp_state` topics; subscribers must migrate.
Flange poses exclude the controller's tool offset and are expressed in the active
external reference frame (`left_external_ref` / `right_external_ref`).
The gripper TCP uses a flange-local offset of `[0, 0, -0.11]` metres:
`gripper.xyz = flange.xyz + R_ref_flange * [0, 0, -0.11]`, with identical
orientation and timestamp. The offset rotates with the wrist and the resulting
position is expressed in the same external reference frame as the flange.
Publishing this point does not change SDK tool settings or MoveL/ServoL command
semantics.

The cutter TCP starts from the flange frame, rotates +15 degrees about local Z,
then translates -0.11 metres along the rotated local X axis:
`T_ref_cut = T_ref_flange * Rz(15 degrees) * Tx(-0.11)`.
Thus `R_ref_cut = R_ref_flange * Rz(15 degrees)` and
`p_ref_cut = p_ref_flange + R_ref_cut * [-0.11, 0, 0]`.
Both the position and rotated orientation are published in the flange's external
reference frame with the same timestamp.

Each Jacobian topic uses `std_msgs/msg/Float64MultiArray`. Its `data` field is
the xCoreSDK row-major `6 x 7` flange Jacobian: rows are
`vx, vy, vz, wx, wy, wz`, and columns are `joint_1 ... joint_7`. The layout
dimensions are labelled `twist` (size 6, stride 42) and `joint` (size 7,
stride 7). The matrix is expressed relative to each robot's base frame.

`publish_jacobian` is enabled for both state-publisher instances in
`dual_arm.yaml`. It can disable only the Jacobian publishers while leaving
joint and TCP state publishing active. The publisher skips model computation
when the topic has no subscribers. The launch argument
`start_state_publisher:=false` instead disables the complete state publisher,
including joint, TCP and Jacobian topics.

Before the first motion test, use one arm, a small target delta, low speed, a
clear workspace and an accessible E-stop. The values in
`rokae_bringup/config/dual_arm.yaml` are initial limits, not a replacement for
the controller's safety configuration.

## Motion programs

`rokae_motion` is an `ament_python` package. Its installed entry points are:

```text
bt_runner
bt_control
vision_target_server
chassis_navigation
```

### Behavior-tree runner

The behavior-tree runner uses the following YAML hierarchy:

```text
task -> location -> behavior -> action
```

The supported action types are `move_absj`, `move_l`, `move_l_relative`,
`vision`, `move_l_vision`, `navigate`, `hand` and `wait`.
The Python files have deliberately separate responsibilities:

```text
run_bt.py             command-line arguments and program entry point
control.py            fixed workflow groups, steps and repeat counts
bt_runner_config.py   dry-run, confirmation and failure policy
bt_executor.py        YAML parsing, validation, filtering and execution flow
bt_actions.py         concrete arm, hand and Wait behaviors
executor_services.py  low-level ROS 2 action and service calls
chassis_navigation.py scheduler-command to Seer station bridge
vision_motion.py      pure motion_mode 1-6 target calculations
vision_target_server.py  detector trigger and in-memory target cache
```

Validate and display the installed safe template without moving:

```bash
ros2 run rokae_motion bt_runner
```

Use a YAML file outside the package:

```bash
ros2 run rokae_motion bt_runner /path/to/tree.yaml
```

Select any hierarchy level for focused debugging:

```bash
ros2 run rokae_motion bt_runner /path/to/tree.yaml \
  --task-id task_001 \
  --location-id location_001 \
  --behavior-id behavior_001 \
  --action-id action_002
```

Dry-run is always the default. Actual execution requires all three:

1. Valid values for every enabled motion action.
2. `execution_locked: false` at the YAML root.
3. The `--execute` option followed by typing `EXECUTE` interactively.

```bash
ros2 run rokae_motion bt_runner /path/to/checked_tree.yaml \
  --action-id action_002 --execute
```

For a fixed multi-step workflow, edit `BEHAVIOR_TREE_PATH` and the `WORKFLOW`
list in `rokae_motion/control.py`, then run it without an extra execution
option:

```bash
ros2 run rokae_motion bt_control
```

Before its first enabled action, `bt_control` automatically calls
`/robot_02/upperlimb/initialize/dual_arm` to initialize and power on both arms. If either arm
cannot be initialized, the workflow stops without sending an action. The
bringup launch must therefore be running with
`start_initializer_service:=true` (the default). This automatic step applies
only to `bt_control`; the general-purpose `bt_runner` does not power on the
robots automatically.

The source file can also be run directly after sourcing ROS and the workspace:

```bash
python3 /home/niic/aide/hardware/arm/xCoreROS/src/rokae_motion/rokae_motion/control.py
```

Use `--dry-run` when only a preview is needed:

```bash
ros2 run rokae_motion bt_control --dry-run
```

`--dry-run` returns before creating the ROS executor, so it neither calls the
initialization service nor sends a robot command.

Each workflow group supports `name`, `repeat` and ordered `steps`. A step must
contain `task_id` and may narrow the selection with `location_id`,
`behavior_id` and `action_id`. Actual workflow execution still respects the
YAML `execution_locked` safeguard, but `bt_control` does not require the
interactive `EXECUTE` input. `bt_runner` remains dry-run by default and still
requires its `--execute` option.

An action with `enabled: false` is displayed but skipped. This permits the
checked-in template to retain null target placeholders safely. An action with
`optional: true` logs a runtime failure and continues; all other failures stop
the selected sequence. `--continue-on-action-fail` can be used during
deliberate debugging to continue after any action failure.

A relative linear action offsets the current TCP position and preserves its
seven-axis elbow. Only the fields that need to change have to be written.
Omitted `dx`, `dy` or `dz` fields mean zero displacement. Omitted `rx`, `ry`
or `rz` fields preserve the corresponding current orientation component;
written orientation fields are absolute RPY targets:

```yaml
- id: lift_arms
  name: dual_arm_relative_lift
  type: move_l_relative
  relative_offset:
    rotate_with_waist: false
    left:
      dx: 0.02
      rz: -0.7
    right:
      dz: 0.02
  speed_mm_s: 30.0
  zone_mm: 0.0
```

Offsets are metres in each controller's external-reference frame. Unlike the
reference humanoid project, Rokae has no waist frame, so
`rotate_with_waist: true` is rejected. Orientation values are absolute XYZ
Euler angles in radians, not increments added to the current orientation. If
only `left` or `right` is present, only that arm receives a service request.

### Seer station navigation

The `navigate` behavior uses the same station IDs as the Seer map:

```text
LM1 -> ARRIVE_HOME
LM2 -> ARRIVE_A
LM3 -> ARRIVE_B
```

Start the bridge by itself:

```bash
ros2 run rokae_motion chassis_navigation
```

Or include it in the bringup launch:

```bash
ros2 launch rokae_bringup dual_arm.launch.py \
  start_chassis_navigation:=true
```

The Seer driver must provide the `/seer/navigate` Action. Navigation belongs
to the `location` entry and runs once before that location's behaviors:

```yaml
locations:
  - id: navigate_to_a
    name: navigate_chassis_to_a
    command: LM2
    behaviors:
      # The location's behaviors run after ARRIVE_A.
      ...
```

The stable navigation type and connection/response timeouts are stored in
`rokae_motion/config/navigation.yaml`, so behavior trees only contain the
per-location command.

Use `--skip-nav` with either `bt_runner` or `bt_control` to omit all enabled
location navigation. A timeout, navigation failure or Ctrl+C requests
`/bt_navigation_server/cancel_navigation`, with the Seer cancellation service
used as a fallback.

### Vision target cache and motion modes

The vision target node has been removed from `dual_arm.launch.py` and boot
autostart. Its source remains available for explicit standalone use; the
interfaces below are not created by the dual-arm launch.

The ROS 2 port keeps the reference project's topic and service names:

```text
subscribed JSON:
  /yolo_vision/front_points_base_json
  /yolo_vision/wall_angle
  /yolo_vision/mode5_points_json

subscribed ArUco pose:
  /tool/pose

published triggers:
  /yolo_vision/control
  /yolo_vision/target_labels
  /aruco/enable

services:
  /bt_target_server/trigger_detect
  /bt_target_server/get_target
  /bt_target_server/set_offset
```

This component stores parsed detector JSON, named points, box left/right
points, scalar offsets and ArUco tool poses in memory. It does not save
RGB/depth images and its cache is cleared when the node restarts. The detector
itself must be started separately.

Trigger and cache a pair of points manually:

```bash
ros2 service call /bt_target_server/trigger_detect \
  rokae_interfaces/srv/GetVisionTarget \
  "{source: yolo, key: sample, trigger_value: 2, labels: [tray], motion_mode: 0, \
point_names: [left, right]}"
```

Read the same cached points without triggering the detector:

```bash
ros2 service call /bt_target_server/get_target \
  rokae_interfaces/srv/GetVisionTarget \
  "{key: sample, motion_mode: 1, point_names: [left, right]}"
```

For the ArUco tool detector, start `aruco_tool_launch.py` separately and use:

```bash
# Reuse an existing /camera color/depth publisher (default).
ros2 launch aruco_scanner aruco_tool_launch.py

# Or start this workspace's configured RealSense camera. It publishes aligned
# /camera/color/image_raw, /camera/depth/image_rect_raw and camera_info.
ros2 launch aruco_scanner aruco_tool_launch.py start_camera:=true
```

The installed vision stack requires a NumPy 1.x version compatible with the
NVIDIA PyTorch build and an OpenCV release that still supports NumPy 1.x. The
tested versions on this computer are NumPy 1.26.4 and OpenCV 4.11.0; these
constraints are recorded in `aruco_scanner/setup.py`.

Use the following behavior-tree action after the detector is running:

```yaml
- id: detect_tool
  name: detect_target
  type: vision
  source: aruco
  echo_topic: /tool/pose
  pub_topic: /aruco/enable
  key: "1-1-1-1"
  trigger_value: 1
  points: [tool]
```

`pub_topic` and `echo_topic` select the actual trigger publisher and result
subscription used by `vision_target_server`. It publishes `1`, waits for the
first result, continues sampling for one second, caches the newest complete
result, and then publishes `0`.

The two-point box detector uses `source: box_grab_points`,
`pub_topic: /box/enable`, `echo_topic: /box_grab_points`, and
`points: [left_center, right_center]`. Its `BoxGrabPoints` message must use
`base_link`.

Use `type: vision` only to trigger and cache data, followed by
`type: move_l_vision` to consume it. `motion_mode` belongs only on the
`move_l_vision` action. Its six imported modes are:

1. Two named points become independent absolute left/right targets.
2. One point becomes the desired dual-arm midpoint while preserving the
   current arm formation.
3. One point plus per-arm offsets; only written arms move.
4. One target point minus one current point is applied to both arms.
5. Independent target-current point differences are applied per arm.
6. A cached scalar such as `y_offset` corrects a selected axis and hand set.

Modes 4, 5 and 6 support a skip threshold. Modes 2 through 6 preserve current
orientation unless an absolute `rx`, `ry` or `rz` is written. Mode 1 can add
per-arm `dx/dy/dz` offsets and absolute RPY from `relative_offset`; omitted
orientation axes fall back to the active entry in
`config/vision_offsets.yaml`.

The full locked template for all six modes is
`behavior_trees/examples/全部动作示例.yaml`. This Rokae port intentionally
does not include the reference project's whole-body IK, waist motion or waist
coordinate rotation. `rotate_with_waist: true` is rejected during YAML
validation.

A hand action can address one or both hands. The explicit `targets` form is
recommended because it makes the side and six-motor order clear:

```yaml
- id: set_hand_positions
  name: set_dual_hand_positions
  type: hand
  command: motors
  targets:
    # M1-M6: thumb bend, thumb side swing, index, middle, ring, little finger
    left: [255, 160, 69, 69, 69, 69]
    right: [255, 160, 69, 69, 69, 69]
```

Commands `open`, `half`, `close` and `pressure` use a `hands` list:

```yaml
- id: open_left
  type: hand
  command: open
  hands: [left]
```

For compatibility with `zj_robot_bt_action`, `q` may contain exactly 12
values: left M1-M6 followed by right M1-M6. Unlike that humanoid project's
joint values, Rokae Linker Hand values are integer CAN protocol bytes in
`[0, 255]`, not radians:

```yaml
- id: close_both
  type: hand
  q: [69, 69, 69, 69, 69, 69,
      69, 69, 69, 69, 69, 69]
```

The installed templates include
`rokae_motion/behavior_trees/examples/展会动作.yaml` and
`rokae_motion/behavior_trees/examples/全部动作示例.yaml`. The complete
example is execution-locked and covers every supported action type, including
hand speed, presets, independent motor positions, the compatible 12-value
`q` layout and pressure reads.
Version-controlled programs belong under `behavior_trees/examples`,
`behavior_trees/debug` or `behavior_trees/production`. Temporary on-site
programs can live under `/home/niic/aide/hardware/arm/xCoreROS/programs` and be passed to
`bt_runner` by absolute path.
