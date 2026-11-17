# PICO 双臂 ROS ServoL 摇操

只需启动操作时，使用 [简明启动文档：双臂＋头部＋腰部](PICO_QUICKSTART.md)。
只动头部和腰部时，用 `python3 tools/pico_servol_ros.py --no-arms`；仅需头腰 CSP 节点，
无需启动机械臂驱动。对应 Grip、方向扇区、死区、限位和停止条件保持相同，该模式不创建机械臂接口。

`pico_servol_ros.py` 监听 PICO 的 UDP 5050 数据包，将左右手柄的末端位姿和摇杆 X
发布到 `/robot_02/upperlimb/servol/jog/{left_arm,right_arm}`。ROS 驱动订阅该话题，
使用现有的 ServoL 实时控制器驱动机械臂。桥接程序本身不创建 SDK 连接。

首次使用或更新 A 键示教功能时构建消息和驱动（当前工作区已完成构建；腰部构建见下文）：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
colcon build --symlink-install --packages-up-to rokae_bringup
source install/local_setup.bash
```

终端一启动驱动（已有 `aide-arm-sdk` 运行时，先停止旧实例）：

```bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
bash tools/start_upperlimb.sh start_head_camera:=false start_servol:=true start_cartesian_drag_safe_service:=true
```

终端二加载相同 ROS 环境并启动桥接：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source install/local_setup.bash
python3 tools/pico_servol_ros.py
```

控制器软限位必须启用。启动桥接后先松开左右 Grip；按住某侧 Grip 时，桥接先
调用该侧 `/robot_02/upperlimb/power_on/{left_arm,right_arm}` 服务，确认上电成功后
再读取当前位姿并请求 ServoL，不需要另开终端手动调用上电服务。
驱动须启用 `start_initializer_service:=true`（统一启动脚本默认启用）。
按住某侧 Grip 控制对应机械臂，同时按住可控制双臂。松开 Grip 或超过 250 ms
未收到该侧有效 UDP 数据时，桥接程序向该侧 ServoL 发送停止请求。
驱动停止实时运动、关闭 ServoJoint 和状态接收后，切回 `NrtCommand`。
正常清理及指令排空完成后，桥接打印 `ServoL stopped; NrtCommand restored`；
若模式恢复或状态检查失败，会打印错误，不能视为正常完成。
上电或位姿查询期间松开 Grip 会取消后续遥操流程；已发出的上电请求可能仍完成，
但桥接不会因此发送运动目标。上电失败或运动保护报错后不会在持续按住 Grip 时
自动重试，须先松开再按。急停或安全门停止时，上电服务会拒绝请求。
摇杆映射如下，需按住对应侧 Grip，且该侧机械臂 ServoL 已就绪：

| 手柄输入 | 动作 |
| --- | --- |
| 左摇杆向左 / 向右 | 左臂臂角减小 / 增大，固定 5°/s |
| 右摇杆向左 / 向右 | 右臂臂角减小 / 增大，固定 5°/s |
| 右摇杆前推 / 后拉 | 头部 PHA（CAN 节点 1）抬头 / 低头，角度增大为抬头 |
| 左摇杆前推（上）/ 后拉（下） | 腰部抬起 / 弯腰，抬起时减小俯仰角 |

臂角使用 X 分量，左右各 ±30° 扇区包含边界；头腰使用 Y 分量，前后各 ±30°。
45° 斜推处于两组扇区之外，不改变臂角或头腰目标。有效轴分量的死区均为 0.15。
臂角范围仍为 ±180°，超出死区后固定 5°/s，不随摇杆幅度变化；回中保持臂角。
扇区角度在桥接脚本的 `ELBOW_JOYSTICK_HALF_ANGLE_DEG = 30.0`，臂角速度在
驱动 `include/rokae_driver/servol_math.hpp` 的 `kElbowSpeedDegS = 5.0`。
`ServoLJog.joystick_y` 为兼容现有消息保留名称，桥接现在向该字段写入筛选后的 X。
手柄位姿对末端的控制保持不变。

头腰使用现有 CSP 驱动的绝对角度接口，从位置反馈建立目标，不跳转到零位。
头部目标变化速度读取 `headmotor.yaml` 的 `pha_csp_velocity_deg_s`（当前 15°/s），
PHA 的 CSP 规划加、减速度分别为 `pha_csp_acceleration_deg_s2`、
`pha_csp_deceleration_deg_s2`，当前均为 60°/s²。PHU 和原 PP 参数保持原值。
腰部目标变化速度为现有默认 10°/s；实际速度由各 CSP 驱动限制。
PHA 范围从同一份配置换算，当前约 ±42.103°；腰部范围保持 0～50°（向前为正）。
头部目标领先实测位置最多为 150 ms 行程（15°/s 时 2.25°），腰部仍为 100 ms 行程；
腰部的反馈约束只限制后续推进，反馈回摆时保持既有目标，固定方向推杆时不因反馈
反向拉回目标；每次目标更新不超过对应摇杆速度允许的步长。此规则用于两种启动模式。
定时延误不会补发大步长目标。头部满速时按 60°/s² 减速的理论时间为 0.25 s，
理论制动行程为 1.875°；实际跟随和制动效果需实机确认。
腰部位置反馈使用深度 1 的 best-effort 订阅，避免可靠重传带来的旧反馈反复触发
1° 目标领先限制，形成推进／保持交替的顿挫；目标指令仍使用原可靠发布方式。
摇杆回中、超出扇区、松开 Grip、UDP 超时、非示教状态下机械臂退出遥操或位置反馈超时，
均停止该头腰轴的目标流并请求保持。头部按原驱动减速后保持使能；腰部保持最后
已发出的 CSP 设定值。普通停止后可再次推杆；故障后须先松开 Grip 并检查驱动状态。
头腰节点未启动或反馈无效时只提示等待，不发送该轴目标，机械臂遥操仍可使用。

头腰驱动需要另外启动：所有终端统一使用 `ROS_DOMAIN_ID=0`、
`ROS_LOCALHOST_ONLY=0` 和相同 RMW（本文使用 `rmw_fastrtps_cpp`）。
机械臂启动终端也先执行这三个 `export`，桥接终端同样设置。
CAN 接口沿用现有配置：头部 `can0`、腰部 `can1`，均为 1 Mbps。
先结束同一电机的 PP 节点、独立控制脚本及重复的 CSP 实例。

头部终端：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/headmotor/yiyou/yiyou_niic_ros
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# 首次使用 CSP、尚未构建该包时执行：
colcon build --base-paths . --packages-select yiyou_niic_ros --symlink-install
source install/local_setup.bash
ros2 launch yiyou_niic_ros yiyou_csp.launch.py
```

桥接在首次有效右 Y 输入时调用 `/headmotor/csp/start`，等待 `running` 和反馈后才发
PHA 目标，无需手动使能。现有头部 CSP 启动服务会使能 PHA 和 PHU，并保持各自
当前位置；桥接仅发布 PHA 目标，PHU 保持原位。若使用自定义 `head_config`，桥接
也应传入同一文件：`python3 tools/pico_servol_ros.py --head-config /绝对路径/headmotor.yaml`。

腰部终端（此次新增 hold/resume 服务，须重新构建并重启旧腰部节点）：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/waist_pitch_motor
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
colcon build --base-paths waist_pitch_ros --packages-select waist_pitch_ros --symlink-install
source install/local_setup.bash
ros2 run waist_pitch_ros waist_pitch_csp_node
```

腰部 CSP 节点启动时以当前位置使能保持，桥接用 `/resume` 接受新目标、用 `/hold`
保持；这些服务不改变原来的 `/stop` 失能含义。仅测试手臂时，可启动
`python3 tools/pico_servol_ros.py --no-head --no-waist`。

右手柄 A/Down（UDP 按键字节 bit 2）统一切换双臂笛卡尔拖动示教：
第一次按下双臂开始，第二次按下双臂停止；左 A 不触发示教。
启动时已按住的右 A 不触发，须先松开再按，且两手柄均需有新鲜 UDP 数据。
旧的 A 键回初始点功能已删除；桥接不再调用 MoveAbsJ 或保存初始点。

右 A 请求双臂示教后，先停止**左右两臂** ServoL，等待两臂均确认非实时模式恢复
和清理完成，才允许示教开始。示教准备、运行和停止清理期间，左右两臂的手柄
位姿和臂角遥操全部屏蔽，Grip 也不会给机械臂上电；头部和腰部仍由对应 Grip＋
摇杆独立控制。右 A 切换不会打断头腰目标。开始前确认双臂的启停服务均可用，
避免仅因缺少一侧服务而单臂启动；运行中任一臂结束或失败，都会请求另一臂停止。
两臂均完成清理前，双臂遥操继续屏蔽。

统一驱动需启用 `start_cartesian_drag_safe_service:=true`（`start_upperlimb.sh`
现已默认启用），重启统一驱动后生效，无需另开独立 SDK 示教节点。
共享连接接口如下，均使用 `std_srvs/srv/Trigger`：

| 操作 | 左臂 | 右臂 |
| --- | --- | --- |
| 开始示教 | `/robot_02/upperlimb/cartesian_drag_safe/start/left_arm` | `/robot_02/upperlimb/cartesian_drag_safe/start/right_arm` |
| 请求停止 | `/robot_02/upperlimb/cartesian_drag_safe/stop/left_arm` | `/robot_02/upperlimb/cartesian_drag_safe/stop/right_arm` |

每次示教默认 60 秒，使用 SDK `cxl/cartesian_drag_safe.cpp` 的正常物理拖动按钮路径、
默认 0 kg／零质心写入和监控规则。该原始 SDK 文件未修改；此负载行为适用于无附加工具情形。
右 A 只是启停双臂示教模式，**每条手臂仍需按住各自末端实体拖动按钮才可拖动，松开即停止拖动**。
集成显式传入 `enableDrag(..., false)`，禁止绕过实体按钮。SDK 示例要求进入拖动前
切到手动模式并 `setPowerState(false)`，因此保留该准备步骤，不承诺切换全程保持上电。
松开 PICO Grip 不会结束示教；再次按右 A、任一侧 UDP 超时或退出桥接会请求双臂停止，
60 秒到时也结束。单侧服务仍可供独立客户端使用，PICO 统一管理两侧会话。
停止服务的成功只表示接收停止请求，桥接等 `disableDrag` 确认及控制锁释放后才
完成切换；清理无法确认时保持两臂遥操屏蔽，须检查控制器后重新启动桥接。
全部示教结束后，头腰仍可继续操作；要恢复手臂遥操，重新松开、按住对应 Grip，
桥接会重新读取当前位置建立锚点，不沿用示教前的位姿目标。

`cartesian_drag_safe_node` 和 `/cartesian_drag_safe/start`（无左右后缀）仍是独立的
原 SDK 子进程包装，只能在统一驱动退出后单独使用，不与 PICO 集成接口混用。

示教数采使用专用 `/robot_02/upperlimb/cartesian_drag_safe/joint_states/{left_arm,right_arm}`
（`sensor_msgs/JointState`、弧度、sensor-data QoS），由示教循环直接发布其已读取的实测关节值，
约 20 Hz。普通状态发布回调在示教期间可能等待 SDK 锁，采集器不使用它作为示教反馈。
PICO 桥接另转发原右手柄按键字节到 `/robot_02/upperlimb/pico/buttons/right_arm`
（`std_msgs/UInt8`）；bit 3 为右 B，只由独立采集器解释，不改变 A、Grip 或运动逻辑。
16 维双臂＋头腰数采和三相机接入见 [ACT 数采说明](../../../../software/vla/act/xcore_act/README_ACT_XCORE.md#当前-ros-双臂示教数采)。

桥接沿用原 PICO 程序的 38 字节小端数据格式、手柄姿态转换、滤波、
30% 位移映射、25% 旋转映射和追踪跳变限制。初始机器人 TCP 位置由
`get_cartesian_state` 服务读取；该服务和 ServoL 驱动必须都已启动。
驱动公布其起始指令位姿后，桥接会再以该位姿对齐手柄锚点。
每侧会先等待自己的驱动就绪并发布起始位姿，等待期间保持初始目标且不传递摇杆输入。
出现 `ServoL driver ready; anchors aligned` 后，该侧才开始跟随手柄。
左右侧独立完成上述流程，可以同时按住两个 Grip，也可以在一侧遥操时加入另一侧。
驱动要求首个目标与当前 TCP 对齐，并在指令超时、关节越界或追踪异常时退出实时控制。
若出现 `joint J6 reached the configured safety margin`，表示当前运动触及 J6 的
软限位安全余量；驱动日志会包含实测角度和允许范围。这不是单纯未上电，重新上电
不能消除限位问题，应先用示教器检查并将对应关节调整到允许范围内，再尝试遥操。

请勿同时运行占用 UDP 5050 的 `receive.py` 或 SDK 版 PICO 摇操程序，也不要
同时向同一机械臂的普通 ServoL 话题和 jog 话题发布控制目标。
