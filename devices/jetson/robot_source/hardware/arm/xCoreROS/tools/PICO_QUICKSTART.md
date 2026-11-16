# PICO 摇操启动（双臂＋头部＋腰部俯仰）

全身模式按编号顺序启动；只动头腰按下一节操作。各启动终端保持打开。使用当前已编译环境，机械臂命名空间为 `/robot_02/upperlimb`。
先接通机器人和头腰电机电源、解除急停；结束旧摇操、头腰 PP/CSP 节点及独立电机控制脚本。

## 只动头部和腰部（不启动手臂）

完成第 1 步 CAN 配置，**跳过第 2 步机械臂**，按第 3、4 步分别启动头部和腰部 CSP 节点。
先在旧 PICO 终端按 `Ctrl+C`，再在第三个终端启动：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source install/local_setup.bash
python3 tools/pico_servol_ros.py --no-arms
```

PICO 发送目标仍为机器人电脑 IP、UDP 5050。先松开左右 Grip 各一次，再持续按住：
**右 Grip＋右摇杆前后控制抬头／低头；左 Grip＋左摇杆前后控制腰部抬起／弯腰**。
摇杆回中或松开对应 Grip 后请求停止并保持。该模式不创建机械臂控制接口，A 键不触发手臂示教。
退出时先松开 Grip，按 `Ctrl+C` 结束 PICO 桥接、等待停止确认，再退出头腰节点。

## 1. 配置 CAN（每次开机一次）

头部 `can0`、腰部 `can1` 均为 1 Mbps；已经配置好可跳过。配置时头腰节点应处于关闭状态。

```bash
cd /home/niic/aide/AIDE_robot_main
sudo bash hardware/headmotor/yiyou/yiyou_niic_ros/scripts/setup_can.sh can0 1000000
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

## 2. 终端一：启动左右臂

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
bash tools/start_upperlimb.sh \
  start_head_camera:=false \
  start_servol:=true \
  start_movel_service:=true \
  start_moveabsj_service:=true \
  start_initializer_service:=true \
  start_cartesian_drag_safe_service:=true
```

若提示已有机械臂驱动，先在旧驱动终端按 `Ctrl+C`；若是后台服务，执行 `systemctl --user stop aide-arm-sdk`，再运行上述命令。
该终端必须保持运行。PICO 若持续提示 `enable start_servol:=true...`，先松开两个 Grip，检查此终端是否已退出或报错，再启动驱动。

## 3. 终端二：启动头部

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/headmotor/yiyou/yiyou_niic_ros
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source install/local_setup.bash
ros2 launch yiyou_niic_ros yiyou_csp.launch.py
```

头部 PHA 抬头／低头当前为 **15°/s**，加减速度各 **60°/s²**。更新后需重启头部节点和 PICO 桥接，头部启动日志应显示 `PHA max 15 deg/s, PHU max 5 deg/s`。零点与限位保持原值。

## 4. 终端三：启动腰部俯仰

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/waist_pitch_motor
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source install/local_setup.bash
ros2 run waist_pitch_ros waist_pitch_csp_node
```

腰部节点启动即以当前位置使能保持。头部在有效摇杆输入时由桥接自动启动 CSP；现有头部服务会使能 PHA、PHU，但仅 PHA 接收摇操目标，PHU 保持原位。

## 5. 终端四：全身 PICO 摇操（含机械臂）

只动头腰时使用文档开头的 `--no-arms` 命令，跳过本节。

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source install/local_setup.bash
python3 tools/pico_servol_ros.py
```

打开 PICO 端遥操程序，将 UDP 目标设为**这台机器人电脑的局域网 IP，端口 5050**。
看到 `PICO UDP 5050 ready` 后，先松开左右 Grip 各一次，再**持续按住**对应 Grip；程序自动给对应机械臂上电，等待 `ServoL driver ready; anchors aligned` 后开始操作。

| 操作（按住对应侧 Grip） | 功能 |
| --- | --- |
| 移动、转动左／右手柄 | 控制左／右臂末端 |
| 左／右摇杆向左、向右 | 对应臂角减小、增大，固定 5°/s |
| 右摇杆前推、后拉 | 头部抬头、低头（PHA，目标速度 15°/s） |
| 左摇杆前推、后拉 | 腰部抬起、弯腰 |
| 点按右 A（无需 Grip） | 双臂同时进入示教；再按一次同时停止 |

摇杆死区为 `0.15`，各方向有效扇区为 ±30°。头腰摇杆回中即请求停止并保持；松开 Grip 停止对应遥操，机械臂恢复非实时模式。**任意手臂示教期间，两臂位姿与臂角遥操均屏蔽，头腰仍可用 Grip＋摇杆控制。**

左 A 不触发示教。每次双臂示教默认 60 秒，到时或再次按右 A 结束；松开 Grip 不会结束示教。
右 A 只启停示教模式，**每条手臂仍需按住自己的末端实体按钮才可拖动，松开即停止拖动**。
SDK 准备流程需要切到手动模式并下电，再开启带实体按钮的拖动；程序明确禁用免按键拖动，不能保证切换全程保持 `powerState=on`。
负载沿用原 SDK 的 0 kg／零质心设置。任一臂示教结束、失败或任一侧 UDP 超时，均请求双臂停止示教。
等待停止确认后，重新松开、按住 Grip 才恢复手臂遥操。首次更新需重启机械臂驱动和 PICO 桥接，
无需启动独立 `cartesian_drag_safe_node`。

需要 ACT 示教数采时，右 B 单击开始，录制中再次单击停止并自动保存；双击（≤350 ms）丢弃当前记录。
停止后的单击保存需等待 350 ms，以区分双击；看到保存或丢弃确认后，再按 B 开始下一条。
右 A 和实体拖动按钮含义不变。三相机和采集器另外启动，步骤见
[ACT 当前示教数采](../../../../software/vla/act/xcore_act/README_ACT_XCORE.md#当前-ros-双臂示教数采)。

## 6. 结束

先松开两个 Grip，在**终端四**按 `Ctrl+C`，等待停止确认，再在终端一、二、三按 `Ctrl+C` 退出驱动。需要整机断电时，再按设备流程关闭电源。

首次构建、更新后重编译和详细接口见 [PICO_SERVOL_ROS.md](PICO_SERVOL_ROS.md)。

## 腰部启动提示 `SDO 607D:01 timed out`

表示启动时未收到腰部电机的限位读取回复。先检查电机电源、接在 `can1` 的 CAN 线和 1 Mbps 波特率；`ip -details -statistics link show can1` 若显示 `ERROR-PASSIVE` 或 `BUS-OFF`，应先排查链路。
结束腰部节点，确认供电、接线正常后重配 `can1` 并只读测试：

```bash
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
cd /home/niic/aide/AIDE_robot_main/hardware/waist_pitch_motor
python3 waist_read_position.py --interface can1 --node 1
```

打印 `position=...` 后再执行第 4 步；仍超时则先解决电机与 CAN 通信，无需反复启动 CSP。

## 腰部提示 `CSP cycle missed for more than 0.2 s`

表示 CAN 控制周期中断超过 200 ms，节点已请求失能，须结束旧腰部节点后重新启动。
更新后的节点用独立线程发送 CAN 周期，ROS 发布在另一线程执行；200 ms 周期／反馈保护、速度和限位保持原值。
若更新后仍出现此错误，保留日志中的 `gap=...` 数值继续排查调度或 CAN 阻塞。

## 固定方向推杆时腰部抖动

桥接已修正反馈回摆导致腰部目标反向跳变的问题，两种启动模式使用同一修正。
更新后松开 Grip，退出并重新启动 PICO 桥接即可加载，无需重编译。
若仍抖动，需要采集运动期间的目标、反馈及 CAN 指令；不能仅凭模式差异确定原因。

若 CAN 周期正常、固定推杆仍一走一停，还需检查 ROS 反馈接收是否滞后。
桥接已将腰部位置反馈改为只保留最新一条的 best-effort 订阅，避免可靠重传延迟触发目标领先限制；
速度、限位和目标指令传输方式保持原值。此更新只需重启 PICO 桥接，无需重编译或重启腰部节点。
