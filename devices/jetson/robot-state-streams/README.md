# 双臂关节角与双吸盘 ROS 2 状态发布

部署在 Jetson `172.16.12.95`，使用 ROS 2 Humble、Domain ID 1 和 Fast DDS，
与本次三路相机发布环境一致。运行目录为
`/home/jetson/.local/share/robot-state-streams/`。

## 话题

| 数据 | 话题 | 标准消息类型 | 配置频率 |
| --- | --- | --- | --- |
| 左臂 7 关节角 | `/robot/upperlimb/joint_states/left_arm` | `sensor_msgs/msg/JointState` | 20 Hz |
| 右臂 7 关节角 | `/robot/upperlimb/joint_states/right_arm` | `sensor_msgs/msg/JointState` | 20 Hz |
| 左吸盘开关读回 | `/robot/suction/left/state` | `std_msgs/msg/Bool` | 10 Hz |
| 右吸盘开关读回 | `/robot/suction/right/state` | `std_msgs/msg/Bool` | 10 Hz |
| 反馈新鲜度与错误 | `/robot/hardware_state/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 1 Hz |

QoS 为 RELIABLE、VOLATILE、KEEP_LAST(5)，普通可靠订阅和传感器 BEST_EFFORT
订阅均可匹配。没有使用自定义消息包，接收端不需要安装厂商 SDK。

## 数据含义

- 左臂读取 `192.168.1.160`，Jetson 本地接口地址 `192.168.1.10`。
- 右臂读取 `192.168.2.160`，Jetson 本地接口地址 `192.168.2.10`。
- 通过现有 xCoreSDK 的 `jointPos()` 查询实测角度。`position` 为 7 个弧度值，
  按 J1–J7 排列。`name` 分别为 `left_joint_1` 至 `left_joint_7`，
  `right_joint_1` 至 `right_joint_7`。速度与力矩字段为空，没有用零值代替未知反馈。
- 时间戳在各臂完成一次 SDK 读取后生成。左右采样独立，不表示硬件同步。
- 吸盘通过现有 USR-M050 SDK 的 FC01 读取 DO，端点为
  `192.168.1.7:20108`（有线），unit ID 为 1。DO1 对应左吸盘，DO2 对应右吸盘。
  该只读客户端复用 TCP 连接；与 ACT 输出 worker 仍通过现有跨进程锁串行访问设备。
- 吸盘 `true` 表示逻辑输出开启/继电器闭合，`false` 表示逻辑输出关闭。
  这是硬件输出状态的读回，不能用作真空建立或物体已吸住的判断。
- 读取失败时停止发布该路有效样本，并通过诊断话题报告错误；不重复旧值，
  也不把读取失败转换为“关闭”。Bool 本身没有时间戳，消费者应结合接收超时
  和诊断中的 `last_success_age_sec` 判断新鲜度。

## 其他电脑接收

在能够与 Jetson 直接进行 DDS 通信的局域网内执行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 topic echo /robot/upperlimb/joint_states/left_arm
ros2 topic echo /robot/upperlimb/joint_states/right_arm
ros2 topic echo /robot/suction/left/state
ros2 topic echo /robot/suction/right/state
```

每条 echo 持续运行，可分别开终端，或加 `--once` 只读一条。只看角度数组时：

```bash
ros2 topic echo /robot/upperlimb/joint_states/left_arm --field position --once
ros2 topic echo /robot/upperlimb/joint_states/right_arm --field position --once
ros2 topic echo /robot/hardware_state/diagnostics --once
```

统一验证四路数据、诊断状态、消息新鲜度及实际频率：

```bash
scp jetson@172.16.12.95:/home/jetson/.local/share/robot-state-streams/verify_state.py ~/verify_robot_state.py
/usr/bin/python3 ~/verify_robot_state.py --seconds 20 --output ~/robot_state_validation.json
```

## 启动、停止与构建

在 Jetson 上运行：

```bash
~/.local/share/robot-state-streams/start.sh
systemctl --user status robot-hardware-state --no-pager
tail -40 ~/.local/share/robot-state-streams/runtime.log
```

服务是当前用户会话的 systemd 临时服务，异常退出后 3 秒重启；尚未配置开机自启。
重启 Jetson 后按上述命令重新启动。停止命令为：

```bash
systemctl --user stop robot-hardware-state
```

重新构建：

```bash
source /opt/ros/humble/setup.bash
cd ~/.local/share/robot-state-streams
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build -j2
```

链接现有机器人目录中的 `libxCoreSDK.a`、`libxMateModel.a` 和 `libm050.so`。
SDK 连接创建采用串行方式，采样在独立工作线程执行。

## 实现范围

本次新增独立状态发布程序和订阅验证脚本，不启动完整运动驱动或录制工作流。
节点仅调用 SDK 连接、关节查询和 M050 DO 读取接口；不包含运动、上电、模式
切换或 DO 写入调用，也没有为验证开关状态切换吸盘输出。
原机器人仓库的行为树、点位、运动目标、速度和流程配置保持原样；保护文件
哈希核查结果保存在部署目录的验证记录中。

短时实机验证结果见 `VALIDATION.md` 与 `evidence/`。

2026-11-12 接入策略执行后的修复：M050 读取使用
`/run/user/<uid>/robot-m050.lock`，与策略节点的 M050 写入做跨进程互斥。
策略启动直接复用新鲜的 ROS 读回，不再同时打开第二个读取请求；常量 0 是有效读回。
本节点仍只读，不新增 DO 写入、上电或运动接口。
