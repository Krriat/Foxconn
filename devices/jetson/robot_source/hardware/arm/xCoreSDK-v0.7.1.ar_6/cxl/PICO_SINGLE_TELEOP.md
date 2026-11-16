# PICO 单右臂遥操与夹爪

`pico_servol` 使用 PICO 右手柄（UDP 包 `hand_id=1`）控制右臂
`192.168.2.160`，并把 Trigger 状态发布到 STS3215 夹爪驱动。

## 控制映射

- 按住 Grip：以当前手柄和末端姿态为锚点，进行右臂 ServoL 遥操
- 松开 Grip：停止右臂遥操
- 遥操中右摇杆前推：增大臂角；后推：减小臂角；松手回中后保持当前臂角
- PICO 程序启动后，Edge API 网页无需等待 Grip 即可设置臂角：程序保持当前
  末端位姿并进入臂角专用 ServoL；按 Grip 后退出专用模式并进入正常遥操
- 网页“臂角控制”开关关闭时会退出上述专用 ServoL；正常 Grip 遥操不被中断，
  单臂摇杆臂角控制仍按原逻辑工作
- 网页目标会平滑跟随并锁定，后续每条 ServoL 指令都携带该臂角；正常遥操
  期间摇杆臂角输入不再改变网页目标
- 空闲时按 Down：右臂回到已配置的复位关节位置
- 松开 Trigger：夹爪目标位置 `900`
- 按住 Trigger：夹爪目标位置 `1500`

Trigger 使用 UDP 按钮字节的 bit 0。夹爪指令只在目标改变时发布；如果
STS3215 驱动晚于遥操程序启动，检测到订阅者后会自动重发当前目标。

## 构建与自检

```bash
source /home/niic/aide/startup/ros_environment.sh
source /opt/ros/humble/setup.bash
cmake --build /home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6/build \
  --target pico_servol -j2
/home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6/build/bin/pico_servol --self-test
```

## 运行

先启动夹爪驱动：

```bash
source /home/niic/aide/startup/ros_environment.sh
source /opt/ros/humble/setup.bash
source /home/niic/aide/hardware/end_effector/sts3215_ws/install/setup.bash
ros2 launch sts3215_driver sts3215.launch.py
```

再启动单臂遥操：

```bash
source /home/niic/aide/startup/ros_environment.sh
source /opt/ros/humble/setup.bash
/home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6/build/bin/pico_servol
```

同一时间不要运行 `receive.py` 或 `pico_dual_servol`，它们会争用 UDP 5050。
运行前确认机器人工作区无人、急停可用、夹爪 900/1500 均为安全机械位置。

## 从双臂版移植的右臂参数

- 位移增益：默认 `0.618`，可通过环境变量 `XCORE_TELEOP_POSITION_GAIN` 在
  `0.100~1.000` 范围内设置
- 旋转增益：`0.25`
- ServoL 控制频率：`100 Hz`（`10 ms` 周期）
- 最大位置步长：`0.5 mm / 10 ms`（最大速度保持不变）
- 最大旋转步长：`0.25 deg / 10 ms`（最大速度保持不变）
- 录制样本频率仍为 `50 Hz`
- 摇杆臂角死区：`0.15`
- 最大臂角速度：`2 deg/s`，指令限制在 `[-180 deg, 180 deg]`
- 摇杆臂角低通滤波时间常数：`80 ms`
- 最大臂角加速度：`2 deg/s²`（从静止到满速至少约 `1 s`）
- 臂角单周期硬限幅：每个 `100 Hz` ServoL 周期最多变化 `0.01 deg`，因此实际
  连续速度还受 `1 deg/s` 上限约束；网页即使提交较远目标，也只会逐周期逼近
- 网页实时接口通过 `XCORE_TELEOP_ELBOW_COMMAND_PATH` 和
  `XCORE_TELEOP_ELBOW_STATUS_PATH` 指定的运行时文件交换目标与状态，无需重启遥操
- 关闭网页臂角控制时，运行时目标会重置为关闭时的当前臂角，不保留旧目标
- ARC 控制器即使返回 `hasElbow=false`，只要臂角/J3 数值有限且位于上述范围，
  程序仍将其作为初始臂角，并在实时指令中主动设置 `hasElbow=true`
- VR 单帧位置跳变限制：`100 mm`
- VR 单帧旋转跳变限制：`45 deg`
- UDP 超时：`250 ms`
- 控制循环最大延迟：`40 ms`
- 关节安全余量：`12 deg`，并与控制器软限位取交集
