# ACT 时间对齐推理与 Jetson ServoJ 执行

2026-11-12，ROS 2 Humble，Domain 1。电脑 `172.16.12.45` 发布手腕图像，
Jetson `172.16.12.95` 发布头部图像、关节与吸盘反馈，并执行轨迹；
模型服务器 `172.16.12.16` 运行 ACT。部署目录均为 `~/.local/share/robot-policy-streams/`。

当前是 **`enable_motion: false`，待机**。真实三路图像的手动停止和 30 秒自动停止预演已通过；
本次修改尚未执行实际 ServoJ 运动。此前旧版本发生的实机振动记录保存在 `evidence/live_f8d1c589-.../`。

## 三项执行修改

1. **统一时间和延迟补偿。** Jetson 的 chrony 提供局域网 NTP；电脑和模型服务器的
   systemd-timesyncd 都以 `172.16.12.95` 为唯一时间源。复测相对偏差约 1～4 ms，
   双向估计仍受链路不对称影响，不能理解为精确到微秒。每个 chunk 记录三路图像和关节的源时间、
   推理开始/结束、发送、Jetson 接收时间，执行侧记录选点和左右 SDK sendCommand 调用区间。
2. **按同一绝对时间融合完整 chunk。** GPU 每收到新的头部图像就预测一次，名义 20 Hz，
   发送完整 `50×16`、间隔 50 ms 的预测。取消“只取前 10 步、每 0.5 秒替换一次”的执行方式。
   原模型、权重及归一化处理不变。对于执行时刻 `t`，每个 chunk 用
   `index=(t-observation_time)/0.05` 取点，并在邻近动作之间线性插值。
   仅融合覆盖这个时刻的预测，不把不同 chunk 的第 0 步对齐。
   关节使用 ACT 约定的 `exp(-0.01*rank)` 加权，rank=0 为最早的有效预测。
   吸盘使用最新有效 chunk 在同一时刻的值，再经过原有滞回，避免平均不同抓取阶段。
3. **在 C++ ServoJ 循环内平滑。** 100 Hz 控制循环保留上一次输出位置和速度，
   新 chunk 到达时不重置，速度上限 0.5 rad/s、加速度上限 1.0 rad/s²。
   制动速度及目标接近速度共同决定下一周期速度，按实际周期积分。
   相对实测位置的 0.035 rad 限制施加在平滑器的输入目标上，不能在输出端突然截断速度。
   跟踪误差超过 0.15 rad、周期超过 50 ms 等情况仍会停止。

当前 `execution_lead_seconds: 0`，表示没有虚构控制器内部响应延迟。
C++ 会根据已记录的本机选点到 SDK 发送延迟更新补偿估计；两臂的实际 SDK 调用时间各自记录。
图像源时间是驱动读取图像后的主机时间，并非硬件曝光时间；SDK 调用时间也不等于编码器开始运动时间。
控制器内部响应延迟需要下一轮实机轨迹/编码器日志才能辨识。

## 每轮行为与停止

用户确认：**每轮归位一次，手动停止，执行最多 30 秒**。

- `/start` 检查模型、输入、新版 C++ 轨迹驱动心跳、重复命令发布者和保存的数采回位点。
- 初始化双臂，然后通过 MoveAbsJ 去数采 B 键保存的位置；参数仍为速度比例 0.05、
  speed=50 mm/s、zone=0，实测误差不超过 0.01 rad。
- ServoJ 从实测位置启动。SDK 启动完成前的 chunk 被丢弃，等待启动后的新观测再推进。
  30 秒计时从双方已运行且收到第一份有效轨迹开始，归位和 SDK 启动不计入。
- 手动 `/stop` 停止双臂并关闭左吸盘 DO1，保留右 DO2；等待在途 IO 和关闭读回完成后才重新允许开始。
  自动超时/故障保持吸盘最后状态；故障后显式 `/stop` 确认也会请求关闭左吸盘。
  预演始终不写任何吸盘输出。
- 停止或断流后禁止在同一个 run_id 下自行重启，下一轮必须再次显式开始并归位。
  归位中途停止会在所有在途 MoveAbsJ 返回后补发最终停止。

回位点来源保持为原机器人文件
`software/vla/act/xcore_act/scripts/recording/teach_suction_return_pose.json`，每轮开始前比对，未改写。
动作顺序为左臂 7 关节、右臂 7 关节、左吸盘、右吸盘；单位 rad，吸盘为 0.6 开/0.4 关滞回。
本任务右吸盘恒 0 是有效状态，不人为制造右吸盘切换或固定时刻的抓取动作。

## 话题与服务

| 名称 | 类型 | 用途 |
| --- | --- | --- |
| `/robot/policy/action_chunk` | String / JSON | GPU 完整预测，名义 20 Hz，schema `dual_arm_suction_chunk_v2` |
| `/robot/policy/validated_chunk` | String / JSON | Jetson 校验身份、心跳、全部目标和时间后转给本机 C++ |
| `/robot/policy/servo_trace` | String / JSON | 100 Hz 选点、融合数、位置、速度、时间戳及实机 SDK 调用区间 |
| `/robot/policy/trajectory_status` | String / JSON | 新版 C++ 轨迹驱动就绪心跳，1 Hz |
| `/robot/policy/status` | String / JSON | 状态机、run_id、停止原因、轨迹状态，10 Hz |
| `/robot/policy/inference_status` | String / JSON | 模型就绪、权重、输入帧龄、推理耗时 |
| `/robot/policy/{check,start,stop,preview}` | Trigger | 检查、实机开始、停止、无运动预演 |

旧 `/robot/policy/action` 的逐点 v1 协议已经替换，不能混用新旧节点。
`accepted_actions` 是兼容保留的状态字段，在新版中计数完整 chunk。
动作通道使用 Best Effort / Volatile / Keep Last 1，不积压待执行队列。
每轮拒绝更换发布者、错误权重、序号回退、非有限值、超限目标、过期心跳及不一致的时间戳。
源观测在 Jetson 接收时需不超过 300 ms，时间戳允许 10 ms 的残余同步误差。
250 ms 无新预测或 100 ms 无本地 ServoJ keepalive 会停止；执行时检测大于 10 ms 的时钟跳变。

`servo_trace` 的 `newest_index` 表示当前实际需要使用新预测的第几步，允许小数。
`contributors` 是参与融合的段数，`command`/`velocity` 是平滑后的 14 维目标和速度。
预演的 `hardware_commands_sent` 始终为 false；实机才包含左右 SDK 调用区间和 RT 关节读回。

## 启动与操作

电脑保持手腕相机服务运行。模型服务器执行：

```bash
~/.local/share/robot-policy-streams/start.sh inference
```

Jetson 执行：

```bash
~/.local/share/robot-policy-streams/start.sh driver
~/.local/share/robot-policy-streams/start.sh executor
```

任意 ROS 电脑先设置环境：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 service call /robot/policy/preview std_srvs/srv/Trigger '{}'
ros2 topic echo /robot/policy/status
```

预演可手动停止，或等 30 秒自动停止：

```bash
ros2 service call /robot/policy/stop std_srvs/srv/Trigger '{}'
```

进行下一轮实机测试时，操作者在 Jetson 的 `config.yaml` 中把 `enable_motion` 改为 true，
在待机时重启 executor。其他速度/加速度限制保持当前配置。

```bash
systemctl --user restart robot-policy-executor
```

随后在控制终端执行：

```bash
ros2 service call /robot/policy/check std_srvs/srv/Trigger '{}'
ros2 service call /robot/policy/start std_srvs/srv/Trigger '{}'
```

`success=true` 表示接受异步请求，需在状态中确认 `RUNNING`。
停止也需等待 `IDLE` 或 `FAULT` 才表示双臂清理完成。
日志位于部署目录 `executor_events.jsonl`、`inference.log`、`driver.log` 和 `executor.log`。

## 构建、回滚与验证范围

新驱动在私有 `native/` 目录构建，复用原 SDK 的完整静态库和原初始化/MoveAbsJ 实现。
原机器人源码、原驱动二进制、BT 和数采回位点均未修改。原节点文件备份到
`backup_before_timed_20261112_152529/`。回滚时需同时恢复 GPU、Jetson Python 节点和 `run.sh`，
不能把旧逐点推理与新 chunk 接收器混用。

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/AIDE_robot_main/hardware/arm/xCoreROS/install/setup.bash
cmake -S ~/.local/share/robot-policy-streams/native -B ~/.local/share/robot-policy-streams/native/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/.local/share/robot-policy-streams/native/build -j2
~/.local/share/robot-policy-streams/native/build/test_timed_policy
```

纯轨迹数学测试验证同一时刻取点、插值、过期、重复段、冲突预测融合、反复回退目标下的
位置/速度连续性和速度/加速度上限；Release 构建保留测试断言。
Python 协议/IO 单元测试 14 项，隔离 Domain 77 的假硬件状态机测试 10 项。
真实网络预演使用真实三路相机、关节和 ACT 模型，但不发初始化、归位、ServoJ 或吸盘写入。
详细结果、时延分位数和后续限制见 `TIMING_VALIDATION.md`；不能由预演推断实机任务成功率。

## 反馈队列修复（2026-11-12）

当前原生驱动的健康消息必须包含 `feedback_reader: drain_latest_v1`。每个 100 Hz 控制周期先以非阻塞、有界循环清空 SDK 接收队列，再用最后一帧关节角计算相对位置限位和跟踪误差。启动时也先清理积压帧。超过读取预算或持续没有新反馈会停止，不能使用尚未清空的历史队列继续控制。

`servo_trace` 的 `left_feedback`、`right_feedback` 记录本次读取帧数、舍弃的历史帧数、读取耗时、累计帧数及 `host_read_age_ms`。该帧龄是主机清空队列读取后的经过时间，SDK 未提供控制器采样时间戳，不能解释成真实端到端编码器采样延迟。模型 chunk 另外记录 `observation_state`，用于比较模型实际输入和执行侧反馈。

只读监测：模型服务器或 Jetson 运行 `~/.local/share/robot-policy-streams/monitor.sh --seconds 30 --output ~/feedback_check`；172.16.12.45 运行 `~/.local/share/robot-policy-monitor/monitor.sh --seconds 30 --output ~/feedback_check`。这些监测程序只有订阅，不会发送机器人命令。

当前回合时长配置保留操作者设定：Jetson 的 `max_round_seconds: null` 表示手动停止；通信、反馈和心跳看门狗仍然有效。实际运动开关以当前 `/robot/policy/status` 为准。
