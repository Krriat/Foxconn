# 双 ACT：手动确认阶段完成，每阶段先双臂归位

当前流程（2026-11-12）：整套执行流程不做箱盖判断。
`start_sequence` 先让双臂回到数据集 B 键初始位置，再运行远盖 ACT。
操作者调用 `finish_far` 后，停止、确认吸盘关闭、双臂再次归位，再运行近盖 ACT。
调用 `finish_near` 后停止并关闭吸盘，整轮结束。程序不判断箱盖是否已经打开。

## 当前命令

模型服务器 `jj@172.16.12.16` 启动待命推理服务（不会触发机械臂动作）：

```bash
~/.local/share/robot-policy-streams/start.sh inference
```

Jetson `jetson@172.16.12.95` 启动待命控制服务：

```bash
~/.local/share/robot-policy-streams/start.sh driver
~/.local/share/robot-policy-streams/start.sh executor
```

三路相机、关节和吸盘状态流沿用原启动方式。`task-monitor` 不再自动启动，执行器
不订阅 `task_scene`，也不加载箱盖识别模板。相机仍是 ACT 本身的必要输入。
腕部 D405 保持与数采相同的 RGB+depth 采集配置，只发布 RGB，深度帧丢弃。

在 Jetson 或同一 ROS 网络的终端执行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 1. 双臂归位，然后运行原 ACT 开远盖
ros2 service call /robot/policy/start_sequence std_srvs/srv/Trigger '{}'

# 2. 远盖动作完成、可以回撤归位时，手动执行：停止、释放、双臂归位、运行近盖 ACT
ros2 service call /robot/policy/finish_far std_srvs/srv/Trigger '{}'

# 3. 近盖动作完成时，手动执行：停止并关闭吸盘，结束整轮
ros2 service call /robot/policy/finish_near std_srvs/srv/Trigger '{}'
```

以上三条动作服务按现场进度分开调用，不要一次粘贴连续执行。
返回 `Accepted` 表示接受异步操作；通过下面的状态话题查看实际阶段和停止结果。

```bash
ros2 topic echo /robot/policy/status
# 任意时刻停止，并取消尚未执行的阶段
ros2 service call /robot/policy/stop std_srvs/srv/Trigger '{}'
```

如果只运行近盖：调用 `/robot/policy/start_near`，同样先归位并使用近盖候选；
最后调用 `/robot/policy/finish_near`。`/robot/policy/start` 保留原 ACT 单轮行为，
不进入两阶段，使用 `/robot/policy/stop` 停止。

## 归位与阶段规则

流程为：
`INITIALIZING → HOMING → VERIFY_HOME → WAITING_ACTION → RUNNING(far)`
`→ 手动 finish_far → STOPPING → STAGE_RELEASE → INITIALIZING → HOMING`
`→ VERIFY_HOME → WAITING_ACTION → RUNNING(near) → 手动 finish_near`
`→ STOPPING → STAGE_RELEASE → IDLE`。

每阶段双臂都执行 MoveAbsJ；以新鲜实测反馈检查 14 个关节的最大误差 ≤ 0.01 rad
（约 0.57°），不以服务返回成功代替到位验证。归位速度为 50 mm/s、关节速度比例
0.05。未到位禁止执行模型动作；首段预测必须使用归位确认后的新图像。

阶段结束只接受操作者服务请求。`finish_far` 仅允许在远盖 RUNNING 时调用，
`finish_near` 仅允许在近盖 RUNNING 时调用；错误阶段、归位、停止、故障或重复请求
都会被拒绝。分别命名可防止迟到的远盖完成请求误结束近盖。

换阶段先确认驱动停止和两个吸盘关闭，然后更换 `run_id`，清除旧预测、融合、
速度和序号状态。近盖重新归位并验证后才开始，旧轮次或错误权重的预测被拒绝。
不根据吸盘、姿态、静止或箱盖画面自动判定任务完成；无场景识别超时。
每阶段仍保留 90 秒运行上限，超时会停止并取消后续阶段，不会自动切到近盖。
关节限位、通信超时、跟踪误差、速度和加速度约束均保留。

`completion_mode=manual`、`box_scene_check_enabled=false` 在状态中明确标示。
手动确认记录的 `stage_complete` 带 `completion_source=operator_service`，
只表示操作者确认，不是程序验证开盖成功。以后外部完成判断可以调用同样的两个服务。

## 两个模型与证据

2026-11-13 的排查已将推理进程的循环垃圾回收移到待机阶段，避免第 2 代回收在
推理中产生约 110 ms 的停顿。修复后的 120 秒无运动预览通过；模型、归一化、
融合、限速及动作超时保护未变。开盖轨迹退化仍未验证恢复，不能将此预览视为
实机开盖成功。完整证据见 `training_verification/manual_disconnect_20261113/README.md`。

- 远盖：原 ACT step 14000，
  `model_outputs/act_suction_20261111_213453/checkpoints/014000/pretrained_model`；
  SHA-256 `0d71f2e37a180b2e829bc7bf423bbe8171945c3afa327f8d08f2a72297d00d93`。
- 近盖：成功候选 ACT step 1000，
  `model_outputs/act_near_success_20261112/pretrained_model`；
  SHA-256 `ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8`。

两个模型均预加载、核验和预热，按阶段选择。近盖成功归档、三路录像、训练及验证
参数位于 `model_outputs/act_near_success_20261112/`，此次修改未变动归档。

手动阶段控制的隔离 ROS 检查位于 `training_verification/manual_stage_20261112/`，
假设备只在 Domain 77、localhost-only 中运行。测试涵盖完全没有箱盖监视节点时
的完整两阶段执行、两次归位、手动完成、重复/错阶段请求、停止取消、权重检查和超时。
这次仅更新待命服务，没有再次触发实机开盖；完整两阶段的实机表现仍待验证。

此前模型加载和执行链路回归位于 `training_verification/dual_act_deployment_20261112/`。
已停用的箱盖判断逻辑、早期试验细节和旧启动方式仅作历史证据，保存在
`training_verification/manual_stage_20261112/README_before.md`，不作为当前运行说明。
