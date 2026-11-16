# Jetson 双 ACT + 模型服务器 Qwen LoRA

> 当前任务已按用户要求暂停。已恢复“模型服务器运行原双 ACT、Jetson 执行、手动阶段完成”的流程，
> Qwen 自动调度和 Jetson 模型推理均已停止。下方自动链路启动命令暂不使用。
> 手动流程使用 `/robot/policy/start_sequence`、`finish_far`、`finish_near`；两个 ACT 前均自动双臂归位。

Jetson `172.16.12.95` 是 Orin Nano 8GB。远盖与近盖两个 ACT 均在 Jetson 常驻，
由阶段控制器选择当前权重；模型服务器 `172.16.12.16` 只运行 Qwen3.5-9B LoRA，
通过 ROS 完成消息调度阶段。`172.16.12.45` 的 umie 电脑提供左右腕相机。

## 启动服务

umie 电脑：

```bash
bash ~/.local/share/robot-camera-streams/start.sh wrists
```

Jetson：

```bash
bash ~/.local/share/robot-policy-streams/start_stack.sh
```

模型服务器：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act/start_brain.sh
```

这些命令启动待命服务，不开始机械臂动作。重复执行会保留已运行的服务。
桌面旧 ACT 推理入口已标记为迁移，避免同时存在两个动作发布者。

## 执行一次完整流程

将箱盖恢复到示教起始形态。在模型服务器运行：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act/run_sequence.sh
```

这条命令会实际驱动双臂，等待模型就绪后执行：

1. 初始化并让两臂回到原数据集 B 键初始位置，以新关节反馈确认两臂均归位。
2. Jetson 执行远盖 ACT。
3. Qwen LoRA 连续确认远盖完成，发送带阶段、序列 UUID、适配器哈希和图像时间的完成消息。
4. Jetson 停止、释放吸盘，重建执行 UUID，再次让两臂归位并核对反馈。
5. Jetson 切到近盖 ACT；Qwen 连续确认两盖均已打开后停止、释放吸盘。

终端打印归位、运行、切换与停止状态，并保存三路视频和 ROS 日志。
记录位于 `training_verification/jetson_dual_act_20261113/user_runs/`。
Ctrl+C 会请求停止。另一个终端也可以运行：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act/stop_sequence.sh
```

若只使用原生 ROS 命令，在已设置 ROS 环境的 Jetson 终端调用：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 service call /robot/policy/start_sequence_brain std_srvs/srv/Trigger '{}'
```

自动模式使用 `/start_sequence_brain`；原 `/start_sequence` 保留手动阶段完成服务。
时长不作为任务完成的依据：阶段超时只停止，不据此切换到下一阶段。

## 模型与约束

- 远盖 ACT：`0d71f2e37a180b2e829bc7bf423bbe8171945c3afa327f8d08f2a72297d00d93`。
- 近盖 ACT：`ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8`。
- Qwen LoRA：`63e882d82b66a58da2269ce32f22dedc41b42959b2165d080a131ddeeabf0ade`。
- 两个 ACT 目录在 Jetson 的 `~/.local/share/robot-policy-streams/jetson_act_runtime_20261113/models/`。
  每路保留原权重、配置和归一化文件，并校验完整文件哈希。
- 使用三路原始 640×480 RGB 与 16 维状态，输出原始 20Hz 的 50×16 chunk。
  没有缩小图像、重新训练或改变关节／吸盘定义。
- chunk fusion、速度／加速度限制、250ms 动作看门狗、50ms 控制定时保护和关节范围保持。
- Qwen 仅调用带当前阶段标识的完成接口；首次机械臂执行仍需用户启动。
- Qwen 每秒检查画面，至少连续 3 张不同图像、持续至少 1.5 秒确认，且每阶段至少运行 5 秒。
  图像质量不足时保持不确定，旧阶段、旧图像和旧权重的完成消息不能推进流程。

## 验证记录

记录集中在 `training_verification/jetson_dual_act_20261113/`。
`local_equivalence.json` 是独立推理实现与原 LeRobot 推理的对照；
`bundle/onnx_manifest.json` 记录导出网络来源。
TensorRT 引擎及数值／速度核对结果保存在 Jetson 的同名运行目录。
实机测试目录 `live_01` 保留了原 FP32 路径发生动作断流的记录及停止后的反馈确认。
后续复测使用新目录，原失败记录保留。
