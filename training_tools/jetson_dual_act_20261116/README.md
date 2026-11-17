# Jetson 双 ACT + 电脑 Qwen LoRA：全程原速

当前两份 ACT 已安装到 Jetson Orin Nano 8GB；电脑只运行已训练的 Qwen3.5-9B LoRA。
**全程 `execution_speed_scale: 1.0`、`execution_speed_profile: uniform`，加速度上限恢复为 `1.0 rad/s²`。**

当前按用户最新要求关闭右臂等待：`coordinate_far_right_arm: false`，部署工具保留现有配置，不会自行重新开启。关闭后完成一轮两盖自动开箱：远盖 ACT 43.9 秒、近盖 ACT 30.7 秒；仍由 Qwen LoRA 判断完成并发起切换，每阶段前双臂归位，结束时 IDLE、吸盘关闭。该单轮对照记录：`training_verification/no_right_hold_20261117_102134/comparison.json`，完整录像与成功配置位于其 `trial02_no_right_hold/`。
相比启用等待那轮远盖 62.2 秒，本轮少约 18.3 秒。两种配置各一次成功不能确定长期可靠性，也不能认定此前失败唯一由右臂过早动作造成。

2026-11-17 增加远盖双臂顺序约束后，完成一轮实机自动远盖、归位、近盖流程；录像中两盖打开，结束后 IDLE、吸盘关闭。这是单轮结果，不是长期成功率。
证据在 `training_verification/grasp_repair_20261117_095228/coordination01/`，包括 `test_report.json`、`waiting_breakdown.json`、录像和成功部署快照 `successful_setup/`。
仍只有远盖、近盖两份双臂 ACT，没有拆成左臂、右臂模型。历史的 `coordinate_far_right_arm: true` 试验会在自动远盖阶段延后右臂执行，直到实际关节反馈满足左臂起抬条件；随后继续执行同一 ACT 的右臂预测，使用原 chunk fusion 和限幅。当前该功能已关闭。
启用等待的历史试验从远盖 ACT 开始到放行右臂约 24.5 秒，Qwen 首次判定远盖打开约 50.8 秒，远盖最终完成约 62.2 秒；前段左臂仍有小幅调整，没有固定等待计时。

以下保留迁移初期的验证记录，早于本页顶部的两轮实机成功，不代表导出时的服务状态。迁移时完成模型校验、精度/耗时测量，以及真实 Jetson ACT + 真实电脑 Qwen + 模拟双臂的隔离两阶段联调。
2026-11-17 已重新连接两台设备并通过 60 秒真实相机无运动预览：533 个 ACT 动作块、5948 条有效轨迹记录，运动命令为 0，Qwen 阶段切换调用为 0，双臂编码器读数保持不动、吸盘关闭。随后连续 12 秒待命就绪检查通过。
上述迁移预览没有启动现场机械臂。该验证确认部署、实时输入和推理/控制计算链路可用，不代表实机开盖成功率已验证。

## 启动待命服务

umie（172.16.12.45）：

```bash
bash ~/.local/share/robot-relay-bridge/start.sh
bash ~/.local/share/robot-camera-streams/start.sh wrists
```

Jetson（172.16.12.95）：

```bash
bash ~/.local/share/robot-camera-streams/start.sh head
bash ~/.local/share/robot-state-streams/start.sh
bash ~/.local/share/robot-policy-streams/start.sh driver
bash ~/.local/share/robot-policy-streams/start.sh executor
bash ~/.local/share/robot-policy-streams/start.sh inference
```

电脑（172.16.12.16）：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act_20261116/start_brain.sh
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
/usr/bin/python3 /home/jj/code/ACT/training_tools/jetson_dual_act_20261116/check_ready.py
```

以上命令只启动待命服务或检查状态，不发起归位/开盖。检查要求状态连续正常 12 秒；网络或图像不正常时返回失败。

## 实际启动一轮

恢复以上待命服务并通过就绪检查，两盖处于示教起始状态，且人员离开工作区后，在电脑执行：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act_20261116/run_sequence.sh
```

此命令会实际驱动机械臂并记录结果。启动后命令立即返回，监督录像在 systemd 用户服务中持续运行；用下方停止命令停机，不能以关闭启动终端代替停止。顺序为：

1. 双臂归位并通过实时关节反馈确认。
2. Jetson 运行远盖 ACT；电脑 Qwen 连续确认阶段完成。
3. 停止、释放吸盘，双臂再次归位并确认。
4. Jetson 切换近盖 ACT；Qwen 连续确认完成后停止、释放吸盘。

Qwen 至少要求阶段运行 5 秒、3 张不同新图像且持续至少 1.5 秒确认；不确定时保持当前阶段。
2026-11-17 增加动作进度门控：`require_manipulation_sweep_for_brain: true`。
Qwen 即使把竖起的盖子识别为打开，也必须等待实际关节反馈表明完整翻盖动作已进入收尾，才允许完成阶段。
该门控在电脑决策端和 Jetson 执行端分别检查，每次换阶段清空进度；它不是箱盖或真空传感器，不能单凭关节姿态认定开盖成功。
自动阶段超过 90 秒未确认只停止，不会按时间强制切到下一模型。

停止：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act_20261116/stop_sequence.sh
```

也可在设置 ROS 环境后直接调用 `/robot/policy/start_sequence_brain`；原 `/start_sequence` 仍是手动切换流程。

## 模型与验证

- 远盖：当前 `act_far_current_recovery_20261113_03/checkpoints/001000`，权重 SHA256 `af60ccd4cfca37c420d1107e440c8fcb78904d22a28e802d39b4b5925a9055f9`。
- 近盖：`act_near_success_20261112`，权重 SHA256 `ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8`。
- Qwen LoRA：`model_outputs/task_brain_lora_20261112/adapter`，SHA256 `63e882d82b66a58da2269ce32f22dedc41b42959b2165d080a131ddeeabf0ade`。
- Jetson 模型目录：`~/.local/share/robot-policy-streams/jetson_act_runtime_20261116/models/{far,near}`。
- TensorRT 使用全 FP32，关闭 FP16/TF32；36 组输入对照最大关节差约 0.017°，吸盘阈值判断无差异。
- 两份模型常驻，推理中位耗时约 104 ms（含 CPU RGB 预处理、GPU 推理、反归一化和结果回传，不含 JPEG 接收/解码）。原动作时间轴为 20 Hz，ServoJ 为 100 Hz；实际重规划更新低于 20 Hz。
- 电脑 ACT 入口通过 `inference_remote.txt` 禁用，避免与 Jetson 重复发布动作。
- Jetson 推理的 DDS 传输限定到实际互通的数据接口；机械臂 SDK 网口和继电器 TCP 配置不变。[Fast DDS 官方接口白名单说明](https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/transport/whitelist.html)。
- umie 腕相机采用相同的接口限制，排除与模型通信无关的继电器专用网口。90 秒无动作检查中三路相机最大接收间隔为 149/143/165 ms，均未超过 250 ms；这不代表长期运行绝不会断流。
- 继电器长期会话出现的周期性读取超时，本次迁移未修复。

本次证据：`training_verification/jetson_dual_act_20261116_171114/`。
状态说明 `migration_status.json`，精度/速度 `tensorrt_validation.json`，隔离完整链路 `sandbox_report.json`。

如需重复无运动预览，在已设置 ROS 环境的电脑执行 `verify_preview.py --preview-seconds 60 --output 新目录`；该工具只创建 preview/stop 客户端，不调用 start/home/ServoJ。

2026-11-17 的复验记录：`training_verification/jetson_dual_act_resume_20261117_084613/`，其中 `model_and_engine_checks.json` 保存重新开机后的模型/引擎完整性检查，`preview_summary.json` 保存真实输入预览结果，`final_readiness.json` 保存最终待命状态。该轮实时推理中位耗时 108.63 ms、P95 117.15 ms、最大 172.36 ms。

动作进度门控与通信修改证据：`training_verification/completion_contact_fix_20261117_090845/`。
`progress_replay.json` 记录了两段完整成功轨迹可通过门控、今天两次中途完成被拦截；该检查不是新的实机成功率。
`launch_supervised_test.py --execute --output 新目录` 在 systemd 用户服务中运行监督录像，监督程序退出时再发送停止请求，避免交互终端结束后失去监督；录像逐帧时间保存在 `video_frames.jsonl`。

远盖吸盘释放检查 `guard_far_suction_release: true` 仅用于自动远盖流程。ACT 请求关闭左吸盘前，应已观察到示教中的左臂起抬关节区间；缺少该动作时最多保持吸气 1.5 秒，然后取消本轮，避免继续空翻。该区间不是吸附压力或接触检测，不能保证吸牢；显式停止、手动流程及近盖不受此过滤影响。历史离线检查中，原 29 段远盖示教回放的吸盘时序保持不变，未吸起记录在提前释放时被拦截；随后本页顶部两轮成功复测均保留了此设置。

在添加顺序约束和本页顶部成功复测之前，曾有一轮实机未吸起远盖，未进入近盖，90 秒自动阶段超时后停止。随后完成释放检查修改，并通过 20 秒无运动预览（175 个 ACT 块、1950 条预览轨迹、实际运动命令 0、吸盘关闭）。该历史检查的结论与限制见上述目录的 `final_review.json`；它本身不能证明吸附问题已解决。
