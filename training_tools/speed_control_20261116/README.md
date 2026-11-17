# 1.5 倍动作时间缩放，2026-11-16

这是执行器改动，不修改 ACT 权重或相机/模型训练时间基准。

- 配置 `execution_speed_scale: 1.5`。
- 配置 `max_joint_acceleration_rad_s2: 2.25`。
- 每个预测块仍为 50 个、间隔 50 ms 的训练动作点；执行器按 `1.5 * (执行时刻 - 该块观察时刻) / 50 ms` 取点，等效相邻点时间约 33.33 ms。
- 所有参与融合的预测块和最新块的吸盘目标使用同一映射。这里是整段统一倍率，没有自动识别接触阶段，也不额外保留吸附等待时间。
- 倍率每轮固定。推理消息、执行器状态、原生驱动必须匹配；一轮内不能改变倍率。
- 相机源时间戳、推理起止和通信时间戳保持真实时间；过期检查仍按原 300/250 ms 等限制执行。
- 速度上限保留 0.5 rad/s，相对目标、跟踪误差、关节边界、每阶段双臂归位和人工完成服务均保留。
- 当前平滑响应时间保留 0.10 秒。1.5 是动作采样倍率，受重规划、限幅和平滑影响，不能据此承诺实机任务时间恰好缩短为原来的 2/3。

`native/`、`protocol.py`、`inference_node.py`、`executor_node.py` 是本次开发源；`config.yaml` 为 Jetson 配置。实际部署文件在电脑/Jetson 的 `~/.local/share/robot-policy-streams/`。

无运动检查命令（在电脑原项目环境运行）：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
/usr/bin/python3 training_tools/speed_control_20261116/verify_preview.py \
  --output training_verification/speed_control_20261116/new_preview \
  --preview-seconds 10
```

该程序只持有 `preview`、`stop` 服务客户端；没有实机启动、归位或吸盘写入客户端。

真实链路预演已通过：10 秒内 200 个预测块、990 次有效轨迹计算；1.5 倍采样下标误差为 0，硬件运动指令数为 0。

随后于 2026-11-16 完成实机验证：远盖、近盖均使用 ACT 在线推理，按“远盖 → 双臂再次归位 → 近盖”完成一轮，两盖均打开，无执行故障。倍率 1.5、加速度上限 2.25；最后 `finish_near` 正常结束、双臂 `IDLE`、两路吸盘 OFF。远盖仍是 `af60ccd4...` 权重，近盖为 `ec6e21a2...` 权重，没有使用轨迹回放或未部署候选。

完整证据、真实权重哈希、阶段归位记录及现场部署快照在 `training_verification/speed_control_20261116/both_act_1p5/`，结果为 `verification.json`。此前单独的 `far_reference_1p5/` 则是回放测试，两者明确区分。有限次成功不能解释为一般成功率或严格 1.5 倍任务耗时保证。

之后同参数、同权重的复测 `both_act_1p5_repeat01/` 在远盖阶段失败：模型吸住后主要做接触区调整，没有及时生成右臂接力翻盖轨迹，60 秒测试时限触发监督停止；没有执行器故障或指令拒收，近盖未启动。用户确认吸盘已经吸住。已保存预测动作块与上一轮成功对照；触发策略差异的具体输入/接触因素尚未隔离。因此目前不能将这版描述为已经稳定。

开启实机近盖仍调用原服务，会先归位：

```bash
ros2 service call /robot/policy/start_near std_srvs/srv/Trigger '{}'
```

近盖完成后：

```bash
ros2 service call /robot/policy/finish_near std_srvs/srv/Trigger '{}'
```

只有调用实机服务才会开始实际动作。近盖单独测试使用上面的 `start_near`；完整双 ACT 流程仍使用 `start_sequence`，远盖实际打开后调用 `finish_far`，近盖实际打开后调用 `finish_near`。没有箱盖自动判定，本轮阶段完成服务由测试监督程序根据人工画面检查发出。

备份：

- 电脑：`training_verification/speed_control_20261116/backup/pc/`。
- Jetson：`~/.local/share/robot-policy-streams/backup_before_speed15_20261116/`，包含原驱动二进制、源码和配置。
- 全部测试、部署、预演证据：`training_verification/speed_control_20261116/`。

构建原生驱动时使用独立目录并全量重编译。部署脚本检查二进制包含新的能力标识，启动前执行器再检查运行中驱动报告的能力；不能仅依据源文件内容判断部署成功。
