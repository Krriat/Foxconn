# Foxconn

双臂开箱系统的三设备源码与实机记录快照，保存于 **2026-11-16（Asia/Shanghai）**。

## 本次成功结果的准确范围

| 测试 | 实际执行方式 | 结果 |
|---|---|---|
| 远盖 ACT 对照 | 电脑 ACT 推理 → Jetson 执行 | 未吸住远盖，保留失败记录 |
| 远盖成功对照 | 保存的成功轨迹回放 → 原执行器和限幅器 | 远盖打开，无执行故障；不是 ACT 在线推理 |
| 近盖第一次 | 原近盖 ACT，电脑推理 → Jetson 执行 | 近盖打开，正常结束 |
| 近盖复测 | 同一近盖 ACT | 再次打开；推理中位数 9.05 ms，P95 10.20 ms |

每次实机执行前都确认双臂到数据集初始位置。最后一轮通过 `finish_near` 正常结束，双臂 `IDLE`、两路吸盘关闭。这里只记录有限次实测，不将其解释为一般成功率。

今天额外微调的远盖候选**没有部署、没有实机验收**。Qwen/LoRA 自动调度也不属于这次成功运行链路。

## 三台设备

| 设备 | 地址 | 本次职责 | 源码快照 |
|---|---|---|---|
| 电脑 jj | 172.16.12.16 | ACT 推理、参考轨迹诊断、录包及训练工具 | `devices/pc/`、`training_tools/` |
| Jetson | 172.16.12.95 | 双臂控制、100 Hz 原生 ServoJ、头相机、关节和吸盘状态 | `devices/jetson/` |
| umie | 172.16.12.45 | 左右腕相机、有线继电器 TCP 桥接 | `devices/umie/` |

ROS 2 Humble，Domain 1，`rmw_fastrtps_cpp`。ACT 动作时间基准为 20 Hz，每次预测 50 步，执行器保留 chunk fusion、速度/加速度限制、输入时效及跟踪误差保护。此次归档没有修改提速参数。

继电器链路为：`Jetson 192.168.50.10 → umie 192.168.50.20:20108 → 192.168.1.7:20108`。继电器直接接 umie 的 USB 网卡。桥接源码和两端配置已保存。双臂控制器地址为 `192.168.1.160`、`192.168.2.160`。

## 目录与完整数据

- `devices/`：从设备当前目录读取的代码、配置和构建文件；以这里的部署快照为准。
- `training_tools/`：训练、回放、通信检查和测试工具的开发归档。部分历史脚本有旧路径或 SSH socket，不应逐个直接运行。
- `AIDE_ACT_dataset_20261111_195100/act/`：ACT / interACT 项目源码；没有复制原始 28 GB 数据集。
- `vendor/lerobot/`：电脑当前环境实际使用的 LeRobot 源码快照及原有许可资料。
- `requirements/`：电脑 ACT 环境的依赖版本记录。
- `evidence/20261116/`：可直接在 GitHub 查看结果、对照报告和末帧图像。
- `manifests/`：源文件 SHA-256、设备环境、权重身份、附件校验和。

完整 ROS bags、所有相机帧、动作块、伺服轨迹、模型权重和 Jetson 必需运行库在同一仓库的 [snapshot-20261116 Release](https://github.com/Krriat/Foxconn/releases/tag/snapshot-20261116)，不是只上传摘要：

| 附件 | 内容 / 解包布局 |
|---|---|
| `foxconn-physical-runs-20261116.tar.gz` | 整个 `training_verification/box_retest_20261116_1005/`，含失败对照、远盖成功回放、两次近盖 ACT 录包及完整日志 |
| `foxconn-act-checkpoints-20261116.tar.gz` | 当前远盖 ACT、成功近盖 ACT、参考回放所对应的原 014000 ACT 权重；加载所需训练元数据；远/近盖历史成功参考轨迹 |
| `foxconn-jetson-runtime-assets-20261116.tar.gz` | `devices/jetson/` 布局下的原生可执行文件、SDK 库、ROS 接口安装文件、吸盘库和初始位置文件 |

下载并校验（需要仓库读取权限和 `gh` 登录）：

```bash
git clone https://github.com/Krriat/Foxconn.git
cd Foxconn
bash scripts/download_assets.sh
```

脚本只下载和校验，不启动机器人、不覆盖设备配置。完整素材下载后位于 `downloads/`。

## 权重与恢复说明

成功近盖 ACT：

```text
model_outputs/act_near_success_20261112/pretrained_model
ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8
```

本次测试结束恢复的远盖 ACT 配置：

```text
model_outputs/act_far_current_recovery_20261113_03/checkpoints/001000/pretrained_model
af60ccd4cfca37c420d1107e440c8fcb78904d22a28e802d39b4b5925a9055f9
```

这个远盖 ACT 不是今天成功远盖回放的算法。回放工具为 `training_tools/far_recovery/reference_session.py` / `reference_actor.py`，实际成功配置和哈希见完整附件中的 `successful_setup/`。回放使用原 `014000` 模型身份和保存的参考轨迹，期间仍使用新 run ID、双臂归位和原保护机制。

本仓库是实际安装目录的快照，并非一键安装包。恢复时：

1. 将对应设备的 `robot-*` 目录恢复到该用户的 `~/.local/share/`；代码保留原绝对路径，需核对主机用户名和安装位置。
2. 电脑的项目源码、`training_tools/`、权重和训练元数据对应 `/home/jj/code/ACT/`。权重和实机数据附件以这个目录为解包根。
3. Jetson 的 `robot_source/` 对应 `/home/jetson/AIDE_robot_main/`。源码中 CMake 所引用 SDK、USR-M050 库和 ROS 接口都在源码或运行附件中。
4. 所有来源、校验和及网络拓扑见 `manifests/`。保留原安全限制；归档动作本身不执行安装或启动。

本次当前代码仍使用人工阶段完成服务；归档不启用箱盖判定或自动切换。近盖单阶段实测调用的是 `start_near`，会先归位。`finish_near` 是结束并释放吸盘，不能在刚启动时调用。

仓库不含 SSH/GitHub 登录凭据；原有第三方代码许可证保留。尚未验收的候选模型只保存训练审计信息，没有替换成功近盖权重。

