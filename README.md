# Foxconn — Jetson 双 ACT + Qwen LoRA 自动调度

分支 `jetson-act-qwen-20261117` 基于旧 `main` 的 `a7e15b19a74f064e8d9fde2bf03753845b68cf04`，归档于 **2026-11-17（Asia/Shanghai）**。旧分支与旧 Release 保留。

**当前默认：两个 ACT 均在 Jetson Orin Nano 8GB 在线推理；电脑运行 Qwen3.5-9B LoRA 判断完成并切换；全程 1.0 倍速；每个 ACT 前双臂归位；右臂等待约束关闭。**

这里归档的是成功测试时的开盖程序与配置。导出时设备已进入另一轮示教采集，旧策略服务可能显示停止或不存在；服务状态单独保存在 `manifests/*services-20261117.txt`，导出没有重启服务或干预采集。

## 与旧快照的区别

| 项目 | 2026-11-16 旧快照 | 当前分支 |
|---|---|---|
| 远盖成功方式 | 保存轨迹回放，非 ACT 在线推理 | Jetson 远盖 ACT 在线推理 |
| 近盖推理位置 | 电脑推理、Jetson 执行 | Jetson 推理与执行 |
| 电脑职责 | ACT、参考轨迹与训练诊断 | Qwen3.5-9B NF4 + LoRA 完成状态判断 |
| 完成与切换 | 人工阶段完成服务 | Qwen 连续视觉确认 + 关节收尾检查 |
| 模型结构 | 双臂 ACT | 保持远盖、近盖两份双臂 ACT，未拆左右臂模型 |
| 新增右臂等待 | 无 | 当前关闭，另保存开启时的成功对照 |

详见 [版本差异](docs/CHANGES_20261117.md) 和 [旧版原始说明](docs/SNAPSHOT_20261116_BASELINE.md)。

## 补充：示教数采源码

同分支已补入 [data_collection/](data_collection/README.md)：Jetson 键盘示教、数据写入/校验/导出、双臂和头相机驱动，以及 umie 网络腕相机与继电器桥接。该目录保留设备原路径，并附启动说明和逐文件来源哈希；与本页成功推理快照分开保存，未改动 `devices/` 中的成功配置。数采使用原有 Domain 0，推理仍使用 Domain 1。

## 成功实机记录

| 试验 | 右臂等待约束 | 远盖阶段 | 近盖阶段 | 结果 |
|---|---|---:|---:|---|
| **当前默认** | **关闭** | **43.9 s** | **30.7 s** | 两盖打开，Qwen 自动切换，IDLE，吸盘关闭 |
| 先前对照 | 开启 | 62.2 s | 31.0 s | 两盖打开，Qwen 自动切换，IDLE，吸盘关闭 |

时间从各阶段 `RUNNING` 到 `STOPPING`，包含执行与完成确认、排除归位。两轮权重、IEEE FP32 引擎、1.0 倍速相同，均未用保存轨迹回放。每种设置各一次成功，不能推算长期成功率或证明此前失败只有一个原因。取消约束也成功，当前保持 `coordinate_far_right_arm: false`。

当前 ACT 推理中位数 **108.44 ms**、P95 **117.39 ms**、最大 **162.27 ms**。计算延迟未恢复到桌面约 9 ms；实际约 8–9 次/秒重规划，50 步动作保持 20 Hz 时间基准，经时间对齐、chunk fusion、插值和限幅，以 100 Hz 下发。

## 三台设备

| 设备 | 当前职责 | 代码目录 |
|---|---|---|
| jj `172.16.12.16` | Qwen LoRA、监督录像、离线工具；桌面 ACT 发布禁用 | `devices/pc/`、`training_tools/task_brain/` |
| Jetson `172.16.12.95` | 双 ACT、头相机、状态、原生 ServoJ | `devices/jetson/` |
| umie `172.16.12.45` | 双腕相机、有线继电器桥接 | `devices/umie/` |

ROS 2 Humble，Domain 1，`rmw_fastrtps_cpp`。Jetson 活动入口是 `devices/jetson/robot-policy-streams/run.sh`、`run_inference_jetson.sh` 和同目录 `config.yaml`，模型目录为 `jetson_act_runtime_20261116/`。模型目录保留的旧联调脚本不是活动执行器入口。

待命与启动说明见 [当前操作说明](training_tools/jetson_dual_act_20261116/README.md)。恢复原路径、依赖与模型后，在电脑开始真实运动：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act_20261116/run_sequence.sh
```

停止：

```bash
bash /home/jj/code/ACT/training_tools/jetson_dual_act_20261116/stop_sequence.sh
```

自动入口是 `/robot/policy/start_sequence_brain`；`/start_sequence` 仍是手动切换。监督录像由 systemd 用户服务保持，关闭启动终端不能代替停止。归档、下载和校验不会启动机器人。

## 完整附件

Git 保存源码、配置、报告与末帧；大文件在同仓库 [snapshot-20261117-jetson-act-qwen Release](https://github.com/Krriat/Foxconn/releases/tag/snapshot-20261117-jetson-act-qwen)：

| 附件 | 内容 |
|---|---|
| `foxconn-successful-runs-20261117.tar.gz` | 完整 `training_verification/no_right_hold_20261117_102134/` 与 `grasp_repair_20261117_095228/`：两轮录像、全部动作/伺服/Qwen 日志、逐帧时间、配置、对照输入与审计；排除临时模型链接 |
| `foxconn-models-20261117.tar.gz` | Jetson 两个 ACT 权重与归一化、电脑 Qwen LoRA 适配器；基座来源、配置、许可证（无 19 GB 基座权重） |
| `foxconn-jetson-runtime-assets-20261117.tar.gz` | 两个对应 TensorRT 引擎、原生执行器/状态程序、SDK/M050/ROS 接口、cuDSS 运行库 |

```bash
git clone --branch jetson-act-qwen-20261117 https://github.com/Krriat/Foxconn.git
cd Foxconn
bash scripts/download_assets.sh
```

脚本只下载并检查 SHA-256，不解包或安装。实机数据以原电脑 `/home/jj/code/ACT/` 为解包根；设备代码/资产按 `devices/` 布局还原。代码保留原用户名和绝对路径，不是任意设备的一键安装包。运行库需匹配现有 JetPack、ROS 2 Humble、Python 3.10 与 aarch64。

Qwen 基座使用既有安装，或依据 `manifests/qwen-base-source.json` 的固定版本、逐文件哈希获取；获取脚本为 `training_tools/task_brain/download_base.py`。LoRA 基座是 Safetensors 版本，不能用 Ollama GGUF 直接替代。

## 权重身份与范围

| 组件 | SHA-256 |
|---|---|
| 远盖 ACT | `af60ccd4cfca37c420d1107e440c8fcb78904d22a28e802d39b4b5925a9055f9` |
| 近盖 ACT | `ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8` |
| Qwen LoRA | `63e882d82b66a58da2269ce32f22dedc41b42959b2165d080a131ddeeabf0ade` |

本次没有重新训练或替换这些权重。引擎保持 IEEE FP32、FP16/TF32 禁用；没有启用试验性混合精度引擎或时钟锁定。开发工具中保留的其他模式不代表活动设置。

环境、启动入口、哈希与附件清单见 `manifests/`。不含 SSH/GitHub 凭据、虚拟环境、完整原始 28 GB 数据集和无关历史日志；保留第三方源码和许可证。
