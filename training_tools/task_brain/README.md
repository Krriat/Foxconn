# Qwen3.5-9B LoRA 任务大脑

本组件直接在 Qwen3.5-9B 上训练 LoRA，输入箱体图像和当前 ACT 阶段，输出场景与任务完成状态。没有独立的 ResNet 分类器，也没有“大模型复核分类器”的第二条识别链路。

## 模型与输入

- 基座：官方 `Qwen/Qwen3.5-9B`，ModelScope 固定版本及各文件 SHA-256 见 `model_outputs/base/Qwen3.5-9B/source_manifest.json`。
- 为适配 RTX 5070 Ti 16GB，基座语言线性层使用 NF4；训练真正的 LoRA A/B 参数，属于 QLoRA。原始 BF16 权重与现有 Ollama 模型均保留。
- LoRA：rank 8、alpha 16、dropout 0.05，语言部分第 24–31 层注意力/MLP，加视觉投影层，共 5,156,864 个可训练参数。
- 视觉主干冻结；直接通过 Qwen 视觉编码器理解图像。输入取头部相机的箱体区域 `x=100:580, y=90:480`。腕部画面继续用于 ACT；任务大脑排除腕部画面，避免把腕部姿态当成箱盖完成证据。
- 只向模型提供当前阶段 `far/near`，不提供采集段编号、帧号、任务耗时、目标关节或吸盘状态来暗示完成。
- 输出为 `scene=closed/far_open/both_open/unknown` 和 `status=running/complete/uncertain`。部署通过同一个 LoRA 模型对四种合法文本响应的条件似然评分读取结果；这不是另一个分类网络。评分间隔不是校准后的概率。

输入、提示词和推理方式必须与已保存模型配套；不要直接把此 NF4 适配器加载到原 Ollama GGUF 模型上。

## 数据与评估

原数据共 58 段。按整段分为 44 段训练、6 段验证、8 段开发回放，拆分表见 `training_verification/task_brain_20261112/dataset/dataset.json`。检查过所有示教的开始/结束画面，使用短初始窗口、终止窗口，并补充视觉审核过的前期运动负样本。训练包含亮度增强和明确标记的遮挡/不可见合成样本。

8 段开发回放和早期实机录像曾用于发现中途误判、调整输入设计，因此最终报告不把它们称为完全未参与开发的独立测试集。所有这些回放均未参与梯度训练。训练未使用实际成功/失败录像的图像作为训练样本。

端点准确率不能代替整段行为测试。最终结果见 `training_verification/task_brain_20261112/` 中各评估报告与 `release_manifest.json`；第一、二轮及废弃的 ResNet 准备代码也保留在证据目录，便于追溯。未测试过的箱体、相机位置、严重遮挡和真实故障仍需补充验证。

## 只观察：不会发送切换请求

模型服务器 `jj@172.16.12.16`：

```bash
bash /home/jj/code/ACT/training_tools/task_brain/start.sh observe
```

只观察进程不创建机器人服务客户端，仅订阅图像/执行器状态、发布 `/robot/policy/brain_status`。它不调用归位、上电、ServoJ、吸盘或任务启动服务。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic echo /robot/policy/brain_status
```

## 以后实际使用自动切换

以下会启用未来任务中的自动阶段完成，属于实机运行操作；本次验证不执行这些启动步骤。

模型服务器先切换大脑模式：

```bash
systemctl --user stop robot-policy-brain.service
bash /home/jj/code/ACT/training_tools/task_brain/start.sh auto
```

等待 `/robot/policy/brain_status` 显示 `ready: true`、`transitions_enabled: true`。然后在 Jetson 的 ROS 环境调用：

```bash
ros2 service call /robot/policy/start_sequence_brain std_srvs/srv/Trigger '{}'
```

流程为：双臂归位并验证实测位置 → 远盖 ACT → LoRA 连续确认完成 → 停止并释放吸盘 → 双臂重新归位并验证 → 近盖 ACT → LoRA 连续确认完成 → 停止并释放吸盘。

原来的 `/start_sequence` 仍是手动阶段切换模式，使用 `/finish_far` 和 `/finish_near`。自动大脑不会结束手动模式中的阶段；操作者仍可手动完成或停止自动序列。

首次启动需要加载和预热。画面过暗、过曝或缺乏可用对比度时，大脑输出 `unknown/uncertain`，`ready` 保持 false，自动序列不接受启动；这只是图像可见性检查，不判断初始箱盖位置。观察模式仍会持续检查新画面。

```bash
ros2 service call /robot/policy/stop std_srvs/srv/Trigger '{}'
```

仅停止大脑观察进程：

```bash
systemctl --user stop robot-policy-brain.service
```

停止大脑不会替代机械臂 `/stop`；若自动任务正在运行而大脑退出，执行器会保留现有超时保护。

## 切换约束与无运动测试

阶段至少运行 5 秒；要求多个不同的新图像连续确认完成，且状态、观测和模型结果足够新鲜。`/robot/policy/brain_finish` 使用带任务 UUID、序列 UUID、阶段、图像时间和适配器哈希的请求，执行器逐项验证。旧结果、错误模型、归位前图像和手动序列均拒绝自动完成。近盖不会继承远盖的判断。

单元测试检查迟到/重复判断、相反场景和过期数据。真实执行器状态机在 `ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1` 下连接模拟双臂，验证每次 ACT 前都须确认双臂归位。测试进程没有加载机械臂 SDK，未连接物理控制器。真实相机测试只使用观察模式及原有无运动预览模式。

训练环境 `.venv-brain` 与 ACT 的 `.venv-act` 分开；新增包只装在前者，后者版本保持原样。大脑常驻时不要在 Ollama 中同时再载入另一份 9B 权重，以免挤占 GPU 显存。

## 复现入口

```bash
cd /home/jj/code/ACT
.venv-brain/bin/python -m training_tools.task_brain.download_base
.venv-brain/bin/python -m training_tools.task_brain.train_lora --help
.venv-brain/bin/python -m training_tools.task_brain.evaluate_lora --help
.venv-brain/bin/python -m unittest training_tools.task_brain.test_decision -v
```

模型的精确训练参数、续训来源、日志、保留适配器和最佳步数均在 `model_outputs/task_brain_lora_20261112/run_*/`。部署入口固定到 `model_outputs/task_brain_lora_20261112/adapter/`，发布前会与测试过的权重哈希核对。

参考：[官方模型](https://huggingface.co/Qwen/Qwen3.5-9B)、[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[PEFT 量化与 LoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)。
