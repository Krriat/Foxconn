# 远盖、近盖两个 RGB interACT 专项模型

入口只做离线训练和离线评估，没有 ROS、吸盘或机械臂接口，也不改部署配置。
正式方案为两个任务独立训练：ResNet18 使用 ImageNet 初始化，interACT 层随机初始化。
两个任务各自创建模型、归一化参数、优化器和权重输出。
原混合任务 interACT 的最佳权重（第 29,000 步）只作离线对照，不用于正式模型初始化。

## 数据与参数

- 使用原数据导出的连续片段，通过 `keyboard_teach_source.json` 映射回原始示教编号。
  偶数原始编号为远盖，奇数为近盖；不能按导出后的 episode 编号判断任务。
- 沿用原始整条示教划分：远盖 27 条训练、2 条验证（1,079 帧）；
  近盖 25 条训练、4 条验证（1,724 帧）。同一示教所有片段属于同一侧。
- 实际采样率 20 Hz。输入头部、左腕、右腕三路 RGB（各 224×224）及 16 维状态；
  输出 50 步×16 维动作。`use_point_cloud=false`，无需深度或点云。
- 各自只用对应任务训练集的有效连续片段计算归一化参数，不使用验证数据。
- batch 8，梯度累积 1，BF16 训练，4 个加载进程，seed 1000；
  主学习率 1e-4，ResNet18 学习率 1e-5，AdamW，权重衰减 1e-4，梯度范数上限 10。
- 每任务最多 30,000 次优化更新；每 1,000 步完整验证，每 5,000 步常规保存，
  验证改善时额外保存。至少训练 10,000 步，连续 8 次验证未改善则提前停止。
- 用完整验证集、FP32 推理、14 个关节有效动作目标的平均绝对角度误差选择最佳模型。
  原模型基线单独保存为 `baseline.json`；第 0 步是各自新模型的初始化状态。
  所有模型均反归一化为真实角度和吸盘值后比较，避免不同统计量造成指标不可比。
- 架构和主训练参数沿用本机之前验证过的 RGB interACT 方案，没有声称全局最优。

## 启动与查看

在 `/home/jj/code/ACT` 执行，`--output` 必须是不存在的新目录：

```bash
.venv-act/bin/python -u training_tools/interact_tasks/run_pair.py \
  --output /home/jj/code/ACT/model_outputs/interact_task_pair_NEW
```

只训练一路：

```bash
bash training_tools/interact_tasks/run.sh --stage far \
  --output /home/jj/code/ACT/model_outputs/interact_far_NEW
bash training_tools/interact_tasks/run.sh --stage near \
  --output /home/jj/code/ACT/model_outputs/interact_near_NEW
```

每个任务目录：

- `pretrained_model/`：指向最佳模型，包含权重、配置和 `normalization.json`。
- `checkpoints/best`、`checkpoints/last`：最佳与最后保存的训练状态。
- `manifest.json`：任务、原始示教与片段、参数、源权重哈希及评价限制。
- `source_snapshot/`：训练器与实际 interACT 模型源码快照。
- `training.jsonl`、`validation.jsonl`、`status.json`：训练、全量验证与进度。
- `result.json`：最佳模型与原模型对比，含前 10 步、0/10/49 步误差、吸盘开启区间误差、
  吸盘阈值错误率、重载一致性与 batch=1 推理计时。

总目录的 `report.json` 汇总两个模型，并核对既有 ACT、大脑权重及本机机器人配置哈希未变。

## 本次运行

`model_outputs/interact_task_pair_20261113` 是首轮混合权重专项微调探测：
两个任务各训练 3,000 步，但完整验证关节误差均未优于原模型，故保留基线且不交付为新模型。
正式独立训练使用新的输出目录 `model_outputs/interact_task_pair_fresh_20261113`，
两轮的记录均保留。

## 验证范围

验证集用于挑选权重，不能称为独立测试集；远盖仅两条验证示教，覆盖有限。
离线评估使用示教图像和示教状态，不代表闭环开盖或吸附成功率。
纯模型推理计时使用已放在 GPU 上的张量，不含 ROS、网络、JPEG 解码和预处理。
原数据右吸盘标签全为关，模型无法学会主动打开右吸盘。
新模型未替换现有两阶段 ACT，未连接机械臂测试。
