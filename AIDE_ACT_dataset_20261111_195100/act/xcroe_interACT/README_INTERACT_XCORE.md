# xCore 双臂 RGB InterACT（可选头部 3D Encoder）

## 2026-11-13：远盖、近盖分别重训

已完成两路独立 RGB-only interACT 训练，无需头部点云。ResNet18 使用 ImageNet 初始化，
其余层随机初始化，任务各自使用训练数据计算归一化参数。没有替换现有 ACT 或大脑，未驱动机械臂。

| 任务 | 实际训练步数 | 最佳权重步数 | 原混合 interACT 验证误差 | 新模型验证误差 |
| --- | ---: | ---: | ---: | ---: |
| 远盖 | 19,000 | 11,000 | 2.9294° | 3.0291° |
| 近盖 | 13,000 | 5,000 | 2.5418° | 2.6761° |

误差是完整验证集有效动作目标上 14 个关节的平均绝对角度误差。
这轮两路均未优于原混合 interACT，不代表实机开盖效果已有提升。
权重目录、校验和、完整结果见[本次训练报告](../../../model_outputs/interact_task_pair_fresh_20261113/README.md)，
独立训练命令和参数见[分任务训练说明](../../../training_tools/interact_tasks/README.md)。

## 当前训练机的实测配置（2026-11-11）

`/home/jj/code/ACT` 已准备独立训练环境和按原始示教分组的数据。训练 RGB interACT：

```bash
/home/jj/code/ACT/.venv-act/bin/python /home/jj/code/ACT/training_tools/run_training.py interact
```

依次训练两套模型，可执行 `bash /home/jj/code/ACT/training_tools/train_both.sh`。
入口自动设置本机环境、源码和数据路径，并按时间创建新的输出目录。

实测选定：batch **8**、梯度累积 **1**、BF16、**4** 个数据加载进程、主学习率 **1e-4**、
ResNet18 学习率 **1e-5**、权重衰减 **1e-4**、梯度裁剪 **10**、chunk **50**、seed **1000**。
三路 RGB 缩放为 **224×224**，`d_model=256`，3 层交互编码，8 个注意力头，
左右臂各 2 层预解码和 2 层后解码。输入和输出均为 16 维双吸盘布局，不需要点云。

数据实际 **20 Hz**；训练/验证各 52/6 条原始示教，导出后各 60/7 个连续片段、
25,063/2,803 帧，原始示教分组不重叠。过滤了 132 帧过短片段，原始 PNG/NPZ 保留。
训练上限 **60,000** 步，每 **1,000** 步全量验证，每 **5,000** 步常规保存；
验证改善时额外保存，最佳权重位于输出目录的 `checkpoints/best/pretrained_model`。

[最终参数](../../../training_tools/validated_config.json) 和
[验证记录](../../../training_verification/final_report.json) 保存本轮运行依据。
batch 2×累积4、4×累积2、8、16、32、64，以及 FP16/BF16 和加载进程数均有实测记录。
短测配置通过运行检查，完整收敛及任务效果仍由后续训练验证。
当前右吸盘标签全为 0，无法据此学习开启右吸盘。

续训添加 `--resume-from /实际输出目录/checkpoints/last`；`--steps` 表示累计目标步数。
`--dry-run` 显示命令，`--smoke` 执行 20 步短测。

这是一个独立工程，不依赖 `xcore_act` 的源码。它覆盖双臂数据采集、数据审计与筛选、训练、离线评估和安全部署。相机获取方式与 `xcore_act` 一致：只订阅公共 ROS 2 相机话题，不直接打开 RealSense。

默认模型只需要头部、左腕、右腕三路 RGB 和 16 维状态，不需要头部深度、相机内参或点云。
训练、采集和数据校验只有显式添加 `--use-point-cloud` 才启用原 3D 分支。
旧 `interact3d` checkpoint 加载时自动保留原四分组结构；新 RGB checkpoint 保存为 `model_type=interact_rgb`。
脚本及 Python 类保留原文件名以兼容现有调用。

### 当前键盘吸盘数据

训练器按 LeRobot 特征名称自动识别两种 16 维排列：

- 双夹爪：左 7 关节、左夹爪、右 7 关节、右夹爪。
- 双吸盘：左 7 关节、右 7 关节、左吸盘、右吸盘；吸盘开=1、关=0。

双吸盘数据在编码器中按各臂 7 关节和对应吸盘分组，模型输出仍保持原数据排列，
归一化、动作损失和离线评估沿用原维度。布局记录在 checkpoint 的 `state_layout` 中，
读取不匹配布局的数据会报错。现有实机部署器仅控制舵机夹爪，不能用于吸盘 checkpoint。

`act_teach_keyboard_suction_30hz` 的实际采样率取 `schema.json`，当前为 **20 Hz**。
原始 PNG/NPZ 需通过 `xcore_act/scripts/dataset_tools/export_teach_lerobot.py` 导出，
不需要补造点云；连续片段会导出为独立 LeRobot episode。
在准备好的 LeRobot 0.4.3 训练环境中，审阅数据检查报告后可执行：

```bash
DATA_ROOT=/home/jj/code/ACT/AIDE_ACT_dataset_20261111_195100
export XCORE_INTERACT_ROOT="$DATA_ROOT/act/xcroe_interACT"
export PYTHONPATH="$XCORE_INTERACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python "$DATA_ROOT/act/xcore_act/scripts/dataset_tools/export_teach_lerobot.py" \
  --root "$DATA_ROOT/act_teach_keyboard_suction_30hz" \
  --destination "$DATA_ROOT/act_teach_keyboard_suction_20hz_lerobot" \
  --repo-id local/dual_arm_suction
python "$XCORE_INTERACT_ROOT/scripts/training/train_interact3d.py" \
  --dataset-root "$DATA_ROOT/act_teach_keyboard_suction_20hz_lerobot" \
  --dataset-repo-id local/dual_arm_suction \
  --output-dir "$DATA_ROOT/model_outputs/interact_rgb_suction_smoke" \
  --steps 10 --save-freq 10 --log-freq 1 --batch-size 2 --grad-accum 1
```

以上导出和模型输出目录须为新目录。当前这批原始数据的右吸盘标签全为 0，
因此不能从中学习开启右吸盘；采集末尾缺帧等质量提示也应结合画面复核。
当前训练机需单独准备训练环境；下文 `environment/activate.sh` 使用旧 Jetson 的绝对路径，
不能直接作为 `/home/jj` 训练机的环境安装脚本。

## 当前硬件和数据定义

- 左臂：机器人 `192.168.1.160`，本机网卡 `192.168.1.10`。
- 左臂摇操/复位初始关节位：`{1.66969, 1.365856, -1.272963, 1.476893, -1.735811, -0.026802, 0.165558}`。
- 右臂：机器人 `192.168.2.160`，本机网卡 `192.168.2.10`。
- 头部 D435i：默认只用 ROS 彩色；可选对齐深度生成 `1024×6` 的 `XYZRGB` 点云。
- 左腕 D405：ROS `/left_wrist/color/image_raw`（640x480@30），只用 RGB。
- 右腕 D405：ROS `/right_wrist/color/image_raw`（640x480@30），只用 RGB。
- 原舵机夹爪采集器的状态和动作为 `左 7 关节 + 左夹爪 + 右 7 关节 + 右夹爪`；训练器也支持上述键盘双吸盘 schema。
- 左夹爪已安装：舵机 ID `1`，转换器 `5B79015009`，话题为 `/left_sts3215/position` 和 `/left_sts3215/state`。
- 夹爪标定：左臂张开/闭合=`4000/3600`，右臂张开/闭合=`1500/900`。PICO 扳机按下表示张开，松开表示闭合。

相机硬件统一由 `hardware/robot_vision_ws` 中的官方 RealSense 驱动打开。InterACT 的采集、检查和部署都是 ROS 消费者，可与 `xcore_act`、Web 预览同时订阅，不会再争抢 USB 设备。头部 RGB 和深度仍要求稳定的 USB 3.x 链路。

VR 视频由网页总览中的“发送 VR 视频”按钮统一控制。独立节点
`hardware/robot_vision_ws/src/robot_vision_tools/robot_vision_tools/udp_video_sender.py`
订阅三路公共 ROS 彩色话题，并分别发送到 UDP 6000、6001、6002。该链路与
PICO 手柄输入使用的 UDP 5050 相互独立。

公共 ROS 话题契约：

| 输入 | ROS 2 话题 | 格式 |
| --- | --- | --- |
| 头部彩色 | `/head/color/image_raw` | `bgr8`, 960x540 |
| 头部对齐深度 | `/head/aligned_depth_to_color/image_raw` | `16UC1` 毫米, 960x540 |
| 头部深度内参 | `/head/aligned_depth_to_color/camera_info` | 彩色坐标系内参 |
| 左腕彩色 | `/left_wrist/color/image_raw` | `bgr8`, 640x480 |
| 右腕彩色 | `/right_wrist/color/image_raw` | `bgr8`, 640x480 |

所有订阅使用 sensor-data QoS。默认只订阅三路 RGB；启用点云时才订阅头部对齐深度和内参，检查头部彩色与深度的时间对齐。过期或错分辨率的数据会被拒绝。

## 模型不是普通 ACT 改名

编码器保留 InterACT 的两级交互：先在各 segment 内做 attention，再让各 segment 的 CLS token 做跨 segment attention。默认三段分别是：

1. 左臂 8 维状态；
2. 右臂 8 维状态；
3. 头部、左腕、右腕的共享 ResNet18 RGB token；

启用 `--use-point-cloud` 时才增加第四段：头部对齐深度生成的 PointNet 风格 XYZRGB token。

解码器为左右臂分别做 pre-decoding，中间通过 multi-head synchronization attention 交换双臂信息，最后分别 post-decode 并输出两个 8 维 action head。RGB 模型不会创建或读取点云分支；可选 3D 模型额外使用真实三维坐标。

默认模型为 `d_model=256`、3 个 interaction block、128 个点云 token；这是针对 Orin 16 GB 统一内存的稳妥起点。训练默认物理 batch 2、梯度累积 4，等效 batch 8。

## 1. 激活环境

```bash
source /home/niic/aide/software/vla/act/xcroe_interACT/environment/activate.sh
export XCORE_INTERACT_DATASETS_ROOT=/home/niic/aide/assets/dataset/xcroe_interACT/datasets
export XCORE_INTERACT_MODEL_ROOT=/home/niic/aide/assets/model/xcroe_interACT/model_outputs
```

工程变量为 `$XCORE_INTERACT_ROOT`。不要再使用旧的 `$XCORE_ACT_ROOT`。

## 2. 启动 Python 双臂 PICO ServoL

先运行纯软件自检；它不会连接或移动机器人：

```bash
source "$XCORE_INTERACT_ROOT/environment/activate.sh"
python "$XCORE_INTERACT_ROOT/scripts/teleoperation/pico_dual_servol.py" --self-test
```

正常启动会连接并允许 PICO 驱动两条机械臂，因此必须先清空工作空间、确认急停有效，再执行：

```bash
python "$XCORE_INTERACT_ROOT/scripts/teleoperation/pico_dual_servol.py" \
  --confirm-motion=I_UNDERSTAND_DUAL_ARM_MOTION
```

不要同时运行旧的 `pico_servol`、`pico_dual_servol`、`pico_dual_recording`
或 `receive.py`，它们会争用 UDP 5050 或机器人控制会话。

左右手柄逻辑与 `xcore_act` 单右臂摇操一致，并且彼此独立：

- 左 Grip 单击：开启左臂 ServoL；再次单击：停止左臂 ServoL。
- 右 Grip 单击：开启右臂 ServoL；再次单击：停止右臂 ServoL。
- 左 Down：停止左臂摇操，只让左臂回默认位置；右臂不复位。
- 右 Down：停止右臂摇操，只让右臂回 `xcore_act` 默认位置；左臂不复位。
- 右 Up/B：通过 `/xcore/dual/pico_buttons` 转发给录制器；左 Up 明确不使用。
- 左/右 Trigger 分别控制对应夹爪。按下均为张开：左 `4000`、右 `1500`；松开均为闭合：左 `3600`、右 `900`。

单臂复位在该臂自己的后台线程执行，另一只手仍能立即开停另一条机械臂。任意一臂复位期间 `/xcore/dual/teleop_active` 会暂时为 false，录制器不会把复位轨迹写进示教数据。

默认复位关节位：

- 左臂：`{1.66969, 1.365856, -1.272963, 1.476893, -1.735811, -0.026802, -0.165558}`。
- 右臂：`{-1.969919, 1.365856, 1.272963, 1.476893, 1.735811, -0.026802, 0.165558}`。

位置增益默认与 `xcore_act` 相同为 `0.618`，仍可使用
`XCORE_TELEOP_POSITION_GAIN=0.3` 或 `--position-gain=0.3` 调整，允许范围
`0.1~1.0`。旋转增益、50 Hz 频率、单周期 `1 mm / 0.5°` 限幅、PICO
跳变拒绝、250 ms 超时和 12° 关节余量也与现行单臂逻辑一致。

Edge API 启动程序时还会传入 `XCORE_TELEOP_ELBOW_COMMAND_PATH` 和
`XCORE_TELEOP_ELBOW_STATUS_PATH`。PICO 程序启动后无需按 Grip，网页即可分别
更新左右臂角；对应机械臂会保持当前末端位姿并进入臂角专用 ServoL，该过程
不发布为采集遥操状态。两臂均以 `2°/s` 最大速度、`2°/s²` 最大加速度平滑
跟随；每个 `50 Hz` ServoL 周期另有 `0.01°` 硬限幅，因此实际连续速度不超过
`0.5°/s`，较远目标只能逐周期逼近。每条笛卡尔指令都会设置 `hasElbow=true`
并携带当前臂角目标。关闭网页臂角控制会把目标重置为关闭时的当前臂角；按
Grip 会退出专用模式并进入正常手柄遥操。

Python 控制器发布：

- `/xcore/dual/record_state`
- `/xcore/dual/record_action`
- `/xcore/dual/teleop_active`
- `/xcore/dual/pico_buttons`（只转发右手柄完整按键字节）

右夹爪使用 `/sts3215/position` 和 `/sts3215/state`；左夹爪使用 `/left_sts3215/position` 和 `/left_sts3215/state`。双夹爪状态必须都可用，采集控制器才会发布完整的 16 维样本。

## 3. 无运动检查

只订阅 ROS 相机并检查点云（不会打开相机硬件）：

```bash
python "$XCORE_INTERACT_ROOT/scripts/diagnostics/preflight.py" --cameras
```

只以只读方式连接两条机械臂并读取 16 维状态，不会发运动指令：

```bash
python "$XCORE_INTERACT_ROOT/scripts/diagnostics/preflight.py" --robots
```

## 4. 采集双臂 RGB 数据（可选深度）

先确认公共相机节点已发布三路 RGB，再启动双夹爪 ROS 驱动和上面的 Python `pico_dual_servol.py`，最后开另一个终端运行。可选 3D 采集还需深度和内参，并添加 `--use-point-cloud`：

```bash
ros2 topic hz /head/color/image_raw
ros2 topic hz /left_wrist/color/image_raw
ros2 topic hz /right_wrist/color/image_raw
```

```bash
source "$XCORE_INTERACT_ROOT/environment/activate.sh"
export DATASET_NAME="bimanual_pick_$(date +%Y%m%d_%H%M%S)"
python "$XCORE_INTERACT_ROOT/scripts/recording/record_dual_rgbd.py" \
  --dataset-root="$XCORE_INTERACT_DATASETS_ROOT/$DATASET_NAME" \
  --repo-id="local/$DATASET_NAME" \
  --task='pick up the objects with both arms' \
  --fps=10 \
  --num-episodes=50 \
  --episode-time-s=20 \
  --segments-per-episode=4 \
  --segment-title='left arm approaches the object' \
  --segment-title='left arm grasps the object' \
  --segment-title='both arms transfer the object' \
  --segment-title='right arm places the object'
```

录制器本身不控制机械臂，只读取 Python ServoL 控制器发布的同时间戳双臂状态/动作并采集相机。每个 Episode 必须按顺序完成 1–8 个子任务：每个子任务按一次右手柄 Up/B 开始，再按一次完成；对应采集帧会把该子任务描述写入 LeRobot `task` 字段。只有所有子任务都完成才保存整个 Episode；中途安全结束时会丢弃不完整 Episode。子任务完成后 0.5 秒内再次按 Up/B 会丢弃当前 Episode，左手柄 Up 不参与录制。

创建采集任务时只检查采集器和相机；网页或右手柄 Up/B 请求开始后，如果左右 Grip 尚未开启 ServoL，就保持“等待遥操”且不写帧；收到 `teleop_active=true` 和新鲜双臂样本后才开始写入。录制期间遥操短暂失活或样本过期时会暂停写帧，恢复新鲜样本后继续，不会把旧样本写入或直接结束整个任务。需要旧的 Enter + 固定时长方式时加 `--keyboard-control`，无人值守定时录制可加 `--no-prompt`。相机失败、无法恢复的程序异常或循环落后仍会终止当前未保存 Episode，避免写入半截数据。

三路 RGB 保存在视频中。默认不保存点云；启用点云时额外保存在 Parquet 数值列。腕部相机深度不送进模型。

## 5. 审计和筛选

```bash
export RAW_ROOT="$XCORE_INTERACT_DATASETS_ROOT/$DATASET_NAME"
python "$XCORE_INTERACT_ROOT/scripts/dataset_tools/validate_dataset.py" \
  --dataset-root="$RAW_ROOT"
```

报告写到 `meta/validation_interact3d.json`，检查 16 维 schema、NaN/Inf、关节和夹爪/吸盘范围。添加 `--use-point-cloud` 才检查点云及有效点比例。按报告只复制合格 episode：

```bash
export CLEAN_ROOT="${RAW_ROOT}_clean"
python "$XCORE_INTERACT_ROOT/scripts/dataset_tools/filter_episodes.py" \
  --source-root="$RAW_ROOT" \
  --destination-root="$CLEAN_ROOT" \
  --source-repo-id="local/$DATASET_NAME" \
  --destination-repo-id="local/${DATASET_NAME}_clean" \
  --validation-report="$RAW_ROOT/meta/validation_interact3d.json"
```

也可以人工指定：`--keep=0,1,3,7`。目标目录必须为空，工具不会覆盖已有数据。

## 6. 训练

先跑 10 step smoke test，确认显存峰值和数据解码正常：

```bash
export TRAIN_ROOT="$CLEAN_ROOT"
export RUN_DIR="$XCORE_INTERACT_MODEL_ROOT/interact3d_smoke_$(date +%Y%m%d_%H%M%S)"
python "$XCORE_INTERACT_ROOT/scripts/training/train_interact3d.py" \
  --dataset-root="$TRAIN_ROOT" \
  --dataset-repo-id="local/${DATASET_NAME}_clean" \
  --output-dir="$RUN_DIR" \
  --batch-size=2 \
  --grad-accum=4 \
  --num-workers=1 \
  --steps=10 \
  --save-freq=10 \
  --log-freq=1
```

正式训练：

```bash
export RUN_DIR="$XCORE_INTERACT_MODEL_ROOT/interact3d_$(date +%Y%m%d_%H%M%S)"
python "$XCORE_INTERACT_ROOT/scripts/training/train_interact3d.py" \
  --dataset-root="$TRAIN_ROOT" \
  --dataset-repo-id="local/${DATASET_NAME}_clean" \
  --output-dir="$RUN_DIR" \
  --batch-size=2 \
  --grad-accum=4 \
  --num-workers=1 \
  --chunk-size=50 \
  --steps=60000 \
  --save-freq=10000 \
  --log-freq=100
```

这是一套独立训练循环，不调用 `lerobot-train`。checkpoint 位于 `checkpoints/060000/pretrained_model`，同时保存模型 config、normalization、optimizer、AMP scaler 和随机数状态。断点续训示例：

```bash
python "$XCORE_INTERACT_ROOT/scripts/training/train_interact3d.py" \
  --dataset-root="$TRAIN_ROOT" \
  --dataset-repo-id="local/${DATASET_NAME}_clean" \
  --output-dir="$RUN_DIR" \
  --resume-from="$RUN_DIR/checkpoints/030000" \
  --steps=60000
```

默认加载 ImageNet ResNet18 权重；完全离线且本机没有缓存时可加 `--no-pretrained-backbone`，但通常收敛会更慢。

## 7. 离线评估

把从未参与训练的 episode（例如 45–49）作为 held-out：

```bash
export POLICY="$RUN_DIR/checkpoints/060000/pretrained_model"
python "$XCORE_INTERACT_ROOT/scripts/evaluation/evaluate_offline.py" \
  --policy-path="$POLICY" \
  --dataset-root="$TRAIN_ROOT" \
  --dataset-repo-id="local/${DATASET_NAME}_clean" \
  --episodes=45,46,47,48,49 \
  --output="$RUN_DIR/offline_eval.json"
```

## 8. 部署

先跑真实相机、真实双臂状态和真实模型，但绝不发送命令：

```bash
python "$XCORE_INTERACT_ROOT/scripts/deployment/deploy_interact3d.py" \
  --policy-path="$POLICY" \
  --duration-s=20
```

确认预测、相机、循环时序和急停后，才允许真实运动：

```bash
python "$XCORE_INTERACT_ROOT/scripts/deployment/deploy_interact3d.py" \
  --policy-path="$POLICY" \
  --duration-s=20 \
  --enable-motion \
  --confirm-motion=I_UNDERSTAND_DUAL_ARM_MOTION
```

部署器以 50 Hz 在后台分别维持左右 ServoJ，策略以 10 Hz 运行；每次目标相对当前观测最多变化 `0.035 rad`，关节还会经过 12° 安全边界裁剪。策略循环超过 250 ms 或 ServoJ 主机循环严重超时会报错并进入清理路径。默认连续执行预测 chunk 的前 10 点，正好覆盖约 1 秒，不做隔 3 点抽样。

## 9. 双夹爪状态与兼容模式

双舵机开机入口已经部署，但当前服务尚未重启；下一次开机后新左夹爪话题才会由 `aide-startup.service` 自动提供。在此之前可以手动启动 `dual_sts3215.launch.py`，但不要同时启动两份驱动占用同一串口。

新数据默认同时记录并训练左右夹爪。左、右夹爪安装方向相反，部署器按各自标定端点做最近邻量化，不使用统一的“大值=闭合”假设。

`--left-gripper-disabled` 只用于读取没有左夹爪的旧数据或临时诊断；用该兼容模式训练的 checkpoint 不应控制左夹爪。已有右臂 ACT 数据的开合标签与本工程双夹爪标定不一定一致，不要直接混入新 InterACT 数据集。

## 目录

```text
src/xcore_interact/        模型、点云、ROS 相机订阅、双臂 ServoJ 与数据封装
native/                    双 PICO 采集控制器及 ROS 遥测
scripts/teleoperation/     Python 双手柄、双臂 ServoL 与 ROS 遥测
scripts/recording/         数据采集
scripts/dataset_tools/     审计、筛选
scripts/training/          独立训练和 checkpoint
scripts/evaluation/        离线评估
scripts/deployment/        无运动 dry-run 与真实部署
scripts/diagnostics/       相机/机器人只读检查
tests/                     点云、schema、模型和夹爪占位测试
```

InterACT 来源和许可证见 `THIRD_PARTY_NOTICES.md` 与 `LICENSE`。

## 当前吸盘开箱实机部署：两阶段执行

2026-11-12 新增“远侧开盖完成 → 验证释放与回撤 → 归位 → 近侧开盖”的执行器流程。
当前接入的是已部署的 ACT 检查点；详细启动命令、完成条件、验证结果及限制见
[两阶段开箱说明](../../../training_tools/two_stage_policy/README.md)。
完整任务使用 `/robot/policy/start_sequence`，远侧已打开时使用 `/robot/policy/start_near`；
原来的 `/robot/policy/start` 仍是单轮执行。
每次接受 `start_sequence` 启动请求后，先归位并确认双臂关节到位，再检查箱盖画面、开始推理。
