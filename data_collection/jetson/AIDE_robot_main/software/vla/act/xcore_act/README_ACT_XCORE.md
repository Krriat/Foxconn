# xCore ACT 使用说明

本项目用于 xCore 右臂的数据录制、数据集处理、ACT 训练和模型评估。

## 当前键盘双臂双吸盘采集与训练

本流程使用 `scripts/recording/record_xcore_teach_suction.py`，不需要 Pico 或头腰电机。
16 维顺序固定为左臂 7 关节、右臂 7 关节、左吸盘、右吸盘。关节单位为弧度；吸盘开=1、关=0。
吸盘 observation 是逻辑 DO 读回，action 是键盘请求目标；两者按图像时间取过去样本，不代表吸附成功检测。

```bash
bash /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/start_xcore_teach_suction.sh \
  --restart-cameras --fps 20 --task "双臂吸盘抓取"
```

默认采样为 **20 Hz**，相机沿用原 **30 Hz** 输出，由采集器降采样。
源时间戳断点阈值随采样率计算：20 Hz 时为 **80 ms**（30 Hz 时为 53.3 ms）；
正常选帧的 33.3/66.7 ms 间隔可保持连续，100 ms 间断仍会切段。
配对相机和关节时，在同步容差内优先选择能保持源时间连续的实测样本，再比较与参考图像的时间差，
避免相机相位漂移时因极小的时间差优势跳过已有样本。真实缺帧仍按上述阈值分段。
继续使用原目录 `act_teach_keyboard_suction_30hz` 保存 20 Hz 数据，目录名保留历史名称；
目录内 `schema.json` 和每条记录中的 `fps` 才是实际采样率。
此前的 30 Hz 数据归档在该目录的 `archive_30hz_*` 子目录中，不参与新数据的直接导出。

双腕接电脑、通过有线网络传图时，先保持电脑端相机发送程序运行，再在 Jetson 使用网络入口：

```bash
bash /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/start_xcore_teach_suction_network.sh \
  --fps 20 --task "双臂吸盘抓取"
```

该入口使用电脑 `192.168.50.20:9800` 的双腕图像、Jetson 本机头部相机和 0.20 秒配对缓冲，
同时启动双臂服务；图像标签保留经时钟校正的拍摄时刻。

A 开示教，R 开始持续录制，1/2 控制左吸盘开/关，3/4 控制右吸盘开/关；
B 回已保存位置并继续录制，最后 S 保存；Z 结束示教或停止回位，D 丢弃未保存数据。
保存前校验时序及标签，失败的数据保留在临时目录，不提升为正式 episode。
保存成功后出现 `[训练数据需复核]` 表示文件已保存，但存在低帧率、末尾缺帧、短片段、未录到回位等质量问题。

批量训练前执行只读检查；`--chunk-size` 应与训练配置一致：

```bash
cd /home/jetson/AIDE_robot_main/software/vla/act/xcore_act
python3 scripts/diagnostics/check_teach_training.py \
  --root /home/jetson/AIDE_robot_main/assets/dataset/act_teach_keyboard_suction_30hz \
  --chunk-size 50
```

返回码 0 表示结构、时序与自动质量检查通过；1 表示数据无效；2 表示需人工复核。
检查包括三路全部图像、16 维标签、IO 因果时间、真实间断分段、整段保存帧率、连续动作块长度、
回位后是否有帧、数据集左右吸盘 0/1 覆盖。即使通过，也需观看画面确认任务成功和完整性。
同目录已有的调试、相机中断和失败记录不会被自动删除或改写；筛选后在单独目录组织正式训练数据。

在安装了 LeRobot 0.4.3 的训练环境中导出、训练（导出目标和模型输出须为新目录）：

```bash
python scripts/dataset_tools/export_teach_lerobot.py \
  --root /path/to/approved_raw_suction \
  --destination /path/to/new_lerobot_suction --repo-id local/dual_arm_suction
python scripts/training/train_act.py \
  --dataset-root /path/to/new_lerobot_suction --dataset-repo-id local/dual_arm_suction \
  --output-dir /path/to/new_act_suction_model --job-name dual_arm_suction \
  --chunk-size 50 --n-action-steps 10
```

导出将每个连续片段保存成独立 LeRobot episode，动作块不跨缺帧或控制模式切换。
ACT 从数据集特征推导 16 维输入/输出；应新建双吸盘模型，不能直接恢复旧右臂或头腰模型检查点。
当前 `train_act.py` 要求 CUDA 可用；先在训练环境确认 `torch.cuda.is_available()` 为真。

键盘双吸盘采集默认通过 `127.0.0.1:29701` 接收双臂实测关节，保留 SDK 采样时间；
相机继续通过 ROS 接收并按原时间戳配对。SDK 采样线程先发送本机数据，再将 ROS 话题交给
独立、有界的发布队列，DDS 发布阻塞不再阻塞该采样线程。元数据的
`capture_diagnostics.local_joint_feedback` 记录本机序号缺失、发送错误和 ROS 队列覆盖计数。
ROS 队列覆盖不等于本机关节通道丢包；真实样本缺失仍会分段，不补帧。

一键启动命令保持不变，会仅给本次双臂驱动设置 `AIDE_JOINT_FEEDBACK_PORT=29701`，
并在进入键盘界面前确认双臂本机反馈就绪。更新后须退出旧双臂驱动，再启动新版本。
单独启动驱动时，在同一条命令前设置该环境变量；单独启动采集器时沿用默认端口。
若需要另一个端口，给采集命令添加 `--joint-feedback-port 端口`，一键入口会同步设置驱动。
`--joint-feedback ros` 显式使用旧 ROS 反馈通道，供兼容检查；该通道仍可能受到 DDS 阻塞影响。
本机通道没有改动运动目标、速度、示教约束、按键或吸盘含义。控制器模式切换期间的真实反馈缺口仍需复测。

## 原 Pico 与头腰 ROS 双臂示教数采

本节对应 `/home/niic/aide/摇操步骤.md` 当前部署：右 A 同时启停双臂示教，各臂实体按钮控制是否可拖动；PICO 仍控制头部 PHA 和腰部俯仰。
旧单右臂 VR 录制流程保留在后文。新采集器不连接 SDK、CAN 或 RealSense，不调用电机启停／运动服务。

### 启动与按键

按摇操步骤启动双臂、头腰节点和全身 PICO；更新后须重启双臂驱动和 PICO，加载新增的只读反馈与按键话题。
首次或在另一台机器上更新驱动时，只编译，不启动机器人：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source install/local_setup.bash
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select rokae_driver --symlink-install
```

另开相机终端：

```bash
bash /home/niic/aide/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_cameras_30hz.sh
```

采集终端使用系统 Python 与当前 ROS 域，无需旧 ACT 环境脚本：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd /home/niic/aide/AIDE_robot_main/software/vla/act/xcore_act
python3 scripts/recording/record_xcore_teach.py \
  --root /home/niic/aide/assets/dataset/act_teach_30hz \
  --fps 30 \
  --udp-video-ip 172.16.13.184 \
  --task "双臂抓取并放置物体"
```

启动采集器后，复用现有三路图像订阅自动转发 UDP JPEG：头部 `172.16.13.184:6000`、左腕 `:6001`、右腕 `:6002`。默认 IP 即 `172.16.13.184`，可用 `--udp-video-ip` 改地址，`--no-udp-video` 关闭。无需另启独立视频发送节点；等待示教、录制暂停及 `--check` 期间均持续转发，退出时关闭。
报文与原 `robot_vision_tools/udp_video_sender.py` 相同：每个 UDP 数据报包含一张完整 JPEG，没有自定义包头。默认 JPEG 质量 50，超过 65,507 字节时逐步降低预览质量；不改变用于数采的无损 RGB。编码和发送在独立线程，每台仅保留最新待发送帧，发送拥塞或网络错误仅影响预览并报告警告。

右 A 开启双臂示教后，等待两侧 `teaching enabled`，再等至少 1 秒供实际频率检查。先松开右 B 一次，再按以下方式操作：

| 右 B 操作 | 效果 |
| --- | --- |
| 空闲时单击 | 立即开始录制一条新数据 |
| 采集中再次单击 | 立即停止采样；等待 350 ms 双击判定窗口后自动保存 |
| 双击（两次按下间隔 ≤350 ms） | 停止并丢弃当前未保存记录；已保存的 episode 保留 |

每次点击需松开再按，长按不连发；按钮间隔按 ROS 回调接收时刻判定。等待 `Saved:` 或 `Pending episode discarded` 后，再按 B 开始下一条。刚开始后立即双击，也会丢弃这一条。
缺图、反馈超时或示教退出仍会暂停采集；此时单击 B 保存已录部分、双击 B 丢弃，不自动续录。保存失败时保留当前记录，可按 B 重试或双击丢弃。终端 `s` / `d` 保留为停止后的备用保存／丢弃入口。
输入 `q` 回车或按 `Ctrl+C` 退出；未确认保存（含退出时仍处于 350 ms 窗口）的数据保留为 `.partial`，不会自动变成完成的 episode。
录制与电机控制独立：B、保存或退出采集器都不会关闭示教或电机；右 A 和实体按钮含义不变。示教仍每次默认 60 秒，到时采集暂停，不自动重新开启示教。

仅检查准备状态、不写数据，可加 `--check`；检查时两臂须已进入示教，三相机、头部和腰部均有有效反馈：

```bash
python3 scripts/recording/record_xcore_teach.py --fps 30 --check
```

若提示 `head_pha` 无反馈，先按原摇操流程让头部 CSP 就绪并产生位置反馈；采集器不代替头腰使能。

### 数据内容与同步

| 数据 | 来源／顺序 |
| --- | --- |
| `state` / `action` | 左臂 J1～J7、右臂 J1～J7、头部 PHA、腰部俯仰，共 16 维，全部为弧度 |
| 左右臂实测值 | `/robot_02/upperlimb/cartesian_drag_safe/joint_states/{left_arm,right_arm}`，目标 100 Hz，同一 SDK 连接在示教期间读取 `jointPos`；原 50 ms 安全检查周期与阈值保留 |
| 头部 PHA | `/headmotor/csp/joint_states` 中的 `head_pha_joint` |
| 腰部俯仰 | `/robot/waist_pitch/csp/position_deg`，角度转弧度，前倾为正 |
| 三路 RGB | `/robot/sensors/cameras/{head,left_wrist,right_wrist}/color/image_raw` |
| 右 B | `/robot_02/upperlimb/pico/buttons/right_arm`，原按键字节 bit 3；A 为 bit 2 |

动作语义沿用参考录制器：同一采样时刻的实测位置同时写入 state 与 action，不伪造示教期间不存在的 ServoL 命令，也不增加外部 IK 或未来帧偏移。
当前默认 **30 Hz、640×480**，CLI 支持 `1..30 Hz`。相机使用独立的 `cameras_30hz.yaml`：三路彩色与深度均设 30 Hz，保留原深度、对齐与流同步开关；原 `cameras.yaml` 的 6/5/5 Hz 配置不变。三台设备各自列出了 VGA 30 Hz 模式，但当前共用 USB 2.0 集线器，需要改接 USB 3.x 后验证三路同时输出。采集器不设置相机参数，也不调整电机速度、模式或运动目标。
变更相机分辨率后，采集端应显式传相同的 `--width` / `--height`。更新后必须重启机械臂驱动，才能使用新的高频示教反馈。
手臂和头部按消息时间戳与图像对齐；腰部 Float64 无源时间戳，使用 ROS 接收时间并在元数据中注明。
默认跨输入时间差不超过 `min(150 ms, 1/fps)`，30 Hz 时为 **33.3 ms**；这是软件时间戳对齐，不保证相机曝光硬同步。状态接收超时 250 ms、图像源／接收超时 500 ms。每路源时间戳和接收时间都需至少 1 秒的测频历史，频率须达到请求值的 95%；`--check` 显示各路实测 Hz。采集中各路位置与图像源时间戳都必须持续前进，单帧跳过会暂停，不以重复值伪造 30 Hz。计时采用固定周期截止时间，避免轮询延迟累计降低采集频率。
高于 20 Hz 时使用 3 个图像写入线程、无损 PNG 压缩级别 0；三路 VGA 约 **83 MB/s、5 GB/分钟**。写盘队列仍有界，超载会暂停；需使用能持续写入的本地磁盘。较低频率保留原压缩级别 1。
使用新的 `act_teach_30hz` 数据根目录；已有 5 Hz schema 不允许直接改为 30 Hz 混写。恢复原低频流程可用原相机入口并显式传 `--fps 5 --root /home/niic/aide/assets/dataset/act_teach`。

磁盘结构如下，PNG 为 RGB 对应的无损图像；临时记录与完成记录分开，已有数据不覆盖，不兼容的 schema 拒绝混写：

```text
act_teach_30hz/
  schema.json
  episode_000000/
    episode.json          # 任务、16维顺序/单位、相机、连续片段边界
    trajectory.npz        # state/action [N,16]，帧及各输入时间戳
    images/{head,left_wrist,right_wrist}/000000.png
  .episode_<uuid>.partial/ # 尚未确认保存或异常退出的记录，不参与导出
```

### 校验与导出到现有 ACT 训练格式

当前机器未安装旧 LeRobot 环境。原始采集与校验仅需要现有 ROS、NumPy 和 OpenCV，训练机再使用参考工程的 **LeRobot 0.4.3** 环境导出。
下列校验无需 LeRobot：

```bash
python3 scripts/dataset_tools/export_teach_lerobot.py \
  --root /home/niic/aide/assets/dataset/act_teach_30hz --validate-only
```

在已安装 LeRobot 0.4.3 的环境中执行导出，目标目录必须不存在：

```bash
python scripts/dataset_tools/export_teach_lerobot.py \
  --root /home/niic/aide/assets/dataset/act_teach_30hz \
  --destination /home/niic/aide/assets/dataset/act_teach_30hz_lerobot \
  --repo-id local/xcore_physical_teach
```

导出字段为 `observation.state`、`action`（各 16 维）和 `observation.images.{head,left_wrist,right_wrist}`。
每个连续录制片段导出为一个独立 LeRobot episode，不让动作块跨过暂停期间未记录的运动；原始 episode 与片段对应关系写入 `meta/physical_teach_source.json`。
生成的数据可供本目录 `scripts/training/train_act.py` 从新 16 维特征训练；原 8 维右臂模型与部署器不直接兼容，本次不改部署或训练参数。

---

项目根目录：

```bash
export XCORE_ACT_ROOT=/home/niic/aide/software/vla/act/xcore_act
export AIDE_ROOT=/home/niic/aide
export XCORE_ACT_DATASETS_ROOT=/home/niic/aide/assets/dataset/xcroe_ACT/datasets
export XCORE_ACT_MODEL_ROOT=/home/niic/aide/assets/model/xcroe_ACT/model_outputs
```

每个新终端都应先执行下面的环境命令。不能在尚未定义
`XCORE_ACT_ROOT` 时直接运行 `source "$XCORE_ACT_ROOT/..."`；否则变量会展开为空，
路径将错误地变成 `/environment/activate.sh`。环境脚本会自动导出
`XCORE_ACT_ROOT`、`AIDE_ROOT`、`PYTHONPATH`，并加载与开机服务一致的本机
ROS 2 通信配置（Cyclone DDS、domain 89）。

## 目录

```text
xcore_act/                         # 源码，不存放运行资产
├── src/                 # xCore 机器人适配器
├── scripts/             # 录制、训练、评估和数据处理脚本
├── environment/         # 环境脚本和依赖
├── logs/                # 日志
├── assets/              # 音频等资源
└── tests/               # 测试

/home/niic/aide/assets/
├── dataset/xcroe_ACT/datasets/    # 右臂 ACT 数据集
├── dataset/xcroe_ACT/datasets_left/ # 左臂 ACT 数据集
├── model/xcroe_ACT/model_outputs/ # 右臂 ACT 模型
└── model/xcroe_ACT/model_outputs_left/ # 左臂 ACT 模型
```

## 1. 进入环境

```bash
source /home/niic/aide/software/vla/act/xcore_act/environment/activate.sh
```

只读环境检查：

```bash
python "$XCORE_ACT_ROOT/scripts/diagnostics/preflight.py" --no-network
```

检查统一三相机 ROS 话题（不会直接打开相机设备）：

```bash
python "$XCORE_ACT_ROOT/scripts/diagnostics/preflight.py" --check-cameras
```

相机由 `hardware/robot_vision_ws` 中的官方 RealSense 驱动唯一占用。需要手动启动三相机时：

```bash
bash /home/jetson/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_cameras.sh
```

## 2. 启动遥操

编译并自检 `pico_servol`：

```bash
source /home/niic/aide/startup/ros_environment_debug.sh
source /opt/ros/humble/setup.bash
cmake --build "$AIDE_ROOT/hardware/arm/xCoreSDK-v0.7.1.ar_6/build" \
  --target pico_servol -j2
"$AIDE_ROOT/hardware/arm/xCoreSDK-v0.7.1.ar_6/build/bin/pico_servol" --self-test
```

终端 1，启动夹爪：

```bash
cd "$AIDE_ROOT/hardware/end_effector/sts3215_ws"
source /home/niic/aide/startup/ros_environment_debug.sh
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch sts3215_driver sts3215.launch.py
```

终端 2，启动 xCore/PICO 遥操：

```bash
source /home/niic/aide/startup/ros_environment_debug.sh
source /opt/ros/humble/setup.bash
"$AIDE_ROOT/hardware/arm/xCoreSDK-v0.7.1.ar_6/build/bin/pico_servol"
```

## 3. 录制数据

```bash
python "$XCORE_ACT_ROOT/scripts/recording/record_xcore_pico.py" \
  --repo-id local/xcore_right_arm_servoj_qm_v1 \
  --root "$XCORE_ACT_DATASETS_ROOT/xcore_right_arm_servoj_qm_v1" \
  --task "approach → grasp → place" \
  --segments-per-episode 3 \
  --segment-title "approach the green cube" \
  --segment-title "grasp the green cube" \
  --segment-title "place it into the blue box" \
  --num-episodes 40 \
  --episode-len 4500 \
  --min-episode-frames 90 \
  --udp-status-port 6005
```

默认只订阅右臂 ACT 所需的两路 `sensor_msgs/Image`（sensor-data QoS）：

- `head`：`/head/color/image_raw`，960x540@30
- `right_wrist`：`/right_wrist/color/image_raw`，640x480@30

新数据集字段固定为 `observation.images.head` 和
`observation.images.right_wrist`。如果 `--root` 指向包含 `left_wrist` 或其他不兼容
相机 schema 的数据集，录制器会保留原数据，并自动使用带时间戳的新目录，避免混写。

VR 相机视频不再由录制程序附带发送。请在网页总览点击“发送 VR 视频”，由
`hardware/robot_vision_ws` 中的 `robot_vision_tools/udp_video_sender` 统一将 head、left_wrist、right_wrist 发送到
UDP 6000、6001、6002；ACT 仍只把 head 与 right_wrist 写入数据集。

每个 Episode 可配置 1–8 个子任务。每个子任务按一次 Up/B 开始、再按一次 Up/B 完成；
全部子任务完成后才保存 Episode，每帧使用当前子任务描述作为独立 `task` 标签。完成子任务后
在确认窗口内再按 B 会放弃整个当前 Episode。Grip 切换遥操，A/down 停止遥操并回到网页
“遥操设置”保存的右臂初始点。

数据动作语义为：同一时刻的实测关节 `q_m` 同时写入
`observation.state` 和 `action`，不使用 `state[t+3]`，也不做外部 IK。

录制后验证：

```bash
python "$XCORE_ACT_ROOT/scripts/dataset_tools/prepare_xcore_act_dataset.py" \
  --root "$XCORE_ACT_DATASETS_ROOT/xcore_right_arm_servoj_qm_v1"
```

## 4. 数据集工具

合并数据集：

```bash
python "$XCORE_ACT_ROOT/scripts/dataset_tools/merge_lerobot_datasets.py" \
  --src-root "$XCORE_ACT_DATASETS_ROOT/DATASET_A" \
  --src-root "$XCORE_ACT_DATASETS_ROOT/DATASET_B" \
  --dst-root "$XCORE_ACT_DATASETS_ROOT/DATASET_MERGED" \
  --repo-id local/DATASET_MERGED \
  --video-backend pyav
```

删除指定 episode，源数据不会被修改：

```bash
python "$XCORE_ACT_ROOT/scripts/dataset_tools/fast_prune_dataset.py" \
  --src-root "$XCORE_ACT_DATASETS_ROOT/xcore_right_arm_vr" \
  --dst-root "$XCORE_ACT_DATASETS_ROOT/xcore_right_arm_vr_clean" \
  --drop 2,38 \
  --overwrite
```

导出预览：

```bash
python "$XCORE_ACT_ROOT/scripts/dataset_tools/export_episode_previews.py" \
  --root "$XCORE_ACT_DATASETS_ROOT/xcore_right_arm_vr" \
  --repo-id local/right_arm_vr \
  --out-dir "$XCORE_ACT_ROOT/artifacts/previews/right_arm_vr"
```

## 5. 训练 ACT

建议在 tmux 中训练：

```bash
tmux new -s act_pickup
source /home/niic/aide/software/vla/act/xcore_act/environment/activate.sh

python "$XCORE_ACT_ROOT/scripts/training/train_act.py" \
  --dataset-root="$XCORE_ACT_DATASETS_ROOT/green-box-819_clean_v1" \
  --dataset-repo-id=local/pickup \
  --video-backend=pyav \
  --output-dir="$XCORE_ACT_MODEL_ROOT/act_custom" \
  --job-name=pickup_act_custom \
  --chunk-size=50 \
  --n-action-steps=10 \
  --batch-size=8 \
  --num-workers=1 \
  --steps=60000 \
  --save-freq=10000 \
  --log-freq=100 \
  --mixed-precision=fp16
```

重新进入训练会话：

```bash
tmux ls
tmux attach -t act_pickup
```

断点续训：

```bash
python "$XCORE_ACT_ROOT/scripts/training/train_act.py" \
  --resume-from="$XCORE_ACT_MODEL_ROOT/act_custom/checkpoints/040000" \
  --steps=60000
```

## 6. 模型评估

当前已有模型：

```bash
export XCORE_POLICY_PATH="$XCORE_ACT_MODEL_ROOT/pickup/checkpoints/060000/pretrained_model"
```

### 6.1 只读推理

运行模型并保存评估数据，但不驱动机械臂和夹爪：

```bash
source /home/niic/aide/software/vla/act/xcore_act/environment/activate.sh

export XCORE_POLICY_PATH="$XCORE_ACT_MODEL_ROOT/pickup/checkpoints/060000/pretrained_model"
export XCORE_EVAL_NAME="eval_xcore_act_read_only_060000_$(date +%Y%m%d_%H%M%S)"

python "$XCORE_ACT_ROOT/scripts/recording/record_with_torque_retry.py" \
  --chunk-fusion \
  --chunk-fusion-replan-interval=2 \
  --chunk-fusion-old-weight=0.3 \
  --chunk-fusion-action-stride=3 \
  --robot.type=xcore_right_arm \
  --robot.id=xcore_right \
  --robot.enable_motion=false \
  --robot.cameras='{head: {type: ros_image, topic: "/head/color/image_raw", width: 960, height: 540, fps: 30}, right_wrist: {type: ros_image, topic: "/right_wrist/color/image_raw", width: 640, height: 480, fps: 30}}' \
  --policy.path="$XCORE_POLICY_PATH" \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=null \
  --policy.n_action_steps=10 \
  --policy.temporal_ensemble_coeff=null \
  --policy.use_amp=false \
  --dataset.repo_id="local/$XCORE_EVAL_NAME" \
  --dataset.root="$XCORE_ACT_DATASETS_ROOT/$XCORE_EVAL_NAME" \
  --dataset.overwrite=true \
  --dataset.single_task='pick up the green cube and put it into the blue box' \
  --dataset.fps=10 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=false
```

### 6.2 实机推理

确认只读推理正常后再运行。执行前必须确认急停可用，工作空间内没有人员或障碍物。

```bash
source /home/niic/aide/software/vla/act/xcore_act/environment/activate.sh

export XCORE_POLICY_PATH="$XCORE_ACT_MODEL_ROOT/pickup/checkpoints/060000/pretrained_model"
export XCORE_EVAL_NAME="eval_xcore_act_real_060000_$(date +%Y%m%d_%H%M%S)"

python "$XCORE_ACT_ROOT/scripts/recording/record_with_torque_retry.py" \
  --chunk-fusion \
  --chunk-fusion-replan-interval=2 \
  --chunk-fusion-old-weight=0.3 \
  --chunk-fusion-action-stride=3 \
  --robot.type=xcore_right_arm \
  --robot.id=xcore_right \
  --robot.enable_motion=true \
  --robot.move_to_initial_pose=true \
  --robot.require_gripper_state=true \
  --robot.gripper_motion_hold_s=0.0 \
  --robot.cameras='{head: {type: ros_image, topic: "/head/color/image_raw", width: 960, height: 540, fps: 30}, right_wrist: {type: ros_image, topic: "/right_wrist/color/image_raw", width: 640, height: 480, fps: 30}}' \
  --policy.path="$XCORE_POLICY_PATH" \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=null \
  --policy.n_action_steps=10 \
  --policy.temporal_ensemble_coeff=null \
  --policy.use_amp=false \
  --dataset.repo_id="local/$XCORE_EVAL_NAME" \
  --dataset.root="$XCORE_ACT_DATASETS_ROOT/$XCORE_EVAL_NAME" \
  --dataset.overwrite=true \
  --dataset.single_task='pick up the green cube and put it into the blue box' \
  --dataset.fps=10 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=false
```

### 6.3 CUDA 离线重放

对已保存的数据进行 CUDA 离线重放，不会连接机械臂：

```bash
python "$XCORE_ACT_ROOT/scripts/evaluation/replay_eval_cuda.py" \
  --dataset-root "$XCORE_ACT_DATASETS_ROOT/EVAL_DATASET_NAME" \
  --policy-path "$XCORE_POLICY_PATH" \
  --output-prefix "$XCORE_ACT_DATASETS_ROOT/EVAL_DATASET_NAME/cuda_replay"
```

### 6.4 ROI-ACT `green_box_v1`

`/home/niic/aide/models/green_box_v1` 是独立的 ROI-ACT checkpoint 格式，不使用
LeRobot 的 `config.json + model.safetensors` loader。已配置专用运行器，并接入移动端
现有的“只读推理 → 实机推理”安全门禁。模型目录会以 `roi_act/green_box_v1` 出现在
ACT 模型列表中。

不连接机器人进行 CUDA smoke test：

```bash
python "$XCORE_ACT_ROOT/scripts/deployment/run_roi_act.py" \
  --model-dir /home/niic/aide/models/green_box_v1 \
  --device cuda \
  --smoke-test
```

完整的只读和实机命令见
`/home/niic/aide/models/green_box_v1/README_INFERENCE.md`。

### 6.5 全帧 ACT `green_box_full_frame_act`

`/home/niic/aide/models/green_box_full_frame_act` 使用 `vision_mode=full_frame`。
推理只需要 `head`、`right_wrist` 完整图像和右臂状态，不运行 HSV/目标检测，也不接受
外部 ROI。网络预测的夹爪框和目标框只是训练辅助输出，不是动作推理输入。

CUDA smoke test：

```bash
python "$XCORE_ACT_ROOT/scripts/deployment/run_roi_act.py" \
  --model-dir /home/niic/aide/models/green_box_full_frame_act \
  --device cuda \
  --smoke-test
```

模型会以 `roi_act/green_box_full_frame_act` 出现在网页 ACT 模型列表中，并沿用同模型
只读推理成功后才能实机推理的安全门禁。完整命令见
`/home/niic/aide/models/green_box_full_frame_act/README_INFERENCE.md`。

## 7. Web 训练与推理

Edge API 会直接调用本 README 中的训练和推理脚本。启动并查看服务：

```bash
systemctl --user restart xcore-edge-api.service
systemctl --user status xcore-edge-api.service
```

机器人电脑打开：

```text
http://127.0.0.1:8000/
```

同一 Wi-Fi 下的手机或电脑打开：

```text
http://172.16.13.92:8000/
```

训练页面可配置数据集、输出目录、steps、batch size、chunk size、
action steps、保存/日志间隔、workers 和混合精度。推理页面可配置
checkpoint、任务、Episode 数、控制 FPS 以及每段时长。
所选模型的 `chunk_size` 和 `n_action_steps` 由 checkpoint 训练配置只读展示；
推理高级设置可调整动作采样间隔、固定重新推理间隔和新预测融合权重。后端会
校验采样后的原始动作索引没有超过模型动作块。

只读推理由后端固定为 `robot.enable_motion=false`。实机推理必须先用同一
checkpoint 完成一次只读推理，并确认急停可用、工作区无人；训练、推理、
采集、遥操和相机预览不能争用同一机器人资源。

## 注意

- 不要在 ACT/Web 业务程序中直接打开 `/dev/video*` 或 RealSense；ACT 只订阅 `all_camera` 发布的 `head` 与 `right_wrist`，interACT 才使用三路。
- 含 `left_wrist` 的旧三相机 ACT 数据集/checkpoint 仍可离线回放和审计，但网页 ACT 推理会明确拒绝该 checkpoint；请用两相机数据重新训练。
- 两相机 checkpoint 的视觉特征必须是 `head` 与 `right_wrist`，不会静默忽略或补造输入。
- `xcore_right_arm_vr_act_v2` 等旧数据使用不同动作语义，不要直接混入同帧 `q_m` 数据。
- 当前录制和推理常用数据帧率为 10 FPS，相机采集帧率为 30 FPS。
- 实机动作异常时立即按 `Ctrl-C` 或急停。
