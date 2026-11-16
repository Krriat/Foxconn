# 本机与 Jetson 的远程修改位置

更新：2026-11-12。此清单只列本机与 Jetson。

## 本机 umie@172.16.12.45

完整源文件、配置、测试及诊断证据目录：

```text
/home/umie/xuguang1(2)/robot_review_2026-10-14/
├── camera_streaming/             相机发布源码、手腕/头部配置
├── robot_state_streaming/        双臂关节与吸盘状态发布源码
├── policy_streaming/             执行节点、协议、吸盘输出、C++ 驱动、测试及日志证据
└── communication_diagnostics/    延迟测试、时钟配置脚本和测量结果
```

进入目录时用引号：

```bash
cd '/home/umie/xuguang1(2)/robot_review_2026-10-14'
```

本机实际运行的手腕相机目录：`/home/umie/.local/share/robot-camera-streams/`。
主要修改 `camera_publisher.py`、`wrists.yaml`；对应用户服务 `robot-wrist-cameras`。

```bash
ssh -o ClearAllForwardings=yes umie@172.16.12.45
systemctl --user restart robot-wrist-cameras
```

## Jetson jetson@172.16.12.95

```bash
ssh -o ClearAllForwardings=yes jetson@172.16.12.95
cd /home/jetson/.local/share/robot-policy-streams
```

实际运行的策略与 ServoJ 代码：`/home/jetson/.local/share/robot-policy-streams/`。

| 文件 | 修改用途 |
| --- | --- |
| `config.yaml` | 模型选择、频率、归位点、限速/限加速度、运动使能、任务时长 |
| `executor_node.py` | 每轮初始化/归位、开始/停止、看门狗、吸盘调度 |
| `protocol.py` | 配置、chunk、时间戳和关节目标校验 |
| `suction_io.py` | 左右吸盘写入、跨进程锁、手动停止释放左吸盘 |
| `driver.yaml` | ServoJ SDK 周期、lookahead、gain 和底层步长限制 |
| `native/timed_policy.hpp` | 绝对时间 chunk fusion、位置/速度连续性、速度/加速度限制 |
| `native/vendor/src/ros_servoj_subscriber.cpp` | 实际 ServoJ 收发、SDK 调用、状态与时间戳记录 |
| `native/main.cpp`、`native/CMakeLists.txt` | 私有 C++ 驱动入口和编译定义 |
| `start.sh`、`run.sh` | 用户服务启动和环境/路径定义 |

其他实际运行目录：

```text
/home/jetson/.local/share/robot-state-streams/
  hardware_state.cpp        双臂关节、吸盘读回
  CMakeLists.txt
  run_state.sh / start.sh
  build/robot_hardware_state

/home/jetson/.local/share/robot-camera-streams/
  camera_publisher.py       头部相机发布
  head.yaml
  run_camera.sh / start.sh
```

日志在策略目录下 `executor_events.jsonl`、`executor.log`、`driver.log`；
状态发布日志在 `robot-state-streams/runtime.log`。
本轮中断的证据在本机 `policy_streaming/evidence/interruption_latest/`，
Jetson 原始本轮记录在 `robot-policy-streams/evidence/live_cc335440-187f-477f-bf0a-ed94087a841d/run.jsonl`。

## 修改后生效

在 IDLE 状态修改/重启。Python 或 YAML 改动后，重启对应用户服务：

```bash
systemctl --user restart robot-policy-executor    # executor / protocol / suction_io / config
systemctl --user restart robot-policy-driver      # driver.yaml
systemctl --user restart robot-hardware-state     # 已编译的硬件反馈节点
systemctl --user restart robot-head-camera        # 头部相机
```

C++ 改动需要重新编译。ServoJ 私有驱动在 Jetson 编译：

```bash
cd ~/.local/share/robot-policy-streams
systemctl --user stop robot-policy-executor robot-policy-driver
source /opt/ros/humble/setup.bash
source /home/jetson/AIDE_robot_main/hardware/arm/xCoreROS/install/setup.bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j2
./start.sh driver
./start.sh executor
```

双臂反馈节点在 Jetson 编译：

```bash
cd ~/.local/share/robot-state-streams
systemctl --user stop robot-hardware-state
source /opt/ros/humble/setup.bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./start.sh
```

这些是 systemd --user 临时服务。若停止后 unit 已被回收，使用各目录的 `start.sh` 启动。
`.local` 是隐藏目录，VS Code Remote SSH 可直接“打开文件夹”并粘贴绝对路径。
本机源文件与 Jetson 运行目录没有自动同步，直接修改运行目录后请同步保存到本机源目录。
`/home/jetson/AIDE_robot_main/` 提供原机器人工程、SDK 与数采回位点依赖；
本次定制 ServoJ 的修改入口是上述私有 `native/` 目录。

## 当前任务时长与系统校时

已按用户要求取消原 30 秒任务上限：Jetson `config.yaml` 当前为
`max_round_seconds: null`，保留用户设置的 `enable_motion: true`。
开始后由手动 `/robot/policy/stop` 结束，通信、反馈、跟踪、时钟和周期异常仍会停止。
部署后只做了只读就绪检查，未自动开始实机任务。

系统校时配置需要管理员权限：

- 本机：`/etc/systemd/timesyncd.conf.d/90-robot-lan.conf`
- Jetson：`/etc/chrony/conf.d/robot-lan.conf`、`/etc/chrony/chrony.conf`
