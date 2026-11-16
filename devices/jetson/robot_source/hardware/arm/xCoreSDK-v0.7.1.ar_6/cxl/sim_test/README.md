# AR5 测量准备与分步运行

本目录根据 [test.md](test.md) 实现准备工具。原有 `cxl` 运动程序、参数和 SDK 均未修改。

本次已在当前机器编译好采集器，并生成 `plans/left_J1`～`left_J7`、`plans/right_J1`～`right_J7` 共14份预览，以及本目录的 [capabilities.json](capabilities.json)。可以直接从第3节开始分别采集；第4节是重新生成计划的方法，已有目录需换新名称。

**当前可运行：本地能力调查、静止机器人只读采集、逐轴轨迹预览、mock 故障演练、离线分析及打包。真机 ServoJ 运动后端尚未实现，`--backend sdk --execute` 会在连接前拒绝执行。** 不能把 mock 结果用于机械臂辨识，也不能靠将配置中的 `false` 改成 `true` 开放运动。

当前缺的是经现场验证的控制权/释放、独立保护及 SDK 阻塞处理、控制器 watchdog 和停止轨迹、保护遥测、TCP/碰撞空间核验，以及最终部署控制接口的确定。公开 SDK 未提供完整验证所需的读取接口。需要这些实际证据后再实现、验证实机运动适配器。详细缺口在 [SDK_FINDINGS.md](SDK_FINDINGS.md) 和各计划 `README_plan.md` 中。

## 1. 环境与编译（离线，不连接机器人）

```bash
cd /home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6/cxl/sim_test
export MPLCONFIGDIR="$PWD/.mplconfig"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python3 -m unittest -v test_measurement
```

当前环境的 Python 3、NumPy、PyYAML、Matplotlib 已可用，测试和分析不依赖 SciPy。移到其他机器缺依赖时，在自己的 Python 环境中安装 `requirements.txt`。该独立 CMake 工程直接链接同一 SDK 的静态库，不要求启动 ROS，也不会编译/运行现有运动示例。

## 2. 先生成本地能力调查

```bash
python3 inspect_robot_capabilities.py --out runs/local_survey
```

默认不连接任何地址。命令输出新建的 `run_日期_时间_随机标识` 目录，内含 `capabilities.json`、元数据和力矩能力未知说明。

## 3. 左右臂分别做静止只读采集

先查看 [measurement_config.yaml](measurement_config.yaml)。地址取自现有 `../pos.cpp`：左臂 `192.168.1.160 / 192.168.1.10`，右臂 `192.168.2.160 / 192.168.2.10`，尚未经本次连接验证，现场不同就修改。无需填猜测端口。

**运行下面命令前，机械臂应已静止，其他遥操作/运动/状态采集客户端应已退出。** 采集器读取 `operationState`，非 idle 会退出；不会抢占、停止别的程序。SDK 文档明确说 `disconnectFromRobot` 会停止机器人运动，析构的全部副作用未公开，因此“只读”表示没有运动/配置调用，并不承诺与其他控制程序并行连接毫无影响。不要自动调用旧的 `op`、`stop`、`servoj_demo` 或启动整个机器人栈。

左臂：

```bash
python3 inspect_robot_capabilities.py --connect --arm left --seconds 10 --out runs/left_stationary
```

右臂（左臂采集结束后）：

```bash
python3 inspect_robot_capabilities.py --connect --arm right --seconds 10 --out runs/right_stationary
```

默认请求 4 ms 反馈包，即 250 Hz **请求收包率**，不是已确认的原生传感器采样率。可配置 1/2/4/8/1000 ms。运行后自动离线统计静止噪声和收包间隔；实际固件若不支持 `tau_m`，该次失败记录会保留，不自动重试。随后可手动运行位置/速度采集：

```bash
python3 inspect_robot_capabilities.py --connect --arm left --seconds 10 --without-torque --out runs/left_stationary
```

每次使用新目录，退出码非零表示失败。查看 `raw_vendor/bridge_stderr.log`、`native_events.jsonl` 和 `metadata.json`。父进程对采集子进程设置总时限；结束子进程不等于确认机器人停止。采集器没有运动，所以不对不属于本工具的运动发送停止指令。

## 4. 左右臂逐轴生成轨迹计划

```bash
python3 plan_servo_identification.py --arm left --joint 1 --out plans/left_J1_reviewed
python3 plan_servo_identification.py --arm right --joint 1 --out plans/right_J1_reviewed
```

`--joint` 可分别选择 1～7；每个计划仅激励一轴，输出全臂保持目标。已存在的输出目录不会被覆盖，重新生成时换一个目录名。

默认每个计划有 3 个 `0.1 Hz / 单边0.5°` 重复段和 1 个 `0.125 Hz` 留出段；每段开始静止 10 秒、进入 10 秒、至少 3 个稳定周期、退出 10 秒、结束静止 5 秒。高频幅值依据**包含包络导数**的速度/加速度/jerk 连续时间保守界自动减小，不会提高限额。每周期至少 10 个命令点。

每个计划提供：

- `plan.json`：配置快照、哈希、轨迹配方、所有未完成核验项。
- `trajectory_preview.csv`：全臂位置以及激励轴的解析速度/加速度/jerk。
- `trajectory_preview.png`：可独立查看的四条曲线。
- `README_plan.md`：逐段幅值和界限表。

未填 `center_q_rad` 时，CSV 的零点只是相对预览原点，**不是发给机械臂的姿态**。现场中心必须用原来已验证的方式到达并读回。配置/计划被更改后必须重新生成；运行入口会重建并验证整个计划，不能仅手改哈希。

需要预览候选频率时，可另外生成计划：

```bash
python3 plan_servo_identification.py --arm left --joint 1 --frequencies 0.1 0.25 0.5 1 2 4 --out plans/left_J1_multifrequency
```

这仍是关节空间预览。TCP 速度、双臂/漏斗碰撞空间、J6/J7 耦合、停止余量尚无验证模型，输出不会声称检查通过。

## 5. 无硬件 mock 演练与回放

仅查看计划（默认行为，无运动）：

```bash
python3 run_servo_identification.py --plan plans/left_J1/plan.json
```

显式运行 mock（虚拟时间，不连接机器人）：

```bash
python3 run_servo_identification.py --plan plans/left_J1/plan.json --backend mock --execute --out runs/mock
```

日志模拟独立 250 Hz 设备反馈、50 Hz 位置命令、已知一阶响应及延迟、少量噪声和轴间耦合。`qd_reference_rad_s` 是参考轨迹速度，实际未发送速度前馈，所以 `qd_cmd_rad_s` 和 `tau_ff_cmd_nm` 留空。合成力矩标记 `synthetic`。

故障注入示例（预期退出码 1，并留下失败记录）：

```bash
python3 run_servo_identification.py --plan plans/left_J1/plan.json --backend mock --execute --fault dropout --out runs/faults
python3 run_servo_identification.py --plan plans/left_J1/plan.json --backend mock --execute --fault stop_failure --out runs/faults
```

可选故障：`duplicate`、`dropout`、`nan`、`sequence_rollback`、`clock_rollback`、`mode_change`、`drive_fault`、`command_timeout`、`log_overflow`、`operator_stop`、`stop_failure`。停止请求会永久撤销本次运行的发送许可；停止请求和停止读回确认分别记录。停止失败保留 `stop_unconfirmed`，不会重连或继续后续段。

原始日志可完全离线回放，把下面路径替换为实际运行输出路径：

```bash
python3 replay_actuator_logs.py runs/mock/run_实际目录
```

该回放检查帧完整性、时序、异常值及停止锁存；阈值是 mock 测试值，不是现场保护限额，不能证明实机停止。没有设备序号时，不把主机计数当作设备序号检测去重。

## 6. 离线分析、绘图、打包

```bash
python3 analyze_actuator_logs.py runs/mock/run_实际目录 --archive
```

分析器没有 SDK 导入、连接或控制入口。原始数据不重采样、不覆盖；只更新衍生结果。压缩包在运行目录旁边，已有包不覆盖。

交付文件包括 `README_results.md`、`summary.json`、`servo_estimates.csv`、`torque_vs_speed.csv`、`torque_statistics.json`、`available_torque_envelope.json`、`plots/`，以及全部原始日志、计划和哈希。`--no-plots` 可跳过画图。

解释边界：

- 主机发送和主机收包用同一时钟分析；不同设备时钟没有同步时不直接相减。单频相位换算只给有效总滞后，存在整周期歧义，不能分离通信延迟与机械响应。
- 至少三个有效辨识频率才尝试 `K/(1+sT) × exp(-s·delay)` 模型；搜索范围、边界解、近最优范围和适用工况均记录。该范围不是置信区间。留出整段只用于验证，不参与拟合；默认首轮单频不硬拟合模型。
- 原始哈希不符或采集失败时不生成有效模型。重复读包不伪装成新增设备采样；SDK无设备序号时原生采样率/缺帧/重复保持未知。
- 速度 RMSE 使用原始反馈速度与解析参考速度对比；没有对位置二阶差分推断力矩。其他轴的谐波幅值用于查看耦合。
- `tau_m` 的流式力矩来源尚未实机核实：原值保存在 `raw_sdk_tau_nm`，来源未证实时 `torque_vs_speed.csv` 的 `torque_nm` 留空。已确认输出端传感器字段才可标为 `joint_output_sensor`；禁止仅改标签冒充标定。
- CSV 按轴、方向、姿态、负载、温度及进入/稳态/退出段保存；汇总温度分箱为 5°C，缺温度单列未知。正弦稳态不是匀速工况；本工具没有匀速/台架极限搜索轨迹。
- `available_torque_envelope.status` 始终为 `unavailable`。182/134/35 N·m 限额不转换成连续转速曲线。

## 7. 真机运动前还需要的现场资料

见配置中的 `null` 与各计划的 blockers，至少包括：逐轴索引/符号/零点（尤其 J4）、J6/J7 耦合范围、控制器实测中心姿态、限位/动态限制来源、工具安装总质量（漏斗252g不包含未知连接件）、工具质心/惯量、最终部署接口与 ServoJ 周期/增益/前瞻设置、TCP/碰撞/停止轨迹验证、控制权和保护遥测、已验证 watchdog 和停止失败流程。

先返回左右臂静止采集日志即可补齐型号、控制器/SDK版本、当前软限位和工具配置等只读信息。缺失力矩不会阻塞位置/速度采集。运动适配器仍需要继续开发和现场验证；本次不会将未测条件填成已通过。
