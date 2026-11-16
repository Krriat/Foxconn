# 本地 SDK 调查依据

调查范围为同目录 SDK 源码、公开头文件、`1.pdf` 与现有 cxl 控制程序；未连接控制器。

| 项目 | 本地证据 | 结论与实现 |
|---|---|---|
| AR 7轴 | `include/rokae/robot.h: ArRobot : Cobot<7>` | 采集器使用真实 `ArRobot`，连接后检查7轴；型号用 `robotInfo().type` 读回 |
| 版本 | `CMakeLists.txt: project(xCoreSDK VERSION 0.7.1)`；`BaseRobot::sdkVersion()`；`Info.version` | 文件夹为0.7.1.ar_6；实际SDK/控制器版本由运行日志记录；无独立固件版本字段 |
| 地址 | `cxl/pos.cpp: ArmConfig arms` | 左4网段、右2网段；作为来源明确的待核验配置，不猜端口 |
| 状态读取 | `robot.h: operationState / operateMode / powerState` | 无写调用；非idle退出；并未找到运动控制模式或控制权token的完整读取接口 |
| 新数据 | `robot.h: updateRobotState(timeout)` | 返回接收长度，0为未收到；只有大于0才调用 `getStateData` 并记样本 |
| 采样身份 | `data_types.h: RtSupportedFields` | 没有设备采样序号/时间戳；主机收包计数不能证明设备样本唯一；相同位置也不能证明重复 |
| 反馈周期 | `robot.h: startReceiveRobotState(interval, fields)` | 允许1/2/4/8ms/1s，首帧等待3秒；不是原生传感器采样率证明 |
| 位置/速度 | `RtSupportedFields::jointPos_m=q_m / jointVel_m=dq_m` | 七轴数组，rad、rad/s；采集器必选 |
| 力矩 | `RtSupportedFields::tau_m`；`Robot_T::jointTorque` | 流式tau_m注释只写关节力矩Nm；另一个jointTorque接口明确写关节力传感器。没有证据证明两者同一信号，因此tau_m保留来源未核实标签 |
| 其他力矩 | `tau_c / motor_tau / tau_ext_base / tau_ext_stiff` | 命令、未注明单位电机力矩、外部残差不能互换；不采集/换算这些字段 |
| 电流/温度/电压 | `RtSupportedFields` | 未找到可用公开字段；留空，不以零或关节峰值代替 |
| 软限位 | `Robot_T::getSoftLimit(limits, ec)` | 读取启用标记和七轴上下界rad；不设置软限位；无J6/J7耦合空间保证 |
| 载荷 | `BaseRobot::toolset()`，`data_types.h: Load` | 读取mass kg、cog m、三项inertia kg·m²和工具/参考系；不是安装实测质量/惯量 |
| 位置流式候选 | `example/rt/servoj_demo.cpp`，`motion_control_rt.h: setServoJoint / startMove / sendCommand` | 示例20ms ServoJ位置命令；没有已验证的位置+速度前馈+独立重力effort联合接口。不能直接复制仿真500/50为SDK增益 |
| 周期差异 | `sendCommand`注释、`1.pdf §3.5`要求1ms；ServoJ示例20ms | 候选模式差异必须在现场固件核实；不能用普通RT模式直接发送50Hz命令 |
| 命令语义 | `sendCommand`返回void | 不代表执行完成；无实际应用时间戳。非实时MoveAbsJ内部排队规划不等同最终策略接口 |
| 受控停止 | `motion_control_rt.h: stopMove / Finishable::setFinished` | SDK声明可结束实时运动，失败抛异常；没有本次已验证的停止轨迹/耗时/独立阻塞保护。`stopLoop`只是调度停止，不能当物理停止确认 |
| 超时 | `robot.h`接收超时；`1.pdf §3.3.2`本地rt._timeout_为1～20ms | 是主机接收等待，不是已证明的控制器watchdog；未找到send/stop/connect全链路有界阻塞保证 |
| 丢包 | `BaseCobot::setRtNetworkTolerance` | 可设置丢包容忍比例，但没有已核实的停止时限及轨迹；本工具不修改 |
| 连接释放 | `robot.h: disconnectFromRobot` | 明确写断开前停止运动；无可证明与现有客户端并行的只读会话语义，只能安排在其他程序退出后静止采集 |

## 已交付与未实现的对应关系

| test.md 需求 | 本次状态 |
|---|---|
| 能力调查与短时反馈 | 离线报告和真实SDK只读采集器已实现、已编译；实机采集由操作者运行 |
| 平滑单轴轨迹、重复、留出、高频减幅 | 已实现解析C3包络和连续时间导数保守界；关节限位有输入时检查全七轴 |
| TCP/碰撞/停止轨迹预览 | 尚无现场验证模型，明确不通过执行门槛；CSV/图不声称完成这项检查 |
| 真实 RobotAdapter.read_state | 使用 startReceiveRobotState → updateRobotState → getStateData |
| 真实 send_setpoint / request_controlled_stop | 仅完成真实SDK接口映射调查；运动实现和独立保护链路未完成，SDK执行入口拒绝连接 |
| 写盘不阻塞采集 | C++使用4096帧有界SPSC队列，溢出/写盘异常中止采集；不涉及运动控制 |
| 故障测试 | mock/replay覆盖重复、断流、回退、异常值、模式/故障、超时/溢出、操作者停止、停止失败；不等同硬件watchdog或制动验证 |
| 日志和完全离线分析 | 已实现独立命令/反馈CSV、元数据/哈希、回放、相位/误差/耦合、重复散布、多频一阶延迟候选及整段留出误差、打包 |
| 力矩—转速工作区数据 | 只读/合成日志均保留来源；未核实tau_m只交付原始反馈，正式运动中的需求测量尚未完成 |
| 真正能力包络 | 未测得；尚无逐轴连续/峰值厂家曲线或台架；unavailable |

没有关闭限位、碰撞或驱动保护，没有调用原示例的MoveJ到示例姿态。也没有用进程退出替代物理停止确认。
