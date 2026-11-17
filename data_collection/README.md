# 示教数采代码归档

2026-11-17 从 Jetson 和 umie 只读复制，接在 `jetson-act-qwen-20261117` 分支的成功推理快照之后。共 589 个设备源文件、约 4.5 MB，每个文件均与设备上的 SHA-256 对照一致。详见 [来源与校验记录](manifests/capture.json) 和 [逐文件索引](manifests/source-files.json)。

这里是数采程序及其依赖源码。此前成功的 ACT/Qwen 推理配置继续保存在仓库根目录的 `devices/` 和 `training_tools/`，本次没有覆盖它们，也没有替换权重或改变机器人动作参数。复制期间未启动采集、重启服务或发送机械臂/吸盘命令。

## 文件和职责

| 目录 | 内容 |
|---|---|
| `jetson/AIDE_robot_main/software/vla/act/xcore_act/` | 键盘示教采集器、统一启动器、源时间匹配、关节本地反馈、PNG/轨迹写入、质量校验、LeRobot 导出及对应测试 |
| `jetson/AIDE_robot_main/hardware/arm/xCoreROS/` | 数采所用双臂 ROS 驱动、接口、启动器和示教反馈源码 |
| `jetson/AIDE_robot_main/hardware/robot_vision_ws/` | 头相机/三相机的 ROS 工作区源码、配置及启动脚本 |
| `jetson/AIDE_robot_main/hardware/USR-M050/` | 吸盘继电器 SDK、封装源码与构建说明 |
| `jetson/AIDE_robot_main/startup/` | 数采原有 ROS 环境配置 |
| `umie/aide_pc_wrist_cameras/` | 网络数采模式的双腕相机 TCP 服务、PNG 无损传输、跨机时钟比较与诊断工具 |
| `umie/.local/share/robot-camera-streams/` | 另一套 ROS 双腕图像发布器，包含其原始配置 |
| `umie/.local/share/robot-relay-bridge/` | 接在 umie 的有线继电器 TCP 桥接 |

`umie/.local/` 是隐藏目录，命令行用 `ls -a` 可查看。两套腕相机程序都保存了源码；网络数采启动器使用 `aide_pc_wrist_cameras` 的 TCP 服务。它们使用相同 USB 相机，运行时应选择一套，不能同时占用设备。

## 恢复位置与依赖

保留了原目录层级，因为采集器通过 `parents[6]` 定位机器人根目录。应将 `jetson/AIDE_robot_main/` 恢复到 `/home/jetson/AIDE_robot_main/`，将 `umie/` 下对应目录恢复到 `/home/umie/`。直接在当前 Git 目录执行采集器会得到不同的根目录。

本次归档不含 `build/`、`install/`、虚拟环境、原始数据、模型权重或凭据。需按各工作区原有说明构建 ROS 2 Humble、RealSense、xCoreROS 和 USR-M050；M050 库默认路径为 `hardware/USR-M050/build/libm050.so`。ROKAE SDK 头文件在仓库原有 `devices/jetson/robot_source/hardware/arm/xCoreSDK-v0.7.1.ar_6/`，二进制库在既有 Jetson runtime Release 附件；恢复时仍需保持其原相对目录。

数采入口使用 **ROS Domain 0**、`/robot_02/...` 接口；之前的在线 ACT 链路使用 **Domain 1**、`/robot/...`。本次原样保存两套配置，没有为了统一名称而改动它们。

## 原网络数采入口示例

以下是恢复依赖后供人工使用的入口，本次归档没有执行。当前代码默认保存 20 Hz、640×480、头/左腕/右腕三路 RGB，腕相机源频率为 30 Hz；可通过参数更改。状态是左右各 7 关节角和 2 路吸盘反馈，动作保存关节示教位置和键盘请求的吸盘目标。继电器开关反馈不表示真空吸附已形成。

先在 umie 启动有线继电器桥接和网络腕相机服务：

```bash
bash /home/umie/.local/share/robot-relay-bridge/start.sh
bash /home/umie/aide_pc_wrist_cameras/start_pc_cameras.sh
```

然后在 Jetson 启动数采界面。该示例使用原有有线相机链路和 umie 继电器桥接：

```bash
bash /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/start_xcore_teach_suction_network.sh \
  --fps 20 \
  --root /home/jetson/AIDE_robot_main/assets/dataset/act_teach_box_left_right_flaps_20hz \
  --task "开箱：打开左盖和右盖" \
  --m050-host 192.168.50.20 --m050-port 20108
```

Jetson 原有有线地址为 `192.168.50.10`，umie 为 `192.168.50.20`，腕相机端口为 `9800`。这些是源码中保存的现场配置，需要与实际网口和线路对应。数采入口原有 `--restart-cameras` 会切换它识别的旧相机组，不应将此命令用作在线推理服务的通用启动命令。

键盘保持原逻辑：`A` 开示教，`Z` 结束示教，`B` 双臂回到保存位置，`R` 开始录制，`S` 停止保存，`D` 丢弃未保存记录，`1/2` 左吸盘开/关，`3/4` 右吸盘开/关，`Q` 退出。入口保留原始回位点 `teach_suction_return_pose.json`。

网络传输、纯相机检查和时间对齐详见 [原网络相机说明](umie/aide_pc_wrist_cameras/README.md)。这里保留原配置和功能说明，未重新做实机数采验证。
