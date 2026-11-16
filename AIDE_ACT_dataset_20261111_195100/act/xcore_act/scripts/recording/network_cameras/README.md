# 电脑采集双腕相机，Jetson 接收并数采

有线直连地址：电脑 Ubuntu `192.168.50.20`，Jetson `192.168.50.10`。默认 TCP 端口 `9800`。
Jetson 图像路由应走 `enx00e04c2b7928`，电脑有线网口为 `enp3s0`；两端已配置同一有线子网。
原来的 `172.16.12.45` 是电脑 Wi-Fi 地址，继续使用该地址仍会走 Wi-Fi。
电脑不需要 ROS，也不连接机械臂或吸盘。电脑负责两台 D405 的 RGB 采集和压缩；
Jetson 保留头部 D435i，并接收左右腕 RGB，原键盘示教和 16 维训练数据格式保持不变。
相机源频率 30Hz，数据集保存频率仍为 20Hz。

## 1. 在电脑上安装和启动

把 `aide_pc_wrist_cameras.zip` 解压到电脑的任意目录，进入该目录。
使用 Python 3.10 或更新版本。Ubuntu 使用 Python 3.10 且缺少 venv 时，先执行：

```bash
sudo apt update
sudo apt install python3.10-venv
```

系统包安装需要 `sudo`；安装成功后运行下面的脚本。如果之前创建虚拟环境失败，
新版 `setup_pc.sh` 会补建缺少 pip 的环境，不删除原目录。

```bash
bash setup_pc.sh
bash start_pc_cameras.sh --list
bash start_pc_cameras.sh
```

发送程序默认监听 `0.0.0.0:9800`，同时接受有线连接，电脑端代码无需因为换网线而重装。
保持上述程序运行后，再启动 Jetson 接收器。电脑防火墙若曾只放行旧 Wi-Fi 地址，需要放行下面的有线地址。

默认序列号是 RealSense SDK 序列号：

| 相机 | 序列号 |
| --- | --- |
| 左腕 D405 | 352122274013 |
| 右腕 D405 | 261122274758 |

如果 `--list` 显示不同序列号，在电脑启动时传入
`--left-serial 序列号 --right-serial 序列号`，并同步修改 Jetson 桥接入口的期望序列号。
不要使用 lsusb 显示的 USB 描述符序列号替代 SDK 序列号。

如果电脑上 USB 访问被拒绝，请按 [RealSense 官方 Linux 安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md)
安装 udev 规则，再重新插入相机。
如果启用了 UFW，仅需允许 Jetson 访问电脑 TCP 9800，例如：

```bash
sudo ufw allow from 192.168.50.10 to any port 9800 proto tcp
```

默认 PNG 无损压缩，不改变 RGB 像素。有线模式先保留该默认值；链路带宽不足时可以在明确接受有损压缩后使用：

```bash
bash start_pc_cameras.sh --codec jpeg --jpeg-quality 95
```

`--codec raw` 不压缩，两路 640×480 RGB@30Hz 的图像净数据约 55.3 MB/s，通常不适合 Wi-Fi。
压缩和联网不能保证 Wi-Fi 始终满足 30Hz；程序会报告实际频率、延迟和跳帧。

## 2. 在 Jetson 上先做纯网络接收检查

此命令只接收图片和测量对时，不启动 ROS 发布、机械臂或吸盘，也不写训练数据：

```bash
ip route get 192.168.50.20
python3 /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/network_cameras/jetson_camera_bridge.py \
  --host 192.168.50.20 --check-seconds 20
```

路由输出应包含 `dev enx00e04c2b7928 src 192.168.50.10`。若连接被拒绝，先确认电脑上的发送程序仍在运行。

看每秒输出中的 `fresh=true`、两路 `receive_hz`、`capture_age_p95_ms`、
`clock_rtt_ms` 和 `clock_error_bound_ms`。开始几秒会等待相机时间和时钟比较稳定。
检查退出码为0还要求两路最近接收速率达到20Hz；低于目标会返回非0。
如果没有有效图像，检查电脑发送进程、地址、端口和防火墙；不会改用接收时刻冒充拍摄时刻。

## 3. 一键启动 Jetson 的采集功能

先保持电脑发送程序运行，然后在 Jetson 执行：

```bash
bash /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/start_xcore_teach_suction_network.sh \
  --fps 20 --task "双臂吸盘抓取"
```

这个入口启动现有双臂服务、Jetson 头部相机、网络双腕接收器和原键盘采集器。
Jetson 不再打开左右腕 USB 相机。旧的全 USB 相机模式入口仍可使用。
`A/Z/B/R/S/1/2/3/4` 功能、回位点和动作参数不变。
`R` 后持续录制，`S` 停止保存；接收临时中断会等待恢复并记录分段。
网络模式有约0.20秒配对缓冲。最后一个动作完成后保持最终状态约0.5秒，再按 `S`，
让末尾动作图像到达并完成匹配后再停止保存。

只启动三路相机、完全不启动机械臂/吸盘/采集器：

```bash
bash /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/start_xcore_teach_suction_network.sh \
  --cameras-only
```

`Ctrl+C` 退出 Jetson 本次启动的进程。电脑发送程序在电脑终端单独用 `Ctrl+C` 退出。
本次启动目录 `/tmp/aide-teach-suction-*/` 中：

- `wrist_cameras.log`：接收、对时、延迟和错误。
- `network_cameras.jsonl`：每秒的双腕接收统计。
- `cameras.log`：Jetson 头部相机。

如果电脑 IP 改变，可使用原统一入口：

```bash
bash /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/recording/start_xcore_teach_suction.sh \
  --wrist-camera-host 新IP --restart-cameras --fps 20 --task "双臂吸盘抓取"
```

相机入口在启动采集器前也检查三路有效图像接收速率至少达到目标保存频率的90%。

## 时间怎么对齐

1. 电脑使用相机 SDK 的 `GLOBAL_TIME`，把相机采集时间戳映射到电脑 UTC Unix 时间。
   启动收敛阶段、时间倒退和超前/过期的时间戳不发送；不退回使用到达时间。
2. Jetson 与电脑通过独立小数据连接持续进行四时间戳比较。
   按 [RFC 5905](https://www.rfc-editor.org/rfc/rfc5905.html) 的 offset/delay 公式，
   从近期低往返延迟样本估计“电脑时钟－Jetson 时钟”，将采集时间戳换算到 Jetson UTC。
   这个过程不设置或跳变任何一台机器的系统时钟。
3. ROS `Image.header.stamp` 保存换算后的**采集时间**，传输延迟不会被加到采集时刻。
   关节、头部图像和腕部图像按这个时间匹配，仍保存每个源的真实时间戳。
4. 网络模式默认等待 0.20 秒再匹配，给晚到的腕部图像留出时间。
   同步时间差上限仍为原来的 50ms，连续性阈值仍为 20Hz 对应的 80ms。

仅将两台电脑设成同一时区不够；UTC 是时间基准，时区只影响显示。
建议采集前让两机通过系统 NTP 服务同步时间（`timedatectl status` 查看）。
采集期间不要手工跳变系统时间。检测到时钟跳变、对时失效或图像过期时会暂停有效图像输出。

默认跨主机时钟比较误差估计上限 10ms，图像最大采集年龄 350ms。
误差估计含半个往返时延和时钟漂移余量，**不包含相机 SDK 曝光时间映射误差**。
Wi-Fi 上下行不对称、系统调度和相机独立曝光会限制精度；这是软件时间对齐，不是三相机曝光硬同步。
对实际网络的对时精度和持续帧率需在搬好相机后实测。
如果网络延迟经常超过默认等待，可传 `--camera-lookahead 0.25`；
持续超过约350ms应先改善网络，不能靠修改图片时间戳掩盖延迟。

## 已提供的测试方式

### 定位偶发间断：电脑取帧、编码和 USB

电脑端先按 `Ctrl+C` 结束旧发送程序，然后在新版电脑代码包目录运行：

```bash
bash diagnose_pc_cameras.sh
```

相机模式、源频率和 PNG 无损压缩保持原设置。终端打印 `PC_DIAGNOSTICS_DIR`，
该目录保存异步的 `capture_*.jsonl`、发送程序输出 `server.log`、USB 拓扑 `system.log`
及内核日志 `kernel.log`。只记录帧号、时戳和耗时，不另存图像。
`capture` 是 SDK 取到的帧，`encoded` 是通过时间检查并编码的帧，`sent` 是已发送的帧。
`timestamp_guard_rejected` 包含正常启动收敛阶段，不能直接当成硬件丢帧。
诊断缓冲有界；尾部 `trace_summary.dropped` 必须为0才能据此比较完整计数。
若内核日志提示权限不足，不能用空日志推断没有 USB 错误。

Jetson 可在不启动 ROS、机械臂和吸盘的情况下持续读取原始帧号：

```bash
python3 /home/jetson/AIDE_robot_main/software/vla/act/xcore_act/scripts/diagnostics/trace_network_wrists.py \
  --host 192.168.50.20 --seconds 180 --out /tmp/aide-wrist-trace-new
```

它兼容旧发送程序。新版发送程序启用诊断后，网络包额外携带 `pc_capture_trace`，
可比较相机帧号、编码序号、前一轮处理时间、编码耗时和收图间隔。
原始帧号跳跃只能说明 SDK 交给应用的帧已不连续，仍需结合耗时和 USB 日志区分
应用漏取帧与硬件传输问题；编码序号连续本身不能证明相机没有丢帧。

`--synthetic` 只生成诊断图像，正常 Jetson 入口拒绝这种图像。
开发测试可在隔离环境显式使用桥接器的 `--allow-synthetic`，这些图像不能用于真实示教训练。
`--test-clock-offset-ms` 仅允许在 synthetic 模式模拟电脑时钟偏差。
真实电脑端 SDK 和实际 Wi-Fi 吞吐需在电脑部署后验证。
