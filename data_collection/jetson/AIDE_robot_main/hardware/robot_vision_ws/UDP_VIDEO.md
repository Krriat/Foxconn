# VR UDP 视频发送

原独立目录的发送功能统一由 `robot_vision_tools/udp_video_sender` 提供。
节点 `/robot/vision/video_sender` 只订阅 ROS 彩色图像，不打开相机，不启动机械臂。
每帧仍是一个完整 JPEG、一个 UDP 数据报，无自定义包头，最大 65,507 字节；
默认目标 IP `172.16.13.184`、JPEG 质量 50、每帧发送，超长 JPEG 自动降低质量。

| 相机 | UDP 端口 | 当前启动脚本订阅的话题 |
| --- | --- | --- |
| head | 6000 | `/robot_02/sensors/cameras/head/color/image_raw` |
| left_wrist | 6001 | 当前头部和右腕配置不启用左腕 |
| right_wrist | 6002 | `/robot_02/sensors/cameras/right_wrist/color/image_raw` |

先按原方式启动头部和右腕采集：

```bash
bash /home/jetson/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_head_camera.sh
```

首次构建发送工具：

```bash
cd /home/jetson/AIDE_robot_main/hardware/robot_vision_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash
colcon build --symlink-install --packages-select robot_vision_tools
```

只检查 JPEG 编码，不发送网络数据：

```bash
bash scripts/start_udp_video.sh --self-test
```

另开终端发送视频（目标 IP 可覆盖）：

```bash
bash /home/jetson/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_udp_video.sh \
  --target-ip 172.16.13.184 \
  --status-file /tmp/xcore-udp-video-status.json
```

脚本默认读取采集入口使用的 `config/head_camera.yaml`（位于
`src/robot_vision_bringup`），只等待并发送其中已启用的彩色相机。
状态 JSON 继续包含 `status`、`target_ip`、`ports`、`cameras`、`frames_sent`、
`bytes_sent`、`fps`、`jpeg_quality`、`error` 和 `updated_at`，按 `Ctrl+C` 停止。
默认 10 秒内未收到配置中的全部相机图像会报告失败；不要同时启动多个发送实例。

使用三相机采集入口时指定对应注册表，订阅 `/robot/sensors/cameras/` 下的三路彩色：

```bash
bash scripts/start_udp_video.sh \
  --camera-registry "$PWD/src/robot_vision_bringup/config/cameras.yaml" \
  --target-ip 172.16.13.184 \
  --status-file /tmp/xcore-udp-video-status.json
```

其他参数 `--jpeg-quality`、`--send-every`、`--startup-timeout` 保持可用。
直接调用 `ros2 run robot_vision_tools udp_video_sender` 或 Python 模块时，
不传 `--camera-registry` 仍使用原有三相机 `/robot/sensors/cameras/` 话题，
保留既有网页调用方式。
