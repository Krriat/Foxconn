# 三路相机 ROS 2 发布

部署日期：2026-11-12。两台电脑均为 Ubuntu 22.04 / ROS 2 Humble。

电脑 `172.16.12.45` 打开两台手腕 D405，Jetson 控制器 `172.16.12.95`
打开头部 D435i 的 RGB 模组。每台设备只由其所在机器上的相机节点采集一次，
其他程序通过 ROS 订阅；本目录的预览和验证程序均不会再次打开相机硬件。

## 硬件对应关系

左右关系通过用户拔掉右腕相机、重新枚举、插回后的二次枚举确认。

| 位置 | 型号 | 采集机器 | 用于绑定的标识 | USB 设备序列号 |
| --- | --- | --- | --- | --- |
| 左腕 | D405 | 172.16.12.45 | SDK 序列号 `352122274013` | `254623078816` |
| 右腕 | D405 | 172.16.12.45 | SDK 序列号 `261122274758` | `261923072940` |
| 头部 RGB | D435i | 172.16.12.95 | USB 序列号 + 接口 `03` | `262923063591` |

SDK 序列号与 USB 描述符中的序列号不同，请不要混用。头部控制器上的
`/dev/v4l/by-id` 链接在深度与 RGB 模组间重名；程序通过 sysfs 中的 USB
序列号、接口 `03`、视频索引 `0` 定位 RGB 模组，避免误选红外接口。
插拔后 `/dev/videoN` 改变时会重新定位设备。

## 话题契约

三路均为 640×480，**硬件采集 30 fps，ROS 图像发布目标 20 Hz**，
与 ACT 数据集的 20 Hz 观测时间基准对齐。2026-11-12 已在两个发布主机部署，
其他电脑订阅三路 JPEG 的 60 秒实测为头部 20.00 Hz、左腕 19.90 Hz、右腕 19.91 Hz。
以下 `CAMERA` 分别替换为
`head`、`left_wrist`、`right_wrist`。

| 话题 | 消息类型 | 内容 |
| --- | --- | --- |
| `/robot/sensors/cameras/CAMERA/color/image_raw` | `sensor_msgs/msg/Image` | `bgr8` 原始图像；有订阅者时发送 |
| `/robot/sensors/cameras/CAMERA/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG，质量 80；适合当前无线网络 |
| `/robot/sensors/cameras/CAMERA/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 图像尺寸、坐标系、标定状态 |
| `/robot/sensors/cameras/status/robot_wrist_cameras` | `std_msgs/msg/String` | 两路手腕的采集状态、帧率、重连计数 |
| `/robot/sensors/cameras/status/robot_head_camera` | `std_msgs/msg/String` | 头部采集状态、帧率、重连计数 |

图像和 CameraInfo 的 QoS 为 `BEST_EFFORT / VOLATILE / KEEP_LAST(1)`。
订阅程序应使用 `rclpy.qos.qos_profile_sensor_data` 或等效 QoS。
状态话题为可靠传输，每秒更新一次。

配置中的 `fps: 30` 指硬件采集档位，`publish_fps: 20` 指 ROS 发布目标。
当前 D405 彩色 640×480 支持 15/30 等档位，没有原生 20 fps 档位；
头部 UVC 的该分辨率支持 6/15/30/60 fps。
采集线程持续读取，发布线程每 50 ms 选取最新、尚未发布的新帧；不排队补发，
不重复发送同一采集帧来凑帧率。SDK 图像复制后存入单帧缓存，避免底层复用缓冲区。
相机无新帧或采集失败时不会维持虚假的 20 Hz 输出。
状态中的 `capture_fps` 为实测采集帧率，`fps` 为实测发布帧率，
`requested_capture_fps` 和 `requested_fps` 分别为两者的目标值。

对应图像与 CameraInfo 使用同一采集时间戳，选帧后不改写为发送时刻。frame_id 为
`head_color_optical_frame`、`left_wrist_color_optical_frame`、
`right_wrist_color_optical_frame`。时间戳来自各自主机时钟；这些流没有实施
硬件同步或跨主机时钟同步验证。

手腕 CameraInfo 来自 RealSense SDK。头部使用 UVC 彩色采集，尚未载入
标定内参，CameraInfo 的 `K[0] == 0` 明确表示未标定，不能把它作为已标定
相机进行几何计算。本次配置发布彩色画面；不涉及深度或点云。

## 在其他电脑订阅

订阅端连接能与两台发布机器互通的局域网，并允许 ROS 2 DDS 的 UDP 发现和
数据通信。本次已验证电脑与 Jetson 的双向发现和图像订阅；其他网段或经过
隔离的网络需要另行配置。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 topic list --no-daemon --spin-time 3
ros2 topic hz /robot/sensors/cameras/head/color/image_raw/compressed
```

复制本目录到订阅端，可运行纯订阅预览程序。需要 `rclpy`、`sensor_msgs`、
NumPy 和带 GUI 支持的 OpenCV；发布机器已具备这些依赖。

```bash
python3 view_cameras.py
# 也可以只看其中一路
python3 view_cameras.py --cameras head
```

预览按 Q 或 Escape 退出。无图形桌面时，可运行验证程序，把订阅收到的
图像保存为 JPEG，并统计帧率、尺寸、时间戳和解码错误：

```bash
python3 verify_streams.py --seconds 20 --output ./received_frames
# 检查发布频率是否在 20 ± 1 Hz 内，并统计时序、状态和最近时间戳配对差
python3 verify_streams.py --seconds 60 --expected-fps 20 --output ./received_20hz
```

原始消息订阅的类型为 `sensor_msgs.msg.Image`，压缩消息类型为
`sensor_msgs.msg.CompressedImage`。压缩数据可用
`cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)` 解码。

## 后台运行和维护

运行文件部署在两端各自的 `~/.local/share/robot-camera-streams/`。

电脑上的服务为 `robot-wrist-cameras.service`：

```bash
~/.local/share/robot-camera-streams/start.sh wrists
systemctl --user status robot-wrist-cameras --no-pager
systemctl --user stop robot-wrist-cameras
```

Jetson 上的服务为 `robot-head-camera.service`，登录后执行：

```bash
~/.local/share/robot-camera-streams/start.sh head
systemctl --user status robot-head-camera --no-pager
systemctl --user stop robot-head-camera
```

这些是当前会话的 systemd 用户临时服务，异常退出时自动重启，摄像头断开后
每 2 秒重试。没有启用开机自启；重启或用户会话结束后，按上述命令重新启动。
使用 Fast DDS 异步发送，降低网络较慢的原始图像订阅者对采集线程的影响。

本目录的源码与配置为部署副本的来源，原有 `source_snapshot/` 阅读快照保持原状。
更改配置后，需要同步到相应主机的运行目录，再重启对应服务。

## 实测证据

`evidence/laptop_jpeg/` 保存电脑通过 ROS 收到的三路预览和 20 秒统计，
其中头部图像经过无线网络从 Jetson 传回；`evidence/jetson_jpeg/` 保存
Jetson 收到的三路预览和统计，其中两个手腕图像经过无线网络从电脑传来。
`evidence/laptop_raw/` 和 `evidence/jetson_raw/` 保存原始消息测试结果。
具体测量结果见 `VALIDATION.md`。

上述无 `20hz_20261112` 前缀的目录保留了最初 15 fps 发布的历史测试。
当前 20 Hz 的订阅统计、图像、时钟估计和只读模型测试见
`evidence/20hz_20261112/`，说明见 `20HZ_VALIDATION.md`。
