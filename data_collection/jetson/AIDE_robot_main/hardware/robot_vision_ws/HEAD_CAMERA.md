# 当前机器人头部和右腕相机

本机 D435i 序列号为 `263022075613`，通过 USB 2.1 连接。
单独配置见 `src/robot_vision_bringup/config/head_camera.yaml`，使用已验证的
1280×720、15 Hz 彩色模式及 640×480、15 Hz 深度模式，
通过现有官方 RealSense ROS 驱动发布。

同一入口现已增加右腕 D405，SDK 序列号 `352122274535`，彩色和深度均为
640×480、30 Hz，发布到 `/robot_02/sensors/cameras/right_wrist` 下。
提供 `color/image_raw`、`color/camera_info`、`depth/image_rect_raw`、
`depth/camera_info`、`aligned_depth_to_color/image_raw` 和
`aligned_depth_to_color/camera_info`。也可通过
`bash /home/jetson/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_head_camera.sh`
启动同一相机组。VR 视频发送见 [UDP 视频说明](UDP_VIDEO.md)。

| 接口 | ROS 类型 |
| --- | --- |
| `/robot_02/sensors/cameras/head/color/image_raw` | `sensor_msgs/msg/Image` |
| `/robot_02/sensors/cameras/head/color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/robot_02/sensors/cameras/head/color/metadata` | `realsense2_camera_msgs/msg/Metadata` |
| `/robot_02/sensors/cameras/head/depth/image_rect_raw` | `sensor_msgs/msg/Image` |
| `/robot_02/sensors/cameras/head/depth/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/robot_02/sensors/cameras/head/depth/metadata` | `realsense2_camera_msgs/msg/Metadata` |
| `/robot_02/sensors/cameras/head/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` |
| `/robot_02/sensors/cameras/head/aligned_depth_to_color/camera_info` | `sensor_msgs/msg/CameraInfo` |

本机实测图像编码为 `rgb8`，内参对应当前彩色分辨率。
深度图编码为 `16UC1`，像素值单位为毫米，`0` 表示无有效深度。
原始深度图采用 `head_depth_optical_frame`，对应自身的深度内参。
已启用 `align_depth` 和同步帧：对齐深度图为 1280×720、`16UC1`，
采用 `head_color_optical_frame` 和彩色相机内参。
驱动根据两个传感器的内参及它们之间的外参，将深度映射到彩色图的像素网格；
彩色图 `(u, v)` 位置可从对齐深度图同一像素读取深度，单位仍为毫米。
遮挡、视野差异或没有测量结果的位置可能为 `0`；对齐后的尺寸不提高深度传感器的物理分辨率。

算法侧可配置：

```python
COLOR_TOPIC = "/robot_02/sensors/cameras/head/color/image_raw"
DEPTH_TOPIC = "/robot_02/sensors/cameras/head/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/robot_02/sensors/cameras/head/aligned_depth_to_color/camera_info"
```

相机内部静态 TF 发布到 `/robot_02/sensors/cameras/tf_static`；
相机到机器人基座的安装外参尚未标定。

构建与启动：

```bash
cd /home/jetson/AIDE_robot_main/hardware/robot_vision_ws
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# 本机 SDK 开发库位于项目内；使用系统 SDK 的机器可省略此行。
export CMAKE_PREFIX_PATH="/home/jetson/AIDE_robot_main/.tools/realsense-sdk/opt/ros/humble${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
MAKEFLAGS=-j2 colcon build --symlink-install --packages-up-to robot_vision_bringup \
  --executor sequential --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
bash scripts/start_head_camera.sh
```

推荐通过双臂统一入口同时启动相机：

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/arm/xCoreROS
bash tools/start_state_publisher.sh
```

`start_upperlimb.sh`、`start_state_publisher.sh`、`start_sdk_nodes.sh` 都使用同一入口。
它先验证彩色、原始深度和对齐深度，再启动双臂接口。
已运行的相机会被复用；新启动的相机随统一入口退出，复用的相机保持运行。
这些入口不会启动箱子检测算法，也未配置开机自启。
仅需双臂时可传 `start_head_camera:=false`。

若单独使用临时后台服务 `aide-head-camera.service`，可通过以下命令管理：

```bash
systemctl --user status aide-head-camera
systemctl --user restart aide-head-camera
systemctl --user stop aide-head-camera
```

前台运行启动脚本时先停止后台实例，确保同一相机只有一个采集进程。
独立脚本与统一入口使用同一相机启动锁，并检查已运行的头部驱动，防止重复占用 USB。
后续消费者应订阅 ROS 话题，拍照时也不再单独打开 RealSense pipeline。

检查与预览：

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
source install/local_setup.bash
ros2 topic hz /robot_02/sensors/cameras/head/color/image_raw
ros2 topic hz /robot_02/sensors/cameras/head/depth/image_rect_raw
ros2 topic hz /robot_02/sensors/cameras/head/aligned_depth_to_color/image_raw
ros2 topic echo /robot_02/sensors/cameras/head/color/camera_info --once
ros2 topic echo /robot_02/sensors/cameras/head/depth/camera_info --once
ros2 topic echo /robot_02/sensors/cameras/head/aligned_depth_to_color/camera_info --once
ros2 run rqt_image_view rqt_image_view
```

当前启用同步彩色、原始深度、对齐到彩色的深度及对应内参。
三相机配置 `cameras.yaml` 已接入当前头部及双腕，入口为 `bash scripts/start_cameras.sh`，
使用 `/robot/sensors/cameras/` 前缀；左右腕分别绑定 `352122274013` 和 `352122274535`。
`start_head_camera.sh` 现在启动头部和右腕，不启动左腕或视觉算法，
与三相机入口共用占用锁。头部的既有参数保持原样。

彩色初次部署已验证：1280×720 RGB 图像持续更新，订阅约 13.8 Hz，只有一个图像发布者，
图像与 CameraInfo 时间戳及光学坐标系一致；从 ROS 订阅画面成功保存照片。

启用深度后已同时验证彩色与深度持续发布：彩色约 14.2 Hz、深度约 14.6 Hz。
深度编码为 `16UC1`、640×480，采样画面约 90.2% 像素具有非零深度；
深度图与对应内参时间戳一致，并成功保存原始 16 位深度 PNG。

启用对齐后已验证：彩色及对齐深度约 14.6 Hz，原始深度约 14.8 Hz；
对齐深度为 1280×720、`16UC1`，与彩色图共用时间戳及光学坐标系，
对应 CameraInfo 的 K/P/R/D 与彩色内参一致；采样图约 94.0% 像素具有非零深度。

右腕 D405 已单独实测：640×480 彩色约 29.2 Hz、原始深度约 29.5 Hz、
对齐深度约 29.3 Hz；图像和对应内参时间戳一致，对齐深度使用彩色内参和
`right_wrist_color_optical_frame`。绑定使用 SDK 返回的 `352122274535`，
不使用不同的 USB 设备标签或随插拔变化的 videoN。
测试时仅检测到右腕设备，尚未验证头部和右腕同时出图。
