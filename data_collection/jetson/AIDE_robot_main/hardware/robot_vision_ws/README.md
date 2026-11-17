# 机器人相机与视觉统一工作区

当前三台相机使用 `bash scripts/start_cameras.sh`，配置见下表。
左右腕序列号已按实际安装位置纠正；三个采集节点统一发布到 `/robot/sensors/cameras/`。
头部和右腕入口 `bash scripts/start_head_camera.sh` 使用 `/robot_02/sensors/cameras/head`
和 `/robot_02/sensors/cameras/right_wrist`，右腕 D405 序列号为 `352122274535`；
本机配置和检查见 [头部相机说明](HEAD_CAMERA.md)。两种入口共用占用锁，不能同时运行。

本机以 `/home/niic/aide/AIDE_robot_main/hardware/robot_vision_ws` 为相机与视觉功能入口。
统一使用 `ROS_DOMAIN_ID=0`、Fast DDS；下面列出的功能由脚本按需启动，不代表开机自启已部署。
相机驱动是唯一打开 RealSense 硬件的进程；网页、录制、算法和 VR 只订阅图像。
视觉节点统一使用 `/robot/vision/...`，相机驱动节点使用
`/robot/sensors/cameras/{head,left_wrist,right_wrist}`；完整节点命名见
[双臂与视觉节点清单](../../startup/ROS_NODE_NAMES.md)。

## 已迁移的功能

| 原功能 | 本工作区实现 | 启动方式 |
| --- | --- | --- |
| 三相机采集、序列号绑定、对齐深度 | `robot_vision_bringup/multi_realsense.launch.py` + 官方 RealSense 驱动 | `start_cameras.sh` |
| 相机在线状态、离线检测 | `robot_vision_core/camera_manager`，原生 `CameraStatus` 和信息查询服务 | 随三相机启动 |
| 平台原有 JSON 健康状态 | `/robot/vision/camera_status_json` 聚合状态 | 随三相机启动 |
| VR JPEG/UDP 转发 | `robot_vision_tools/udp_video_sender`，端口 6000/6001/6002 | `start_udp_video.sh`，按需启动，见 [说明](UDP_VIDEO.md) |
| 大箱左右抓取点、角度检测 | `robot_vision_core/big_box_detection`，含原模型和消息 | 否 |
| 小箱稳定目标点检测 | `robot_vision_core/small_box_detection`，含原模型和消息 | 否 |
| ArUco ID、去重、启停控制 | `robot_vision_core/aruco_scanner`，改为订阅相机图像 | 否 |
| ArUco 工具位姿与补偿、启停控制 | `robot_vision_core/arucode_position_detection` | 否 |
| 原展示脚本的视觉进程调度 | `robot_vision_core/visual_manager` | 否 |
| RGB-D 数据检查 | `robot_vision_tools/rgbd_probe` | 否 |

相机采集和 UDP 转发统一保留在本工作区，旧相机目录已删除。
原 `startup/startup.launch.py` 中关于旧相机的注释和启动条目已删除。
YOLO、ArUco 和箱体检测均不会因开机而启动，`perception.launch.py` 的默认开关也全部关闭。

## 本机相机配置

三相机注册表：`src/robot_vision_bringup/config/cameras.yaml`。
序列号绑定当前连接的设备，不能用 USB 的 videoN 编号代替。

| 相机 | 型号 | SDK 序列号 | 彩色 | 深度 |
| --- | --- | --- | --- | --- |
| head | D435i | 263022075613 | 640×480，6 Hz | 640×480，6 Hz，另发布彩色对齐深度 |
| left_wrist | D405 | 352122274013 | 640×480，5 Hz | 640×480，5 Hz，另发布彩色对齐深度 |
| right_wrist | D405 | 352122274535 | 640×480，5 Hz | 640×480，5 Hz，另发布彩色对齐深度 |

三台目前共用 USB 2.0 链路，采用低帧率同时采集配置；换用 USB 3.0 后需重新验证高帧率。

ACT 30 Hz 采集另用 `config/cameras_30hz.yaml`（位于 `src/robot_vision_bringup`）：
三台彩色与深度均为 640×480@30，原深度与对齐开关保留。先退出旧相机入口并确认三台接到 USB 3.x，再运行：

```bash
bash /home/niic/aide/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_cameras_30hz.sh
```

该入口复用原占用锁和相同话题；原低帧率入口、注册表不变。设备枚举已确认三台各自支持 VGA 30 Hz，但三台同时 30 Hz 尚待实际 USB 3.x 接线验证。ACT 端的 `--fps 30 --check` 会检查三路图像和四路位置的实测频率及时间戳对齐。

胸部、底盘相机配置仍为禁用占位。未连接的已配置相机会持续等待设备；其他相机可独立出图。
左右腕深度在统一配置中启用，旧采集配置只有头部发布深度。

相机图像与内参话题使用 `/robot/sensors/cameras/<camera_id>/` 前缀：

- `color/image_raw`、`color/camera_info`
- `depth/image_rect_raw`、`depth/camera_info`
- `aligned_depth_to_color/image_raw`、`aligned_depth_to_color/camera_info`

所有相机共用以下诊断和静态 TF 话题，消息中的诊断名称、硬件 ID 和 frame ID 区分相机：

- `/robot/sensors/cameras/diagnostics`
- `/robot/sensors/cameras/tf_static`

驱动不再向公共 `/diagnostics`、`/tf_static` 发布。`/parameter_events` 仍是多节点共用的话题。
静态 TF 的 frame ID 和 transient-local QoS 保留；RViz 或 TF 消费者需订阅相机共用的新路径。
例如查看相机坐标系：

```bash
ros2 run rviz2 rviz2 --ros-args -r /tf_static:=/robot/sensors/cameras/tf_static
```

图像订阅使用传感器 QoS；彩色编码由驱动实际消息给出，消费者已支持 RGB/BGR。
深度图为 `16UC1` 毫米；对齐深度使用彩色像素坐标。
相机到机器人基座的外参尚未完成统一标定接入，不能把预留或旧检测外参当成本机标定结果。

## 域与调用方

相机、状态管理在域 0；网页预览、VR、ACT/InterACT 图像接收上下文也使用域 0。
电机和语音域配置不变。原域桥从域 0 向域 1 转发头部彩色、对齐深度、对齐内参和
JSON 状态，以兼容内部跟随与健康监测功能。图像转发使用 best-effort、队列深度 1。
已有网页、ACT、InterACT、GR00T、rosbag 和平台配置已迁移到新话题名称。

## 构建与运行

```bash
cd /home/niic/aide/AIDE_robot_main/hardware/robot_vision_ws
source /home/niic/aide/AIDE_robot_main/startup/ros_environment.sh
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j2 colcon build --symlink-install \
  --packages-up-to robot_vision_bringup robot_vision_tools --executor sequential \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
source install/local_setup.bash
```

官方驱动固定 4.55.1，匹配本机 `/usr/local` 中的 librealsense 2.55.1；
`src/vendor` 同时包含缺少的 diagnostic_updater 4.0.3、vision_msgs 4.1.1。
RealSense/vision_msgs 的 `SOURCE.json` 记录上游提交和导入文件；不升级系统 ROS 核心库。
默认构建只包含上述目标的依赖，不包含无关的 vendor 示例或诊断工具包。
不参与交付的 vendor 包已用 `COLCON_IGNORE` 排除；`colcon.meta` 为精简导入的官方驱动关闭上游测试构建。

本机手动启动三台相机（`Ctrl+C` 退出并释放设备）：

```bash
bash /home/niic/aide/AIDE_robot_main/hardware/robot_vision_ws/scripts/start_cameras.sh
```

若已单独运行头部相机或通过双臂入口启动头部，先退出该相机采集再运行三相机入口。
本次接入不修改双臂启动流程或开机服务。网页、拍照和算法应订阅上述 ROS 话题。

可选功能（先加载上面的 ROS 和本工作区环境）：

```bash
ros2 launch robot_vision_core perception.launch.py run_big_box:=true
ros2 launch robot_vision_core perception.launch.py run_small_box:=true
ros2 launch robot_vision_core perception.launch.py run_aruco_id:=true
ros2 launch robot_vision_core perception.launch.py run_aruco_pose:=true
bash scripts/start_udp_video.sh --self-test
```

ArUco 默认暂停识别，向 `/robot/vision/aruco/enable` 发 `std_msgs/Int32` 的 1/0 启停。
箱体仍保留 `/box/enable`、`/small_box/enable` 和原输出消息类型，默认暂停识别。
可选 `visual_manager` 的 `/robot/vision/control_cmd`（String）保留原实际行为：
0 启动小箱检测、1 停止小箱检测、2 启动 ArUco ID、3 停止 ArUco ID。

## 当前三相机接入验证

2026-10-16：通过 `scripts/start_cameras.sh` 同时启动三台官方驱动及状态管理节点。
按上表序列号绑定头部、左腕和右腕，三台均取得彩色、原始深度、对齐深度及对应内参。
15 秒可靠 QoS 订阅实测：头部三路收到 87/89/88 帧，左腕 74/75/74 帧，
右腕 74/75/76 帧；全部为 640×480，彩色 `rgb8`，深度 `16UC1`。
注册表与状态监测的 7 项现有测试通过；三相机运行时重复启动三相机或单头部入口均被锁拒绝。

当前 Fast DDS 环境下，best-effort 订阅大图像存在丢帧，双腕状态监测偶尔显示
`STALE`，监测帧率可能低于实际采集值；同一时间可靠 QoS 订阅仍持续收到接近 5 Hz 的双腕图像。
需要完整图像的本机消费者可使用 `rclpy.qos.QoSProfile(depth=5,
reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)`；不能以一次状态 `streaming` 代替长期稳定性检查。
本次未修改系统 DDS 设置、状态监测实现或其他消费者的 QoS。

## 历史部署验收（原设备与配置）

2026-09-10：9 个 ROS 包构建成功；相机注册表/健康监测 7 项、网页后端 51 项、
域桥 17 项、InterACT 相机/配置 9 项相关测试通过。VR JPEG 编码自检通过，
大小箱模型加载、ArUco 启停和深度几何换算通过。

当前实际检测到的左腕相机已收到 640×480 彩色、原始深度和对齐深度，状态为 OK，
信息查询可返回内参；网页订阅器已生成真实 JPEG 预览。
头部和右腕未找到配置的设备，显示 OFFLINE，尚未完成这两路真机验收。
配置的 30 Hz 是目标值，实际订阅帧率还受设备和 DDS 负载影响；未作三路满负载帧率验收。
