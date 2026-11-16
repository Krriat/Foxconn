# 从检测话题选择点并写入 text_dct.conf

`text_points.py` 是配套的 Python ROS 2 选点工具，C++运动程序不变。
它只订阅检测结果、预览和编辑配置，不发布运动指令、不初始化DO2。
保存后仍由用户编辑 `[steps]`，下一次启动 text_dct 才读取新配置。

## 启动与使用

在 `cxl` 目录，选择能收到检测话题的 ROS 域；这里不强制设0或1。
检测进程和选点进程必须使用相同域（或已有桥接），不一定等于夹爪驱动的域。

```bash
source /opt/ros/humble/setup.bash
# export ROS_DOMAIN_ID=实际检测节点所在域
python3 text_points.py --check
python3 text_points.py
```

也可指定 `--sources ./other.sources.json --conf ./other.conf`。
Python解释器需要能导入rclpy、rosidl_runtime_py及所订阅的消息包；
自定义消息请先source对应工作空间的install/setup.bash。工具不依赖YOLO、OpenCV或检测代码的推理环境。

进入交互命令：

```text
list
pick part_left
show
save detected_pick pose2
save detected_above pose2 0 0 0.05
quit
```

- `list`：列出每个来源最新候选点、序号、坐标系、类别/置信度、接收后经过的秒数或错误。
- `pick part_left`：冻结该来源此刻的最新有效点；后续检测变化不会偷偷替换已选点。
- `show`：查看冻结数据。
- `save detected_pick pose2`：新增点名detected_pick，XYZ来自选点变换结果，RPY沿用当前conf中pose2。
- 后面的三个偏移量单位m，在**目标控制器参考系**中相加。例如 `0 0 0.05` 是目标系Z+5cm，并不保证是实际竖直向上。
- `save detected_pick pose2 replace`：显式允许覆盖已有detected_pick；默认重名拒绝。
- 保存前打印conf差异及冻结时间，输入 `SAVE` 才写入。自动生成 `.bak.时间戳` 备份；检测到conf在预览后被修改会拒绝覆盖。
- 只增改指定 `[poses]` 行，其余点、关节、步骤和注释保持原样；不会自动插入运动步骤。

保存后自行编辑conf，例如点属于左臂时：

```conf
move left detected_above 30
move left detected_pick 30
```

这只是语法，不代表路径已经验证。再执行 `../build/bin/text_dct --check` 查看完整流程。
选中的点是静态快照，之后目标移动、相机/腰部位置或控制器坐标变化时，需要重新检测或标定。

## 当前检测节点提供什么

`/home/niic/zhenguo/tianjin_part_detection/part_detection.py` 默认发布：

| 话题 | 类型 | 信息 |
| --- | --- | --- |
| `/vision/part_target` | std_msgs/msg/String，内部JSON | detected、XYZ、frame_id、class、confidence、stamp |
| `/vision/material_coord` | geometry_msgs/msg/PoseStamped | XYZ和时间戳；四元数固定为单位四元数 |

优先使用JSON来源，可校验detected并显示类别。检测器每帧只输出最左侧目标，
工具的list是**各话题最新点列表**，不是画面所有零件列表，也不是目标跟踪ID列表。
如果需要选择同一画面的全部零件，需扩展检测器发布全部三维目标，当前工具未修改检测器。
PoseStamped并未估计抓取姿态，因此所有来源当前均采用已有conf点的RPY，不自动使用消息四元数。

## 必须明确坐标变换

当前按用户约定：检测端负责变换到所用机械臂坐标系。因此随附来源配置已使用单位矩阵，
直接导入XYZ，不做额外坐标变换。frame_id按检测代码当前值保留base_link校验；
若检测端修改frame_id，同步修改expected_frame。
part_left和part_right目前订阅同一话题，是供选择目标臂的两个配置入口，
**不会自动把左臂坐标变成右臂坐标**。只选择与检测端实际输出匹配的入口；
后续左右臂分别发布时，分别修改对应topic。

来源默认frame为base_link，但text_dct的 `[poses]` 使用对应臂控制器外部参考系。
不能只因坐标值看起来相近就当作同一系。`text_points.sources.json` 中左右臂各自有：

```json
"arm": "left",
"target_frame": "",
"transform": null
```

未标定时允许接收和选点，禁止写入运动点。确认坐标映射后填实际target_frame说明和4×4矩阵：

```text
p_controller = T_controller_from_detection × [x, y, z, 1]^T
```

平移单位m；旋转须正交、行列式+1。不查询或改写控制器BaseFrame/工具配置，不自动获取TF。
**只有已确认检测frame和控制器外部参考系完全一致时**，才使用单位矩阵：

```json
"target_frame": "left_controller_external",
"transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
```

这里的target_frame是映射说明，并不是修改ROS TF的指令。左右臂映射应分别填写。
姿态模板已经处于目标控制器系，因此工具仅变换XYZ，不再次旋转模板RPY。

## 扩展话题和消息类型

在JSON的sources里复制一个来源并改名：

- `topic`：订阅话题；`type`：已安装ROS消息类型，如geometry_msgs/msg/PointStamped或自定义包/msg/类型。
- `json_field`：当消息某字段内包含JSON时指定，例如String的data；普通结构化ROS消息省略此项。
- `xyz`：三个字段路径，例如PointStamped使用 `["point.x", "point.y", "point.z"]`。
- `frame_path` / `expected_frame`：坐标系字段和预期值，不匹配拒绝。
- `stamp_path`：可选时间戳路径；配置后必须是sec/nanosec结构，检查消息是否过期或来自未来。
- `require`：可选字段匹配条件，如 `{"detected": true}`；无效结果会清除该来源的最新候选。
- `metadata`：可选展示字段，如类别/置信度；不要把无时间关联的另一个话题类别拼接到点上。
- `scale`：XYZ到米的倍率，米1、毫米0.001。
- `max_age_s`：消息时间和接收时间的新鲜度上限，默认5秒；pick拒绝过期数据。
- `reliability`：reliable或best_effort，按发布端QoS配置；默认volatile、不读取历史持久化目标。
- `arm` / `target_frame` / `transform`：目标臂、控制器参考系说明、刚体变换。

字段路径支持字典/消息字段和数组索引，如 `targets.0.position.x`。
一个来源映射一个点；动态长度目标列表、四元数抓取姿态推导等更复杂行为需新增解析逻辑，
不要仅更改type就假定所有语义相同。缺少stamp_path的类型只能检查接收时间，无法判断上游是否重发旧数据。
配置在工具启动时加载，修改来源配置后重启工具。

离线测试：`python3 test_text_points.py`。测试不连接ROS、不改真实conf。
