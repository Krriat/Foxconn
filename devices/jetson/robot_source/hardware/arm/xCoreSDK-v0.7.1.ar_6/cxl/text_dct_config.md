# text_dct 流程配置

按物料类别旋转底盘：`spin_if_class pose2cs 2 0.25 0.2`，
类别来自与坐标同帧的检测JSON；配置和服务启动见 [text_chassis.md](text_chassis.md)。

默认运行一次；`../build/bin/text_dct --loop`循环执行，每轮重新取话题点，
下一轮无新点时结束。循环详情见[text_topic_points.md](text_topic_points.md)。

自动化话题取点：在 `[topic_poses]` 声明点名，然后直接在运动步骤引用；
`text_dct` 内部C++模块自动订阅和解析，无需Python选点工具。
配置、取点时机和扩展方式见 [text_topic_points.md](text_topic_points.md)。

偏移点在 `[offset_poses]` 中写 `新点名 基准点名 DX DY DZ`（米），例如
`pose2cs_above pose2cs 0 0 0.05`。新点XYZ叠加偏移、RPY保持不变，原点仍可使用；
运动步骤直接引用新点名。多个偏移点复用同一话题点快照，详见上述文档的偏移点说明。

## 独立分支同时运行

原有 `parallel_begin` / `parallel_end` 保留：一组单步动作同时执行，全部完成才继续。
新增 `concurrent_begin` / `concurrent_end`：多个分支同时启动，每个分支内部顺序执行
多条动作，各自完成一步就执行下一步，不需要与其他分支的步骤数量、时间对应。
所有分支完成后才执行 `concurrent_end` 后面的步骤。提前完成的分支不会自动循环。

格式示例（请替换实际点名和夹爪计数后使用）：

```conf
[steps]
concurrent_begin
branch left_work
move left left_pick 30
wait 2
move left left_place 30
move left left_pick 30
branch_end

branch right_work
move right right_home 30
move right right_pick 30
move right right_home 30
branch_end

branch gripper_work
gripper right 2000
wait 1
gripper right 2100
wait 1
gripper right 2000
branch_end
concurrent_end
wait 2
```

2000/2100仅用于展示整数格式，不代表已确认的张开/闭合位置。
`wait` 只等待所在分支；重复写动作即可让一个执行器反复运动。
每个分支可使用多个设备，例如同一分支内“左臂到点→左夹爪夹紧→左臂离开”。
有先后依赖的操作应放在同一分支。不同分支没有逐步等待关系。

每块至少两个非空分支，名称不得重复；分支及并行块暂不支持嵌套。
同一执行器只能被一个分支使用，但该分支可以多次操作它；左右臂、左右夹爪、
升降轴、DO2分别算独立设备。跨分支冲突会在 `--check` 阶段拒绝。
该规则避免同时向同一设备发送不同目标。

异常或Ctrl+C会取消其他分支的后续动作，按现有接口尝试停止正在执行的臂/升降运动，
等待工作线程退出后清理。夹爪仍受“驱动没有停止接口”的限制。
这是软件并发，不是多设备轨迹联动；不增加避碰或腰部外轴补偿。

编辑同目录 `text_dct.conf`，保存后在下一次启动时生效，无需重新编译。
程序启动时读取配置快照；执行中修改配置不会改变正在运行的流程。
默认读取源文件旁的配置，不依赖终端当前目录。

在 `cxl` 目录：

```bash
# 只检查配置并显示步骤，不连接机械臂、不申请或改变 GPIO
../build/bin/text_dct --check
# 执行默认配置，输入 MOVE 确认后执行一次
../build/bin/text_dct
# 也可保存多套流程并指定配置路径（相对当前终端目录）
../build/bin/text_dct --check ./other.conf
../build/bin/text_dct ./other.conf
```
示教
左臂：
../build/bin/cartesian_drag_safe left 300
右臂：
../build/bin/cartesian_drag_safe right 300
升降
1
source /opt/ros/humble/setup.bash
source /home/niic/aide/hardware/liftmotor/install/setup.bash
ros2 launch liftmotor_rs485 liftmotor_rs485.launch.py  # 仅在开机流程未自动启动时使用
2
source /opt/ros/humble/setup.bash
source /home/niic/aide/hardware/liftmotor/install/setup.bash

# 先查看连接和报警状态
ros2 topic echo --once /robot/lift/state
3
# 上升 0.5 毫米
ros2 run liftmotor_rs485 move_relative_client -0.5

## 配置格式

- `[settings]`：
  - `speed_mm_s 50`：未单独指定时的默认 MoveL 速度，范围 1–250 mm/s。
  - `max_move_distance_m 0.30`：每段运动起终点的最大距离，范围 0.001–2.0 m。
    它同时检查配置中同一机械臂的相邻目标，以及执行时当前位置到目标点的距离。
- `[poses]`：每行 `名称 X Y Z RX RY RZ`，位置单位米，姿态为 RPY 弧度。
  使用拖动程序输出的 `endInRef` 数据，保持相同 TCP、工具负载及参考系。
  左右臂位姿放在同一区域；名称可自定义、不能包含空格或重复，建议右臂名称加 `right_` 前缀。
  未记录的肘角沿用每段运动开始时的当前值。
- `[steps]`：按从上到下的顺序执行，可增删、重复或调整步骤：
  - `move left 名称`：左臂用 `[settings]` 的默认速度 MoveL 到指定点。
  - `move right 名称 30`：右臂本段使用 30 mm/s MoveL 到指定点。
  - 旧格式 `move 名称 [速度]` 继续支持，并默认表示左臂。
  - `lift -0.5`：腰部升降轴相对上升 0.5 mm；正数下降、负数上升，单位毫米。
    程序等待升降轴停止后才进入下一步；升降速度使用升降节点 YAML 的 `speed_rpm`。
    `text_dct` 不限制相对位移幅度；实际允许行程由升降节点、驱动器和硬件限位决定。
  - `lift_abs 300`：腰部移动到驱动器内部坐标 300 mm。绝对位置允许为 0，
    程序等待升降轴停止且反馈连续 3 次进入目标 ±100 pulse（当前标定约 ±0.05 mm）后
    才进入下一步。该坐标基于驱动器零点，并非离地高度。
  - `do 0`：DO2 输出低电平（本项目约定“关”）。
  - `do 1`：DO2 输出高电平（本项目约定“开”）。
  - `wait 10`：保持当前输出和位置，等待 10 秒；支持小数，如 `wait 0.5`，范围 0–3600。
- `#` 后为注释。每行只写一个位姿、设置或步骤，不使用逗号或等号。

默认配置已写入用户要求的 11 个步骤；`pose2`、`pose3`、`pose4` 需要填写实际坐标。
缺少位姿、数字无效、重复名称或步骤拼写错误会在任何硬件操作前报错。
若只需两个点，删除 `[steps]` 中对 pose2/pose3/pose4 的引用即可。

## 运行行为

左臂地址固定为 `192.168.1.160`，本机地址 `192.168.1.10`；右臂地址固定为
`192.168.2.160`，本机地址 `192.168.2.10`。程序只连接步骤中出现的机械臂。
块外按顺序执行；`parallel_begin` 到 `parallel_end` 内并发执行，所有动作完成后才继续。
配置包含 `lift` 时，须确保 `/robot/lift` ROS 2 节点已经运行；机器人开机流程通常会自动启动它，
不要再启动第二个节点占用同一串口。在运行 `text_dct` 的终端中
source ROS Humble 和 `/home/niic/aide/hardware/liftmotor/install/setup.bash`。
DO2 固定为 `/dev/gpiochip0` 的 line 6；不要同时运行占用它的 Python 脚本。
程序开始时将输出初始化为 0；正常结束、Ctrl+C 或异常退出时尝试恢复为 0。
这些初始化/清理动作始终存在，即使配置中未写 `do 0`。
程序保持 GPIO 句柄直到退出；释放后的持续电平取决于 GPIO 驱动/硬件。
正常结束后机械臂保持上电。Ctrl+C 是软件停止，依赖通信，不是实体急停。

每段位移限制由 `[settings]` 的 `max_move_distance_m` 设置；保留60秒运动超时、
1 mm位置和0.01 rad姿态到达判断。
所有机械臂运动到点停止（zone=0）。配置检查包含相邻目标间距离；第一段与当前位置的距离在连接后检查。
`--check` 不验证实机可达性、奇异位形或碰撞；路径、负载和工具配置仍需实际确认。

只有修改 C++ 代码才需要重新编译：

```bash
cmake --build ../build --target text_dct -j2
```

## 关节运动、点到点与并行块

- `move` / `movel left pose0 100`：读取 `[poses]` 的6维末端位姿，MoveL直线，速度mm/s。
- `movej left pose0 0.2`：读取 `[poses]`，MoveJ关节插补到指定末端位姿，TCP路径不保证直线。
- `moveabsj right right_home 0.2`：读取 `[joints]` 的7个关节角(rad)，MoveAbsJ关节插补。
- MoveJ和MoveAbsJ速度为比例 `(0,1]`，0.2表示20%，不是rad/s或mm/s。
  省略速度采用 `[settings]` 中的 `joint_speed_ratio`，默认0.2。
- 每个动作均到点停止。MoveAbsJ完成要求各轴误差不超过 joint_tolerance_rad；
  末端位姿动作要求位置误差1 mm内、姿态误差0.01 rad内，并收到对应指令完成事件。
- `max_move_distance_m` 检查起终点距离。MoveAbsJ在连接后通过FK计算终点并检查。
  该限制不能保证PTP的中间路径长度、关节行程或碰撞安全。

格式示例（点名必须预先定义，关节值必须用实际示教数据）：

```conf
[joints]
# left_home J1 J2 J3 J4 J5 J6 J7
# right_home J1 J2 J3 J4 J5 J6 J7

[steps]
# moveabsj left left_home 0.2
parallel_begin
movej left pose0 0.2
movej right pose0_r 0.2
parallel_end
wait 2
```

并行块可包含左右臂、左右夹爪、lift/lift_abs、do及wait。每个块中同一臂最多一条运动、
升降最多一条、DO2最多一条；不支持嵌套。块内wait与其它动作同时开始，
不会延迟后面一行的开始时间。DO2仍为同一根GPIO输出，不按左右臂区分。
任一动作报错或Ctrl+C会取消组内尚未完成的动作，并尝试停止正在运动的设备；
所有工作线程退出后再清理DO2，且不执行后续步骤。停止依赖SDK/ROS通信。

并行是软件并发启动，不是控制器时钟同步，不保证同时到达，也不做双臂互相避碰。
升降与机械臂并发时，TCP相对桌面的运动还叠加了腰部位移，程序没有外轴联动补偿。
先用 `--check` 查看动作、目标臂、速度单位及并行组编号；它不验证实机路径。

离线回归验证：`python3 cxl/test_text_dct_offline.py`（在SDK根目录运行，需先编译text_dct）。

完成事件后的反馈校验：最多等待2秒，连续3次满足配置的到位阈值才继续。
MoveAbsJ正常到位时隐藏七轴误差日志；持续超差时打印每轴目标值、实际值和误差(rad)，并停止流程，不自动重发运动。

`[settings]` 的 `joint_tolerance_rad` 控制 MoveAbsJ 各轴到位验收容差，单位rad，
范围 `(0,0.01]`，未填写时默认0.001；当前conf显式设置0.002。
它不修改驱动器参数、目标关节角或MoveL/MoveJ的末端位姿验收阈值。

## 左右夹爪（STS3215）

左右夹爪驱动须提前运行，且与本终端使用相同的 ROS_DOMAIN_ID。
使用已有 `sts3215_driver` 驱动；若开机流程已启动，不要重复启动。
话题消息均为 `std_msgs/msg/Int32`：

| 夹爪 | 位置命令话题 | 位置反馈话题 |
| --- | --- | --- |
| 左 | `/robot/gripper/left/command/position` | `/robot/gripper/left/state` |
| 右 | `/robot/gripper/right/command/position` | `/robot/gripper/right/state` |

在 `[steps]` 中使用：

```conf
gripper_read left   # 读取并打印左夹爪当前计数
gripper_read right  # 读取并打印右夹爪当前计数
# gripper left 2000  # 示例格式：左夹爪运动到2000计数，实际值请按机构确认
# gripper right 2000
```

目标是整数位置计数，范围0–4095，不是弧度或毫米，也不是开关值。
驱动配置的 min_position/max_position 可能进一步缩小范围。
先读取已确认张开/闭合位置的计数，再将对应值用于流程；不默认0代表张开。
速度沿用驱动的 move_speed 参数，当前位置话题不支持每条命令单独指定速度。

可在现有 `[settings]` 区域设置：

```conf
gripper_timeout_s 10          # 到位等待超时，范围(0,120]秒
gripper_tolerance_counts 20   # 到位容差，整数0–4095计数
```

`gripper` 先确认存在命令订阅者和位置反馈，再发送一次目标；连续3次新反馈位于
目标±容差内才进入下一步。连接/初次读取最多等待5秒，运动中反馈中断2秒会报错。
`gripper_read` 只读取该夹爪，不发送夹爪目标；程序原有DO2初始化/退出清理仍执行。
驱动没有命令确认消息，位置反馈到位是本程序的完成依据。

左右夹爪可以放进 `parallel_begin` / `parallel_end`，也可以与同侧机械臂并行。
每块内同一侧夹爪最多一条操作（读取也占用该夹爪）。块结束等待全部操作完成。

驱动目前没有停止或夹持力接口。抓住物体后若达不到位置目标，可能超时；
超时/Ctrl+C会结束等待并终止后续流程，但不能撤销已经发送的夹爪命令。
退出不自动松开夹爪；并行取消时该限制同样适用。
