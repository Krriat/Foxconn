# text_dct运行时话题点（C++）

直接运行text_dct即可订阅取点，无需运行text_points.py，无需人工选点/保存。
实现集中在text_topic_points.hpp/.cpp，主程序只负责配置注册和按名称resolve。
现有手工点、关节运动、并行块和独立分支继续支持。

## conf用法

```conf
[poses]
# 姿态模板：使用你已有的实机示教点；XYZ在话题点解析时不会沿用
# pose2 X Y Z RX RY RZ

[topic_poses]
# 名称 目标臂 话题 预期frame_id 姿态模板 [等待超时 消息最大年龄]
target_left left /vision/part_target base_link pose2 10 5

[steps]
move left target_left 30
# 后续夹爪、DO、lift及其他步骤照常填写
```

上例需要已有pose2。只在实际运动步骤引用话题点时建立订阅，声明但不使用不会等待话题。
`target_left`由你命名，不能与静态点重名；绑定left的点不能用于right动作。
`move`/`movel`/`movej`支持话题点，`moveabsj`仍只接受七轴关节点。
话题点XYZ直接使用上游转换完成后的坐标，单位米；不做额外变换。
RPY沿用指定静态点，单位弧度。当前检测器单位四元数不是估计的抓取姿态，因此不自动使用它。

## 取点时机

输入MOVE后，在初始化任何执行设备及执行任何步骤之前，预先获取本流程引用的全部话题点。
每个点等待从其检查开始后收到的一条新有效消息，取得后固定成六维目标。
任一点无发布者、无有效目标或超时，整个流程报错退出；左右臂、夹爪、升降和DO步骤都不会启动，DO2也不会初始化。
只有全部点取得后才启动设备及顺序/并行流程。多个点依次检查，等待总时长可能是各点超时之和。
本轮流程再次引用同一点名时复用同一个坐标；不会在正在执行运动时追逐视觉更新。
程序重新启动或--loop开始下一轮后重新取点。当前不提供流程内刷新命令或多个点之间的关联快照，
不同点名在前置检查中分别取样；需同一目标多段运动时请复用同一点名。
如果目标已变化而流程需要重新检测，需进入下一轮或启动下一次流程，不能把冻结点当成实时跟踪。

无消息、detected=false、缺字段、非有限坐标、frame不匹配、消息过期都不会产生目标运动。
到等待超时会报告最后一个原因并终止流程；并行分支按现有取消逻辑退出。
等候支持Ctrl+C（最多约50ms轮询间隔，不包括其他设备停止延迟）。
超时默认10秒、最大消息年龄默认5秒，可在声明末尾一起填写，范围(0,120]秒。
订阅深度1、volatile、best_effort，可接收reliable或best_effort发布者的新数据。
消息时间戳必须有效，发布端和本机时钟须同步；不自动使用旧缓存位置兜底。

`--check`只校验声明、模板、目标臂和步骤，不启动ROS、不访问硬件。
动态点的坐标在运行时才能确定，因此静态距离预检查跳过该段；
运动前仍按实际当前位置与解析目标检查max_move_distance_m及原有运行时保护。
不新增可达性、避碰或动态目标追踪规划。

## ROS域

默认沿用启动终端的ROS_DOMAIN_ID。若夹爪/升降在域1、检测在域0：

```bash
source /opt/ros/humble/setup.bash
source /home/niic/aide/hardware/liftmotor/install/setup.bash
export ROS_DOMAIN_ID=1
export TEXT_DCT_POINTS_DOMAIN_ID=0
../build/bin/text_dct --check
../build/bin/text_dct
```

TEXT_DCT_POINTS_DOMAIN_ID只改变取点模块的独立ROS上下文，不改变夹爪/升降所用域。
两者相同则无需设置它；`unset TEXT_DCT_POINTS_DOMAIN_ID`恢复跟随默认域。
取点模块不启动检测器，检测程序须提前运行。

## 扩展接口

运行时自动查询话题发布者的ROS消息类型并选择解析器，conf无需填写解析器。
没有发布者时在取点超时内等待；不支持的类型或同话题多个不同类型会报错。
旧的显式解析器格式仍兼容。当前支持：

| 解析器 | ROS消息类型 | XYZ/有效性 |
| --- | --- | --- |
| part_json | std_msgs/msg/String | 当前part_detection.py的JSON，检查detected、position、frame_id、stamp |
| pose_stamped | geometry_msgs/msg/PoseStamped | pose.position、header；忽略orientation，使用模板RPY |
| point_stamped | geometry_msgs/msg/PointStamped | point、header；使用模板RPY |

已有类型新增话题只增加[topic_poses]声明，无需改主程序。
新增完全不同的消息类型时，在模块adapterForType中注册消息类型映射，在Impl::subscribe中增加订阅，
通过receive把消息转换成Sample（米制XYZ、frame、秒时间戳）；校验、冻结和等待逻辑可以复用。
在CMake为模块添加消息包依赖，重新编译即可，通常无需改text_dct.cpp。
JSON不是任意结构自动映射：part_json专门匹配当前检测器，其它JSON格式需新增适配器。

```bash
cmake --build ../build --target text_dct -j2
```

离线验证（SDK根目录）：

```bash
cmake --build build --target text_topic_points_test -j2
./build/bin/text_topic_points_test
python3 cxl/test_text_dct_offline.py
```

注意：ROS类型String只能表明字符串，不能自动推断任意JSON语义。当前String默认解析现有检测JSON；不同JSON结构需要在模块内增加对应适配规则，conf不暴露解析细节。

## 偏移生成新点

可以在conf增加：

```conf
[offset_poses]
# 新点名 基准点名 DX DY DZ（米）
pose2cs_above pose2cs 0 0 0.05
pose2cs_shift pose2cs 0.01 -0.02 0
pose2cs_higher pose2cs_above 0 0 0.03
```

原pose2cs仍可使用，above为原点Z+0.05m，shift为X+0.01m/Y-0.02m，
higher为原点Z+0.08m。RPY完全沿用基准点，不改变原点。
偏移沿控制器外部参考系坐标轴，不是工具局部轴；Z+只有在该轴实际竖直朝上时才是“正上方”。
支持话题点、静态[poses]点、其它偏移点作为基准，不支持七轴[joints]点作为基准。

在steps中可以这样安排（请按实际流程插入）：

```conf
move left pose2cs_above 30
move left pose2cs 20
# 执行抓取动作
move left pose2cs_above 30
```

前置检查获取话题根点一次，所有依赖它的偏移点基于同一快照计算；
即使steps只使用above，也必须先收到pose2cs。新点不重新订阅、不触发第二次检测。
运行前打印用到的动态偏移点最终六维坐标一次。静态偏移点在--check中直接显示结果。
配置中顺序不限；未定义基准、重名、循环依赖和非法数值会在启动前拒绝。
话题派生点继承目标臂，绑定左臂的点不能用于右臂。
现有实际起终点距离限制保留，新点不保证可达或路径无碰撞。

## 循环运行模式

默认仍执行一次，`--loop`才开启循环：

```bash
../build/bin/text_dct --loop --check
../build/bin/text_dct --loop
# 指定其他配置：../build/bin/text_dct --loop ./other.conf
```

输入一次MOVE确认整次循环运行。每轮开始重新建立取点快照、接收新消息、
重新计算全部偏移点，全部通过才执行整份steps（包括parallel/concurrent分支）。
每轮结束DO2置0并完成本轮设备清理，然后进入下一轮前置检查。
默认不重读conf；运行中修改文件不会改变本次循环，修改流程需停止并重新启动。

下一轮在话题配置的timeout时间内没有有效新点时结束循环，打印完成轮数，退出码0，
不执行该轮任何动作。首轮就没点同样完成0轮后退出。单次模式保持缺点报错行为。
解析错误、错误坐标系/消息类型、非法/过期数据、设备故障和Ctrl+C仍报错退出，
不会因为动作失败就自动重试下一轮。轮次开始前采集的点不会用于新一轮。
循环配置必须在运动步骤中真正使用至少一个话题点（可通过偏移点间接引用），
只含固定点的流程不能开启--loop，以免无结束条件地重复运行。

PoseStamped没有“无物料”字段。按当前约定，**不再发布点**表示下一轮没有目标；
检测器停机/断连也可能造成没有点，程序只报告“无新目标”，不能证明实际料空。
持续发布带新时间戳的同一坐标会被视为一直有料；测试发布器应在模拟料空时Ctrl+C停止。
停止发布不会中断已经开始的本轮动作，而是在下一轮前置检查超时后结束。
