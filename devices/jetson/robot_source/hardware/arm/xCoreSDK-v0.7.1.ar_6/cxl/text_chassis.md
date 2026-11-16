# 按物料类别控制底盘旋转

模块为text_chassis.cpp/.hpp；独立ROS上下文，默认域0。
当前检测器的material_coord是PoseStamped，不含类别。配置：

```conf
[topic_classes]
pose2cs /vision/part_target
```

保留从/vision/material_coord取得位置；从/vision/part_target的JSON按同一时间戳、frame、
XYZ匹配类别（允许JSON坐标四舍五入造成的0.11mm差异）。part1/part2映射1/2，也接受"1"/"2"。
不会把无时间戳的/vision/material_class最新值拼接到另一帧坐标上。
类别在每轮前置检查与坐标一起冻结，转回时使用同一类别。
有位置但匹配不到类别属于前置检查错误，不会当作料空启动动作或继续循环。
未知/缺失类别不能静默当成类别1。

conf步骤：

```conf
# 类别2才执行；类别1跳过，速度rad/s、圈数正逆时针负顺时针
spin_if_class pose2cs 2 0.25 0.2
# ...原有顺序/并行/独立分支...
spin_if_class pose2cs 2 -0.25 0.2
```

格式：spin_if_class 点名 类别 圈数 [角速度]，省略速度默认0.2。
点名也可为由话题派生的偏移点，类别沿用根点。只支持放在parallel/concurrent块外，
确保先等待所有臂动作完成再转底盘。条件不符合时打印Skip，不连接/调用底盘服务。
当前conf已在最后一组“同时异步运动”之前和之后放入+0.25/-0.25两步。

类别2在任何本轮设备动作前检查旋转/停止服务及新状态是否可用、是否已有任务运行。
运动步骤到达后调用/robot/chassis/spin；accepted只是受理，按返回task_id等待
/robot/chassis/spin/status的succeeded，之后等待0.6秒让服务完成补发零速度。
超时/failed/canceled/状态中断/取消会报错并尝试调用/robot/chassis/spin/stop；
明确拒绝的请求不停止服务中其他任务。故障后不自动执行后续转回或下一轮。
对称圈数是地图朝向反馈下的相对转动，不保证精确恢复物理位置/朝向。

启动底盘服务（与底盘同域，当前为0）：

```bash
source /opt/ros/humble/setup.bash
source /home/niic/aide/Exhibition/9_15_tianjing_ws/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run chassis_spin spin_server
```

text_dct终端仍可用域1控制夹爪/升降，视觉域按现有TEXT_DCT_POINTS_DOMAIN_ID：

```bash
source /home/niic/aide/Exhibition/9_15_tianjing_ws/install/setup.bash
export TEXT_DCT_CHASSIS_DOMAIN_ID=0
../build/bin/text_dct --check
../build/bin/text_dct --loop
```

只有上面运行命令会在输入MOVE后执行真实流程。--check不访问设备。
不要同时让导航/遥控发布底盘速度，服务没有跨节点速度仲裁。

测试发布器已支持同帧发布坐标和类别：

```bash
python3 publish_test_point.py --class-id 1
# 或 --class-id 2（随后执行text_dct会按配置真实转动底盘）
```

测试脚本只发布检测信息，但若text_dct已在运行，会触发它的对应动作。
不要与真实检测器同时发布测试目标。独立类别话题、消息格式或命名若改变，需同步适配配置/模块。
