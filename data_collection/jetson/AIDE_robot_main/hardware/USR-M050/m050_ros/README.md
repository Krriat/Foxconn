# USR-M050 ROS 2 节点

ROS 2 Humble C++ 节点，复用上层目录的 C SDK 并静态链接，无需另外安装 SDK 动态库。
构建时须保留整个 `USR-M050` 目录。现有 SDK、命令行工具继续独立使用。

## 编译与启动

```bash
source /opt/ros/humble/setup.bash
cd ~/AIDE_robot_main/hardware/USR-M050
colcon build --base-paths m050_ros --packages-select m050_ros \
  --build-base build_ros --install-base install_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install_ros/local_setup.bash
ros2 launch m050_ros m050.launch.py
```

每个调用端终端也须加载 ROS 和 `install_ros/local_setup.bash`，使用与节点相同的 ROS_DOMAIN_ID。
不需要管理员权限，不改现有自启。直接启动也可以：`ros2 run m050_ros m050_node`。

启动后默认每 500 ms 查询 DO1/DO2、DI1/DI2。启动、退出、网络故障均不写输出。
输出只在收到 `set_do` 服务请求时改变；每个请求只写指定的一路，写后读取两路 DO 并校验。
所有网络操作在该独立节点单线程串行执行，一次 SDK 调用最多等待 `timeout_ms`；
忙碌期间其他服务和轮询需等待。两次完整查询之间的状态变化可能无法捕获。

## 接口

默认节点 `/m050`。数组下标 0 对应通道 1，下标 1 对应通道 2。

| 接口 | 类型 | 内容 |
| --- | --- | --- |
| `/m050/state` | `m050_ros/msg/IOState` | 每次完整查询后的两路 DO、两路 DI、有效标记、完成时间、错误码与说明 |
| `/m050/get_io` | `m050_ros/srv/GetIO` | 立即查询完整 IO，结果也发布到 state |
| `/m050/set_do` | `m050_ros/srv/SetDO` | 单路 DO 通断，响应包含两路 DO 的校验读回值 |

```bash
# 读取，不改变继电器
ros2 topic echo /m050/state
ros2 service call /m050/get_io m050_ros/srv/GetIO '{}'
```

以下命令各自控制一路，按业务需要执行：

```bash
ros2 service call /m050/set_do m050_ros/srv/SetDO '{channel: 1, closed: true}'
ros2 service call /m050/set_do m050_ros/srv/SetDO '{channel: 1, closed: false}'
ros2 service call /m050/set_do m050_ros/srv/SetDO '{channel: 2, closed: true}'
ros2 service call /m050/set_do m050_ros/srv/SetDO '{channel: 2, closed: false}'
```

`closed: true` 表示闭合，`false` 表示断开。DI 由物理输入决定，仅支持读取。

查询失败时 `state.valid=false`，数组填零但不表示设备实际断开，`error_code/message` 描述原因。
查询成功才可使用数组；DO 和 DI 顺序读取，非同一时刻的原子快照。话题使用 reliable、volatile、depth=1，
`stamp` 是该次查询完成时间，业务端须按所需时效判断数据是否过期。

`set_do` 的 `success=true` 表示 FC05 回显与 DO 读回校验均通过，`do_state_valid=true` 才能使用响应数组。
`success=false` 且 `write_may_have_applied=true` 表示输出可能已动作，不能推断为断开；
先调用 `get_io` 获取状态，再由业务决定下一步。节点不重试写指令。
`set_do` 不附加读取 DI；`state` 在下一次轮询或 `get_io` 后更新，与该请求的响应分开判断时效。
错误码沿用 [SDK 头文件](../include/m050/m050.h)，例如 `-1` 参数错误、`-4` 连接失败、
`-5` 超时、`-7` 协议错误、`-8` Modbus 异常、`-9` 读回不匹配。

## 参数

参数在启动时设置，运行中只读；修改后重启节点。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `host` | `192.168.1.7` | 设备有线数字 IPv4 地址 |
| `port` | `20108` | TCP 端口，1～65535 |
| `unit_id` | `1` | Modbus 站号，1～247 |
| `timeout_ms` | `3000` | 一次 SDK 网络操作总超时，正整数毫秒 |
| `source_ip` | 空字符串 | 可选源 IPv4；空值由系统选择 |
| `poll_interval_ms` | `500` | 轮询周期；0 关闭自动轮询，保留查询与控制服务 |

```bash
ros2 launch m050_ros m050.launch.py \
  host:=192.168.1.7 port:=20108 unit_id:=1 timeout_ms:=1000 poll_interval_ms:=500

# 只按服务请求访问设备
ros2 run m050_ros m050_node --ros-args -p poll_interval_ms:=0
```

启动多个模块可用 `namespace:=io_a`，接口变为 `/io_a/m050/...`；
直接运行可用 ROS 参数 `-r __ns:=/io_a`，节点及接口也支持标准 ROS 重映射。

## 验证

```bash
colcon test --base-paths m050_ros --build-base build_ros --install-base install_ros --packages-select m050_ros \
  --event-handlers console_direct+
colcon test-result --test-result-base build_ros --verbose
```

集成测试使用 `127.0.0.1` 上的 Modbus 模拟器和隔离 ROS 域，覆盖服务、话题、两路控制与互不影响、
失败状态、读回校验、写结果不确定、故障恢复、参数校验及启动/退出不写输出。
实机通断和触点测量需在实际接线条件下验证；封装测试不向实机写入。
