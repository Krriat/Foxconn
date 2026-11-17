# USR-M050 C / C++ SDK

本目录提供机器人端可直接使用的 Linux C99 SDK、C++11 封装、动态库、静态库和命令行工具。
SDK 使用系统 socket / pthread，不依赖 libmodbus 或其他第三方运行库。支持 Jetson ARM64；其他 Linux 平台可从源码编译。

ROS 2 Humble 节点见 [m050_ros/README.md](m050_ros/README.md)，提供 IO 状态话题、实时查询与单路 DO 控制服务。

默认连接：`192.168.1.7:20108`（以太网），Modbus Unit ID：`1`，网络操作超时：`3000 ms`。

| 接口 | 能力 | 协议 |
| --- | --- | --- |
| DO1、DO2 | 两路独立闭合 / 断开，查询状态 | FC01 读、FC05 写，地址 0 / 1 |
| DI1、DI2 | 读取外部输入有效状态 | FC02 读，地址 0 / 1 |

DO 的 `1 / true / on` 表示闭合，`0 / false / off` 表示断开。DI 是物理输入，只能读取；输入信号由外部接线决定。
DO 读回值是设备报告的逻辑状态，实际触点导通需要现场测量确认。

## 立即测试

已在本机编译，以下命令可直接执行。直接运行 `m050ctl` 默认只查询状态。

```bash
cd /home/jetson/AIDE_robot_main/hardware/USR-M050

# 查询两路 DO 和两路 DI
./build/m050ctl status

# 单独查询输出或输入
./build/m050ctl read-do
./build/m050ctl read-di
```

下面每条写命令均独立执行，根据现场需要选择；执行成功会打印两路 DO 的读回状态。

```bash
./build/m050ctl set-do 1 on     # DO1 闭合
./build/m050ctl set-do 1 off    # DO1 断开
./build/m050ctl set-do 2 on     # DO2 闭合
./build/m050ctl set-do 2 off    # DO2 断开
```

修改连接参数时，选项放在命令前：

```bash
./build/m050ctl --host 192.168.1.7 --port 20108 --unit 1 --timeout 3000 status
./build/m050ctl --source 192.168.1.10 status
./build/m050ctl --help
```

一般无需指定 `--source`，系统会按路由选择网卡。只接受数字 IPv4 地址。
命令退出码：`0` 成功、`1` 通信 / 协议 / 读回错误、`2` 参数错误。

## 编译和目录

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

构建需要 CMake、C/C++ 编译器。自动测试额外需要 Python 3 标准库，测试仅连接 `127.0.0.1` 模拟器，不接触实机。

| 路径 | 内容 |
| --- | --- |
| `include/m050/m050.h` | C API |
| `include/m050/m050.hpp` | C++ RAII 封装 |
| `src/m050.c` | Modbus TCP 和 socket 实现 |
| `build/libm050.so` | 本机动态库，SONAME 为 `libm050.so.1` |
| `build/libm050.a` | 本机静态库 |
| `build/m050ctl` | 状态读取 / 单路控制工具 |
| `examples/read_io.c`、`examples/read_io.cpp` | 默认只读的完整示例 |
| `tests/test_protocol.py` | 本地模拟设备测试 |

## C++ 调用

```cpp
#include <m050/m050.hpp>
#include <iostream>

int main() {
    try {
        m050::Client io("192.168.1.7", 20108, 1);
        const auto state = io.read_io();
        std::cout << "DI1=" << state.di.ch1 << " DI2=" << state.di.ch2 << '\n';

        // 以下为独立业务调用示例，按所需动作调用。
        io.set_do(1, true);    // DO1 闭合；返回两路 DO 的读回状态
        io.set_do(1, false);   // DO1 断开
        io.set_do(2, true);    // DO2 闭合
        io.set_do(2, false);   // DO2 断开
    } catch (const m050::Error &error) {
        std::cerr << error.what() << '\n';
        if (error.write_may_have_applied()) {
            // 输出可能已经动作。恢复通信后先读状态，再由业务决定下一步。
        }
        return 1;
    }
}
```

单独读取：`io.read_do()`、`io.read_di()`，返回 `m050::Channels`，成员为 `ch1`、`ch2`。
可通过构造函数第 4 个参数设置毫秒超时，第 5 个参数设置源 IP。

编译现有 C++ 只读示例并静态链接：

```bash
g++ -std=c++11 -Iinclude examples/read_io.cpp build/libm050.a -pthread -o build/my_m050_cpp
./build/my_m050_cpp
```

## C 调用

```c
#include <m050/m050.h>
#include <stdio.h>

int main(void) {
    m050_config config;
    m050_client *io = NULL;
    m050_error error;
    m050_channels state;
    int rc;

    m050_config_default(&config);
    config.host = "192.168.1.7";
    rc = m050_create(&config, &io, &error);
    if (rc == M050_OK) rc = m050_read_do(io, &state, &error);
    if (rc == M050_OK) {
        printf("DO1=%u DO2=%u\n", (unsigned)state.ch1, (unsigned)state.ch2);
        // 按业务需要写入，例如 DO1 闭合：
        // rc = m050_set_do(io, 1, 1, &state, &error);
        // 每次调用后检查 rc；失败时检查 error.write_may_have_applied。
    }
    if (rc != M050_OK) fprintf(stderr, "%s\n", error.message);
    m050_destroy(io);
    return rc == M050_OK ? 0 : 1;
}
```

`m050_read_di()` 读取两路 DI；`m050_read_io()` 同时返回 DO 和 DI 两组字段。
`m050_set_do(io, channel, closed, verified, error)` 的通道只能为 `1 / 2`，状态只能为 `0 / 1`。
`verified` 可为 `NULL`；即使不接收返回状态，也会执行读回校验。`error` 可为 `NULL`。
函数返回 `M050_OK` 或负错误码，失败时不修改状态输出，详细错误通过每次调用独立的 `m050_error` 返回。

```bash
gcc -std=c99 -Iinclude examples/read_io.c build/libm050.a -pthread -o build/my_m050_c
./build/my_m050_c
```

## 接入现有 CMake 工程

直接引用源码目录：

```cmake
add_subdirectory(/home/jetson/AIDE_robot_main/hardware/USR-M050 m050-sdk)
target_link_libraries(your_cpp_target PRIVATE m050::m050_cpp)
# C 动态链接：m050::m050；C / C++ 静态链接：m050::m050_static
# C++ 静态链接时，为业务目标设置 C++11 或更高标准。
```

也可以安装到本目录下的 `install/`，不需要管理员权限：

```bash
cmake --install build --prefix "$PWD/install"
./install/bin/m050ctl status
```

安装后的库支持：

```cmake
find_package(m050 1 CONFIG REQUIRED)
target_link_libraries(your_cpp_target PRIVATE m050::m050_cpp)
```

配置业务工程时使用 `-DCMAKE_PREFIX_PATH=/home/jetson/AIDE_robot_main/hardware/USR-M050/install`。

## 通信行为

- 默认每次 API 调用建立短连接，完成后关闭。创建 / 析构客户端不会改变输出。
- C API `m050_set_connection_reuse(client, 1, &error)` 可启用连接复用；设置本身不连接设备、不写输出。成功调用保留连接，失败关闭连接且不重发请求，下次调用重新连接。ACT 吸盘 worker 与硬件状态发布器已启用；其跨进程锁仍串行处理设备请求。
- M050 最多允许两个 TCP 客户端。上述两个模块运行并持有连接时，通过现有 ROS 状态话题观察反馈，避免再启动第三个设备客户端。
- 单路写入先校验 FC05 回显，再在同一连接内用 FC01 读取并核对目标通道；不自动写另一通道。
- 一次调用的连接、收发和读回共用一个单调时钟超时；同一客户端等待线程锁的时间另计。
- 不自动重试。失败且 `write_may_have_applied=1` 表示写指令可能已经执行，不能当成“断开”或直接认定写入失败。
- 同一客户端的并发调用通过互斥锁串行处理；各次调用使用独立的错误结构。销毁前需要结束所有使用该客户端的调用。
- `read_io()` 在同一连接中先读 DO 再读 DI，两次读取不是同一时刻的原子快照。
- 设备输出由设备维持，关闭程序不发送断开指令。启动时先读取；通信失败时由业务层将状态视为未知。
- 在业务工作线程调用，避免网络等待阻塞机器人运动控制循环。由一个业务模块统一管理输出，避免多程序相互覆盖。

## 本机验证记录

2026-11-13（Asia/Shanghai）有线切换：

- M050 网口固定为 `192.168.1.7/24`，网关为 `192.168.1.10`，网络选择为“仅以太网”。Jetson 自带网口 `enP8p1s0` 保留已有的 `192.168.1.10/24`；通信不依赖两端 Wi-Fi 所属网段。
- Jetson 的机械臂地址、默认上网路由、行为树、运动目标和 ACT 动作参数保持原样。SDK、ROS 启动、采集脚本与硬件状态发布器的 M050 默认目标已改为有线地址。
- 实机只读压力测试 600 次全部成功；经授权两路各执行一次开 / 关，四次读回均成功，结束时两路均断开。TCP 连接复用已通过本地模拟器的连接恢复、失败不重发和读回校验测试。
- 实机 Modbus 写入及读回仍可能超过 ACT 20 Hz 的 50 ms 帧周期；本次未运行完整 ACT 实机动作，不能据此保证推理期间零漏执行。反馈发布周期仍为原有的 10 Hz。

2026-11-02（Asia/Shanghai）：

- 网络出口：`wlP1p1s0`，机器人源 IP：`172.16.12.95/23`，目标：`172.16.13.167:20108`。
- 实机 FC01：`DO1=0 DO2=0`；FC02：`DI1=0 DI2=0`。
- 已通过命令行、C 示例、C++ 示例读取实机；本轮未向实机发送 FC05 通断命令。
- 模拟器覆盖两路 DO 开 / 关、另一通道状态保持、写后验证、四种输入 / 输出位组合、TCP 分包、错误帧、Modbus 异常、连接失败、超时、部分读取失败、线程串行及 C 静态链接。
- 机器人现有行为树、动作流程和其他设备配置保持原样。现场继电器触点与外接输入变化由操作者后续测试。

参考：[厂家产品资料](https://www.usr.cn/Product/484.html)、[厂家说明书入口](https://www.pusr.com/support/download/User-Manual-USR-M050-User-Manual.html)。
本 SDK 使用的 FC01 / FC02 地址已通过当前设备只读实测，FC05 对照用户提供的既有控制代码实现并以本地模拟器验证。
