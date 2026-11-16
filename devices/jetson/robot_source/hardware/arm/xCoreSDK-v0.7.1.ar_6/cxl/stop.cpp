#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "rokae/robot.h"

using namespace rokae;

namespace {

struct ArmConfig {
  const char *name;
  const char *robot_ip;
  const char *local_ip;
};

bool stopArm(const ArmConfig &config) {
  try {
    std::cout << "\n========== 停止" << config.name << " =========="
              << std::endl;
    ArRobot robot(config.robot_ip, config.local_ip);
    error_code ec;

    robot.stop(ec);
    std::cout << "stop ec: " << ec.value();
    if (ec) std::cout << ", " << ec.message();
    std::cout << std::endl;

    // 等待规划停止完成，最长5秒。
    for (int i = 0; i < 100; ++i) {
      const auto state = robot.operationState(ec);
      if (ec || state == OperationState::idle ||
          state == OperationState::unknown) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 清空未执行轨迹，确保不能通过moveStart继续。
    robot.moveReset(ec);
    std::cout << "moveReset ec: " << ec.value();
    if (ec) std::cout << ", " << ec.message();
    std::cout << std::endl;

    const auto state = robot.operationState(ec);
    std::cout << "operationState: " << static_cast<int>(state) << std::endl;
    return !ec;
  } catch (const std::exception &e) {
    std::cerr << config.name << "停止失败: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace

int main() {
  const ArmConfig arms[] = {
      {"右臂", "192.168.2.160", "192.168.2.10"},
      {"左臂", "192.168.1.160", "192.168.1.10"},
  };

  int failed = 0;
  for (const auto &arm : arms) {
    if (!stopArm(arm)) ++failed;
  }

  if (failed != 0) {
    std::cerr << "\n停止结束，失败机械臂数量: " << failed << std::endl;
    return 1;
  }

  std::cout << "\n左右臂均已停止并清空未执行轨迹。" << std::endl;
  return 0;
}
