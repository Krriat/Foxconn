/**
 * @file op.cpp
 * @brief Initialize both AR arms after controller startup.
 *
 * This program selects non-realtime command mode, switches to automatic mode,
 * and powers on each arm. It does not send any motion command.
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "rokae/robot.h"

namespace {

struct ArmConfig {
  const char *name;
  const char *robot_ip;
  const char *local_ip;
};

bool checkError(const std::error_code &ec, const std::string &arm,
                const char *operation) {
  if (!ec) {
    return true;
  }
  std::cerr << arm << " " << operation << "失败: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

const char *powerStateName(rokae::PowerState state) {
  switch (state) {
    case rokae::PowerState::on: return "已上电";
    case rokae::PowerState::off: return "已下电";
    case rokae::PowerState::estop: return "急停按下";
    case rokae::PowerState::gstop: return "安全门打开";
    default: return "未知";
  }
}

bool initializeArm(const ArmConfig &config) {
  const std::string arm(config.name);
  try {
    std::cout << "\n========== 初始化" << arm << " ==========" << std::endl
              << "机器人IP: " << config.robot_ip
              << "，本机IP: " << config.local_ip << std::endl;

    std::unique_ptr<rokae::ArRobot> robot;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(90);
    while (!robot) {
      try {
        robot = std::make_unique<rokae::ArRobot>(config.robot_ip,
                                                  config.local_ip);
      } catch (const std::exception &e) {
        if (std::string(e.what()).find("network:") != 0 ||
            std::chrono::steady_clock::now() >= deadline) {
          throw;
        }
        std::cerr << arm << "网络连接暂不可用，2 秒后重试" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }
    std::error_code ec;

    const auto info = robot->robotInfo(ec);
    if (!checkError(ec, arm, "读取机器人信息")) return false;
    std::cout << "机型: " << info.type << "，控制器版本: " << info.version
              << std::endl;

    robot->setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
    if (!checkError(ec, arm, "切换非实时指令模式")) return false;
    std::cout << "非实时指令模式设置成功" << std::endl;

    robot->setOperateMode(rokae::OperateMode::automatic, ec);
    if (!checkError(ec, arm, "切换自动模式")) return false;
    std::cout << "自动模式设置成功" << std::endl;

    robot->setPowerState(true, ec);
    if (!checkError(ec, arm, "上电")) return false;

    const auto power_state = robot->powerState(ec);
    if (!checkError(ec, arm, "读取上电状态")) return false;
    std::cout << "最终状态: " << powerStateName(power_state) << std::endl;

    if (power_state != rokae::PowerState::on) {
      std::cerr << arm << "初始化未完成：机械臂没有进入上电状态" << std::endl;
      return false;
    }

    std::cout << arm << "初始化完成" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << arm << "连接或初始化异常: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace

int main() {
  const ArmConfig arms[] = {
      {"左臂", "192.168.1.160", "192.168.1.10"},
      {"右臂", "192.168.2.160", "192.168.2.10"},
  };

  int failed = 0;
  for (const auto &arm : arms) {
    if (!initializeArm(arm)) {
      ++failed;
    }
  }

  if (failed != 0) {
    std::cerr << "\n初始化结束，失败机械臂数量: " << failed << std::endl;
    return 1;
  }

  std::cout << "\n左右臂均已完成初始化；程序未发送运动指令。" << std::endl;
  return 0;
}
