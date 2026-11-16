/**
 * @file tcp_force_monitor.cpp
 * @brief Read the current TCP wrench from an idle AR arm without commanding it.
 *
 * This utility does not change the operate mode or power state and does not
 * issue a motion or force-control command. Stop other SDK/ROS clients before
 * running it; the program exits if the robot is not idle.
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "rokae/robot.h"

namespace {

constexpr std::chrono::milliseconds kSamplePeriod{200};
std::atomic_bool stop_requested{false};

struct ArmConfig {
  const char *name;
  const char *robot_ip;
  const char *local_ip;
};

void handleSignal(int) {
  stop_requested.store(true, std::memory_order_relaxed);
}

double norm(const std::array<double, 3> &values) {
  return std::sqrt(values[0] * values[0] + values[1] * values[1] +
                   values[2] * values[2]);
}

bool checkError(const std::error_code &error, const char *operation) {
  if (!error) return true;
  std::cerr << operation << "失败: code=" << error.value()
            << ", message=" << error.message() << std::endl;
  return false;
}

const ArmConfig *selectArm(const std::string &selection) {
  static constexpr ArmConfig left{
      "左臂", "192.168.1.160", "192.168.1.10"};
  static constexpr ArmConfig right{
      "右臂", "192.168.2.160", "192.168.2.10"};
  if (selection == "left") return &left;
  if (selection == "right") return &right;
  return nullptr;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "用法: " << argv[0] << " left|right" << std::endl;
    return 2;
  }

  const ArmConfig *const arm = selectArm(argv[1]);
  if (arm == nullptr) {
    std::cerr << "机械臂必须是 left 或 right" << std::endl;
    return 2;
  }

  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);
  std::cout << std::fixed << std::setprecision(3);

  try {
    std::cout << "连接" << arm->name << ": robot=" << arm->robot_ip
              << ", local=" << arm->local_ip << std::endl;
    rokae::ArRobot robot(arm->robot_ip, arm->local_ip);
    std::error_code error;

    const auto info = robot.robotInfo(error);
    if (!checkError(error, "读取机器人信息")) return 1;
    std::cout << "机型: " << info.type << std::endl
              << "TCP坐标系当前力/力矩（Ctrl+C 退出）" << std::endl;

    auto force_control = robot.forceControl();
    while (!stop_requested.load(std::memory_order_relaxed)) {
      error.clear();
      const auto state = robot.operationState(error);
      if (!checkError(error, "读取运行状态")) return 1;
      if (state != rokae::OperationState::idle) {
        std::cerr << "机器人不再处于 idle，停止只读监视" << std::endl;
        return 1;
      }

      std::array<double, 7> joint_torque{};
      std::array<double, 7> external_torque{};
      std::array<double, 3> tcp_torque{};
      std::array<double, 3> tcp_force{};
      error.clear();
      force_control.getEndTorque(
          rokae::FrameType::tool, joint_torque, external_torque,
          tcp_torque, tcp_force, error);
      if (!checkError(error, "读取TCP力")) return 1;

      std::cout << "F[N]=[" << tcp_force[0] << ", " << tcp_force[1]
                << ", " << tcp_force[2] << "] |F|=" << norm(tcp_force)
                << "  M[Nm]=[" << tcp_torque[0] << ", "
                << tcp_torque[1] << ", " << tcp_torque[2]
                << "] |M|=" << norm(tcp_torque) << std::endl;
      std::this_thread::sleep_for(kSamplePeriod);
    }
  } catch (const std::exception &error) {
    std::cerr << "连接或读取异常: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}
