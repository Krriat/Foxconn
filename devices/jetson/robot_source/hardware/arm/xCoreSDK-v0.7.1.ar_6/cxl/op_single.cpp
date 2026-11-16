/**
 * @file op_single.cpp
 * @brief 单臂初始化：非实时指令模式 -> 自动模式 -> 上电 -> 状态确认。
 * 修改 ARM_SELECTION 后重新编译运行。只连接选中的机械臂，不发送运动指令。
 * 运行前停止占用该机械臂的 ROS 驱动、遥操作或其他 SDK 控制程序。
 */

#include <iostream>
#include <string>

#include "rokae/robot.h"

// ===================== 用户配置 =====================
constexpr int ARM_SELECTION = 1;  // 1：左臂；2：右臂
// ====================================================
static_assert(ARM_SELECTION == 1 || ARM_SELECTION == 2,
              "ARM_SELECTION must be 1 (left) or 2 (right)");

namespace {
struct ArmConfig {
  const char *name;
  const char *robot_ip;
  const char *local_ip;
};

bool checkError(const std::error_code &ec, const char *operation) {
  if (!ec) return true;
  std::cerr << operation << "失败: code=" << ec.value()
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
}  // namespace

int main() {
  constexpr ArmConfig left{"左臂", "192.168.1.160", "192.168.1.10"};
  constexpr ArmConfig right{"右臂", "192.168.2.160", "192.168.2.10"};
  const auto &arm = ARM_SELECTION == 1 ? left : right;

  try {
    std::cout << "初始化" << arm.name << "，机器人IP: " << arm.robot_ip
              << "，本机IP: " << arm.local_ip << std::endl;
    rokae::ArRobot robot(arm.robot_ip, arm.local_ip);
    std::error_code ec;

    const auto info = robot.robotInfo(ec);
    if (!checkError(ec, "读取机器人信息")) return 1;
    std::cout << "机型: " << info.type << "，控制器版本: " << info.version
              << std::endl;

    robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
    if (!checkError(ec, "切换非实时指令模式")) return 1;
    std::cout << "非实时指令模式设置成功" << std::endl;

    robot.setOperateMode(rokae::OperateMode::automatic, ec);
    if (!checkError(ec, "切换自动模式")) return 1;
    std::cout << "自动模式设置成功" << std::endl;

    robot.setPowerState(true, ec);
    if (!checkError(ec, "上电")) return 1;
    const auto power = robot.powerState(ec);
    if (!checkError(ec, "读取上电状态")) return 1;
    std::cout << arm.name << "最终状态: " << powerStateName(power) << std::endl;
    if (power != rokae::PowerState::on) {
      std::cerr << "初始化未完成：机械臂没有进入上电状态" << std::endl;
      return 1;
    }
    std::cout << arm.name << "初始化完成；未发送运动指令。" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << arm.name << "连接或初始化异常: " << error.what() << std::endl;
    return 1;
  }
}
