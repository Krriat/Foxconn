/**
 * @file pos.cpp
 * @brief Read joint and Cartesian positions from both AR arms.
 *
 * This program is read-only. It does not change mode, power on, or move.
 */

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

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

void printCartesian(const std::array<double, 6> &pose) {
  std::cout << "  位置 XYZ: [" << pose[0] << ", "
            << pose[1] << ", " << pose[2]
            << "] m" << std::endl
            << "  姿态 RPY: [" << pose[3] << ", " << pose[4] << ", "
            << pose[5] << "] rad" << std::endl;
}

bool readArm(const ArmConfig &config) {
  const std::string arm(config.name);
  try {
    std::cout << "\n========== " << arm << " ==========" << std::endl
              << "机器人IP: " << config.robot_ip
              << "，本机IP: " << config.local_ip << std::endl;

    rokae::ArRobot robot(config.robot_ip, config.local_ip);
    std::error_code ec;

    const auto joints = robot.jointPos(ec);
    if (!checkError(ec, arm, "读取关节角度")) return false;

    std::cout << "关节角度:" << std::endl;
    for (std::size_t i = 0; i < joints.size(); ++i) {
      std::cout << "  J" << i + 1 << ": " << joints[i] << " rad"
                << std::endl;
    }

    const auto tcp_pose = robot.posture(rokae::CoordinateType::endInRef, ec);
    if (!checkError(ec, arm, "读取末端笛卡尔位姿")) return false;
    std::cout << "旋转后的TCP（末端相对外部参考坐标系）:" << std::endl;
    printCartesian(tcp_pose);

    ec.clear();
    const auto raw_flange =
        robot.posture(rokae::CoordinateType::flangeInBase, ec);
    if (!checkError(ec, arm, "读取原始法兰位姿")) return false;
    std::cout << "原始法兰位姿（法兰相对机器人基坐标系）:" << std::endl;
    printCartesian(raw_flange);

    ec.clear();
    const auto cart_pose =
        robot.cartPosture(rokae::CoordinateType::endInRef, ec);
    if (!checkError(ec, arm, "读取完整笛卡尔位姿")) return false;

    std::cout << "  臂角: " << cart_pose.elbow << " rad" << std::endl
              << "  臂角有效: " << (cart_pose.hasElbow ? "是" : "否")
              << std::endl
              << "  构型: [";
    for (std::size_t i = 0; i < cart_pose.confData.size(); ++i) {
      if (i != 0) std::cout << ", ";
      std::cout << cart_pose.confData[i];
    }
    std::cout << "]" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << arm << "连接或读取异常: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(6);

  const ArmConfig arms[] = {
      {"左臂", "192.168.1.160", "192.168.1.10"},
      {"右臂", "192.168.2.160", "192.168.2.10"},
  };

  int failed = 0;
  for (const auto &arm : arms) {
    if (!readArm(arm)) {
      ++failed;
    }
  }

  if (failed != 0) {
    std::cerr << "\n读取结束，失败机械臂数量: " << failed << std::endl;
    return 1;
  }
  return 0;
}
