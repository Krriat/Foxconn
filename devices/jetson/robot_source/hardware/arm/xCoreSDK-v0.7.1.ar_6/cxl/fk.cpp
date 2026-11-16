/**
 * @file fk.cpp
 * @brief 输入七个关节角，计算末端位姿。只计算，不会运动。
 */

#include <array>
#include <iomanip>
#include <iostream>

#include "rokae/robot.h"

namespace {

// ======================== 在这里修改参数 ========================
constexpr const char *kRobotIp = "192.168.2.160";  // 右臂
constexpr const char *kLocalIp = "192.168.2.10";

// 七个关节角，单位: rad
const std::array<double, 7> kJointInput = {
    1.7, -1.0, -2.2, 0.5, 0.5, 0.5, 0.0,
};
// ================================================================

bool checkError(const std::error_code &ec, const char *operation) {
  if (!ec) return true;
  std::cerr << operation << "失败: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

void printJoints(const std::array<double, 7> &joints) {
  std::cout << "输入关节角 [rad]: [";
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << joints[i];
  }
  std::cout << "]" << std::endl;
}

void printPose(const rokae::CartesianPosition &pose) {
  std::cout << "FK结果（末端相对外部参考坐标系）:" << std::endl
            << "  位置 XYZ: [" << pose.trans[0] << ", " << pose.trans[1]
            << ", " << pose.trans[2] << "] m" << std::endl
            << "  姿态 RPY: [" << pose.rpy[0] << ", " << pose.rpy[1]
            << ", " << pose.rpy[2] << "] rad" << std::endl
            << "  臂角: " << pose.elbow << " rad" << std::endl
            << "  臂角有效: " << (pose.hasElbow ? "是" : "否") << std::endl
            << "  构型: [";
  for (std::size_t i = 0; i < pose.confData.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << pose.confData[i];
  }
  std::cout << "]" << std::endl;
}

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(9);

  try {
    rokae::ArRobot robot(kRobotIp, kLocalIp);
    std::error_code ec;

    const auto toolset = robot.toolset(ec);
    if (!checkError(ec, "读取工具工件坐标系")) return 1;

    auto model = robot.model();
    const auto pose = model.calcFk(kJointInput, toolset, ec);
    if (!checkError(ec, "计算FK")) return 1;

    printJoints(kJointInput);
    printPose(pose);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FK程序异常: " << e.what() << std::endl;
    return 1;
  }
}
