/**
 * @file ik.cpp
 * @brief 输入末端位姿，计算七个关节角。只计算，不会运动。
 */

#include <array>
#include <iomanip>
#include <iostream>

#include "rokae/robot.h"

namespace {

// ======================== 在这里修改参数 ========================
constexpr const char *kRobotIp = "192.168.2.160";  // 右臂
constexpr const char *kLocalIp = "192.168.2.10";

// 末端相对外部参考坐标系: {X, Y, Z, Rx, Ry, Rz}
// XYZ单位: m，RPY单位: rad
const std::array<double, 6> kPoseInput = {
    0.528835, -0.376112, -0.339850,
    1.536745, -0.948863, 1.693023,
};

// true表示计算IK时指定七轴臂角；false表示选择最接近当前关节角的解。
constexpr bool kUseElbow = false;
constexpr double kElbow = 1.570796327;  // rad
// ================================================================

bool checkError(const std::error_code &ec, const char *operation) {
  if (!ec) return true;
  std::cerr << operation << "失败: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

void printPose(const rokae::CartesianPosition &pose) {
  std::cout << "输入末端位姿（相对外部参考坐标系）:" << std::endl
            << "  位置 XYZ: [" << pose.trans[0] << ", " << pose.trans[1]
            << ", " << pose.trans[2] << "] m" << std::endl
            << "  姿态 RPY: [" << pose.rpy[0] << ", " << pose.rpy[1]
            << ", " << pose.rpy[2] << "] rad" << std::endl;
  if (pose.hasElbow) {
    std::cout << "  臂角: " << pose.elbow << " rad" << std::endl;
  }
}

void printJoints(const std::array<double, 7> &joints) {
  std::cout << "IK结果 [rad]: [";
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << joints[i];
  }
  std::cout << "]" << std::endl;

  for (std::size_t i = 0; i < joints.size(); ++i) {
    std::cout << "  J" << i + 1 << ": " << joints[i] << " rad"
              << std::endl;
  }
}

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(9);

  try {
    rokae::ArRobot robot(kRobotIp, kLocalIp);
    std::error_code ec;

    rokae::CartesianPosition target(
        {kPoseInput[0], kPoseInput[1], kPoseInput[2],
         kPoseInput[3], kPoseInput[4], kPoseInput[5]});
    if (kUseElbow) {
      target.elbow = kElbow;
      target.hasElbow = true;
    }

    const auto toolset = robot.toolset(ec);
    if (!checkError(ec, "读取工具工件坐标系")) return 1;

    auto model = robot.model();
    const auto joints = model.calcIk(target, toolset, ec);
    if (!checkError(ec, "计算IK")) return 1;

    printPose(target);
    printJoints(joints);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "IK程序异常: " << e.what() << std::endl;
    return 1;
  }
}
