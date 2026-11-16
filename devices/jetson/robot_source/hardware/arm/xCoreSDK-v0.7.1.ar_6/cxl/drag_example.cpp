/**
 * @file drag_example.cpp
 * @brief AR 7-axis robot joint-space free-drag example.
 */

#include <array>
#include <iostream>
#include <string>

#include "rokae/robot.h"
#include "rokae/utility.h"

namespace {

bool checkError(const std::error_code &ec, const char *operation) {
  if (!ec) {
    return true;
  }
  std::cerr << operation << "失败: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

void printJointPosition(rokae::ArRobot &robot) {
  std::error_code ec;
  const auto joints = robot.jointPos(ec);
  if (!checkError(ec, "读取关节角度")) {
    return;
  }

  std::cout << "当前关节角度:" << std::endl;
  for (std::size_t i = 0; i < joints.size(); ++i) {
    std::cout << "  第" << i + 1 << "轴: "
              << rokae::Utils::radToDeg(joints[i]) << " deg" << std::endl;
  }
}

void printToolLoad(rokae::ArRobot &robot) {
  std::error_code ec;
  const auto toolset = robot.toolset(ec);
  if (!checkError(ec, "读取当前工具负载")) {
    return;
  }

  std::cout << "当前工具负载:" << std::endl
            << "  质量: " << toolset.load.mass << " kg" << std::endl
            << "  质心: [" << toolset.load.cog[0] << ", "
            << toolset.load.cog[1] << ", " << toolset.load.cog[2]
            << "] m" << std::endl
            << "  惯量: [" << toolset.load.inertia[0] << ", "
            << toolset.load.inertia[1] << ", "
            << toolset.load.inertia[2] << "] kg*m^2" << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string robot_ip = argc > 1 ? argv[1] : "192.168.2.160";
  const std::string local_ip = argc > 2 ? argv[2] : "192.168.2.10";

  try {
    std::cout << "连接机器人 " << robot_ip << "，本机地址 " << local_ip
              << std::endl;
    rokae::ArRobot robot(robot_ip, local_ip);
    std::error_code ec;

    // 上一次程序若未正常按回车退出，控制器可能仍保持拖拽状态。
    // 此时重新运行本程序，先关闭拖拽并直接退出。
    const auto initial_state = robot.operationState(ec);
    if (!checkError(ec, "读取机器人状态")) {
      return 1;
    }
    if (initial_state == rokae::OperationState::drag) {
      robot.disableDrag(ec);
      if (!checkError(ec, "关闭遗留拖拽")) {
        return 1;
      }
      std::cout << "检测到拖拽已经开启，现已关闭。" << std::endl;
      return 0;
    }

    printJointPosition(robot);
    printToolLoad(robot);

    // Drag mode requires the robot to be powered off and in manual mode.
    robot.setPowerState(false, ec);
    if (!checkError(ec, "机械臂下电")) {
      return 1;
    }

    robot.setOperateMode(rokae::OperateMode::manual, ec);
    if (!checkError(ec, "切换手动模式")) {
      return 1;
    }

    std::cout << "请先扶稳机械臂，按回车开启轴空间自由拖拽。" << std::endl;
    std::cin.get();

    // true: drag directly without holding the end drag button.
    robot.enableDrag(rokae::DragParameter::jointSpace,
                     rokae::DragParameter::freely, ec, true);
    if (!checkError(ec, "开启拖拽")) {
      return 1;
    }

    std::cout << "拖拽已开启。请将第6、7轴缓慢移回 -50 deg ~ 50 deg 内；"
                 "完成后按回车关闭拖拽。"
              << std::endl;
    std::cin.get();

    robot.disableDrag(ec);
    if (!checkError(ec, "关闭拖拽")) {
      return 1;
    }

    std::cout << "拖拽已关闭。" << std::endl;
    printJointPosition(robot);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "连接或执行失败: " << e.what() << std::endl;
    return 1;
  }
}

// 左
// ./build/bin/drag_example 192.168.1.160 192.168.1.10
// 右
// ./build/bin/drag_example 192.168.2.160 192.168.2.10
