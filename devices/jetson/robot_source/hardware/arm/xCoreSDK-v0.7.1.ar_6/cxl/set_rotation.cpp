/**
 * @file set_rotation.cpp
 * @brief Set the base-frame transformations for the left and right AR arms.
 *
 * setBaseFrame() stores the values in the controller. The controller must be
 * restarted before the new transformations take effect.
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
  std::array<double, 6> base_frame;
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

void printFrame(const char *label, const std::array<double, 6> &frame) {
  std::cout << label << " [";
  for (std::size_t i = 0; i < frame.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << frame[i];
  }
  std::cout << "]" << std::endl;
}

bool setArmBaseFrame(const ArmConfig &config) {
  const std::string arm(config.name);
  try {
    std::cout << "\n========== 设置" << arm << " ==========" << std::endl
              << "机器人IP: " << config.robot_ip
              << "，本机IP: " << config.local_ip << std::endl;
    printFrame("目标baseFrame:", config.base_frame);

    rokae::ArRobot robot(config.robot_ip, config.local_ip);
    std::error_code ec;

    const rokae::Frame frame(config.base_frame);
    robot.setBaseFrame(frame, ec);
    if (!checkError(ec, arm, "设置baseFrame")) {
      return false;
    }

    const auto stored_frame = robot.baseFrame(ec);
    if (!checkError(ec, arm, "读取baseFrame")) {
      return false;
    }

    printFrame("控制器已保存:", stored_frame);
    std::cout << arm << "设置成功，重启该控制器后生效。" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << arm << "连接或设置异常: " << e.what() << std::endl;
    return false;
  }
}

}  // namespace

int main() {
  std::cout << std::fixed << std::setprecision(7);

  const ArmConfig arms[] = {
      {"左臂",
       "192.168.1.160",
       "192.168.1.10",
      //  {0.0, 0.105, 0.0, -1.570796327,-0.134728006, 0.0}},
      {0.0, 0.105, 0.0, -1.570796327, 0.1308996939, 0.0}},
      {"右臂",
       "192.168.2.160",
       "192.168.2.10",
       {0.0, -0.105, 0.00, 1.570796327, 0.1308996939, 0}},
      // {0.0, -0.105, 0.0, -1.570796327, -0.128343267, 3.14159}},
  };

  int failed = 0;
  for (const auto &arm : arms) {
    if (!setArmBaseFrame(arm)) {
      ++failed;
    }
  }

  if (failed != 0) {
    std::cerr << "\n设置结束，失败机械臂数量: " << failed << std::endl;
    return 1;
  }

  std::cout << "\n左右臂baseFrame均已保存。请分别重启两个控制器使设置生效。"
            << std::endl;
  return 0;
}
