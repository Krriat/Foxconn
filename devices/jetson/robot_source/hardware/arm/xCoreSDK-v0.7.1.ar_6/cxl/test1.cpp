#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include "rokae/robot.h"

using namespace rokae;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void signalHandler(int) {
  g_stop_requested = 1;
}

bool checkEc(const error_code &ec, const char *operation) {
  if (!ec) return true;

  std::cerr << operation << " ec: " << ec.value() << ", " << ec.message()
            << std::endl;
  return false;
}

bool waitRobot(BaseRobot &robot) {
  while (!g_stop_requested) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    error_code ec;
    const auto state = robot.operationState(ec);
    if (!checkEc(ec, "operationState")) return false;
    if (state == OperationState::idle || state == OperationState::unknown) {
      return true;
    }
  }

  return false;
}

bool moveTo(ArRobot &robot, const JointPosition &target, const char *name) {
  error_code ec;
  MoveAbsJCommand command(target, 5000, 10);
  std::string command_id;

  std::cout << "运动到" << name << std::endl;
  robot.moveAppend({command}, command_id, ec);
  if (!checkEc(ec, "moveAppend")) return false;

  robot.moveStart(ec);
  if (!checkEc(ec, "moveStart")) return false;

  return waitRobot(robot);
}

}  // namespace

int main() {
  std::signal(SIGINT, signalHandler);

  try {
    ArRobot robot("192.168.2.160", "192.168.2.10");
    error_code ec;

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkEc(ec, "setMotionControlMode(NrtCommand)")) return 1;

    const JointPosition point_a = {1.18856966, -1.570796327, -1.570796327, 0.0, 0.0, 0.0, 0.0};
    const JointPosition point_b = {1.9, -1.1, -2.0, 0.5235987756, 0.5, 0.5, 0.5};

    std::cout << "右臂开始在A、B两点间循环运动，按Ctrl+C停止。" << std::endl;
    while (!g_stop_requested) {
      if (!moveTo(robot, point_a, "A点")) break;
      if (!moveTo(robot, point_b, "B点")) break;
    }

    if (g_stop_requested) {
      std::cout << "\n收到Ctrl+C，停止右臂并清空运动队列。" << std::endl;
      robot.stop(ec);
      checkEc(ec, "stop");
      robot.moveReset(ec);
      checkEc(ec, "moveReset");
    } else {
      std::cerr << "循环运动因SDK错误停止。" << std::endl;
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
