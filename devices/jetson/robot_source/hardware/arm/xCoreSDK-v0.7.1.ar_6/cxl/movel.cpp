#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "rokae/robot.h"

using namespace rokae;
std::ostream &os = std::cout;

void waitRobot(BaseRobot &robot) {
  bool running = true;
  while (running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    error_code ec;
    auto st = robot.operationState(ec);
    if (st == OperationState::idle || st == OperationState::unknown) {
      running = false;
    }
  }
}

void movel(ArRobot &robot, error_code &ec) {
  // Same physical right-arm flange target as the original example, expressed
  // in the current external reference frame configured by set_rotation.cpp:
  // baseFrame = [0, -0.105, 0, +pi/2, -0.132645, 0].
  CartesianPosition cart_pos1(
      {0.3431931361, -0.3895329999, -0.3578980637,
       -2.6692857074, 0.2414876934, -0.7195532289});
  cart_pos1.elbow = 0.540357 * M_PI / 180;
 
  // CartesianPosition cart_pos2(
  //     {0.53466, -0.351432, 1.69627e-16, 1.5708, 2.06757e-16, 1.5708});
  // cart_pos2.elbow = 16.6993 * M_PI / 180;

  MoveLCommand movejcmd(cart_pos1, 50, 0);
  // MoveLCommand movejcmd2(cart_pos2, 100, 10);

  std::string cmdID = "";
  // robot.moveAppend({movejcmd, movejcmd2}, cmdID, ec);
  robot.moveAppend({movejcmd}, cmdID, ec);
  if (ec)
    os << "ec: " << ec.value() << std::endl;

  robot.moveStart(ec);
  if (ec)
    os << "ec: " << ec.value() << std::endl;

  waitRobot(robot);
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (ec) {
    os << "ec: " << ec.value() << std::endl;
    return;
  }

  os << "最近5条控制器错误日志:" << std::endl;
  for (const auto &log : logs) {
    os << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

int main() {
  try {
    ArRobot robot("192.168.2.160", "192.168.2.10");
    error_code ec;

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (ec)
      os << "ec: " << ec.value() << std::endl;

    movel(robot, ec);
    printControllerErrors(robot);
  } catch (const std::exception &e) {
    os << e.what() << std::endl;
    return 1;
  }

  return 0;
}
