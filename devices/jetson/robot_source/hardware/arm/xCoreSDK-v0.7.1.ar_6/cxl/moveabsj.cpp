#include <chrono>
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

void moveabsj(ArRobot &robot, error_code &ec) {
  MoveAbsJCommand absjcmd1(
      // 右原点
      // {-1.70344 ,1.570796327, 0, 0, 0, 0, 0},
      // {-1.4399, 1.570796327, 1.570796327, 0.0, 0, 0.0, 0.0},
      // {0, 1.570796327, 1.570796327, 0.0, 0, 0.0, 0.0},
      // 右初始点
      // {-1.179841224,  1.570796327, 1.57079632, 1.04719755, 0.0, 0, 0.0},
      // 右抓取点
      // {-1.969919, 1.365856, 1.272963, 1.476893, 1.735811,  -0.026802, 0.165558},
      //右测试
      // {0, 0, 0, 0, 0, 0, 0},
      {-0.859278, 1.252637, 0.908581, 1.105772, -1.085419, -0.423597, -0.193035},


      // 左原点
      // {1.4399, 1.570796327, -1.570796327, 0.00, 0, 0.0, 0.0},
      // 左初始点
      // {1.3, 1.2, -1.0, 0.5, -0.5, 0.5, 0.0},
      // 左抓取点
      // {1.66969, 1.365856, -1.272963, 1.476893, -1.735811, -0.026802, -0.165558},
      // 左测试
      // {0, 0, 0, 2.2689, 0, 0.8726, 0},
      // {-0.1308996939, 1.570796327, 0, 0, 0, 0.0, 0.0},


      100, 10);

  std::string cmdID = "";
  robot.moveAppend({absjcmd1}, cmdID, ec);
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
    //左
    // ArRobot robot("192.168.1.160", "192.168.1.10");
    //右
    ArRobot robot("192.168.2.160", "192.168.2.10");
    error_code ec;

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (ec)
      os << "ec: " << ec.value() << std::endl;

    moveabsj(robot, ec);
    printControllerErrors(robot);
  } catch (const std::exception &e) {
    os << e.what() << std::endl;
    return 1;
  }

  return 0;
}

