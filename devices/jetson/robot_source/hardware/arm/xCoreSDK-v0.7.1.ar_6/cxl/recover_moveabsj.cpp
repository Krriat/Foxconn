/**
 * 软限位恢复后 MoveAbsJ 回位。用法：recover_moveabsj [--show-target]
 * 和 moveabsj.cpp 一样，通过注释选择目标坐标及左/右臂连接；两处必须对应。
 * 执行前结束其他机械臂控制程序并确认回位路径畅通。
 */
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include "soft_limit_recovery.hpp"

using namespace rokae;

namespace {
volatile std::sig_atomic_t stopRequested = 0;
void requestStop(int) { stopRequested = 1; }

soft_limit_recovery::Joints targetJoints() {
  return {
      // 右原点
      {-1.4399, 1.570796327, 1.570796327, 0.0, 0, 0.0, 0.0},
      //-1.70344, 1.570796327, 0, 0.0, 0, 0.0, 0.0
      // 右初始点
      // {-1.179841224,  1.570796327, 1.57079632, 1.04719755, 0.0, 0, 0.0},
      // 右抓取点
      // {-1.969919, 1.365856, 1.272963, 1.476893, 1.735811,  -0.026802, 0.165558},
      //右测试
      // {0, 0, 0, 0, 0, 0, 0},
      // {-1.50344, 1.270796327, 0, 0.0, 0, 0.0, 0.0},

      // 左原点
      // {1.4399, 1.570796327, -1.570796327, 0.00, 0, 0.0, 0.0},
      // 左初始点1.70344, 1.570796327, 0, 0.00, -1.570796327, 0.0, 0.0
      // {1.3, 1.2, -1.0, 0.5, -0.5, 0.5, 0.0},
      // 左抓取点
      // {1.66969, 1.365856, -1.272963, 1.476893, -1.735811, -0.026802, -0.165558},
      // 左测试
      // {1.10344, 1.570796327, -1.570796327, 0.8, 0, 0.0, 0.0},
      // {0, 0, 0, 0, 0, 0, 0},
  };
}
}

int main(int argc, char **argv) {
  if(argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "用法：recover_moveabsj [--show-target]\n"
                 "在源码中注释选择目标坐标和左/右臂连接，重新编译后运行。\n"
                 "--show-target：只显示目标，不连接控制器。\n";
    return 0;
  }
  if(argc > 2 || (argc == 2 && std::string(argv[1]) != "--show-target")) {
    std::cerr << "用法：recover_moveabsj [--show-target]" << std::endl;
    return 2;
  }
  const auto target = targetJoints();
  std::cout << std::setprecision(12) << "目标关节角(rad)：[";
  for(size_t i=0; i<target.size(); ++i) std::cout << (i ? ", " : "") << target[i];
  std::cout << "]" << std::endl;
  if(argc == 2) return 0;
  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);
  try {
    //右
    ArRobot robot("192.168.2.160", "192.168.2.10");
    //左
    // ArRobot robot("192.168.1.160", "192.168.1.10");
    error_code ec;
    try {
      soft_limit_recovery::Recovery<ArRobot> recovery(
          robot, std::cout, [] { return stopRequested != 0; });
      recovery.recoverAndMoveAbsJ(target);
    } catch(...) {
      const auto logs = robot.queryControllerLog(5, {rokae::LogInfo::error}, ec);
      if(!ec) for(const auto &log : logs)
        std::cerr << "[" << log.timestamp << "] " << log.content << std::endl;
      throw;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << "恢复回位失败：" << e.what() << std::endl;
    return 1;
  }
}
