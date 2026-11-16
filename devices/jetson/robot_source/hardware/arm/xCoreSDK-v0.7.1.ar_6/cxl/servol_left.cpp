// Left-arm MoveAbsJ preparation followed by realtime Cartesian ServoL.
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;
std::ostream &os = std::cout;

namespace {

constexpr double kPlanPeriod = 0.02;
constexpr double kMoveTime = 10.0;
constexpr double kYDistance = 1.0;
constexpr double kJointLimitMargin = 15.0 * M_PI / 180.0;

const std::array<double, 7> kInitialJoint = {
    1.3, 1.2, -1.0, 0.5, -0.5, 0.5, 0.0
};

const std::array<double, 7> kLowerLimit = {
    -3.10669 + kJointLimitMargin,
    -2.0944 + kJointLimitMargin,
    -3.10669 + kJointLimitMargin,
    -1.0472 + kJointLimitMargin,
    -3.10669 + kJointLimitMargin,
    -1.0472 + kJointLimitMargin,
    -1.0472 + kJointLimitMargin,
};

const std::array<double, 7> kUpperLimit = {
    3.10669 - kJointLimitMargin,
    2.0944 - kJointLimitMargin,
    3.10669 - kJointLimitMargin,
    2.53073 - kJointLimitMargin,
    3.10669 - kJointLimitMargin,
    1.0472 - kJointLimitMargin,
    1.0472 - kJointLimitMargin,
};

volatile std::sig_atomic_t g_stop_requested = 0;

void signalHandler(int) {
  g_stop_requested = 1;
}

bool checkEc(const error_code &ec, const char *operation) {
  if (!ec) return true;
  os << operation << " ec: " << ec.value() << ", " << ec.message()
     << std::endl;
  return false;
}

bool insideJointLimit(const std::array<double, 7> &joints) {
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (joints[i] < kLowerLimit[i] || joints[i] > kUpperLimit[i]) {
      os << "关节越界: J" << i + 1 << "=" << joints[i]
         << " rad, limit=[" << kLowerLimit[i] << ", " << kUpperLimit[i]
         << "] rad" << std::endl;
      return false;
    }
  }
  return true;
}

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

bool moveToInitialJoint(ArRobot &robot, error_code &ec) {
  if (!insideJointLimit(kInitialJoint)) return false;

  MoveAbsJCommand absjcmd1(
      {1.3, 1.2, -1.0, 0.5, -0.5, 0.5, 0.0},
      1000, 10);

  std::string cmdID = "";
  robot.moveAppend({absjcmd1}, cmdID, ec);
  if (!checkEc(ec, "moveAppend")) return false;

  robot.moveStart(ec);
  if (!checkEc(ec, "moveStart")) return false;

  waitRobot(robot);

  const auto joints = robot.jointPos(ec);
  if (!checkEc(ec, "jointPos")) return false;
  return insideJointLimit(joints);
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (ec) {
    os << "queryControllerLog ec: " << ec.value() << std::endl;
    return;
  }

  os << "最近5条控制器错误日志:" << std::endl;
  for (const auto &log : logs) {
    os << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

}  // namespace

int main() {
  std::signal(SIGINT, signalHandler);

  ArRobot robot;
  error_code ec;
  try {
    robot.connectToRobot("192.168.1.160", "192.168.1.10");

    robot.setOperateMode(OperateMode::automatic, ec);
    if (!checkEc(ec, "setOperateMode")) return 1;

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkEc(ec, "setMotionControlMode(NrtCommand)")) return 1;

    robot.setPowerState(true, ec);
    if (!checkEc(ec, "setPowerState")) return 1;

    os << "先执行MoveAbsJ到ServoL初始关节位置" << std::endl;
    if (!moveToInitialJoint(robot, ec)) {
      printControllerErrors(robot);
      return 1;
    }

    robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
    if (!checkEc(ec, "setMotionControlMode(RtCommand)")) return 1;

    auto rtCon = robot.getRtMotionController().lock();
    if (!rtCon) {
      os << "获取实时运动控制器失败" << std::endl;
      return 1;
    }

    robot.startReceiveRobotState(
        std::chrono::milliseconds(1),
        {RtSupportedFields::jointPos_m, RtSupportedFields::tcpPose_m});
    while (robot.updateRobotState(std::chrono::steady_clock::duration::zero()));

    std::array<double, 7> measured_joints{};
    robot.updateRobotState(std::chrono::milliseconds(1));
    if (robot.getStateData(RtSupportedFields::jointPos_m, measured_joints) != 0 ||
        !insideJointLimit(measured_joints)) {
      os << "读取关节状态失败或当前关节超出限制" << std::endl;
      return 1;
    }

    CartesianPosition initial_pose;
    const auto flange_pose = robot.posture(CoordinateType::flangeInBase, ec);
    if (!checkEc(ec, "posture(flangeInBase)")) return 1;
    Utils::postureToTransArray(flange_pose, initial_pose.pos);

    Eigen::Matrix3d initial_rotation;
    Eigen::Vector3d initial_translation;
    Utils::arrayToTransMatrix(
        initial_pose.pos, initial_rotation, initial_translation);

    os << "ServoL起点: X=" << initial_translation[0]
       << ", Y=" << initial_translation[1]
       << ", Z=" << initial_translation[2] << " m" << std::endl;

    rtCon->setServoJoint(kPlanPeriod, kPlanPeriod, 0, ec);
    if (!checkEc(ec, "setServoJoint")) return 1;
    rtCon->startMove(RtControllerMode::cartesianPosition);

    const std::size_t total_frames =
        static_cast<std::size_t>(std::ceil(kMoveTime / kPlanPeriod));
    CartesianPosition cmd;
    cmd.pos = initial_pose.pos;
    auto next_cycle = std::chrono::steady_clock::now();
    bool aborted = false;

    os << "ServoL开始：沿基坐标系Y+移动0.1 m，按Ctrl+C可正常结束。"
       << std::endl;

    for (std::size_t i = 1; i <= total_frames; ++i) {
      if (g_stop_requested) {
        os << "收到Ctrl+C，结束ServoL" << std::endl;
        aborted = true;
        break;
      }

      robot.updateRobotState(std::chrono::milliseconds(1));
      if (robot.getStateData(RtSupportedFields::jointPos_m, measured_joints) != 0 ||
          !insideJointLimit(measured_joints)) {
        os << "关节限制触发，结束ServoL" << std::endl;
        aborted = true;
        break;
      }

      const double ratio = static_cast<double>(i) / total_frames;
      const double smooth_ratio = 0.5 * (1.0 - std::cos(M_PI * ratio));
      Eigen::Vector3d target_translation = initial_translation;
      target_translation[0] -= kYDistance * smooth_ratio;
      Utils::transMatrixToArray(initial_rotation, target_translation, cmd.pos);

      if (i == total_frames) cmd.setFinished();
      rtCon->sendCommand(cmd);

      next_cycle += std::chrono::milliseconds(20);
      std::this_thread::sleep_until(next_cycle);
    }

    if (aborted) {
      cmd.setFinished();
      rtCon->sendCommand(cmd);
    }

    rtCon->stopServoJoint();
    robot.stopReceiveRobotState();

    if (!aborted) {
      os << "ServoL完成：Y方向移动0.1 m" << std::endl;
    }

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkEc(ec, "恢复NrtCommand模式")) return 1;

    printControllerErrors(robot);
    return aborted ? 1 : 0;
  } catch (const std::exception &e) {
    os << "ServoL异常: " << e.what() << std::endl;
    printControllerErrors(robot);
    return 1;
  }
}
