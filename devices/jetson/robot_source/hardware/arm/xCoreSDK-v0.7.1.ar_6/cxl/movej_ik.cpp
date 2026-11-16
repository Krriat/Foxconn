/**
 * @file movej_ik.cpp
 * @brief Convert an absolute Cartesian target to joint positions with the
 *        SDK IK solver, then execute it as a non-realtime MoveAbsJ.
 *
 * Usage:
 *   movej_ik <left|right> <x_m> <y_m> <z_m>
 *            <roll_rad> <pitch_rad> <yaw_rad>
 *
 * The target is the TCP pose in the configured external reference frame.
 * This is joint-space motion: only the endpoint is Cartesian; the TCP path
 * between the start and target is not guaranteed to be a straight line.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rokae/model.h"
#include "rokae/robot.h"

using namespace rokae;

namespace {

constexpr std::size_t kJointCount = 7;
constexpr double kPi = 3.14159265358979323846;
constexpr double kSpeedMmS = 80.0;
constexpr double kJointSpeedScale = 0.30;
constexpr double kZoneMm = 0.0;
constexpr double kMaxTranslationM = 0.40;
constexpr double kMaxRotationRad = 60.0 * kPi / 180.0;
constexpr double kMaxJointDeltaRad = 0.6;
constexpr double kSoftLimitMarginRad = 0.08;
constexpr double kIkPositionToleranceM = 0.005;
constexpr double kIkRotationToleranceRad = 2.0 * kPi / 180.0;
constexpr double kFinalJointToleranceRad = 0.01;

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

struct NetworkConfig {
  const char *name;
  const char *robotIp;
  const char *localIp;
};

using RotationMatrix = std::array<std::array<double, 3>, 3>;

bool checkError(const error_code &ec, const std::string &operation) {
  if (!ec) return true;
  std::cerr << operation << " failed: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

NetworkConfig parseArm(const std::string &side) {
  if (side == "left") {
    return {"left", "192.168.1.160", "192.168.1.10"};
  }
  if (side == "right") {
    return {"right", "192.168.2.160", "192.168.2.10"};
  }
  throw std::invalid_argument("arm must be 'left' or 'right'");
}

double parseNumber(const char *text, const char *name) {
  std::size_t parsed = 0;
  const std::string valueText(text);
  const double value = std::stod(valueText, &parsed);
  if (parsed != valueText.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) +
                                " must be a finite number");
  }
  return value;
}

double positionDistance(const CartesianPosition &a,
                        const CartesianPosition &b) {
  const double dx = a.trans[0] - b.trans[0];
  const double dy = a.trans[1] - b.trans[1];
  const double dz = a.trans[2] - b.trans[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

RotationMatrix rpyToRotation(const std::array<double, 3> &rpy) {
  const double cr = std::cos(rpy[0]);
  const double sr = std::sin(rpy[0]);
  const double cp = std::cos(rpy[1]);
  const double sp = std::sin(rpy[1]);
  const double cy = std::cos(rpy[2]);
  const double sy = std::sin(rpy[2]);

  return {{
      {{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr}},
      {{sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr}},
      {{-sp, cp * sr, cp * cr}},
  }};
}

double orientationDistance(const CartesianPosition &a,
                           const CartesianPosition &b) {
  const RotationMatrix ra = rpyToRotation(a.rpy);
  const RotationMatrix rb = rpyToRotation(b.rpy);
  double traceRelative = 0.0;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      traceRelative += ra[row][col] * rb[row][col];
    }
  }
  const double cosine =
      std::clamp((traceRelative - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cosine);
}

void printPose(const char *label, const CartesianPosition &pose) {
  std::cout << label << std::endl
            << "  XYZ = [" << pose.trans[0] << ", " << pose.trans[1]
            << ", " << pose.trans[2] << "] m" << std::endl
            << "  RPY = [" << pose.rpy[0] << ", " << pose.rpy[1] << ", "
            << pose.rpy[2] << "] rad" << std::endl
            << "  elbow = " << pose.elbow
            << " rad, valid=" << (pose.hasElbow ? "yes" : "no")
            << std::endl;
}

template <typename JointContainer>
void printJoints(const char *label, const JointContainer &joints) {
  std::cout << label << std::endl;
  for (std::size_t i = 0; i < joints.size(); ++i) {
    std::cout << "  J" << i + 1 << " = " << joints[i] << " rad ("
              << joints[i] * 180.0 / kPi << " deg)" << std::endl;
  }
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (!checkError(ec, "query controller error logs")) return;
  if (logs.empty()) return;
  std::cout << "Latest controller error logs:" << std::endl;
  for (const auto &log : logs) {
    std::cout << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

void stopMotion(ArRobot &robot) {
  error_code ec;
  robot.stop(ec);
  if (ec) {
    std::cerr << "stop failed: " << ec.message() << std::endl;
  }
  ec.clear();
  robot.moveReset(ec);
  if (ec) {
    std::cerr << "moveReset after stop failed: " << ec.message()
              << std::endl;
  }
}

bool waitRobotIdle(ArRobot &robot,
                   std::chrono::seconds timeout =
                       std::chrono::seconds(60)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    error_code ec;
    const auto state = robot.operationState(ec);
    if (!checkError(ec, "read operation state")) return false;
    if (state == OperationState::idle) return true;
    if (state == OperationState::unknown) {
      std::cerr << "Robot entered unknown operation state" << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::cerr << (gStopRequested ? "Ctrl+C received"
                               : "MoveAbsJ timed out")
            << std::endl;
  return false;
}

template <typename JointContainer>
bool validateIkSolution(
    ArRobot &robot, const JointContainer &targetJoints,
    const std::array<double, kJointCount> &startJoints) {
  if (targetJoints.size() != kJointCount) {
    std::cerr << "IK returned " << targetJoints.size()
              << " joints; expected " << kJointCount << std::endl;
    return false;
  }

  std::array<double[2], kJointCount> softLimits{};
  error_code ec;
  const bool softLimitsEnabled = robot.getSoftLimit(softLimits, ec);
  if (!checkError(ec, "read controller soft limits")) return false;
  if (!softLimitsEnabled) {
    std::cerr << "Controller soft limits are disabled; refusing to move"
              << std::endl;
    return false;
  }

  for (std::size_t i = 0; i < kJointCount; ++i) {
    const double target = targetJoints[i];
    if (!std::isfinite(target)) {
      std::cerr << "IK returned a non-finite value for J" << i + 1
                << std::endl;
      return false;
    }

    const double lower = softLimits[i][0] + kSoftLimitMarginRad;
    const double upper = softLimits[i][1] - kSoftLimitMarginRad;
    if (lower >= upper || target < lower || target > upper) {
      std::cerr << "IK target J" << i + 1
                << " violates the controller soft-limit margin: target="
                << target << ", allowed=[" << lower << ", " << upper
                << "] rad" << std::endl;
      return false;
    }

    const double delta = std::abs(target - startJoints[i]);
    if (delta > kMaxJointDeltaRad) {
      std::cerr << "IK target J" << i + 1 << " changes by " << delta
                << " rad, exceeding the " << kMaxJointDeltaRad
                << " rad demo limit" << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr
        << "Usage: " << argv[0]
        << " <left|right> <x_m> <y_m> <z_m>"
           " <roll_rad> <pitch_rad> <yaw_rad>"
        << std::endl;
    return 2;
  }

  NetworkConfig config{};
  std::array<double, 6> targetValues{};
  try {
    config = parseArm(argv[1]);
    const std::array<const char *, 6> names = {
        "x", "y", "z", "roll", "pitch", "yaw"};
    for (std::size_t i = 0; i < targetValues.size(); ++i) {
      targetValues[i] = parseNumber(argv[i + 2], names[i]);
    }
  } catch (const std::exception &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return 2;
  }

  std::signal(SIGINT, signalHandler);
  std::cout << std::fixed << std::setprecision(6);

  try {
    std::cout << "Connecting " << config.name << " arm: robot="
              << config.robotIp << ", local=" << config.localIp
              << std::endl;
    ArRobot robot(config.robotIp, config.localIp);
    error_code ec;

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkError(ec, "switch to NrtCommand")) return 1;
    robot.setOperateMode(OperateMode::automatic, ec);
    if (!checkError(ec, "set automatic mode")) return 1;
    robot.setPowerState(true, ec);
    if (!checkError(ec, "power on")) return 1;

    const auto state = robot.operationState(ec);
    if (!checkError(ec, "read initial operation state")) return 1;
    if (state != OperationState::idle) {
      std::cerr << "Robot must be idle before IK MoveJ; current state="
                << static_cast<int>(state) << std::endl;
      return 1;
    }

    const std::array<double, kJointCount> startJoints =
        robot.jointPos(ec);
    if (!checkError(ec, "read current joint positions")) return 1;

    const CartesianPosition start =
        robot.cartPosture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read current TCP pose")) return 1;

    // Copying the current CartesianPosition preserves its elbow angle,
    // configuration data and external-axis data as the IK branch seed.
    CartesianPosition target = start;
    target.trans = {targetValues[0], targetValues[1], targetValues[2]};
    target.rpy = {targetValues[3], targetValues[4], targetValues[5]};

    printPose("Current TCP pose in external reference frame:", start);
    printPose("Requested TCP target:", target);

    const double translation = positionDistance(start, target);
    const double rotation = orientationDistance(start, target);
    std::cout << "Requested change: translation=" << translation
              << " m, rotation=" << rotation * 180.0 / kPi << " deg"
              << std::endl;
    if (translation > kMaxTranslationM) {
      std::cerr << "Translation exceeds the " << kMaxTranslationM
                << " m demo limit" << std::endl;
      return 1;
    }
    if (rotation > kMaxRotationRad) {
      std::cerr << "Rotation exceeds the "
                << kMaxRotationRad * 180.0 / kPi
                << " deg demo limit" << std::endl;
      return 1;
    }

    const Toolset activeToolset = robot.toolset(ec);
    if (!checkError(ec, "read active tool/workobject")) return 1;

    auto model = robot.model();
    const auto ikResult = model.calcIk(target, activeToolset, ec);
    if (!checkError(ec, "calculate IK")) {
      printControllerErrors(robot);
      return 1;
    }
    printJoints("IK joint target:", ikResult);

    if (!validateIkSolution(robot, ikResult, startJoints)) return 1;

    const CartesianPosition fkCheck =
        model.calcFk(ikResult, activeToolset, ec);
    if (!checkError(ec, "verify IK with FK")) return 1;
    printPose("FK pose of the IK result:", fkCheck);

    const double ikPositionError = positionDistance(target, fkCheck);
    const double ikRotationError = orientationDistance(target, fkCheck);
    std::cout << "IK/FK check error: position="
              << ikPositionError * 1000.0 << " mm, rotation="
              << ikRotationError * 180.0 / kPi << " deg" << std::endl;
    if (ikPositionError > kIkPositionToleranceM ||
        ikRotationError > kIkRotationToleranceRad) {
      std::cerr << "IK/FK verification exceeded the demo tolerance; "
                   "refusing to move"
                << std::endl;
      return 1;
    }

    std::cout
        << "This is joint-space motion; the TCP path is not straight."
        << std::endl
        << "Speed=" << kSpeedMmS
        << " mm/s, joint speed scale=" << kJointSpeedScale
        << ", zone=" << kZoneMm << " mm." << std::endl
        << "Verify the full swept volume is clear and the E-stop is "
           "reachable. Type MOVE to execute: ";

    std::string confirmation;
    std::getline(std::cin, confirmation);
    if (confirmation != "MOVE") {
      std::cout << "Cancelled; no motion command was started" << std::endl;
      return 0;
    }
    if (gStopRequested) return 0;

    robot.moveReset(ec);
    if (!checkError(ec, "moveReset")) return 1;

    const std::vector<double> targetJointVector(
        ikResult.begin(), ikResult.end());
    MoveAbsJCommand command(
        JointPosition(targetJointVector), kSpeedMmS, kZoneMm);
    command.jointSpeed = kJointSpeedScale;

    std::string commandId;
    robot.moveAppend({command}, commandId, ec);
    if (!checkError(ec, "moveAppend(MoveAbsJ from IK)")) {
      printControllerErrors(robot);
      return 1;
    }
    std::cout << "Command ID: " << commandId << std::endl
              << "moveStart() is now triggering robot motion."
              << std::endl;

    robot.moveStart(ec);
    if (!checkError(ec, "moveStart(MoveAbsJ from IK)")) {
      printControllerErrors(robot);
      return 1;
    }
    if (!waitRobotIdle(robot)) {
      stopMotion(robot);
      printControllerErrors(robot);
      return 1;
    }

    const auto reachedJoints = robot.jointPos(ec);
    if (!checkError(ec, "read final joint positions")) return 1;
    printJoints("Reached joint positions:", reachedJoints);

    double maximumJointError = 0.0;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      maximumJointError = std::max(
          maximumJointError,
          std::abs(reachedJoints[i] - targetJointVector[i]));
    }
    std::cout << "Maximum final joint error = " << maximumJointError
              << " rad" << std::endl;
    if (maximumJointError > kFinalJointToleranceRad) {
      std::cerr << "Robot became idle outside the final joint tolerance"
                << std::endl;
      return 1;
    }

    const CartesianPosition reachedPose =
        robot.cartPosture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read final TCP pose")) return 1;
    printPose("Reached TCP pose:", reachedPose);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "IK MoveJ program exception: " << e.what() << std::endl;
    return 1;
  }
}
