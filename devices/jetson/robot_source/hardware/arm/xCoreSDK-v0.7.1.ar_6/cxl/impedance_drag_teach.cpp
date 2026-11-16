/**
 * @file impedance_drag_teach.cpp
 * @brief Conservative fixed-equilibrium Cartesian impedance demo.
 *
 * Usage:
 *   impedance_drag_teach <left|right> [duration_seconds]
 *
 * This is an impedance demonstration, not the controller's built-in drag and
 * path-recording mode. The desired wrench is deliberately zero so the robot
 * does not actively push in any direction. The equilibrium pose is fixed at
 * the startup pose; the arm therefore returns toward that pose after release.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kControlPeriod = 0.001;  // 1 kHz.
constexpr double kMaxDistanceFromStart = 0.1;  // m.
constexpr double kMaxRotationFromStart = 20.0 * kPi / 180.0;
constexpr double kMaxLinearSpeed = 0.25;   // m/s.
constexpr double kMaxAngularSpeed = 1.20;  // rad/s.
constexpr double kSpeedFilterAlpha = 0.05;
constexpr double kStartupCheckDuration = 2.0;  // s.
constexpr double kStartupMaxPositionDrift = 0.010;  // m.
constexpr double kStartupMaxRotationDrift = 3.0 * kPi / 180.0;
constexpr double kJointLimitMargin = 12.0 * kPi / 180.0;

// A conservative starting point for hand guiding. Increase the first three
// values if the arm feels too loose, and decrease them if it is too hard to
// move. The last three values are rotational stiffnesses.
const std::array<double, 6> kCartesianStiffness = {
    250.0, 250.0, 250.0, 18.0, 18.0, 18.0,
};

// Manual guiding should not contain an active force/torque bias.
const std::array<double, 6> kDesiredWrench = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

enum class StopReason {
  durationReached,
  ctrlC,
  stateReadFailed,
  jointLimit,
  workspaceLimit,
  orientationLimit,
  speedLimit,
  startupDrift,
};

struct NetworkConfig {
  const char *name;
  const char *robotIp;
  const char *localIp;
};

bool checkError(const error_code &ec, const std::string &operation) {
  if (!ec) return true;
  std::cerr << operation << " failed: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

bool shutdownRobot(ArRobot &robot) {
  bool success = true;
  error_code ec;
  robot.setPowerState(false, ec);
  if (!checkError(ec, "power off")) success = false;
  ec.clear();
  robot.setOperateMode(OperateMode::manual, ec);
  if (!checkError(ec, "set manual mode")) success = false;
  ec.clear();
  robot.setMotionControlMode(MotionControlMode::Idle, ec);
  if (!checkError(ec, "switch to Idle")) success = false;
  return success;
}

class RobotShutdownGuard {
 public:
  explicit RobotShutdownGuard(ArRobot &robot) : robot_(robot) {}

  ~RobotShutdownGuard() {
    if (!dismissed_) {
      try {
        shutdownRobot(robot_);
      } catch (const std::exception &) {
      }
    }
  }

  void dismiss() { dismissed_ = true; }

 private:
  ArRobot &robot_;
  bool dismissed_{false};
};

NetworkConfig parseArm(const std::string &side) {
  if (side == "left") {
    return {"left", "192.168.1.160", "192.168.1.10"};
  }
  if (side == "right") {
    return {"right", "192.168.2.160", "192.168.2.10"};
  }
  throw std::invalid_argument("arm must be 'left' or 'right'");
}

double parseDuration(int argc, char **argv) {
  if (argc < 3) return 60.0;
  std::size_t parsed = 0;
  const std::string text(argv[2]);
  const double duration = std::stod(text, &parsed);
  if (parsed != text.size() || !std::isfinite(duration) || duration < 1.0 ||
      duration > 600.0) {
    throw std::invalid_argument("duration must be between 1 and 600 seconds");
  }
  return duration;
}

bool insideJointLimits(const std::array<double, 7> &joints) {
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (joints[i] < kLowerLimit[i] || joints[i] > kUpperLimit[i]) {
      return false;
    }
  }
  return true;
}

void printJointPositions(const std::array<double, 7> &joints) {
  std::cout << "Current joint positions:" << std::endl;
  for (std::size_t i = 0; i < joints.size(); ++i) {
    std::cout << "  J" << i + 1 << " = " << joints[i] << " rad ("
              << joints[i] * 180.0 / kPi << " deg)" << std::endl;
  }
}

const char *stopReasonText(StopReason reason) {
  switch (reason) {
    case StopReason::durationReached:
      return "configured duration reached";
    case StopReason::ctrlC:
      return "Ctrl+C received";
    case StopReason::stateReadFailed:
      return "real-time state read failed";
    case StopReason::jointLimit:
      return "joint safety boundary reached";
    case StopReason::workspaceLimit:
      return "TCP moved more than 0.05 m from the start pose";
    case StopReason::orientationLimit:
      return "TCP rotated more than 20 deg from the start pose";
    case StopReason::speedLimit:
      return "TCP speed exceeded the configured safety limit";
    case StopReason::startupDrift:
      return "TCP drifted during the initial 2 s hands-off check";
  }
  return "unknown";
}

void printToolLoad(ArRobot &robot) {
  error_code ec;
  const auto toolset = robot.toolset(ec);
  if (!checkError(ec, "read tool load")) return;
  std::cout << "Configured tool load: mass=" << toolset.load.mass
            << " kg, cog=[" << toolset.load.cog[0] << ", "
            << toolset.load.cog[1] << ", " << toolset.load.cog[2]
            << "] m" << std::endl;
}

void printRobotInfo(ArRobot &robot) {
  error_code ec;
  const auto info = robot.robotInfo(ec);
  if (!checkError(ec, "read robot info")) return;
  std::cout << "Robot type: " << info.type
            << ", controller version: " << info.version << std::endl;
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (!checkError(ec, "query controller error logs")) return;
  if (logs.empty()) {
    std::cout << "Controller returned no error log entries." << std::endl;
    return;
  }
  std::cout << "Latest controller error logs:" << std::endl;
  for (const auto &log : logs) {
    std::cout << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

bool receiveInitialState(ArRobot &robot, std::array<double, 16> &pose,
                         std::array<double, 7> &joints) {
  while (robot.updateRobotState(
      std::chrono::steady_clock::duration::zero())) {
  }
  if (!robot.updateRobotState(std::chrono::milliseconds(100))) return false;
  while (robot.updateRobotState(
      std::chrono::steady_clock::duration::zero())) {
  }
  return robot.getStateData(RtSupportedFields::tcpPose_m, pose) == 0 &&
         robot.getStateData(RtSupportedFields::jointPos_m, joints) == 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " <left|right> [duration_seconds]" << std::endl;
    return 2;
  }

  NetworkConfig config{};
  double duration = 0.0;
  try {
    config = parseArm(argv[1]);
    duration = parseDuration(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return 2;
  }

  std::signal(SIGINT, signalHandler);

  try {
    std::cout << "Connecting " << config.name << " arm: robot="
              << config.robotIp << ", local=" << config.localIp << std::endl;
    ArRobot robot(config.robotIp, config.localIp);
    RobotShutdownGuard shutdownGuard(robot);
    error_code ec;

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkError(ec, "switch to NrtCommand")) return 1;
    robot.setOperateMode(OperateMode::automatic, ec);
    if (!checkError(ec, "set automatic mode")) return 1;
    robot.setPowerState(true, ec);
    if (!checkError(ec, "power on")) return 1;

    printRobotInfo(robot);
    printToolLoad(robot);
    std::cout << "WARNING: verify the tool mass and center of gravity before "
                 "hand guiding. Hold the E-stop and support the arm."
              << std::endl
              << "Press Enter to enable low-stiffness impedance mode..."
              << std::endl;
    std::cin.get();
    if (gStopRequested) return 0;

    // This must be configured before entering RT command mode.
    robot.setRtNetworkTolerance(20, ec);
    if (!checkError(ec, "set RT network tolerance")) return 1;
    robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
    if (!checkError(ec, "switch to RtCommand")) return 1;
    robot.setOperateMode(OperateMode::automatic, ec);
    if (!checkError(ec, "confirm automatic mode")) return 1;
    robot.setPowerState(true, ec);
    if (!checkError(ec, "confirm power on")) return 1;

    auto rtController = robot.getRtMotionController().lock();
    if (!rtController) {
      std::cerr << "Failed to obtain real-time controller" << std::endl;
      return 1;
    }

    const std::array<double, 16> identityFrame = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    // This SDK version accepts the Cartesian impedance frame as a tool frame.
    // Identity means that the force-control tool frame coincides with flange.
    rtController->setFcCoor(identityFrame, FrameType::tool, ec);
    if (!checkError(ec, "set force-control frame")) return 1;
    rtController->setCartesianImpedance(kCartesianStiffness, ec);
    if (!checkError(ec, "set Cartesian impedance")) return 1;
    rtController->setCartesianImpedanceDesiredTorque(kDesiredWrench, ec);
    if (!checkError(ec, "set zero desired wrench")) return 1;

    bool receivingState = false;
    try {
      robot.startReceiveRobotState(
          std::chrono::milliseconds(1),
          {RtSupportedFields::tcpPose_m, RtSupportedFields::jointPos_m});
      receivingState = true;

      std::array<double, 16> initialPose{};
      std::array<double, 7> initialJoints{};
      if (!receiveInitialState(robot, initialPose, initialJoints) ||
          !insideJointLimits(initialJoints)) {
        std::cerr << "Initial robot state is unavailable or outside the "
                     "configured joint safety margin"
                  << std::endl;
        robot.stopReceiveRobotState();
        return 1;
      }
      printJointPositions(initialJoints);

      Eigen::Matrix3d equilibriumRotation;
      Eigen::Vector3d equilibriumPosition;
      Utils::arrayToTransMatrix(initialPose, equilibriumRotation,
                                equilibriumPosition);
      const Eigen::Vector3d initialPosition = equilibriumPosition;
      const Eigen::Matrix3d initialRotation = equilibriumRotation;
      Eigen::Vector3d previousPosition = initialPosition;
      Eigen::Matrix3d previousRotation = initialRotation;
      double filteredLinearSpeed = 0.0;
      double filteredAngularSpeed = 0.0;

      std::size_t cycles = 0;
      StopReason stopReason = StopReason::durationReached;

      std::function<CartesianPosition(void)> callback = [&]() {
        CartesianPosition command{};
        std::array<double, 16> measuredPose{};
        std::array<double, 7> measuredJoints{};

        if (robot.getStateData(RtSupportedFields::tcpPose_m, measuredPose) !=
                0 ||
            robot.getStateData(RtSupportedFields::jointPos_m,
                               measuredJoints) != 0) {
          stopReason = StopReason::stateReadFailed;
          Utils::transMatrixToArray(equilibriumRotation,
                                    equilibriumPosition, command.pos);
          command.setFinished();
          return command;
        }

        if (!insideJointLimits(measuredJoints)) {
          stopReason = StopReason::jointLimit;
          Utils::transMatrixToArray(equilibriumRotation,
                                    equilibriumPosition, command.pos);
          command.setFinished();
          return command;
        }

        Eigen::Matrix3d measuredRotation;
        Eigen::Vector3d measuredPosition;
        Utils::arrayToTransMatrix(measuredPose, measuredRotation,
                                  measuredPosition);
        const double positionOffset =
            (measuredPosition - initialPosition).norm();
        if (positionOffset > kMaxDistanceFromStart) {
          stopReason = StopReason::workspaceLimit;
          Utils::transMatrixToArray(equilibriumRotation,
                                    equilibriumPosition, command.pos);
          command.setFinished();
          return command;
        }

        const double rotationOffset = std::abs(
            Eigen::AngleAxisd(measuredRotation * initialRotation.transpose())
                .angle());
        if (rotationOffset > kMaxRotationFromStart) {
          stopReason = StopReason::orientationLimit;
          Utils::transMatrixToArray(equilibriumRotation,
                                    equilibriumPosition, command.pos);
          command.setFinished();
          return command;
        }

        const double instantaneousLinearSpeed =
            (measuredPosition - previousPosition).norm() / kControlPeriod;
        const double instantaneousAngularSpeed = std::abs(
            Eigen::AngleAxisd(measuredRotation * previousRotation.transpose())
                .angle()) /
            kControlPeriod;
        filteredLinearSpeed += kSpeedFilterAlpha *
                               (instantaneousLinearSpeed -
                                filteredLinearSpeed);
        filteredAngularSpeed += kSpeedFilterAlpha *
                                (instantaneousAngularSpeed -
                                 filteredAngularSpeed);
        previousPosition = measuredPosition;
        previousRotation = measuredRotation;

        if (filteredLinearSpeed > kMaxLinearSpeed ||
            filteredAngularSpeed > kMaxAngularSpeed) {
          stopReason = StopReason::speedLimit;
          Utils::transMatrixToArray(equilibriumRotation,
                                    equilibriumPosition, command.pos);
          command.setFinished();
          return command;
        }

        const double elapsed = cycles * kControlPeriod;
        if (elapsed < kStartupCheckDuration &&
            (positionOffset > kStartupMaxPositionDrift ||
             rotationOffset > kStartupMaxRotationDrift)) {
          stopReason = StopReason::startupDrift;
          Utils::transMatrixToArray(equilibriumRotation,
                                    equilibriumPosition, command.pos);
          command.setFinished();
          return command;
        }

        // Keep the virtual spring equilibrium fixed at the startup pose.
        // Unlike the previous version, measured drift is never copied into
        // the command, so load or force-sensor bias cannot ratchet the target.
        Utils::transMatrixToArray(equilibriumRotation, equilibriumPosition,
                                  command.pos);

        ++cycles;
        if (gStopRequested) {
          stopReason = StopReason::ctrlC;
          command.setFinished();
        } else if (cycles * kControlPeriod >= duration) {
          stopReason = StopReason::durationReached;
          command.setFinished();
        }
        return command;
      };

      rtController->setControlLoop(callback, 0, true);
      rtController->startMove(RtControllerMode::cartesianImpedance);

      std::cout << "Impedance mode active for up to " << duration
                << " s. KEEP HANDS OFF for the first 2 s; then push/pull "
                   "slowly. Press Ctrl+C to stop."
                << std::endl
                << "Stiffness [X Y Z Rx Ry Rz] = [250 250 250 18 18 18], "
                   "desired wrench = [0 0 0 0 0 0]."
                << std::endl;
      rtController->startLoop(true);

      std::cout << "Hand guiding stopped: " << stopReasonText(stopReason)
                << std::endl;
      robot.stopReceiveRobotState();
      receivingState = false;
    } catch (const std::exception &e) {
      std::cerr << "Real-time impedance exception: " << e.what()
                << std::endl;
      try {
        rtController->stopMove();
      } catch (const std::exception &) {
      }
      if (receivingState) {
        try {
          robot.stopReceiveRobotState();
        } catch (const std::exception &) {
        }
      }
      printControllerErrors(robot);
      return 1;
    }

    const bool shutdownOk = shutdownRobot(robot);
    shutdownGuard.dismiss();
    return shutdownOk ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "Program exception: " << e.what() << std::endl;
    return 1;
  }
}
