/**
 * @file cartesian_drag_safe.cpp
 * @brief Conservative Cartesian-space hand-guiding demo for an AR robot.
 *
 * Usage:
 *   cartesian_drag_safe <left|right> <duration_seconds>
 *
 * The active tool load must already have been identified and stored in the
 * controller. This demo deliberately uses the SDK's normal drag-button path;
 * it does not enable direct drag.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDefaultToolMass = 0.0;          // kg; no attached tool.
constexpr std::array<double, 3> kDefaultToolCog{
    0.0, 0.0, 0.0};                              // m from flange.
constexpr double kMaxToolMass = 5.0;             // kg; AR5 payload guard.
constexpr double kMaxCogDistance = 0.60;         // m.
constexpr double kMaxTranslation = 0.50;         // m from the start TCP.
constexpr double kMaxRotation = 90.0 * kPi / 180.0;
constexpr double kMaxLinearSpeed = 1;         // m/s.
constexpr double kMaxJointSpeed = 2.0;           // rad/s.
constexpr int kSpeedViolationSamples = 2;
constexpr auto kMonitorPeriod = std::chrono::milliseconds(50);
constexpr auto kTcpPrintPeriod = std::chrono::milliseconds(200);

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

struct NetworkConfig {
  const char *name;
  const char *robotIp;
  const char *localIp;
};

struct MonitorResult {
  bool safe;
  std::string reason;
};

template <std::size_t N>
bool allFinite(const std::array<double, N> &values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

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

double parseDuration(const char *text) {
  std::size_t parsed = 0;
  const std::string valueText(text);
  const double duration = std::stod(valueText, &parsed);
  if (parsed != valueText.size() || !std::isfinite(duration) ||
      duration < 1.0 || duration > 600.0) {
    throw std::invalid_argument(
        "duration must be a finite number between 1 and 600 seconds");
  }
  return duration;
}

void printUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " <left|right> <duration_seconds>" << std::endl;
}

void validateLoad(const Load &load) {
  if (!std::isfinite(load.mass) || load.mass < 0.0 ||
      load.mass > kMaxToolMass) {
    throw std::runtime_error(
        "configured tool mass must be between 0 and 5 kg");
  }

  double squaredDistance = 0.0;
  for (const double value : load.cog) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("configured tool CoG contains a non-finite value");
    }
    squaredDistance += value * value;
  }
  if (std::sqrt(squaredDistance) > kMaxCogDistance) {
    throw std::runtime_error(
        "configured tool CoG is more than 0.60 m from the flange");
  }
}

void printLoad(const Load &load) {
  std::cout << "Current controller tool load:" << std::endl
            << "  mass = " << load.mass << " kg" << std::endl
            << "  cog = [" << load.cog[0] << ", " << load.cog[1] << ", "
            << load.cog[2] << "] m" << std::endl
            << "  inertia = [" << load.inertia[0] << ", "
            << load.inertia[1] << ", " << load.inertia[2]
            << "] kg*m^2" << std::endl;
}

bool loadMatchesDefaults(const Load &load) {
  constexpr double kTolerance = 1e-6;
  if (std::abs(load.mass - kDefaultToolMass) > kTolerance) return false;
  for (std::size_t i = 0; i < kDefaultToolCog.size(); ++i) {
    if (std::abs(load.cog[i] - kDefaultToolCog[i]) > kTolerance) return false;
  }
  return true;
}

void printPose(const char *label, const std::array<double, 6> &pose) {
  std::cout << label << std::endl
            << "  XYZ = [" << pose[0] << ", " << pose[1] << ", "
            << pose[2] << "] m" << std::endl
            << "  RPY = [" << pose[3] << ", " << pose[4] << ", "
            << pose[5] << "] rad" << std::endl;
}

Eigen::Matrix3d rotationFromPose(const std::array<double, 6> &pose) {
  std::array<double, 16> transform{};
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  Utils::postureToTransArray(pose, transform);
  Utils::arrayToTransMatrix(transform, rotation, translation);
  return rotation;
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (!checkError(ec, "query controller error logs") || logs.empty()) return;

  std::cout << "Latest controller error logs:" << std::endl;
  for (const auto &log : logs) {
    std::cout << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

class DragGuard {
 public:
  explicit DragGuard(ArRobot &robot) : robot_(robot) {}

  ~DragGuard() {
    if (!enabled_) return;
    try {
      error_code ec;
      robot_.disableDrag(ec);
    } catch (const std::exception &) {
    }
  }

  void setEnabled(bool enabled) { enabled_ = enabled; }

 private:
  ArRobot &robot_;
  bool enabled_{false};
};

MonitorResult monitorDrag(ArRobot &robot, ForceControl_T<7> &forceControl,
                          const std::array<double, 6> &startPose,
                          double duration) {
  error_code ec;
  auto previousPose = robot.posture(CoordinateType::endInRef, ec);
  if (!checkError(ec, "read initial TCP pose")) {
    return {false, "failed to read the initial TCP pose"};
  }
  auto previousJoints = robot.jointPos(ec);
  if (!checkError(ec, "read initial joint positions")) {
    return {false, "failed to read the initial joint positions"};
  }
  if (!allFinite(previousPose) || !allFinite(previousJoints)) {
    return {false, "initial robot feedback contains a non-finite value"};
  }

  const Eigen::Matrix3d startRotation = rotationFromPose(startPose);
  auto previousTime = std::chrono::steady_clock::now();
  auto previousTcpPrint = previousTime - kTcpPrintPeriod;
  const auto deadline = previousTime + std::chrono::milliseconds(
                                         static_cast<int>(duration * 1000));
  int speedViolationCount = 0;

  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(kMonitorPeriod);
    const auto now = std::chrono::steady_clock::now();

    const auto pose = robot.posture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read TCP pose during Cartesian drag")) {
      return {false, "TCP feedback failed"};
    }
    const auto joints = robot.jointPos(ec);
    if (!checkError(ec, "read joints during Cartesian drag")) {
      return {false, "joint feedback failed"};
    }
    if (!allFinite(pose) || !allFinite(joints)) {
      return {false, "robot feedback contains a non-finite value"};
    }

    const Eigen::Vector3d startPosition(startPose[0], startPose[1],
                                        startPose[2]);
    const Eigen::Vector3d position(pose[0], pose[1], pose[2]);
    const double translation = (position - startPosition).norm();
    if (translation > kMaxTranslation) {
      return {false, "TCP moved more than 0.50 m from the start pose"};
    }

    const Eigen::Matrix3d relativeRotation =
        rotationFromPose(pose) * startRotation.transpose();
    const double rotation =
        std::abs(Eigen::AngleAxisd(relativeRotation).angle());
    if (rotation > kMaxRotation) {
      return {false, "TCP rotated more than 90 deg from the start pose"};
    }

    const double dt = std::chrono::duration<double>(now - previousTime).count();
    const Eigen::Vector3d previousPosition(
        previousPose[0], previousPose[1], previousPose[2]);
    const double linearSpeed = (position - previousPosition).norm() / dt;

    double fastestJointSpeed = 0.0;
    std::size_t fastestJoint = 0;
    for (std::size_t i = 0; i < joints.size(); ++i) {
      const double speed = std::abs(joints[i] - previousJoints[i]) / dt;
      if (speed > fastestJointSpeed) {
        fastestJointSpeed = speed;
        fastestJoint = i;
      }
    }

    if (linearSpeed > kMaxLinearSpeed || fastestJointSpeed > kMaxJointSpeed) {
      ++speedViolationCount;
    } else {
      speedViolationCount = 0;
    }

    if (speedViolationCount >= kSpeedViolationSamples) {
      if (linearSpeed > kMaxLinearSpeed) {
        return {false, "TCP linear speed exceeded 0.20 m/s"};
      }
      return {false, "J" + std::to_string(fastestJoint + 1) +
                         " speed exceeded 1.0 rad/s"};
    }

    if (now - previousTcpPrint >= kTcpPrintPeriod) {
      std::array<double, 7> jointTorque{};
      std::array<double, 7> externalTorque{};
      std::array<double, 3> tcpTorque{};
      std::array<double, 3> tcpForce{};
      ec.clear();
      forceControl.getEndTorque(
          FrameType::tool, jointTorque, externalTorque,
          tcpTorque, tcpForce, ec);
      if (!checkError(ec, "read TCP wrench during Cartesian drag")) {
        return {false, "TCP wrench feedback failed"};
      }
      if (!allFinite(tcpForce) || !allFinite(tcpTorque)) {
        return {false, "TCP wrench feedback contains a non-finite value"};
      }

      const double forceNorm = std::sqrt(
          tcpForce[0] * tcpForce[0] + tcpForce[1] * tcpForce[1] +
          tcpForce[2] * tcpForce[2]);
      const double torqueNorm = std::sqrt(
          tcpTorque[0] * tcpTorque[0] + tcpTorque[1] * tcpTorque[1] +
          tcpTorque[2] * tcpTorque[2]);
      std::cout << "TCP XYZ[m]=[" << pose[0] << ", " << pose[1] << ", "
                << pose[2] << "] RPY[rad]=[" << pose[3] << ", "
                << pose[4] << ", " << pose[5] << "] F[N]=["
                << tcpForce[0] << ", " << tcpForce[1] << ", "
                << tcpForce[2] << "] |F|=" << forceNorm << " M[Nm]=["
                << tcpTorque[0] << ", " << tcpTorque[1] << ", "
                << tcpTorque[2] << "] |M|=" << torqueNorm << std::endl;
      previousTcpPrint = now;
    }

    previousPose = pose;
    previousJoints = joints;
    previousTime = now;
  }

  if (gStopRequested) return {true, "Ctrl+C requested"};
  return {true, "requested duration reached"};
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    printUsage(argv[0]);
    return 2;
  }

  NetworkConfig config{};
  double duration = 0.0;
  try {
    config = parseArm(argv[1]);
    duration = parseDuration(argv[2]);
  } catch (const std::exception &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    printUsage(argv[0]);
    return 2;
  }

  std::signal(SIGINT, signalHandler);
  std::cout << std::fixed << std::setprecision(6);

  try {
    std::cout << "Connecting " << config.name << " arm: robot="
              << config.robotIp << ", local=" << config.localIp << std::endl;
    ArRobot robot(config.robotIp, config.localIp);
    DragGuard dragGuard(robot);
    error_code ec;

    const auto initialState = robot.operationState(ec);
    if (!checkError(ec, "read operation state")) return 1;
    if (initialState == OperationState::drag) {
      robot.disableDrag(ec);
      if (!checkError(ec, "disable existing drag mode")) return 1;
      std::cout << "Existing drag mode was disabled. Run the program again."
                << std::endl;
      return 0;
    }

    Toolset toolset = robot.toolset(ec);
    if (!checkError(ec, "read current toolset")) return 1;
    toolset.load.mass = kDefaultToolMass;
    toolset.load.cog = kDefaultToolCog;
    try {
      validateLoad(toolset.load);
    } catch (const std::exception &e) {
      std::cerr << "Configured tool load is unusable: " << e.what()
                << std::endl
                << "Correct the built-in tool load before enabling "
                   "Cartesian drag."
                << std::endl;
      return 1;
    }

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkError(ec, "switch to NrtCommand")) return 1;
    robot.setOperateMode(OperateMode::manual, ec);
    if (!checkError(ec, "set manual mode")) return 1;
    robot.setPowerState(false, ec);
    if (!checkError(ec, "power off")) return 1;

    ec.clear();
    robot.setToolset(toolset, ec);
    if (!checkError(ec, "set default tool load")) return 1;
    ec.clear();
    const Toolset storedToolset = robot.toolset(ec);
    if (!checkError(ec, "verify default tool load")) return 1;
    if (!loadMatchesDefaults(storedToolset.load)) {
      std::cerr << "Controller did not retain the default tool mass/CoG"
                << std::endl;
      printLoad(storedToolset.load);
      return 1;
    }
    std::cout << "Default tool load stored in controller:" << std::endl;
    printLoad(storedToolset.load);

    const auto startPose = robot.posture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read start TCP pose")) return 1;
    printPose("Start TCP pose in external reference frame:", startPose);

    std::cout
        << "WARNING: the program writes a fixed 0 kg tool load with CoG "
           "[0, 0, 0] m; use it only when no tool is attached.\n"
        << "Support the arm, clear the workspace and keep the E-stop ready.\n"
        << "The physical drag button must be held while moving."
        << std::endl;

    // Official SDK path: Cartesian free drag with the physical drag button.
    // Deliberately omit the optional direct-drag=true argument.
    robot.enableDrag(DragParameter::cartesianSpace,
                     DragParameter::freely, ec);
    if (!checkError(ec, "enable Cartesian-space drag")) {
      printControllerErrors(robot);
      return 1;
    }
    dragGuard.setEnabled(true);
    auto forceControl = robot.forceControl();

    std::cout << "Cartesian drag enabled. Waiting 2 s for mode switching..."
              << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Hold the physical drag button and move slowly. Release the "
                 "button immediately if motion is unexpected. Monitoring for "
              << duration << " s; press Ctrl+C to stop." << std::endl
              << "Software guards: 0.50 m translation, 90 deg rotation, "
                 "0.20 m/s TCP speed and 1.0 rad/s joint speed."
              << std::endl;

    const MonitorResult result =
        monitorDrag(robot, forceControl, startPose, duration);

    robot.disableDrag(ec);
    if (!checkError(ec, "disable Cartesian-space drag")) {
      printControllerErrors(robot);
      return 1;
    }
    dragGuard.setEnabled(false);

    std::cout << "Cartesian drag disabled: " << result.reason << std::endl;
    if (!result.safe) {
      std::cerr << "A software safety guard stopped the drag session. Check "
                   "the load, sensor zero and gravity parameters before retrying."
                << std::endl;
      return 1;
    }

    const auto finalPose = robot.posture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read final TCP pose")) return 1;
    printPose("Final TCP pose in external reference frame:", finalPose);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Cartesian drag program exception: " << e.what()
              << std::endl;
    return 1;
  }
}
