/**
 * @file movel_relative.cpp
 * @brief Move the AR arm by a Cartesian translation relative to its current
 *        end pose, using a non-realtime MoveL command.
 *
 * Usage:
 *   movel_relative <left|right> <dx_m> <dy_m> <dz_m>
 *
 * The XYZ offsets are expressed in the configured external reference frame.
 * The current orientation, elbow angle and configuration are preserved.
 */

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

#include "rokae/robot.h"

using namespace rokae;

namespace {

constexpr double kMaxOffset = 0.30;  // Maximum combined XYZ offset, metres.
constexpr double kSpeed = 50.0;       // MoveL linear speed, mm/s.
constexpr double kZone = 0.0;        // Reach the target without blending.
constexpr double kGoalPositionTolerance = 0.002;  // metres.
constexpr double kTravelOverrunTolerance = 0.003;  // metres.
constexpr auto kMotionStartTimeout = std::chrono::seconds(5);

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

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
                                " must be a finite number in metres");
  }
  return value;
}

void printPose(const char *label, const CartesianPosition &pose) {
  std::cout << label << std::endl
            << "  XYZ = [" << pose.trans[0] << ", " << pose.trans[1]
            << ", " << pose.trans[2] << "] m" << std::endl
            << "  RPY = [" << pose.rpy[0] << ", " << pose.rpy[1] << ", "
            << pose.rpy[2] << "] rad" << std::endl
            << "  elbow = " << pose.elbow << " rad" << std::endl;
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

double positionError(const CartesianPosition &actual,
                     const CartesianPosition &target) {
  return std::sqrt(std::pow(actual.trans[0] - target.trans[0], 2) +
                   std::pow(actual.trans[1] - target.trans[1], 2) +
                   std::pow(actual.trans[2] - target.trans[2], 2));
}

bool waitRobotAtTarget(
    ArRobot &robot, const CartesianPosition &start,
    const CartesianPosition &target, const std::array<double, 3> &offset,
    std::chrono::seconds timeout = std::chrono::seconds(60)) {
  const auto commandStart = std::chrono::steady_clock::now();
  const auto motionStartDeadline = commandStart + kMotionStartTimeout;
  const auto deadline = commandStart + timeout;
  bool motionObserved = false;

  // moveStart() may return before the controller changes its operation state.
  // Do not treat that initial idle state as motion completion.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    error_code ec;
    const auto state = robot.operationState(ec);
    if (!checkError(ec, "read operation state")) return false;
    if (state == OperationState::unknown) {
      std::cerr << "Robot entered unknown operation state" << std::endl;
      return false;
    }

    const auto pose = robot.cartPosture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read Cartesian pose while waiting")) return false;
    const double error = positionError(pose, target);
    const auto now = std::chrono::steady_clock::now();

    const double requestedDistance = std::sqrt(
        offset[0] * offset[0] + offset[1] * offset[1] +
        offset[2] * offset[2]);
    const std::array<double, 3> travelled = {
        pose.trans[0] - start.trans[0], pose.trans[1] - start.trans[1],
        pose.trans[2] - start.trans[2]};
    const double progress =
        (travelled[0] * offset[0] + travelled[1] * offset[1] +
         travelled[2] * offset[2]) /
        requestedDistance;
    const double travelledDistance =
        std::sqrt(travelled[0] * travelled[0] +
                  travelled[1] * travelled[1] +
                  travelled[2] * travelled[2]);

    // Do not rely only on operationState: if feedback passes the requested
    // endpoint or leaves the requested straight-line travel envelope, return
    // immediately so the caller issues stop() and moveReset().
    if (progress < -kTravelOverrunTolerance ||
        progress > requestedDistance + kTravelOverrunTolerance ||
        travelledDistance > requestedDistance + kTravelOverrunTolerance) {
      std::cerr << "MoveL exceeded the requested travel; progress="
                << progress * 1000.0
                << " mm, total travel=" << travelledDistance * 1000.0
                << " mm. Stopping motion." << std::endl;
      return false;
    }

    if (state != OperationState::idle) {
      motionObserved = true;
    } else if (error <= kGoalPositionTolerance) {
      return true;
    } else if (motionObserved) {
      std::cerr << "Robot became idle before reaching the MoveL target; "
                << "position error=" << error * 1000.0 << " mm"
                << std::endl;
      return false;
    } else if (now >= motionStartDeadline) {
      std::cerr << "MoveL did not enter a running state within "
                << std::chrono::duration<double>(kMotionStartTimeout).count()
                << " s; position error=" << error * 1000.0 << " mm"
                << std::endl;
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::cerr << (gStopRequested ? "Ctrl+C received" : "MoveL timed out")
            << std::endl;
  return false;
}

void stopMotion(ArRobot &robot) {
  error_code ec;
  robot.stop(ec);
  if (ec) {
    std::cerr << "stop failed: " << ec.message() << std::endl;
    return;
  }

  const auto stopDeadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < stopDeadline) {
    ec.clear();
    const auto state = robot.operationState(ec);
    if (ec) {
      std::cerr << "read operation state after stop failed: " << ec.message()
                << std::endl;
      return;
    }
    if (state == OperationState::idle || state == OperationState::unknown) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  ec.clear();
  const auto finalState = robot.operationState(ec);
  if (ec) {
    std::cerr << "read final operation state after stop failed: "
              << ec.message() << std::endl;
    return;
  }
  if (finalState != OperationState::idle &&
      finalState != OperationState::unknown) {
    std::cerr << "robot did not stop within 10 s; motion queue was not reset"
              << std::endl;
    return;
  }

  ec.clear();
  robot.moveReset(ec);
  if (ec) {
    std::cerr << "moveReset after stop failed: " << ec.message()
              << std::endl;
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <left|right> <dx_m> <dy_m> <dz_m>" << std::endl;
    return 2;
  }

  NetworkConfig config{};
  std::array<double, 3> offset{};
  try {
    config = parseArm(argv[1]);
    offset = {parseNumber(argv[2], "dx"), parseNumber(argv[3], "dy"),
              parseNumber(argv[4], "dz")};
  } catch (const std::exception &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return 2;
  }

  const double offsetNorm = std::sqrt(offset[0] * offset[0] +
                                      offset[1] * offset[1] +
                                      offset[2] * offset[2]);
  if (offsetNorm < 1e-6) {
    std::cerr << "The requested offset is effectively zero" << std::endl;
    return 2;
  }
  if (offsetNorm > kMaxOffset) {
    std::cerr << "The combined offset is " << offsetNorm
              << " m, exceeding the " << kMaxOffset << " m safety limit"
              << std::endl;
    return 2;
  }

  std::signal(SIGINT, signalHandler);
  std::cout << std::fixed << std::setprecision(6);

  try {
    std::cout << "Connecting " << config.name << " arm: robot="
              << config.robotIp << ", local=" << config.localIp << std::endl;
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
      std::cerr << "Robot must be idle before MoveL; current state="
                << static_cast<int>(state) << std::endl;
      return 1;
    }

    CartesianPosition start =
        robot.cartPosture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read current Cartesian pose")) return 1;
    // Use the measured endInRef pose directly and change only XYZ. This keeps
    // the controller's current orientation, elbow and configuration while
    // making the commanded endpoint exactly match the printed target.
    CartesianPosition target = start;
    for (std::size_t i = 0; i < offset.size(); ++i) {
      target.trans[i] += offset[i];
    }
    if (!std::isfinite(target.elbow)) {
      std::cerr << "Current elbow feedback is invalid" << std::endl;
      return 1;
    }
    target.hasElbow = true;

    printPose("Current end pose in external reference frame:", start);
    std::cout << "Requested XYZ offset = [" << offset[0] << ", "
              << offset[1] << ", " << offset[2] << "] m" << std::endl;
    printPose("MoveL target pose:", target);

    const auto currentJoints = robot.jointPos(ec);
    if (!checkError(ec, "read current joints for path validation")) return 1;
    const std::vector<double> startJoints(
        currentJoints.begin(), currentJoints.end());
    ec.clear();
    const auto targetJoints = robot.checkPath(start, startJoints, target, ec);
    if (!checkError(ec, "validate MoveL path")) {
      printControllerErrors(robot);
      return 1;
    }
    if (targetJoints.size() != currentJoints.size()) {
      std::cerr << "MoveL path validation returned " << targetJoints.size()
                << " joints; expected " << currentJoints.size() << std::endl;
      return 1;
    }
    std::cout << "MoveL path validation passed." << std::endl;

    std::cout << "MoveL speed=" << kSpeed << " mm/s, zone=" << kZone
              << ". Verify the path is clear, then press Enter to move."
              << std::endl;
    std::cin.get();
    if (gStopRequested) return 0;

    robot.setDefaultConfOpt(false, ec);
    if (!checkError(ec, "disable default configuration selection")) return 1;
    robot.moveReset(ec);
    if (!checkError(ec, "moveReset")) return 1;

    MoveLCommand command(target, kSpeed, kZone);
    std::string commandId;
    robot.moveAppend({command}, commandId, ec);
    if (!checkError(ec, "moveAppend(MoveL)")) {
      printControllerErrors(robot);
      return 1;
    }
    std::cout << "MoveL command ID: " << commandId << std::endl;

    robot.moveStart(ec);
    if (!checkError(ec, "moveStart(MoveL)")) {
      printControllerErrors(robot);
      return 1;
    }
    if (!waitRobotAtTarget(robot, start, target, offset)) {
      stopMotion(robot);
      printControllerErrors(robot);
      return 1;
    }

    const CartesianPosition reached =
        robot.cartPosture(CoordinateType::endInRef, ec);
    if (!checkError(ec, "read final Cartesian pose")) return 1;
    printPose("Reached end pose:", reached);

    const double finalPositionError = positionError(reached, target);
    std::cout << "Final XYZ error = " << finalPositionError * 1000.0 << " mm"
              << std::endl;
    if (finalPositionError > kGoalPositionTolerance) {
      std::cerr << "MoveL ended outside the "
                << kGoalPositionTolerance * 1000.0
                << " mm goal tolerance" << std::endl;
      printControllerErrors(robot);
      return 1;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "MoveL program exception: " << e.what() << std::endl;
    return 1;
  }
}
