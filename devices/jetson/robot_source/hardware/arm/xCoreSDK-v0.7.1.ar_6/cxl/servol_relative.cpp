// Relative TCP translation in endInRef, streamed as flangeInBase ServoL poses.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void signalHandler(int) { interrupted = 1; }
void checked(error_code ec, const char* op) {
  if (ec) throw std::runtime_error(std::string(op) + ": " + ec.message());
}
double number(const char* s) {
  std::size_t n;
  double v = std::stod(s, &n);
  if (s[n] || !std::isfinite(v)) throw std::runtime_error("Invalid numeric argument");
  return v;
}
Eigen::Matrix4d matrix(const std::array<double, 6>& p) {
  std::array<double, 16> a{};
  Utils::postureToTransArray(p, a);
  Eigen::Matrix4d m;
  Utils::arrayToTransMatrix_all(a, m);
  if (!m.allFinite()) throw std::runtime_error("Invalid pose feedback");
  return m;
}
void printPose(const char* label, const std::array<double, 6>& p) {
  std::cout << label;
  for (double v : p) std::cout << ' ' << v;
  std::cout << " [m, rad]\n";
}
}

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "Usage: servol_relative <left|right> dx_m dy_m dz_m [duration_s]\n"
                 "External-reference TCP offsets; peak speed <= 50 mm/s.\n";
    return 2;
  }
  try {
    const std::string side(argv[1]);
    if (side != "left" && side != "right") throw std::runtime_error("Invalid arm");
    const Eigen::Vector3d delta(number(argv[2]), number(argv[3]), number(argv[4]));
    const double distance = delta.norm();
    if (distance < 1e-6 || distance > 0.30) throw std::runtime_error("Distance must be in (0, 0.30] m");
    // Quintic interpolation has maximum ds/du = 1.875.
    const double minimumDuration = std::max(1.0, 1.875 * distance / 0.05);
    const double duration = argc == 6 ? number(argv[5]) : minimumDuration;
    if (duration < minimumDuration || duration > 300)
      throw std::runtime_error("Duration must be >= " + std::to_string(minimumDuration) + " and <= 300 s");
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::cout << std::fixed << std::setprecision(6);
    ArRobot robot(side == "left" ? "192.168.1.160" : "192.168.2.160",
                  side == "left" ? "192.168.1.10" : "192.168.2.10");
    error_code ec;
    const auto state = robot.operationState(ec);
    checked(ec, "operationState");
    if (state != OperationState::idle) throw std::runtime_error("Robot must be idle");
    const auto start = robot.posture(CoordinateType::endInRef, ec);
    checked(ec, "endInRef");
    const auto base = robot.baseFrame(ec); checked(ec, "baseFrame");
    const auto tool = robot.toolset(ec); checked(ec, "toolset");
    const auto cart = robot.cartPosture(CoordinateType::endInRef, ec);
    checked(ec, "cartPosture");
    if (!std::isfinite(cart.elbow)) throw std::runtime_error("Invalid elbow");
    auto target = start;
    for (int i = 0; i < 3; ++i) target[i] += delta[i];
    const auto flangeStart = Utils::EndInRefToFlanInBase(base, tool, start);
    const auto flangeTarget = Utils::EndInRefToFlanInBase(base, tool, target);
    const auto anchor = matrix(flangeStart);
    const auto goal = matrix(flangeTarget);
    const Eigen::Vector3d translation = goal.block<3,1>(0,3) - anchor.block<3,1>(0,3);
    const auto measured = robot.posture(CoordinateType::flangeInBase, ec);
    checked(ec, "flangeInBase");
    if ((matrix(measured) - anchor).norm() > 0.002)
      throw std::runtime_error("baseFrame/toolset conversion disagrees with measured flange; aborting");
    printPose("Current endInRef:", start);
    printPose("Target endInRef:", target);
    printPose("Current flangeInBase:", flangeStart);
    printPose("Target flangeInBase:", flangeTarget);
    std::cout << "Duration=" << duration << " s, peak speed="
              << 1875 * distance / duration << " mm/s. Press Enter to move.\n";
    if (std::cin.get() != '\n' || interrupted) return 1;
    const auto recheck = robot.posture(CoordinateType::flangeInBase, ec);
    checked(ec, "recheck start");
    if ((matrix(recheck) - anchor).norm() > 0.002)
      throw std::runtime_error("Robot moved since preview; aborting");
    robot.setOperateMode(OperateMode::automatic, ec); checked(ec, "automatic");
    robot.setMotionControlMode(MotionControlMode::RtCommand, ec); checked(ec, "RtCommand");
    robot.setPowerState(true, ec); checked(ec, "power on");
    auto rt = robot.getRtMotionController().lock();
    if (!rt) throw std::runtime_error("No RT controller");
    bool receiving = false, servo = false, moving = false;
    auto cleanup = [&]() {
      bool ok = true;
      if (moving) {
        try { rt->stopMove(); } catch (const std::exception& e) {
          std::cerr << "stopMove failed: " << e.what() << '\n'; ok = false;
        }
        moving = false;
      }
      if (servo) { rt->stopServoJoint(); servo = false; }
      if (receiving) {
        try { robot.stopReceiveRobotState(); } catch (const std::exception& e) {
          std::cerr << e.what() << '\n'; ok = false;
        }
        receiving = false;
      }
      return ok;
    };
    try {
      std::array<double,16> identity{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      rt->setEndEffectorFrame(identity, ec); checked(ec, "RT flange frame");
      robot.startReceiveRobotState(std::chrono::milliseconds(1),
          {RtSupportedFields::tcpPose_m, RtSupportedFields::jointPos_m});
      receiving = true;
      rt->setServoJoint(0.02, 0.02, 0, ec); checked(ec, "setServoJoint");
      servo = true;
      moving = true;
      rt->startMove(RtControllerMode::cartesianPosition);
      CartesianPosition command;
      command.elbow = cart.elbow;
      command.hasElbow = true;
      const auto frames = static_cast<std::size_t>(std::ceil(duration / 0.02));
      bool reached = false;
      auto next = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i <= frames + 100; ++i) {
        if (interrupted) throw std::runtime_error("Interrupted");
        if (std::chrono::steady_clock::now() - next > std::chrono::milliseconds(100))
          throw std::runtime_error("Servo scheduling overrun");
        if (robot.updateRobotState(std::chrono::milliseconds(5)) == 0)
          throw std::runtime_error("Missing fresh feedback");
        std::array<double,16> feedback{};
        if (robot.getStateData(RtSupportedFields::tcpPose_m, feedback) != 0)
          throw std::runtime_error("Cannot read TCP feedback");
        Eigen::Matrix4d actual;
        Utils::arrayToTransMatrix_all(feedback, actual);
        if (!actual.allFinite()) throw std::runtime_error("Nonfinite feedback");
        std::array<double,7> joints{};
        if (robot.getStateData(RtSupportedFields::jointPos_m, joints) != 0)
          throw std::runtime_error("Cannot read joint feedback");
        // Conservative limits inherited from the local ServoL example.
        const std::array<double,7> lower{-3.10669,-2.0944,-3.10669,-1.0472,-3.10669,-1.0472,-1.0472};
        const std::array<double,7> upper{3.10669,2.0944,3.10669,2.53073,3.10669,1.0472,1.0472};
        for (std::size_t j=0; j<joints.size(); ++j)
          if (!std::isfinite(joints[j]) || joints[j] < lower[j]+0.20 || joints[j] > upper[j]-0.20)
            throw std::runtime_error("Joint limit margin reached");
        if ((actual.block<3,3>(0,0) - anchor.block<3,3>(0,0)).norm() > 0.05)
          throw std::runtime_error("Unexpected orientation change");
        const Eigen::Vector3d travelled = actual.block<3,1>(0,3) - anchor.block<3,1>(0,3);
        const double progress = travelled.dot(translation.normalized());
        const double lateral = (travelled - progress * translation.normalized()).norm();
        if (progress < -0.003 || progress > distance + 0.003 || lateral > 0.003)
          throw std::runtime_error("Reverse/overrun/path deviation detected");
        const double u = std::min(1.0, double(i) / frames);
        const double s = u*u*u*(10 + u*(-15 + 6*u));
        auto planned = anchor;
        planned.block<3,1>(0,3) += s * translation;
        if ((actual.block<3,1>(0,3) - planned.block<3,1>(0,3)).norm() > 0.015)
          throw std::runtime_error("Tracking error exceeds 15 mm");
        Utils::transMatrixToArray_all(planned, command.pos);
        rt->sendCommand(command);
        if (i >= frames && (actual.block<3,1>(0,3) - goal.block<3,1>(0,3)).norm() < 0.001) {
          reached = true; break;
        }
        next += std::chrono::milliseconds(20);
        std::this_thread::sleep_until(next);
      }
      if (!reached) throw std::runtime_error("Final target not reached within settling time");
      if (!cleanup()) return 1;
      const auto finalPose = robot.posture(CoordinateType::endInRef, ec);
      checked(ec, "final pose");
      printPose("Final endInRef:", finalPose);
      double error = 0;
      for (int i=0; i<3; ++i) error += std::pow(finalPose[i]-target[i], 2);
      if (!std::isfinite(error) || std::sqrt(error) > 0.002)
        throw std::runtime_error("Final external-reference target mismatch");
      std::cout << "Completed. Robot remains powered on.\n";
    } catch (...) { cleanup(); throw; }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ServoL relative failed: " << e.what() << '\n';
    return 1;
  }
}
