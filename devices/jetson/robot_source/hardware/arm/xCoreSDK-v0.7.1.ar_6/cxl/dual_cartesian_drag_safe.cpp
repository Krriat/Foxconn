/**
 * @file dual_cartesian_drag_safe.cpp
 * @brief Enable Cartesian hand-guiding on both AR arms in one session.
 *
 * Usage: dual_cartesian_drag_safe <duration_seconds>
 *
 * Follows cartesian_drag_safe.cpp: writes a 0 kg load / zero CoG for each
 * arm, and requires each arm's physical drag button. Use with no tool attached.
 * This enables hand-guiding and prints feedback; it does not record a path.
 * Any feedback/guard failure ends the session and attempts to disable both
 * arms. Software monitoring cannot replace controller safety or an E-stop.
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

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kPi = 3.14159265358979323846;
constexpr double kDefaultToolMass = 0.0;
constexpr std::array<double, 3> kDefaultToolCog{0.0, 0.0, 0.0};
// Load guards follow the single-arm source; motion guards are widened 3x.
constexpr double kMaxToolMass = 5.0;
constexpr double kMaxCogDistance = 0.60;
constexpr double kMaxTranslation = 1.50;
// The shortest relative rotation is at most 180 deg, so 270 deg never trips.
constexpr double kMaxRotation = 270.0 * kPi / 180.0;
constexpr double kMaxLinearSpeed = 3.0;
constexpr double kMaxJointSpeed = 6.0;
constexpr int kSpeedViolationSamples = 2;
constexpr auto kMonitorPeriod = std::chrono::milliseconds(50);
constexpr auto kTcpPrintPeriod = std::chrono::milliseconds(200);

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) { gStopRequested = 1; }

template <std::size_t N>
bool allFinite(const std::array<double, N> &values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

Eigen::Matrix3d rotationFromPose(const std::array<double, 6> &pose) {
  std::array<double, 16> transform{};
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  Utils::postureToTransArray(pose, transform);
  Utils::arrayToTransMatrix(transform, rotation, translation);
  return rotation;
}

double parseDuration(const char *text) {
  std::size_t parsed = 0;
  const std::string input(text);
  const double duration = std::stod(input, &parsed);
  if (parsed != input.size() || !std::isfinite(duration) ||
      duration < 1.0 || duration > 600.0) {
    throw std::invalid_argument("duration must be between 1 and 600 seconds");
  }
  return duration;
}

class ArmSession {
 public:
  ArmSession(const char *name, const char *robotIp, const char *localIp)
      : name_(name), robot_(robotIp, localIp),
        forceControl_(robot_.forceControl()) {
    std::cout << '[' << name_ << "] Connected: robot=" << robotIp
              << ", local=" << localIp << std::endl;
  }

  ~ArmSession() { stop(); }
  ArmSession(const ArmSession &) = delete;
  ArmSession &operator=(const ArmSession &) = delete;

  bool detectExistingDrag() {
    error_code ec;
    const auto state = robot_.operationState(ec);
    require(ec, "read operation state");
    needsCleanup_ = state == OperationState::drag;
    return needsCleanup_;
  }

  void prepare() {
    if (gStopRequested) fail("stop requested during setup");
    error_code ec;
    Toolset toolset = robot_.toolset(ec);
    require(ec, "read toolset");
    toolset.load.mass = kDefaultToolMass;
    toolset.load.cog = kDefaultToolCog;
    validateLoad(toolset.load);

    robot_.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    require(ec, "switch to NrtCommand");
    robot_.setOperateMode(OperateMode::manual, ec);
    require(ec, "set manual mode");
    robot_.setPowerState(false, ec);
    require(ec, "power off");
    robot_.setToolset(toolset, ec);
    require(ec, "set default tool load");
    const auto storedToolset = robot_.toolset(ec);
    require(ec, "verify default tool load");
    validateLoad(storedToolset.load);
    constexpr double kTolerance = 1e-6;
    if (std::abs(storedToolset.load.mass - kDefaultToolMass) > kTolerance) {
      fail("controller did not retain the default tool mass");
    }
    for (std::size_t i = 0; i < kDefaultToolCog.size(); ++i) {
      if (std::abs(storedToolset.load.cog[i] - kDefaultToolCog[i]) >
          kTolerance) {
        fail("controller did not retain the default tool CoG");
      }
    }
    std::cout << '[' << name_ << "] Stored load: mass="
              << storedToolset.load.mass << " kg, CoG=["
              << storedToolset.load.cog[0] << ", "
              << storedToolset.load.cog[1] << ", "
              << storedToolset.load.cog[2] << "] m" << std::endl;

    startPose_ = robot_.posture(CoordinateType::endInRef, ec);
    require(ec, "read start TCP pose");
    previousJoints_ = robot_.jointPos(ec);
    require(ec, "read initial joint positions");
    if (!allFinite(startPose_) || !allFinite(previousJoints_)) {
      fail("initial feedback contains a non-finite value");
    }
    startRotation_ = rotationFromPose(startPose_);
    previousPose_ = startPose_;
    previousTime_ = Clock::now();
    previousTcpPrint_ = previousTime_ - kTcpPrintPeriod;
    printPose("Start", startPose_);
  }

  void enable() {
    if (gStopRequested) fail("stop requested before enabling drag");
    // Track the attempt too, so a partially successful enable is cleaned up.
    needsCleanup_ = true;
    error_code ec;
    robot_.enableDrag(DragParameter::cartesianSpace, DragParameter::freely, ec);
    require(ec, "enable Cartesian drag");
    std::cout << '[' << name_ << "] Cartesian drag enabled." << std::endl;
  }

  bool stop() noexcept {
    if (!needsCleanup_) return true;
    error_code ec;
    robot_.disableDrag(ec);
    if (ec) {
      std::cerr << '[' << name_ << "] disable drag failed: code="
                << ec.value() << ", message=" << ec.message() << std::endl;
      return false;
    }
    needsCleanup_ = false;
    std::cout << '[' << name_ << "] Cartesian drag disabled." << std::endl;
    return true;
  }

  void monitor() {
    error_code ec;
    const auto pose = robot_.posture(CoordinateType::endInRef, ec);
    require(ec, "read TCP pose");
    const auto joints = robot_.jointPos(ec);
    require(ec, "read joint positions");
    const auto now = Clock::now();
    if (!allFinite(pose) || !allFinite(joints)) {
      fail("feedback contains a non-finite value");
    }

    const Eigen::Vector3d position(pose[0], pose[1], pose[2]);
    const Eigen::Vector3d startPosition(startPose_[0], startPose_[1], startPose_[2]);
    if ((position - startPosition).norm() > kMaxTranslation) {
      fail("TCP moved more than 1.50 m from the start pose");
    }
    const Eigen::Matrix3d relativeRotation =
        rotationFromPose(pose) * startRotation_.transpose();
    if (std::abs(Eigen::AngleAxisd(relativeRotation).angle()) > kMaxRotation) {
      fail("TCP rotated more than 270 deg from the start pose");
    }

    const double dt = std::chrono::duration<double>(now - previousTime_).count();
    if (dt <= 0.0) fail("invalid feedback sample interval");
    const Eigen::Vector3d previousPosition(
        previousPose_[0], previousPose_[1], previousPose_[2]);
    const double linearSpeed = (position - previousPosition).norm() / dt;
    double fastestJointSpeed = 0.0;
    std::size_t fastestJoint = 0;
    for (std::size_t i = 0; i < joints.size(); ++i) {
      const double speed = std::abs(joints[i] - previousJoints_[i]) / dt;
      if (speed > fastestJointSpeed) {
        fastestJointSpeed = speed;
        fastestJoint = i;
      }
    }
    if (linearSpeed > kMaxLinearSpeed || fastestJointSpeed > kMaxJointSpeed) {
      ++speedViolationCount_;
    } else {
      speedViolationCount_ = 0;
    }
    if (speedViolationCount_ >= kSpeedViolationSamples) {
      if (linearSpeed > kMaxLinearSpeed) {
        fail("TCP speed exceeded " + std::to_string(kMaxLinearSpeed) + " m/s");
      }
      fail("J" + std::to_string(fastestJoint + 1) + " speed exceeded " +
           std::to_string(kMaxJointSpeed) + " rad/s");
    }

    if (now - previousTcpPrint_ >= kTcpPrintPeriod) {
      std::array<double, 7> jointTorque{}, externalTorque{};
      std::array<double, 3> tcpTorque{}, tcpForce{};
      forceControl_.getEndTorque(FrameType::tool, jointTorque, externalTorque,
                                 tcpTorque, tcpForce, ec);
      require(ec, "read TCP wrench");
      if (!allFinite(tcpForce) || !allFinite(tcpTorque)) {
        fail("TCP wrench contains a non-finite value");
      }
      printPose("TCP", pose);
      std::cout << '[' << name_ << "] F[N]=["
                << tcpForce[0] << ", " << tcpForce[1] << ", " << tcpForce[2]
                << "] M[Nm]=[" << tcpTorque[0] << ", " << tcpTorque[1]
                << ", " << tcpTorque[2] << ']' << std::endl;
      previousTcpPrint_ = now;
    }
    previousPose_ = pose;
    previousJoints_ = joints;
    previousTime_ = now;
  }

  void printFinalPose() {
    error_code ec;
    const auto pose = robot_.posture(CoordinateType::endInRef, ec);
    require(ec, "read final TCP pose");
    if (!allFinite(pose)) fail("final TCP pose contains a non-finite value");
    printPose("Final", pose);
  }

 private:
  [[noreturn]] void fail(const std::string &message) const {
    throw std::runtime_error("[" + name_ + "] " + message);
  }

  void require(const error_code &ec, const char *operation) const {
    if (ec) {
      fail(std::string(operation) + " failed: code=" +
           std::to_string(ec.value()) + ", message=" + ec.message());
    }
  }

  void validateLoad(const Load &load) const {
    if (!std::isfinite(load.mass) || load.mass < 0.0 ||
        load.mass > kMaxToolMass || !allFinite(load.cog)) {
      fail("configured tool mass/CoG is invalid");
    }
    double squaredDistance = 0.0;
    for (const double value : load.cog) squaredDistance += value * value;
    if (std::sqrt(squaredDistance) > kMaxCogDistance) {
      fail("configured tool CoG is more than 0.60 m from the flange");
    }
  }

  void printPose(const char *label, const std::array<double, 6> &pose) const {
    std::cout << '[' << name_ << "] " << label << " XYZ[m]=["
              << pose[0] << ", " << pose[1] << ", " << pose[2]
              << "] RPY[rad]=[" << pose[3] << ", " << pose[4] << ", "
              << pose[5] << ']' << std::endl;
  }

  std::string name_;
  ArRobot robot_;
  ForceControl_T<7> forceControl_;
  bool needsCleanup_{false};
  std::array<double, 6> startPose_{}, previousPose_{};
  std::array<double, 7> previousJoints_{};
  Eigen::Matrix3d startRotation_{};
  Clock::time_point previousTime_{}, previousTcpPrint_{};
  int speedViolationCount_{0};
};

bool stopBoth(ArmSession &left, ArmSession &right) {
  // Do not short-circuit: always attempt to disable both arms.
  const bool leftStopped = left.stop();
  const bool rightStopped = right.stop();
  return leftStopped && rightStopped;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "Usage: " << argv[0] << " <duration_seconds: 1..600>\n"
              << "Both arms; hold each physical drag button. No tool attached.\n"
              << "Prints feedback only; does not record a path." << std::endl;
    return 0;
  }
  double duration = 0.0;
  try {
    if (argc != 2) throw std::invalid_argument("expected duration_seconds");
    duration = parseDuration(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "Invalid argument: " << e.what() << '\n'
              << "Usage: " << argv[0] << " <duration_seconds: 1..600>" << std::endl;
    return 2;
  }

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "WARNING: this program writes a 0 kg tool load and zero CoG "
               "to BOTH controllers. Use only with no tool attached.\n"
            << "Support both arms, clear the workspace and keep the E-stop ready.\n"
            << "Each arm requires its physical drag button to be held." << std::endl;
  try {
    // Connect both before configuring or enabling either arm.
    ArmSession left("left", "192.168.1.160", "192.168.1.10");
    ArmSession right("right", "192.168.2.160", "192.168.2.10");
    const bool leftDragging = left.detectExistingDrag();
    const bool rightDragging = right.detectExistingDrag();
    if (leftDragging || rightDragging) {
      if (!stopBoth(left, right)) return 1;
      std::cout << "Existing drag mode disabled. Run the program again." << std::endl;
      return 0;
    }
    left.prepare();
    right.prepare();
    left.enable();
    right.enable();

    std::cout << "Both arms enabled; allow 2 s for mode switching.\n"
              << "Software guards per arm: " << kMaxTranslation
              << " m translation, 270 deg rotation, " << kMaxLinearSpeed
              << " m/s TCP speed, " << kMaxJointSpeed << " rad/s joint speed.\n"
              << "Session duration: " << duration << " s after mode switching; "
              << "press Ctrl+C to stop both arms." << std::endl;
    const auto settlingEnd = Clock::now() + std::chrono::seconds(2);
    const auto deadline = settlingEnd + std::chrono::duration_cast<Clock::duration>(
                                          std::chrono::duration<double>(duration));
    // Monitor during mode switching too; there is no unmonitored startup sleep.
    while (!gStopRequested && Clock::now() < deadline) {
      const auto nextSample = Clock::now() + kMonitorPeriod;
      left.monitor();
      if (gStopRequested) break;
      right.monitor();
      std::this_thread::sleep_until(std::min(nextSample, deadline));
    }
    if (!stopBoth(left, right)) return 1;
    std::cout << "Dual-arm drag ended: "
              << (gStopRequested ? "stop requested" : "requested duration reached")
              << std::endl;
    left.printFinalPose();
    right.printFinalPose();
    return 0;
  } catch (const std::exception &e) {
    // ArmSession destructors attempt both disables before entering this handler.
    std::cerr << "Dual-arm drag stopped: " << e.what() << std::endl;
    return 1;
  }
}
