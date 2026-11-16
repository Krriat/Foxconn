#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;

namespace {

constexpr uint16_t kUdpPort = 5050;
constexpr std::size_t kPacketSize = 38;
constexpr double kPi = 3.14159265358979323846;
constexpr double kPlanPeriod = 0.02;  // 50 Hz ServoL command period.

constexpr double kPositionFilterAlpha = 0.20;
constexpr double kRotationFilterAlpha = 0.15;
constexpr double kPositionGain = 0.30;
constexpr double kRotationGain = 0.25;
constexpr double kPositionDeadband = 0.0005;
constexpr double kRotationDeadband = 0.3 * kPi / 180.0;
constexpr double kMaxPositionStep = 0.001;
constexpr double kMaxRotationStep = 0.5 * kPi / 180.0;
constexpr double kMaxRawPositionJump = 0.10;
constexpr double kMaxRawRotationJump = 45.0 * kPi / 180.0;
constexpr double kMaxRawPositionOffset = 0.40;
constexpr double kMaxRawRotationOffset = 120.0 * kPi / 180.0;
constexpr auto kUdpTimeout = std::chrono::milliseconds(250);
constexpr auto kMaxCycleLateness = std::chrono::milliseconds(40);
constexpr std::size_t kMaxRejectedCycles = 100;  // 2 s at 50 Hz.

constexpr double kJointLimitMargin = 12.0 * kPi / 180.0;
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

enum class Side : std::size_t { left = 0, right = 1 };

const std::array<Side, 2> kSides = {Side::left, Side::right};

std::size_t sideIndex(Side side) {
  return static_cast<std::size_t>(side);
}

const char *sideName(Side side) {
  return side == Side::left ? "left" : "right";
}

struct ArmConfig {
  Side side;
  const char *robotIp;
  const char *localIp;
  std::array<double, 7> resetJoint;
};

const ArmConfig kLeftArmConfig = {
    Side::left,
    "192.168.1.160",
    "192.168.1.10",
    {1.167045, 1.300690, -1.524894, 0.588013,
     -0.254441, 0.509970, -0.364911},
};

const ArmConfig kRightArmConfig = {
    Side::right,
    "192.168.2.160",
    "192.168.2.10",
    {-1.652252, 1.369718, 1.524893, 0.984033,  0.261897, 0.196173, -0.145317},
};

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

struct HandPacket {
  Side side{Side::left};
  std::array<double, 3> positionMm{};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  bool grip{false};
  bool down{false};
};

struct HandState {
  std::array<double, 3> positionM{};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  std::array<double, 3> rawPositionM{};
  Eigen::Quaterniond rawOrientation{Eigen::Quaterniond::Identity()};
  bool grip{false};
  bool down{false};
  bool valid{false};
  bool rawValid{false};
  std::uint64_t sequence{0};
  std::chrono::steady_clock::time_point timestamp{};
};

struct ArmSession {
  const ArmConfig *config{nullptr};
  std::unique_ptr<ArRobot> robot;
  std::array<double, 7> activeLowerLimit{};
  std::array<double, 7> activeUpperLimit{};
  bool jointLimitsReady{false};
};

double clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

float readFloatLe(const std::uint8_t *data) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(data[0]) |
      (static_cast<std::uint32_t>(data[1]) << 8U) |
      (static_cast<std::uint32_t>(data[2]) << 16U) |
      (static_cast<std::uint32_t>(data[3]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void writeFloatLe(std::uint8_t *data, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  data[0] = static_cast<std::uint8_t>(bits & 0xffU);
  data[1] = static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
  data[2] = static_cast<std::uint8_t>((bits >> 16U) & 0xffU);
  data[3] = static_cast<std::uint8_t>((bits >> 24U) & 0xffU);
}

bool finiteQuaternion(const Eigen::Quaterniond &q) {
  return std::isfinite(q.w()) && std::isfinite(q.x()) &&
         std::isfinite(q.y()) && std::isfinite(q.z());
}

double quaternionAngularDistance(const Eigen::Quaterniond &a,
                                 const Eigen::Quaterniond &b) {
  const double dot = clamp(std::abs(a.normalized().dot(b.normalized())),
                           -1.0, 1.0);
  return 2.0 * std::acos(dot);
}

// Keep the same PICO-to-robot orientation convention as pico_servol.cpp.
Eigen::Quaterniond convertPicoOrientation(double qx, double qy, double qz,
                                          double qw) {
  Eigen::Quaterniond raw(qw, qx, qy, qz);
  if (!finiteQuaternion(raw) || raw.norm() < 1e-8) {
    return Eigen::Quaterniond::Identity();
  }
  raw.normalize();

  qx = raw.x();
  qy = raw.y();
  qz = raw.z();
  qw = raw.w();

  const double roll = std::atan2(2.0 * (qw * qx + qy * qz),
                                 1.0 - 2.0 * (qx * qx + qy * qy));
  const double sinPitch = clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0);
  const double pitch = std::asin(sinPitch);
  const double yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                1.0 - 2.0 * (qy * qy + qz * qz));

  const double rx = pitch;
  const double ry = roll - kPi / 4.0;
  const double rz = -yaw - kPi / 2.0;
  const Eigen::Matrix3d rotation =
      (Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  return Eigen::Quaterniond(rotation).normalized();
}

bool parseHandPacket(const std::uint8_t *data, std::size_t size,
                     HandPacket &packet) {
  if (size != kPacketSize || data[0] > 1U) return false;

  packet.side = data[0] == 0U ? Side::left : Side::right;
  std::array<double, 9> values{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = readFloatLe(data + 1 + i * sizeof(float));
    if (!std::isfinite(values[i])) return false;
  }

  packet.positionMm = {values[0], values[1], values[2]};
  const Eigen::Quaterniond rawOrientation(
      values[6], values[3], values[4], values[5]);
  if (!finiteQuaternion(rawOrientation) ||
      rawOrientation.norm() < 1e-8) {
    return false;
  }
  packet.orientation =
      convertPicoOrientation(values[3], values[4], values[5], values[6]);
  const std::uint8_t buttons = data[37];
  packet.grip = (buttons & (1U << 1U)) != 0;
  packet.down = (buttons & (1U << 2U)) != 0;
  return true;
}

bool parserSelfTest() {
  const std::array<float, 9> values = {
      172.14F, -178.71F, -221.11F, 0.0F, 0.0F,
      0.0F,   1.0F,     0.0F,     0.0F,
  };

  bool allOk = true;
  for (const Side side : kSides) {
    std::array<std::uint8_t, kPacketSize> data{};
    data[0] = static_cast<std::uint8_t>(sideIndex(side));
    for (std::size_t i = 0; i < values.size(); ++i) {
      writeFloatLe(data.data() + 1 + i * sizeof(float), values[i]);
    }
    data[37] = static_cast<std::uint8_t>((1U << 1U) | (1U << 2U));

    HandPacket packet;
    const bool ok = parseHandPacket(data.data(), data.size(), packet) &&
                    packet.side == side &&
                    std::abs(packet.positionMm[0] - values[0]) < 1e-3 &&
                    std::abs(packet.positionMm[1] - values[1]) < 1e-3 &&
                    std::abs(packet.positionMm[2] - values[2]) < 1e-3 &&
                    packet.grip && packet.down &&
                    finiteQuaternion(packet.orientation);
    std::cout << "UDP parser self-test (" << sideName(side)
              << "): " << (ok ? "PASS" : "FAIL") << std::endl;
    allOk = allOk && ok;
  }
  return allOk;
}

class UdpReceiver {
 public:
  explicit UdpReceiver(uint16_t port) : port_(port) {}

  ~UdpReceiver() { stop(); }

  void start() {
    running_.store(true);
    thread_ = std::thread(&UdpReceiver::run, this);
  }

  void stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
  }

  HandState latest(Side side) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_[sideIndex(side)];
  }

  std::array<HandState, 2> latestBoth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_;
  }

  std::string error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
  }

 private:
  void setError(const std::string &message) {
    std::lock_guard<std::mutex> lock(mutex_);
    error_ = message;
  }

  void update(const HandPacket &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    HandState &state = states_[sideIndex(packet.side)];
    state.grip = packet.grip;
    state.down = packet.down;

    std::array<double, 3> rawPositionM{};
    for (std::size_t i = 0; i < rawPositionM.size(); ++i) {
      rawPositionM[i] = packet.positionMm[i] / 1000.0;
    }

    // While Grip is released the controller is not allowed to move the robot,
    // so it is safe to accept a new tracking origin. This also lets the
    // operator recover from a PICO re-localization without restarting the
    // program. While Grip is held, a large one-frame jump is rejected.
    if (state.rawValid && packet.grip) {
      const Eigen::Vector3d previous(
          state.rawPositionM[0], state.rawPositionM[1],
          state.rawPositionM[2]);
      const Eigen::Vector3d current(
          rawPositionM[0], rawPositionM[1], rawPositionM[2]);
      const double positionJump = (current - previous).norm();
      const double rotationJump =
          quaternionAngularDistance(state.rawOrientation,
                                    packet.orientation);
      if (positionJump > kMaxRawPositionJump ||
          rotationJump > kMaxRawRotationJump) {
        // Do not refresh the pose timestamp. A persistent tracking jump will
        // therefore stop teleoperation through the existing UDP stale guard.
        // Grip release is still copied above and always remains able to stop.
        return;
      }
    }

    state.rawPositionM = rawPositionM;
    state.rawOrientation = packet.orientation;
    state.rawValid = true;

    if (!state.valid) {
      for (std::size_t i = 0; i < 3; ++i) {
        state.positionM[i] = rawPositionM[i];
      }
      state.orientation = packet.orientation;
      state.valid = true;
    } else {
      for (std::size_t i = 0; i < 3; ++i) {
        state.positionM[i] +=
            kPositionFilterAlpha *
            (rawPositionM[i] - state.positionM[i]);
      }
      Eigen::Quaterniond target = packet.orientation;
      if (state.orientation.dot(target) < 0.0) target.coeffs() *= -1.0;
      state.orientation =
          state.orientation.slerp(kRotationFilterAlpha, target).normalized();
    }

    state.timestamp = std::chrono::steady_clock::now();
    ++state.sequence;
  }

  void run() {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
      setError("Failed to create UDP socket");
      return;
    }

    const int reuse = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    timeval timeout{};
    timeout.tv_usec = 100000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port_);
    if (::bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) <
        0) {
      setError("Failed to bind UDP port 5050; stop receive.py first");
      ::close(fd);
      return;
    }

    std::array<std::uint8_t, 1024> buffer{};
    std::array<bool, 2> sourcePrinted{false, false};
    while (running_.load() && !gStopRequested) {
      sockaddr_in sender{};
      socklen_t senderLength = sizeof(sender);
      const ssize_t length =
          ::recvfrom(fd, buffer.data(), buffer.size(), 0,
                     reinterpret_cast<sockaddr *>(&sender), &senderLength);
      if (length < 0) continue;

      HandPacket packet;
      if (!parseHandPacket(buffer.data(), static_cast<std::size_t>(length),
                           packet)) {
        continue;
      }
      update(packet);

      const std::size_t index = sideIndex(packet.side);
      if (!sourcePrinted[index]) {
        char ip[INET_ADDRSTRLEN]{};
        ::inet_ntop(AF_INET, &sender.sin_addr, ip, sizeof(ip));
        std::cout << "Received " << sideName(packet.side)
                  << " controller UDP data from " << ip << ":"
                  << ntohs(sender.sin_port) << std::endl;
        sourcePrinted[index] = true;
      }
    }
    ::close(fd);
  }

  uint16_t port_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  std::array<HandState, 2> states_{};
  std::string error_;
};

bool checkEc(const error_code &ec, const std::string &operation) {
  if (!ec) return true;
  std::cerr << operation << " failed: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

bool loadActiveJointLimits(ArmSession &arm) {
  arm.jointLimitsReady = false;
  std::array<double[2], 7> controllerLimits{};
  error_code ec;
  const bool softLimitsEnabled =
      arm.robot->getSoftLimit(controllerLimits, ec);
  if (!checkEc(ec, std::string(sideName(arm.config->side)) +
                       " getSoftLimit")) {
    return false;
  }
  if (!softLimitsEnabled) {
    std::cerr << sideName(arm.config->side)
              << " controller soft limits are disabled; refusing "
                 "teleoperation"
              << std::endl;
    return false;
  }

  for (std::size_t i = 0; i < arm.activeLowerLimit.size(); ++i) {
    const double controllerLower =
        controllerLimits[i][0] + kJointLimitMargin;
    const double controllerUpper =
        controllerLimits[i][1] - kJointLimitMargin;
    arm.activeLowerLimit[i] =
        std::max(kLowerLimit[i], controllerLower);
    arm.activeUpperLimit[i] =
        std::min(kUpperLimit[i], controllerUpper);
    if (!std::isfinite(arm.activeLowerLimit[i]) ||
        !std::isfinite(arm.activeUpperLimit[i]) ||
        arm.activeLowerLimit[i] >= arm.activeUpperLimit[i]) {
      std::cerr << sideName(arm.config->side)
                << " arm has an invalid usable range for J" << i + 1
                << " after applying the safety margin" << std::endl;
      return false;
    }
  }
  arm.jointLimitsReady = true;
  return true;
}

bool insideJointLimits(const std::array<double, 7> &joints,
                       const ArmSession &arm) {
  if (!arm.jointLimitsReady) {
    std::cerr << sideName(arm.config->side)
              << " arm joint limits have not been initialized" << std::endl;
    return false;
  }
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (!std::isfinite(joints[i]) ||
        joints[i] < arm.activeLowerLimit[i] ||
        joints[i] > arm.activeUpperLimit[i]) {
      std::cerr << sideName(arm.config->side)
                << " arm joint safety boundary: J" << i + 1
                << "=" << joints[i] << " rad, limit=["
                << arm.activeLowerLimit[i] << ", "
                << arm.activeUpperLimit[i] << "] rad" << std::endl;
      return false;
    }
  }
  return true;
}

bool handStateFresh(const HandState &state) {
  return state.valid &&
         std::chrono::steady_clock::now() - state.timestamp <= kUdpTimeout;
}

bool teleoperationRequested(const HandState &state) {
  return handStateFresh(state) && state.grip;
}

bool gripModeSelfTest() {
  HandState left;
  HandState right;
  left.valid = true;
  right.valid = true;

  const auto check = [&](bool leftGrip, bool rightGrip,
                         bool expectedLeft, bool expectedRight) {
    const auto now = std::chrono::steady_clock::now();
    left.timestamp = now;
    right.timestamp = now;
    left.grip = leftGrip;
    right.grip = rightGrip;
    return teleoperationRequested(left) == expectedLeft &&
           teleoperationRequested(right) == expectedRight;
  };

  const bool ok = check(false, false, false, false) &&
                  check(true, false, true, false) &&
                  check(false, true, false, true) &&
                  check(true, true, true, true);
  std::cout << "Independent-Grip mode self-test: "
            << (ok ? "PASS" : "FAIL") << std::endl;
  return ok;
}

bool updateLatestJointState(ArmSession &arm,
                            std::array<double, 7> &joints) {
  bool updated =
      arm.robot->updateRobotState(std::chrono::milliseconds(1));
  while (arm.robot->updateRobotState(
      std::chrono::steady_clock::duration::zero())) {
    updated = true;
  }
  return updated &&
         arm.robot->getStateData(RtSupportedFields::jointPos_m, joints) == 0;
}

bool waitRobotIdle(ArmSession &arm,
                   std::chrono::seconds timeout = std::chrono::seconds(60)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    error_code ec;
    const auto state = arm.robot->operationState(ec);
    if (!checkEc(ec, std::string(sideName(arm.config->side)) +
                         " operationState")) {
      return false;
    }
    if (state == OperationState::idle) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cerr << sideName(arm.config->side)
            << " arm idle wait timed out or Ctrl+C received" << std::endl;
  return false;
}

bool switchToNrt(ArmSession &arm) {
  error_code ec;
  for (int attempt = 0; attempt < 20; ++attempt) {
    arm.robot->setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!ec) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return checkEc(ec, std::string(sideName(arm.config->side)) +
                         " switch to NrtCommand");
}

bool restoreNrtReady(ArmSession &arm) {
  if (!switchToNrt(arm)) return false;

  error_code ec;
  arm.robot->setOperateMode(OperateMode::automatic, ec);
  if (!checkEc(ec, std::string(sideName(arm.config->side)) +
                       " set automatic mode")) {
    return false;
  }

  arm.robot->setPowerState(true, ec);
  if (!checkEc(ec, std::string(sideName(arm.config->side)) + " power on")) {
    return false;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    ec.clear();
    const auto state = arm.robot->powerState(ec);
    if (!checkEc(ec, std::string(sideName(arm.config->side)) +
                         " read power state")) {
      return false;
    }
    if (state == PowerState::on) {
      std::cout << sideName(arm.config->side)
                << " arm NRT/automatic/power-on ready" << std::endl;
      return true;
    }
    if (state == PowerState::estop || state == PowerState::gstop) {
      std::cerr << sideName(arm.config->side)
                << " arm cannot power on: E-stop or safety stop is active"
                << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cerr << sideName(arm.config->side) << " arm power-on timeout"
            << std::endl;
  return false;
}

bool connectArm(ArmSession &arm) {
  try {
    std::cout << "Connecting " << sideName(arm.config->side)
              << " arm: robot=" << arm.config->robotIp
              << ", local=" << arm.config->localIp << std::endl;
    arm.robot =
        std::make_unique<ArRobot>(arm.config->robotIp, arm.config->localIp);
    if (!restoreNrtReady(arm)) {
      arm.robot.reset();
      return false;
    }
    if (!loadActiveJointLimits(arm)) {
      arm.robot.reset();
      return false;
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Connect " << sideName(arm.config->side)
              << " arm exception: " << e.what() << std::endl;
    arm.robot.reset();
    return false;
  }
}

bool connectBothArms(ArmSession &left, ArmSession &right) {
  if (!connectArm(left)) return false;
  if (!connectArm(right)) {
    left.robot.reset();
    return false;
  }
  return true;
}

bool reconnectBothArms(ArmSession &left, ArmSession &right) {
  left.robot.reset();
  right.robot.reset();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  return connectBothArms(left, right);
}

void stopNrtMotion(ArmSession &arm) {
  error_code ec;
  arm.robot->stop(ec);
  if (ec) {
    std::cerr << sideName(arm.config->side)
              << " arm stop failed: " << ec.message() << std::endl;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    ec.clear();
    if (arm.robot->operationState(ec) == OperationState::idle) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  ec.clear();
  arm.robot->moveReset(ec);
  if (ec) {
    std::cerr << sideName(arm.config->side)
              << " arm moveReset failed: " << ec.message() << std::endl;
  }
}

bool resetArm(ArmSession &arm) {
  const Side side = arm.config->side;
  std::cout << "Resetting " << sideName(side) << " arm with MoveAbsJ..."
            << std::endl;
  if (!insideJointLimits(arm.config->resetJoint, arm)) return false;
  if (!restoreNrtReady(arm)) return false;

  error_code ec;
  arm.robot->moveReset(ec);
  if (!checkEc(ec, std::string(sideName(side)) + " moveReset")) return false;

  MoveAbsJCommand command(
      JointPosition(std::vector<double>(arm.config->resetJoint.begin(),
                                        arm.config->resetJoint.end())),
      1000, 10);
  std::string commandId;
  arm.robot->moveAppend({command}, commandId, ec);
  if (!checkEc(ec, std::string(sideName(side)) + " moveAppend(reset)")) {
    return false;
  }
  arm.robot->moveStart(ec);
  if (!checkEc(ec, std::string(sideName(side)) + " moveStart(reset)")) {
    return false;
  }
  if (!waitRobotIdle(arm)) {
    stopNrtMotion(arm);
    return false;
  }

  const auto joints = arm.robot->jointPos(ec);
  if (!checkEc(ec, std::string(sideName(side)) + " jointPos(after reset)")) {
    return false;
  }
  std::cout << sideName(side) << " arm reset complete" << std::endl;
  return insideJointLimits(joints, arm);
}

bool resetBothArms(ArmSession &left, ArmSession &right) {
  // NRT reset commands are deliberately serialized. Teleoperation itself is
  // synchronized in one 50 Hz loop below.
  if (!resetArm(left)) return false;
  return resetArm(right);
}

Eigen::Vector3d limitedPositionStep(const Eigen::Vector3d &current,
                                    const Eigen::Vector3d &target) {
  const Eigen::Vector3d difference = target - current;
  const double distance = difference.norm();
  if (distance < kPositionDeadband) return current;
  if (distance <= kMaxPositionStep) return target;
  return current + difference * (kMaxPositionStep / distance);
}

Eigen::Matrix3d limitedRotationStep(const Eigen::Matrix3d &current,
                                    const Eigen::Matrix3d &target) {
  const Eigen::AngleAxisd difference(target * current.transpose());
  const double angle = std::abs(difference.angle());
  if (angle < kRotationDeadband) return current;
  const double step = std::min(angle, kMaxRotationStep);
  return Eigen::AngleAxisd(step, difference.axis()).toRotationMatrix() * current;
}

Eigen::Matrix3d scaledRotation(const Eigen::Matrix3d &rawOffset) {
  const Eigen::AngleAxisd rawAngleAxis(rawOffset);
  if (std::abs(rawAngleAxis.angle()) < kRotationDeadband) {
    return Eigen::Matrix3d::Identity();
  }
  const double mappedAngle =
      std::abs(rawAngleAxis.angle()) * kRotationGain;
  return Eigen::AngleAxisd(mappedAngle, rawAngleAxis.axis())
      .toRotationMatrix();
}

struct PoseMapping {
  std::array<double, 6> baseFrame{};
  Toolset toolset{};
  Eigen::Vector3d handAnchorPosition{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d handAnchorRotation{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d robotAnchorPosition{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d robotAnchorRotation{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d commandedPosition{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d commandedRotation{Eigen::Matrix3d::Identity()};
};

struct PoseCandidate {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
  CartesianPosition command{};
};

bool initializeMapping(ArmSession &arm, const HandState &handAnchor,
                       PoseMapping &mapping, CartesianPosition &command) {
  error_code ec;
  const auto robotEndPosture =
      arm.robot->posture(CoordinateType::endInRef, ec);
  if (!checkEc(ec, std::string(sideName(arm.config->side)) +
                       " read end-in-reference pose")) {
    return false;
  }
  mapping.baseFrame = arm.robot->baseFrame(ec);
  if (!checkEc(ec, std::string(sideName(arm.config->side)) +
                       " read base frame")) {
    return false;
  }
  mapping.toolset = arm.robot->toolset(ec);
  if (!checkEc(ec,
               std::string(sideName(arm.config->side)) + " read toolset")) {
    return false;
  }

  std::array<double, 16> robotEndMatrix{};
  Utils::postureToTransArray(robotEndPosture, robotEndMatrix);
  Utils::arrayToTransMatrix(robotEndMatrix, mapping.robotAnchorRotation,
                            mapping.robotAnchorPosition);

  mapping.handAnchorPosition =
      Eigen::Vector3d(handAnchor.positionM[0], handAnchor.positionM[1],
                      handAnchor.positionM[2]);
  mapping.handAnchorRotation = handAnchor.orientation.toRotationMatrix();
  mapping.commandedPosition = mapping.robotAnchorPosition;
  mapping.commandedRotation = mapping.robotAnchorRotation;

  const auto initialFlangePose = Utils::EndInRefToFlanInBase(
      mapping.baseFrame, mapping.toolset, robotEndPosture);
  Utils::postureToTransArray(initialFlangePose, command.pos);
  return true;
}

bool buildPoseCandidate(const HandState &hand, const PoseMapping &mapping,
                        PoseCandidate &candidate, std::string &reason,
                        Side side) {
  const Eigen::Vector3d handPosition(hand.positionM[0], hand.positionM[1],
                                     hand.positionM[2]);
  const Eigen::Vector3d rawPositionOffset =
      handPosition - mapping.handAnchorPosition;
  const Eigen::Matrix3d rawRotationOffset =
      hand.orientation.toRotationMatrix() *
      mapping.handAnchorRotation.transpose();
  const Eigen::AngleAxisd rawRotationAngle(rawRotationOffset);

  if (rawPositionOffset.norm() > kMaxRawPositionOffset ||
      std::abs(rawRotationAngle.angle()) > kMaxRawRotationOffset) {
    reason = std::string(sideName(side)) +
             " controller offset exceeded raw safety range";
    return false;
  }

  const Eigen::Vector3d mappedPositionOffset =
      kPositionGain * rawPositionOffset;
  const Eigen::Matrix3d mappedRotationOffset =
      scaledRotation(rawRotationOffset);
  const Eigen::Vector3d targetPosition =
      mapping.robotAnchorPosition + mappedPositionOffset;
  const Eigen::Matrix3d targetRotation =
      mappedRotationOffset * mapping.robotAnchorRotation;
  candidate.position =
      limitedPositionStep(mapping.commandedPosition, targetPosition);
  candidate.rotation =
      limitedRotationStep(mapping.commandedRotation, targetRotation);

  std::array<double, 16> targetEndMatrix{};
  Utils::transMatrixToArray(candidate.rotation, candidate.position,
                            targetEndMatrix);
  std::array<double, 6> targetEndPose{};
  Utils::transArrayToPosture(targetEndMatrix, targetEndPose);
  const auto targetFlangePose = Utils::EndInRefToFlanInBase(
      mapping.baseFrame, mapping.toolset, targetEndPose);
  Utils::postureToTransArray(targetFlangePose, candidate.command.pos);
  return true;
}

void commitPoseCandidate(const PoseCandidate &candidate,
                         PoseMapping &mapping,
                         CartesianPosition &command) {
  mapping.commandedPosition = candidate.position;
  mapping.commandedRotation = candidate.rotation;
  command = candidate.command;
}

void reanchorPoseMapping(const HandState &hand, PoseMapping &mapping) {
  mapping.handAnchorPosition =
      Eigen::Vector3d(hand.positionM[0], hand.positionM[1], hand.positionM[2]);
  mapping.handAnchorRotation = hand.orientation.toRotationMatrix();
  mapping.robotAnchorPosition = mapping.commandedPosition;
  mapping.robotAnchorRotation = mapping.commandedRotation;
}

bool poseFilterSelfTest() {
  const Eigen::Vector3d scaledPosition =
      kPositionGain * Eigen::Vector3d(0.10, 0.0, 0.0);
  const bool positionGainOk =
      std::abs(scaledPosition.x() - 0.03) < 1e-9 &&
      std::abs(scaledPosition.y()) < 1e-9 &&
      std::abs(scaledPosition.z()) < 1e-9;

  const Eigen::Matrix3d rawRotation =
      Eigen::AngleAxisd(90.0 * kPi / 180.0,
                        Eigen::Vector3d::UnitZ())
          .toRotationMatrix();
  const Eigen::AngleAxisd mappedRotation(
      scaledRotation(rawRotation));
  const bool rotationGainOk =
      std::abs(mappedRotation.angle() - 22.5 * kPi / 180.0) < 1e-9;

  const Eigen::Vector3d stepped =
      limitedPositionStep(Eigen::Vector3d::Zero(),
                          Eigen::Vector3d(0.01, 0.0, 0.0));
  const bool rateLimitOk =
      std::abs(stepped.norm() - kMaxPositionStep) < 1e-9;

  const bool ok = positionGainOk && rotationGainOk && rateLimitOk;
  std::cout << "Pose-filter math self-test: "
            << (ok ? "PASS" : "FAIL") << std::endl;
  return ok;
}

bool runIndependentServo(
    ArmSession &left, ArmSession &right, UdpReceiver &receiver,
    const std::array<HandState, 2> &handAnchors) {
  const bool leftInitiallyActive =
      teleoperationRequested(handAnchors[sideIndex(Side::left)]);
  const bool rightInitiallyActive =
      teleoperationRequested(handAnchors[sideIndex(Side::right)]);
  if (!leftInitiallyActive && !rightInitiallyActive) {
    std::cerr << "At least one fresh controller stream with Grip held is "
                 "required before teleoperation"
              << std::endl;
    return true;
  }

  PoseMapping leftMapping;
  PoseMapping rightMapping;
  CartesianPosition leftCommand;
  CartesianPosition rightCommand;
  if (!initializeMapping(left, handAnchors[sideIndex(Side::left)],
                         leftMapping, leftCommand) ||
      !initializeMapping(right, handAnchors[sideIndex(Side::right)],
                         rightMapping, rightCommand)) {
    return false;
  }

  error_code ec;
  left.robot->setRtNetworkTolerance(20, ec);
  if (!checkEc(ec, "left setRtNetworkTolerance")) return false;
  ec.clear();
  right.robot->setRtNetworkTolerance(20, ec);
  if (!checkEc(ec, "right setRtNetworkTolerance")) return false;

  ec.clear();
  left.robot->setMotionControlMode(MotionControlMode::RtCommand, ec);
  if (!checkEc(ec, "left switch to RtCommand")) return false;
  ec.clear();
  right.robot->setMotionControlMode(MotionControlMode::RtCommand, ec);
  if (!checkEc(ec, "right switch to RtCommand")) {
    switchToNrt(left);
    return false;
  }

  auto leftController = left.robot->getRtMotionController().lock();
  auto rightController = right.robot->getRtMotionController().lock();
  if (!leftController || !rightController) {
    std::cerr << "Failed to obtain both real-time motion controllers"
              << std::endl;
    error_code idleEc;
    left.robot->setMotionControlMode(MotionControlMode::Idle, idleEc);
    idleEc.clear();
    right.robot->setMotionControlMode(MotionControlMode::Idle, idleEc);
    return false;
  }

  bool leftReceiving = false;
  bool rightReceiving = false;
  bool leftServoConfigured = false;
  bool rightServoConfigured = false;
  bool leftMoveStarted = false;
  bool rightMoveStarted = false;

  auto finishBoth = [&]() {
    bool success = true;
    try {
      if (leftMoveStarted) leftController->stopMove();
    } catch (const std::exception &e) {
      std::cerr << "left stopMove exception: " << e.what() << std::endl;
      success = false;
    }
    try {
      if (rightMoveStarted) rightController->stopMove();
    } catch (const std::exception &e) {
      std::cerr << "right stopMove exception: " << e.what() << std::endl;
      success = false;
    }
    try {
      if (leftServoConfigured) leftController->stopServoJoint();
    } catch (const std::exception &e) {
      std::cerr << "left stopServoJoint exception: " << e.what()
                << std::endl;
      success = false;
    }
    try {
      if (rightServoConfigured) rightController->stopServoJoint();
    } catch (const std::exception &e) {
      std::cerr << "right stopServoJoint exception: " << e.what()
                << std::endl;
      success = false;
    }

    if (leftReceiving) {
      try {
        left.robot->stopReceiveRobotState();
      } catch (const std::exception &e) {
        std::cerr << "left stopReceiveRobotState exception: " << e.what()
                  << std::endl;
        success = false;
      }
    }
    if (rightReceiving) {
      try {
        right.robot->stopReceiveRobotState();
      } catch (const std::exception &e) {
        std::cerr << "right stopReceiveRobotState exception: " << e.what()
                  << std::endl;
        success = false;
      }
    }

    error_code idleEc;
    left.robot->setMotionControlMode(MotionControlMode::Idle, idleEc);
    if (!checkEc(idleEc, "left switch to Idle")) success = false;
    idleEc.clear();
    right.robot->setMotionControlMode(MotionControlMode::Idle, idleEc);
    if (!checkEc(idleEc, "right switch to Idle")) success = false;
    return success;
  };

  try {
    left.robot->startReceiveRobotState(
        std::chrono::milliseconds(1), {RtSupportedFields::jointPos_m});
    leftReceiving = true;
    right.robot->startReceiveRobotState(
        std::chrono::milliseconds(1), {RtSupportedFields::jointPos_m});
    rightReceiving = true;

    while (left.robot->updateRobotState(
        std::chrono::steady_clock::duration::zero())) {
    }
    while (right.robot->updateRobotState(
        std::chrono::steady_clock::duration::zero())) {
    }

    std::array<double, 7> leftJoints{};
    std::array<double, 7> rightJoints{};
    if (!updateLatestJointState(left, leftJoints) ||
        !updateLatestJointState(right, rightJoints) ||
        !insideJointLimits(leftJoints, left) ||
        !insideJointLimits(rightJoints, right)) {
      std::cerr << "Failed initial dual-arm joint safety check" << std::endl;
      finishBoth();
      return false;
    }

    leftController->setServoJoint(kPlanPeriod, kPlanPeriod, 0, ec);
    if (!checkEc(ec, "left setServoJoint")) {
      finishBoth();
      return false;
    }
    leftServoConfigured = true;
    ec.clear();
    rightController->setServoJoint(kPlanPeriod, kPlanPeriod, 0, ec);
    if (!checkEc(ec, "right setServoJoint")) {
      finishBoth();
      return false;
    }
    rightServoConfigured = true;

    leftController->startMove(RtControllerMode::cartesianPosition);
    leftMoveStarted = true;
    rightController->startMove(RtControllerMode::cartesianPosition);
    rightMoveStarted = true;

    // Real-time setup takes a finite amount of time. Re-anchor from the
    // latest hand samples so controller motion during setup cannot become an
    // unintended robot offset when the first ServoL command is sent.
    const auto activationHands = receiver.latestBoth();
    bool leftActive =
        teleoperationRequested(activationHands[sideIndex(Side::left)]);
    bool rightActive =
        teleoperationRequested(activationHands[sideIndex(Side::right)]);
    if (leftActive) {
      reanchorPoseMapping(activationHands[sideIndex(Side::left)], leftMapping);
    }
    if (rightActive) {
      reanchorPoseMapping(activationHands[sideIndex(Side::right)],
                          rightMapping);
    }

    std::string stopReason = "both Grip buttons were released";
    std::string leftLastRejectedReason;
    std::string rightLastRejectedReason;
    std::size_t leftRejectedCycles = 0;
    std::size_t rightRejectedCycles = 0;
    auto nextCycle = std::chrono::steady_clock::now();
    std::cout << "Independent teleoperation active. Hold left Grip for the "
                 "left arm, right Grip for the right arm, or both for both "
                 "arms. Releasing one Grip stops only that arm."
              << std::endl;
    std::cout << "Input filters: position gain=" << kPositionGain
              << ", rotation gain=" << kRotationGain
              << ", raw jump limit=" << kMaxRawPositionJump * 1000.0
              << " mm / "
              << kMaxRawRotationJump * 180.0 / kPi
              << " deg"
              << ", max command step=" << kMaxPositionStep * 1000.0
              << " mm / "
              << kMaxRotationStep * 180.0 / kPi
              << " deg, joint margin="
              << kJointLimitMargin * 180.0 / kPi << " deg" << std::endl;

    while (!gStopRequested) {
      const auto hands = receiver.latestBoth();
      const HandState &leftHand = hands[sideIndex(Side::left)];
      const HandState &rightHand = hands[sideIndex(Side::right)];

      const bool leftRequested = teleoperationRequested(leftHand);
      const bool rightRequested = teleoperationRequested(rightHand);
      if (!leftRequested && !rightRequested) {
        stopReason =
            "both Grip buttons were released or both UDP streams became stale";
        break;
      }

      if (leftRequested && !leftActive) {
        reanchorPoseMapping(leftHand, leftMapping);
        leftRejectedCycles = 0;
        leftLastRejectedReason.clear();
        std::cout << "left arm teleoperation enabled" << std::endl;
      } else if (!leftRequested && leftActive) {
        std::cout << "left arm teleoperation paused" << std::endl;
      }
      if (rightRequested && !rightActive) {
        reanchorPoseMapping(rightHand, rightMapping);
        rightRejectedCycles = 0;
        rightLastRejectedReason.clear();
        std::cout << "right arm teleoperation enabled" << std::endl;
      } else if (!rightRequested && rightActive) {
        std::cout << "right arm teleoperation paused" << std::endl;
      }
      leftActive = leftRequested;
      rightActive = rightRequested;

      if (!updateLatestJointState(left, leftJoints) ||
          !updateLatestJointState(right, rightJoints) ||
          !insideJointLimits(leftJoints, left) ||
          !insideJointLimits(rightJoints, right)) {
        stopReason = "joint state read failed or a joint safety limit was hit";
        break;
      }

      bool candidateTimeout = false;
      if (leftActive) {
        PoseCandidate leftCandidate;
        std::string rejectedReason;
        if (buildPoseCandidate(leftHand, leftMapping, leftCandidate,
                               rejectedReason, Side::left)) {
          leftController->sendCommand(leftCandidate.command);
          commitPoseCandidate(leftCandidate, leftMapping, leftCommand);
          leftRejectedCycles = 0;
          leftLastRejectedReason.clear();
        } else {
          leftController->sendCommand(leftCommand);
          ++leftRejectedCycles;
          if (rejectedReason != leftLastRejectedReason ||
              leftRejectedCycles % 25 == 1) {
            std::cerr << "Holding last valid left-arm target: "
                      << rejectedReason << std::endl;
            leftLastRejectedReason = rejectedReason;
          }
          if (leftRejectedCycles >= kMaxRejectedCycles) {
            stopReason =
                "left candidate pose remained invalid for 2 seconds: " +
                rejectedReason;
            candidateTimeout = true;
          }
        }
      } else {
        leftController->sendCommand(leftCommand);
        leftRejectedCycles = 0;
        leftLastRejectedReason.clear();
      }

      if (rightActive) {
        PoseCandidate rightCandidate;
        std::string rejectedReason;
        if (buildPoseCandidate(rightHand, rightMapping, rightCandidate,
                               rejectedReason, Side::right)) {
          rightController->sendCommand(rightCandidate.command);
          commitPoseCandidate(rightCandidate, rightMapping, rightCommand);
          rightRejectedCycles = 0;
          rightLastRejectedReason.clear();
        } else {
          rightController->sendCommand(rightCommand);
          ++rightRejectedCycles;
          if (rejectedReason != rightLastRejectedReason ||
              rightRejectedCycles % 25 == 1) {
            std::cerr << "Holding last valid right-arm target: "
                      << rejectedReason << std::endl;
            rightLastRejectedReason = rejectedReason;
          }
          if (rightRejectedCycles >= kMaxRejectedCycles) {
            stopReason =
                "right candidate pose remained invalid for 2 seconds: " +
                rejectedReason;
            candidateTimeout = true;
          }
        }
      } else {
        rightController->sendCommand(rightCommand);
        rightRejectedCycles = 0;
        rightLastRejectedReason.clear();
      }
      if (candidateTimeout) break;

      nextCycle += std::chrono::milliseconds(20);
      const auto now = std::chrono::steady_clock::now();
      if (now > nextCycle + kMaxCycleLateness) {
        stopReason = "50 Hz control loop overrun exceeded 40 ms";
        break;
      }
      std::this_thread::sleep_until(nextCycle);
    }

    if (gStopRequested) stopReason = "Ctrl+C received";
    std::cout << "Teleoperation session stopped: " << stopReason << std::endl;
    return finishBoth();
  } catch (const std::exception &e) {
    std::cerr << "Dual ServoL exception: " << e.what() << std::endl;
    finishBoth();
    return false;
  }
}

void printControllerErrors(ArmSession &arm) {
  if (!arm.robot) return;
  error_code ec;
  const auto logs = arm.robot->queryControllerLog(5, {LogInfo::error}, ec);
  if (ec) return;
  std::cout << sideName(arm.config->side)
            << " arm latest controller error logs:" << std::endl;
  for (const auto &log : logs) {
    std::cout << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    const bool parserOk = parserSelfTest();
    const bool filterOk = poseFilterSelfTest();
    const bool gripModeOk = gripModeSelfTest();
    return parserOk && filterOk && gripModeOk ? 0 : 1;
  }
  if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [--self-test]" << std::endl;
    return 2;
  }

  std::signal(SIGINT, signalHandler);

  UdpReceiver receiver(kUdpPort);
  receiver.start();
  std::cout << "Listening for left(id=0) and right(id=1) controller data on "
               "UDP port "
            << kUdpPort << ". Stop receive.py if it occupies this port."
            << std::endl;

  ArmSession left{&kLeftArmConfig, nullptr};
  ArmSession right{&kRightArmConfig, nullptr};

  try {
    std::cout << "Waiting for BOTH controller streams..." << std::endl;
    while (!gStopRequested) {
      const auto hands = receiver.latestBoth();
      if (hands[sideIndex(Side::left)].valid &&
          hands[sideIndex(Side::right)].valid) {
        break;
      }
      const std::string receiverError = receiver.error();
      if (!receiverError.empty()) {
        std::cerr << receiverError << std::endl;
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (gStopRequested) return 0;

    if (!connectBothArms(left, right)) return 1;

    auto hands = receiver.latestBoth();
    bool gripArmed = !hands[sideIndex(Side::left)].grip &&
                     !hands[sideIndex(Side::right)].grip;
    bool resetArmed = !hands[sideIndex(Side::left)].down &&
                      !hands[sideIndex(Side::right)].down;

    std::cout << "Ready. Hold either Grip to control its matching arm, or "
                 "hold both Grips to control both arms. Releasing one Grip "
                 "stops only that arm. While idle, press BOTH Down buttons "
                 "to reset both arms."
              << std::endl;

    while (!gStopRequested) {
      hands = receiver.latestBoth();
      const bool leftFresh =
          handStateFresh(hands[sideIndex(Side::left)]);
      const bool rightFresh =
          handStateFresh(hands[sideIndex(Side::right)]);
      if (!leftFresh && !rightFresh) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }

      const HandState &leftHand = hands[sideIndex(Side::left)];
      const HandState &rightHand = hands[sideIndex(Side::right)];

      if (!leftHand.grip && !rightHand.grip) gripArmed = true;
      if (!leftHand.down && !rightHand.down) resetArmed = true;

      const bool anyFreshGrip =
          teleoperationRequested(leftHand) ||
          teleoperationRequested(rightHand);
      const bool bothDown = leftHand.down && rightHand.down;

      if (resetArmed && leftFresh && rightFresh && bothDown &&
          !leftHand.grip && !rightHand.grip) {
        resetArmed = false;
        if (!resetBothArms(left, right)) {
          printControllerErrors(left);
          printControllerErrors(right);
          return 1;
        }
        std::cout << "Both arms reset. Release both Down buttons before the "
                     "next reset."
                  << std::endl;
      } else if (gripArmed && anyFreshGrip) {
        gripArmed = false;
        if (!runIndependentServo(left, right, receiver, hands)) {
          printControllerErrors(left);
          printControllerErrors(right);
          return 1;
        }

        // The SDK session is recreated after leaving RT mode, matching the
        // recovery behavior used by pico_servol.cpp.
        if (!reconnectBothArms(left, right)) {
          std::cerr << "Failed to recreate both NRT sessions after ServoL"
                    << std::endl;
          return 1;
        }
        std::cout << "Both arms are idle. Release both Grip buttons to re-arm."
                  << std::endl;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (left.robot) switchToNrt(left);
    if (right.robot) switchToNrt(right);
    std::cout << "Program exited" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Program exception: " << e.what() << std::endl;
    printControllerErrors(left);
    printControllerErrors(right);
    return 1;
  }
}
