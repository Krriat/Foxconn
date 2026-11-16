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
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;

namespace {

constexpr const char *kRobotIp = "192.168.2.160";
constexpr const char *kLocalIp = "192.168.2.10";
constexpr uint16_t kUdpPort = 5050;
constexpr std::size_t kPacketSize = 38;

constexpr double kPi = 3.14159265358979323846;
constexpr double kPlanPeriod = 0.01;  // 100 Hz ServoL command period.
constexpr auto kPlanCycle = std::chrono::milliseconds(10);
constexpr std::size_t kDatasetPublishDivider = 2;  // Keep recording at 50 Hz.
constexpr double kPositionFilterAlpha = 0.20;
constexpr double kRotationFilterAlpha = 0.15;
constexpr double kDefaultPositionGain = 0.618;
constexpr double kMinPositionGain = 0.10;
constexpr double kMaxPositionGain = 1.00;
constexpr double kRotationGain = 0.25;
constexpr double kPositionDeadband = 0.0005;
constexpr double kRotationDeadband = 0.3 * kPi / 180.0;
constexpr double kMaxPositionStep = 0.0005;
constexpr double kMaxRotationStep = 0.25 * kPi / 180.0;
constexpr double kMaxRawPositionJump = 0.10;
constexpr double kMaxRawRotationJump = 45.0 * kPi / 180.0;
constexpr double kMaxRawPositionOffset = 0.40;
constexpr double kMaxRawRotationOffset = 120.0 * kPi / 180.0;
constexpr double kJoystickDeadband = 0.15;
constexpr double kMaxElbowVelocity = 2.0 * kPi / 180.0;
constexpr double kJoystickLowPassTimeConstant = 0.08;
constexpr double kMaxElbowAcceleration = 2.0 * kPi / 180.0;
constexpr double kMaxElbowStep = 0.01 * kPi / 180.0;
constexpr double kMinElbow = -kPi;
constexpr double kMaxElbow = kPi;
constexpr auto kElbowLogPeriod = std::chrono::milliseconds(200);  // 5 Hz.
constexpr auto kUdpTimeout = std::chrono::milliseconds(250);
constexpr auto kMaxCycleLateness = std::chrono::milliseconds(40);
constexpr std::size_t kMaxRejectedCycles = 200;  // 2 s at 100 Hz.

constexpr int kGripperReleasedPosition = 900;
constexpr int kGripperPressedPosition = 1500;

constexpr double kJointLimitMargin = 12.0 * kPi / 180.0;
const std::array<double, 7> kLowerLimit = {
    -3.10669 + kJointLimitMargin,
    -2.0944 + kJointLimitMargin,
    -3.10669 + kJointLimitMargin,
    0 + kJointLimitMargin,
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

// A/down返回的初始关节位置，依次为J1...J7，单位rad。
const std::array<double, 7> kDefaultInitialJoint = {
    -1.969919, 1.365856, 1.272963, 1.476893,
    1.735811,  -0.026802, 0.165558,
};

std::array<double, 7> gActiveLowerLimit{};
std::array<double, 7> gActiveUpperLimit{};
bool gJointLimitsReady = false;
double gPositionGain = kDefaultPositionGain;
std::array<double, 7> gInitialJoint = kDefaultInitialJoint;

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

double positionGainFromEnvironment() {
  const char *raw = std::getenv("XCORE_TELEOP_POSITION_GAIN");
  if (raw == nullptr || *raw == '\0') return kDefaultPositionGain;

  char *end = nullptr;
  const double value = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || !std::isfinite(value) ||
      value < kMinPositionGain || value > kMaxPositionGain) {
    std::cerr << "忽略无效的XCORE_TELEOP_POSITION_GAIN='" << raw
              << "'，使用默认值 " << kDefaultPositionGain << "（允许范围 "
              << kMinPositionGain << "~" << kMaxPositionGain << "）"
              << std::endl;
    return kDefaultPositionGain;
  }
  return value;
}

std::array<double, 7> initialJointFromEnvironment() {
  const char *raw = std::getenv("XCORE_TELEOP_RIGHT_RESET_JOINTS");
  if (raw == nullptr || *raw == '\0') return kDefaultInitialJoint;

  std::array<double, 7> values{};
  std::istringstream input(raw);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!(input >> values[index]) || !std::isfinite(values[index]) ||
        values[index] < kLowerLimit[index] ||
        values[index] > kUpperLimit[index]) {
      std::cerr << "忽略无效的XCORE_TELEOP_RIGHT_RESET_JOINTS='" << raw
                << "'，J" << index + 1 << "必须是关节安全范围内的有限数值"
                << std::endl;
      return kDefaultInitialJoint;
    }
  }
  std::string extra;
  if (input >> extra) {
    std::cerr << "忽略无效的XCORE_TELEOP_RIGHT_RESET_JOINTS='" << raw
              << "'，必须恰好包含7个关节值" << std::endl;
    return kDefaultInitialJoint;
  }
  return values;
}

struct HandPacket {
  std::array<double, 3> positionMm{};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  std::array<double, 2> joystick{};
  bool trigger{false};
  bool grip{false};
  bool down{false};
  std::uint8_t buttons{0};
};

struct HandState {
  std::array<double, 3> positionM{};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  std::array<double, 3> rawPositionM{};
  Eigen::Quaterniond rawOrientation{Eigen::Quaterniond::Identity()};
  std::array<double, 2> joystick{};
  bool trigger{false};
  bool grip{false};
  bool down{false};
  std::uint8_t buttons{0};
  bool valid{false};
  bool rawValid{false};
  std::uint64_t sequence{0};
  std::chrono::steady_clock::time_point timestamp{};
};

double clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

double joystickAfterDeadband(double value) {
  const double limited = clamp(value, -1.0, 1.0);
  const double magnitude = std::abs(limited);
  if (magnitude <= kJoystickDeadband) return 0.0;
  return std::copysign(
      (magnitude - kJoystickDeadband) / (1.0 - kJoystickDeadband), limited);
}

struct ElbowControlState {
  double filteredJoystickY{0.0};
  double velocity{0.0};
};

double nextElbowCommand(double currentElbow, double joystickY,
                        ElbowControlState &state) {
  const double targetJoystick = joystickAfterDeadband(joystickY);
  const double filterAlpha =
      kPlanPeriod / (kJoystickLowPassTimeConstant + kPlanPeriod);
  state.filteredJoystickY +=
      filterAlpha * (targetJoystick - state.filteredJoystickY);

  const double targetVelocity =
      state.filteredJoystickY * kMaxElbowVelocity;
  const double maxVelocityChange = kMaxElbowAcceleration * kPlanPeriod;
  state.velocity += clamp(targetVelocity - state.velocity,
                          -maxVelocityChange, maxVelocityChange);

  const double step = clamp(
      state.velocity * kPlanPeriod, -kMaxElbowStep, kMaxElbowStep);
  state.velocity = step / kPlanPeriod;
  const double unconstrained = currentElbow + step;
  const double command = clamp(unconstrained, kMinElbow, kMaxElbow);
  if ((command <= kMinElbow && state.velocity < 0.0) ||
      (command >= kMaxElbow && state.velocity > 0.0)) {
    state.velocity = 0.0;
  }
  return command;
}

double nextElbowTargetCommand(double currentElbow, double targetElbow,
                              ElbowControlState &state) {
  targetElbow = clamp(targetElbow, kMinElbow, kMaxElbow);
  const double error = targetElbow - currentElbow;
  const double targetVelocity =
      clamp(error / kPlanPeriod, -kMaxElbowVelocity, kMaxElbowVelocity);
  const double maxVelocityChange = kMaxElbowAcceleration * kPlanPeriod;
  state.velocity += clamp(targetVelocity - state.velocity,
                          -maxVelocityChange, maxVelocityChange);
  const double step = clamp(
      state.velocity * kPlanPeriod, -kMaxElbowStep, kMaxElbowStep);
  state.velocity = step / kPlanPeriod;
  const double candidate = currentElbow + step;
  if ((error >= 0.0 && candidate >= targetElbow) ||
      (error <= 0.0 && candidate <= targetElbow)) {
    state.velocity = 0.0;
    return targetElbow;
  }
  return clamp(candidate, kMinElbow, kMaxElbow);
}

struct ElbowIpcCommand {
  double target{0.0};
  bool applyTarget{false};
  bool enabled{true};
};

class ElbowIpc {
 public:
  ElbowIpc() {
    const char *command = std::getenv("XCORE_TELEOP_ELBOW_COMMAND_PATH");
    const char *status = std::getenv("XCORE_TELEOP_ELBOW_STATUS_PATH");
    if (command != nullptr) commandPath_ = command;
    if (status != nullptr) statusPath_ = status;
  }

  bool readRightCommand(ElbowIpcCommand &command) {
    if (commandPath_.empty()) return false;
    std::ifstream input(commandPath_);
    std::uint64_t generation = 0;
    double left = 0.0;
    double right = 0.0;
    int leftApply = 0;
    int rightApply = 1;
    int enabled = 1;
    if (!(input >> generation >> left >> right) ||
        generation <= generation_ || !std::isfinite(right) ||
        right < kMinElbow || right > kMaxElbow) {
      return false;
    }
    if (!(input >> leftApply >> rightApply >> enabled)) {
      leftApply = 0;
      rightApply = 1;
      enabled = 1;
    }
    generation_ = generation;
    enabled_ = enabled != 0;
    command.target = right;
    command.applyTarget = rightApply != 0;
    command.enabled = enabled_;
    return true;
  }

  void writeRightStatus(double current, double target, bool active) const {
    if (statusPath_.empty() || !std::isfinite(current) ||
        !std::isfinite(target)) {
      return;
    }
    const double timestamp =
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    std::ostringstream json;
    json << std::setprecision(15)
         << "{\"timestamp\":" << timestamp
         << ",\"generation\":" << generation_
         << ",\"enabled\":" << (enabled_ ? "true" : "false")
         << ",\"arms\":{\"right\":{\"current_rad\":" << current
         << ",\"target_rad\":" << target
         << ",\"active\":" << (active ? "true" : "false") << "}}}\n";
    const std::string temporary = statusPath_ + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) return;
    output << json.str();
    output.close();
    if (output.fail()) return;
    std::rename(temporary.c_str(), statusPath_.c_str());
  }

 private:
  std::string commandPath_;
  std::string statusPath_;
  std::uint64_t generation_{0};
  bool enabled_{true};
};

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

bool finiteQuaternion(const Eigen::Quaterniond &q) {
  return std::isfinite(q.w()) && std::isfinite(q.x()) &&
         std::isfinite(q.y()) && std::isfinite(q.z());
}

double quaternionAngularDistance(const Eigen::Quaterniond &a,
                                 const Eigen::Quaterniond &b) {
  const double dot =
      clamp(std::abs(a.normalized().dot(b.normalized())), -1.0, 1.0);
  return 2.0 * std::acos(dot);
}

// 与receive.py的quaternion_to_euler()轴交换和固定角度修正保持一致。
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

  // receive.py输出: (原pitch, 原roll-45度, -原yaw-90度)。
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

bool parseRightHandPacket(const std::uint8_t *data, std::size_t size,
                          HandPacket &packet) {
  if (size != kPacketSize || data[0] != 1) return false;

  std::array<double, 9> values{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = readFloatLe(data + 1 + i * sizeof(float));
    if (!std::isfinite(values[i])) return false;
  }

  packet.positionMm = {values[0], values[1], values[2]};
  const Eigen::Quaterniond rawOrientation(
      values[6], values[3], values[4], values[5]);
  if (!finiteQuaternion(rawOrientation) || rawOrientation.norm() < 1e-8) {
    return false;
  }
  packet.orientation =
      convertPicoOrientation(values[3], values[4], values[5], values[6]);
  packet.joystick = {clamp(values[7], -1.0, 1.0),
                     clamp(values[8], -1.0, 1.0)};
  const std::uint8_t buttons = data[37];
  packet.buttons = buttons;
  packet.trigger = (buttons & (1U << 0U)) != 0;
  packet.grip = (buttons & (1U << 1U)) != 0;
  packet.down = (buttons & (1U << 2U)) != 0;
  return true;
}

void writeFloatLe(std::uint8_t *data, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  data[0] = static_cast<std::uint8_t>(bits & 0xffU);
  data[1] = static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
  data[2] = static_cast<std::uint8_t>((bits >> 16U) & 0xffU);
  data[3] = static_cast<std::uint8_t>((bits >> 24U) & 0xffU);
}

bool parserSelfTest() {
  std::array<std::uint8_t, kPacketSize> data{};
  data[0] = 1;
  const std::array<float, 9> values = {
      172.14F, -178.71F, -221.11F, 0.0F, 0.0F,
      0.0F,   1.0F,     -0.25F,   0.75F,
  };
  for (std::size_t i = 0; i < values.size(); ++i) {
    writeFloatLe(data.data() + 1 + i * sizeof(float), values[i]);
  }
  data[37] = static_cast<std::uint8_t>(
      (1U << 0U) | (1U << 1U) | (1U << 2U) | (1U << 3U));

  HandPacket packet;
  const bool ok = parseRightHandPacket(data.data(), data.size(), packet) &&
                  std::abs(packet.positionMm[0] - values[0]) < 1e-3 &&
                  std::abs(packet.positionMm[1] - values[1]) < 1e-3 &&
                  std::abs(packet.positionMm[2] - values[2]) < 1e-3 &&
                  packet.trigger && packet.grip && packet.down &&
                  std::abs(packet.joystick[0] + 0.25) < 1e-6 &&
                  std::abs(packet.joystick[1] - 0.75) < 1e-6 &&
                  packet.buttons == 0x0fU &&
                  finiteQuaternion(packet.orientation);
  std::cout << "UDP解析自检: " << (ok ? "通过" : "失败") << std::endl;
  return ok;
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

  HandState latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  void setTeleoperationActive(bool active) {
    teleoperationActive_.store(active);
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
    state_.trigger = packet.trigger;
    state_.grip = packet.grip;
    state_.down = packet.down;
    state_.buttons = packet.buttons;
    state_.joystick = packet.joystick;

    std::array<double, 3> rawPositionM{};
    for (std::size_t i = 0; i < rawPositionM.size(); ++i) {
      rawPositionM[i] = packet.positionMm[i] / 1000.0;
    }

    // Match the dual-arm controller's PICO re-localization protection. In
    // click-toggle mode this must remain active after the Grip is released.
    if (state_.rawValid && teleoperationActive_.load()) {
      const Eigen::Vector3d previous(
          state_.rawPositionM[0], state_.rawPositionM[1],
          state_.rawPositionM[2]);
      const Eigen::Vector3d current(
          rawPositionM[0], rawPositionM[1], rawPositionM[2]);
      const double positionJump = (current - previous).norm();
      const double rotationJump = quaternionAngularDistance(
          state_.rawOrientation, packet.orientation);
      if (positionJump > kMaxRawPositionJump ||
          rotationJump > kMaxRawRotationJump) {
        return;
      }
    }

    state_.rawPositionM = rawPositionM;
    state_.rawOrientation = packet.orientation;
    state_.rawValid = true;

    if (!state_.valid) {
      state_.positionM = rawPositionM;
      state_.orientation = packet.orientation;
      state_.valid = true;
    } else {
      for (std::size_t i = 0; i < 3; ++i) {
        state_.positionM[i] +=
            kPositionFilterAlpha *
            (rawPositionM[i] - state_.positionM[i]);
      }
      Eigen::Quaterniond target = packet.orientation;
      if (state_.orientation.dot(target) < 0.0) target.coeffs() *= -1.0;
      state_.orientation =
          state_.orientation.slerp(kRotationFilterAlpha, target).normalized();
    }

    state_.timestamp = std::chrono::steady_clock::now();
    ++state_.sequence;
  }

  void run() {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
      setError("创建UDP socket失败");
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
      setError("绑定UDP端口失败，确认receive.py没有占用5050端口");
      ::close(fd);
      return;
    }

    std::array<std::uint8_t, 1024> buffer{};
    bool printedSource = false;
    while (running_.load() && !gStopRequested) {
      sockaddr_in sender{};
      socklen_t senderLength = sizeof(sender);
      const ssize_t length =
          ::recvfrom(fd, buffer.data(), buffer.size(), 0,
                     reinterpret_cast<sockaddr *>(&sender), &senderLength);
      if (length < 0) continue;

      HandPacket packet;
      if (!parseRightHandPacket(buffer.data(), static_cast<std::size_t>(length),
                                packet)) {
        continue;
      }
      update(packet);

      if (!printedSource) {
        char ip[INET_ADDRSTRLEN]{};
        ::inet_ntop(AF_INET, &sender.sin_addr, ip, sizeof(ip));
        std::cout << "收到右手柄UDP数据: " << ip << ":"
                  << ntohs(sender.sin_port) << std::endl;
        printedSource = true;
      }
    }
    ::close(fd);
  }

  uint16_t port_;
  std::atomic<bool> running_{false};
  std::atomic<bool> teleoperationActive_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  HandState state_;
  std::string error_;
};

bool waitForFreshHandState(UdpReceiver &receiver, HandState &state,
                           std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    state = receiver.latest();
    if (state.valid &&
        std::chrono::steady_clock::now() - state.timestamp <= kUdpTimeout) {
      return true;
    }
    if (!receiver.error().empty()) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

bool waitForTriggerState(UdpReceiver &receiver, bool pressed, HandState &state,
                         std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    state = receiver.latest();
    if (state.valid &&
        std::chrono::steady_clock::now() - state.timestamp <= kUdpTimeout &&
        state.trigger == pressed) {
      return true;
    }
    if (!receiver.error().empty()) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

bool waitForHorizontalMovement(UdpReceiver &receiver,
                               const Eigen::Vector3d &start,
                               double minimumDistance,
                               HandState &state,
                               std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    state = receiver.latest();
    if (state.valid &&
        std::chrono::steady_clock::now() - state.timestamp <= kUdpTimeout) {
      const double dx = state.rawPositionM[0] - start.x();
      const double dz = state.rawPositionM[2] - start.z();
      if (std::hypot(dx, dz) >= minimumDistance) return true;
    }
    if (!receiver.error().empty()) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

int diagnosePicoPositionMapping() {
  std::cout << "PICO输入诊断模式：不会连接、上电或控制机器人。" << std::endl;
  UdpReceiver receiver(kUdpPort);
  receiver.start();

  HandState state;
  if (!waitForFreshHandState(receiver, state, std::chrono::seconds(15))) {
    const std::string error = receiver.error();
    std::cerr << (error.empty() ? "15秒内未收到有效右手柄UDP数据" : error)
              << std::endl;
    return 1;
  }

  // Require a release before each capture so a held Trigger cannot
  // accidentally record both endpoints from the same position.
  std::cout << "步骤1/2：将右手柄放在起始位置，松开Trigger，然后按下Trigger记录起点。"
            << std::endl;
  if (!waitForTriggerState(receiver, false, state, std::chrono::seconds(60)) ||
      !waitForTriggerState(receiver, true, state, std::chrono::seconds(60))) {
    std::cerr << "等待起点Trigger操作超时或UDP断流" << std::endl;
    return 1;
  }
  const Eigen::Vector3d start(state.rawPositionM[0], state.rawPositionM[1],
                              state.rawPositionM[2]);
  std::cout << std::fixed << std::setprecision(1)
            << "已记录起点(mm): x=" << start.x() * 1000.0
            << ", y=" << start.y() * 1000.0
            << ", z=" << start.z() * 1000.0 << std::endl;

  if (!waitForTriggerState(receiver, false, state, std::chrono::seconds(60))) {
    std::cerr << "等待Trigger松开超时或UDP断流" << std::endl;
    return 1;
  }
  std::cout << "步骤2/2：松开Trigger，只将手柄水平向前移动10到20厘米。"
            << std::endl;
  if (!waitForHorizontalMovement(receiver, start, 0.08, state,
                                 std::chrono::seconds(60))) {
    std::cerr << "等待至少80毫米水平位移超时或UDP断流" << std::endl;
    return 1;
  }
  std::cout << "已检测到足够的水平位移；保持终点不动，松开后再按一次Trigger。"
            << std::endl;
  if (!waitForTriggerState(receiver, false, state, std::chrono::seconds(60))) {
    std::cerr << "等待终点Trigger松开超时或UDP断流" << std::endl;
    return 1;
  }
  if (!waitForTriggerState(receiver, true, state, std::chrono::seconds(60))) {
    std::cerr << "等待终点Trigger操作超时或UDP断流" << std::endl;
    return 1;
  }
  const Eigen::Vector3d end(state.rawPositionM[0], state.rawPositionM[1],
                            state.rawPositionM[2]);
  const Eigen::Vector3d delta = end - start;
  // PICO/OpenXR position uses Y as the vertical axis. Diagnose forward motion
  // only in the horizontal XZ plane; otherwise lowering/raising the controller
  // can be mistaken for forward movement.
  const double horizontalMagnitude =
      std::hypot(delta.x(), delta.z());
  const double verticalMagnitude = std::abs(delta.y());
  const Eigen::Index dominantIndex =
      std::abs(delta.x()) >= std::abs(delta.z()) ? 0 : 2;
  const double dominantMagnitude = std::abs(delta[dominantIndex]);
  const char *axisNames[] = {"X", "Y", "Z"};

  std::cout << "已记录终点(mm): x=" << end.x() * 1000.0
            << ", y=" << end.y() * 1000.0
            << ", z=" << end.z() * 1000.0 << std::endl;
  std::cout << "手柄向前位移增量(mm): dx=" << delta.x() * 1000.0
            << ", dy=" << delta.y() * 1000.0
            << ", dz=" << delta.z() * 1000.0 << std::endl;

  std::cout << "水平位移=" << horizontalMagnitude * 1000.0
            << " mm，高度变化=" << delta.y() * 1000.0 << " mm"
            << std::endl;
  if (horizontalMagnitude < 0.08 || horizontalMagnitude > 0.25) {
    std::cerr << "采样无效：水平前移应保持在80到250毫米，请重试。"
              << std::endl;
    return 2;
  }
  if (verticalMagnitude > horizontalMagnitude * 0.35) {
    std::cerr << "采样无效：高度变化超过水平位移的35%，请保持同一高度重试。"
              << std::endl;
    return 2;
  }

  const double dominantValue = delta[dominantIndex];
  const char sign = dominantValue >= 0.0 ? '+' : '-';
  const double mappedMillimeters =
      dominantValue * gPositionGain * 1000.0;
  std::cout << "诊断结果：PICO的“向前”主要对应 "
            << axisNames[dominantIndex] << sign << "（"
            << dominantMagnitude * 1000.0 << " mm）。" << std::endl;
  std::cout << "PICO水平前向单位向量(X,Y,Z)=("
            << delta.x() / horizontalMagnitude << ", 0.000, "
            << delta.z() / horizontalMagnitude << ")。" << std::endl;
  std::cout << "当前pico_servol没有坐标变换，会把它直接映射为机器人参考系 "
            << axisNames[dominantIndex] << sign << "，增益后约 "
            << mappedMillimeters << " mm。" << std::endl;
  std::cout << "如果机器人实际向后运动，则可确认是PICO与机器人参考系的轴/符号映射错误，"
               "不是滤波或增益造成的。"
            << std::endl;
  return 0;
}

int gripperPositionForTrigger(bool trigger) {
  return trigger ? kGripperPressedPosition : kGripperReleasedPosition;
}

class GripperPublisher {
 public:
  explicit GripperPublisher(UdpReceiver &receiver)
      : receiver_(receiver),
        node_(std::make_shared<rclcpp::Node>("pico_gripper_control")),
        publisher_(node_->create_publisher<std_msgs::msg::Int32>(
            "/sts3215/position", 10)) {}

  ~GripperPublisher() { stop(); }

  void start() {
    running_.store(true);
    thread_ = std::thread(&GripperPublisher::run, this);
  }

  void stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
  }

 private:
  void update(bool trigger) {
    rclcpp::spin_some(node_);
    const int position = gripperPositionForTrigger(trigger);
    const std::size_t subscriptions = publisher_->get_subscription_count();
    const bool subscriberJustAppeared =
        lastSubscriptionCount_ == 0 && subscriptions > 0;
    if (!initialized_ || position != lastPosition_ ||
        subscriberJustAppeared) {
      std_msgs::msg::Int32 message;
      message.data = position;
      publisher_->publish(message);
      if (!initialized_ || position != lastPosition_) {
        std::cout << "Trigger " << (trigger ? "pressed" : "released")
                  << ": gripper target=" << position << std::endl;
      }
      if (subscriptions == 0 && !missingDriverWarningPrinted_) {
        std::cerr << "No /sts3215/position subscriber detected; start the "
                     "sts3215_driver node"
                  << std::endl;
        missingDriverWarningPrinted_ = true;
      }
      initialized_ = true;
      lastPosition_ = position;
    }
    lastSubscriptionCount_ = subscriptions;
  }

  void run() {
    while (running_.load() && !gStopRequested) {
      const HandState state = receiver_.latest();
      if (state.valid &&
          std::chrono::steady_clock::now() - state.timestamp <= kUdpTimeout) {
        update(state.trigger);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  UdpReceiver &receiver_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  bool initialized_{false};
  bool missingDriverWarningPrinted_{false};
  int lastPosition_{kGripperReleasedPosition};
  std::size_t lastSubscriptionCount_{0};
};

class DatasetStatePublisher {
 public:
  DatasetStatePublisher()
      : node_(std::make_shared<rclcpp::Node>("pico_dataset_state")),
        statePublisher_(node_->create_publisher<sensor_msgs::msg::JointState>(
            "/xcore/right/record_state", 10)),
        actionPublisher_(node_->create_publisher<sensor_msgs::msg::JointState>(
            "/xcore/right/record_action", 10)),
        activePublisher_(node_->create_publisher<std_msgs::msg::Bool>(
            "/xcore/right/teleop_active", 10)),
        buttonsPublisher_(node_->create_publisher<std_msgs::msg::UInt8>(
            "/xcore/right/pico_buttons", 10)),
        stopSubscription_(node_->create_subscription<std_msgs::msg::Bool>(
            "/xcore/right/record_stop_teleop", 10,
            [this](const std_msgs::msg::Bool &message) {
              if (message.data) stopTeleopRequested_.store(true);
            })),
        gripperStateSubscription_(
            node_->create_subscription<std_msgs::msg::Int32>(
                "/sts3215/state", 10,
                [this](const std_msgs::msg::Int32 &message) {
                  if (message.data >= 0 && message.data <= 4095) {
                    gripperMeasuredPosition_.store(message.data);
                    gripperStateReady_.store(true);
                  }
                })) {
    for (std::size_t i = 0; i < 7; ++i) {
      stateNames_.push_back("joint_" + std::to_string(i + 1) + ".pos");
    }
    stateNames_.push_back("gripper.pos");
    actionNames_ = stateNames_;
  }

  void publishActive(bool active) {
    std_msgs::msg::Bool message;
    message.data = active;
    activePublisher_->publish(message);
    rclcpp::spin_some(node_);
  }

  void publishSample(const std::array<double, 7> &measuredJoints,
                     bool trigger) {
    rclcpp::spin_some(node_);

    // A missing gripper observation must not be replaced with its target: that
    // would silently make observation and action mean the same thing.
    if (!gripperStateReady_.load()) {
      if (++missingGripperStateCycles_ % 250 == 1) {
        std::cerr << "尚未收到/sts3215/state，本周期不发布录制样本"
                  << std::endl;
      }
      return;
    }

    sensor_msgs::msg::JointState stateMessage;
    stateMessage.header.stamp = node_->now();
    stateMessage.name = stateNames_;
    stateMessage.position.assign(measuredJoints.begin(), measuredJoints.end());
    stateMessage.position.push_back(
        static_cast<double>(gripperMeasuredPosition_.load()));
    statePublisher_->publish(stateMessage);

    sensor_msgs::msg::JointState actionMessage;
    actionMessage.header.stamp = stateMessage.header.stamp;
    actionMessage.name = actionNames_;
    // Match the original working recorder and the SDK servol_demo: joint
    // angles are read from jointPos_m.  ServoL Cartesian mode on this
    // controller does not provide a usable jointPos_c stream.
    actionMessage.position.assign(measuredJoints.begin(),
                                  measuredJoints.end());
    actionMessage.position.push_back(
        static_cast<double>(gripperPositionForTrigger(trigger)));
    actionPublisher_->publish(actionMessage);

    // Re-publish the active level with every sample so a recorder started
    // after ServoL has already begun still observes the current state.
    std_msgs::msg::Bool activeMessage;
    activeMessage.data = true;
    activePublisher_->publish(activeMessage);
    rclcpp::spin_some(node_);
  }

  void publishButtons(std::uint8_t buttons) {
    std_msgs::msg::UInt8 message;
    message.data = buttons;
    buttonsPublisher_->publish(message);
    rclcpp::spin_some(node_);
  }

  bool stopTeleopRequested() const {
    return stopTeleopRequested_.load();
  }

  void clearStopTeleopRequest() {
    stopTeleopRequested_.store(false);
  }

 private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr statePublisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr actionPublisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr activePublisher_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr buttonsPublisher_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stopSubscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr
      gripperStateSubscription_;
  std::atomic<bool> stopTeleopRequested_{false};
  std::atomic<bool> gripperStateReady_{false};
  std::atomic<int> gripperMeasuredPosition_{kGripperReleasedPosition};
  std::size_t missingGripperStateCycles_{0};
  std::vector<std::string> stateNames_;
  std::vector<std::string> actionNames_;
};

bool checkEc(const error_code &ec, const char *operation) {
  if (!ec) return true;
  std::cerr << operation << "失败: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

bool loadActiveJointLimits(ArRobot &robot) {
  gJointLimitsReady = false;
  std::array<double[2], 7> controllerLimits{};
  error_code ec;
  const bool softLimitsEnabled = robot.getSoftLimit(controllerLimits, ec);
  if (!checkEc(ec, "getSoftLimit")) return false;
  if (!softLimitsEnabled) {
    std::cerr << "控制器软限位未启用，拒绝进入遥操" << std::endl;
    return false;
  }

  for (std::size_t i = 0; i < gActiveLowerLimit.size(); ++i) {
    const double controllerLower =
        controllerLimits[i][0] + kJointLimitMargin;
    const double controllerUpper =
        controllerLimits[i][1] - kJointLimitMargin;
    gActiveLowerLimit[i] = std::max(kLowerLimit[i], controllerLower);
    gActiveUpperLimit[i] = std::min(kUpperLimit[i], controllerUpper);
    if (!std::isfinite(gActiveLowerLimit[i]) ||
        !std::isfinite(gActiveUpperLimit[i]) ||
        gActiveLowerLimit[i] >= gActiveUpperLimit[i]) {
      std::cerr << "应用安全余量后J" << i + 1
                << "没有有效可用范围" << std::endl;
      return false;
    }
  }
  gJointLimitsReady = true;
  return true;
}

bool insideJointLimits(const std::array<double, 7> &joints) {
  if (!gJointLimitsReady) {
    std::cerr << "关节限位尚未初始化" << std::endl;
    return false;
  }
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (!std::isfinite(joints[i]) || joints[i] < gActiveLowerLimit[i] ||
        joints[i] > gActiveUpperLimit[i]) {
      std::cerr << "关节安全边界触发: J" << i + 1 << "=" << joints[i]
                << " rad, limit=[" << gActiveLowerLimit[i] << ", "
                << gActiveUpperLimit[i] << "] rad" << std::endl;
      return false;
    }
  }
  return true;
}

bool waitRobotIdle(BaseRobot &robot,
                   std::chrono::seconds timeout = std::chrono::seconds(60)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    error_code ec;
    const auto state = robot.operationState(ec);
    if (!checkEc(ec, "operationState")) return false;
    if (state == OperationState::idle) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cerr << "等待机器人停止超时或收到Ctrl+C" << std::endl;
  return false;
}

bool switchToNrt(ArRobot &robot) {
  error_code ec;
  for (int attempt = 0; attempt < 20; ++attempt) {
    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!ec) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return checkEc(ec, "切换NrtCommand模式");
}

bool restoreNrtReady(ArRobot &robot) {
  if (!switchToNrt(robot)) return false;

  error_code ec;
  robot.setOperateMode(OperateMode::automatic, ec);
  if (!checkEc(ec, "恢复automatic模式")) return false;

  robot.setPowerState(true, ec);
  if (!checkEc(ec, "恢复机器人上电")) return false;

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    ec.clear();
    const auto state = robot.powerState(ec);
    if (!checkEc(ec, "确认机器人上电状态")) return false;
    if (state == PowerState::on) {
      std::cout << "NRT模式、自动模式和上电状态已恢复" << std::endl;
      return true;
    }
    if (state == PowerState::estop) {
      std::cerr << "恢复上电失败：急停已按下" << std::endl;
      return false;
    }
    if (state == PowerState::gstop) {
      std::cerr << "恢复上电失败：安全门已打开" << std::endl;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cerr << "恢复上电超时：机器人未进入PowerState::on" << std::endl;
  return false;
}

void stopNrtMotion(ArRobot &robot) {
  error_code ec;
  robot.stop(ec);
  if (ec) {
    std::cerr << "stop(复位运动)失败: code=" << ec.value()
              << ", message=" << ec.message() << std::endl;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < deadline) {
    ec.clear();
    if (robot.operationState(ec) == OperationState::idle) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  ec.clear();
  robot.moveReset(ec);
  if (ec) {
    std::cerr << "moveReset(停止复位运动)失败: code=" << ec.value()
              << ", message=" << ec.message() << std::endl;
  }
}

bool resetRobot(ArRobot &robot) {
  std::cout << "A/down触发回初始位置，执行MoveAbsJ..." << std::endl;
  if (!insideJointLimits(gInitialJoint)) return false;
  if (!restoreNrtReady(robot)) return false;

  error_code ec;
  robot.moveReset(ec);
  if (!checkEc(ec, "moveReset")) return false;

  MoveAbsJCommand command(
      JointPosition(
          std::vector<double>(gInitialJoint.begin(), gInitialJoint.end())),
      1000, 10);
  std::string commandId;
  robot.moveAppend({command}, commandId, ec);
  if (!checkEc(ec, "moveAppend(复位)")) return false;
  robot.moveStart(ec);
  if (!checkEc(ec, "moveStart(复位)")) return false;
  if (!waitRobotIdle(robot)) {
    stopNrtMotion(robot);
    return false;
  }

  const auto joints = robot.jointPos(ec);
  if (!checkEc(ec, "jointPos(复位后)")) return false;
  std::cout << "MoveAbsJ复位完成，复位后J6=" << joints[5] << " rad"
            << std::endl;
  return true;
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

bool poseFilterSelfTest() {
  const Eigen::Vector3d scaledPosition =
      kDefaultPositionGain * Eigen::Vector3d(0.10, 0.0, 0.0);
  const bool positionGainOk =
      std::abs(scaledPosition.x() - 0.0618) < 1e-9 &&
      std::abs(scaledPosition.y()) < 1e-9 &&
      std::abs(scaledPosition.z()) < 1e-9;

  const Eigen::Matrix3d rawRotation =
      Eigen::AngleAxisd(90.0 * kPi / 180.0,
                        Eigen::Vector3d::UnitZ())
          .toRotationMatrix();
  const Eigen::AngleAxisd mappedRotation(scaledRotation(rawRotation));
  const bool rotationGainOk =
      std::abs(mappedRotation.angle() - 22.5 * kPi / 180.0) < 1e-9;

  const Eigen::Vector3d stepped =
      limitedPositionStep(Eigen::Vector3d::Zero(),
                          Eigen::Vector3d(0.01, 0.0, 0.0));
  const bool rateLimitOk =
      std::abs(stepped.norm() - kMaxPositionStep) < 1e-9;

  const double elbowStart = 0.2;
  ElbowControlState positiveElbowState;
  ElbowControlState negativeElbowState;
  ElbowControlState deadbandElbowState;
  ElbowControlState upperLimitElbowState;
  ElbowControlState lowerLimitElbowState;
  const double positiveElbow =
      nextElbowCommand(elbowStart, 1.0, positiveElbowState);
  const double negativeElbow =
      nextElbowCommand(elbowStart, -1.0, negativeElbowState);
  const bool elbowMappingOk =
      positiveElbow > elbowStart && negativeElbow < elbowStart &&
      nextElbowCommand(elbowStart, kJoystickDeadband,
                       deadbandElbowState) == elbowStart &&
      nextElbowCommand(kMaxElbow, 1.0,
                       upperLimitElbowState) == kMaxElbow &&
      nextElbowCommand(kMinElbow, -1.0,
                       lowerLimitElbowState) == kMinElbow;
  const bool elbowSmoothingOk =
      positiveElbowState.filteredJoystickY > 0.0 &&
      positiveElbowState.filteredJoystickY < 1.0 &&
      std::abs(positiveElbow - elbowStart) <= kMaxElbowStep + 1e-12 &&
      std::abs(negativeElbow - elbowStart) <= kMaxElbowStep + 1e-12 &&
      std::abs(positiveElbowState.velocity) <=
          kMaxElbowAcceleration * kPlanPeriod + 1e-12;
  ElbowControlState webElbowState;
  const double webElbow =
      nextElbowTargetCommand(elbowStart, 1.0, webElbowState);
  const bool webElbowTargetOk =
      webElbow > elbowStart && webElbow < 1.0 &&
      std::abs(webElbow - elbowStart) <= kMaxElbowStep + 1e-12 &&
      webElbowState.velocity > 0.0 &&
      webElbowState.velocity <=
          kMaxElbowAcceleration * kPlanPeriod + 1e-12;
  bool elbowStepLimitOk = true;
  double stepLimitedElbow = 0.0;
  ElbowControlState stepLimitState;
  for (int cycle = 0; cycle < 200; ++cycle) {
    const double previous = stepLimitedElbow;
    stepLimitedElbow =
        nextElbowTargetCommand(stepLimitedElbow, 1.0, stepLimitState);
    elbowStepLimitOk = elbowStepLimitOk &&
        stepLimitedElbow > previous &&
        stepLimitedElbow - previous <= kMaxElbowStep + 1e-12;
  }

  const bool ok =
      positionGainOk && rotationGainOk && rateLimitOk && elbowMappingOk &&
      elbowSmoothingOk && webElbowTargetOk && elbowStepLimitOk;
  std::cout << "位姿滤波自检: " << (ok ? "通过" : "失败") << std::endl;
  return ok;
}

bool gripperMappingSelfTest() {
  const bool ok =
      gripperPositionForTrigger(false) == kGripperReleasedPosition &&
      gripperPositionForTrigger(true) == kGripperPressedPosition;
  std::cout << "Trigger夹爪映射自检: " << (ok ? "通过" : "失败")
            << std::endl;
  return ok;
}

bool handStateFresh(const HandState &state) {
  return state.valid &&
         std::chrono::steady_clock::now() - state.timestamp <= kUdpTimeout;
}

bool teleoperationStartRequested(const HandState &state) {
  return handStateFresh(state) && state.grip && !state.down;
}

bool updateLatestMeasuredJointState(ArRobot &robot,
                                    std::array<double, 7> &measuredJoints) {
  bool updated = robot.updateRobotState(std::chrono::milliseconds(1));
  while (robot.updateRobotState(
      std::chrono::steady_clock::duration::zero())) {
    updated = true;
  }
  return updated && robot.getStateData(RtSupportedFields::jointPos_m,
                                       measuredJoints) == 0;
}

bool runServo(ArRobot &robot, UdpReceiver &receiver,
              DatasetStatePublisher &datasetPublisher,
              const HandState &handAnchor,
              bool &returnToInitialRequested, ElbowIpc &elbowIpc,
              bool elbowOnly, double initialElbowTarget) {
  returnToInitialRequested = false;
  if (!handStateFresh(handAnchor) ||
      (!elbowOnly && !teleoperationStartRequested(handAnchor))) {
    std::cerr << "右手柄数据已超时，不能开启摇操" << std::endl;
    return true;
  }

  error_code ec;
  const auto robotEndPose = robot.posture(CoordinateType::endInRef, ec);
  if (!checkEc(ec, "读取ServoL末端锚点")) return false;
  const auto robotEndCartesian =
      robot.cartPosture(CoordinateType::endInRef, ec);
  if (!checkEc(ec, "读取ServoL完整笛卡尔锚点")) return false;
  if (!std::isfinite(robotEndCartesian.elbow) ||
      robotEndCartesian.elbow < kMinElbow ||
      robotEndCartesian.elbow > kMaxElbow) {
    std::cerr << "机器人返回的臂角数值无效，拒绝进入带臂角控制的ServoL: "
              << robotEndCartesian.elbow << " rad" << std::endl;
    return false;
  }
  if (!robotEndCartesian.hasElbow) {
    // ARC controllers in this SDK may return a usable elbow value (the J3
    // angle) while leaving hasElbow false.  Use the finite, range-checked
    // value as the anchor and explicitly enable elbow control on commands.
    std::cerr << "警告：控制器返回hasElbow=false；按ARC定义使用当前J3/臂角 "
              << robotEndCartesian.elbow << " rad作为ServoL初始臂角"
              << std::endl;
  }
  const auto baseFrame = robot.baseFrame(ec);
  if (!checkEc(ec, "读取baseFrame")) return false;
  const auto toolset = robot.toolset(ec);
  if (!checkEc(ec, "读取toolset")) return false;

  std::array<double, 16> robotEndMatrix{};
  Utils::postureToTransArray(robotEndPose, robotEndMatrix);
  Eigen::Matrix3d robotAnchorRotation;
  Eigen::Vector3d robotAnchorPosition;
  Utils::arrayToTransMatrix(robotEndMatrix, robotAnchorRotation,
                            robotAnchorPosition);

  robot.setRtNetworkTolerance(20, ec);
  if (!checkEc(ec, "setRtNetworkTolerance")) return false;
  robot.setMotionControlMode(MotionControlMode::RtCommand, ec);
  if (!checkEc(ec, "切换RtCommand模式")) return false;

  auto rtController = robot.getRtMotionController().lock();
  if (!rtController) {
    std::cerr << "获取实时运动控制器失败" << std::endl;
    switchToNrt(robot);
    return false;
  }

  bool receivingState = false;
  bool servoStarted = false;
  CartesianPosition command;
  const auto initialFlangePose =
      Utils::EndInRefToFlanInBase(baseFrame, toolset, robotEndPose);
  Utils::postureToTransArray(initialFlangePose, command.pos);
  command.hasElbow = true;
  command.elbow = robotEndCartesian.elbow;
  command.confData = robotEndCartesian.confData;
  command.external = robotEndCartesian.external;

  auto finishServo = [&]() {
    bool success = true;
    receiver.setTeleoperationActive(false);
    datasetPublisher.clearStopTeleopRequest();
    datasetPublisher.publishActive(false);
    if (servoStarted) {
      try {
        // 不再发送带setFinished的笛卡尔点，避免结束时产生位置跳变。
        rtController->stopMove();
        rtController->stopServoJoint();
      } catch (const std::exception &e) {
        std::cerr << "结束ServoL异常: " << e.what() << std::endl;
        success = false;
      }
      servoStarted = false;
    }
    if (receivingState) {
      robot.stopReceiveRobotState();
      receivingState = false;
    }

    error_code idleEc;
    robot.setMotionControlMode(MotionControlMode::Idle, idleEc);
    if (!checkEc(idleEc, "切换Idle模式")) success = false;
    return success;
  };

  try {
    robot.startReceiveRobotState(
        std::chrono::milliseconds(1),
        {RtSupportedFields::jointPos_m});
    receivingState = true;
    while (robot.updateRobotState(std::chrono::steady_clock::duration::zero())) {
    }

    std::array<double, 7> measuredJoints{};
    if (!updateLatestMeasuredJointState(robot, measuredJoints) ||
        !insideJointLimits(measuredJoints)) {
      std::cerr << "读取关节状态失败或当前关节超出安全边界" << std::endl;
      finishServo();
      return false;
    }
    rtController->setServoJoint(kPlanPeriod, kPlanPeriod, 0, ec);
    if (!checkEc(ec, "setServoJoint")) {
      finishServo();
      return false;
    }
    rtController->startMove(RtControllerMode::cartesianPosition);
    servoStarted = true;

    // Real-time setup takes time. Re-anchor from the latest sample so setup
    // latency cannot become an unintended first command, matching dual-arm.
    const HandState activationHand = receiver.latest();
    if (!handStateFresh(activationHand)) {
      std::cout << "右手柄数据超时，取消进入摇操" << std::endl;
      return finishServo();
    }
    receiver.setTeleoperationActive(!elbowOnly);
    const Eigen::Vector3d handAnchorPosition(
        activationHand.positionM[0], activationHand.positionM[1],
        activationHand.positionM[2]);
    const Eigen::Matrix3d handAnchorRotation =
        activationHand.orientation.toRotationMatrix();
    Eigen::Vector3d commandedPosition = robotAnchorPosition;
    Eigen::Matrix3d commandedRotation = robotAnchorRotation;
    double commandedElbow = robotEndCartesian.elbow;
    double externalElbowTarget = elbowOnly
        ? clamp(initialElbowTarget, kMinElbow, kMaxElbow)
        : commandedElbow;
    bool externalElbowEnabled = elbowOnly;
    ElbowControlState elbowControl;
    bool gripWasPressed = activationHand.grip;
    bool downWasPressed = activationHand.down;
    std::string stopReason = elbowOnly
        ? "点击Grip进入正常遥操"
        : "再次点击Grip停止摇操";
    std::string lastRejectedReason;
    std::size_t rejectedCycles = 0;
    std::size_t controlCycle = 0;
    auto nextCycle = std::chrono::steady_clock::now();
    auto nextElbowLog = nextCycle;

    datasetPublisher.publishActive(!elbowOnly);

    std::cout << (elbowOnly
        ? "网页臂角控制已开启：保持当前末端位姿；点击Grip切换正常遥操。"
        : "单臂遥操已开启：再次点击Grip停止；按A/down回初始位置。")
              << std::endl;
    std::cout << "输入参数: position gain=" << gPositionGain
              << ", rotation gain=" << kRotationGain
              << ", raw jump limit=" << kMaxRawPositionJump * 1000.0
              << " mm / " << kMaxRawRotationJump * 180.0 / kPi
              << " deg, max step=" << kMaxPositionStep * 1000.0
              << " mm / " << kMaxRotationStep * 180.0 / kPi
              << " deg" << std::endl;
    std::cout << "臂角控制: 初始=" << commandedElbow * 180.0 / kPi
              << " deg，摇杆前推增大/后推减小，死区="
              << kJoystickDeadband << "，最大速度="
              << kMaxElbowVelocity * 180.0 / kPi << " deg/s，低通时间常数="
              << kJoystickLowPassTimeConstant * 1000.0
              << " ms，最大加速度="
              << kMaxElbowAcceleration * 180.0 / kPi << " deg/s^2"
              << "，单周期最大变化="
              << kMaxElbowStep * 180.0 / kPi << " deg"
              << std::endl;
    std::cout << "录制关节来源: observation/action均读取q_m(实测); "
                 "与SDK servol_demo一致，不读取q_c、不解外部IK。"
              << std::endl;

    while (!gStopRequested) {
      const HandState hand = receiver.latest();
      datasetPublisher.publishButtons(hand.buttons);
      if (datasetPublisher.stopTeleopRequested()) {
        stopReason = "录制达到最大帧数，请求停止摇操";
        break;
      }
      if (!handStateFresh(hand)) {
        stopReason = "UDP数据超过250 ms未更新，安全停止摇操";
        break;
      }

      const bool gripPressedNow = hand.grip;
      const bool downPressedNow = hand.down;
      if (downPressedNow && !downWasPressed) {
        returnToInitialRequested = true;
        stopReason = "A/down已按下，停止摇操并回初始位置";
        break;
      }
      if (gripPressedNow && !gripWasPressed) {
        stopReason = elbowOnly
            ? "Grip已按下，退出网页臂角专用控制"
            : "再次点击Grip，停止摇操";
        break;
      }
      gripWasPressed = gripPressedNow;
      downWasPressed = downPressedNow;

      if (!updateLatestMeasuredJointState(robot, measuredJoints) ||
          !insideJointLimits(measuredJoints)) {
        stopReason = "关节安全边界触发，停止摇操";
        break;
      }
      if (elbowOnly) {
        ElbowIpcCommand request;
        if (elbowIpc.readRightCommand(request)) {
          if (!request.enabled) {
            externalElbowTarget = commandedElbow;
            stopReason = "网页已关闭臂角控制";
            break;
          }
          if (request.applyTarget) {
            externalElbowTarget = request.target;
            elbowControl.filteredJoystickY = 0.0;
          }
        }
        commandedElbow = nextElbowTargetCommand(
            commandedElbow, externalElbowTarget, elbowControl);
        command.elbow = commandedElbow;
        rtController->sendCommand(command);
      } else {
      const Eigen::Vector3d handPosition(hand.positionM[0], hand.positionM[1],
                                         hand.positionM[2]);
      const Eigen::Vector3d rawPositionOffset =
          handPosition - handAnchorPosition;
      const Eigen::Matrix3d rawRotationOffset =
          hand.orientation.toRotationMatrix() * handAnchorRotation.transpose();
      const Eigen::AngleAxisd rawRotationAngle(rawRotationOffset);

      if (rawPositionOffset.norm() > kMaxRawPositionOffset ||
          std::abs(rawRotationAngle.angle()) > kMaxRawRotationOffset) {
        const std::string reason = "手柄偏移超过原始安全范围";
        rtController->sendCommand(command);
        ++rejectedCycles;
        if (reason != lastRejectedReason || rejectedCycles % 25 == 1) {
          std::cerr << "保持最后有效目标: " << reason << std::endl;
          lastRejectedReason = reason;
        }
        if (rejectedCycles >= kMaxRejectedCycles) {
          stopReason = "目标连续无效2秒，安全停止摇操";
          break;
        }
      } else {
        ElbowIpcCommand request;
        if (elbowIpc.readRightCommand(request)) {
          externalElbowEnabled = request.enabled;
          if (request.enabled && request.applyTarget) {
            externalElbowTarget = request.target;
            elbowControl.filteredJoystickY = 0.0;
            std::cout << "网页臂角目标已更新为 "
                      << externalElbowTarget * 180.0 / kPi << " deg"
                      << std::endl;
          }
        }
        const Eigen::Vector3d mappedPositionOffset =
            gPositionGain * rawPositionOffset;
        const Eigen::Matrix3d mappedRotationOffset =
            scaledRotation(rawRotationOffset);
        const Eigen::Vector3d targetPosition =
            robotAnchorPosition + mappedPositionOffset;
        const Eigen::Matrix3d targetRotation =
            mappedRotationOffset * robotAnchorRotation;
        commandedPosition =
            limitedPositionStep(commandedPosition, targetPosition);
        commandedRotation =
            limitedRotationStep(commandedRotation, targetRotation);
        commandedElbow = externalElbowEnabled
            ? nextElbowTargetCommand(commandedElbow, externalElbowTarget,
                                     elbowControl)
            : nextElbowCommand(commandedElbow, hand.joystick[1], elbowControl);

        std::array<double, 16> targetEndMatrix{};
        Utils::transMatrixToArray(commandedRotation, commandedPosition,
                                  targetEndMatrix);
        std::array<double, 6> targetEndPose{};
        Utils::transArrayToPosture(targetEndMatrix, targetEndPose);
        const auto targetFlangePose =
            Utils::EndInRefToFlanInBase(baseFrame, toolset, targetEndPose);
        Utils::postureToTransArray(targetFlangePose, command.pos);
        command.elbow = commandedElbow;
        rtController->sendCommand(command);
        rejectedCycles = 0;
        lastRejectedReason.clear();
      }
      }

      if (!elbowOnly && controlCycle % kDatasetPublishDivider == 0) {
        datasetPublisher.publishSample(measuredJoints, hand.trigger);
      }

      const auto logNow = std::chrono::steady_clock::now();
      if (logNow >= nextElbowLog) {
        elbowIpc.writeRightStatus(
            commandedElbow,
            externalElbowEnabled ? externalElbowTarget : commandedElbow,
            true);
        std::cout << std::fixed << std::setprecision(4)
                  << "[臂角 5Hz] 指令=" << commandedElbow << " rad ("
                  << commandedElbow * 180.0 / kPi
                  << " deg)，实测J3=" << measuredJoints[2] << " rad ("
                  << measuredJoints[2] * 180.0 / kPi << " deg)，速度="
                  << elbowControl.velocity * 180.0 / kPi
                  << " deg/s，滤波摇杆Y="
                  << elbowControl.filteredJoystickY
                  << std::endl;
        do {
          nextElbowLog += kElbowLogPeriod;
        } while (nextElbowLog <= logNow);
      }

      ++controlCycle;
      nextCycle += kPlanCycle;
      const auto now = std::chrono::steady_clock::now();
      if (now > nextCycle + kMaxCycleLateness) {
        stopReason = "100 Hz控制循环超时超过40 ms";
        break;
      }
      std::this_thread::sleep_until(nextCycle);
    }

    if (gStopRequested) stopReason = "收到Ctrl+C，停止摇操";
    elbowIpc.writeRightStatus(
        commandedElbow,
        externalElbowEnabled ? externalElbowTarget : commandedElbow,
        false);
    std::cout << stopReason << std::endl;
    return finishServo();
  } catch (const std::exception &e) {
    std::cerr << "ServoL异常: " << e.what() << std::endl;
    finishServo();
    return false;
  }
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (ec) return;
  std::cout << "最近5条控制器错误日志:" << std::endl;
  for (const auto &log : logs) {
    std::cout << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

std::unique_ptr<ArRobot> connectReadyRobot() {
  auto robot = std::make_unique<ArRobot>(kRobotIp, kLocalIp);
  if (!restoreNrtReady(*robot)) return nullptr;
  if (!loadActiveJointLimits(*robot)) return nullptr;
  return robot;
}

class RosContextGuard {
 public:
  RosContextGuard(int argc, char **argv) { rclcpp::init(argc, argv); }
  ~RosContextGuard() {
    if (rclcpp::ok()) rclcpp::shutdown();
  }

  RosContextGuard(const RosContextGuard &) = delete;
  RosContextGuard &operator=(const RosContextGuard &) = delete;
};

}  // namespace

int main(int argc, char **argv) {
  gPositionGain = positionGainFromEnvironment();
  gInitialJoint = initialJointFromEnvironment();
  ElbowIpc elbowIpc;
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    const bool parserOk = parserSelfTest();
    const bool filterOk = poseFilterSelfTest();
    const bool gripperOk = gripperMappingSelfTest();
    return parserOk && filterOk && gripperOk ? 0 : 1;
  }

  std::signal(SIGINT, signalHandler);
  if (argc == 2 && std::string(argv[1]) == "--diagnose-input") {
    return diagnosePicoPositionMapping();
  }
  RosContextGuard rosContext(argc, argv);

  UdpReceiver receiver(kUdpPort);
  receiver.start();
  GripperPublisher gripper(receiver);
  gripper.start();
  DatasetStatePublisher datasetPublisher;
  datasetPublisher.publishActive(false);
  std::cout << "柄臂位置比例: " << std::fixed << std::setprecision(3)
            << gPositionGain << "（旋转比例保持 " << kRotationGain << "）"
            << std::endl;
  std::cout << "右臂复位初始点(rad):";
  for (const double joint : gInitialJoint) std::cout << ' ' << joint;
  std::cout << std::endl;
  std::cout << "监听右手柄UDP端口 " << kUdpPort
            << "，请先关闭占用该端口的receive.py" << std::endl;

  try {
    std::cout << "等待右手柄数据..." << std::endl;
    while (!gStopRequested && !receiver.latest().valid) {
      const std::string receiverError = receiver.error();
      if (!receiverError.empty()) {
        std::cerr << receiverError << std::endl;
        return 1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (gStopRequested) return 0;

    HandState state = receiver.latest();

    auto robot = connectReadyRobot();
    if (!robot) return 1;
    {
      error_code elbowEc;
      const auto cartesian =
          robot->cartPosture(CoordinateType::endInRef, elbowEc);
      if (!elbowEc && std::isfinite(cartesian.elbow)) {
        elbowIpc.writeRightStatus(cartesian.elbow, cartesian.elbow, false);
      }
    }

    state = receiver.latest();
    bool gripArmed = !state.grip;
    bool resetArmed = !state.down;
    std::cout << "就绪：点击Grip开启/停止右臂遥操；按A/down回初始位置；"
                 "Trigger松开/按下控制夹爪900/1500。"
              << std::endl;

    while (!gStopRequested) {
      state = receiver.latest();
      datasetPublisher.publishButtons(state.buttons);
      if (!handStateFresh(state)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }

      if (!state.grip) gripArmed = true;
      if (!state.down) resetArmed = true;

      if (resetArmed && state.down && !state.grip) {
        resetArmed = false;
        if (!resetRobot(*robot)) {
          printControllerErrors(*robot);
          return 1;
        }
        state = receiver.latest();
      } else {
        const bool normalTeleoperationRequested =
            gripArmed && teleoperationStartRequested(state);
        ElbowIpcCommand idleElbowCommand;
        const bool elbowCommandReceived =
            !normalTeleoperationRequested &&
            elbowIpc.readRightCommand(idleElbowCommand);
        const bool elbowOnlyRequested =
            elbowCommandReceived && idleElbowCommand.enabled &&
            idleElbowCommand.applyTarget;
        if (elbowCommandReceived) {
          error_code elbowEc;
          const auto cartesian =
              robot->cartPosture(CoordinateType::endInRef, elbowEc);
          if (!elbowEc && std::isfinite(cartesian.elbow)) {
            elbowIpc.writeRightStatus(
                cartesian.elbow,
                idleElbowCommand.applyTarget
                    ? idleElbowCommand.target
                    : cartesian.elbow,
                false);
          }
        }
        if (normalTeleoperationRequested || elbowOnlyRequested) {
        if (normalTeleoperationRequested) gripArmed = false;
        bool returnToInitialRequested = false;
        if (!runServo(*robot, receiver, datasetPublisher, state,
                      returnToInitialRequested, elbowIpc,
                      elbowOnlyRequested, idleElbowCommand.target)) {
          printControllerErrors(*robot);
          return 1;
        }

        // 控制器从RT退出后，同一SDK会话无法可靠恢复上电。
        // 销毁旧会话并按op.cpp的顺序新建NRT会话。
        robot.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        robot = connectReadyRobot();
        if (!robot) {
          std::cerr << "ServoL结束后重新建立NRT会话失败" << std::endl;
          return 1;
        }
        {
          error_code elbowEc;
          const auto cartesian =
              robot->cartPosture(CoordinateType::endInRef, elbowEc);
          if (!elbowEc && std::isfinite(cartesian.elbow)) {
            elbowIpc.writeRightStatus(cartesian.elbow, cartesian.elbow, false);
          }
        }

        if (returnToInitialRequested) {
          // Ignore the same held A/down press after reset; it must be released
          // before another return can be requested.
          resetArmed = false;
          if (!resetRobot(*robot)) {
            printControllerErrors(*robot);
            return 1;
          }
        }

        state = receiver.latest();
        std::cout << "ServoL已停止；PICO继续等待网页臂角或Grip命令。"
                  << std::endl;
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    switchToNrt(*robot);
    datasetPublisher.publishActive(false);
    std::cout << "程序退出" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "程序异常: " << e.what() << std::endl;
    return 1;
  }
}
