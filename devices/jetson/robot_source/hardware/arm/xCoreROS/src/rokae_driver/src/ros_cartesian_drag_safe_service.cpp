/** Shared-connection Cartesian drag for PICO A toggles.
 * Monitor/load/physical-button path adapted from SDK cxl/cartesian_drag_safe.cpp.
 * The SDK example and independent subprocess wrapper remain unchanged.
 */
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "rclcpp/rclcpp.hpp"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_driver/local_joint_feedback.hpp"
#include "rokae/robot.h"
#include "rokae/utility.h"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace rokae;
namespace {
struct MonitorResult { bool safe; std::string reason; };
struct SessionResult { bool success; bool cleaned; std::string message; };
using JointSample = std::function<void(const std::array<double, 7> &)>;
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
// Measured recording telemetry has its own cadence. Guard samples and their
// two-sample speed debounce remain on the original 50 ms monitor cadence.
constexpr auto kJointSamplePeriod = std::chrono::milliseconds(10);
constexpr auto kTcpPrintPeriod = std::chrono::milliseconds(200);

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

template <typename Robot, typename Force>
MonitorResult monitorDrag(Robot &robot, Force &forceControl,
                          const std::array<double, 6> &startPose,
                          double duration, std::atomic<bool> &stopRequested,
                          const std::function<bool()> &running = [] { return rclcpp::ok(); },
                          const JointSample &onSample = {}, const char *traceSide = "test") {
  error_code ec;
  auto previousPose = rokae_driver::traceFeedbackCall(traceSide, "drag.initial_posture", [&] {
    return robot.posture(CoordinateType::endInRef, ec);
  });
  if (!checkError(ec, "read initial TCP pose")) {
    return {false, "failed to read the initial TCP pose"};
  }
  auto previousJoints = rokae_driver::traceFeedbackCall(traceSide, "drag.initial_jointPos", [&] {
    return robot.jointPos(ec);
  });
  if (!checkError(ec, "read initial joint positions")) {
    return {false, "failed to read the initial joint positions"};
  }
  if (!allFinite(previousPose) || !allFinite(previousJoints)) {
    return {false, "initial robot feedback contains a non-finite value"};
  }
  if (onSample) onSample(previousJoints);

  const Eigen::Matrix3d startRotation = rotationFromPose(startPose);
  auto previousTime = std::chrono::steady_clock::now();
  auto nextMonitor = previousTime + kMonitorPeriod;
  auto nextSample = previousTime + kJointSamplePeriod;
  auto previousTcpPrint = previousTime - kTcpPrintPeriod;
  const auto deadline = previousTime + std::chrono::milliseconds(
                                         static_cast<int>(duration * 1000));
  int speedViolationCount = 0;

  while (running() && !stopRequested.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_until(std::min(deadline,
        onSample ? std::min(nextMonitor, nextSample) : nextMonitor));
    const auto now = std::chrono::steady_clock::now();
    if (!running() || stopRequested.load() || now >= deadline) break;
    if (now < nextMonitor) {
      // Use the same exclusively owned SDK connection. Extra recording reads
      // never replace previousJoints/previousTime used by the safety guards.
      const auto joints = rokae_driver::traceFeedbackCall(traceSide, "drag.jointPos", [&] {
        return robot.jointPos(ec);
      });
      if (!checkError(ec, "read recording joints during Cartesian drag") || !allFinite(joints)) {
        return {false, "recording joint feedback failed or contains a non-finite value"};
      }
      onSample(joints);
      nextSample = std::chrono::steady_clock::now() + kJointSamplePeriod;
      continue;
    }

    const auto pose = rokae_driver::traceFeedbackCall(traceSide, "drag.monitor_posture", [&] {
      return robot.posture(CoordinateType::endInRef, ec);
    });
    if (!checkError(ec, "read TCP pose during Cartesian drag")) {
      return {false, "TCP feedback failed"};
    }
    const auto joints = rokae_driver::traceFeedbackCall(traceSide, "drag.monitor_jointPos", [&] {
      return robot.jointPos(ec);
    });
    if (!checkError(ec, "read joints during Cartesian drag")) {
      return {false, "joint feedback failed"};
    }
    if (!allFinite(pose) || !allFinite(joints)) {
      return {false, "robot feedback contains a non-finite value"};
    }
    if (onSample) onSample(joints);

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
        return {false, "TCP linear speed exceeded " + std::to_string(kMaxLinearSpeed) + " m/s"};
      }
      return {false, "J" + std::to_string(fastestJoint + 1) +
                         " speed exceeded " + std::to_string(kMaxJointSpeed) + " rad/s"};
    }

    if (now - previousTcpPrint >= kTcpPrintPeriod) {
      std::array<double, 7> jointTorque{};
      std::array<double, 7> externalTorque{};
      std::array<double, 3> tcpTorque{};
      std::array<double, 3> tcpForce{};
      ec.clear();
      rokae_driver::traceFeedbackCall(traceSide, "drag.getEndTorque", [&] {
        forceControl.getEndTorque(FrameType::tool, jointTorque, externalTorque,
                                  tcpTorque, tcpForce, ec);
      });
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
      rokae_driver::traceFeedbackCall(traceSide, "drag.console_output", [&] {
        std::cout << "TCP XYZ[m]=[" << pose[0] << ", " << pose[1] << ", "
                << pose[2] << "] RPY[rad]=[" << pose[3] << ", "
                << pose[4] << ", " << pose[5] << "] F[N]=["
                << tcpForce[0] << ", " << tcpForce[1] << ", "
                << tcpForce[2] << "] |F|=" << forceNorm << " M[Nm]=["
                << tcpTorque[0] << ", " << tcpTorque[1] << ", "
                  << tcpTorque[2] << "] |M|=" << torqueNorm << std::endl;
      });
      previousTcpPrint = now;
    }

    previousPose = pose;
    previousJoints = joints;
    previousTime = now;
    nextMonitor = std::chrono::steady_clock::now() + kMonitorPeriod;
    nextSample = std::chrono::steady_clock::now() + kJointSamplePeriod;
  }

  if (!running()) return {true, "driver exiting"};
  if (stopRequested.load()) return {true, "A/stop requested"};
  return {true, "requested duration reached"};
}

template <typename Robot>
SessionResult runDrag(Robot &robot, double duration, std::atomic<bool> &stopRequested,
                      const std::function<void()> &onReady,
                      const std::function<bool()> &running = [] { return rclcpp::ok(); },
                      const JointSample &onSample = {}, const char *traceSide = "test") {
  rokae_driver::FeedbackSpan sessionTrace(traceSide, "drag.session");
  bool dragAttempted = false;
  SessionResult result{false, true, "drag did not start"};
  error_code ec;
  const auto checked = [&](const std::string &stage) {
    if (ec) throw std::runtime_error(stage + ": " + ec.message());
  };
  try {
    const auto initial = rokae_driver::traceFeedbackCall(traceSide, "drag.operationState", [&] {
      return robot.operationState(ec);
    });
    checked("read operation state");
    if (initial == OperationState::drag) {
      dragAttempted = true;
      result = {true, true, "existing drag session stopped; press A again to start"};
    } else if (stopRequested.load() || !running()) {
      result = {true, true, "cancelled before setup"};
    } else {
      Toolset toolset = rokae_driver::traceFeedbackCall(traceSide, "drag.toolset", [&] {
        return robot.toolset(ec);
      });
      checked("read toolset");
      toolset.load.mass = kDefaultToolMass;
      toolset.load.cog = kDefaultToolCog;
      validateLoad(toolset.load);
      rokae_driver::traceFeedbackCall(traceSide, "drag.setMotionControlMode", [&] {
        robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
      });
      checked("switch NrtCommand");
      rokae_driver::traceFeedbackCall(traceSide, "drag.setOperateMode", [&] {
        robot.setOperateMode(OperateMode::manual, ec);
      });
      checked("set manual mode");
      rokae_driver::traceFeedbackCall(traceSide, "drag.setPowerState", [&] {
        robot.setPowerState(false, ec);
      });
      checked("power off");
      rokae_driver::traceFeedbackCall(traceSide, "drag.setToolset", [&] {
        robot.setToolset(toolset, ec);
      });
      checked("set original SDK default tool load");
      const auto stored = rokae_driver::traceFeedbackCall(traceSide, "drag.verify_toolset", [&] {
        return robot.toolset(ec);
      });
      checked("verify default tool load");
      if (!loadMatchesDefaults(stored.load)) {
        throw std::runtime_error("controller did not retain the SDK default tool mass/CoG");
      }
      printLoad(stored.load);
      const auto startPose = rokae_driver::traceFeedbackCall(traceSide, "drag.start_posture", [&] {
        return robot.posture(CoordinateType::endInRef, ec);
      });
      checked("read start TCP pose");
      printPose("Cartesian drag start TCP:", startPose);
      if (stopRequested.load() || !running()) {
        result = {true, true, "cancelled before enabling drag"};
      } else {
        // SDK false requires the physical end button. True bypasses it.
        dragAttempted = true;
        rokae_driver::traceFeedbackCall(traceSide, "drag.enableDrag", [&] {
          robot.enableDrag(DragParameter::cartesianSpace, DragParameter::freely, ec, false);
        });
        checked("enable Cartesian drag");
        const auto switchDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (running() && !stopRequested.load() &&
               std::chrono::steady_clock::now() < switchDeadline) {
          std::this_thread::sleep_for(kMonitorPeriod);
        }
        if (stopRequested.load() || !running()) {
          result = {true, true, "cancelled during mode switching"};
        } else {
          onReady();
          auto forceControl = rokae_driver::traceFeedbackCall(traceSide, "drag.forceControl", [&] {
            return robot.forceControl();
          });
          const auto monitored = monitorDrag(robot, forceControl, startPose, duration,
                                             stopRequested, running, onSample, traceSide);
          result = {monitored.safe, true, monitored.reason};
        }
      }
    }
  } catch (const std::exception &error) {
    result.success = false;
    result.message = error.what();
  }
  if (dragAttempted) {
    try {
      ec.clear();
      rokae_driver::traceFeedbackCall(traceSide, "drag.disableDrag", [&] {
        robot.disableDrag(ec);
      });
      checked("disable Cartesian drag");
      const auto state = rokae_driver::traceFeedbackCall(traceSide, "drag.verify_disabled", [&] {
        return robot.operationState(ec);
      });
      checked("verify drag disabled");
      if (state == OperationState::drag) throw std::runtime_error("robot still reports drag mode");
    } catch (const std::exception &error) {
      result.success = false;
      result.cleaned = false;
      result.message += "; cleanup failed: " + std::string(error.what());
    }
  }
  return result;
}
}  // namespace

class RokaeCartesianDragSafeService : public rclcpp::Node {
  using Trigger = std_srvs::srv::Trigger;
  struct Arm {
    std::string side;
    std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
    std::atomic<bool> active{false}, stop{false};
    rclcpp::CallbackGroup::SharedPtr group;
    rclcpp::Service<Trigger>::SharedPtr startService, stopService;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr recordState;
    std::unique_ptr<rokae_driver::AsyncJointPublisher> recordingWorker;
  };
 public:
  RokaeCartesianDragSafeService()
      : Node("pico_cartesian_drag_safe", rokae_driver::interface_names::kNodeNamespace) {
    duration_ = declare_parameter<double>("duration_seconds", 600.0);
    if (!std::isfinite(duration_) || duration_ < 1.0 || duration_ > 600.0) {
      throw std::invalid_argument("duration_seconds must be 1..600");
    }
    stopGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    for (const auto &side : {std::string("left"), std::string("right")}) {
      auto arm = std::make_unique<Arm>();
      arm->side = side;
      const std::string prefix = side + "_arm";
      const auto robotIp = declare_parameter<std::string>(prefix + ".robot_ip",
          side == "left" ? "192.168.1.160" : "192.168.2.160");
      const auto localIp = declare_parameter<std::string>(prefix + ".local_ip",
          side == "left" ? "192.168.1.10" : "192.168.2.10");
      arm->hardware = rokae_driver::sharedArm(side, robotIp, localIp);
      arm->group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      arm->status = create_publisher<std_msgs::msg::String>(
          endpoint("status", side), rclcpp::QoS(1).reliable().transient_local());
      // Ordinary state callbacks wait for sdk_mutex_ while drag owns it.
      // Publish the monitor's measured joints without another SDK connection.
      arm->recordState = create_publisher<sensor_msgs::msg::JointState>(
          endpoint("joint_states", side), rclcpp::SensorDataQoS().keep_last(5));
      arm->recordingWorker = std::make_unique<rokae_driver::AsyncJointPublisher>(
          [publisher=arm->recordState, side](const rokae_driver::JointFeedbackSample &sample) {
            if (publisher->get_subscription_count() == 0) return;
            sensor_msgs::msg::JointState msg;
            msg.header.stamp = rclcpp::Time(sample.stamp_ns);
            for (int i=1;i<=7;++i) msg.name.push_back(side+"_joint_"+std::to_string(i));
            msg.position.assign(sample.joints.begin(),sample.joints.end());
            rokae_driver::traceFeedbackCall(side.c_str(),"drag.publish",[&] { publisher->publish(msg); });
            rokae_driver::FeedbackTrace::instance().emit(side.c_str(),"drag.publish","sample",
                0,0,sample.stamp_ns);
          });
      Arm *ptr = arm.get();
      arm->startService = create_service<Trigger>(endpoint("start", side),
          [this, ptr](const std::shared_ptr<Trigger::Request>,
                      std::shared_ptr<Trigger::Response> response) { start(*ptr, *response); },
          rmw_qos_profile_services_default, arm->group);
      arm->stopService = create_service<Trigger>(endpoint("stop", side),
          [ptr](const std::shared_ptr<Trigger::Request>,
                std::shared_ptr<Trigger::Response> response) {
            response->success = ptr->active.load();
            if (response->success) ptr->stop.store(true);
            response->message = response->success ? "stop requested; await drag cleanup" : "no drag session active";
          }, rmw_qos_profile_services_default, stopGroup_);
      publish(*arm, "idle");
      arms_.push_back(std::move(arm));
    }
    RCLCPP_INFO(get_logger(), "PICO Cartesian drag services ready: %.1f s; physical drag button required; SDK default load 0 kg / zero CoG", duration_);
  }
 private:
  static std::string endpoint(const std::string &operation, const std::string &side) {
    return rokae_driver::interface_names::arm("cartesian_drag_safe/" + operation, side);
  }
  void publish(Arm &arm, const std::string &state) {
    std_msgs::msg::String msg;
    msg.data = state;
    arm.status->publish(msg);
    const char *phase = state == "teaching" ? "teaching" : state == "starting" ? "starting" :
        state == "stopped" ? "stopped" : state == "idle" ? "idle" : "error";
    rokae_driver::FeedbackTrace::instance().emit(arm.side.c_str(), "drag.status", phase);
  }
  void start(Arm &arm, Trigger::Response &response) {
    std::unique_lock<std::mutex> commandLock(arm.hardware->commandMutex(), std::try_to_lock);
    if (!commandLock.owns_lock()) {
      response.success = false;
      response.message = "arm is occupied by another command";
      return;
    }
    arm.stop.store(false);
    arm.active.store(true);
    publish(arm, "starting");
    RCLCPP_INFO(get_logger(), "%s drag preparation: SDK requires manual/power-off; physical end button required, direct drag disabled", arm.side.c_str());
    SessionResult result{false, true, "setup failed"};
    try {
      // The selected arm's existing SDK connection is exclusively owned until
      // disableDrag has completed, just as in its ServoL session.
      result = arm.hardware->withRobot([&](ArRobot &robot) {
        return runDrag(robot, duration_, arm.stop, [&]() { publish(arm, "teaching"); },
            [] { return rclcpp::ok(); }, [&](const std::array<double, 7> &joints) {
              rokae_driver::FeedbackSpan sampleTrace(arm.side.c_str(), "drag.sample_callback");
              const rokae_driver::JointFeedbackSample sample{
                  static_cast<std::uint8_t>(arm.side=="left"?0:1),0,
                  static_cast<std::uint64_t>(now().nanoseconds()),
                  rokae_driver::LocalJointFeedback::monoNow(),joints};
              rokae_driver::LocalJointFeedback::instance().offer(sample,arm.recordingWorker->overwritten());
              rokae_driver::FeedbackTrace::instance().emit(arm.side.c_str(),"drag.local_send","sample",
                  0,0,sample.stamp_ns);
              arm.recordingWorker->offer(sample);
            }, arm.side.c_str());
      });
    } catch (const std::exception &error) {
      result = {false, false, error.what()};
    }
    arm.active.store(false);
    commandLock.unlock();
    response.success = result.success;
    response.message = result.cleaned ? result.message : "cleanup unverified:" + result.message;
    publish(arm, result.cleaned ? "stopped" : "error:" + result.message);
    if (result.success) RCLCPP_INFO(get_logger(), "%s drag ended: %s", arm.side.c_str(), result.message.c_str());
    else RCLCPP_ERROR(get_logger(), "%s drag failed: %s", arm.side.c_str(), result.message.c_str());
  }
  double duration_;
  rclcpp::CallbackGroup::SharedPtr stopGroup_;
  std::vector<std::unique_ptr<Arm>> arms_;
};

namespace rokae_driver {
std::shared_ptr<rclcpp::Node> makeCartesianDragSafeService() {
  return std::make_shared<RokaeCartesianDragSafeService>();
}
}  // namespace rokae_driver
