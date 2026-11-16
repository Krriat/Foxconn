#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include <nlohmann/json.hpp>
#include "timed_policy.hpp"
#include "latest_feedback.hpp"
#include "std_msgs/msg/string.hpp"
#include "rokae_interfaces/msg/dual_arm_servo_j.hpp"
#include "rokae_interfaces/msg/servo_j.hpp"

#include "rokae/data_types.h"
#include "rokae/motion_control_rt.h"
#include "rokae/robot.h"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"

namespace {

constexpr std::size_t kJointCount = 7;
using JointArray = std::array<double, kJointCount>;
using RtController = rokae::RtMotionControlCobot<kJointCount>;
using Clock = std::chrono::steady_clock;

struct Options {
  double periodS{0.01};
  double lookaheadS{0.01};
  double gain{0.0};
  double commandTimeoutS{0.10};
  double maxCommandStepRad{0.01};
  double softLimitMarginRad{0.08};
};

struct ArmContext {
  std::string side;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
};

struct SingleFrame {
  bool enable{false};
  JointArray positions{};
  Clock::time_point received{};
  std::uint64_t sequence{0};
};

struct DualFrame {
  bool enable{false};
  JointArray left{};
  JointArray right{};
  Clock::time_point received{};
  std::uint64_t sequence{0};
};

template <typename Frame>
struct CommandBuffer {
  std::mutex mutex;
  std::condition_variable condition;
  Frame latest;
  bool hasMessage{false};
};

struct ArmSession {
  std::shared_ptr<RtController> controller;
  std::array<double[2], kJointCount> softLimits{};
  JointArray lastCommand{}, measured{};
  std::int64_t sendStartNs{}, sendEndNs{};
  std::int64_t feedbackHostReadNs{};
  Clock::time_point feedbackReadMono{};
  latest_feedback::DrainResult feedbackDrain;
  std::uint64_t feedbackFramesTotal{}, feedbackDiscardedTotal{};
  bool servoConfigured{false};
  bool moveStarted{false};
  bool realtimeMode{false};
  bool stateReceiving{false};
};

bool finite(const JointArray &values) {
  return std::all_of(
      values.begin(), values.end(),
      [](double value) { return std::isfinite(value); });
}

std::string sdkError(const std::string &operation,
                     const std::error_code &error) {
  return operation + " failed: code=" + std::to_string(error.value()) +
         ", message=" + error.message();
}

using Json = nlohmann::json;
std::int64_t wallNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
bool activePhase(const std::string &phase) {
  return phase == "WAITING_ACTION" || phase == "STARTING_SERVO" || phase == "RUNNING" || phase == "PREVIEW";
}

class RokaeServoJSubscriber : public rclcpp::Node {
 public:
  RokaeServoJSubscriber() : Node("joint_servo", rokae_driver::interface_names::kNodeNamespace) {
    const auto leftRobotIp = declare_parameter<std::string>("left_arm.robot_ip", "192.168.1.160");
    const auto leftLocalIp = declare_parameter<std::string>("left_arm.local_ip", "192.168.1.10");
    const auto rightRobotIp = declare_parameter<std::string>("right_arm.robot_ip", "192.168.2.160");
    const auto rightLocalIp = declare_parameter<std::string>("right_arm.local_ip", "192.168.2.10");
    options_.periodS = declare_parameter<double>("period_s", .01);
    options_.lookaheadS = declare_parameter<double>("lookahead_s", .01);
    options_.gain = declare_parameter<double>("gain", 0.);
    options_.commandTimeoutS = declare_parameter<double>("command_timeout_s", .1);
    options_.maxCommandStepRad = declare_parameter<double>("max_command_step_rad", .01);
    options_.softLimitMarginRad = declare_parameter<double>("soft_limit_margin_rad", .08);
    // On these controllers, T=10 ms / lookahead=20 ms / gain=0 produced
    // exactly half of the requested displacement. Both-arm 0.004-rad
    // tracking probes verified unity displacement with T=lookahead=10 ms.
    if (options_.lookaheadS != options_.periodS)
      throw std::invalid_argument("ServoJ lookahead_s must equal period_s; unequal values change displacement gain");
    if (options_.periodS != .01 || options_.gain != 0. ||
        !std::isfinite(options_.maxCommandStepRad) || options_.maxCommandStepRad <= 0 || options_.maxCommandStepRad > .01 ||
        !std::isfinite(options_.commandTimeoutS) || options_.commandTimeoutS < .02 || options_.commandTimeoutS > .25 ||
        !std::isfinite(options_.softLimitMarginRad) || options_.softLimitMarginRad < .08 || options_.softLimitMarginRad > .3)
      throw std::invalid_argument("timed policy driver requires reviewed 100-Hz limits");
    left_ = {"left", rokae_driver::sharedArm("left", leftRobotIp, leftLocalIp)};
    right_ = {"right", rokae_driver::sharedArm("right", rightRobotIp, rightLocalIp)};
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    statusPublisher_ = create_publisher<std_msgs::msg::String>("/robot_02/upperlimb/servoj/status", 20);
    tracePublisher_ = create_publisher<std_msgs::msg::String>("/robot/policy/servo_trace", rclcpp::QoS(20).best_effort());
    healthPublisher_ = create_publisher<std_msgs::msg::String>("/robot/policy/trajectory_status", 1);
    healthTimer_ = create_wall_timer(std::chrono::seconds(1), [this] {
      std::lock_guard<std::mutex> lock(policyMutex_);
      std_msgs::msg::String message;
      message.data = Json({{"protocol","timed_chunk_v2"},{"ready",true},{"servo_hz",100},
                           {"feedback_reader","drain_latest_v1"},{"feedback_max_host_age_ms",30},
                           {"servoj_period_s",options_.periodS},{"servoj_lookahead_s",options_.lookaheadS},
                           {"servoj_gain",options_.gain},
                           {"run_id",runId_},{"last_error",policyError_}}).dump();
      healthPublisher_->publish(message);
    });
    statusSubscription_ = create_subscription<std_msgs::msg::String>("/robot/policy/status", 1,
        [this](std_msgs::msg::String::ConstSharedPtr message) { onStatus(message->data); });
    chunkSubscription_ = create_subscription<std_msgs::msg::String>("/robot/policy/validated_chunk", qos,
        [this](std_msgs::msg::String::ConstSharedPtr message) { onChunk(message->data); });
    dualSubscription_ = create_subscription<rokae_interfaces::msg::DualArmServoJ>(
        rokae_driver::interface_names::dualArm("servoj"), qos,
        [this](rokae_interfaces::msg::DualArmServoJ::ConstSharedPtr message) {
          std::lock_guard<std::mutex> lock(dualBuffer_.mutex);
          dualBuffer_.latest.enable = message->enable;
          dualBuffer_.latest.left = message->left_positions;
          dualBuffer_.latest.right = message->right_positions;
          dualBuffer_.latest.received = Clock::now();
          ++dualBuffer_.latest.sequence;
          dualBuffer_.hasMessage = true;
          dualBuffer_.condition.notify_one();
        });
    previewTimer_ = create_wall_timer(std::chrono::milliseconds(10), [this] { previewTick(); });
    dualThread_ = std::thread([this] { runDual(); });
    RCLCPP_WARN(get_logger(), "Timed chunk fusion driver ready; no legacy single-arm ServoJ inputs; no automatic motion");
  }
  ~RokaeServoJSubscriber() override {
    running_.store(false);
    dualBuffer_.condition.notify_all();
    if (dualThread_.joinable()) dualThread_.join();
  }
 private:
  void report(const std::string &side, const std::string &state) {
    std_msgs::msg::String message; message.data = side + ":" + state; statusPublisher_->publish(message);
  }
  void trace(Json value) {
    value["schema"] = "timed_servo_trace_v2";
    value["run_id"] = runId_;
    value["mode"] = mode_;
    value["trace_sequence"] = ++traceSequence_;
    std_msgs::msg::String message; message.data = value.dump(); tracePublisher_->publish(message);
  }
  void onStatus(const std::string &data) {
    try {
      const auto value = Json::parse(data);
      std::lock_guard<std::mutex> lock(policyMutex_);
      if (value.value("native_protocol", "") != "timed_chunk_v2") return;
      const auto id = value.at("run_id").get<std::string>();
      if (id.empty()) return;
      if (id != runId_) {
        runId_ = id; closedRunId_.clear(); fusion_.reset(); policyError_.clear(); newestSequence_ = 0;
        anchor_ = value.at("last_command").get<timed_policy::Joints>();
        const double speed = value.at("max_joint_speed_rad_s").get<double>();
        const double accel = value.at("max_joint_acceleration_rad_s2").get<double>();
        relativeLimit_ = value.at("max_relative_target_rad").get<double>();
        trackingLimit_ = value.at("max_tracking_error_rad").get<double>();
        leadNs_ = value.at("execution_lead_ns").get<std::int64_t>();
        if (relativeLimit_ <= 0 || relativeLimit_ > .035 || trackingLimit_ <= 0 || trackingLimit_ > .15 || leadNs_ < 0 || leadNs_ > 50000000)
          throw std::runtime_error("invalid policy motion limits");
        fusion_ = timed_policy::Fusion(value.at("temporal_ensemble_coefficient").get<double>());
        limiter_ = timed_policy::Limiter(speed, accel, value.at("trajectory_response_seconds").get<double>());
        limiter_.reset(anchor_);
        lastCycle_ = readyMono_ = Clock::now(); clockWall_ = wallNs(); clockMono_ = lastCycle_;
        readyNs_ = clockWall_; dispatchLeadNs_ = 0.; traceSequence_ = 0; lastChunk_ = Clock::time_point{};
      }
      mode_ = value.at("mode").get<std::string>();
      phase_ = value.at("phase").get<std::string>();
      motionAllowed_ = value.value("enable_motion", false);
      lastStatus_ = Clock::now();
    } catch (const std::exception &error) {
      std::lock_guard<std::mutex> lock(policyMutex_); policyError_ = error.what();
    }
  }
  void onChunk(const std::string &data) {
    try {
      const auto value = Json::parse(data);
      std::lock_guard<std::mutex> lock(policyMutex_);
      if (value.value("run_id", "") != runId_ || !activePhase(phase_)) return;
      timed_policy::Plan plan;
      plan.sequence = value.at("sequence").get<std::uint64_t>();
      plan.observation_ns = value.at("observation_time_ns").get<std::int64_t>();
      plan.inference_start_ns = value.at("inference_start_ns").get<std::int64_t>();
      plan.inference_end_ns = value.at("inference_end_ns").get<std::int64_t>();
      plan.send_ns = value.at("send_time_ns").get<std::int64_t>();
      plan.receive_ns = value.at("jetson_receive_ns").get<std::int64_t>();
      plan.step_ns = value.at("step_ns").get<std::int64_t>();
      plan.actions = value.at("actions").get<std::vector<timed_policy::Action>>();
      if (plan.observation_ns < readyNs_ || plan.sequence <= newestSequence_) return;
      const auto now = wallNs();
      if (now - plan.observation_ns > 300000000 || plan.observation_ns > now + 10000000 ||
          plan.send_ns > now + 10000000 || plan.send_ns < plan.inference_end_ns ||
          plan.inference_end_ns < plan.inference_start_ns)
        throw std::runtime_error("native chunk clock/freshness check failed");
      if (fusion_.add(std::move(plan))) {
        lastChunk_ = Clock::now(); newestSequence_ = value.at("sequence").get<std::uint64_t>();
      }
    } catch (const std::exception &error) {
      std::lock_guard<std::mutex> lock(policyMutex_); policyError_ = error.what();
    }
  }
  Json evaluate(const timed_policy::Joints &measured, bool live, bool &hasPlan) {
    const auto mono = Clock::now(); const auto now = wallNs();
    if (!policyError_.empty()) throw std::runtime_error(policyError_);
    if (mono - lastStatus_ > std::chrono::milliseconds(400)) throw std::runtime_error("executor heartbeat lost");
    const auto monoElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(mono-clockMono_).count();
    if (std::abs((now-clockWall_) - monoElapsed) > 10000000)
      throw std::runtime_error("host wall clock stepped during trajectory");
    if (mono-lastChunk_ > std::chrono::milliseconds(250) && mono-readyMono_ > std::chrono::milliseconds(500))
      throw std::runtime_error("native prediction watchdog expired");
    double dt = std::chrono::duration<double>(mono-lastCycle_).count(); lastCycle_ = mono;
    timed_policy::Fused fused;
    const auto executionNs = now+leadNs_+static_cast<std::int64_t>(dispatchLeadNs_);
    hasPlan = fusion_.sample(executionNs, fused);
    Json result = {{"selection_time_ns",now},{"requested_execution_time_ns",executionNs},{"estimated_dispatch_lead_ns",dispatchLeadNs_},
                   {"dt_s",dt},{"contributors",fused.contributors},{"active",hasPlan}};
    if (!hasPlan) { result["command"] = limiter_.position(); result["velocity"] = limiter_.velocity(); return result; }
    timed_policy::Joints desired;
    for (unsigned j=0;j<14;++j) {
      if (live && std::abs(limiter_.position()[j]-measured[j]) > trackingLimit_)
        throw std::runtime_error("measured joints do not track ServoJ trajectory");
      desired[j] = std::clamp(fused.target[j], measured[j]-relativeLimit_, measured[j]+relativeLimit_);
    }
    const auto &command = limiter_.step(desired, dt);
    result["command"] = command; result["velocity"] = limiter_.velocity();
    result["fused_target"] = fused.target; result["limited_target"] = desired;
    result["newest_index"] = fused.newest_index; result["oldest_index"] = fused.oldest_index;
    result["chunk_sequence"] = fused.newest_sequence;
    result["observation_time_ns"] = fused.newest_observation_ns;
    result["inference_start_ns"] = fused.inference_start_ns; result["inference_end_ns"] = fused.inference_end_ns;
    result["send_time_ns"] = fused.send_ns; result["jetson_receive_ns"] = fused.receive_ns;
    result["observation_age_ms"] = (executionNs-fused.newest_observation_ns)/1e6;
    result["network_to_executor_ms"] = (fused.receive_ns-fused.send_ns)/1e6;
    return result;
  }
  void previewTick() {
    std::lock_guard<std::mutex> lock(policyMutex_);
    if (mode_ != "preview" || phase_ != "PREVIEW" || runId_.empty()) return;
    try {
      bool active = false;
      auto value = evaluate(anchor_, false, active);
      value["hardware_commands_sent"] = false; trace(std::move(value));
    } catch (const std::exception &error) {
      trace({{"error",error.what()},{"hardware_commands_sent",false}});
    }
  }
  bool validateTarget(const ArmContext &arm, const ArmSession &session,
                      const JointArray &target, bool first,
                      std::string &reason) const {
    if (!finite(target)) {
      reason = "joint target contains NaN or infinity";
      return false;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      const double lower = session.softLimits[index][0] +
                           options_.softLimitMarginRad;
      const double upper = session.softLimits[index][1] -
                           options_.softLimitMarginRad;
      if (lower >= upper || target[index] < lower ||
          target[index] > upper) {
        reason = arm.side + " J" + std::to_string(index + 1) +
                 " violates the controller soft-limit margin";
        return false;
      }
      const double step =
          std::abs(target[index] - session.lastCommand[index]);
      if (step > options_.maxCommandStepRad) {
        reason = arm.side + " J" + std::to_string(index + 1) +
                 (first ? " first target" : " target step") +
                 " exceeds max_command_step_rad";
        return false;
      }
    }
    return true;
  }

  bool prepareArm(const ArmContext &arm, ArmSession &session,
                  std::string &reason) {
    std::error_code error;
    const auto power = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) { return robot.powerState(error); });
    if (error) {
      reason = sdkError("powerState", error);
      return false;
    }
    if (power != rokae::PowerState::on) {
      reason = arm.side +
               " robot is not powered on; call " +
               std::string(rokae_driver::interface_names::kInitializeDualArm) +
               " first";
      return false;
    }

    error.clear();
    const auto state = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) {
          return robot.operationState(error);
        });
    if (error) {
      reason = sdkError("operationState", error);
      return false;
    }
    if (state != rokae::OperationState::idle) {
      reason = arm.side + " robot must be idle before starting ServoJ";
      return false;
    }

    error.clear();
    const bool limitsEnabled = arm.hardware->withRobot(
        [&session, &error](rokae::ArRobot &robot) {
          return robot.getSoftLimit(session.softLimits, error);
        });
    if (error) {
      reason = sdkError("getSoftLimit", error);
      return false;
    }
    if (!limitsEnabled) {
      reason = arm.side + " controller soft limits are disabled";
      return false;
    }

    error.clear();
    session.lastCommand = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) { return robot.jointPos(error); });
    if (error) {
      reason = sdkError("jointPos", error);
      return false;
    }
    if (!finite(session.lastCommand)) {
      reason = arm.side + " joint feedback contains NaN or infinity";
      return false;
    }
    return true;
  }

  bool startArm(const ArmContext &arm, ArmSession &session,
                std::string &reason) {
    std::error_code error;
    try {
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        robot.setOperateMode(rokae::OperateMode::automatic, error);
        if (error) return;
        robot.setMotionControlMode(
            rokae::MotionControlMode::RtCommand, error);
        if (error) return;
        session.realtimeMode = true;
        session.controller = robot.getRtMotionController().lock();
        robot.startReceiveRobotState(
            std::chrono::milliseconds(1),
            {rokae::RtSupportedFields::jointPos_m});
        session.stateReceiving = true;
      });
      if (error) {
        reason = sdkError("enter realtime mode", error);
        return false;
      }
      if (!session.controller) {
        reason = arm.side + " getRtMotionController returned an empty handle";
        return false;
      }

      arm.hardware->withSdkLock([&] {
        session.controller->setServoJoint(
            options_.periodS, options_.lookaheadS, options_.gain, error);
      });
      if (error) {
        reason = sdkError("setServoJoint", error);
        return false;
      }
      session.servoConfigured = true;
      arm.hardware->withSdkLock([&] {
        session.controller->startMove(
            rokae::RtControllerMode::jointPosition);
      });
      session.moveStarted = true;
      return true;
    } catch (const std::exception &exception) {
      reason = arm.side + " ServoJ startup exception: " + exception.what();
      return false;
    }
  }

  bool refreshFeedback(const ArmContext &arm, ArmSession &session,
                       std::chrono::milliseconds budget, std::string &reason) {
    try {
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        session.feedbackDrain = latest_feedback::drain(
            [&] { return robot.updateRobotState(Clock::duration::zero()); },
            [] { return Clock::now(); }, budget);
        if (session.feedbackDrain.frames) {
          JointArray newest;
          if (robot.getStateData(rokae::RtSupportedFields::jointPos_m, newest) != 0 || !finite(newest))
            throw std::runtime_error("missing or nonfinite latest realtime measured joints");
          session.measured = newest;
          session.feedbackReadMono = Clock::now();
          session.feedbackHostReadNs = wallNs();
          session.feedbackFramesTotal += session.feedbackDrain.frames;
          session.feedbackDiscardedTotal += session.feedbackDrain.discarded();
        }
      });
      if (!latest_feedback::fresh(session.feedbackReadMono, Clock::now()))
        throw std::runtime_error("no fresh SDK feedback after FIFO drain (host-read age exceeds 30 ms)");
      return true;
    } catch (const std::exception &exception) {
      reason = arm.side + " feedback exception: " + exception.what();
      return false;
    }
  }

  static Json feedbackDiagnostics(const ArmSession &session) {
    return {{"reader","drain_latest_v1"}, {"frames_drained",session.feedbackDrain.frames},
            {"historical_frames_discarded",session.feedbackDrain.discarded()},
            {"drain_duration_ms",session.feedbackDrain.duration_ms},
            {"frames_total",session.feedbackFramesTotal}, {"discarded_total",session.feedbackDiscardedTotal},
            {"last_host_read_ns",session.feedbackHostReadNs},
            {"host_read_age_ms",std::chrono::duration<double,std::milli>(Clock::now()-session.feedbackReadMono).count()},
            {"fresh",latest_feedback::fresh(session.feedbackReadMono,Clock::now())},
            {"age_scope","host time since latest drained SDK read; controller capture timestamp unavailable"}};
  }

  bool sendArm(const ArmContext &arm, ArmSession &session,
               const JointArray &target, std::string &reason) {
    try {
      rokae::JointPosition command(std::vector<double>(target.begin(), target.end()));
      arm.hardware->withSdkLock([&] {
        session.sendStartNs = wallNs();
        session.controller->sendCommand(command);
        session.sendEndNs = wallNs();
      });
      session.lastCommand = target;
      report(arm.side, "running");
      return true;
    } catch (const std::exception &exception) {
      reason = arm.side + " sendCommand exception: " + exception.what();
      return false;
    }
  }

  void stopArm(const ArmContext &arm, ArmSession &session) noexcept {
    bool cleanupFailed = false;
    report(arm.side, "stopped");
    if (session.controller) {
      if (session.moveStarted) {
        try {
          // Use the stop sequence verified in the small-amplitude ServoJ
          // probes. A finish flag followed immediately by stopServoJoint can
          // race the controller's RT completion and leave the left arm off.
          arm.hardware->withSdkLock(
              [&] { session.controller->stopMove(); });
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception &exception) {
          RCLCPP_ERROR(
              get_logger(), "%s ServoJ stopMove failed: %s",
              arm.side.c_str(), exception.what());
          try {
            arm.hardware->withSdkLock(
                [&] { session.controller->stopMove(); });
          } catch (const std::exception &stopException) {
            cleanupFailed = true;
            RCLCPP_ERROR(
                get_logger(), "%s ServoJ stopMove failed: %s",
                arm.side.c_str(), stopException.what());
          }
        }
      }
      if (session.servoConfigured) {
        try {
          arm.hardware->withSdkLock(
              [&] { session.controller->stopServoJoint(); });
        } catch (const std::exception &error) {
          cleanupFailed = true;
          RCLCPP_ERROR(get_logger(), "%s stopServoJoint failed: %s", arm.side.c_str(), error.what());
        }
      }
    }

    if (session.realtimeMode) {
      std::error_code error;
      try {
        arm.hardware->withRobot([&error, &session](rokae::ArRobot &robot) {
          if (session.stateReceiving) robot.stopReceiveRobotState();
          robot.setMotionControlMode(
              rokae::MotionControlMode::NrtCommand, error);
        });
      } catch (const std::exception &exception) {
        cleanupFailed = true;
        RCLCPP_ERROR(get_logger(), "%s NRT cleanup failed: %s", arm.side.c_str(), exception.what());
      }
      if (error) {
        cleanupFailed = true;
        RCLCPP_ERROR(
            get_logger(), "%s", sdkError(
                arm.side + " restore NrtCommand", error).c_str());
      }
    }
    session = ArmSession{};
    // Stopping ServoJ must not issue a power-off command. Verify the
    // controller state after cleanup instead of silently claiming success.
    try {
      std::error_code error;
      const auto power = arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) { return robot.powerState(error); });
      if (error) {
        report(arm.side, "error:" + sdkError("powerState after stop", error));
      } else if (cleanupFailed) {
        report(arm.side, "error:ServoJ cleanup failed; refusing automatic stage transition");
      } else if (power == rokae::PowerState::on) {
        report(arm.side, "idle_power_on");
        RCLCPP_INFO(get_logger(), "%s ServoJ stopped; power remains ON",
                    arm.side.c_str());
      } else {
        report(arm.side, "error:controller is not powered on after ServoJ stop");
        RCLCPP_ERROR(get_logger(),
                     "%s power state after stop=%d; no power-off command sent",
                     arm.side.c_str(), static_cast<int>(power));
      }
    } catch (const std::exception &error) {
      report(arm.side, std::string("error:power verification failed: ") + error.what());
    }
  }


  void runDual() {
    std::uint64_t consumed=0;
    while (running_.load()) {
      DualFrame frame;
      {
        std::unique_lock<std::mutex> lock(dualBuffer_.mutex);
        dualBuffer_.condition.wait(lock,[&] { return !running_.load() ||
          (dualBuffer_.hasMessage && dualBuffer_.latest.sequence != consumed); });
        if (!running_.load()) break;
        frame=dualBuffer_.latest; consumed=frame.sequence;
      }
      if (!frame.enable) continue;
      std::string sessionId;
      {
        std::lock_guard<std::mutex> lock(policyMutex_);
        if (mode_ != "live" || phase_ != "STARTING_SERVO" || !motionAllowed_ || !policyError_.empty() || closedRunId_ == runId_) continue;
        sessionId=runId_;
      }
      std::unique_lock<std::mutex> leftLock(left_.hardware->commandMutex(),std::defer_lock);
      std::unique_lock<std::mutex> rightLock(right_.hardware->commandMutex(),std::defer_lock);
      if (std::try_lock(leftLock,rightLock) != -1) { report("left","error:arm occupied"); report("right","error:arm occupied"); continue; }
      const auto leftStop=left_.hardware->stopGeneration(), rightStop=right_.hardware->stopGeneration();
      ArmSession leftSession,rightSession; std::string reason;
      bool ready=prepareArm(left_,leftSession,reason) && prepareArm(right_,rightSession,reason);
      // Begin at measured positions, never at an old prediction prefix.
      if (ready) ready=startArm(left_,leftSession,reason);
      if (ready) ready=startArm(right_,rightSession,reason);
      // Both SDK startups can accumulate feedback before the control loop runs.
      // Drain that backlog before initializing the trajectory at current joints.
      if (ready) ready=refreshFeedback(left_,leftSession,std::chrono::milliseconds(50),reason) &&
                       refreshFeedback(right_,rightSession,std::chrono::milliseconds(50),reason);
      if (ready) {
        std::lock_guard<std::mutex> lock(policyMutex_);
        timed_policy::Joints measured;
        leftSession.lastCommand=leftSession.measured; rightSession.lastCommand=rightSession.measured;
        std::copy(leftSession.measured.begin(),leftSession.measured.end(),measured.begin());
        std::copy(rightSession.measured.begin(),rightSession.measured.end(),measured.begin()+7);
        limiter_.reset(measured); anchor_=measured;
        readyNs_=wallNs(); readyMono_=lastCycle_=Clock::now();
        clockWall_=readyNs_; clockMono_=readyMono_; fusion_.discard_before(readyNs_);
      }
      auto next=Clock::now();
      while (ready && running_.load()) {
        {
          std::lock_guard<std::mutex> lock(dualBuffer_.mutex);
          frame=dualBuffer_.latest; consumed=frame.sequence;
        }
        if (!frame.enable) break;
        if (Clock::now()-frame.received > std::chrono::duration<double>(options_.commandTimeoutS)) {
          reason="local ServoJ keepalive watchdog expired"; break;
        }
        if (leftStop!=left_.hardware->stopGeneration() || rightStop!=right_.hardware->stopGeneration()) {
          reason="controller stop endpoint requested"; break;
        }
        try {
          Json value; bool active=false;
          // Feedback used by relative-position/tracking checks is refreshed
          // BEFORE target selection, not read once after selecting the command.
          if (!refreshFeedback(left_,leftSession,std::chrono::milliseconds(5),reason) ||
              !refreshFeedback(right_,rightSession,std::chrono::milliseconds(5),reason))
            throw std::runtime_error(reason);
          {
            std::lock_guard<std::mutex> lock(policyMutex_);
            if (runId_!=sessionId || mode_!="live" || (phase_!="STARTING_SERVO" && phase_!="RUNNING") || !motionAllowed_) break;
            timed_policy::Joints measured;
            std::copy(leftSession.measured.begin(),leftSession.measured.end(),measured.begin());
            std::copy(rightSession.measured.begin(),rightSession.measured.end(),measured.begin()+7);
            value=evaluate(measured,true,active);
            auto q=limiter_.position();
            std::copy_n(q.begin(),7,frame.left.begin()); std::copy_n(q.begin()+7,7,frame.right.begin());
          }
          if (!validateTarget(left_,leftSession,frame.left,false,reason) ||
              !validateTarget(right_,rightSession,frame.right,false,reason) ||
              !sendArm(left_,leftSession,frame.left,reason) || !sendArm(right_,rightSession,frame.right,reason))
            throw std::runtime_error(reason);
          value["left_sdk_send_start_ns"]=leftSession.sendStartNs; value["left_sdk_send_end_ns"]=leftSession.sendEndNs;
          value["right_sdk_send_start_ns"]=rightSession.sendStartNs; value["right_sdk_send_end_ns"]=rightSession.sendEndNs;
          value["left_measured"]=leftSession.measured; value["right_measured"]=rightSession.measured;
          value["left_feedback"]=feedbackDiagnostics(leftSession);
          value["right_feedback"]=feedbackDiagnostics(rightSession);
          value["hardware_commands_sent"]=true;
          { std::lock_guard<std::mutex> lock(policyMutex_);
            const double elapsed = .5*((leftSession.sendStartNs-value.at("selection_time_ns").get<std::int64_t>()) +
                                        (rightSession.sendStartNs-value.at("selection_time_ns").get<std::int64_t>()));
            dispatchLeadNs_ = .9*dispatchLeadNs_+.1*std::clamp(elapsed,0.,20000000.);
            trace(std::move(value)); }
        } catch (const std::exception &error) { reason=error.what(); break; }
        next+=std::chrono::milliseconds(10);
        if (next<=Clock::now()) next=Clock::now()+std::chrono::milliseconds(10);
        std::unique_lock<std::mutex> lock(dualBuffer_.mutex);
        dualBuffer_.condition.wait_until(lock,next,[&] { return !running_.load() || !dualBuffer_.latest.enable; });
      }
      if (!reason.empty()) { report("left","error:"+reason); report("right","error:"+reason); }
      if (rightSession.realtimeMode || rightSession.controller) stopArm(right_,rightSession);
      if (leftSession.realtimeMode || leftSession.controller) stopArm(left_,leftSession);
      // Require a new round after any native failure; never restart on keepalive.
      { std::lock_guard<std::mutex> lock(policyMutex_);
        closedRunId_=sessionId;
        if (runId_==sessionId && !reason.empty()) policyError_=reason; }
    }
  }
  Options options_;
  ArmContext left_,right_;
  std::atomic<bool> running_{true};
  CommandBuffer<DualFrame> dualBuffer_;
  std::thread dualThread_;
  std::mutex policyMutex_;
  timed_policy::Fusion fusion_;
  timed_policy::Limiter limiter_;
  timed_policy::Joints anchor_{};
  std::string runId_,closedRunId_,mode_,phase_,policyError_;
  bool motionAllowed_{false};
  double relativeLimit_{.035},trackingLimit_{.15},dispatchLeadNs_{};
  std::uint64_t newestSequence_{},traceSequence_{};
  std::int64_t readyNs_{},clockWall_{},leadNs_{};
  Clock::time_point lastCycle_{},readyMono_{},clockMono_{},lastStatus_{},lastChunk_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr statusPublisher_,tracePublisher_,healthPublisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr statusSubscription_,chunkSubscription_;
  rclcpp::Subscription<rokae_interfaces::msg::DualArmServoJ>::SharedPtr dualSubscription_;
  rclcpp::TimerBase::SharedPtr previewTimer_,healthTimer_;
};
}  // namespace
namespace rokae_driver {
std::shared_ptr<rclcpp::Node> makeServoJSubscriber() { return std::make_shared<RokaeServoJSubscriber>(); }
}
