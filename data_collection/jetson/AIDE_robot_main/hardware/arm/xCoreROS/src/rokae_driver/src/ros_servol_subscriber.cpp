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
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

#include "rokae/motion_control_rt.h"
#include "rokae/utility.h"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/servol_math.hpp"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_interfaces/msg/dual_arm_servo_l.hpp"
#include "rokae_interfaces/msg/servo_l_jog.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Matrix = rokae_driver::servol::Matrix;
using Pose = geometry_msgs::msg::Pose;
using Controller = rokae::RtMotionControlCobot<7>;

enum class StopCode {
  none,
  requested,
  watchdog,
  shutdown,
  alignment_timeout,
  target_step,
  joint_limit,
  state_timeout,
  cycle_late,
  sdk_failure,
};

struct CommandBuffer {
  std::mutex mutex;
  std::condition_variable condition;
  // ROS targets use the same TCP-in-external-reference convention as PICO.
  std::array<Matrix, 2> targets{};
  double joystick_y{0.0};
  Clock::time_point received{};
  std::uint64_t sequence{0};
  bool stop{false};
};

// Non-blocking bridge from the control thread to the ROS publishing timer.
// This is the generated EndInRef command, not measured robot feedback.
struct CommandPoseSlot {
  CommandPoseSlot() {
    for (auto &value : matrix) value.store(0.0, std::memory_order_relaxed);
  }

  void write(const Matrix &source) noexcept {
    sequence.fetch_add(1, std::memory_order_acq_rel);
    for (std::size_t index = 0; index < source.size(); ++index) {
      matrix[index].store(source[index], std::memory_order_relaxed);
    }
    sequence.fetch_add(1, std::memory_order_release);
    active.store(true, std::memory_order_release);
  }

  bool read(Matrix &destination) const noexcept {
    for (int attempt = 0; attempt < 3; ++attempt) {
      const auto before = sequence.load(std::memory_order_acquire);
      if ((before & 1U) != 0U) continue;
      for (std::size_t index = 0; index < destination.size(); ++index) {
        destination[index] = matrix[index].load(std::memory_order_relaxed);
      }
      const auto after = sequence.load(std::memory_order_acquire);
      if (before == after && (after & 1U) == 0U) return true;
    }
    return false;
  }

  std::array<std::atomic<double>, 16> matrix;
  std::atomic<std::uint64_t> sequence{0};
  std::atomic<bool> active{false};
};

struct Arm {
  std::string side;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
};

struct Session {
  Arm *arm{nullptr};
  const char *preparation_step{nullptr};
  std::shared_ptr<Controller> controller;
  std::array<double[2], 7> limits{};
  std::array<double, 6> base_frame{};
  rokae::Toolset toolset{};
  Matrix initial_end{};
  Matrix command_end{};
  double elbow{0.0};
  std::vector<int> configuration;
  std::vector<double> external;
  bool realtime{false};
  bool receiving_state{false};
  bool servo_enabled{false};
  bool move_started{false};
};

struct SharedRun {
  std::atomic<StopCode> stop{StopCode::none};
  std::atomic<int> aligned_mask{0};
  std::mutex error_mutex;
  std::string sdk_error;
};

struct LoopState {
  CommandBuffer *buffer{nullptr};
  CommandPoseSlot *command_pose{nullptr};
  SharedRun *shared{nullptr};
  std::atomic<int> *stage{nullptr};
  int target_index{0};
  int alignment_bit{1};
  int expected_alignment_mask{1};
  std::uint64_t seen_sequence{0};
  Clock::time_point last_received{};
  Clock::time_point alignment_deadline{};
  Matrix goal_end{};
  double joystick_y{0.0};
  bool aligned{false};
};

void check(const std::error_code &error, const char *where) {
  if (error) {
    throw std::runtime_error(std::string(where) + ": " + error.message());
  }
}

const char *powerStateName(rokae::PowerState state) noexcept {
  switch (state) {
    case rokae::PowerState::on: return "on";
    case rokae::PowerState::off: return "off";
    case rokae::PowerState::estop: return "emergency stop";
    case rokae::PowerState::gstop: return "safety gate stop";
    default: return "unknown";
  }
}

void requestStop(SharedRun &run, StopCode code) noexcept {
  auto expected = StopCode::none;
  run.stop.compare_exchange_strong(
      expected, code, std::memory_order_acq_rel);
}

Matrix limitedPoseStep(
    const Matrix &current, const Matrix &target,
    double max_translation, double max_rotation) {
  const auto current_pose = rokae_driver::servol::pose(current);
  const auto target_pose = rokae_driver::servol::pose(target);
  const double translation = std::hypot(
      std::hypot(
          target_pose.position.x - current_pose.position.x,
          target_pose.position.y - current_pose.position.y),
      target_pose.position.z - current_pose.position.z);
  const double dot = std::clamp(
      std::abs(
          current_pose.orientation.x * target_pose.orientation.x +
          current_pose.orientation.y * target_pose.orientation.y +
          current_pose.orientation.z * target_pose.orientation.z +
          current_pose.orientation.w * target_pose.orientation.w),
      0.0, 1.0);
  const double rotation = 2.0 * std::acos(dot);
  double ratio = 1.0;
  if (translation > max_translation) {
    ratio = std::min(ratio, max_translation / translation);
  }
  if (rotation > max_rotation) {
    ratio = std::min(ratio, max_rotation / rotation);
  }
  return rokae_driver::servol::interpolate(current, target, ratio);
}

class ServoL : public rclcpp::Node {
 public:
  ServoL()
      : Node("cartesian_servo", rokae_driver::interface_names::kNodeNamespace) {
    period_ = declare_parameter("period_s", 0.01);
    lookahead_ = declare_parameter("lookahead_s", period_);
    gain_ = declare_parameter("gain", 0.0);
    watchdog_ = declare_parameter("command_timeout_s", 0.10);
    state_timeout_ = declare_parameter("state_timeout_s", 0.10);
    max_cycle_lateness_ =
        declare_parameter("max_cycle_lateness_s", 0.04);
    translation_step_ =
        declare_parameter("max_translation_step_m", 0.0005);
    rotation_step_ = declare_parameter(
        "max_rotation_step_rad", 0.004363323129985824);
    max_target_jump_translation_ =
        declare_parameter("max_target_jump_m", 0.10);
    max_target_jump_rotation_ =
        declare_parameter("max_target_jump_rad", 0.7853981633974483);
    joint_margin_ = declare_parameter("soft_limit_margin_rad", 0.08);
    alignment_timeout_ =
        declare_parameter("initial_alignment_timeout_s", 2.0);
    network_tolerance_ =
        declare_parameter("rt_network_tolerance_percent", 20);

    for (double value : {
             period_, lookahead_, gain_, watchdog_, state_timeout_,
             max_cycle_lateness_, translation_step_, rotation_step_,
             max_target_jump_translation_, max_target_jump_rotation_,
             joint_margin_, alignment_timeout_}) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("non-finite ServoL parameter");
      }
    }
    if (period_ < 0.005 || period_ > 0.1 || lookahead_ < period_ ||
        lookahead_ > 1.0 || gain_ < 0.0 || gain_ > 1.0 ||
        watchdog_ < 2.0 * period_ || watchdog_ > 2.0 ||
        state_timeout_ < 2.0 * period_ || state_timeout_ > 2.0 ||
        max_cycle_lateness_ < period_ || max_cycle_lateness_ > 1.0 ||
        translation_step_ <= 0.0 || translation_step_ > 0.01 ||
        rotation_step_ <= 0.0 || rotation_step_ > 0.1 ||
        max_target_jump_translation_ < translation_step_ ||
        max_target_jump_translation_ > 1.0 ||
        max_target_jump_rotation_ < rotation_step_ ||
        max_target_jump_rotation_ > 3.14159265358979323846 ||
        joint_margin_ < 0.0 || joint_margin_ > 0.3 ||
        alignment_timeout_ < 0.1 || alignment_timeout_ > 10.0 ||
        network_tolerance_ < 20 || network_tolerance_ > 100) {
      throw std::invalid_argument("invalid ServoL timing/safety parameter");
    }

    for (int index = 0; index < 2; ++index) {
      const auto side = index == 0 ? std::string("left") : std::string("right");
      const auto prefix = side + "_arm.";
      const auto robot_ip = declare_parameter(
          prefix + "robot_ip",
          index == 0 ? "192.168.1.160" : "192.168.2.160");
      const auto local_ip = declare_parameter(
          prefix + "local_ip",
          index == 0 ? "192.168.1.10" : "192.168.2.10");
      arms_[index] = {
          side, rokae_driver::sharedArm(side, robot_ip, local_ip)};
    }

    status_publisher_ = create_publisher<std_msgs::msg::String>(
        "/robot_02/upperlimb/servol/status", 10);
    const auto command_qos =
        rclcpp::QoS(1).best_effort().durability_volatile();
    for (int index = 0; index < 2; ++index) {
      command_pose_publishers_[index] =
          create_publisher<geometry_msgs::msg::PoseStamped>(
              "/robot_02/upperlimb/servol/command_pose/" +
                  arms_[index].side + "_arm",
              command_qos);
      single_subscriptions_[index] = create_subscription<Pose>(
          "/robot_02/upperlimb/servol/" + arms_[index].side + "_arm",
          command_qos,
          [this, index](Pose::ConstSharedPtr message) {
            receive(index, {*message, Pose{}});
          });
      jog_subscriptions_[index] =
          create_subscription<rokae_interfaces::msg::ServoLJog>(
              "/robot_02/upperlimb/servol/jog/" +
                  arms_[index].side + "_arm",
              command_qos,
              [this, index](
                  rokae_interfaces::msg::ServoLJog::ConstSharedPtr message) {
                receive(index, {message->pose, Pose{}},
                        message->joystick_y);
              });
    }
    dual_subscription_ =
        create_subscription<rokae_interfaces::msg::DualArmServoL>(
            "/robot_02/upperlimb/servol/dual_arm", command_qos,
            [this](
                rokae_interfaces::msg::DualArmServoL::ConstSharedPtr message) {
              receive(2, {message->left_pose, message->right_pose});
            });

    global_stop_subscription_ = create_subscription<std_msgs::msg::Bool>(
        "/robot_02/upperlimb/servol/stop", 10,
        [this](std_msgs::msg::Bool::ConstSharedPtr message) {
          if (!message->data) return;
          for (int mode = 0; mode < 3; ++mode) stopMode(mode);
        });
    const std::array<std::string, 3> mode_names{
        "left_arm", "right_arm", "dual_arm"};
    for (int mode = 0; mode < 3; ++mode) {
      mode_stop_subscriptions_[mode] =
          create_subscription<std_msgs::msg::Bool>(
              "/robot_02/upperlimb/servol/stop/" + mode_names[mode], 10,
              [this, mode](std_msgs::msg::Bool::ConstSharedPtr message) {
                if (message->data) stopMode(mode);
              });
    }

    publish_timer_ = create_wall_timer(
        std::chrono::milliseconds(10),
        [this]() { publishCommandState(); });
    for (int mode = 0; mode < 3; ++mode) {
      workers_[mode] = std::thread([this, mode]() { run(mode); });
    }
    RCLCPP_WARN(
        get_logger(),
        "ServoL ready: ROS TCP targets and SDK ServoL commands %.0f Hz; "
        "EndInRef is converted to FlanInBase using active base/tool frames; "
        "no automatic power-on",
        1.0 / period_);
  }

  ~ServoL() override {
    running_.store(false, std::memory_order_release);
    for (int mode = 0; mode < 3; ++mode) stopMode(mode);
    for (auto &worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

 private:
  void report(int mode, const std::string &state) noexcept {
    try {
      std_msgs::msg::String message;
      message.data =
          (mode == 2 ? std::string("dual") : arms_[mode].side) +
          ":" + state;
      status_publisher_->publish(message);
    } catch (...) {
    }
  }

  void stopMode(int mode) noexcept {
    auto &buffer = buffers_[mode];
    {
      std::lock_guard<std::mutex> lock(buffer.mutex);
      buffer.stop = true;
    }
    buffer.condition.notify_all();
  }

  void receive(int mode, const std::array<Pose, 2> &poses,
               double joystick_y = 0.0) {
    if (!std::isfinite(joystick_y) || std::abs(joystick_y) > 1.0) {
      report(mode, "error:invalid joystick Y");
      return;
    }
    std::array<Matrix, 2> targets{};
    try {
      const int count = mode == 2 ? 2 : 1;
      for (int index = 0; index < count; ++index) {
        targets[index] = rokae_driver::servol::matrix(poses[index]);
      }
    } catch (const std::exception &error) {
      report(
          mode,
          "error:invalid Cartesian target: " + std::string(error.what()));
      return;
    }

    auto &buffer = buffers_[mode];
    {
      std::lock_guard<std::mutex> lock(buffer.mutex);
      buffer.targets = targets;
      buffer.joystick_y = joystick_y;
      buffer.received = Clock::now();
      ++buffer.sequence;
    }
    buffer.condition.notify_one();
  }

  bool jointsAreSafe(
      const Session &session, const std::array<double, 7> &joints,
      int &bad_joint) const noexcept {
    for (std::size_t index = 0; index < joints.size(); ++index) {
      if (!std::isfinite(joints[index]) ||
          joints[index] < session.limits[index][0] + joint_margin_ ||
          joints[index] > session.limits[index][1] - joint_margin_) {
        bad_joint = static_cast<int>(index);
        return false;
      }
    }
    return true;
  }

  std::string jointLimitDetails(
      const Session &session, const std::array<double, 7> &joints,
      int joint) const {
    return " (actual=" + std::to_string(joints[joint]) +
           " rad, allowed=[" +
           std::to_string(session.limits[joint][0] + joint_margin_) + ", " +
           std::to_string(session.limits[joint][1] - joint_margin_) +
           "] rad, controller limits=[" +
           std::to_string(session.limits[joint][0]) + ", " +
           std::to_string(session.limits[joint][1]) +
           "] rad, margin=" + std::to_string(joint_margin_) + " rad)";
  }

  Matrix endInRefToFlanInBase(
      const Session &session, const Matrix &end_matrix) const {
    std::array<double, 6> end_pose{};
    rokae::Utils::transArrayToPosture(end_matrix, end_pose);
    const auto flange_pose = rokae::Utils::EndInRefToFlanInBase(
        session.base_frame, session.toolset, end_pose);
    Matrix flange_matrix{};
    rokae::Utils::postureToTransArray(flange_pose, flange_matrix);
    (void)rokae_driver::servol::pose(flange_matrix);
    return flange_matrix;
  }

  bool updateMeasuredJoints(
      rokae::ArRobot &robot, std::array<double, 7> &joints) {
    bool updated = robot.updateRobotState(std::chrono::milliseconds(1));
    while (robot.updateRobotState(Clock::duration::zero())) updated = true;
    return updated &&
           robot.getStateData(rokae::RtSupportedFields::jointPos_m, joints) ==
               0;
  }

  void logPreparationPower(
      rokae::ArRobot &robot, const Session &session, const char *stage) {
    std::error_code error;
    const auto power = robot.powerState(error);
    if (error) {
      RCLCPP_WARN(
          get_logger(), "%s ServoL %s: powerState failed: %s",
          session.arm->side.c_str(), stage, error.message().c_str());
      return;
    }
    RCLCPP_INFO(
        get_logger(), "%s ServoL %s: power=%s (%d), robot=%s",
        session.arm->side.c_str(), stage, powerStateName(power),
        static_cast<int>(power), session.arm->hardware->robotIp().c_str());
  }

  void prepare(Session &session, CommandPoseSlot &command_pose) {
    session.preparation_step = "initial power check";
    session.arm->hardware->withRobot([&](rokae::ArRobot &robot) {
      std::error_code error;
      const auto power = robot.powerState(error);
      check(error, "powerState");
      RCLCPP_INFO(
          get_logger(), "%s ServoL initial power=%s (%d), robot=%s",
          session.arm->side.c_str(), powerStateName(power),
          static_cast<int>(power), session.arm->hardware->robotIp().c_str());
      if (power != rokae::PowerState::on) {
        throw std::runtime_error(
            std::string("robot is not powered on: ") + powerStateName(power));
      }
      session.preparation_step = "idle and soft-limit checks";
      const auto operation = robot.operationState(error);
      check(error, "operationState");
      if (operation != rokae::OperationState::idle) {
        throw std::runtime_error("robot must be idle");
      }

      const bool limits_enabled = robot.getSoftLimit(session.limits, error);
      check(error, "getSoftLimit");
      if (!limits_enabled) {
        throw std::runtime_error("controller soft limits must be enabled");
      }
      for (const auto &limit : session.limits) {
        if (!std::isfinite(limit[0]) || !std::isfinite(limit[1]) ||
            limit[0] + joint_margin_ >= limit[1] - joint_margin_) {
          throw std::runtime_error("invalid controller soft limits");
        }
      }

      // Match pico_servol.cpp: anchor in EndInRef before entering RT mode,
      // preserve seven-axis metadata, and cache the active frame mapping.
      session.preparation_step = "read TCP and frames";
      const auto end_pose =
          robot.posture(rokae::CoordinateType::endInRef, error);
      check(error, "posture(endInRef)");
      const auto cartesian =
          robot.cartPosture(rokae::CoordinateType::endInRef, error);
      check(error, "cartPosture(endInRef)");
      session.base_frame = robot.baseFrame(error);
      check(error, "baseFrame");
      session.toolset = robot.toolset(error);
      check(error, "toolset");

      rokae::Utils::postureToTransArray(end_pose, session.initial_end);
      (void)rokae_driver::servol::pose(session.initial_end);
      session.command_end = session.initial_end;
      session.elbow = cartesian.elbow;
      if (!std::isfinite(session.elbow) ||
          std::abs(session.elbow) > 3.14159265358979323846) {
        throw std::runtime_error("controller returned an invalid arm angle");
      }
      session.configuration = cartesian.confData;
      session.external = cartesian.external;
      (void)endInRefToFlanInBase(session, session.initial_end);

      session.preparation_step = "setOperateMode(automatic)";
      robot.setOperateMode(rokae::OperateMode::automatic, error);
      check(error, "automatic mode");
      logPreparationPower(robot, session, "after setOperateMode(automatic)");
      session.preparation_step = "setRtNetworkTolerance";
      robot.setRtNetworkTolerance(
          static_cast<unsigned>(network_tolerance_), error);
      check(error, "setRtNetworkTolerance");
      session.preparation_step = "setMotionControlMode(RtCommand)";
      robot.setMotionControlMode(rokae::MotionControlMode::RtCommand, error);
      check(error, "RtCommand");
      session.realtime = true;
      logPreparationPower(robot, session, "after setMotionControlMode(RtCommand)");
      session.preparation_step = "getRtMotionController";
      session.controller = robot.getRtMotionController().lock();
      if (!session.controller) {
        throw std::runtime_error("empty realtime controller");
      }

      session.preparation_step = "startReceiveRobotState";
      robot.startReceiveRobotState(
          std::chrono::milliseconds(1),
          {rokae::RtSupportedFields::jointPos_m});
      session.receiving_state = true;
      session.preparation_step = "initial realtime joint feedback";
      std::array<double, 7> joints{};
      if (!updateMeasuredJoints(robot, joints)) {
        throw std::runtime_error("initial realtime joint feedback timeout");
      }
      int bad_joint = -1;
      if (!jointsAreSafe(session, joints, bad_joint)) {
        throw std::runtime_error(
            "initial joint J" + std::to_string(bad_joint + 1) +
            " violates soft-limit margin" +
            jointLimitDetails(session, joints, bad_joint));
      }
      command_pose.write(session.initial_end);
    });
    session.preparation_step = nullptr;
  }

  void runArmLoop(Session &session, LoopState &state) noexcept {
    try {
      session.arm->hardware->withRobot([&](rokae::ArRobot &robot) {
        std::error_code error;
        session.controller->setServoJoint(
            period_, lookahead_, gain_, error);
        check(error, "setServoJoint for ServoL");
        session.servo_enabled = true;
        session.controller->startMove(
            rokae::RtControllerMode::cartesianPosition);
        session.move_started = true;

        auto next_cycle = Clock::now();
        auto last_state = next_cycle;
        std::array<double, 7> measured_joints{};

        while (running_.load(std::memory_order_acquire) &&
               state.shared->stop.load(std::memory_order_acquire) ==
                   StopCode::none) {
          if (updateMeasuredJoints(robot, measured_joints)) {
            last_state = Clock::now();
            int bad_joint = -1;
            if (!jointsAreSafe(session, measured_joints, bad_joint)) {
              {
                std::lock_guard<std::mutex> lock(state.shared->error_mutex);
                state.shared->sdk_error =
                    session.arm->side + " joint J" +
                    std::to_string(bad_joint + 1) +
                    " reached the configured safety margin" +
                    jointLimitDetails(session, measured_joints, bad_joint);
              }
              requestStop(*state.shared, StopCode::joint_limit);
              break;
            }
          } else if (
              Clock::now() - last_state >
              std::chrono::duration<double>(state_timeout_)) {
            requestStop(*state.shared, StopCode::state_timeout);
            break;
          }

          std::array<Matrix, 2> targets{};
          Clock::time_point received{};
          std::uint64_t sequence = 0;
          double joystick_y = 0.0;
          bool stop_requested = false;
          {
            std::lock_guard<std::mutex> lock(state.buffer->mutex);
            targets = state.buffer->targets;
            received = state.buffer->received;
            sequence = state.buffer->sequence;
            joystick_y = state.buffer->joystick_y;
            stop_requested = state.buffer->stop;
          }
          if (stop_requested) {
            requestStop(*state.shared, StopCode::requested);
            break;
          }

          const auto now = Clock::now();
          if (received != Clock::time_point{}) state.last_received = received;
          if (now - state.last_received >
              std::chrono::duration<double>(watchdog_)) {
            requestStop(*state.shared, StopCode::watchdog);
            break;
          }

          if (sequence != state.seen_sequence) {
            const Matrix &target = targets[state.target_index];
            if (!state.aligned) {
              if (rokae_driver::servol::withinStep(
                      session.initial_end, target,
                      translation_step_, rotation_step_)) {
                state.aligned = true;
                state.goal_end = target;
                const int mask = state.shared->aligned_mask.fetch_or(
                                     state.alignment_bit,
                                     std::memory_order_acq_rel) |
                                 state.alignment_bit;
                if (mask == state.expected_alignment_mask) {
                  state.stage->store(3, std::memory_order_release);
                }
              }
            } else if (!rokae_driver::servol::withinStep(
                           state.goal_end, target,
                           max_target_jump_translation_,
                           max_target_jump_rotation_)) {
              requestStop(*state.shared, StopCode::target_step);
              break;
            } else {
              state.goal_end = target;
            }
            state.joystick_y = joystick_y;
            state.seen_sequence = sequence;
          }

          if (!state.aligned && now > state.alignment_deadline) {
            requestStop(*state.shared, StopCode::alignment_timeout);
            break;
          }
          const bool all_aligned =
              state.shared->aligned_mask.load(std::memory_order_acquire) ==
              state.expected_alignment_mask;
          if (state.aligned && all_aligned) {
            session.command_end = limitedPoseStep(
                session.command_end, state.goal_end,
                translation_step_, rotation_step_);
            session.elbow = rokae_driver::servol::nextElbowCommand(
                session.elbow, state.joystick_y, period_);
          }

          rokae::CartesianPosition command;
          command.pos = endInRefToFlanInBase(session, session.command_end);
          command.hasElbow = true;
          command.elbow = session.elbow;
          command.confData = session.configuration;
          command.external = session.external;
          session.controller->sendCommand(command);
          state.command_pose->write(session.command_end);

          next_cycle += std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(period_));
          const auto before_sleep = Clock::now();
          if (before_sleep - next_cycle >
              std::chrono::duration<double>(max_cycle_lateness_)) {
            requestStop(*state.shared, StopCode::cycle_late);
            break;
          }
          std::this_thread::sleep_until(next_cycle);
        }
        if (!running_.load(std::memory_order_acquire)) {
          requestStop(*state.shared, StopCode::shutdown);
        }
      });
    } catch (const std::exception &error) {
      {
        std::lock_guard<std::mutex> lock(state.shared->error_mutex);
        if (state.shared->sdk_error.empty()) {
          state.shared->sdk_error =
              session.arm->side + " SDK loop: " + error.what();
        }
      }
      requestStop(*state.shared, StopCode::sdk_failure);
    } catch (...) {
      requestStop(*state.shared, StopCode::sdk_failure);
    }
  }

  bool cleanup(Session &session) noexcept {
    if (!session.arm) return true;
    bool okay = true;
    auto attempt = [&](const char *stage, auto action) {
      try {
        action();
      } catch (const std::exception &error) {
        okay = false;
        RCLCPP_ERROR(
            get_logger(), "%s ServoL cleanup %s: %s",
            session.arm->side.c_str(), stage, error.what());
      }
    };

    if (session.move_started) {
      attempt("stopMove", [&]() {
        session.arm->hardware->withRobot(
            [&](rokae::ArRobot &) { session.controller->stopMove(); });
      });
      session.move_started = false;
    }
    if (session.servo_enabled) {
      attempt("stopServoJoint", [&]() {
        session.arm->hardware->withRobot(
            [&](rokae::ArRobot &) {
              session.controller->stopServoJoint();
            });
      });
      session.servo_enabled = false;
    }
    if (session.receiving_state) {
      attempt("stopReceiveRobotState", [&]() {
        session.arm->hardware->withRobot(
            [&](rokae::ArRobot &robot) { robot.stopReceiveRobotState(); });
      });
      session.receiving_state = false;
    }
    if (session.realtime) {
      attempt("restore NrtCommand", [&]() {
        session.arm->hardware->withRobot([&](rokae::ArRobot &robot) {
          std::error_code error;
          robot.setMotionControlMode(
              rokae::MotionControlMode::NrtCommand, error);
          check(error, "restore NrtCommand");
        });
      });
      attempt("verify power", [&]() {
        session.arm->hardware->withRobot([&](rokae::ArRobot &robot) {
          std::error_code error;
          const auto power = robot.powerState(error);
          check(error, "power after stop");
          if (power != rokae::PowerState::on) {
            throw std::runtime_error(
                "power is not on after stop; not repowering");
          }
        });
      });
      session.realtime = false;
    }
    return okay;
  }

  std::string stopReason(const SharedRun &run) {
    switch (run.stop.load(std::memory_order_acquire)) {
      case StopCode::none:
      case StopCode::requested:
      case StopCode::shutdown:
        return {};
      case StopCode::watchdog:
        return "command watchdog expired";
      case StopCode::alignment_timeout:
        return "first ROS target did not align with the current EndInRef TCP pose";
      case StopCode::target_step:
        return "EndInRef ROS target jump exceeds the configured safety limit";
      case StopCode::joint_limit:
      case StopCode::sdk_failure: {
        std::lock_guard<std::mutex> lock(
            const_cast<SharedRun &>(run).error_mutex);
        return run.sdk_error.empty()
                   ? "SDK realtime loop failed"
                   : run.sdk_error;
      }
      case StopCode::state_timeout:
        return "realtime measured-joint feedback timeout";
      case StopCode::cycle_late:
        return "100 Hz ServoL loop exceeded the lateness limit";
    }
    return "unknown ServoL stop reason";
  }

  void run(int mode) {
    auto &buffer = buffers_[mode];
    std::uint64_t consumed = 0;
    while (running_.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lock(buffer.mutex);
        buffer.condition.wait(lock, [&]() {
          return !running_.load(std::memory_order_acquire) ||
                 buffer.sequence != consumed || buffer.stop;
        });
        if (!running_.load(std::memory_order_acquire)) break;
        if (buffer.stop) {
          buffer.stop = false;
          consumed = buffer.sequence;
          continue;
        }
        consumed = buffer.sequence;
      }

      const int count = mode == 2 ? 2 : 1;
      Session sessions[2];
      for (int index = 0; index < count; ++index) {
        sessions[index].arm = &arms_[mode == 2 ? index : mode];
      }
      std::unique_lock<std::mutex> left(
          sessions[0].arm->hardware->commandMutex(), std::defer_lock);
      std::unique_lock<std::mutex> right;
      if (count == 2) {
        right = std::unique_lock<std::mutex>(
            sessions[1].arm->hardware->commandMutex(), std::defer_lock);
      }

      SharedRun shared;
      std::string error;
      bool clean = true;
      std::vector<std::thread> loops;
      try {
        if (count == 2 ? std::try_lock(left, right) != -1
                       : !left.try_lock()) {
          throw std::runtime_error("arm occupied by another command");
        }
        stage_[mode].store(1, std::memory_order_release);
        report(mode, "starting");
        for (int index = 0; index < count; ++index) {
          const int arm_index = mode == 2 ? index : mode;
          prepare(sessions[index], command_pose_[arm_index]);
          command_pose_[arm_index].active.store(
              true, std::memory_order_release);
        }

        LoopState states[2];
        const auto deadline = Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(alignment_timeout_));
        {
          std::lock_guard<std::mutex> lock(buffer.mutex);
          for (int index = 0; index < count; ++index) {
            states[index].buffer = &buffer;
            states[index].command_pose =
                &command_pose_[mode == 2 ? index : mode];
            states[index].shared = &shared;
            states[index].stage = &stage_[mode];
            states[index].target_index = mode == 2 ? index : 0;
            states[index].alignment_bit = 1 << index;
            states[index].expected_alignment_mask = (1 << count) - 1;
            states[index].last_received = buffer.received;
            states[index].alignment_deadline = deadline;
            states[index].goal_end = sessions[index].initial_end;
          }
        }

        stage_[mode].store(2, std::memory_order_release);
        loops.reserve(count);
        for (int index = 0; index < count; ++index) {
          loops.emplace_back(
              [this, &sessions, &states, index]() {
                runArmLoop(sessions[index], states[index]);
              });
        }
        for (auto &loop : loops) {
          if (loop.joinable()) loop.join();
        }
      } catch (const std::exception &exception) {
        error = exception.what();
        for (int index = 0; index < count; ++index) {
          if (sessions[index].preparation_step) {
            error = sessions[index].arm->side + " prepare [" +
                    sessions[index].preparation_step + "]: " + error;
            break;
          }
        }
        requestStop(shared, StopCode::sdk_failure);
        for (auto &loop : loops) {
          if (loop.joinable()) loop.join();
        }
      }

      stage_[mode].store(0, std::memory_order_release);
      for (int index = 0; index < count; ++index) {
        command_pose_[mode == 2 ? index : mode].active.store(
            false, std::memory_order_release);
      }
      for (int index = count - 1; index >= 0; --index) {
        clean = cleanup(sessions[index]) && clean;
      }
      if (error.empty()) error = stopReason(shared);
      if (!clean) {
        if (!error.empty()) error += "; ";
        error += "cleanup/power verification failed";
      }
      if (!error.empty()) {
        // Report faults immediately so the bridge stops its target stream.
        report(mode, "error:" + error);
        RCLCPP_ERROR(
            get_logger(), "ServoL mode %d: %s",
            mode + 1, error.c_str());
      }
      std::unique_lock<std::mutex> lock(buffer.mutex);
      while (running_.load(std::memory_order_acquire) &&
             Clock::now() - buffer.received <
                 std::chrono::duration<double>(watchdog_)) {
        buffer.condition.wait_for(
            lock, std::chrono::duration<double>(watchdog_));
      }
      consumed = buffer.sequence;
      buffer.stop = false;
      // Keep this arm occupied until the previous target stream has drained.
      // Otherwise a new power/pose request can succeed while this worker is
      // still discarding incoming targets from the preceding session.
      lock.unlock();
      if (left.owns_lock()) left.unlock();
      if (right.owns_lock()) right.unlock();
      // A normal stop is acknowledged only when NRT restoration, draining,
      // and command ownership release have all completed.
      if (error.empty()) report(mode, "stopped");
    }
  }

  void publishCommandState() {
    for (int index = 0; index < 2; ++index) {
      if (!command_pose_[index].active.load(
              std::memory_order_acquire)) {
        continue;
      }
      Matrix matrix{};
      if (!command_pose_[index].read(matrix)) continue;
      try {
        geometry_msgs::msg::PoseStamped message;
        message.header.stamp = now();
        message.header.frame_id =
            arms_[index].side + "_external_ref_command";
        message.pose = rokae_driver::servol::pose(matrix);
        command_pose_publishers_[index]->publish(message);
      } catch (const std::exception &error) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "%s ServoL command-pose conversion failed: %s",
            arms_[index].side.c_str(), error.what());
      }
    }
    for (int mode = 0; mode < 3; ++mode) {
      const int stage = stage_[mode].load(std::memory_order_acquire);
      if (stage == 1) report(mode, "starting");
      if (stage == 2) report(mode, "ready");
      if (stage == 3) report(mode, "running");
    }
  }

  double period_{0.01};
  double lookahead_{0.01};
  double gain_{0.0};
  double watchdog_{0.10};
  double state_timeout_{0.10};
  double max_cycle_lateness_{0.04};
  double translation_step_{0.0005};
  double rotation_step_{0.004363323129985824};
  double max_target_jump_translation_{0.10};
  double max_target_jump_rotation_{0.7853981633974483};
  double joint_margin_{0.08};
  double alignment_timeout_{2.0};
  int network_tolerance_{20};
  Arm arms_[2];
  CommandBuffer buffers_[3];
  CommandPoseSlot command_pose_[2];
  std::thread workers_[3];
  std::atomic<bool> running_{true};
  std::atomic<int> stage_[3]{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      command_pose_publishers_[2];
  rclcpp::Subscription<Pose>::SharedPtr single_subscriptions_[2];
  rclcpp::Subscription<rokae_interfaces::msg::ServoLJog>::SharedPtr
      jog_subscriptions_[2];
  rclcpp::Subscription<rokae_interfaces::msg::DualArmServoL>::SharedPtr
      dual_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr
      global_stop_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr
      mode_stop_subscriptions_[3];
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

}  // namespace

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeServoLSubscriber() {
  return std::make_shared<ServoL>();
}

}  // namespace rokae_driver
