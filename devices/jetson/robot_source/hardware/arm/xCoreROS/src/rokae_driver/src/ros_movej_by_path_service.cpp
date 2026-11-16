/**
 * @file ros_movej_by_path_service.cpp
 * @brief Multi-waypoint joint-space motion through the xCoreSDK queue.
 *
 * The service sends one vector of MoveAbsJCommand objects with moveAppend(),
 * then starts the queue once. A positive zone_mm lets the controller blend
 * adjacent waypoints; the service never emulates a path by calling MoveAbsJ
 * once per point.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rokae/data_types.h"
#include "rokae/robot.h"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_driver/cartesian_movej.hpp"
#include "rokae_interfaces/srv/move_j_by_path.hpp"

namespace {

using MoveJByPath = rokae_interfaces::srv::MoveJByPath;
constexpr std::size_t kJointCount = 7;
constexpr std::size_t kMaxWaypoints = 100;
constexpr auto kPollPeriod = std::chrono::milliseconds(100);
constexpr auto kMotionStartGracePeriod = std::chrono::milliseconds(300);
using JointArray = std::array<double, kJointCount>;

struct Options {
  double speedMmS{80.0};
  double jointSpeedScale{0.10};
  double zoneMm{5.0};
  double timeoutS{120.0};
  double softLimitMarginRad{0.08};
  double maxWaypointDeltaRad{0.35};
  double maxCartesianJointDeltaRad{2.3};
  double goalToleranceRad{0.02};
};

struct ArmContext {
  std::string side;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
  rclcpp::Service<MoveJByPath>::SharedPtr service;
};

struct Execution {
  ArmContext *arm{nullptr};
  std::unique_lock<std::mutex> commandLock;
  bool started{false};
  std::string commandId;
};

std::string sdkError(const std::string &operation,
                     const std::error_code &error) {
  return operation + " failed: code=" + std::to_string(error.value()) +
         ", message=" + error.message();
}

bool finite(const JointArray &joints) {
  return std::all_of(joints.begin(), joints.end(),
                     [](double value) { return std::isfinite(value); });
}

bool finite(const std::vector<double> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

std::vector<JointArray> decodePath(const std::vector<double> &raw,
                                   std::string &reason) {
  if (raw.size() < 2 * kJointCount || raw.size() % kJointCount != 0) {
    reason = "joint_positions must contain 2..100 complete waypoints "
             "(N * 7 values)";
    return {};
  }
  const auto count = raw.size() / kJointCount;
  if (count > kMaxWaypoints) {
    reason = "a path may contain at most 100 waypoints";
    return {};
  }
  if (!finite(raw)) {
    reason = "path positions must all be finite";
    return {};
  }

  std::vector<JointArray> path;
  path.reserve(count);
  for (std::size_t point = 0; point < count; ++point) {
    JointArray joints{};
    std::copy_n(raw.begin() + point * kJointCount, kJointCount,
                joints.begin());
    path.push_back(joints);
  }
  return path;
}

class RokaeMoveJByPathService : public rclcpp::Node {
 public:
  RokaeMoveJByPathService()
      : Node("joint_path", rokae_driver::interface_names::kNodeNamespace) {
    options_.speedMmS = declare_parameter<double>("speed_mm_s", 80.0);
    options_.jointSpeedScale =
        declare_parameter<double>("joint_speed_scale", 0.10);
    options_.zoneMm = declare_parameter<double>("zone_mm", 5.0);
    options_.timeoutS = declare_parameter<double>("timeout_s", 120.0);
    options_.softLimitMarginRad = declare_parameter<double>(
        "soft_limit_margin_rad", 0.08);
    options_.maxWaypointDeltaRad = declare_parameter<double>(
        "max_waypoint_delta_rad", 0.35);
    options_.maxCartesianJointDeltaRad = declare_parameter<double>(
        "max_cartesian_joint_delta_rad", 2.3);
    options_.goalToleranceRad = declare_parameter<double>(
        "goal_tolerance_rad", 0.02);
    validateOptions();

    left_ = createArm("left", "192.168.1.160", "192.168.1.10");
    right_ = createArm("right", "192.168.2.160", "192.168.2.10");
    motionGroup_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    left_->service = create_service<MoveJByPath>(
        rokae_driver::interface_names::arm("movej_by_path", "left"),
        [this](const MoveJByPath::Request::SharedPtr request,
               MoveJByPath::Response::SharedPtr response) {
          handleSingle(*left_, *request, *response);
        }, rmw_qos_profile_services_default, motionGroup_);
    right_->service = create_service<MoveJByPath>(
        rokae_driver::interface_names::arm("movej_by_path", "right"),
        [this](const MoveJByPath::Request::SharedPtr request,
               MoveJByPath::Response::SharedPtr response) {
          handleSingle(*right_, *request, *response);
        }, rmw_qos_profile_services_default, motionGroup_);
    dualService_ = create_service<MoveJByPath>(
        rokae_driver::interface_names::dualArm("movej_by_path"),
        [this](const MoveJByPath::Request::SharedPtr request,
               MoveJByPath::Response::SharedPtr response) {
          handleDual(*request, *response);
        }, rmw_qos_profile_services_default, motionGroup_);

    RCLCPP_INFO(
        get_logger(),
        "MoveJ-by-path services ready under %s/movej_by_path; "
        "SDK queue limit=%zu waypoints, default blending zone=%.1f mm",
        rokae_driver::interface_names::kUpperlimbPrefix, kMaxWaypoints,
        options_.zoneMm);
  }

 private:
  rclcpp::CallbackGroup::SharedPtr motionGroup_;
  std::unique_ptr<ArmContext> createArm(
      const std::string &side, const std::string &defaultRobotIp,
      const std::string &defaultLocalIp) {
    const std::string prefix = side + "_arm";
    const auto robotIp = declare_parameter<std::string>(
        prefix + ".robot_ip", defaultRobotIp);
    const auto localIp = declare_parameter<std::string>(
        prefix + ".local_ip", defaultLocalIp);
    auto arm = std::make_unique<ArmContext>();
    arm->side = side;
    arm->hardware = rokae_driver::sharedArm(side, robotIp, localIp);
    return arm;
  }

  void validateOptions() const {
    if (!std::isfinite(options_.speedMmS) || options_.speedMmS <= 0.0 ||
        options_.speedMmS > 4000.0) {
      throw std::invalid_argument("speed_mm_s must be in (0, 4000]");
    }
    if (!std::isfinite(options_.jointSpeedScale) ||
        options_.jointSpeedScale < 0.01 ||
        options_.jointSpeedScale > 1.0) {
      throw std::invalid_argument(
          "joint_speed_scale must be in [0.01, 1.0]");
    }
    if (!std::isfinite(options_.zoneMm) || options_.zoneMm < 0.0 ||
        options_.zoneMm > 200.0) {
      throw std::invalid_argument("zone_mm must be in [0, 200]");
    }
    if (!std::isfinite(options_.timeoutS) || options_.timeoutS < 1.0 ||
        options_.timeoutS > 600.0) {
      throw std::invalid_argument("timeout_s must be in [1, 600]");
    }
    if (!std::isfinite(options_.softLimitMarginRad) ||
        options_.softLimitMarginRad < 0.0 ||
        options_.softLimitMarginRad > 0.30) {
      throw std::invalid_argument(
          "soft_limit_margin_rad must be in [0, 0.30]");
    }
    if (!std::isfinite(options_.maxWaypointDeltaRad) ||
        options_.maxWaypointDeltaRad <= 0.0 ||
        options_.maxWaypointDeltaRad > 3.14159265358979323846) {
      throw std::invalid_argument(
          "max_waypoint_delta_rad must be in (0, pi]");
    }
    if (!std::isfinite(options_.goalToleranceRad) ||
        options_.goalToleranceRad <= 0.0 ||
        options_.goalToleranceRad > 0.20) {
      throw std::invalid_argument("goal_tolerance_rad must be in (0, 0.20]");
    }
    if (!std::isfinite(options_.maxCartesianJointDeltaRad) ||
        options_.maxCartesianJointDeltaRad <= 0.0 ||
        options_.maxCartesianJointDeltaRad > 3.14159265358979323846) {
      throw std::invalid_argument("max_cartesian_joint_delta_rad must be in (0, pi]");
    }
  }

  bool validatePath(
      ArmContext &arm, const std::vector<JointArray> &path,
      std::array<double[2], kJointCount> &softLimits,
      std::string &reason) {
    std::error_code error;
    const auto power = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) { return robot.powerState(error); });
    if (error) {
      reason = sdkError("powerState", error);
      return false;
    }
    if (power != rokae::PowerState::on) {
      reason = arm.side + " robot is not powered on; call " +
               std::string(rokae_driver::interface_names::kInitializeDualArm) +
               " first";
      return false;
    }

    const auto state = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) {
          return robot.operationState(error);
        });
    if (error) {
      reason = sdkError("operationState", error);
      return false;
    }
    if (state != rokae::OperationState::idle) {
      reason = arm.side + " arm must be idle before accepting a path";
      return false;
    }

    const bool limitsEnabled = arm.hardware->withRobot(
        [&softLimits, &error](rokae::ArRobot &robot) {
          return robot.getSoftLimit(softLimits, error);
        });
    if (error) {
      reason = sdkError("getSoftLimit", error);
      return false;
    }
    if (!limitsEnabled) {
      reason = "controller soft limits are disabled";
      return false;
    }

    const auto current = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) { return robot.jointPos(error); });
    if (error) {
      reason = sdkError("jointPos", error);
      return false;
    }
    if (!finite(current)) {
      reason = "current joint feedback contains a non-finite value";
      return false;
    }

    JointArray previous = current;
    for (std::size_t point = 0; point < path.size(); ++point) {
      const auto &joints = path[point];
      if (!finite(joints)) {
        reason = "path positions must all be finite";
        return false;
      }
      for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        const double lower =
            softLimits[joint][0] + options_.softLimitMarginRad;
        const double upper =
            softLimits[joint][1] - options_.softLimitMarginRad;
        if (lower >= upper) {
          reason = "invalid soft-limit interval on J" +
                   std::to_string(joint + 1);
          return false;
        }
        if (joints[joint] < lower || joints[joint] > upper) {
          reason = "waypoint " + std::to_string(point + 1) + " J" +
                   std::to_string(joint + 1) +
                   " violates the controller soft-limit margin";
          return false;
        }
        if (std::abs(joints[joint] - previous[joint]) >
            options_.maxWaypointDeltaRad) {
          reason = "waypoint " + std::to_string(point + 1) + " J" +
                   std::to_string(joint + 1) +
                   " exceeds max_waypoint_delta_rad";
          return false;
        }
      }
      previous = joints;
    }
    return true;
  }

  bool start(ArmContext &arm, const std::vector<JointArray> &path,
             const std::array<double[2], kJointCount> &softLimits,
             Execution &execution, std::string &reason) {
    (void)softLimits;
    std::vector<rokae::MoveAbsJCommand> commands;
    commands.reserve(path.size());
    for (const auto &joints : path) {
      rokae::MoveAbsJCommand command(
          rokae::JointPosition(
              std::vector<double>(joints.begin(), joints.end())),
          options_.speedMmS, options_.zoneMm);
      command.jointSpeed = options_.jointSpeedScale;
      commands.push_back(command);
    }

    std::error_code error;
    arm.hardware->withRobot([&error](rokae::ArRobot &robot) {
      robot.moveReset(error);
    });
    if (error) {
      reason = sdkError("moveReset", error);
      return false;
    }
    arm.hardware->withRobot([&](rokae::ArRobot &robot) {
      robot.moveAppend(commands, execution.commandId, error);
    });
    if (error) {
      reason = sdkError("moveAppend", error);
      return false;
    }
    // A successful append leaves a live queue even if moveStart fails. Mark
    // it as active before starting so the common failure path always issues
    // moveReset and clears that queue.
    execution.started = true;
    arm.hardware->withRobot([&error](rokae::ArRobot &robot) {
      robot.moveStart(error);
    });
    if (error) {
      reason = sdkError("moveStart", error);
      return false;
    }
    return true;
  }

  bool monitor(Execution &execution, const JointArray &target,
               std::string &reason) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<
                              std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(
                                  options_.timeoutS));
    const auto graceDeadline = std::chrono::steady_clock::now() +
                               kMotionStartGracePeriod;
    bool sawMoving = false;
    while (std::chrono::steady_clock::now() < deadline) {
      std::error_code error;
      const auto state = execution.arm->hardware->withRobot(
          [&error](rokae::ArRobot &robot) {
            return robot.operationState(error);
          });
      if (error) {
        reason = sdkError("operationState", error);
        return false;
      }
      if (state == rokae::OperationState::moving) {
        sawMoving = true;
      } else if (state == rokae::OperationState::unknown) {
        reason = "controller returned unknown operation state";
        return false;
      } else if (state == rokae::OperationState::idle &&
                 (sawMoving || std::chrono::steady_clock::now() >=
                                   graceDeadline)) {
        const auto joints = execution.arm->hardware->withRobot(
            [&error](rokae::ArRobot &robot) { return robot.jointPos(error); });
        if (error) {
          reason = sdkError("jointPos", error);
          return false;
        }
        if (!finite(joints)) {
          reason = "final joint feedback contains a non-finite value";
          return false;
        }
        double maximumError = 0.0;
        for (std::size_t joint = 0; joint < kJointCount; ++joint) {
          maximumError = std::max(
              maximumError, std::abs(joints[joint] - target[joint]));
        }
        if (maximumError > options_.goalToleranceRad) {
          reason = "final joint error " + std::to_string(maximumError) +
                   " rad exceeds goal_tolerance_rad";
          return false;
        }
        return true;
      }
      std::this_thread::sleep_for(kPollPeriod);
    }
    reason = "path execution timed out";
    return false;
  }

  void stop(Execution &execution) {
    if (!execution.started) return;
    std::error_code error;
    execution.arm->hardware->withRobot(
        [&error](rokae::ArRobot &robot) { robot.moveReset(error); });
    if (error) {
      RCLCPP_ERROR(
          get_logger(), "%s moveReset after path failure: %s",
          execution.arm->side.c_str(), error.message().c_str());
    }
    execution.started = false;
  }

  void fail(MoveJByPath::Response &response, const std::string &reason) {
    response.success = false;
    response.message = reason;
  }

  void handleSingle(ArmContext &arm, const MoveJByPath::Request &request,
                    MoveJByPath::Response &response) {
    if (request.use_cartesian_target) {
      std::unique_lock<std::mutex> lock(arm.hardware->commandMutex(), std::try_to_lock);
      if (!lock.owns_lock()) {
        fail(response, arm.side + " arm is occupied by another command");
        return;
      }
      try {
        response.message = rokae_driver::cartesian_movej::execute(
            *arm.hardware, request, options_.softLimitMarginRad,
            options_.maxCartesianJointDeltaRad);
        response.success = true;
      } catch (const std::exception &error) {
        fail(response, error.what());
      }
      return;
    }
    if (!request.left_joint_positions.empty() ||
        !request.right_joint_positions.empty()) {
      fail(response,
           "single-arm movej_by_path uses joint_positions only");
      return;
    }
    std::string reason;
    const auto path = decodePath(request.joint_positions, reason);
    if (path.empty()) {
      fail(response, reason);
      return;
    }
    Execution execution{&arm, std::unique_lock<std::mutex>(
                                 arm.hardware->commandMutex(),
                                 std::try_to_lock)};
    if (!execution.commandLock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another command");
      return;
    }
    std::array<double[2], kJointCount> softLimits{};
    if (!validatePath(arm, path, softLimits, reason) ||
        !start(arm, path, softLimits, execution, reason)) {
      stop(execution);
      fail(response, reason);
      return;
    }
    if (!monitor(execution, path.back(), reason)) {
      stop(execution);
      fail(response, reason);
      return;
    }
    execution.started = false;
    response.success = true;
    response.message = "completed " + std::to_string(path.size()) +
                       " waypoints (command_id=" + execution.commandId +
                       ")";
  }

  void handleDual(const MoveJByPath::Request &request,
                  MoveJByPath::Response &response) {
    if (request.use_cartesian_target) {
      fail(response, "Cartesian MoveJ is supported by single-arm endpoints only");
      return;
    }
    if (!request.joint_positions.empty()) {
      fail(response,
           "dual-arm movej_by_path uses left_joint_positions and "
           "right_joint_positions");
      return;
    }
    std::string reason;
    const auto leftPath = decodePath(request.left_joint_positions, reason);
    if (leftPath.empty()) {
      fail(response, "left path: " + reason);
      return;
    }
    const auto rightPath = decodePath(request.right_joint_positions, reason);
    if (rightPath.empty()) {
      fail(response, "right path: " + reason);
      return;
    }
    if (leftPath.size() != rightPath.size()) {
      fail(response, "left and right paths must have the same waypoint count");
      return;
    }

    std::unique_lock<std::mutex> leftLock(
        left_->hardware->commandMutex(), std::defer_lock);
    std::unique_lock<std::mutex> rightLock(
        right_->hardware->commandMutex(), std::defer_lock);
    if (std::try_lock(leftLock, rightLock) != -1) {
      fail(response, "one or both arms are occupied by another command");
      return;
    }
    Execution leftExecution{left_.get(), std::move(leftLock)};
    Execution rightExecution{right_.get(), std::move(rightLock)};
    std::array<double[2], kJointCount> leftLimits{};
    std::array<double[2], kJointCount> rightLimits{};
    if (!validatePath(*left_, leftPath, leftLimits, reason) ||
        !validatePath(*right_, rightPath, rightLimits, reason) ||
        !start(*left_, leftPath, leftLimits, leftExecution, reason) ||
        !start(*right_, rightPath, rightLimits, rightExecution, reason)) {
      stop(rightExecution);
      stop(leftExecution);
      fail(response, reason);
      return;
    }
    const bool leftOk = monitor(leftExecution, leftPath.back(), reason);
    const std::string leftReason = reason;
    const bool rightOk = monitor(rightExecution, rightPath.back(), reason);
    if (!leftOk || !rightOk) {
      stop(rightExecution);
      stop(leftExecution);
      fail(response, !leftOk ? "left path: " + leftReason
                             : "right path: " + reason);
      return;
    }
    leftExecution.started = false;
    rightExecution.started = false;
    response.success = true;
    response.message = "completed synchronized dual-arm path with " +
                       std::to_string(leftPath.size()) + " waypoints";
  }

  Options options_;
  std::unique_ptr<ArmContext> left_;
  std::unique_ptr<ArmContext> right_;
  rclcpp::Service<MoveJByPath>::SharedPtr dualService_;
};

}  // namespace

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeMoveJByPathService() {
  return std::make_shared<RokaeMoveJByPathService>();
}

}  // namespace rokae_driver

#ifndef ROKAE_UNIFIED_DRIVER
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<RokaeMoveJByPathService>();
    rclcpp::spin(node);
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("rokae_movej_by_path"), "%s",
                 error.what());
  }
  rclcpp::shutdown();
  return 0;
}
#endif
