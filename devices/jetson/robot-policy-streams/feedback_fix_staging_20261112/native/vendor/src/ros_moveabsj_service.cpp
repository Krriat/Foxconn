/**
 * @file ros_moveabsj_service.cpp
 * @brief Blocking ROS 2 services backed by xCoreSDK MoveAbsJCommand.
 *
 * Services:
 *   /robot_02/upperlimb/move_absj/left_arm
 *   /robot_02/upperlimb/move_absj/right_arm
 */

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rokae/data_types.h"
#include "rokae/robot.h"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_driver/joint_goal_delta.hpp"
#include "rokae_driver/moveabsj_path.hpp"
#include "rokae_driver/cartesian_movej.hpp"
#include "rokae_interfaces/srv/move_abs_j.hpp"

namespace {

using MoveAbsJ = rokae_interfaces::srv::MoveAbsJ;
using MoveAbsJPath = rokae_interfaces::srv::MoveAbsJPath;
constexpr std::size_t kJointCount = 7;
constexpr auto kPollPeriod = std::chrono::milliseconds(50);
constexpr auto kMotionStartGracePeriod = std::chrono::milliseconds(100);
constexpr int kSpeedViolationSamples = 2;
using JointArray = std::array<double, kJointCount>;

struct Options {
  double jointSpeedScale{0.10};
  double timeoutS{60.0};
  double motionStartTimeoutS{3.0};
  double maxSpeedMmS{4000.0};
  double maxZoneMm{200.0};
  double softLimitMarginRad{0.08};
  double maxGoalDeltaRad{2.3};
  double maxJointSpeedRadS{0.60};
  double goalToleranceRad{0.01};
};

struct ArmContext {
  std::string side;
  rokae_driver::joint_goal_delta::Limits maxGoalDeltaRad;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
  rclcpp::Service<MoveAbsJ>::SharedPtr service;
  rclcpp::Service<MoveAbsJPath>::SharedPtr pathService;
};

std::string sdkError(const std::string &operation,
                     const std::error_code &error) {
  return operation + " failed: code=" + std::to_string(error.value()) +
         ", message=" + error.message();
}

bool finite(const JointArray &values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

class RokaeMoveAbsJService : public rclcpp::Node {
 public:
  RokaeMoveAbsJService()
      : Node("joint_motion_service",
             rokae_driver::interface_names::kNodeNamespace) {
    options_.jointSpeedScale =
        declare_parameter<double>("joint_speed_scale", 0.10);
    options_.timeoutS = declare_parameter<double>("timeout_s", 60.0);
    options_.motionStartTimeoutS =
        declare_parameter<double>("motion_start_timeout_s", 3.0);
    options_.maxSpeedMmS =
        declare_parameter<double>("max_speed_mm_s", 4000.0);
    options_.maxZoneMm = declare_parameter<double>("max_zone_mm", 200.0);
    options_.softLimitMarginRad =
        declare_parameter<double>("soft_limit_margin_rad", 0.08);
    options_.maxGoalDeltaRad =
        declare_parameter<double>("max_goal_delta_rad", 2.3);
    options_.maxJointSpeedRadS =
        declare_parameter<double>("max_joint_speed_rad_s", 0.60);
    options_.goalToleranceRad =
        declare_parameter<double>("goal_tolerance_rad", 0.01);
    validateOptions();

    left_ = createArm("left", "192.168.1.160", "192.168.1.10");
    right_ = createArm("right", "192.168.2.160", "192.168.2.10");
    callbackGroup_ = create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    createService(*left_);
    createService(*right_);

    RCLCPP_WARN(
        get_logger(),
        "MoveAbsJ services are ready under %s/move_absj. They do not power "
        "on the robots; verify the target and workspace before calling.",
        rokae_driver::interface_names::kUpperlimbPrefix);
  }

 private:
  std::unique_ptr<ArmContext> createArm(
      const std::string &side, const std::string &defaultRobotIp,
      const std::string &defaultLocalIp) {
    auto arm = std::make_unique<ArmContext>();
    arm->side = side;
    const std::string prefix = side + "_arm";
    const auto j3Delta = declare_parameter<double>(
        prefix + ".j3_max_goal_delta_rad", options_.maxGoalDeltaRad);
    const auto j5Delta = declare_parameter<double>(
        prefix + ".j5_max_goal_delta_rad", options_.maxGoalDeltaRad);
    arm->maxGoalDeltaRad = rokae_driver::joint_goal_delta::makeLimits(
        options_.maxGoalDeltaRad, j5Delta, j3Delta);
    RCLCPP_INFO(get_logger(), "%s J3 maximum goal change: %.9f rad (%.3f degrees)",
                side.c_str(), j3Delta, j3Delta * 180.0 / rokae_driver::joint_goal_delta::kPi);
    RCLCPP_INFO(get_logger(), "%s J5 maximum goal change: %.9f rad (%.3f degrees)",
                side.c_str(), j5Delta, j5Delta * 180.0 / rokae_driver::joint_goal_delta::kPi);
    const auto robotIp = declare_parameter<std::string>(
        prefix + ".robot_ip", defaultRobotIp);
    const auto localIp = declare_parameter<std::string>(
        prefix + ".local_ip", defaultLocalIp);
    arm->hardware = rokae_driver::sharedArm(side, robotIp, localIp);
    return arm;
  }

  void createService(ArmContext &arm) {
    ArmContext *const armPtr = &arm;
    const std::string serviceName =
        rokae_driver::interface_names::arm("move_absj", arm.side);
    arm.service = create_service<MoveAbsJ>(
        serviceName,
        [this, armPtr](const MoveAbsJ::Request::SharedPtr request,
                       MoveAbsJ::Response::SharedPtr response) {
          execute(*armPtr, *request, *response);
        },
        rmw_qos_profile_services_default, callbackGroup_);
    arm.pathService = create_service<MoveAbsJPath>(
        rokae_driver::interface_names::arm("move_absj_path", arm.side),
        [this, armPtr](const MoveAbsJPath::Request::SharedPtr request,
                       MoveAbsJPath::Response::SharedPtr response) {
          try {
            MoveAbsJ::Response result;
            if (request->use_cartesian_targets) {
              const auto count = rokae_driver::moveabsj_path::validateCartesian(*request);
              const auto terminal = rokae_driver::moveabsj_path::pointOptions(*request, count - 1);
              execute(*armPtr, terminal, result, {}, request.get());
            } else {
              const auto points = rokae_driver::moveabsj_path::decode(*request);
              execute(*armPtr, points.back(), result, points);
            }
            response->success = result.success;
            response->message = result.message;
          } catch (const std::exception &error) {
            response->success = false;
            response->message = error.what();
          }
        }, rmw_qos_profile_services_default, callbackGroup_);
    RCLCPP_INFO(get_logger(), "%s service is ready", serviceName.c_str());
  }

  void validateOptions() const {
    const std::array<double, 9> values{
        options_.jointSpeedScale, options_.timeoutS,
        options_.motionStartTimeoutS, options_.maxSpeedMmS,
        options_.maxZoneMm, options_.softLimitMarginRad,
        options_.maxGoalDeltaRad, options_.maxJointSpeedRadS,
        options_.goalToleranceRad};
    if (!std::all_of(values.begin(), values.end(), [](double value) {
          return std::isfinite(value);
        })) {
      throw std::invalid_argument("MoveAbsJ parameters must be finite");
    }
    if (options_.jointSpeedScale < 0.01 || options_.jointSpeedScale > 1.0) {
      throw std::invalid_argument("joint_speed_scale must be in [0.01, 1]");
    }
    if (options_.timeoutS < 1.0 || options_.timeoutS > 300.0) {
      throw std::invalid_argument("timeout_s must be in [1, 300]");
    }
    if (options_.motionStartTimeoutS <= 0.0 ||
        options_.motionStartTimeoutS > std::min(options_.timeoutS, 10.0)) {
      throw std::invalid_argument(
          "motion_start_timeout_s must be in (0, min(timeout_s, 10)]");
    }
    if (options_.maxSpeedMmS <= 0.0 || options_.maxSpeedMmS > 5000.0) {
      throw std::invalid_argument("max_speed_mm_s must be in (0, 5000]");
    }
    if (options_.maxZoneMm < 0.0 || options_.maxZoneMm > 200.0) {
      throw std::invalid_argument("max_zone_mm must be in [0, 200]");
    }
    if (options_.softLimitMarginRad < 0.0 ||
        options_.softLimitMarginRad > 0.30) {
      throw std::invalid_argument(
          "soft_limit_margin_rad must be in [0, 0.30]");
    }
    if (options_.maxGoalDeltaRad <= 0.0 || options_.maxGoalDeltaRad > rokae_driver::joint_goal_delta::kMaxGoalDeltaRad) {
      throw std::invalid_argument("max_goal_delta_rad must be in (0, 2*pi]");
    }
    if (options_.maxJointSpeedRadS <= 0.0 ||
        options_.maxJointSpeedRadS > 3.0) {
      throw std::invalid_argument(
          "max_joint_speed_rad_s must be in (0, 3]");
    }
    if (options_.goalToleranceRad <= 0.0 ||
        options_.goalToleranceRad > 0.20) {
      throw std::invalid_argument("goal_tolerance_rad must be in (0, 0.20]");
    }
  }

  bool validateRequest(const MoveAbsJ::Request &request,
                       JointArray &target, std::string &reason) const {
    std::copy(request.joint_positions.begin(), request.joint_positions.end(),
              target.begin());
    if (!finite(target) || !std::isfinite(request.speed_mm_s) ||
        !std::isfinite(request.zone_mm)) {
      reason = "joint_positions, speed_mm_s and zone_mm must be finite";
      return false;
    }
    if (request.speed_mm_s <= 0.0 ||
        request.speed_mm_s > options_.maxSpeedMmS) {
      reason = "speed_mm_s must be in (0, " +
               std::to_string(options_.maxSpeedMmS) + "]";
      return false;
    }
    if (request.zone_mm < 0.0 || request.zone_mm > options_.maxZoneMm) {
      reason = "zone_mm must be in [0, " +
               std::to_string(options_.maxZoneMm) + "]";
      return false;
    }
    if (!std::isfinite(request.joint_speed_scale) ||
        request.joint_speed_scale < 0.0 || request.joint_speed_scale > 1.0) {
      reason = "joint_speed_scale must be 0 (node default) or in (0, 1]";
      return false;
    }
    return true;
  }

  bool validateRobotAndTarget(
      ArmContext &arm, const std::vector<MoveAbsJ::Request> &points,
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
      reason = arm.side + " arm must be idle before MoveAbsJ";
      return false;
    }

    error.clear();
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

    error.clear();
    const JointArray current = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) { return robot.jointPos(error); });
    if (error || !finite(current)) {
      reason = error ? sdkError("jointPos", error)
                     : "current joint feedback is not finite";
      return false;
    }
    JointArray previous = current;
    for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
      const auto &target = points[pointIndex].joint_positions;
      for (std::size_t index = 0; index < kJointCount; ++index) {
        const double lower = softLimits[index][0] + options_.softLimitMarginRad;
        const double upper = softLimits[index][1] - options_.softLimitMarginRad;
        if (lower >= upper || target[index] < lower || target[index] > upper) {
          reason = "J" + std::to_string(index + 1) +
                   " target violates the controller soft-limit margin";
          return false;
        }
        const double delta = std::abs(target[index] - previous[index]);
        if (!rokae_driver::joint_goal_delta::withinLimit(
                previous[index], target[index], index, arm.maxGoalDeltaRad)) {
          reason = "J" + std::to_string(index + 1) + " waypoint " + std::to_string(pointIndex + 1) +
                   " requested change exceeds max_goal_delta_rad: " + std::to_string(delta);
          return false;
        }
      }
      previous = target;
    }
    return true;
  }

  void execute(ArmContext &arm, const MoveAbsJ::Request &request,
               MoveAbsJ::Response &response,
               const std::vector<MoveAbsJ::Request> &waypoints = {},
               const MoveAbsJPath::Request *cartesianPath = nullptr) {
    std::unique_lock<std::mutex> commandLock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!commandLock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another control command");
      return;
    }

    bool motionStarted = false;
    const auto generation = arm.hardware->stopGeneration();
    try {
      if (!std::isfinite(request.timeout_s) || request.timeout_s < 0. || request.timeout_s > 600. ||
          !std::isfinite(request.joint_tolerance_rad) || request.joint_tolerance_rad < 0. || request.joint_tolerance_rad > .2 ||
          request.stable_samples < 1 || request.stable_samples > 100 ||
          request.poll_interval_ms < 10 || request.poll_interval_ms > 1000)
        throw std::runtime_error("invalid MoveAbsJ feedback settings");
      auto points = waypoints.empty() ? std::vector<MoveAbsJ::Request>{request} : waypoints;
      if (cartesianPath) {
        arm.hardware->withRobot([&](rokae::ArRobot &robot) {
          std::error_code error;
          const auto originalTool = robot.toolset(error);
          if (error) throw std::runtime_error(sdkError("toolset(Cartesian MoveAbsJPath)", error));
          // Explicit calculation tool only; do not change the controller TCP.
          const auto tool = rokae_driver::cartesian_movej::selectTool(originalTool, cartesianPath->tcp_type);
          auto model = robot.model();
          points = rokae_driver::moveabsj_path::resolveCartesian(*cartesianPath,
              [&](std::size_t, const geometry_msgs::msg::Pose &pose, bool useElbow, double elbow) {
                if (generation != arm.hardware->stopGeneration())
                  throw std::runtime_error("stop interrupted Cartesian MoveAbsJPath IK");
                const auto q = rokae_driver::cartesian_movej::quaternion(pose);
                rokae::CartesianPosition target;
                target.trans = {pose.position.x, pose.position.y, pose.position.z};
                target.rpy = rokae::Utils::quaternionToEuler(q.w(), q.x(), q.y(), q.z());
                target.hasElbow = useElbow;
                target.elbow = elbow;
                const auto joints = model.calcIk(target, tool, error);
                if (error) throw std::runtime_error(sdkError("calcIk", error));
                const auto solved = model.calcFk(joints, tool, error);
                if (error) throw std::runtime_error(sdkError("calcFk(IK verification)", error));
                const auto sq = rokae::Utils::eulerToQuaternion(solved.rpy);
                const double distance = std::hypot(std::hypot(solved.trans[0] - target.trans[0],
                                                             solved.trans[1] - target.trans[1]),
                                                   solved.trans[2] - target.trans[2]);
                const double dot = sq[0]*q.w() + sq[1]*q.x() + sq[2]*q.y() + sq[3]*q.z();
                const double angle = 2.*std::acos(std::clamp(std::abs(dot), 0., 1.));
                if (!std::isfinite(distance) || !std::isfinite(angle) || distance > 1e-4 || angle > 1e-3 ||
                    (useElbow && (!std::isfinite(solved.elbow) || std::abs(solved.elbow - elbow) > 1e-4)))
                  throw std::runtime_error("IK failed FK pose/elbow verification");
                return joints;
              });
        });
      }
      JointArray target = points.back().joint_positions;
      std::string reason;
      for (const auto &point : points) {
        JointArray validated{};
        if (!validateRequest(point, validated, reason)) {
          fail(response, reason);
          return;
        }
      }
      std::array<double[2], kJointCount> softLimits{};
      if (!validateRobotAndTarget(arm, points, softLimits, reason)) {
        fail(response, reason);
        return;
      }

      std::error_code error;
      std::string commandId;
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        robot.setMotionControlMode(
            rokae::MotionControlMode::NrtCommand, error);
        if (error) return;
        robot.setOperateMode(rokae::OperateMode::automatic, error);
        if (error) return;
        robot.adjustSpeedOnline(1.0, error);
        if (error) return;
        robot.moveReset(error);
        if (error) return;

        std::vector<rokae::MoveAbsJCommand> commands;
        for (std::size_t index = 0; index < points.size(); ++index) {
          const auto &point = points[index];
          // The terminal point has no outgoing segment and must stop exactly.
          rokae::MoveAbsJCommand command(
              rokae::JointPosition(std::vector<double>(point.joint_positions.begin(), point.joint_positions.end())),
              point.speed_mm_s, index + 1 == points.size() ? 0. : point.zone_mm);
          command.jointSpeed = point.joint_speed_scale > 0.0 ? point.joint_speed_scale : options_.jointSpeedScale;
          commands.push_back(command);
        }
        robot.moveAppend(commands, commandId, error);
      });
      if (error) {
        stopAndReset(arm);
        fail(response, sdkError("prepare MoveAbsJ", error));
        return;
      }
      // Treat an exception from moveStart as potentially motion-producing.
      motionStarted = true;
      arm.hardware->withRobot(
          [&](rokae::ArRobot &robot) {
            if (generation != arm.hardware->stopGeneration()) throw std::runtime_error("stop interrupted MoveAbsJ planning");
            robot.moveStart(error);
          });
      if (error) {
        stopAndReset(arm);
        fail(response, sdkError("moveStart", error));
        return;
      }
      RCLCPP_WARN(
          get_logger(),
          "%s MoveAbsJ service started: command_id=%s, speed=%.3f mm/s, "
          "zone=%.3f mm, joint_speed_scale=%.3f, waypoints=%zu",
          arm.side.c_str(), commandId.c_str(), request.speed_mm_s,
          request.zone_mm, request.joint_speed_scale > 0.0
              ? request.joint_speed_scale : options_.jointSpeedScale, points.size());

      const auto commandStart = std::chrono::steady_clock::now();
      const auto deadline = commandStart +
          std::chrono::duration<double>(request.timeout_s > 0. ? request.timeout_s : options_.timeoutS);
      const auto motionStartDeadline = commandStart +
          std::chrono::duration<double>(options_.motionStartTimeoutS);
      bool motionObserved = false;
      int speedViolationCount = 0;
      unsigned stable = 0;
      std::this_thread::sleep_for(kMotionStartGracePeriod);

      while (rclcpp::ok()) {
        if (generation != arm.hardware->stopGeneration()) { reason = "stop service interrupted MoveAbsJ"; break; }
        const JointArray joints = arm.hardware->withRobot(
            [&error](rokae::ArRobot &robot) {
              error.clear();
              return robot.jointPos(error);
            });
        if (error || !finite(joints)) {
          reason = error ? sdkError("jointPos", error)
                         : "joint feedback is not finite";
          break;
        }
        const JointArray velocities = arm.hardware->withRobot(
            [&error](rokae::ArRobot &robot) {
              error.clear();
              return robot.jointVel(error);
            });
        if (error || !finite(velocities)) {
          reason = error ? sdkError("jointVel", error)
                         : "joint velocity is not finite";
          break;
        }

        bool speedExceeded = false;
        std::size_t fastestJoint = 0;
        double fastestSpeed = 0.0;
        for (std::size_t index = 0; index < kJointCount; ++index) {
          if (joints[index] < softLimits[index][0] ||
              joints[index] > softLimits[index][1]) {
            reason = "J" + std::to_string(index + 1) +
                     " crossed the controller soft limit";
            break;
          }
          const double speed = std::abs(velocities[index]);
          if (speed > fastestSpeed) {
            fastestSpeed = speed;
            fastestJoint = index;
          }
          speedExceeded = speedExceeded ||
                          speed > options_.maxJointSpeedRadS;
        }
        if (!reason.empty()) break;
        speedViolationCount = speedExceeded ? speedViolationCount + 1 : 0;
        if (speedViolationCount >= kSpeedViolationSamples) {
          reason = "J" + std::to_string(fastestJoint + 1) +
                   " speed exceeded max_joint_speed_rad_s: " +
                   std::to_string(fastestSpeed);
          break;
        }

        const auto state = arm.hardware->withRobot(
            [&error](rokae::ArRobot &robot) {
              error.clear();
              return robot.operationState(error);
            });
        if (error) {
          reason = sdkError("operationState", error);
          break;
        }
        if (state == rokae::OperationState::unknown) {
          reason = "controller entered unknown operation state";
          break;
        }

        const auto now = std::chrono::steady_clock::now();
        if (state != rokae::OperationState::idle) {
          motionObserved = true;
          stable = 0;
        } else {
          double maximumError = 0.0;
          for (std::size_t index = 0; index < kJointCount; ++index) {
            maximumError = std::max(
                maximumError, std::abs(target[index] - joints[index]));
          }
          const auto event = arm.hardware->withRobot([&](rokae::ArRobot &robot) {
            return robot.queryEventInfo(rokae::Event::moveExecution, error);
          });
          if (error) { reason = sdkError("moveExecution", error); break; }
          using namespace rokae::EventInfoKey::MoveExecution;
          const auto id = event.find(ID), reachedEvent = event.find(ReachTarget), errorEvent = event.find(Error);
          const auto indexEvent = event.find(WaypointIndex);
          const auto *waypointIndex = indexEvent == event.end() ? nullptr : std::any_cast<int>(&indexEvent->second);
          const auto *eventId = id == event.end() ? nullptr : std::any_cast<std::string>(&id->second);
          const auto *eventReached = reachedEvent == event.end() ? nullptr : std::any_cast<bool>(&reachedEvent->second);
          const auto *eventError = errorEvent == event.end() ? nullptr : std::any_cast<std::error_code>(&errorEvent->second);
          if (eventId && *eventId == commandId && eventError && *eventError) { reason = sdkError("moveExecution", *eventError); break; }
          const bool targetReached = maximumError <= (request.joint_tolerance_rad > 0. ? request.joint_tolerance_rad : options_.goalToleranceRad);
          const bool finalWaypoint = points.size() == 1 ||
              (waypointIndex && *waypointIndex == static_cast<int>(points.size() - 1));
          const bool reached = targetReached && finalWaypoint && eventId && *eventId == commandId && eventReached && *eventReached;
          stable = reached ? stable + 1 : 0;
          if (stable >= request.stable_samples) {
            response.success = true;
            response.message = "MoveAbsJ completed; command_id=" + commandId +
                               "; waypoints=" + std::to_string(points.size()) + "; maximum_error_rad=" +
                               std::to_string(maximumError);
            RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
            return;
          }
          if (!targetReached && motionObserved) {
            reason = "robot became idle before reaching the MoveAbsJ target; "
                     "maximum_error_rad=" + std::to_string(maximumError);
            break;
          }
          if (!targetReached && now >= motionStartDeadline) {
            reason = "MoveAbsJ did not enter a running state within " +
                     std::to_string(options_.motionStartTimeoutS) + " s";
            break;
          }
        }
        if (now >= deadline) {
          reason = "MoveAbsJ timed out after " +
                   std::to_string(request.timeout_s > 0. ? request.timeout_s : options_.timeoutS) + " s";
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(request.poll_interval_ms));
      }
      if (reason.empty()) reason = "ROS shutdown interrupted MoveAbsJ";
      stopAndReset(arm);
      fail(response, reason);
    } catch (const std::exception &exception) {
      if (motionStarted) stopAndReset(arm);
      fail(response, std::string("MoveAbsJ service exception: ") +
                         exception.what());
    }
  }

  void stopAndReset(ArmContext &arm) noexcept {
    try {
      std::error_code error;
      arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) { robot.stop(error); });
      error.clear();
      arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) { robot.moveReset(error); });
      if (error) {
        RCLCPP_ERROR(get_logger(), "%s", sdkError("moveReset", error).c_str());
      }
    } catch (const std::exception &exception) {
      RCLCPP_ERROR(get_logger(), "%s stop/reset exception: %s",
                   arm.side.c_str(), exception.what());
    }
  }

  void fail(MoveAbsJ::Response &response, const std::string &message) {
    response.success = false;
    response.message = message;
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
  }

  Options options_;
  std::unique_ptr<ArmContext> left_;
  std::unique_ptr<ArmContext> right_;
  rclcpp::CallbackGroup::SharedPtr callbackGroup_;
};

}  // namespace

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeMoveAbsJService() {
  return std::make_shared<RokaeMoveAbsJService>();
}

}  // namespace rokae_driver

#ifndef ROKAE_UNIFIED_DRIVER
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int exitCode = 0;
  try {
    auto node = std::make_shared<RokaeMoveAbsJService>();
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception &exception) {
    RCLCPP_FATAL(rclcpp::get_logger("rokae_moveabsj_service"), "%s",
                 exception.what());
    exitCode = 1;
  }
  rclcpp::shutdown();
  return exitCode;
}
#endif
