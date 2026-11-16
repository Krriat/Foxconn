/**
 * @file ros_movel_service.cpp
 * @brief ROS 2 services backed by the xCoreSDK MoveL command.
 *
 * Services:
 *   /robot_02/upperlimb/move_l/left_arm
 *   /robot_02/upperlimb/move_l/right_arm
 *   /robot_02/upperlimb/move_l_relative/left_arm
 *   /robot_02/upperlimb/move_l_relative/right_arm
 *   /robot_02/upperlimb/get_cartesian_state/left_arm
 *   /robot_02/upperlimb/get_cartesian_state/right_arm
 *
 * The request pose is [x, y, z, rx, ry, rz] in the robot's configured
 * external-reference frame. Translation is in metres; XYZ Euler angles and
 * the elbow value are in radians. The callback returns only after motion stops,
 * fails, or reaches its timeout.
 */

#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_driver/cartesian_movej.hpp"
#include "rokae_interfaces/srv/get_cartesian_state.hpp"
#include "rokae_interfaces/srv/move_l.hpp"
#include "rokae_interfaces/srv/move_l_relative.hpp"
#include "rokae/robot.h"

namespace {

using GetCartesianState = rokae_interfaces::srv::GetCartesianState;
using MoveL = rokae_interfaces::srv::MoveL;
using MoveLRelative = rokae_interfaces::srv::MoveLRelative;
constexpr auto kPollPeriod = std::chrono::milliseconds(100);
constexpr auto kMotionStartGracePeriod = std::chrono::milliseconds(100);

struct PoseError {
  double translationM{0.0};
  double rotationRad{0.0};
};

std::string sdkError(
    const std::string &operation, const std::error_code &ec) {
  return operation + " failed: code=" + std::to_string(ec.value()) +
         ", message=" + ec.message();
}

struct ArmContext {
  std::string side;
  std::string parameterPrefix;
  std::array<double, 6> expectedBaseFrame{};
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
  rclcpp::CallbackGroup::SharedPtr callbackGroup;
  rclcpp::Service<MoveL>::SharedPtr service;
  rclcpp::Service<MoveLRelative>::SharedPtr relativeService;
  rclcpp::Service<GetCartesianState>::SharedPtr stateService;
};

}  // namespace

class RokaeMoveLService : public rclcpp::Node {
 public:
  RokaeMoveLService()
      : Node("cartesian_motion", rokae_driver::interface_names::kNodeNamespace) {
    left_ = createArm(
        "left", "192.168.1.160", "192.168.1.10",
        {0.0, 0.105, 0.0, -1.570796327, 0.1308996939, 0.0},
        rokae_driver::interface_names::arm("move_l", "left"),
        rokae_driver::interface_names::arm("move_l_relative", "left"));
    right_ = createArm(
        "right", "192.168.2.160", "192.168.2.10",
        {0.0, -0.105, 0.0, 1.570796327, 0.1308996939, 0.0},
        rokae_driver::interface_names::arm("move_l", "right"),
        rokae_driver::interface_names::arm("move_l_relative", "right"));

    RCLCPP_WARN(
        get_logger(),
        "MoveL services are ready. This node does not power on either robot. "
        "Use a small first move, low speed, a clear workspace and an "
        "accessible E-stop.");
  }

 private:
  std::unique_ptr<ArmContext> createArm(
      const std::string &side, const std::string &defaultRobotIp,
      const std::string &defaultLocalIp,
      const std::array<double, 6> &defaultExpectedBaseFrame,
      const std::string &serviceName,
      const std::string &relativeServiceName) {
    const std::string prefix = side + "_arm";
    const auto robotIp = declare_parameter<std::string>(
        prefix + ".robot_ip", defaultRobotIp);
    const auto localIp = declare_parameter<std::string>(
        prefix + ".local_ip", defaultLocalIp);
    const auto expectedBaseFrame = declare_parameter<std::vector<double>>(
        prefix + ".expected_base_frame",
        std::vector<double>(
            defaultExpectedBaseFrame.begin(), defaultExpectedBaseFrame.end()));
    if (expectedBaseFrame.size() != defaultExpectedBaseFrame.size() ||
        !std::all_of(
            expectedBaseFrame.begin(), expectedBaseFrame.end(),
            [](double value) { return std::isfinite(value); })) {
      throw std::invalid_argument(
          prefix + ".expected_base_frame must contain 6 finite values");
    }
    declare_parameter<double>(prefix + ".timeout_s", 60.0);
    declare_parameter<double>(prefix + ".max_translation_delta_m", 0.10);
    declare_parameter<double>(prefix + ".max_rotation_delta_rad", 0.35);
    declare_parameter<double>(prefix + ".max_speed_mm_s", 250.0);
    declare_parameter<double>(prefix + ".max_zone_mm", 20.0);
    declare_parameter<double>(prefix + ".motion_start_timeout_s", 3.0);
    declare_parameter<double>(prefix + ".goal_position_tolerance_m", 0.002);
    declare_parameter<double>(
        prefix + ".goal_rotation_tolerance_rad", 0.02);

    auto arm = std::make_unique<ArmContext>();
    arm->side = side;
    arm->parameterPrefix = prefix;
    std::copy(
        expectedBaseFrame.begin(), expectedBaseFrame.end(),
        arm->expectedBaseFrame.begin());

    RCLCPP_INFO(
        get_logger(), "Using shared %s arm: robot=%s, local=%s",
        side.c_str(), robotIp.c_str(), localIp.c_str());
    arm->hardware = rokae_driver::sharedArm(side, robotIp, localIp);
    std::string baseFrameReason;
    if (!validateBaseFrame(*arm, baseFrameReason)) {
      RCLCPP_ERROR(get_logger(), "%s", baseFrameReason.c_str());
    }
    arm->callbackGroup = create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);

    ArmContext *const armPtr = arm.get();
    arm->service = create_service<MoveL>(
        serviceName,
        [this, armPtr](
            const std::shared_ptr<MoveL::Request> request,
            std::shared_ptr<MoveL::Response> response) {
          execute(*armPtr, *request, *response);
        },
        rmw_qos_profile_services_default,
        arm->callbackGroup);
    RCLCPP_INFO(get_logger(), "%s is ready", serviceName.c_str());

    arm->relativeService = create_service<MoveLRelative>(
        relativeServiceName,
        [this, armPtr](
            const std::shared_ptr<MoveLRelative::Request> request,
            std::shared_ptr<MoveLRelative::Response> response) {
          executeRelative(*armPtr, *request, *response);
        },
        rmw_qos_profile_services_default,
        arm->callbackGroup);
    RCLCPP_INFO(
        get_logger(), "%s is ready", relativeServiceName.c_str());

    const std::string stateServiceName =
        rokae_driver::interface_names::arm("get_cartesian_state", side);
    arm->stateService = create_service<GetCartesianState>(
        stateServiceName,
        [this, armPtr](
            const std::shared_ptr<GetCartesianState::Request>,
            std::shared_ptr<GetCartesianState::Response> response) {
          readCartesianState(*armPtr, *response);
        },
        rmw_qos_profile_services_default,
        arm->callbackGroup);
    RCLCPP_INFO(
        get_logger(), "%s is ready", stateServiceName.c_str());
    return arm;
  }

  void readCartesianState(
      ArmContext &arm, GetCartesianState::Response &response) {
    std::unique_lock<std::mutex> lock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!lock.owns_lock()) {
      fail(
          response,
          arm.side + " arm is occupied by another control command");
      return;
    }

    std::error_code ec;
    const auto current = arm.hardware->withRobot(
        [&ec](rokae::ArRobot &robot) {
          return robot.cartPosture(
              rokae::CoordinateType::endInRef, ec);
        });
    if (ec) {
      fail(response, sdkError("cartPosture(endInRef)", ec));
      return;
    }
    for (std::size_t index = 0; index < 3; ++index) {
      response.pose[index] = current.trans[index];
      response.pose[index + 3] = current.rpy[index];
    }
    response.success = true;
    response.message = "current Cartesian state";
  }

  void executeRelative(
      ArmContext &arm, const MoveLRelative::Request &request,
      MoveLRelative::Response &response) {
    std::unique_lock<std::mutex> lock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!lock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another control command");
      return;
    }

    std::string reason;
    if (!validateBaseFrame(arm, reason)) {
      fail(response, reason);
      return;
    }

    const bool finiteTranslation = std::all_of(
        request.translation.begin(), request.translation.end(),
        [](double value) { return std::isfinite(value); });
    const bool finiteOrientation = std::all_of(
        request.orientation_rpy.begin(), request.orientation_rpy.end(),
        [](double value) { return std::isfinite(value); });
    if (!finiteTranslation || !finiteOrientation) {
      fail(
          response,
          "relative translation and absolute orientation must be finite");
      return;
    }

    if (!validateMotionOptions(
            arm, request.speed_mm_s, request.zone_mm, reason)) {
      fail(response, reason);
      return;
    }

    try {
      std::error_code ec;
      const auto current = arm.hardware->withRobot(
          [&ec](rokae::ArRobot &robot) {
            return robot.cartPosture(
                rokae::CoordinateType::endInRef, ec);
          });
      if (ec) {
        fail(response, sdkError("cartPosture(endInRef)", ec));
        return;
      }

      // Keep the complete SDK CartesianPosition, including its elbow,
      // configuration and external-axis data. AR controllers may report
      // hasElbow=false even though this copied position is a valid MoveL seed.
      rokae::CartesianPosition target = current;
      for (std::size_t i = 0; i < 3; ++i) {
        target.trans[i] += request.translation[i];
        if (request.orientation_override[i]) {
          target.rpy[i] = request.orientation_rpy[i];
        }
      }

      if (!validateDelta(arm, current, target, reason)) {
        fail(response, reason);
        return;
      }
      executeTarget(
          arm, target, request.speed_mm_s, request.zone_mm, response);
    } catch (const std::exception &exception) {
      stopAndReset(arm);
      fail(
          response,
          std::string("relative MoveL exception: ") + exception.what());
    }
  }

  void execute(
      ArmContext &arm, const MoveL::Request &request,
      MoveL::Response &response) {
    std::unique_lock<std::mutex> lock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!lock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another control command");
      return;
    }
    const auto generation = arm.hardware->stopGeneration();
    rokae::Toolset originalTool;
    bool toolChanged = false;
    const auto restoreTool = [&]() {
      if (!toolChanged) return;
      for (int attempt = 0; attempt < 20; ++attempt) {
        const bool idle = arm.hardware->withRobot([](rokae::ArRobot &robot) {
          std::error_code ec;
          return robot.operationState(ec) == rokae::OperationState::idle && !ec;
        });
        if (idle) {
          arm.hardware->withRobot([&](rokae::ArRobot &robot) {
            std::error_code ec;
            robot.setToolset(originalTool, ec);
            if (ec) throw std::runtime_error(sdkError("restore toolset", ec));
          });
          toolChanged = false;
          return;
        }
        std::this_thread::sleep_for(kPollPeriod);
      }
      throw std::runtime_error("MoveL tool restore requires confirmed idle");
    };

    std::string reason;
    if (!validateBaseFrame(arm, reason)) {
      fail(response, reason);
      return;
    }
    if (!validateRequest(arm, request, reason)) {
      fail(response, reason);
      return;
    }

    try {
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        std::error_code ec;
        const auto state = robot.operationState(ec);
        if (ec) throw std::runtime_error(sdkError("operationState", ec));
        if (state != rokae::OperationState::idle) throw std::runtime_error("MoveL requires idle arm");
        originalTool = robot.toolset(ec);
        if (ec) throw std::runtime_error(sdkError("toolset", ec));
        const auto selected = rokae_driver::cartesian_movej::selectTool(originalTool, request.tcp_type);
        toolChanged = selected.end.trans != originalTool.end.trans || selected.end.rpy != originalTool.end.rpy;
        if (toolChanged) {
          robot.setToolset(selected, ec);
          if (ec) throw std::runtime_error(sdkError("select toolset", ec));
        }
      });
      std::error_code ec;
      const auto current = arm.hardware->withRobot(
          [&ec](rokae::ArRobot &robot) {
            return robot.cartPosture(
                rokae::CoordinateType::endInRef, ec);
          });
      if (ec) {
        throw std::runtime_error(sdkError("cartPosture(endInRef)", ec));
      }

      // Start with the complete controller position so the redundant-arm
      // configuration remains unchanged unless an elbow is explicitly used.
      if (request.check_current_pose)
        rokae_driver::cartesian_movej::checkObservation(current, request.expected_current_pose);
      rokae::CartesianPosition target = current;
      for (std::size_t index = 0; index < 3; ++index) {
        if (!request.use_current_pose)
          target.trans[index] = request.position[index] +
                                (request.position_is_relative ? current.trans[index] : 0.0);
        if (!request.use_current_pose && !request.use_current_orientation)
          target.rpy[index] = request.orientation_rpy[index];
      }
      if (request.use_elbow) {
        target.elbow = request.elbow;
        target.hasElbow = true;
      }
      if (!request.conf_data.empty()) {
        target.confData.assign(request.conf_data.begin(), request.conf_data.end());
      }
      if (!validateDelta(arm, current, target, reason)) {
        fail(response, reason);
        restoreTool();
        return;
      }
      if (generation != arm.hardware->stopGeneration()) throw std::runtime_error("stop interrupted MoveL planning");
      executeTarget(
          arm, target, request.speed_mm_s, request.zone_mm, response,
          !request.conf_data.empty(), &request, generation);
      if (!response.success) stopAndReset(arm);
      restoreTool();
    } catch (const std::exception &exception) {
      stopAndReset(arm);
      std::string error = std::string("MoveL exception: ") + exception.what();
      try { restoreTool(); }
      catch (const std::exception &restoreError) { error += "; " + std::string(restoreError.what()); }
      fail(response, error);
    }
  }

  template <typename ResponseT>
  void executeTarget(
      ArmContext &arm, const rokae::CartesianPosition &target,
      double speedMmS, double zoneMm, ResponseT &response,
      bool forceConfiguration = false, const MoveL::Request *request = nullptr,
      std::uint64_t generation = 0) {
    if (!request) generation = arm.hardware->stopGeneration();
    const double timeoutS = request && request->timeout_s > 0. ? request->timeout_s : get_parameter(
        arm.parameterPrefix + ".timeout_s").as_double();
    const double motionStartTimeoutS = std::min(timeoutS, get_parameter(
        arm.parameterPrefix + ".motion_start_timeout_s").as_double());
    const double positionToleranceM = request && request->position_tolerance_m > 0. ? request->position_tolerance_m : get_parameter(
        arm.parameterPrefix + ".goal_position_tolerance_m").as_double();
    const double rotationToleranceRad = request && request->orientation_tolerance_rad > 0. ? request->orientation_tolerance_rad : get_parameter(
        arm.parameterPrefix + ".goal_rotation_tolerance_rad").as_double();

    std::string reason;
    if (!validateMonitorOptions(
            timeoutS, motionStartTimeoutS, positionToleranceM,
            rotationToleranceRad, reason)) {
      fail(response, reason);
      return;
    }
    // Validate the selected-tool straight path and endpoint joint limits before
    // appending a command. Identical current-pose probes need no checkPath.
    arm.hardware->withRobot([&](rokae::ArRobot &robot) {
      std::error_code error;
      if (robot.powerState(error) != rokae::PowerState::on || error)
        throw std::runtime_error("MoveL requires powered-on robot; call initialize first");
      const auto current = robot.cartPosture(rokae::CoordinateType::endInRef, error);
      if (error) throw std::runtime_error(sdkError("cartPosture(path)", error));
      const auto joints = robot.jointPos(error);
      if (error) throw std::runtime_error(sdkError("jointPos(path)", error));
      const auto delta = calculatePoseError(current, target);
      std::vector<double> targetJoints(joints.begin(), joints.end());
      if (delta.translationM > 1e-9 || delta.rotationRad > 1e-9) {
        robot.setDefaultConfOpt(forceConfiguration, error);
        if (error) throw std::runtime_error(sdkError("setDefaultConfOpt(path)", error));
        targetJoints = robot.checkPath(current, std::vector<double>(joints.begin(), joints.end()), target, error);
        if (error) throw std::runtime_error(sdkError("checkPath(MoveL)", error));
      }
      std::array<double[2], 7> limits{};
      const bool enabled = robot.getSoftLimit(limits, error);
      if (error || !enabled || targetJoints.size() != 7) throw std::runtime_error("MoveL needs enabled valid soft limits and seven target joints");
      for (std::size_t i = 0; i < 7; ++i) {
        if (!std::isfinite(joints[i]) || !std::isfinite(targetJoints[i]) ||
            !std::isfinite(limits[i][0]) || !std::isfinite(limits[i][1]) || limits[i][0] >= limits[i][1] ||
            targetJoints[i] < limits[i][0] || targetJoints[i] > limits[i][1])
          throw std::runtime_error("MoveL target violates joint soft limits");
      }
    });
    const unsigned stableSamples = request ? request->stable_samples : 1;
    const auto pollPeriod = request ? std::chrono::milliseconds(request->poll_interval_ms) : kPollPeriod;

    std::error_code ec;
    arm.hardware->withRobot([&ec](rokae::ArRobot &robot) {
      robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
    });
    if (ec) {
      fail(response, sdkError("setMotionControlMode(NrtCommand)", ec));
      return;
    }
    arm.hardware->withRobot([&ec](rokae::ArRobot &robot) {
      robot.setOperateMode(rokae::OperateMode::automatic, ec);
    });
    if (ec) {
      fail(response, sdkError("setOperateMode(automatic)", ec));
      return;
    }
    arm.hardware->withRobot([&ec, forceConfiguration](rokae::ArRobot &robot) {
      robot.setDefaultConfOpt(forceConfiguration, ec);
    });
    if (ec) {
      fail(response, sdkError("setDefaultConfOpt", ec));
      return;
    }
    arm.hardware->withRobot(
        [&ec](rokae::ArRobot &robot) { robot.moveReset(ec); });
    if (ec) {
      fail(response, sdkError("moveReset", ec));
      return;
    }

    const rokae::MoveLCommand command(target, speedMmS, zoneMm);
    std::string commandId;
    arm.hardware->withRobot(
        [&command, &commandId, &ec](rokae::ArRobot &robot) {
          robot.moveAppend({command}, commandId, ec);
        });
    if (ec) {
      fail(response, sdkError("moveAppend(MoveL)", ec));
      return;
    }
    arm.hardware->withRobot(
        [&](rokae::ArRobot &robot) {
          if (generation != arm.hardware->stopGeneration()) throw std::runtime_error("stop interrupted MoveL");
          if (request && request->resource_valid_until_unix_s > 0. &&
              std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                  >= request->resource_valid_until_unix_s) throw std::runtime_error("collected ROS resource expired before MoveL");
          robot.moveStart(ec);
        });
    if (ec) {
      fail(response, sdkError("moveStart", ec));
      return;
    }

    const auto commandStart = std::chrono::steady_clock::now();
    const auto deadline = commandStart +
        std::chrono::duration<double>(timeoutS);
    const auto motionStartDeadline = commandStart +
        std::chrono::duration<double>(motionStartTimeoutS);
    bool motionObserved = false;
    unsigned stable = 0;

    // The controller can still report the previous idle state immediately
    // after moveStart(). The known-good SDK example waits before polling.
    std::this_thread::sleep_for(kMotionStartGracePeriod);

    const auto complete = [&](const PoseError &error) {
      response.success = true;
      response.message =
          "MoveL completed; command_id=" + commandId +
          "; position_error_m=" + std::to_string(error.translationM) +
          "; rotation_error_rad=" + std::to_string(error.rotationRad);
      RCLCPP_INFO(
          get_logger(),
          "%s arm MoveL completed: position_error=%.6f m, "
          "rotation_error=%.6f rad",
          arm.side.c_str(), error.translationM, error.rotationRad);
    };

    while (rclcpp::ok()) {
      if (generation != arm.hardware->stopGeneration()) {
        stopAndReset(arm);
        fail(response, "stop service interrupted MoveL");
        return;
      }
      const auto state = arm.hardware->withRobot(
          [&ec](rokae::ArRobot &robot) {
            return robot.operationState(ec);
          });
      if (ec) {
        stopAndReset(arm);
        fail(response, sdkError("operationState", ec));
        return;
      }
      if (state == rokae::OperationState::unknown) {
        stopAndReset(arm);
        fail(response, "robot operation state became unknown");
        return;
      }

      const auto now = std::chrono::steady_clock::now();
      if (state != rokae::OperationState::idle) {
        motionObserved = true;
        stable = 0;
      } else {
        PoseError goalError;
        if (!readGoalError(arm, target, goalError, reason)) {
          stopAndReset(arm);
          fail(response, reason);
          return;
        }
        const bool targetReached =
            goalError.translationM <= positionToleranceM &&
            goalError.rotationRad <= rotationToleranceRad;
        bool commandReached = targetReached;
        if (request) {
          const auto event = arm.hardware->withRobot([&](rokae::ArRobot &robot) {
            return robot.queryEventInfo(rokae::Event::moveExecution, ec);
          });
          if (ec) { stopAndReset(arm); fail(response, sdkError("moveExecution", ec)); return; }
          using namespace rokae::EventInfoKey::MoveExecution;
          const auto id = event.find(ID), reached = event.find(ReachTarget), error = event.find(Error);
          const auto *eventId = id == event.end() ? nullptr : std::any_cast<std::string>(&id->second);
          const auto *eventReached = reached == event.end() ? nullptr : std::any_cast<bool>(&reached->second);
          const auto *eventError = error == event.end() ? nullptr : std::any_cast<std::error_code>(&error->second);
          if (eventId && *eventId == commandId && eventError && *eventError) {
            stopAndReset(arm); fail(response, sdkError("moveExecution", *eventError)); return;
          }
          commandReached = commandReached && eventId && *eventId == commandId && eventReached && *eventReached;
        }
        if (commandReached) {
          if (++stable >= stableSamples) {
            complete(goalError);
            return;
          }
        } else {
          stable = 0;
        }
        if (!targetReached && motionObserved) {
          stopAndReset(arm);
          fail(
              response,
              "robot became idle before reaching the MoveL target; "
              "position_error_m=" +
                  std::to_string(goalError.translationM) +
                  ", rotation_error_rad=" +
                  std::to_string(goalError.rotationRad));
          return;
        }
        if (!targetReached && now >= motionStartDeadline) {
          stopAndReset(arm);
          fail(
              response,
              "MoveL did not enter a running state within " +
                  std::to_string(motionStartTimeoutS) +
                  " s; position_error_m=" +
                  std::to_string(goalError.translationM) +
                  ", rotation_error_rad=" +
                  std::to_string(goalError.rotationRad));
          return;
        }
      }

      if (now >= deadline) {
        stopAndReset(arm);
        fail(
            response,
            "MoveL timed out after " + std::to_string(timeoutS) +
                " s; motion stopped and reset");
        return;
      }
      std::this_thread::sleep_for(pollPeriod);
    }
    stopAndReset(arm);
    fail(response, "ROS shutdown stopped and reset MoveL");
  }

  bool validateRequest(
      const ArmContext &arm, const MoveL::Request &request,
      std::string &reason) const {
    try { rokae_driver::cartesian_movej::selectTool(rokae::Toolset{}, request.tcp_type); }
    catch (const std::exception &error) { reason = error.what(); return false; }
    for (const double value : {request.timeout_s, request.position_tolerance_m,
                              request.orientation_tolerance_rad, request.resource_valid_until_unix_s}) {
      if (!std::isfinite(value) || value < 0.) { reason = "invalid MoveL monitor/deadline value"; return false; }
    }
    if (request.stable_samples < 1 || request.stable_samples > 100 ||
        request.poll_interval_ms < 10 || request.poll_interval_ms > 1000) {
      reason = "stable_samples must be [1,100]; poll_interval_ms must be [10,1000]";
      return false;
    }
    const bool finitePosition = std::all_of(
        request.position.begin(), request.position.end(),
        [](double value) { return std::isfinite(value); });
    const bool finiteOrientation = std::all_of(
        request.orientation_rpy.begin(), request.orientation_rpy.end(),
        [](double value) { return std::isfinite(value); });
    if (!finitePosition || !finiteOrientation ||
        (request.use_elbow && !std::isfinite(request.elbow))) {
      reason = "position, orientation and enabled elbow must be finite";
      return false;
    }
    if (!request.conf_data.empty() && request.conf_data.size() != 8) {
      reason = "conf_data must be empty or contain 8 integers";
      return false;
    }
    return validateMotionOptions(
        arm, request.speed_mm_s, request.zone_mm, reason);
  }

  bool validateMotionOptions(
      const ArmContext &arm, double speedMmS, double zoneMm,
      std::string &reason) const {
    if (!std::isfinite(speedMmS) || !std::isfinite(zoneMm)) {
      reason = "speed_mm_s and zone_mm must be finite";
      return false;
    }
    const double maxSpeed = get_parameter(
        arm.parameterPrefix + ".max_speed_mm_s").as_double();
    const double maxZone = get_parameter(
        arm.parameterPrefix + ".max_zone_mm").as_double();
    if (speedMmS <= 0.0 || speedMmS > maxSpeed) {
      reason = "speed_mm_s must be in (0, " +
          std::to_string(maxSpeed) + "]";
      return false;
    }
    if (zoneMm < 0.0 || zoneMm > maxZone) {
      reason = "zone_mm must be in [0, " + std::to_string(maxZone) + "]";
      return false;
    }
    return true;
  }

  bool validateMonitorOptions(
      double timeoutS, double motionStartTimeoutS,
      double positionToleranceM, double rotationToleranceRad,
      std::string &reason) const {
    const std::array<double, 4> values{
        timeoutS, motionStartTimeoutS, positionToleranceM,
        rotationToleranceRad};
    if (!std::all_of(
            values.begin(), values.end(),
            [](double value) { return std::isfinite(value); })) {
      reason = "MoveL timeout and goal tolerances must be finite";
      return false;
    }
    if (timeoutS < 1.0 || timeoutS > 300.0) {
      reason = "timeout_s must be in [1, 300]";
      return false;
    }
    if (motionStartTimeoutS <= 0.0 ||
        motionStartTimeoutS > std::min(timeoutS, 10.0)) {
      reason =
          "motion_start_timeout_s must be in (0, min(timeout_s, 10)]";
      return false;
    }
    if (positionToleranceM <= 0.0 || positionToleranceM > 0.05) {
      reason = "goal_position_tolerance_m must be in (0, 0.05]";
      return false;
    }
    if (rotationToleranceRad <= 0.0 || rotationToleranceRad > 0.20) {
      reason = "goal_rotation_tolerance_rad must be in (0, 0.20]";
      return false;
    }
    return true;
  }

  PoseError calculatePoseError(
      const rokae::CartesianPosition &current,
      const rokae::CartesianPosition &target) const {
    double translationSquared = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
      const double translationDelta =
          target.trans[i] - current.trans[i];
      translationSquared += translationDelta * translationDelta;
    }
    const auto a = rokae::Utils::eulerToQuaternion(current.rpy);
    const auto b = rokae::Utils::eulerToQuaternion(target.rpy);
    double dot = 0.;
    for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return {std::sqrt(translationSquared), 2. * std::acos(std::clamp(std::abs(dot), 0., 1.))};
  }

  bool readGoalError(
      ArmContext &arm, const rokae::CartesianPosition &target,
      PoseError &error, std::string &reason) {
    std::error_code ec;
    const auto current = arm.hardware->withRobot(
        [&ec](rokae::ArRobot &robot) {
          return robot.cartPosture(
              rokae::CoordinateType::endInRef, ec);
        });
    if (ec) {
      reason = sdkError("cartPosture(endInRef)", ec);
      return false;
    }
    error = calculatePoseError(current, target);
    return true;
  }

  void stopAndReset(ArmContext &arm) {
    std::error_code ec;
    arm.hardware->withRobot(
        [&ec](rokae::ArRobot &robot) { robot.stop(ec); });
    if (ec) {
      RCLCPP_ERROR(
          get_logger(), "%s arm stop failed: %s",
          arm.side.c_str(), sdkError("stop", ec).c_str());
    }
    ec.clear();
    arm.hardware->withRobot(
        [&ec](rokae::ArRobot &robot) { robot.moveReset(ec); });
    if (ec) {
      RCLCPP_ERROR(
          get_logger(), "%s arm moveReset failed: %s",
          arm.side.c_str(), sdkError("moveReset", ec).c_str());
    }
  }

  bool validateDelta(
      const ArmContext &arm, const rokae::CartesianPosition &current,
      const rokae::CartesianPosition &target, std::string &reason) const {
    const PoseError error = calculatePoseError(current, target);
    const double maxTranslation = get_parameter(
        arm.parameterPrefix + ".max_translation_delta_m").as_double();
    const double maxRotation = get_parameter(
        arm.parameterPrefix + ".max_rotation_delta_rad").as_double();
    if (error.translationM > maxTranslation) {
      reason =
          "translation delta " + std::to_string(error.translationM) +
          " m exceeds max_translation_delta_m=" +
          std::to_string(maxTranslation);
      return false;
    }
    if (error.rotationRad > maxRotation) {
      reason =
          "orientation delta " + std::to_string(error.rotationRad) +
          " rad exceeds max_rotation_delta_rad=" +
          std::to_string(maxRotation);
      return false;
    }
    return true;
  }

  bool validateBaseFrame(
      const ArmContext &arm, std::string &reason) const {
    std::error_code ec;
    const auto actual = arm.hardware->withRobot(
        [&ec](rokae::ArRobot &robot) { return robot.baseFrame(ec); });
    if (ec) {
      reason = arm.side + " arm " + sdkError("baseFrame", ec);
      return false;
    }
    constexpr double tolerance = 1e-6;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (!std::isfinite(actual[index]) ||
          std::abs(actual[index] - arm.expectedBaseFrame[index]) > tolerance) {
        std::ostringstream message;
        message << arm.side
                << " arm baseFrame does not match set_rotation.cpp; actual=[";
        for (std::size_t i = 0; i < actual.size(); ++i) {
          if (i != 0) message << ", ";
          message << actual[i];
        }
        message << "], expected=[";
        for (std::size_t i = 0; i < arm.expectedBaseFrame.size(); ++i) {
          if (i != 0) message << ", ";
          message << arm.expectedBaseFrame[i];
        }
        message << "]; run set_rotation and restart the robot controller";
        reason = message.str();
        return false;
      }
    }
    return true;
  }

  template <typename ResponseT>
  void fail(ResponseT &response, const std::string &message) {
    response.success = false;
    response.message = message;
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
  }

  std::unique_ptr<ArmContext> left_;
  std::unique_ptr<ArmContext> right_;
};

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeMoveLService() {
  return std::make_shared<RokaeMoveLService>();
}

}  // namespace rokae_driver

#ifndef ROKAE_UNIFIED_DRIVER
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<RokaeMoveLService>();
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception &exception) {
    RCLCPP_FATAL(
        rclcpp::get_logger("rokae_movel_service"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
#endif
