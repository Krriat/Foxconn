/**
 * @file ros_cartesian_teach_service.cpp
 * @brief Safe Cartesian hand-guiding services based on xCoreSDK enableDrag.
 *
 * Start services block until stopped, rejected by a software guard, or the
 * fixed 300-second session limit is reached. Stop services allow a client to
 * end a session without shutting down the unified driver.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_interfaces/srv/cartesian_teach.hpp"
#include "rokae/robot.h"
#include "rokae/utility.h"
#include "std_srvs/srv/trigger.hpp"

namespace {

using CartesianTeach = rokae_interfaces::srv::CartesianTeach;
using Trigger = std_srvs::srv::Trigger;
using namespace std::chrono_literals;

constexpr auto kSessionDuration = std::chrono::seconds(300);
constexpr auto kModeSwitchDelay = std::chrono::seconds(2);
constexpr auto kMonitorPeriod = std::chrono::milliseconds(50);
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxToolMassKg = 5.0;
constexpr double kMaxCogDistanceM = 0.60;
constexpr double kMaxTranslationM = 0.50;
constexpr double kMaxRotationRad = 90.0 * kPi / 180.0;
constexpr double kMaxLinearSpeedMps = 0.20;
constexpr double kMaxJointSpeedRadS = 1.0;
constexpr int kSpeedViolationSamples = 2;

struct MonitorResult {
  bool safe{false};
  std::string reason;
};

struct ArmContext {
  std::string side;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
  std::atomic<bool> active{false};
  std::atomic<bool> stopRequested{false};
  rclcpp::CallbackGroup::SharedPtr startGroup;
  rclcpp::Service<CartesianTeach>::SharedPtr startService;
};

std::string sdkError(
    const std::string &operation, const std::error_code &error) {
  return operation + " failed: code=" + std::to_string(error.value()) +
         ", message=" + error.message();
}

template <std::size_t Size>
bool allFinite(const std::array<double, Size> &values) {
  return std::all_of(
      values.begin(), values.end(),
      [](double value) { return std::isfinite(value); });
}

bool validateLoad(const rokae::Load &load, std::string &reason) {
  if (!std::isfinite(load.mass) || load.mass <= 0.0 ||
      load.mass > kMaxToolMassKg) {
    reason = "configured tool mass must be in (0, 5] kg";
    return false;
  }
  if (!allFinite(load.cog)) {
    reason = "configured tool center of gravity contains a non-finite value";
    return false;
  }
  const double distance = std::sqrt(
      load.cog[0] * load.cog[0] + load.cog[1] * load.cog[1] +
      load.cog[2] * load.cog[2]);
  if (distance > kMaxCogDistanceM) {
    reason = "configured tool center of gravity exceeds 0.60 m";
    return false;
  }
  return true;
}

Eigen::Matrix3d rotationFromPose(const std::array<double, 6> &pose) {
  std::array<double, 16> transform{};
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  rokae::Utils::postureToTransArray(pose, transform);
  rokae::Utils::arrayToTransMatrix(transform, rotation, translation);
  return rotation;
}

}  // namespace

class RokaeCartesianTeachService : public rclcpp::Node {
 public:
  RokaeCartesianTeachService()
      : Node("cartesian_teach", rokae_driver::interface_names::kNodeNamespace) {
    left_ = createArm(
        "left", "192.168.1.160", "192.168.1.10");
    right_ = createArm(
        "right", "192.168.2.160", "192.168.2.10");
    stopGroup_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    stopService_ = create_service<Trigger>(
        "/robot_02/upperlimb/cartesian_teach/stop",
        [this](
            const std::shared_ptr<Trigger::Request>,
            std::shared_ptr<Trigger::Response> response) {
          stopTeach(*response);
        },
        rmw_qos_profile_services_default, stopGroup_);
    RCLCPP_WARN(
        get_logger(),
        "Cartesian teach services are ready for left and right arms. "
        "Each request may switch the selected arm to manual drag mode and "
        "power it off; fixed session limit is 300 seconds.");
  }

 private:
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
    arm->startGroup = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    ArmContext *const armPtr = arm.get();
    const auto startName = rokae_driver::interface_names::arm(
        "cartesian_teach", side);
    arm->startService = create_service<CartesianTeach>(
        startName,
        [this, armPtr](
            const std::shared_ptr<CartesianTeach::Request> request,
            std::shared_ptr<CartesianTeach::Response> response) {
          startTeach(*armPtr, *request, *response);
        },
        rmw_qos_profile_services_default, arm->startGroup);

    RCLCPP_INFO(get_logger(), "%s is ready", startName.c_str());
    return arm;
  }

  void stopTeach(Trigger::Response &response) {
    std::string stoppedSides;
    for (ArmContext *arm : {left_.get(), right_.get()}) {
      if (!arm->active.load(std::memory_order_acquire)) continue;
      arm->stopRequested.store(true, std::memory_order_release);
      if (!stoppedSides.empty()) stoppedSides += " and ";
      stoppedSides += arm->side;
    }
    if (stoppedSides.empty()) {
      response.success = false;
      response.message = "no Cartesian teach session is active";
      return;
    }
    response.success = true;
    response.message =
        "Cartesian teach stop requested for " + stoppedSides +
        "; disabling drag";
  }

  void startTeach(
      ArmContext &arm, const CartesianTeach::Request &request,
      CartesianTeach::Response &response) {
    std::unique_lock<std::mutex> commandLock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!commandLock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another command");
      return;
    }

    bool expected = false;
    if (!arm.active.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
      fail(response, arm.side + " Cartesian teach session is already active");
      return;
    }
    arm.stopRequested.store(false, std::memory_order_release);
    bool dragEnabled = false;

    const auto cleanup = [&]() {
      if (dragEnabled) {
        std::error_code error;
        arm.hardware->withRobot(
            [&error](rokae::ArRobot &robot) { robot.disableDrag(error); });
        if (error) {
          RCLCPP_ERROR(
              get_logger(), "%s", sdkError("disableDrag", error).c_str());
        }
      }
      arm.stopRequested.store(false, std::memory_order_release);
      arm.active.store(false, std::memory_order_release);
    };

    try {
      std::error_code error;
      const auto operationState = arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) {
            return robot.operationState(error);
          });
      if (error) {
        cleanup();
        fail(response, sdkError("operationState", error));
        return;
      }
      if (operationState != rokae::OperationState::idle) {
        cleanup();
        fail(
            response,
            "Cartesian teach requires an idle robot; current operation state "
            "is not idle");
        return;
      }

      auto toolset = arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) { return robot.toolset(error); });
      if (error) {
        cleanup();
        fail(response, sdkError("toolset", error));
        return;
      }
      std::string reason;
      if (request.set_tool_load) {
        toolset.load.mass = request.tool_mass_kg;
        toolset.load.cog = request.tool_cog_m;
      }
      if (!validateLoad(toolset.load, reason)) {
        cleanup();
        fail(
            response,
            request.set_tool_load
                ? "requested " + reason
                : reason +
                      "; identify and store the real tool load first, or set "
                      "set_tool_load=true with measured values");
        return;
      }

      const auto startPose = arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) {
            return robot.posture(rokae::CoordinateType::endInRef, error);
          });
      if (error || !allFinite(startPose)) {
        cleanup();
        fail(
            response,
            error ? sdkError("read start TCP pose", error)
                  : "start TCP pose contains a non-finite value");
        return;
      }

      arm.hardware->withRobot([&error](rokae::ArRobot &robot) {
        robot.setMotionControlMode(
            rokae::MotionControlMode::NrtCommand, error);
      });
      if (error) {
        cleanup();
        fail(response, sdkError("setMotionControlMode(NrtCommand)", error));
        return;
      }
      arm.hardware->withRobot([&error](rokae::ArRobot &robot) {
        robot.setOperateMode(rokae::OperateMode::manual, error);
      });
      if (error) {
        cleanup();
        fail(response, sdkError("setOperateMode(manual)", error));
        return;
      }
      arm.hardware->withRobot([&error](rokae::ArRobot &robot) {
        robot.setPowerState(false, error);
      });
      if (error) {
        cleanup();
        fail(response, sdkError("setPowerState(false)", error));
        return;
      }

      if (request.set_tool_load) {
        error.clear();
        arm.hardware->withRobot(
            [&toolset, &error](rokae::ArRobot &robot) {
              robot.setToolset(toolset, error);
            });
        if (error) {
          cleanup();
          fail(response, sdkError("setToolset(tool load)", error));
          return;
        }

        error.clear();
        const auto storedToolset = arm.hardware->withRobot(
            [&error](rokae::ArRobot &robot) {
              return robot.toolset(error);
            });
        if (error) {
          cleanup();
          fail(response, sdkError("verify stored tool load", error));
          return;
        }
        if (!validateLoad(storedToolset.load, reason)) {
          cleanup();
          fail(response, "stored " + reason);
          return;
        }
        constexpr double kLoadVerificationTolerance = 1e-6;
        bool loadMatches =
            std::abs(storedToolset.load.mass - request.tool_mass_kg) <=
            kLoadVerificationTolerance;
        for (std::size_t index = 0; index < request.tool_cog_m.size();
             ++index) {
          loadMatches = loadMatches &&
              std::abs(
                  storedToolset.load.cog[index] - request.tool_cog_m[index]) <=
                  kLoadVerificationTolerance;
        }
        if (!loadMatches) {
          cleanup();
          fail(response, "controller did not retain the requested tool load");
          return;
        }
        RCLCPP_WARN(
            get_logger(),
            "%s controller tool load updated: mass=%.6f kg, "
            "cog=[%.6f, %.6f, %.6f] m",
            arm.side.c_str(), storedToolset.load.mass,
            storedToolset.load.cog[0], storedToolset.load.cog[1],
            storedToolset.load.cog[2]);
      }

      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        robot.enableDrag(
            rokae::DragParameter::cartesianSpace,
            rokae::DragParameter::freely, error,
            request.enable_drag_button);
      });
      if (error) {
        cleanup();
        fail(response, sdkError("enableDrag(cartesianSpace)", error));
        return;
      }
      dragEnabled = true;

      RCLCPP_WARN(
          get_logger(),
          "%s Cartesian teach enabled for at most 300 s; direct_drag=%s. "
          "Keep the E-stop ready and move slowly.",
          arm.side.c_str(), request.enable_drag_button ? "true" : "false");

      const auto switchDeadline = std::chrono::steady_clock::now() +
          kModeSwitchDelay;
      while (rclcpp::ok() &&
             !arm.stopRequested.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < switchDeadline) {
        std::this_thread::sleep_for(kMonitorPeriod);
      }

      MonitorResult result;
      if (!rclcpp::ok()) {
        result = {false, "ROS shutdown requested"};
      } else if (arm.stopRequested.load(std::memory_order_acquire)) {
        result = {true, "stop service requested"};
      } else {
        result = monitor(arm, startPose);
      }

      cleanup();
      response.success = result.safe;
      response.message = arm.side + " Cartesian teach ended: " + result.reason +
          (request.set_tool_load ? "; tool load updated" : "") +
          "; drag disabled; robot remains in manual mode and is not powered on";
      if (response.success) {
        RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "%s", response.message.c_str());
      }
    } catch (const std::exception &exception) {
      cleanup();
      fail(
          response,
          std::string("Cartesian teach exception: ") + exception.what());
    }
  }

  MonitorResult monitor(
      ArmContext &arm, const std::array<double, 6> &startPose) {
    std::error_code error;
    auto previousPose = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) {
          return robot.posture(rokae::CoordinateType::endInRef, error);
        });
    if (error) return {false, sdkError("read initial TCP pose", error)};
    auto previousJoints = arm.hardware->withRobot(
        [&error](rokae::ArRobot &robot) { return robot.jointPos(error); });
    if (error) return {false, sdkError("read initial joints", error)};
    if (!allFinite(previousPose) || !allFinite(previousJoints)) {
      return {false, "initial feedback contains a non-finite value"};
    }

    const Eigen::Vector3d startPosition(
        startPose[0], startPose[1], startPose[2]);
    const Eigen::Matrix3d startRotation = rotationFromPose(startPose);
    auto previousTime = std::chrono::steady_clock::now();
    const auto deadline = previousTime + kSessionDuration;
    int speedViolationCount = 0;

    while (rclcpp::ok() &&
           !arm.stopRequested.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(kMonitorPeriod);
      const auto now = std::chrono::steady_clock::now();

      error.clear();
      const auto pose = arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) {
            return robot.posture(rokae::CoordinateType::endInRef, error);
          });
      if (error) return {false, sdkError("read TCP pose", error)};
      error.clear();
      const auto joints = arm.hardware->withRobot(
          [&error](rokae::ArRobot &robot) { return robot.jointPos(error); });
      if (error) return {false, sdkError("read joints", error)};
      if (!allFinite(pose) || !allFinite(joints)) {
        return {false, "feedback contains a non-finite value"};
      }

      const Eigen::Vector3d position(pose[0], pose[1], pose[2]);
      if ((position - startPosition).norm() > kMaxTranslationM) {
        return {false, "TCP moved more than 0.50 m from the start pose"};
      }
      const Eigen::Matrix3d relativeRotation =
          rotationFromPose(pose) * startRotation.transpose();
      if (std::abs(Eigen::AngleAxisd(relativeRotation).angle()) >
          kMaxRotationRad) {
        return {false, "TCP rotated more than 90 degrees from the start pose"};
      }

      const double dt =
          std::chrono::duration<double>(now - previousTime).count();
      const Eigen::Vector3d previousPosition(
          previousPose[0], previousPose[1], previousPose[2]);
      const double linearSpeed = (position - previousPosition).norm() / dt;
      double fastestJointSpeed = 0.0;
      std::size_t fastestJoint = 0;
      for (std::size_t index = 0; index < joints.size(); ++index) {
        const double speed =
            std::abs(joints[index] - previousJoints[index]) / dt;
        if (speed > fastestJointSpeed) {
          fastestJointSpeed = speed;
          fastestJoint = index;
        }
      }

      if (linearSpeed > kMaxLinearSpeedMps ||
          fastestJointSpeed > kMaxJointSpeedRadS) {
        ++speedViolationCount;
      } else {
        speedViolationCount = 0;
      }
      if (speedViolationCount >= kSpeedViolationSamples) {
        if (linearSpeed > kMaxLinearSpeedMps) {
          return {false, "TCP linear speed exceeded 0.20 m/s"};
        }
        return {
            false,
            "J" + std::to_string(fastestJoint + 1) +
                " speed exceeded 1.0 rad/s"};
      }

      previousPose = pose;
      previousJoints = joints;
      previousTime = now;
    }

    if (!rclcpp::ok()) return {false, "ROS shutdown requested"};
    if (arm.stopRequested.load(std::memory_order_acquire)) {
      return {true, "stop service requested"};
    }
    return {true, "fixed 300-second duration reached"};
  }

  static void fail(
      CartesianTeach::Response &response, const std::string &message) {
    response.success = false;
    response.message = message;
  }

  std::unique_ptr<ArmContext> left_;
  std::unique_ptr<ArmContext> right_;
  rclcpp::CallbackGroup::SharedPtr stopGroup_;
  rclcpp::Service<Trigger>::SharedPtr stopService_;
};

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeCartesianTeachService() {
  return std::make_shared<RokaeCartesianTeachService>();
}

}  // namespace rokae_driver

#ifndef ROKAE_UNIFIED_DRIVER
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int exitCode = 0;
  try {
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<RokaeCartesianTeachService>();
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception &exception) {
    std::cerr << "Cartesian teach service exception: " << exception.what()
              << std::endl;
    exitCode = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return exitCode;
}
#endif
