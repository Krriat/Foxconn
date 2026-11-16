/**
 * @file ros_robot_initializer_service.cpp
 * @brief Initialize and power on both Rokae AR arms without moving them.
 *
 * Service:
 *   /robot_02/upperlimb/initialize/{left_arm,right_arm,dual_arm}
 *
 * Each request follows op.cpp exactly:
 *   connect -> NrtCommand -> automatic -> power on -> verify power state
 *
 * The request reuses the unified driver's one shared SDK session per arm.
 * No motion command is constructed or sent.
 */

#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae/robot.h"
#include "std_srvs/srv/trigger.hpp"

namespace {

using Trigger = std_srvs::srv::Trigger;

struct ArmConfig {
  std::string side;
  std::string robotIp;
  std::string localIp;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
};

std::string sdkError(
    const std::string &operation, const std::error_code &error) {
  return operation + " failed: code=" + std::to_string(error.value()) +
         ", message=" + error.message();
}

const char *powerStateName(rokae::PowerState state) {
  switch (state) {
    case rokae::PowerState::on:
      return "on";
    case rokae::PowerState::off:
      return "off";
    case rokae::PowerState::estop:
      return "emergency stop";
    case rokae::PowerState::gstop:
      return "safety gate stop";
    default:
      return "unknown";
  }
}

}  // namespace

class RokaeRobotInitializerService : public rclcpp::Node {
 public:
  RokaeRobotInitializerService()
      : Node("initializer", rokae_driver::interface_names::kNodeNamespace),
        left_(loadArm(
            "left", "192.168.1.160", "192.168.1.10")),
        right_(loadArm(
            "right", "192.168.2.160", "192.168.2.10")) {
    leftInitializeService_ = create_service<Trigger>(
        "/robot_02/upperlimb/initialize/left_arm",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          initializeSingle(left_, *response);
        });
    rightInitializeService_ = create_service<Trigger>(
        "/robot_02/upperlimb/initialize/right_arm",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          initializeSingle(right_, *response);
        });
    dualInitializeService_ = create_service<Trigger>(
        rokae_driver::interface_names::kInitializeDualArm,
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          initializeBoth(*response);
        });
    leftPowerService_ = create_service<Trigger>(
        "/robot_02/upperlimb/power_on/left_arm",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          powerOn(left_, *response);
        });
    rightPowerService_ = create_service<Trigger>(
        "/robot_02/upperlimb/power_on/right_arm",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          powerOn(right_, *response);
        });
    stopGroup_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    leftStopService_ = create_service<Trigger>(
        "/robot_02/upperlimb/stop/left_arm",
        [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
          stopArm(left_, *response);
        }, rmw_qos_profile_services_default, stopGroup_);
    rightStopService_ = create_service<Trigger>(
        "/robot_02/upperlimb/stop/right_arm",
        [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
          stopArm(right_, *response);
        }, rmw_qos_profile_services_default, stopGroup_);
    RCLCPP_WARN(
        get_logger(),
        "%s is ready. A request switches both arms to NRT automatic mode "
        "and powers them on; it sends no motion command.",
        rokae_driver::interface_names::kInitializeDualArm);
    RCLCPP_WARN(get_logger(),
        "Initialization services ready: /robot_02/upperlimb/initialize/"
        "{left_arm,right_arm,dual_arm}; requests power on selected arms");
  }

 private:
  void stopArm(const ArmConfig &arm, Trigger::Response &response) {
    arm.hardware->requestStop();
    try {
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        std::error_code ec;
        robot.stop(ec);
        if (ec) throw std::runtime_error("stop failed: " + ec.message());
        robot.moveReset(ec);
        if (ec) throw std::runtime_error("moveReset failed: " + ec.message());
      });
      // Do not acquire commandMutex: the active callback must observe the
      // generation change and finish its own cleanup/tool restoration.
      response.success = true;
      response.message = arm.side + " stop and queue reset acknowledged";
    } catch (const std::exception &error) {
      response.success = false;
      response.message = error.what();
    }
  }
  rclcpp::CallbackGroup::SharedPtr stopGroup_;
  rclcpp::Service<Trigger>::SharedPtr leftStopService_, rightStopService_;
  void initializeSingle(const ArmConfig &arm, Trigger::Response &response) {
    std::unique_lock<std::mutex> control(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!control.owns_lock()) {
      response.success = false;
      response.message = arm.side + " initialization rejected: arm is occupied";
      return;
    }
    std::string result;
    response.success = initializeArm(arm, result);
    response.message = arm.side + ": " + result;
    RCLCPP_INFO(get_logger(), "%s", response.message.c_str());
  }

  void powerOn(const ArmConfig &arm, Trigger::Response &response) {
    std::unique_lock<std::mutex> control(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!control.owns_lock()) {
      response.success = false;
      response.message = arm.side + " arm is occupied; power-on rejected";
      return;
    }
    try {
      response.success = arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        std::error_code error;
        const auto power = robot.powerState(error);
        if (error) {
          response.message = sdkError("powerState", error);
          return false;
        }
        if (power == rokae::PowerState::on) {
          response.message = "already powered on; no mode change";
          return true;
        }
        if (power != rokae::PowerState::off) {
          response.message = std::string("power-on rejected: ") + powerStateName(power);
          return false;
        }
        const auto state = robot.operationState(error);
        if (error || state != rokae::OperationState::idle) {
          response.message = error ? sdkError("operationState", error)
                                   : "robot must be idle before power-on";
          return false;
        }
        robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, error);
        if (error) { response.message = sdkError("NrtCommand", error); return false; }
        robot.setOperateMode(rokae::OperateMode::automatic, error);
        if (error) { response.message = sdkError("automatic", error); return false; }
        robot.setPowerState(true, error);
        if (error) { response.message = sdkError("power-on", error); return false; }
        const auto finalPower = robot.powerState(error);
        response.message = error ? sdkError("verify power", error)
            : std::string("power state: ") + powerStateName(finalPower);
        return !error && finalPower == rokae::PowerState::on;
      });
    } catch (const std::exception &error) {
      response.success = false;
      response.message = error.what();
    }
    RCLCPP_WARN(get_logger(), "%s power-on request: %s", arm.side.c_str(),
                response.message.c_str());
  }

  ArmConfig loadArm(
      const std::string &side, const std::string &defaultRobotIp,
      const std::string &defaultLocalIp) {
    const std::string prefix = side + "_arm";
    const auto robotIp = declare_parameter<std::string>(
        prefix + ".robot_ip", defaultRobotIp);
    const auto localIp = declare_parameter<std::string>(
        prefix + ".local_ip", defaultLocalIp);
    return {
        side, robotIp, localIp,
        rokae_driver::sharedArm(side, robotIp, localIp)};
  }

  bool initializeArm(
      const ArmConfig &config, std::string &result) {
    try {
      RCLCPP_INFO(
          get_logger(), "Initializing %s arm: robot=%s, local=%s",
          config.side.c_str(), config.robotIp.c_str(),
          config.localIp.c_str());
      return config.hardware->withRobot(
          [&result](rokae::ArRobot &robot) {
            std::error_code error;
            // Service calls must not interrupt an independently started move
            // or attempt to override an emergency/safety-gate stop.
            const auto power = robot.powerState(error);
            if (error) {
              result = sdkError("powerState before initialization", error);
              return false;
            }
            if (power != rokae::PowerState::on && power != rokae::PowerState::off) {
              result = std::string("initialization rejected: ") + powerStateName(power);
              return false;
            }
            const auto state = robot.operationState(error);
            if (error || state != rokae::OperationState::idle) {
              result = error ? sdkError("operationState", error)
                             : "initialization requires idle robot";
              return false;
            }
            const auto info = robot.robotInfo(error);
            if (error) {
              result = sdkError("robotInfo", error);
              return false;
            }

            robot.setMotionControlMode(
                rokae::MotionControlMode::NrtCommand, error);
            if (error) {
              result = sdkError("setMotionControlMode(NrtCommand)", error);
              return false;
            }

            robot.setOperateMode(rokae::OperateMode::automatic, error);
            if (error) {
              result = sdkError("setOperateMode(automatic)", error);
              return false;
            }

            robot.setPowerState(true, error);
            if (error) {
              result = sdkError("setPowerState(true)", error);
              return false;
            }

            const auto powerState = robot.powerState(error);
            if (error) {
              result = sdkError("powerState", error);
              return false;
            }
            if (powerState != rokae::PowerState::on) {
              result =
                  "power-on verification failed; final state=" +
                  std::string(powerStateName(powerState));
              return false;
            }

            result =
                "initialized and powered on; model=" + info.type +
                ", controller=" + info.version;
            return true;
          });
    } catch (const std::exception &exception) {
      result = std::string("connection/initialization exception: ") +
          exception.what();
      return false;
    }
  }

  void initializeBoth(Trigger::Response &response) {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      response.success = false;
      response.message = "robot initialization is already in progress";
      return;
    }

    RCLCPP_WARN(
        get_logger(),
        "Robot initialization requested: switching both arms to automatic "
        "mode and powering them on");
    std::unique_lock<std::mutex> leftControl(
        left_.hardware->commandMutex(), std::defer_lock);
    std::unique_lock<std::mutex> rightControl(
        right_.hardware->commandMutex(), std::defer_lock);
    if (std::try_lock(leftControl, rightControl) != -1) {
      response.success = false;
      response.message =
          "robot initialization rejected: one or both arms are occupied";
      return;
    }
    std::string leftResult;
    std::string rightResult;
    const bool leftSuccess = initializeArm(left_, leftResult);
    const bool rightSuccess = initializeArm(right_, rightResult);

    response.success = leftSuccess && rightSuccess;
    response.message =
        "left: " + leftResult + "; right: " + rightResult;
    if (response.success) {
      RCLCPP_INFO(
          get_logger(), "Both arms initialized and powered on");
    } else {
      RCLCPP_ERROR(
          get_logger(), "%s", response.message.c_str());
    }
  }

  ArmConfig left_;
  ArmConfig right_;
  std::mutex mutex_;
  rclcpp::Service<Trigger>::SharedPtr leftInitializeService_;
  rclcpp::Service<Trigger>::SharedPtr rightInitializeService_;
  rclcpp::Service<Trigger>::SharedPtr dualInitializeService_;
  rclcpp::Service<Trigger>::SharedPtr leftPowerService_;
  rclcpp::Service<Trigger>::SharedPtr rightPowerService_;
};

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeRobotInitializerService() {
  return std::make_shared<RokaeRobotInitializerService>();
}

}  // namespace rokae_driver

#ifndef ROKAE_UNIFIED_DRIVER
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<RokaeRobotInitializerService>());
  } catch (const std::exception &exception) {
    RCLCPP_FATAL(
        rclcpp::get_logger("rokae_robot_initializer"), "%s",
        exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
#endif
