/**
 * @file ros_kinematics_service.cpp
 * @brief Left/right FK and IK services backed by the xCoreSDK model library.
 *
 * Services:
 *   /robot_02/upperlimb/fk/left_arm
 *   /robot_02/upperlimb/fk/right_arm
 *   /robot_02/upperlimb/ik/left_arm
 *   /robot_02/upperlimb/ik/right_arm
 *
 * Cartesian poses are TCP poses relative to the robot's configured external
 * reference. The active controller tool/workobject is read for every request.
 * These services calculate only and never power or move either robot.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

#include "rokae/model.h"
#include "rokae/robot.h"
#include "rokae/utility.h"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_interfaces/srv/forward_kinematics.hpp"
#include "rokae_interfaces/srv/inverse_kinematics.hpp"

namespace {

using ForwardKinematics = rokae_interfaces::srv::ForwardKinematics;
using InverseKinematics = rokae_interfaces::srv::InverseKinematics;
using CartesianPosition = rokae::CartesianPosition;
using Pose = geometry_msgs::msg::Pose;

constexpr double kMinimumQuaternionNorm = 1e-12;

std::string sdkError(
    const std::string &operation, const std::error_code &error) {
  return operation + " failed: code=" + std::to_string(error.value()) +
         ", message=" + error.message();
}

bool finitePose(const Pose &pose) {
  const std::array<double, 7> values{
      pose.position.x, pose.position.y, pose.position.z,
      pose.orientation.x, pose.orientation.y, pose.orientation.z,
      pose.orientation.w};
  return std::all_of(
      values.begin(), values.end(),
      [](double value) { return std::isfinite(value); });
}

CartesianPosition toSdkPose(const Pose &pose) {
  if (!finitePose(pose)) {
    throw std::invalid_argument("pose contains a non-finite value");
  }

  tf2::Quaternion quaternion(
      pose.orientation.x, pose.orientation.y,
      pose.orientation.z, pose.orientation.w);
  const double norm = quaternion.length();
  if (!std::isfinite(norm) || norm < kMinimumQuaternionNorm) {
    throw std::invalid_argument("pose orientation quaternion is zero");
  }
  quaternion.normalize();
  const tf2::Matrix3x3 rotation(quaternion);

  std::array<double, 16> transform{
      rotation[0][0], rotation[0][1], rotation[0][2], pose.position.x,
      rotation[1][0], rotation[1][1], rotation[1][2], pose.position.y,
      rotation[2][0], rotation[2][1], rotation[2][2], pose.position.z,
      0.0, 0.0, 0.0, 1.0};
  std::array<double, 6> xyzabc{};
  rokae::Utils::transArrayToPosture(transform, xyzabc);
  if (!std::all_of(
          xyzabc.begin(), xyzabc.end(),
          [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error("SDK pose conversion produced invalid RPY");
  }
  return CartesianPosition({
      xyzabc[0], xyzabc[1], xyzabc[2],
      xyzabc[3], xyzabc[4], xyzabc[5]});
}

Pose toRosPose(const CartesianPosition &pose) {
  const std::array<double, 6> xyzabc{
      pose.trans[0], pose.trans[1], pose.trans[2],
      pose.rpy[0], pose.rpy[1], pose.rpy[2]};
  if (!std::all_of(
          xyzabc.begin(), xyzabc.end(),
          [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error("SDK returned a non-finite Cartesian pose");
  }

  std::array<double, 16> transform{};
  rokae::Utils::postureToTransArray(xyzabc, transform);
  const tf2::Matrix3x3 rotation(
      transform[0], transform[1], transform[2],
      transform[4], transform[5], transform[6],
      transform[8], transform[9], transform[10]);
  tf2::Quaternion quaternion;
  rotation.getRotation(quaternion);
  quaternion.normalize();

  Pose result;
  result.position.x = pose.trans[0];
  result.position.y = pose.trans[1];
  result.position.z = pose.trans[2];
  result.orientation.x = quaternion.x();
  result.orientation.y = quaternion.y();
  result.orientation.z = quaternion.z();
  result.orientation.w = quaternion.w();
  return result;
}

double positionError(const Pose &first, const Pose &second) {
  return std::hypot(
      std::hypot(
          first.position.x - second.position.x,
          first.position.y - second.position.y),
      first.position.z - second.position.z);
}

double orientationError(const Pose &first, const Pose &second) {
  tf2::Quaternion a(
      first.orientation.x, first.orientation.y,
      first.orientation.z, first.orientation.w);
  tf2::Quaternion b(
      second.orientation.x, second.orientation.y,
      second.orientation.z, second.orientation.w);
  a.normalize();
  b.normalize();
  const double dot = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return 2.0 * std::acos(dot);
}

struct ArmContext {
  std::string side;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware;
  rclcpp::CallbackGroup::SharedPtr callback_group;
  rclcpp::Service<ForwardKinematics>::SharedPtr fk_service;
  rclcpp::Service<InverseKinematics>::SharedPtr ik_service;
};

}  // namespace

class RokaeKinematicsService : public rclcpp::Node {
 public:
  RokaeKinematicsService()
      : Node("kinematics", rokae_driver::interface_names::kNodeNamespace) {
    verify_position_tolerance_ = declare_parameter<double>(
        "ik_verify_position_tolerance_m", 1e-4);
    verify_orientation_tolerance_ = declare_parameter<double>(
        "ik_verify_orientation_tolerance_rad", 1e-3);
    if (!std::isfinite(verify_position_tolerance_) ||
        verify_position_tolerance_ <= 0.0 ||
        !std::isfinite(verify_orientation_tolerance_) ||
        verify_orientation_tolerance_ <= 0.0) {
      throw std::invalid_argument(
          "IK verification tolerances must be finite and positive");
    }

    left_ = createArm(
        "left", "192.168.1.160", "192.168.1.10");
    right_ = createArm(
        "right", "192.168.2.160", "192.168.2.10");

    RCLCPP_INFO(
        get_logger(),
        "Left/right FK and IK services are ready; calculations do not move "
        "or power either robot");
  }

 private:
  std::unique_ptr<ArmContext> createArm(
      const std::string &side, const std::string &default_robot_ip,
      const std::string &default_local_ip) {
    const std::string prefix = side + "_arm.";
    const auto robot_ip = declare_parameter<std::string>(
        prefix + "robot_ip", default_robot_ip);
    const auto local_ip = declare_parameter<std::string>(
        prefix + "local_ip", default_local_ip);

    auto arm = std::make_unique<ArmContext>();
    arm->side = side;
    arm->hardware = rokae_driver::sharedArm(side, robot_ip, local_ip);
    arm->callback_group = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    ArmContext *const context = arm.get();
    const auto fk_name = rokae_driver::interface_names::arm("fk", side);
    arm->fk_service = create_service<ForwardKinematics>(
        fk_name,
        [this, context](
            const std::shared_ptr<ForwardKinematics::Request> request,
            std::shared_ptr<ForwardKinematics::Response> response) {
          calculateFk(*context, *request, *response);
        },
        rmw_qos_profile_services_default, arm->callback_group);

    const auto ik_name = rokae_driver::interface_names::arm("ik", side);
    arm->ik_service = create_service<InverseKinematics>(
        ik_name,
        [this, context](
            const std::shared_ptr<InverseKinematics::Request> request,
            std::shared_ptr<InverseKinematics::Response> response) {
          calculateIk(*context, *request, *response);
        },
        rmw_qos_profile_services_default, arm->callback_group);

    RCLCPP_INFO(
        get_logger(), "%s and %s are ready",
        fk_name.c_str(), ik_name.c_str());
    return arm;
  }

  void calculateFk(
      ArmContext &arm, const ForwardKinematics::Request &request,
      ForwardKinematics::Response &response) {
    if (!std::all_of(
            request.joints.begin(), request.joints.end(),
            [](double value) { return std::isfinite(value); })) {
      fail(response, "joint positions must all be finite");
      return;
    }

    std::unique_lock<std::mutex> command_lock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!command_lock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another command");
      return;
    }

    try {
      std::error_code error;
      CartesianPosition result;
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        const auto toolset = robot.toolset(error);
        if (error) return;

        // Follow the SDK FK example exactly. xMateModel owns SDK-internal
        // state and must not be retained by copying/moving a temporary model
        // returned during node construction.
        auto model = robot.model();
        result = model.calcFk(request.joints, toolset, error);
      });
      if (error) {
        fail(response, sdkError("FK", error));
        return;
      }

      response.pose = toRosPose(result);
      response.elbow = result.elbow;
      response.has_elbow = result.hasElbow;
      response.configuration.assign(
          result.confData.begin(), result.confData.end());
      response.success = true;
      response.message = arm.side +
          " FK calculated in the active external-reference/tool frame";
    } catch (const std::exception &error) {
      fail(response, std::string("FK exception: ") + error.what());
    }
  }

  void calculateIk(
      ArmContext &arm, const InverseKinematics::Request &request,
      InverseKinematics::Response &response) {
    if (!std::isfinite(request.elbow)) {
      fail(response, "elbow must be finite");
      return;
    }

    std::unique_lock<std::mutex> command_lock(
        arm.hardware->commandMutex(), std::try_to_lock);
    if (!command_lock.owns_lock()) {
      fail(response, arm.side + " arm is occupied by another command");
      return;
    }

    try {
      CartesianPosition target = toSdkPose(request.pose);
      if (request.use_elbow) {
        target.elbow = request.elbow;
        target.hasElbow = true;
      }

      std::error_code error;
      std::array<double, 7> joints{};
      CartesianPosition solved;
      arm.hardware->withRobot([&](rokae::ArRobot &robot) {
        const auto toolset = robot.toolset(error);
        if (error) return;

        // robot.model() is intentionally local to this serialized SDK call,
        // matching the vendor FK/IK examples and their object lifetime.
        auto model = robot.model();
        joints = model.calcIk(target, toolset, error);
        if (!error) solved = model.calcFk(joints, toolset, error);
      });
      if (error) {
        fail(response, sdkError("IK or FK verification", error));
        return;
      }
      if (!std::all_of(
              joints.begin(), joints.end(),
              [](double value) { return std::isfinite(value); })) {
        fail(response, "SDK IK returned a non-finite joint position");
        return;
      }

      response.joints = joints;
      response.solved_pose = toRosPose(solved);
      response.position_error_m =
          positionError(request.pose, response.solved_pose);
      response.orientation_error_rad =
          orientationError(request.pose, response.solved_pose);
      if (response.position_error_m > verify_position_tolerance_ ||
          response.orientation_error_rad > verify_orientation_tolerance_) {
        fail(
            response,
            "IK solution failed FK verification: position_error=" +
                std::to_string(response.position_error_m) +
                " m, orientation_error=" +
                std::to_string(response.orientation_error_rad) + " rad");
        return;
      }

      response.success = true;
      response.message = arm.side +
          " IK calculated and verified with FK in the active frame";
    } catch (const std::exception &error) {
      fail(response, std::string("IK exception: ") + error.what());
    }
  }

  template <typename Response>
  void fail(Response &response, const std::string &message) {
    response.success = false;
    response.message = message;
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
  }

  double verify_position_tolerance_{1e-4};
  double verify_orientation_tolerance_{1e-3};
  std::unique_ptr<ArmContext> left_;
  std::unique_ptr<ArmContext> right_;
};

namespace rokae_driver {

std::shared_ptr<rclcpp::Node> makeKinematicsService() {
  return std::make_shared<RokaeKinematicsService>();
}

}  // namespace rokae_driver
