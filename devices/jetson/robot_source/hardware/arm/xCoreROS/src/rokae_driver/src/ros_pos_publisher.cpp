#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rokae_interfaces/msg/tcp_state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"

#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/interface_names.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_driver/local_joint_feedback.hpp"
#include "rokae/robot.h"
#include "rokae/utility.h"

class RokaeStatePublisher : public rclcpp::Node {
 public:
  RokaeStatePublisher(
      const std::string &node_namespace,
      const std::string &default_arm_name,
      const std::string &default_robot_ip,
      const std::string &default_local_ip,
      const std::string &default_frame_id)
      : Node("state_publisher", node_namespace) {
    arm_name_ =
        declare_parameter<std::string>("arm_name", default_arm_name);
    robot_ip_ =
        declare_parameter<std::string>("robot_ip", default_robot_ip);
    local_ip_ =
        declare_parameter<std::string>("local_ip", default_local_ip);
    frame_id_ =
        declare_parameter<std::string>("frame_id", default_frame_id);
    const double rate_hz =
        declare_parameter<double>("rate_hz", 20.0);
    publish_jacobian_ =
        declare_parameter<bool>("publish_jacobian", true);
    if (!std::isfinite(rate_hz) || rate_hz <= 0.0 ||
        rate_hz > 200.0) {
      throw std::invalid_argument(
          "rate_hz must be a finite number in (0, 200]");
    }

    for (int i = 1; i <= 7; ++i) {
      joint_names_.push_back(
          arm_name_ + "_joint_" + std::to_string(i));
    }

    joint_topic_ = rokae_driver::interface_names::arm(
        "joint_states", arm_name_);
    pose_topic_ = rokae_driver::interface_names::arm(
        "flange_pose", arm_name_);
    jacobian_topic_ = rokae_driver::interface_names::arm(
        "jacobian", arm_name_);

    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        joint_topic_, rclcpp::QoS(10));

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        pose_topic_, rclcpp::QoS(10));
    gripper_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        rokae_driver::interface_names::arm("tcp_gripper_pose", arm_name_),
        rclcpp::QoS(10));
    gripper_state_pub_ = create_publisher<rokae_interfaces::msg::TcpState>(
        rokae_driver::interface_names::arm("tcp_gripper_state", arm_name_),
        rclcpp::QoS(10));
    cut_pos_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        rokae_driver::interface_names::arm("tcp_cut_pos", arm_name_),
        rclcpp::QoS(10));

    RCLCPP_INFO(
        get_logger(),
        "Connecting %s arm: robot=%s, local=%s",
        arm_name_.c_str(), robot_ip_.c_str(), local_ip_.c_str());

    hardware_ = rokae_driver::sharedArm(
        arm_name_, robot_ip_, local_ip_);

    if (publish_jacobian_) {
      // xMateModel has no public default constructor. Direct initialization
      // from robot.model() keeps one loaded model for the node lifetime.
      hardware_->withRobot([this](rokae::ArRobot &robot) {
        model_.reset(new rokae::xMateModel<7>(robot.model()));
      });
      jacobian_pub_ =
          create_publisher<std_msgs::msg::Float64MultiArray>(
              jacobian_topic_, rclcpp::QoS(10));
    }

    const auto period = std::chrono::duration_cast<
        std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / rate_hz));

    timer_ = create_wall_timer(
        period,
        std::bind(&RokaeStatePublisher::publishState, this));

    // An optional joints-only stream keeps 30 Hz recording supplied during
    // MoveAbsJ and after it finishes, without changing the ordinary pose rate.
    recording_joint_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        rokae_driver::interface_names::arm("recording_joint_states", arm_name_),
        rclcpp::SensorDataQoS().keep_last(5));
    recording_worker_ = std::make_unique<rokae_driver::AsyncJointPublisher>(
        [publisher=recording_joint_pub_, side=arm_name_](const rokae_driver::JointFeedbackSample &sample) {
          if (publisher->get_subscription_count() == 0) return;
          sensor_msgs::msg::JointState msg;
          msg.header.stamp = rclcpp::Time(sample.stamp_ns);
          for(int i=1;i<=7;++i) msg.name.push_back(side+"_joint_"+std::to_string(i));
          msg.position.assign(sample.joints.begin(),sample.joints.end());
          rokae_driver::traceFeedbackCall(side.c_str(),"recording.publish",[&] {publisher->publish(msg);});
          rokae_driver::FeedbackTrace::instance().emit(side.c_str(),"recording.publish","sample",0,0,sample.stamp_ns);
        });
    recording_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    recording_timer_ = create_wall_timer(std::chrono::milliseconds(10),
        std::bind(&RokaeStatePublisher::publishRecordingJoints, this), recording_group_);

    RCLCPP_INFO(
        get_logger(),
        "Publishing %s and %s at %.3f Hz; "
        "Jacobian publishing is %s",
        joint_topic_.c_str(), pose_topic_.c_str(),
        rate_hz,
        publish_jacobian_ ? "enabled" : "disabled");
    RCLCPP_INFO(
        get_logger(), "Publishing %s at %.3f Hz",
        rokae_driver::interface_names::arm("tcp_cut_pos", arm_name_).c_str(),
        rate_hz);
  }

 private:
  void publishRecordingJoints() {
    if (!rokae_driver::LocalJointFeedback::instance().enabled() &&
        rokae_driver::traceFeedbackCall(arm_name_.c_str(), "recording.subscription_count", [&] {
          return recording_joint_pub_->get_subscription_count() +
                 recording_joint_pub_->get_intra_process_subscription_count();
        }) == 0) return;
    std::error_code ec;
    std::array<double,7> joints{};
    const bool acquired = hardware_->tryWithRobot(
        [this, &ec, &joints](rokae::ArRobot &robot) {
          joints = rokae_driver::traceFeedbackCall(arm_name_.c_str(), "recording.jointPos", [&] {
            return robot.jointPos(ec);
          });
        });
    if (!acquired) return;  // Active drag supplies the dedicated measured stream.
    if (ec || !std::all_of(joints.begin(), joints.end(),
                          [](double value) { return std::isfinite(value); })) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                          "%s recording joint feedback failed", arm_name_.c_str());
      return;
    }
    const rokae_driver::JointFeedbackSample sample{
        static_cast<std::uint8_t>(arm_name_=="left"?0:1),1,
        static_cast<std::uint64_t>(now().nanoseconds()),
        rokae_driver::LocalJointFeedback::monoNow(),joints};
    rokae_driver::LocalJointFeedback::instance().offer(sample,recording_worker_->overwritten());
    rokae_driver::FeedbackTrace::instance().emit(arm_name_.c_str(),"recording.local_send","sample",0,0,sample.stamp_ns);
    recording_worker_->offer(sample);
  }

  void publishState() {
    std::error_code ec;

    const auto joints = hardware_->withRobot(
        [this, &ec](rokae::ArRobot &robot) {
          return rokae_driver::traceFeedbackCall(arm_name_.c_str(), "state.jointPos", [&] {
            return robot.jointPos(ec);
          });
        });
    if (ec) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "%s jointPos failed: code=%d, message=%s",
          arm_name_.c_str(),
          ec.value(), ec.message().c_str());
      return;
    }

    ec.clear();
    const auto pose = hardware_->withRobot(
        [this, &ec](rokae::ArRobot &robot) {
          const auto flange = rokae_driver::traceFeedbackCall(arm_name_.c_str(), "state.posture", [&] {
            return robot.posture(rokae::CoordinateType::flangeInBase, ec);
          });
          if (ec) return std::array<double, 6>{};
          const auto base = rokae_driver::traceFeedbackCall(arm_name_.c_str(), "state.baseFrame", [&] {
            return robot.baseFrame(ec);
          });
          if (ec) return std::array<double, 6>{};
          auto toolset = rokae_driver::traceFeedbackCall(arm_name_.c_str(), "state.toolset", [&] {
            return robot.toolset(ec);
          });
          if (ec) return std::array<double, 6>{};
          // Keep the active external reference, but remove the tool offset:
          // T_ref_flange = inverse(T_world_ref) * T_world_base * T_base_flange.
          toolset.end = rokae::Frame({0., 0., 0., 0., 0., 0.});
          return rokae::Utils::FlanInBaseToEndInRef(base, toolset, flange);
        });

    if (ec) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "%s posture failed: code=%d, message=%s",
          arm_name_.c_str(),
          ec.value(), ec.message().c_str());
      return;
    }

    const auto stamp = now();

    sensor_msgs::msg::JointState joint_msg;
    joint_msg.header.stamp = stamp;
    joint_msg.name = joint_names_;
    joint_msg.position.assign(joints.begin(), joints.end());
    rokae_driver::traceFeedbackCall(arm_name_.c_str(), "state.publish", [&] {
      joint_pub_->publish(joint_msg);
    });
    rokae_driver::FeedbackTrace::instance().emit(arm_name_.c_str(), "state.publish", "sample",
        0, 0, rclcpp::Time(joint_msg.header.stamp).nanoseconds());

    publishJacobianIfSubscribed(joints);

    // 通过SDK生成旋转矩阵，避免直接猜测RPY旋转顺序。
    std::array<double, 16> transform{};
    rokae::Utils::postureToTransArray(pose, transform);

    tf2::Matrix3x3 rotation(
        transform[0], transform[1], transform[2],
        transform[4], transform[5], transform[6],
        transform[8], transform[9], transform[10]);

    tf2::Quaternion quaternion;
    rotation.getRotation(quaternion);
    quaternion.normalize();

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = frame_id_;

    pose_msg.pose.position.x = pose[0];
    pose_msg.pose.position.y = pose[1];
    pose_msg.pose.position.z = pose[2];

    pose_msg.pose.orientation.x = quaternion.x();
    pose_msg.pose.orientation.y = quaternion.y();
    pose_msg.pose.orientation.z = quaternion.z();
    pose_msg.pose.orientation.w = quaternion.w();
    pose_pub_->publish(pose_msg);

    rokae_interfaces::msg::TcpState state_msg;
    state_msg.header = pose_msg.header;
    state_msg.pose = pose_msg.pose;
    for (std::size_t index = 0; index < 3; ++index) {
      state_msg.orientation_rpy[index] = pose[index + 3];
    }

    // Start from the flange, rotate +15 degrees around its local Z axis,
    // then translate -0.11 m along the rotated local X axis.
    // T_ref_cut = T_ref_flange * Rz(15 deg) * Tx(-0.11 m).
    tf2::Quaternion cut_rotation;
    cut_rotation.setRotation(
        tf2::Vector3(0.0, 0.0, 1.0), std::acos(-1.0) / 12.0);
    auto cut_quaternion = quaternion * cut_rotation;
    cut_quaternion.normalize();
    const auto cut_offset =
        tf2::Matrix3x3(cut_quaternion) * tf2::Vector3(-0.11, 0.0, 0.0);
    auto cut_msg = pose_msg;
    cut_msg.pose.position.x += cut_offset.x();
    cut_msg.pose.position.y += cut_offset.y();
    cut_msg.pose.position.z += cut_offset.z();
    cut_msg.pose.orientation.x = cut_quaternion.x();
    cut_msg.pose.orientation.y = cut_quaternion.y();
    cut_msg.pose.orientation.z = cut_quaternion.z();
    cut_msg.pose.orientation.w = cut_quaternion.w();
    cut_pos_pub_->publish(cut_msg);

    // Gripper TCP is 0.11 m along the flange's local -Z axis.
    // Rotate the tool offset into the external reference frame so it follows
    // wrist rotation: p_ref_tcp = p_ref_flange + R_ref_flange * p_flange_tcp.
    const auto gripper_offset = rotation * tf2::Vector3(0.0, 0.0, -0.11);
    pose_msg.pose.position.x += gripper_offset.x();
    pose_msg.pose.position.y += gripper_offset.y();
    pose_msg.pose.position.z += gripper_offset.z();
    gripper_pose_pub_->publish(pose_msg);
    state_msg.pose = pose_msg.pose;
    gripper_state_pub_->publish(state_msg);
  }

  void publishJacobianIfSubscribed(
      const std::array<double, 7> &joints) {
    if (!publish_jacobian_ || !model_ || !jacobian_pub_) {
      return;
    }

    const auto subscribers =
        jacobian_pub_->get_subscription_count() +
        jacobian_pub_->get_intra_process_subscription_count();
    if (subscribers == 0) {
      return;
    }

    // SDK output is a row-major 6x7 matrix for the flange relative to the
    // robot base. Rows: vx, vy, vz, wx, wy, wz. Columns: joint_1..joint_7.
    std::array<double, 42> values{};
    try {
      values = model_->jacobian(
          joints, rokae::SegmentFrame::flange);
    } catch (const std::exception &error) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "%s Jacobian calculation failed: %s",
          arm_name_.c_str(), error.what());
      return;
    }

    std_msgs::msg::Float64MultiArray message;
    message.layout.dim.resize(2);
    message.layout.dim[0].label = "twist";
    message.layout.dim[0].size = 6;
    message.layout.dim[0].stride = 42;
    message.layout.dim[1].label = "joint";
    message.layout.dim[1].size = 7;
    message.layout.dim[1].stride = 7;
    message.layout.data_offset = 0;
    message.data.assign(values.begin(), values.end());
    jacobian_pub_->publish(message);
  }

  std::string arm_name_;
  std::string robot_ip_;
  std::string local_ip_;
  std::string frame_id_;
  std::string joint_topic_;
  std::string pose_topic_;
  std::string jacobian_topic_;
  bool publish_jacobian_{true};

  std::vector<std::string> joint_names_;
  std::shared_ptr<rokae_driver::SharedArmHardware> hardware_;
  std::unique_ptr<rokae::xMateModel<7>> model_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr recording_joint_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr gripper_pose_pub_;
  rclcpp::Publisher<rokae_interfaces::msg::TcpState>::SharedPtr gripper_state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr cut_pos_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      jacobian_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr recording_timer_;
  rclcpp::CallbackGroup::SharedPtr recording_group_;
  std::unique_ptr<rokae_driver::AsyncJointPublisher> recording_worker_;
};

namespace rokae_driver {

std::vector<std::shared_ptr<rclcpp::Node>> makeStatePublishers() {
  std::vector<std::shared_ptr<rclcpp::Node>> nodes;
  nodes.push_back(std::make_shared<RokaeStatePublisher>(
      interface_names::nodeNamespace("left"), "left",
      "192.168.1.160", "192.168.1.10",
      "left_external_ref"));
  nodes.push_back(std::make_shared<RokaeStatePublisher>(
      interface_names::nodeNamespace("right"), "right",
      "192.168.2.160", "192.168.2.10",
      "right_external_ref"));
  return nodes;
}

}  // namespace rokae_driver

#ifndef ROKAE_UNIFIED_DRIVER
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  int exit_code = 0;
  try {
    auto left = std::make_shared<RokaeStatePublisher>(
        rokae_driver::interface_names::nodeNamespace("left"), "left",
        "192.168.1.160", "192.168.1.10",
        "left_external_ref");
    auto right = std::make_shared<RokaeStatePublisher>(
        rokae_driver::interface_names::nodeNamespace("right"), "right",
        "192.168.2.160", "192.168.2.10",
        "right_external_ref");

    // 两个SDK读取定时器可独立执行，避免一侧网络等待阻塞另一侧。
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(left);
    executor.add_node(right);
    executor.spin();
  } catch (const std::exception &e) {
    std::cerr << "Node exception: " << e.what() << std::endl;
    exit_code = 1;
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return exit_code;
}
#endif
