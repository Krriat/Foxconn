#ifndef ROKAE_DRIVER__CARTESIAN_MOVEJ_HPP_
#define ROKAE_DRIVER__CARTESIAN_MOVEJ_HPP_

#include <cmath>
#include <algorithm>
#include <string>
#include "rokae_driver/shared_arm_hardware.hpp"
#include "rokae_interfaces/srv/move_j_by_path.hpp"
#include "rokae/utility.h"
#include "tf2/LinearMath/Quaternion.h"

namespace rokae_driver::cartesian_movej {
using Request = rokae_interfaces::srv::MoveJByPath::Request;
using Pose = geometry_msgs::msg::Pose;
inline void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
inline tf2::Quaternion quaternion(const Pose &pose) {
  tf2::Quaternion q(pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w);
  require(std::isfinite(q.length2()) && q.length2() > 1e-24,
          "pose needs a finite, nonzero quaternion");
  q.normalize();
  return q;
}
inline rokae::Toolset selectTool(const rokae::Toolset &original, const std::string &type) {
  auto tool = original;
  if (type.empty() || type == "sdk_tcp") return tool;
  if (type == "tcp_gripper_pose") {
    tool.end = rokae::Frame({0., 0., -0.11, 0., 0., 0.});
  } else if (type == "tcp_cut_pos") {
    const double angle = std::acos(-1.) / 12.;
    tool.end = rokae::Frame({-0.11 * std::cos(angle), -0.11 * std::sin(angle), 0., 0., 0., angle});
  } else throw std::runtime_error("unknown tcp_type: " + type);
  return tool;
}
inline void checkObservation(const rokae::CartesianPosition &current, const Pose &observed) {
  const auto q = quaternion(observed);
  const auto actual = rokae::Utils::eulerToQuaternion(current.rpy);
  const double distance = std::hypot(std::hypot(current.trans[0] - observed.position.x,
                                               current.trans[1] - observed.position.y),
                                    current.trans[2] - observed.position.z);
  const double dot = actual[0]*q.w() + actual[1]*q.x() + actual[2]*q.y() + actual[3]*q.z();
  require(std::isfinite(distance) && distance <= .005 &&
          2.*std::acos(std::min(1., std::abs(dot))) <= .035,
          "selected TCP topic observation disagrees with SDK; check frame/tool calibration");
}
inline void validate(const Request &r) {
  require(r.joint_positions.empty() && r.left_joint_positions.empty() && r.right_joint_positions.empty(),
          "Cartesian MoveJ cannot also contain joint waypoints");
  require(std::isfinite(r.joint_speed_scale) && r.joint_speed_scale > 0. && r.joint_speed_scale <= 1.,
          "joint_speed_scale must be in (0,1]");
  require(std::isfinite(r.zone_mm) && r.zone_mm == 0., "Cartesian MoveJ supports only zone_mm=0");
  require(!r.is_async, "Cartesian MoveJ supports only is_async=false");
  require(r.conf_data.empty() || r.conf_data.size() == 8, "conf_data must be empty or have 8 integers");
  require(!r.use_elbow || std::isfinite(r.elbow), "elbow must be finite");
  require(std::isfinite(r.timeout_s) && r.timeout_s > 0. && r.timeout_s <= 600., "timeout_s must be in (0,600]");
  require(std::isfinite(r.resource_valid_until_unix_s) && r.resource_valid_until_unix_s >= 0., "invalid resource validity deadline");
  require(r.stable_samples >= 1 && r.stable_samples <= 100, "stable_samples must be in [1,100]");
  require(r.poll_interval_ms >= 10 && r.poll_interval_ms <= 1000, "poll_interval_ms must be in [10,1000]");
  require(std::isfinite(r.joint_tolerance_rad) && r.joint_tolerance_rad > 0. && r.joint_tolerance_rad <= 0.2,
          "invalid joint_tolerance_rad");
  require(std::isfinite(r.position_tolerance_m) && r.position_tolerance_m > 0. && r.position_tolerance_m <= 0.05,
          "invalid position_tolerance_m");
  require(std::isfinite(r.orientation_tolerance_rad) && r.orientation_tolerance_rad > 0. && r.orientation_tolerance_rad <= 0.2,
          "invalid orientation_tolerance_rad");
  selectTool(rokae::Toolset{}, r.tcp_type);
  if (r.check_current_pose) quaternion(r.expected_current_pose);
  if (!r.use_current_pose) {
    require(std::isfinite(r.pose.position.x) && std::isfinite(r.pose.position.y) && std::isfinite(r.pose.position.z),
            "pose position must be finite");
    if (!r.use_current_orientation) quaternion(r.pose);
  }
}
// The caller owns hardware.commandMutex() for the entire request.
std::string execute(SharedArmHardware &hardware, const Request &request,
                    double soft_limit_margin_rad, double max_goal_delta_rad);
}  // namespace rokae_driver::cartesian_movej
#endif
