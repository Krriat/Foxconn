#ifndef ROKAE_DRIVER__MOVEABSJ_PATH_HPP_
#define ROKAE_DRIVER__MOVEABSJ_PATH_HPP_

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "rokae_interfaces/srv/move_abs_j.hpp"
#include "rokae_interfaces/srv/move_abs_j_path.hpp"

namespace rokae_driver::moveabsj_path {
inline rokae_interfaces::srv::MoveAbsJ::Request pointOptions(
    const rokae_interfaces::srv::MoveAbsJPath::Request &request, std::size_t index) {
  rokae_interfaces::srv::MoveAbsJ::Request point;
  point.joint_speed_scale = request.joint_speed_scales.at(index);
  point.zone_mm = request.zones_mm.at(index);
  point.speed_mm_s = request.speed_mm_s;
  point.joint_tolerance_rad = request.joint_tolerance_rad;
  point.stable_samples = request.stable_samples;
  point.poll_interval_ms = request.poll_interval_ms;
  point.timeout_s = request.timeout_s;
  return point;
}

inline std::size_t validateCartesian(
    const rokae_interfaces::srv::MoveAbsJPath::Request &request) {
  const auto count = request.poses.size();
  if (!request.use_cartesian_targets || !request.joint_positions.empty() || count < 1 || count > 100 ||
      request.use_elbows.size() != count || request.elbows.size() != count ||
      request.joint_speed_scales.size() != count || request.zones_mm.size() != count)
    throw std::invalid_argument("Cartesian MoveAbsJPath requires 1..100 poses, N elbow flags/angles, speeds/zones and no joint positions");
  if (request.zones_mm.back() != 0.)
    throw std::invalid_argument("MoveAbsJPath last waypoint must have zone_mm=0");
  if (request.tcp_type != "sdk_tcp" && request.tcp_type != "tcp_gripper_pose" && request.tcp_type != "tcp_cut_pos")
    throw std::invalid_argument("unknown Cartesian MoveAbsJPath tcp_type");
  for (std::size_t i = 0; i < count; ++i) {
    const auto &p = request.poses[i];
    const auto &q = p.orientation;
    const double norm = std::hypot(std::hypot(q.x, q.y), std::hypot(q.z, q.w));
    if (!std::isfinite(p.position.x) || !std::isfinite(p.position.y) || !std::isfinite(p.position.z) ||
        !std::isfinite(norm) || norm < 1e-12 || !std::isfinite(request.elbows[i]) ||
        !std::isfinite(request.joint_speed_scales[i]) || request.joint_speed_scales[i] <= 0. || request.joint_speed_scales[i] > 1. ||
        !std::isfinite(request.zones_mm[i]) || request.zones_mm[i] < 0. || request.zones_mm[i] > 200.)
      throw std::invalid_argument("Cartesian MoveAbsJPath invalid pose, elbow, speed or zone at waypoint " + std::to_string(i + 1));
  }
  return count;
}

// Compute every IK target before returning any executable command points.
// Solver is supplied by the caller while holding the arm command lock.
template<class Solver>
std::vector<rokae_interfaces::srv::MoveAbsJ::Request> resolveCartesian(
    const rokae_interfaces::srv::MoveAbsJPath::Request &request, Solver solve) {
  const auto count = validateCartesian(request);
  std::vector<rokae_interfaces::srv::MoveAbsJ::Request> result;
  for (std::size_t i = 0; i < count; ++i) {
    auto point = pointOptions(request, i);
    try {
      point.joint_positions = solve(i, request.poses[i], request.use_elbows[i], request.elbows[i]);
      if (!std::all_of(point.joint_positions.begin(), point.joint_positions.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("IK returned non-finite joints");
    } catch (const std::exception &error) {
      throw std::runtime_error("Cartesian MoveAbsJPath waypoint " + std::to_string(i + 1) + ": " + error.what());
    }
    result.push_back(point);
  }
  return result;
}

// Pure decoding: this helper never opens an SDK connection or issues motion.
inline std::vector<rokae_interfaces::srv::MoveAbsJ::Request> decode(
    const rokae_interfaces::srv::MoveAbsJPath::Request &request) {
  if (request.use_cartesian_targets || !request.poses.empty() || !request.use_elbows.empty() || !request.elbows.empty())
    throw std::invalid_argument("Cartesian MoveAbsJPath must resolve all poses before joint decoding");
  const auto count = request.joint_positions.size() / 7;
  if (count < 1 || count > 100 || request.joint_positions.size() % 7 != 0 ||
      request.joint_speed_scales.size() != count || request.zones_mm.size() != count)
    throw std::invalid_argument("MoveAbsJPath requires 1..100 complete seven-joint points and N speeds/zones");
  if (request.zones_mm.back() != 0.)
    throw std::invalid_argument("MoveAbsJPath last waypoint must have zone_mm=0");
  std::vector<rokae_interfaces::srv::MoveAbsJ::Request> result;
  for (std::size_t index = 0; index < count; ++index) {
    auto point = pointOptions(request, index);
    std::copy_n(request.joint_positions.begin() + index * 7, 7, point.joint_positions.begin());
    if (!std::all_of(point.joint_positions.begin(), point.joint_positions.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::isfinite(point.joint_speed_scale) || point.joint_speed_scale <= 0. || point.joint_speed_scale > 1. ||
        !std::isfinite(point.zone_mm) || point.zone_mm < 0. || point.zone_mm > 200.)
      throw std::invalid_argument("MoveAbsJPath contains invalid joint positions, speed fractions or zones");
    result.push_back(point);
  }
  return result;
}
}  // namespace rokae_driver::moveabsj_path
#endif
