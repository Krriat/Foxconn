#include <cassert>
#include <limits>
#include "rokae_driver/moveabsj_path.hpp"

int main() {
  using Request = rokae_interfaces::srv::MoveAbsJPath::Request;
  Request request;
  request.joint_positions = {0, 1, 2, 3, 4, 5, 6, .1, .2, .3, .4, .5, .6, .7};
  request.joint_speed_scales = {.05, .12};
  request.zones_mm = {5., 0.};
  request.joint_tolerance_rad = .01;
  request.timeout_s = 25.;
  const auto points = rokae_driver::moveabsj_path::decode(request);
  assert(points.size() == 2);
  assert(points[0].joint_positions[6] == 6.);
  assert(points[1].joint_positions[6] == .7);
  assert(points[0].joint_speed_scale == .05 && points[1].joint_speed_scale == .12);
  assert(points[0].zone_mm == 5. && points[1].zone_mm == 0.);
  assert(points[1].timeout_s == 25. && points[1].joint_tolerance_rad == .01);
  const auto rejects = [](const Request &value) {
    try { rokae_driver::moveabsj_path::decode(value); }
    catch (const std::invalid_argument &) { return true; }
    return false;
  };
  auto bad = request;
  bad.joint_positions.pop_back();
  assert(rejects(bad));
  bad = request; bad.joint_speed_scales.pop_back(); assert(rejects(bad));
  bad = request; bad.zones_mm.back() = 5.; assert(rejects(bad));
  bad = request; bad.joint_positions[0] = std::numeric_limits<double>::quiet_NaN(); assert(rejects(bad));
  bad = request; bad.zones_mm[0] = 201.; assert(rejects(bad));
  bad = request; bad.joint_speed_scales[0] = 0.; assert(rejects(bad));
  bad = request; bad.joint_positions.resize(707); bad.joint_speed_scales.resize(101, .1); bad.zones_mm.resize(101, 0.);
  assert(rejects(bad));
  request.joint_positions.resize(7); request.joint_speed_scales.resize(1); request.zones_mm = {0.};
  assert(rokae_driver::moveabsj_path::decode(request).size() == 1);

  Request cartesian;
  cartesian.use_cartesian_targets = true;
  cartesian.poses.resize(3);
  for (std::size_t i = 0; i < 3; ++i) {
    cartesian.poses[i].position.x = .4 + .05 * i;
    cartesian.poses[i].orientation.w = 1.;
  }
  cartesian.use_elbows = {true, false, true};
  cartesian.elbows = {.1, 0., .3};
  cartesian.tcp_type = "tcp_gripper_pose";
  cartesian.joint_speed_scales = {.05, .06, .07};
  cartesian.zones_mm = {10., 10., 0.};
  cartesian.timeout_s = 25.;
  assert(rokae_driver::moveabsj_path::validateCartesian(cartesian) == 3);
  assert(rejects(cartesian)); // Cartesian input may never bypass IK via decode().
  std::size_t calls = 0;
  const auto solved = rokae_driver::moveabsj_path::resolveCartesian(cartesian,
      [&](std::size_t i, const geometry_msgs::msg::Pose &pose, bool useElbow, double elbow) {
        assert(i == calls++);
        assert(pose.position.x == cartesian.poses[i].position.x);
        assert(useElbow == cartesian.use_elbows[i] && elbow == cartesian.elbows[i]);
        std::array<double, 7> joints{};
        joints.fill(.1 * i);
        return joints;
      });
  assert(calls == 3 && solved.size() == 3);
  assert(solved[0].zone_mm == 10. && solved[1].zone_mm == 10. && solved[2].zone_mm == 0.);
  assert(solved[0].joint_speed_scale == .05 && solved[2].joint_speed_scale == .07);
  assert(solved[2].joint_positions[6] == .2 && solved[2].timeout_s == 25.);
  calls = 0;
  bool complete = false;
  try {
    rokae_driver::moveabsj_path::resolveCartesian(cartesian,
        [&](std::size_t i, const geometry_msgs::msg::Pose &, bool, double) -> std::array<double, 7> {
          ++calls;
          if (i == 1) throw std::runtime_error("unreachable point");
          return {};
        });
    complete = true;
  } catch (const std::runtime_error &error) {
    assert(std::string(error.what()).find("waypoint 2") != std::string::npos);
  }
  assert(!complete && calls == 2); // No complete command list exists if any IK fails.
  const auto rejectsCartesian = [](const Request &value) {
    try { rokae_driver::moveabsj_path::validateCartesian(value); }
    catch (const std::invalid_argument &) { return true; }
    return false;
  };
  bad = cartesian; bad.joint_positions.resize(7); assert(rejectsCartesian(bad));
  bad = cartesian; bad.use_cartesian_targets = false; assert(rejectsCartesian(bad) && rejects(bad));
  bad = cartesian; bad.use_elbows.pop_back(); assert(rejectsCartesian(bad));
  bad = cartesian; bad.elbows.pop_back(); assert(rejectsCartesian(bad));
  bad = cartesian; bad.poses[1].orientation.w = 0.; assert(rejectsCartesian(bad));
  bad = cartesian; bad.poses[1].position.z = std::numeric_limits<double>::quiet_NaN(); assert(rejectsCartesian(bad));
  bad = cartesian; bad.elbows[0] = std::numeric_limits<double>::infinity(); assert(rejectsCartesian(bad));
  bad = cartesian; bad.tcp_type = "unknown"; assert(rejectsCartesian(bad));
  bad = cartesian; bad.zones_mm.back() = 10.; assert(rejectsCartesian(bad));
  bad = cartesian; bad.joint_speed_scales[1] = 0.; assert(rejectsCartesian(bad));
  bad = cartesian; bad.poses.clear(); assert(rejectsCartesian(bad));
}
