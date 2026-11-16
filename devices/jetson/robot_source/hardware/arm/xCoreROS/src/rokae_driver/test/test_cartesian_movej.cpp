#include <iostream>
#include <limits>
#include "rokae_driver/cartesian_movej.hpp"

using namespace rokae_driver::cartesian_movej;
void expectReject(const Request &request) {
  try { validate(request); }
  catch (const std::runtime_error &) { return; }
  throw std::runtime_error("invalid request was accepted");
}
void near(double actual, double expected) {
  require(std::abs(actual - expected) < 1e-10, "transform mismatch");
}
int main() {
  Request request;
  request.use_cartesian_target = true;
  request.pose.orientation.w = 1.;
  validate(request);
  auto invalid = request;
  invalid.pose.orientation.w = 0.;
  expectReject(invalid);
  invalid = request;
  invalid.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  expectReject(invalid);
  invalid = request;
  invalid.joint_positions = {0., 0., 0., 0., 0., 0., 0.};
  expectReject(invalid);
  invalid = request;
  invalid.conf_data = {1, 0};
  expectReject(invalid);
  invalid = request;
  invalid.is_async = true;
  expectReject(invalid);
  invalid = request;
  invalid.joint_speed_scale = 0.;
  expectReject(invalid);
  invalid = request;
  invalid.zone_mm = 1.;
  expectReject(invalid);
  invalid = request;
  invalid.timeout_s = std::numeric_limits<double>::infinity();
  expectReject(invalid);
  invalid = request;
  invalid.resource_valid_until_unix_s = std::numeric_limits<double>::quiet_NaN();
  expectReject(invalid);
  invalid = request;
  invalid.stable_samples = 0;
  expectReject(invalid);
  invalid = request;
  invalid.poll_interval_ms = 0;
  expectReject(invalid);
  invalid = request;
  invalid.tcp_type = "missing_tool";
  expectReject(invalid);
  request.use_current_pose = true;
  request.pose.orientation.w = 0.;
  validate(request);  // Omitted target orientation is not used for current-pose mode.

  rokae::CartesianPosition current;
  current.trans = {.2, .3, .4};
  current.rpy = {0., 0., 0.};
  Pose observed;
  observed.position.x = .2; observed.position.y = .3; observed.position.z = .4;
  observed.orientation.w = 1.;
  checkObservation(current, observed);
  observed.position.x += .02;
  bool rejected = false;
  try { checkObservation(current, observed); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "inconsistent TCP observation was accepted");

  rokae::Toolset original;
  original.ref = rokae::Frame({0., 0., 0., 0., 0., 0.});
  original.end = rokae::Frame({0.03, 0.02, 0.01, 0.1, 0.2, 0.3});
  require(selectTool(original, "sdk_tcp").end.trans == original.end.trans, "SDK tool was changed");
  const auto gripper = selectTool(original, "tcp_gripper_pose");
  const auto cutter = selectTool(original, "tcp_cut_pos");
  require(cutter.ref.trans == original.ref.trans && cutter.ref.rpy == original.ref.rpy, "reference was changed");
  const std::array<double, 6> base{0., 0., 0., 0., 0., 0.};
  const std::array<double, 6> flange{0.2, 0.3, 0.4, std::acos(-1.) / 2., 0., 0.};
  const auto g = rokae::Utils::FlanInBaseToEndInRef(base, gripper, flange);
  near(g[0], 0.2); near(g[1], 0.41); near(g[2], 0.4);
  const auto c = rokae::Utils::FlanInBaseToEndInRef(base, cutter, flange);
  near(c[0], 0.09374815910820249); near(c[1], 0.3); near(c[2], 0.3715299050387227);
  std::array<double, 16> transform{};
  rokae::Utils::postureToTransArray(c, transform);
  near(transform[0], 0.9659258262890683);
  near(transform[4], 0.0);
  near(transform[8], 0.25881904510252074);
  std::cout << "PASS: request rejection, current-pose mode, SDK tool preservation, and rotated gripper/cutter transforms\n";
}
