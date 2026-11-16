#include "rokae_driver/cartesian_movej.hpp"

#include <algorithm>
#include <any>
#include <chrono>
#include <thread>
#include "rclcpp/rclcpp.hpp"

namespace rokae_driver::cartesian_movej {
namespace {
void check(const std::error_code &ec, const char *operation) {
  require(!ec, std::string(operation) + ": " + ec.message());
}
template <typename T>
const T *field(const rokae::EventInfo &info, const char *key) {
  const auto it = info.find(key);
  if (it == info.end()) return nullptr;
  const auto *value = std::any_cast<T>(&it->second);
  require(value != nullptr, std::string("unexpected event field type: ") + key);
  return value;
}
double angleError(const rokae::CartesianPosition &a, const rokae::CartesianPosition &b) {
  const auto qa = rokae::Utils::eulerToQuaternion(a.rpy);
  const auto qb = rokae::Utils::eulerToQuaternion(b.rpy);
  double dot = 0.;
  for (std::size_t i = 0; i < qa.size(); ++i) dot += qa[i] * qb[i];
  return 2. * std::acos(std::clamp(std::abs(dot), 0., 1.));
}
}  // namespace

std::string execute(SharedArmHardware &hardware, const Request &request,
                    double margin, double max_delta) {
  validate(request);  // Reject malformed requests before any SDK access.
  const auto generation = hardware.stopGeneration();
  rokae::Toolset original;
  bool tool_changed = false;
  bool command_owned = false;
  std::string id;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(request.timeout_s);
  const auto checkpoint = [&]() {
    require(hardware.stopGeneration() == generation, "stop service interrupted MoveJ");
    require(rclcpp::ok(), "ROS shutdown interrupted MoveJ");
    require(std::chrono::steady_clock::now() < deadline, "MoveJ timed out");
    require(std::isfinite(request.resource_valid_until_unix_s) && request.resource_valid_until_unix_s >= 0.,
            "invalid resource validity deadline");
  };
  const auto restore_tool = [&]() {
    if (!tool_changed) return;
    hardware.withRobot([&](rokae::ArRobot &robot) {
      std::error_code ec;
      const auto state = robot.operationState(ec);
      check(ec, "operationState(restore tool)");
      require(state == rokae::OperationState::idle, "tool restore requires confirmed idle");
      robot.setToolset(original, ec);
      check(ec, "restore toolset");
    });
    tool_changed = false;
  };
  try {
    std::array<double[2], 7> limits{};
    std::array<double, 7> target_joints{};
    rokae::CartesianPosition target;
    hardware.withRobot([&](rokae::ArRobot &robot) {
      checkpoint();
      std::error_code ec;
      const auto state = robot.operationState(ec);
      check(ec, "operationState");
      require(state == rokae::OperationState::idle, "arm must be idle before MoveJ");
      const auto power = robot.powerState(ec);
      check(ec, "powerState");
      require(power == rokae::PowerState::on, "robot must be powered on; call initialize first");
      original = robot.toolset(ec);
      check(ec, "toolset");
      const auto selected = selectTool(original, request.tcp_type);
      if (selected.end.trans != original.end.trans || selected.end.rpy != original.end.rpy) {
        tool_changed = true;
        robot.setToolset(selected, ec);
        check(ec, "select TCP toolset");
      }
      const auto current = robot.cartPosture(rokae::CoordinateType::endInRef, ec);
      check(ec, "cartPosture");
      if (request.check_current_pose) checkObservation(current, request.expected_current_pose);
      target = current;
      if (!request.use_current_pose) {
        target.trans = {request.pose.position.x, request.pose.position.y, request.pose.position.z};
        if (request.position_is_relative)
          for (std::size_t i = 0; i < 3; ++i) target.trans[i] += current.trans[i];
        if (!request.use_current_orientation) {
          const auto q = quaternion(request.pose);
          target.rpy = rokae::Utils::quaternionToEuler(q.w(), q.x(), q.y(), q.z());
        }
        target.hasElbow = request.use_elbow;
        target.elbow = request.elbow;
        target.confData.assign(request.conf_data.begin(), request.conf_data.end());
      }
      robot.setDefaultConfOpt(!target.confData.empty(), ec);
      check(ec, "setDefaultConfOpt(IK)");
      auto model = robot.model();
      target_joints = model.calcIk(target, selected, ec);
      check(ec, "calcIk(MoveJ)");
      const auto solved = model.calcFk(target_joints, selected, ec);
      check(ec, "calcFk(MoveJ verification)");
      double error_squared = 0.;
      for (std::size_t i = 0; i < 3; ++i) error_squared += std::pow(solved.trans[i] - target.trans[i], 2);
      require(std::sqrt(error_squared) <= 1e-4 && angleError(solved, target) <= 1e-3,
              "MoveJ IK solution failed FK verification");
      const bool enabled = robot.getSoftLimit(limits, ec);
      check(ec, "getSoftLimit");
      require(enabled, "controller soft limits are disabled");
      const auto joints = robot.jointPos(ec);
      check(ec, "jointPos");
      for (std::size_t i = 0; i < 7; ++i) {
        require(std::isfinite(joints[i]) && std::isfinite(target_joints[i]) &&
                std::isfinite(limits[i][0]) && std::isfinite(limits[i][1]) &&
                limits[i][0] + margin < limits[i][1] - margin,
                "invalid joint feedback or soft limits");
        require(target_joints[i] >= limits[i][0] + margin && target_joints[i] <= limits[i][1] - margin,
                "MoveJ target violates controller soft-limit margin");
        require(std::abs(target_joints[i] - joints[i]) <= max_delta,
                "MoveJ target exceeds max_cartesian_joint_delta_rad");
      }
      checkpoint();
      robot.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec);
      check(ec, "setMotionControlMode");
      robot.setOperateMode(rokae::OperateMode::automatic, ec);
      check(ec, "setOperateMode");
      robot.setDefaultConfOpt(!target.confData.empty(), ec);
      check(ec, "setDefaultConfOpt");
      command_owned = true;
      robot.moveReset(ec);
      check(ec, "moveReset");
      // Follow open_box: the AR 4000 mm/s ceiling avoids capping jointSpeed.
      rokae::MoveJCommand command(target, 4000., 0.);
      command.jointSpeed = request.joint_speed_scale;
      robot.moveAppend({command}, id, ec);
      check(ec, "moveAppend(MoveJ)");
      require(!id.empty(), "SDK returned an empty command ID");
      checkpoint();
      require(request.resource_valid_until_unix_s == 0. ||
              std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                  < request.resource_valid_until_unix_s, "collected ROS resource expired before MoveJ");
      robot.moveStart(ec);
      check(ec, "moveStart");
    });
    unsigned stable = 0;
    while (true) {
      checkpoint();
      bool done = false;
      hardware.withRobot([&](rokae::ArRobot &robot) {
        std::error_code ec;
        const auto event = robot.queryEventInfo(rokae::Event::moveExecution, ec);
        check(ec, "queryEventInfo");
        using namespace rokae::EventInfoKey::MoveExecution;
        const auto *event_id = field<std::string>(event, ID);
        const auto *reached = field<bool>(event, ReachTarget);
        const auto *error = field<std::error_code>(event, Error);
        if (event_id && *event_id == id && error) check(*error, "moveExecution");
        const auto state = robot.operationState(ec);
        check(ec, "operationState(feedback)");
        require(state == rokae::OperationState::moving || state == rokae::OperationState::idle,
                "controller entered an unexpected operation state");
        const auto power = robot.powerState(ec);
        check(ec, "powerState(feedback)");
        require(power == rokae::PowerState::on, "power is no longer on");
        const auto joints = robot.jointPos(ec);
        check(ec, "jointPos(feedback)");
        double joint_error = 0.;
        for (std::size_t i = 0; i < 7; ++i) {
          require(std::isfinite(joints[i]) && joints[i] >= limits[i][0] && joints[i] <= limits[i][1],
                  "joint feedback is invalid or crossed a soft limit");
          joint_error = std::max(joint_error, std::abs(joints[i] - target_joints[i]));
        }
        const auto pose = robot.cartPosture(rokae::CoordinateType::endInRef, ec);
        check(ec, "cartPosture(feedback)");
        double squared = 0.;
        for (std::size_t i = 0; i < 3; ++i) squared += std::pow(pose.trans[i] - target.trans[i], 2);
        done = event_id && *event_id == id && reached && *reached && state == rokae::OperationState::idle
            && joint_error <= request.joint_tolerance_rad
            && std::sqrt(squared) <= request.position_tolerance_m
            && angleError(pose, target) <= request.orientation_tolerance_rad;
      });
      stable = done ? stable + 1 : 0;
      if (stable >= request.stable_samples) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(request.poll_interval_ms));
    }
    command_owned = false;
    restore_tool();
    return "SDK MoveJ completed; command_id=" + id;
  } catch (const std::exception &error) {
    std::string reason = error.what();
    if (command_owned) {
      hardware.withRobot([&](rokae::ArRobot &robot) {
        std::error_code ec;
        robot.stop(ec);
        if (ec) reason += "; stop failed: " + ec.message();
        ec.clear();
        robot.moveReset(ec);
        if (ec) reason += "; reset failed: " + ec.message();
      });
    }
    // A stop acknowledgement can precede the idle feedback. Restore a changed
    // tool only after the controller confirms idle, as in the native backend.
    if (tool_changed) {
      for (int attempt = 0; attempt < 10; ++attempt) {
        const bool idle = hardware.withRobot([](rokae::ArRobot &robot) {
          std::error_code ec;
          return robot.operationState(ec) == rokae::OperationState::idle && !ec;
        });
        if (idle) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    try { restore_tool(); }
    catch (const std::exception &restore_error) { reason += "; " + std::string(restore_error.what()); }
    throw std::runtime_error(reason);
  }
}
}  // namespace rokae_driver::cartesian_movej
