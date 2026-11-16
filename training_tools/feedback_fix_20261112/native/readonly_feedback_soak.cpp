// Diagnose SDK FIFO timing without changing mode, power, targets, or IO.
#include "rokae/robot.h"
#include "rokae/data_types.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
  using Clock = std::chrono::steady_clock;
  using Json = nlohmann::json;
  using namespace std::chrono;
  const int ticks = argc == 2 ? std::stoi(argv[1]) : 6000;
  if (ticks < 100 || ticks > 18000) return 2;
  std::array<std::unique_ptr<rokae::ArRobot>, 2> robots;
  std::array<bool, 2> receiving{};
  Json result = {{"mode", "read_only_dual_arm_SDK_telemetry"}, {"motion_operations", 0},
                 {"slow_reads", Json::array()}, {"arms", Json::array()}};
  std::array<std::vector<double>, 2> durations;
  std::array<int, 2> over5{}, over10{}, over20{};
  std::array<std::size_t, 2> maxFrames{};
  bool passed = false;
  try {
    for (int side=0; side<2; ++side) {
      robots[side] = std::make_unique<rokae::ArRobot>(
          side ? "192.168.2.160" : "192.168.1.160", side ? "192.168.2.10" : "192.168.1.10");
      auto &robot = *robots[side];
      std::error_code ec;
      Json arm = {{"side", side ? "right" : "left"}};
      arm["initial_power"] = static_cast<int>(robot.powerState(ec));
      if (ec) throw std::runtime_error(ec.message());
      arm["initial_joints"] = robot.jointPos(ec);
      if (ec) throw std::runtime_error(ec.message());
      if (robot.operationState(ec) != rokae::OperationState::idle || ec)
        throw std::runtime_error("Read-only soak requires idle arms");
      result["arms"].push_back(arm);
      robot.startReceiveRobotState(milliseconds(1), {rokae::RtSupportedFields::jointPos_m});
      receiving[side] = true;
    }
    auto next = Clock::now();
    for (int tick=0; tick<ticks; ++tick) {
      for (int side=0; side<2; ++side) {
        const auto start = Clock::now();
        std::size_t frames = 0;
        double maxCall = 0, crossed5 = -1;
        while (true) {
          const auto before = Clock::now();
          const double elapsed = duration<double,std::milli>(before-start).count();
          if (elapsed >= 5 && crossed5 < 0) crossed5 = frames;
          if (frames >= 4096 || elapsed >= 50)
            throw std::runtime_error("Diagnostic drain exceeded 50 ms / 4096 frames");
          const auto bytes = robots[side]->updateRobotState(Clock::duration::zero());
          maxCall = std::max(maxCall, duration<double,std::milli>(Clock::now()-before).count());
          if (!bytes) break;
          ++frames;
        }
        const auto elapsed = duration<double,std::milli>(Clock::now()-start).count();
        std::array<double,7> joints;
        if (robots[side]->getStateData(rokae::RtSupportedFields::jointPos_m,joints) != 0)
          throw std::runtime_error("Missing measured joint feedback");
        for (double q:joints) if (!std::isfinite(q)) throw std::runtime_error("Nonfinite feedback");
        durations[side].push_back(elapsed);
        maxFrames[side] = std::max(maxFrames[side],frames);
        over5[side] += crossed5 >= 0;
        over10[side] += elapsed >= 10;
        over20[side] += elapsed >= 20;
        if (elapsed >= 5 || crossed5 >= 0)
          result["slow_reads"].push_back({{"tick",tick},{"side",side ? "right" : "left"},
              {"elapsed_ms",elapsed},{"frames",frames},{"frames_at_5ms",crossed5},{"max_receive_call_ms",maxCall}});
      }
      next += milliseconds(10);
      if (next < Clock::now()) next = Clock::now() + milliseconds(10);
      std::this_thread::sleep_until(next);
    }
    passed = true;
  } catch (const std::exception &e) { result["error"] = e.what(); }
  for (int side=0; side<2; ++side) {
    if (!robots[side]) continue;
    if (receiving[side]) { try { robots[side]->stopReceiveRobotState(); } catch (...) { passed=false; } }
    std::error_code ec;
    auto &arm = result["arms"][side];
    arm["final_power"] = static_cast<int>(robots[side]->powerState(ec));
    if (ec || arm["final_power"] != arm["initial_power"]) passed = false;
    arm["final_joints"] = robots[side]->jointPos(ec);
    if (ec) passed = false;
    arm["samples"] = durations[side].size();
    arm["old_5ms_budget_exceeded"] = over5[side];
    arm["over_10ms"] = over10[side]; arm["over_20ms"] = over20[side];
    arm["maximum_frames"] = maxFrames[side];
    if (!durations[side].empty()) {
      auto values = durations[side]; std::sort(values.begin(),values.end());
      arm["median_ms"] = values[values.size()/2];
      arm["p99_ms"] = values[values.size()*99/100];
      arm["maximum_ms"] = values.back();
    }
  }
  result["passed"] = passed;
  std::cout << result.dump(2) << std::endl;
  return passed ? 0 : 1;
}
