// Read-only SDK telemetry probe. Never changes motion mode, power, targets or IO.
#include "latest_feedback.hpp"
#include "rokae/robot.h"
#include "rokae/data_types.h"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

int main(int argc, char **argv) {
  if (argc != 2 || (std::string(argv[1]) != "left" && std::string(argv[1]) != "right")) return 2;
  const bool left = std::string(argv[1]) == "left";
  using Json = nlohmann::json;
  using Clock = latest_feedback::Clock;
  rokae::ArRobot robot(left ? "192.168.1.160" : "192.168.2.160",
                       left ? "192.168.1.10" : "192.168.2.10");
  bool receiving = false;
  Json result = {{"mode","read_only_SDK_telemetry"},{"side",argv[1]},
                 {"motion_operations",0},{"samples",Json::array()}};
  try {
    std::error_code ec;
    result["initial_power_state"] = static_cast<int>(robot.powerState(ec));
    if (ec) throw std::runtime_error(ec.message());
    result["initial_NRT_joints"] = robot.jointPos(ec);
    if (ec) throw std::runtime_error(ec.message());
    robot.startReceiveRobotState(std::chrono::milliseconds(1),{rokae::RtSupportedFields::jointPos_m});
    receiving = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const auto initial = latest_feedback::drain(
        [&] { return robot.updateRobotState(Clock::duration::zero()); },
        [] { return Clock::now(); },std::chrono::milliseconds(50));
    result["startup_frames_drained"] = initial.frames;
    auto next = Clock::now();
    for (int tick=0; tick<100; ++tick) {
      next += std::chrono::milliseconds(10);
      std::this_thread::sleep_until(next);
      const auto drain = latest_feedback::drain(
          [&] { return robot.updateRobotState(Clock::duration::zero()); },
          [] { return Clock::now(); },std::chrono::milliseconds(5));
      std::array<double,7> joints;
      if (robot.getStateData(rokae::RtSupportedFields::jointPos_m,joints) != 0)
        throw std::runtime_error("SDK joint feedback unavailable");
      for (double q:joints) if (!std::isfinite(q)) throw std::runtime_error("nonfinite SDK joint feedback");
      result["samples"].push_back({{"frames_drained",drain.frames},
          {"historical_frames_discarded",drain.discarded()},
          {"drain_duration_ms",drain.duration_ms},{"joints",joints}});
    }
    robot.stopReceiveRobotState(); receiving=false;
    result["final_power_state"] = static_cast<int>(robot.powerState(ec));
    if (ec) throw std::runtime_error(ec.message());
    result["final_NRT_joints"] = robot.jointPos(ec);
    if (ec) throw std::runtime_error(ec.message());
    result["passed"] = result["initial_power_state"] == result["final_power_state"];
    std::cout << result.dump(2) << '\n';
    return result["passed"].get<bool>() ? 0 : 1;
  } catch (const std::exception &error) {
    if (receiving) robot.stopReceiveRobotState();
    result["passed"]=false; result["error"]=error.what();
    std::cout << result.dump(2) << '\n'; return 1;
  }
}
