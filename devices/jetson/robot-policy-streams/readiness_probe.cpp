// Read-only controller inspection. No power/mode/motion/output setter is used.
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <rokae/robot.h>

void check(const std::error_code &error) {
  if (error) throw std::runtime_error(error.message());
}

const char *power_name(rokae::PowerState value) {
  switch (value) {
    case rokae::PowerState::on: return "on";
    case rokae::PowerState::off: return "off";
    case rokae::PowerState::estop: return "estop";
    case rokae::PowerState::gstop: return "gstop";
    default: return "other";
  }
}

int main() {
  std::cout << std::setprecision(12) << std::boolalpha;
  for (int side = 0; side < 2; ++side) {
    try {
      rokae::ArRobot robot(side == 0 ? "192.168.1.160" : "192.168.2.160",
                           side == 0 ? "192.168.1.10" : "192.168.2.10");
      std::error_code error;
      const auto power = robot.powerState(error); check(error);
      const auto state = robot.operationState(error); check(error);
      const auto joints = robot.jointPos(error); check(error);
      std::array<double[2], 7> limits{};
      const auto enabled = robot.getSoftLimit(limits, error); check(error);
      std::cout << "{\"side\":\"" << (side == 0 ? "left" : "right")
                << "\",\"power\":\"" << power_name(power)
                << "\",\"idle\":" << (state == rokae::OperationState::idle)
                << ",\"operation_state\":" << static_cast<int>(state)
                << ",\"soft_limits_enabled\":" << enabled << ",\"joints\":[";
      for (int i = 0; i < 7; ++i) std::cout << (i ? "," : "") << joints[i];
      std::cout << "],\"limits\":[";
      for (int i = 0; i < 7; ++i)
        std::cout << (i ? "," : "") << "[" << limits[i][0] << "," << limits[i][1] << "]";
      std::cout << "]}" << std::endl;
    } catch (const std::exception &error) {
      std::cerr << (side == 0 ? "left" : "right") << ": " << error.what() << std::endl;
      return 1;
    }
  }
}
