#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "rokae/robot.h"

namespace {

struct ArmConfig {
  const char *name;
  const char *robotIp;
  const char *localIp;
};

const ArmConfig &armConfig(const std::string &side) {
  static const ArmConfig left{"left", "192.168.1.160", "192.168.1.10"};
  static const ArmConfig right{"right", "192.168.2.160", "192.168.2.10"};
  if (side == "left") return left;
  if (side == "right") return right;
  throw std::invalid_argument("arm must be left or right");
}

void checked(const std::error_code &ec, const char *operation) {
  if (ec) {
    throw std::runtime_error(std::string(operation) + " failed: code=" +
                             std::to_string(ec.value()) + ", message=" +
                             ec.message());
  }
}

const char *stateName(rokae::OperationState state) {
  switch (state) {
    case rokae::OperationState::idle: return "idle";
    case rokae::OperationState::jog: return "jog";
    case rokae::OperationState::rtControlling: return "rtControlling";
    case rokae::OperationState::drag: return "drag";
    case rokae::OperationState::rlProgram: return "rlProgram";
    case rokae::OperationState::demo: return "demo";
    case rokae::OperationState::dynamicIdentify: return "dynamicIdentify";
    case rokae::OperationState::frictionIdentify: return "frictionIdentify";
    case rokae::OperationState::loadIdentify: return "loadIdentify";
    case rokae::OperationState::moving: return "moving";
    case rokae::OperationState::jogging: return "jogging";
    default: return "unknown";
  }
}

const char *modeName(rokae::OperateMode mode) {
  switch (mode) {
    case rokae::OperateMode::manual: return "manual";
    case rokae::OperateMode::automatic: return "automatic";
    default: return "unknown";
  }
}

const char *powerName(rokae::PowerState power) {
  switch (power) {
    case rokae::PowerState::on: return "on";
    case rokae::PowerState::off: return "off";
    case rokae::PowerState::estop: return "estop";
    case rokae::PowerState::gstop: return "gstop";
    default: return "unknown";
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <left|right> [--execute]\n";
    return 2;
  }

  const bool execute = argc == 3 && std::string(argv[2]) == "--execute";
  if (argc == 3 && !execute) {
    std::cerr << "The only supported action is --execute\n";
    return 2;
  }

  try {
    const auto &config = armConfig(argv[1]);
    std::cout << "Connecting " << config.name << " controller: robot="
              << config.robotIp << ", local=" << config.localIp << '\n';
    rokae::ArRobot robot(config.robotIp, config.localIp);
    std::error_code ec;

    const auto state = robot.operationState(ec);
    checked(ec, "operationState");
    const auto mode = robot.operateMode(ec);
    checked(ec, "operateMode");
    const auto power = robot.powerState(ec);
    checked(ec, "powerState");
    const auto base = robot.baseFrame(ec);
    checked(ec, "baseFrame");

    std::cout << "state=" << stateName(state) << ", mode=" << modeName(mode)
              << ", power=" << powerName(power) << "\nbaseFrame=[";
    for (std::size_t i = 0; i < base.size(); ++i) {
      if (i) std::cout << ", ";
      std::cout << base[i];
    }
    std::cout << "]\n";

    if (!execute) {
      std::cout << "Status only; controller was not restarted.\n";
      return 0;
    }
    if (state != rokae::OperationState::idle) {
      throw std::runtime_error("refusing reboot: robot is not idle");
    }
    if (mode == rokae::OperateMode::automatic) {
      std::cout << "Switching controller to manual mode for reboot...\n";
      robot.setOperateMode(rokae::OperateMode::manual, ec);
      checked(ec, "setOperateMode(manual)");
    } else if (mode != rokae::OperateMode::manual) {
      throw std::runtime_error("refusing reboot: operation mode is unknown");
    }
    if (power == rokae::PowerState::on) {
      std::cout << "Powering motors off for reboot...\n";
      robot.setPowerState(false, ec);
      checked(ec, "setPowerState(false)");
    } else if (power != rokae::PowerState::off) {
      throw std::runtime_error("refusing reboot: power state is estop/gstop/unknown");
    }

    const auto readyState = robot.operationState(ec);
    checked(ec, "operationState before reboot");
    const auto readyMode = robot.operateMode(ec);
    checked(ec, "operateMode before reboot");
    const auto readyPower = robot.powerState(ec);
    checked(ec, "powerState before reboot");
    if (readyState != rokae::OperationState::idle ||
        readyMode != rokae::OperateMode::manual ||
        readyPower != rokae::PowerState::off) {
      throw std::runtime_error("refusing reboot: final state is not idle/manual/power-off");
    }

    std::cout << "Requesting controller reboot...\n";
    robot.rebootSystem(ec);
    checked(ec, "rebootSystem");
    std::cout << "Controller accepted the reboot request.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
