/**
 * @file joint_drag_safe.cpp
 * @brief Safer AR joint-space hand-guiding demo with explicit tool load.
 *
 * Usage:
 *   joint_drag_safe <left|right> configured [duration_seconds]
 *
 *   joint_drag_safe <left|right> <mass_kg> <cog_x_m> <cog_y_m> <cog_z_m>
 *                   [duration_seconds]
 *
 * Example (replace all load values with measured values):
 *   joint_drag_safe right 1.20 0.0 0.0 0.08 30
 *
 * This program updates the active tool load in the robot controller. It uses
 * the SDK's normal joint-space drag enable path and deliberately does not pass
 * the final direct-drag=true argument.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "rokae/robot.h"
#include "rokae/utility.h"

using namespace rokae;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxToolMass = 5.0;          // kg; AR5 payload guard.
constexpr double kMaxCogDistance = 0.30;      // m.
constexpr double kStartupCheckSeconds = 2.0;  // hands off.
constexpr double kStartupMaxJointDrift = 2.0 * kPi / 180.0;
constexpr double kMaxJointSpeed = 1.00;        // rad/s.
constexpr double kJointLimitMargin = 12.0 * kPi / 180.0;

const std::array<double, 7> kLowerLimit = {
    -3.10669 + kJointLimitMargin,
    -2.0944 + kJointLimitMargin,
    -3.10669 + kJointLimitMargin,
    -1.0472 + kJointLimitMargin,
    -3.10669 + kJointLimitMargin,
    -1.0472 + kJointLimitMargin,
    -1.0472 + kJointLimitMargin,
};

const std::array<double, 7> kUpperLimit = {
    3.10669 - kJointLimitMargin,
    2.0944 - kJointLimitMargin,
    3.10669 - kJointLimitMargin,
    2.53073 - kJointLimitMargin,
    3.10669 - kJointLimitMargin,
    1.0472 - kJointLimitMargin,
    1.0472 - kJointLimitMargin,
};

volatile std::sig_atomic_t gStopRequested = 0;

void signalHandler(int) {
  gStopRequested = 1;
}

struct NetworkConfig {
  const char *name;
  const char *robotIp;
  const char *localIp;
};

struct RequestedLoad {
  bool useConfigured{false};
  double mass{0.0};
  std::array<double, 3> cog{};
  double duration{30.0};
};

bool checkError(const error_code &ec, const std::string &operation) {
  if (!ec) return true;
  std::cerr << operation << " failed: code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

NetworkConfig parseArm(const std::string &side) {
  if (side == "left") {
    return {"left", "192.168.1.160", "192.168.1.10"};
  }
  if (side == "right") {
    return {"right", "192.168.2.160", "192.168.2.10"};
  }
  throw std::invalid_argument("arm must be 'left' or 'right'");
}

double parseNumber(const char *text, const char *name) {
  std::size_t parsed = 0;
  const std::string valueText(text);
  const double value = std::stod(valueText, &parsed);
  if (parsed != valueText.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) +
                                " must be a finite number");
  }
  return value;
}

void validateDuration(double duration) {
  if (duration < 1.0 || duration > 300.0) {
    throw std::invalid_argument("duration must be between 1 and 300 seconds");
  }
}

void validateMassAndCog(double mass, const std::array<double, 3> &cog) {
  if (!std::isfinite(mass) || mass <= 0.0 || mass > kMaxToolMass) {
    throw std::invalid_argument(
        "mass must be greater than 0 and at most 5 kg");
  }
  for (const double value : cog) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("tool CoG values must be finite numbers");
    }
  }
  const double cogDistance =
      std::sqrt(cog[0] * cog[0] + cog[1] * cog[1] + cog[2] * cog[2]);
  if (cogDistance > kMaxCogDistance) {
    throw std::invalid_argument("tool CoG distance must not exceed 0.30 m");
  }
}

RequestedLoad parseLoad(int argc, char **argv) {
  RequestedLoad load;

  if (argc >= 3 && std::string(argv[2]) == "configured") {
    if (argc != 3 && argc != 4) {
      throw std::invalid_argument(
          "configured mode accepts only an optional duration");
    }
    load.useConfigured = true;
    if (argc == 4) load.duration = parseNumber(argv[3], "duration");
    validateDuration(load.duration);
    return load;
  }

  if (argc != 6 && argc != 7) {
    throw std::invalid_argument(
        "manual load mode requires mass and three CoG values");
  }
  load.mass = parseNumber(argv[2], "mass");
  load.cog = {parseNumber(argv[3], "cog_x"),
              parseNumber(argv[4], "cog_y"),
              parseNumber(argv[5], "cog_z")};
  if (argc == 7) load.duration = parseNumber(argv[6], "duration");

  validateMassAndCog(load.mass, load.cog);
  validateDuration(load.duration);
  return load;
}

void printUsage(const char *program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " <left|right> configured [duration_seconds]\n"
      << "  " << program
      << " <left|right> <mass_kg> <cog_x_m> <cog_y_m> <cog_z_m> "
         "[duration_seconds]"
      << std::endl;
}

void printLoad(const char *label, const Load &load) {
  std::cout << label << std::endl
            << "  mass = " << load.mass << " kg" << std::endl
            << "  cog = [" << load.cog[0] << ", " << load.cog[1] << ", "
            << load.cog[2] << "] m" << std::endl
            << "  inertia = [" << load.inertia[0] << ", "
            << load.inertia[1] << ", " << load.inertia[2]
            << "] kg*m^2" << std::endl;
}

void printJoints(const std::array<double, 7> &joints) {
  std::cout << "Current joints:" << std::endl;
  for (std::size_t i = 0; i < joints.size(); ++i) {
    std::cout << "  J" << i + 1 << " = "
              << Utils::radToDeg(joints[i]) << " deg" << std::endl;
  }
}

bool insideJointLimits(const std::array<double, 7> &joints) {
  for (std::size_t i = 0; i < joints.size(); ++i) {
    if (joints[i] < kLowerLimit[i] || joints[i] > kUpperLimit[i]) {
      std::cerr << "Joint safety boundary reached: J" << i + 1 << " = "
                << Utils::radToDeg(joints[i]) << " deg" << std::endl;
      return false;
    }
  }
  return true;
}

void printControllerErrors(ArRobot &robot) {
  error_code ec;
  const auto logs = robot.queryControllerLog(5, {LogInfo::error}, ec);
  if (!checkError(ec, "query controller error logs")) return;
  if (logs.empty()) return;
  std::cout << "Latest controller error logs:" << std::endl;
  for (const auto &log : logs) {
    std::cout << "[" << log.timestamp << "] " << log.content << std::endl;
  }
}

class DragGuard {
 public:
  explicit DragGuard(ArRobot &robot) : robot_(robot) {}

  ~DragGuard() {
    if (!enabled_) return;
    try {
      error_code ec;
      robot_.disableDrag(ec);
    } catch (const std::exception &) {
    }
  }

  void setEnabled(bool enabled) { enabled_ = enabled; }

 private:
  ArRobot &robot_;
  bool enabled_{false};
};

bool checkStartupDrift(ArRobot &robot,
                       const std::array<double, 7> &startJoints) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(
                            static_cast<int>(kStartupCheckSeconds * 1000));
  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    error_code ec;
    const auto joints = robot.jointPos(ec);
    if (!checkError(ec, "read joints during startup drift check")) {
      return false;
    }
    for (std::size_t i = 0; i < joints.size(); ++i) {
      if (std::abs(joints[i] - startJoints[i]) > kStartupMaxJointDrift) {
        std::cerr << "Startup drift detected on J" << i + 1 << ": "
                  << Utils::radToDeg(joints[i] - startJoints[i])
                  << " deg" << std::endl;
        return false;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return !gStopRequested;
}

bool monitorDrag(ArRobot &robot, double duration) {
  error_code ec;
  auto previousJoints = robot.jointPos(ec);
  if (!checkError(ec, "read initial drag joints")) return false;
  auto previousTime = std::chrono::steady_clock::now();
  const auto deadline = previousTime + std::chrono::milliseconds(
                                         static_cast<int>(duration * 1000));

  while (!gStopRequested && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto now = std::chrono::steady_clock::now();
    const auto joints = robot.jointPos(ec);
    if (!checkError(ec, "read joints during drag")) return false;
    if (!insideJointLimits(joints)) return false;

    const double dt =
        std::chrono::duration<double>(now - previousTime).count();
    for (std::size_t i = 0; i < joints.size(); ++i) {
      const double speed = std::abs(joints[i] - previousJoints[i]) / dt;
      if (speed > kMaxJointSpeed) {
        std::cerr << "Joint speed guard triggered: J" << i + 1 << " = "
                  << speed << " rad/s" << std::endl;
        return false;
      }
    }
    previousJoints = joints;
    previousTime = now;
  }
  return !gStopRequested;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    printUsage(argv[0]);
    return 2;
  }

  NetworkConfig config{};
  RequestedLoad requested{};
  try {
    config = parseArm(argv[1]);
    requested = parseLoad(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    printUsage(argv[0]);
    return 2;
  }

  std::signal(SIGINT, signalHandler);
  std::cout << std::fixed << std::setprecision(6);

  try {
    std::cout << "Connecting " << config.name << " arm: robot="
              << config.robotIp << ", local=" << config.localIp << std::endl;
    ArRobot robot(config.robotIp, config.localIp);
    DragGuard dragGuard(robot);
    error_code ec;

    const auto initialState = robot.operationState(ec);
    if (!checkError(ec, "read operation state")) return 1;
    if (initialState == OperationState::drag) {
      robot.disableDrag(ec);
      if (!checkError(ec, "disable existing drag mode")) return 1;
      std::cout << "Existing drag mode was disabled. Run the program again."
                << std::endl;
      return 0;
    }

    robot.setMotionControlMode(MotionControlMode::NrtCommand, ec);
    if (!checkError(ec, "switch to NrtCommand")) return 1;
    robot.setOperateMode(OperateMode::manual, ec);
    if (!checkError(ec, "set manual mode")) return 1;
    robot.setPowerState(false, ec);
    if (!checkError(ec, "power off")) return 1;

    Toolset toolset = robot.toolset(ec);
    if (!checkError(ec, "read current toolset")) return 1;
    printLoad("Current controller tool load:", toolset.load);

    std::string confirmation;
    if (requested.useConfigured) {
      try {
        validateMassAndCog(toolset.load.mass, toolset.load.cog);
      } catch (const std::exception &e) {
        std::cerr << "Configured tool load is unusable: " << e.what()
                  << std::endl
                  << "Run the controller load-identification procedure or "
                     "use manual load arguments before enabling drag."
                  << std::endl;
        return 1;
      }
      std::cout << "Using the existing controller tool load; no load "
                   "parameters were changed."
                << std::endl;
    } else {
      Load requestedLoad = toolset.load;
      requestedLoad.mass = requested.mass;
      requestedLoad.cog = requested.cog;
      printLoad("Requested tool load (existing inertia is preserved):",
                requestedLoad);

      std::cout << "This writes the active tool load to the controller. Type "
                   "APPLY to confirm the measured values: "
                << std::flush;
      std::getline(std::cin, confirmation);
      if (confirmation != "APPLY") {
        std::cout << "Cancelled; tool load and drag mode were not changed."
                  << std::endl;
        return 0;
      }

      toolset.load = requestedLoad;
      robot.setToolset(toolset, ec);
      if (!checkError(ec, "set tool load")) {
        printControllerErrors(robot);
        return 1;
      }
      const auto storedToolset = robot.toolset(ec);
      if (!checkError(ec, "verify stored tool load")) return 1;
      printLoad("Controller tool load after update:", storedToolset.load);
    }

    const auto startJoints = robot.jointPos(ec);
    if (!checkError(ec, "read current joints")) return 1;
    printJoints(startJoints);
    if (!insideJointLimits(startJoints)) return 1;

    std::cout << "Confirm the displayed load matches the physical tool. "
                 "Support the arm and keep the E-stop ready. Press Enter to "
                 "enable joint-space drag. Do NOT press the drag button "
                 "during the first 2 s hands-off drift check."
              << std::endl;
    std::getline(std::cin, confirmation);
    if (gStopRequested) return 0;

    // Official-style call: do not pass direct-drag=true.
    robot.enableDrag(DragParameter::jointSpace,
                     DragParameter::freely, ec);
    if (!checkError(ec, "enable joint-space drag")) {
      printControllerErrors(robot);
      return 1;
    }
    dragGuard.setEnabled(true);

    std::cout << "Joint-space drag mode enabled. KEEP HANDS OFF for 2 s..."
              << std::endl;
    if (!checkStartupDrift(robot, startJoints)) {
      std::cerr << "Startup drift check failed; disabling drag. Verify tool "
                   "load, force-sensor zero and gravity parameters."
                << std::endl;
      robot.disableDrag(ec);
      if (!ec) dragGuard.setEnabled(false);
      checkError(ec, "disable drag after startup drift");
      printControllerErrors(robot);
      return 1;
    }

    std::cout << "Startup drift check passed. Use the controller's normal "
                 "drag enable button and move slowly. Monitoring for "
              << requested.duration << " s; press Ctrl+C to stop."
              << std::endl;
    const bool monitorOk = monitorDrag(robot, requested.duration);

    robot.disableDrag(ec);
    if (!checkError(ec, "disable joint-space drag")) {
      printControllerErrors(robot);
      return 1;
    }
    dragGuard.setEnabled(false);

    if (!monitorOk) {
      std::cerr << "Drag stopped by Ctrl+C or a safety monitor."
                << std::endl;
      return gStopRequested ? 0 : 1;
    }

    std::cout << "Joint-space drag completed and disabled." << std::endl;
    const auto finalJoints = robot.jointPos(ec);
    if (!checkError(ec, "read final joints")) return 1;
    printJoints(finalJoints);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Joint drag program exception: " << e.what() << std::endl;
    return 1;
  }
}
