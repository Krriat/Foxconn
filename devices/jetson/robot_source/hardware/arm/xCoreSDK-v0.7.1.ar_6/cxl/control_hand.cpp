/**
 * @file control_hand.cpp
 * @brief Control a six-motor Linker Hand O6/L6 through the AR end CAN bus.
 *
 * Usage:
 *   control_hand <left|right> open
 *   control_hand <left|right> half
 *   control_hand <left|right> close
 *   control_hand <left|right> position <0-255>
 *   control_hand <left|right> motors <m1> ... <m6>
 *   control_hand <left|right> joints <m1> ... <m6>  (legacy alias)
 *   control_hand <left|right> speed <0-255>
 *   control_hand <left|right> pressure
 *   control_hand <left|right> probe
 *   control_hand <left|right> demo
 */

#include <array>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "rokae/robot.h"

namespace {

// Match the timing used by the vendor example. The hand may not place its
// response in the controller receive queue immediately after a write.
constexpr int kReceiveTimeoutMs = 200;
constexpr int kReceiveAttempts = 3;
constexpr int kReplyDelayMs = 1000;
constexpr int kDemoStepDelayMs = 1000;

struct ArmConfig {
  const char *name;
  const char *robotIp;
  const char *localIp;
  int canId;
};

void setupConsoleUtf8() {
#ifdef _WIN32
  SetConsoleOutputCP(65001);
  SetConsoleCP(65001);
#endif
}

void printUsage(const char *program) {
  std::cout
      << "用法:\n"
      << "  " << program << " <left|right> open\n"
      << "  " << program << " <left|right> half\n"
      << "  " << program << " <left|right> close\n"
      << "  " << program << " <left|right> position <0-255>\n"
      << "  " << program << " <left|right> motors <m1> ... <m6>\n"
      << "  " << program << " <left|right> joints <m1> ... <m6>\n"
      << "  " << program << " <left|right> speed <0-255>\n"
      << "  " << program << " <left|right> pressure\n"
      << "  " << program << " <left|right> probe\n"
      << "  " << program << " <left|right> demo\n\n"
      << "位置数值越大越张开，越小越靠近掌心。\n"
      << "M1-M6依次为拇指弯曲、拇指侧摆、食指、中指、"
         "无名指和小指弯曲。\n"
      << "默认CAN ID: 左手0x28，右手0x27。" << std::endl;
}

bool selectArm(const std::string &side, ArmConfig &config) {
  if (side == "left") {
    config = {"左臂/左手", "192.168.1.160", "192.168.1.10", 0x28};
    return true;
  }
  if (side == "right") {
    config = {"右臂/右手", "192.168.2.160", "192.168.2.10", 0x27};
    return true;
  }
  return false;
}

bool parseByte(const char *text, std::uint8_t &value) {
  if (text == nullptr || *text == '\0') return false;

  errno = 0;
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
      parsed > 255) {
    return false;
  }
  value = static_cast<std::uint8_t>(parsed);
  return true;
}

void printReceivedFrame(const char *tag, const rokae::CANFrame &frame) {
  std::cout << tag << " 返回: id=0x" << std::hex << frame.frame_id
            << std::dec << ", format=" << frame.frame_format
            << ", type=" << frame.frame_type
            << ", length=" << frame.frame_valid_length << ", data=[";
  for (std::size_t i = 0; i < frame.data.size(); ++i) {
    if (i != 0) std::cout << ' ';
    std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(frame.data[i]) << std::dec;
  }
  std::cout << std::setfill(' ') << "]" << std::endl;
}

bool sendAndReceive(rokae::ArRobot &robot, int canId,
                    const std::vector<std::uint8_t> &payload,
                    const char *tag) {
  rokae::CANFrame frame;
  frame.frame_id = canId;
  frame.frame_format = static_cast<int>(rokae::CANFormat::STANDARD);
  frame.frame_type = static_cast<int>(rokae::CANType::CAN);
  frame.frame_valid_length = static_cast<int>(payload.size());
  frame.data = payload;

  std::error_code ec;
  const std::vector<rokae::CANFrame> frames{frame};
  robot.CANSendData("uint8", frames, ec);
  if (ec) {
    std::cerr << tag << " 发送失败: code=" << ec.value()
              << ", message=" << ec.message() << std::endl;
    return false;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(kReplyDelayMs));

  rokae::CANFrame received;
  for (int attempt = 1; attempt <= kReceiveAttempts; ++attempt) {
    ec.clear();
    robot.CANReceiveData(kReceiveTimeoutMs, "uint8", received, ec);
    if (!ec) {
      const bool expectedId =
          received.frame_id == canId || received.frame_id == canId + 8;
      const bool expectedCommand =
          !received.data.empty() && !payload.empty() &&
          received.data.front() == payload.front();
      if (!expectedId || !expectedCommand) {
        std::cerr << tag << " 忽略无关回包: id=0x" << std::hex
                  << received.frame_id << std::dec << std::endl;
        continue;
      }
      printReceivedFrame(tag, received);
      return true;
    }

    if (attempt != kReceiveAttempts) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  std::cerr << tag << " 回包超时: 发送后已等待" << kReplyDelayMs
            << " ms，并接收重试" << kReceiveAttempts
            << "次；最后错误 code=" << ec.value()
            << ", message=" << ec.message() << std::endl;
  return false;
}

bool setSpeed(rokae::ArRobot &robot, int canId, std::uint8_t speed) {
  return sendAndReceive(robot, canId,
                        {0x05, speed, speed, speed, speed, speed, speed},
                        "设置M1-M6速度");
}

bool setPositions(rokae::ArRobot &robot, int canId,
                  const std::array<std::uint8_t, 6> &positions) {
  return sendAndReceive(robot, canId,
                        {0x01, positions[0], positions[1], positions[2],
                         positions[3], positions[4], positions[5]},
                        "设置M1-M6位置");
}

bool setUniformPosition(rokae::ArRobot &robot, int canId,
                        std::uint8_t position) {
  std::array<std::uint8_t, 6> positions{};
  positions.fill(position);
  return setPositions(robot, canId, positions);
}

bool readPressure(rokae::ArRobot &robot, int canId) {
  // The vendor example only documents the request code. Keep the response raw
  // until the hand protocol defines the byte-to-pressure conversion.
  return sendAndReceive(robot, canId, {0x20}, "读取五指法向压力");
}

bool probeCanIds(rokae::ArRobot &robot) {
  bool found = false;
  const std::array<int, 2> canIds = {0x27, 0x28};
  for (const int canId : canIds) {
    std::cout << "探测CAN ID 0x" << std::hex << canId << std::dec
              << "（只发送压力读取命令，不驱动手指）" << std::endl;
    if (readPressure(robot, canId)) {
      std::cout << "CAN ID 0x" << std::hex << canId << std::dec
                << " 有回包。" << std::endl;
      found = true;
    }
  }
  if (!found) {
    std::cerr << "CAN ID 0x27和0x28均无回包。" << std::endl;
  }
  return found;
}

bool runDemo(rokae::ArRobot &robot, int canId) {
  std::cout << "开始演示：设置速度 -> 张开 -> 半握 -> 闭合 -> 压力 -> 张开"
            << std::endl;

  if (!setSpeed(robot, canId, 0x60)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(kDemoStepDelayMs));

  if (!setUniformPosition(robot, canId, 0xff)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(kDemoStepDelayMs));

  if (!setUniformPosition(robot, canId, 0xa0)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(kDemoStepDelayMs));

  const std::array<std::uint8_t, 6> closed = {
      0x45, 0x45, 0x45, 0x45, 0x45, 0x45,
  };
  if (!setPositions(robot, canId, closed)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(kDemoStepDelayMs));

  if (!readPressure(robot, canId)) return false;
  std::this_thread::sleep_for(std::chrono::milliseconds(kDemoStepDelayMs));

  return setUniformPosition(robot, canId, 0xff);
}

}  // namespace

int main(int argc, char *argv[]) {
  setupConsoleUtf8();

  if (argc < 3) {
    printUsage(argv[0]);
    return 2;
  }

  ArmConfig config{};
  if (!selectArm(argv[1], config)) {
    std::cerr << "机械臂参数必须是 left 或 right。" << std::endl;
    printUsage(argv[0]);
    return 2;
  }

  const std::string command = argv[2];
  std::uint8_t value = 0;
  std::array<std::uint8_t, 6> positions{};

  if ((command == "position" || command == "speed") &&
      (argc != 4 || !parseByte(argv[3], value))) {
    std::cerr << command << " 需要一个0-255的数值。" << std::endl;
    return 2;
  }
  if (command == "motors" || command == "joints") {
    if (argc != 9) {
      std::cerr << command << " 需要依次给出M1-M6共6个位置值。"
                << std::endl;
      return 2;
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
      if (!parseByte(argv[3 + static_cast<int>(i)], positions[i])) {
        std::cerr << "M" << i + 1 << "位置无效，应为0-255。"
                  << std::endl;
        return 2;
      }
    }
  }

  try {
    std::cout << "连接" << config.name << ": robot=" << config.robotIp
              << ", local=" << config.localIp << ", CAN ID=0x" << std::hex
              << config.canId << std::dec << std::endl;

    rokae::ArRobot robot(config.robotIp, config.localIp);
    std::error_code infoEc;
    const auto info = robot.robotInfo(infoEc);
    if (!infoEc) {
      std::cout << "机型=" << info.type << "，控制器版本=" << info.version
                << "，SDK版本=" << robot.sdkVersion() << std::endl;
    } else {
      std::cerr << "读取机器人版本失败: code=" << infoEc.value()
                << ", message=" << infoEc.message() << std::endl;
    }
    bool success = false;

    if (command == "open") {
      success = setUniformPosition(robot, config.canId, 0xff);
    } else if (command == "half") {
      success = setUniformPosition(robot, config.canId, 0xa0);
    } else if (command == "close") {
      const std::array<std::uint8_t, 6> closed = {
          0x45, 0x45, 0x45, 0x45, 0x45, 0x45,
      };
      success = setPositions(robot, config.canId, closed);
    } else if (command == "position") {
      success = setUniformPosition(robot, config.canId, value);
    } else if (command == "motors" || command == "joints") {
      success = setPositions(robot, config.canId, positions);
    } else if (command == "speed") {
      success = setSpeed(robot, config.canId, value);
    } else if (command == "pressure") {
      success = readPressure(robot, config.canId);
    } else if (command == "probe") {
      success = probeCanIds(robot);
    } else if (command == "demo") {
      success = runDemo(robot, config.canId);
    } else {
      std::cerr << "未知命令: " << command << std::endl;
      printUsage(argv[0]);
      return 2;
    }

    if (!success) return 1;
    std::cout << "命令执行完成。" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "连接或CAN控制异常: " << e.what() << std::endl;
    return 1;
  }
}
