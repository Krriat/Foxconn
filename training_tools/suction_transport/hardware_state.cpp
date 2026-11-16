// Read-only telemetry: no motion, mode, power or output-writing API is used.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <m050/m050.h>
#include <rokae/robot.h>

using SteadyClock = std::chrono::steady_clock;

struct TelemetryStatus {
  SteadyClock::time_point last_success{};
  std::uint64_t samples{0};
  std::uint64_t failures{0};
  std::string error{"Waiting for first hardware sample"};
  std::string values;
};

class HardwareState : public rclcpp::Node {
 public:
  HardwareState() : Node("robot_hardware_state") {
    joint_hz_ = declare_parameter<double>("joint_hz", 20.0);
    io_hz_ = declare_parameter<double>("io_hz", 10.0);
    if (!std::isfinite(joint_hz_) || joint_hz_ <= 0 || joint_hz_ > 100 ||
        !std::isfinite(io_hz_) || io_hz_ <= 0 || io_hz_ > 50) {
      throw std::invalid_argument("Invalid telemetry rates");
    }
    arm_ips_[0] = declare_parameter<std::string>("left.robot_ip", "192.168.1.160");
    arm_ips_[1] = declare_parameter<std::string>("right.robot_ip", "192.168.2.160");
    local_ips_[0] = declare_parameter<std::string>("left.local_ip", "192.168.1.10");
    local_ips_[1] = declare_parameter<std::string>("right.local_ip", "192.168.2.10");
    io_host_ = declare_parameter<std::string>("suction.host", "172.16.13.167");
    io_port_ = declare_parameter<int>("suction.port", 20108);
    io_unit_ = declare_parameter<int>("suction.unit_id", 1);
    io_timeout_ = declare_parameter<int>("suction.timeout_ms", 1000);
    if (io_port_ < 1 || io_port_ > 65535 || io_unit_ < 1 || io_unit_ > 247 ||
        io_timeout_ < 1 || io_timeout_ > 5000) {
      throw std::invalid_argument("Invalid suction connection parameters");
    }
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable().durability_volatile();
    for (int side = 0; side < 2; ++side) {
      const auto name = side == 0 ? "left" : "right";
      joint_pubs_[side] = create_publisher<sensor_msgs::msg::JointState>(
          "/robot/upperlimb/joint_states/" + std::string(name) + "_arm", qos);
      suction_pubs_[side] = create_publisher<std_msgs::msg::Bool>(
          "/robot/suction/" + std::string(name) + "/state", qos);
    }
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/robot/hardware_state/diagnostics", qos);
    diagnostics_timer_ = create_wall_timer(std::chrono::seconds(1), [this] { diagnostics(); });
    threads_.emplace_back([this] { arm_worker(0); });
    threads_.emplace_back([this] { arm_worker(1); });
    threads_.emplace_back([this] { suction_worker(); });
  }

  ~HardwareState() override {
    stopping_ = true;
    condition_.notify_all();
    for (auto &thread : threads_) if (thread.joinable()) thread.join();
  }

 private:
  bool running() const { return !stopping_ && rclcpp::ok(); }

  void wait(double seconds) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    condition_.wait_for(lock, std::chrono::duration<double>(std::max(0.0, seconds)),
                        [this] { return !running(); });
  }

  void success(int index, const std::string &values) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    auto &state = status_[index];
    state.last_success = SteadyClock::now();
    ++state.samples;
    state.error.clear();
    state.values = values;
  }

  void failure(int index, const std::string &message) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    ++status_[index].failures;
    status_[index].error = message;
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "%s", message.c_str());
  }

  void arm_worker(int side) {
    const std::string name = side == 0 ? "left" : "right";
    std::unique_ptr<rokae::ArRobot> robot;
    while (running()) {
      try {
        if (!robot) {
          // Match the existing driver's serialized connection creation: the
          // vendor SDK initializes process-global resources during construction.
          std::lock_guard<std::mutex> lock(connect_mutex_);
          robot = std::make_unique<rokae::ArRobot>(arm_ips_[side], local_ips_[side]);
          RCLCPP_INFO(get_logger(), "%s: connected for joint feedback", name.c_str());
        }
        const auto start = SteadyClock::now();
        std::error_code error;
        const auto joints = robot->jointPos(error);
        if (error) {
          failure(side, name + " jointPos: " + error.message());
          wait(0.5);
          continue;
        }
        if (joints.size() != 7 || !std::all_of(joints.begin(), joints.end(),
                                             [](double v) { return std::isfinite(v); })) {
          throw std::runtime_error(name + " returned invalid seven-joint feedback");
        }
        sensor_msgs::msg::JointState message;
        message.header.stamp = now();
        message.header.frame_id = name + "_arm_base";
        for (int joint = 1; joint <= 7; ++joint) {
          message.name.push_back(name + "_joint_" + std::to_string(joint));
        }
        message.position.assign(joints.begin(), joints.end());
        // No synthetic velocities or torques: unavailable fields stay empty.
        joint_pubs_[side]->publish(message);
        success(side, "7 measured joint positions; unit=rad");
        const double elapsed = std::chrono::duration<double>(SteadyClock::now() - start).count();
        wait(1.0 / joint_hz_ - elapsed);
      } catch (const std::exception &error) {
        if (!running()) break;
        failure(side, name + ": " + error.what());
        // A live SDK connection is retained on read errors. This reader never
        // invokes disconnectFromRobot(), which the SDK documents as stopping motion.
        wait(1.0);
      }
    }
  }

  void suction_worker() {
    // The gateway can route replies incorrectly when separate TCP clients
    // overlap. Cooperate with the policy output worker across processes.
    const auto lock_path = "/run/user/" + std::to_string(::getuid()) + "/robot-m050.lock";
    const int lock_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (lock_fd < 0) {
      failure(2, "M050 lock open: " + std::string(std::strerror(errno)));
      return;
    }
    m050_config config;
    m050_config_default(&config);
    config.host = io_host_.c_str();
    config.port = static_cast<std::uint16_t>(io_port_);
    config.unit_id = static_cast<std::uint8_t>(io_unit_);
    config.timeout_ms = io_timeout_;
    m050_client *client = nullptr;
    m050_error error{};
    if (m050_create(&config, &client, &error) != M050_OK) {
      failure(2, std::string("M050 create: ") + error.message);
      ::close(lock_fd);
      return;
    }
    SteadyClock::time_point last_published{};
    std::uint64_t attempts = 0;
    while (running()) {
      const auto start = SteadyClock::now();
      m050_channels channels{};
      if (::flock(lock_fd, LOCK_EX) != 0) {
        failure(2, "M050 lock: " + std::string(std::strerror(errno)));
        wait(0.1);
        continue;
      }
      const auto acquired = SteadyClock::now();
      const auto result = m050_read_do(client, &channels, &error);
      const auto read_done = SteadyClock::now();
      ::flock(lock_fd, LOCK_UN);
      if (!running()) break;
      double publish_ms = 0.0, gap_ms = -1.0;
      if (result == M050_OK) {
        std_msgs::msg::Bool left, right;
        left.data = channels.ch1 != 0;
        right.data = channels.ch2 != 0;
        const auto publish_start = SteadyClock::now();
        suction_pubs_[0]->publish(left);
        suction_pubs_[1]->publish(right);
        const auto published = SteadyClock::now();
        publish_ms = std::chrono::duration<double, std::milli>(published-publish_start).count();
        if (last_published.time_since_epoch().count()) {
          gap_ms = std::chrono::duration<double, std::milli>(published-last_published).count();
        }
        last_published = published;
        success(2, "left_DO1=" + std::to_string(channels.ch1) +
                   ", right_DO2=" + std::to_string(channels.ch2));
      } else {
        // An IO failure emits diagnostics, never an invented OFF state.
        failure(2, std::string("M050 DO read: ") + error.message);
      }
      const auto lock_ms = std::chrono::duration<double, std::milli>(acquired-start).count();
      const auto read_ms = std::chrono::duration<double, std::milli>(read_done-acquired).count();
      // Instrument the existing read, without extra device requests or stale-state publication.
      // One normal sample per second, plus every slow/error sample.
      if (++attempts % 10 == 0 || result != M050_OK || lock_ms > 20.0 ||
          read_ms > 120.0 || publish_ms > 10.0 || gap_ms > 200.0) {
        RCLCPP_INFO(get_logger(), "m050_timing attempt=%llu result=%d lock_ms=%.3f read_ms=%.3f publish_ms=%.3f gap_ms=%.3f",
                    static_cast<unsigned long long>(attempts), static_cast<int>(result),
                    lock_ms, read_ms, publish_ms, gap_ms);
      }
      const double elapsed = std::chrono::duration<double>(SteadyClock::now() - start).count();
      wait(result == M050_OK ? 1.0 / io_hz_ - elapsed : 0.5);
    }
    m050_destroy(client);
    ::close(lock_fd);
  }

  void diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray message;
    message.header.stamp = now();
    std::lock_guard<std::mutex> lock(status_mutex_);
    for (int index = 0; index < 3; ++index) {
      const auto &state = status_[index];
      const double age = state.samples == 0 ? -1.0 :
          std::chrono::duration<double>(SteadyClock::now() - state.last_success).count();
      diagnostic_msgs::msg::DiagnosticStatus entry;
      entry.name = index == 0 ? "left_arm_feedback" :
                   index == 1 ? "right_arm_feedback" : "suction_output_feedback";
      entry.hardware_id = index < 2 ? arm_ips_[index] : io_host_;
      entry.level = state.error.empty() && age >= 0 && age < 1.0 ?
          diagnostic_msgs::msg::DiagnosticStatus::OK : diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      entry.message = entry.level == diagnostic_msgs::msg::DiagnosticStatus::OK ?
          "Fresh hardware readback" : state.error.empty() ? "Feedback stale" : state.error;
      auto add = [&entry](const std::string &key, const std::string &value) {
        diagnostic_msgs::msg::KeyValue field;
        field.key = key;
        field.value = value;
        entry.values.push_back(field);
      };
      add("samples", std::to_string(state.samples));
      add("failures", std::to_string(state.failures));
      add("last_success_age_sec", std::to_string(age));
      add("last_values", state.values);
      add("source", index < 2 ? "xCoreSDK jointPos" : "USR-M050 FC01 DO readback");
      if (index == 2) add("semantics", "true=relay closed/suction enabled; not vacuum detection");
      message.status.push_back(entry);
    }
    diagnostics_pub_->publish(message);
  }

  std::array<rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr, 2> joint_pubs_;
  std::array<rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr, 2> suction_pubs_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;
  std::array<std::string, 2> arm_ips_, local_ips_;
  std::array<TelemetryStatus, 3> status_;
  std::mutex status_mutex_, wait_mutex_, connect_mutex_;
  std::condition_variable condition_;
  std::atomic_bool stopping_{false};
  std::vector<std::thread> threads_;
  std::string io_host_;
  int io_port_, io_unit_, io_timeout_;
  double joint_hz_, io_hz_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<HardwareState>());
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("robot_hardware_state"), "%s", error.what());
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return 0;
}
