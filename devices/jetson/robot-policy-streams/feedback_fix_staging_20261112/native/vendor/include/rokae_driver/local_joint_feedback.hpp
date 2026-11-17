#ifndef ROKAE_DRIVER__LOCAL_JOINT_FEEDBACK_HPP_
#define ROKAE_DRIVER__LOCAL_JOINT_FEEDBACK_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace rokae_driver {
struct JointFeedbackSample {
  std::uint8_t side{0}, source{0};  // side: left/right=0/1; source: drag/recording=0/1.
  std::uint64_t stamp_ns{0}, mono_ns{0};
  std::array<double, 7> joints{};
};

// DDS can block. Only this worker calls it; SDK callbacks offer measured
// samples to a bounded queue whose mutex is never held during publication.
class AsyncJointPublisher {
 public:
  using Publish = std::function<void(const JointFeedbackSample &)>;
  explicit AsyncJointPublisher(Publish publish, std::size_t capacity = 16)
      : publish_(std::move(publish)), capacity_(capacity) {
    if (!capacity_) throw std::invalid_argument("joint queue capacity must be positive");
    worker_ = std::thread([this] { run(); });
  }
  AsyncJointPublisher(const AsyncJointPublisher &) = delete;
  ~AsyncJointPublisher() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    condition_.notify_one();
    worker_.join();
  }
  void offer(const JointFeedbackSample &sample) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.size() == capacity_) { queue_.pop_front(); ++overwritten_; }
      queue_.push_back(sample);
    }
    condition_.notify_one();
  }
  std::uint64_t overwritten() const noexcept { return overwritten_.load(); }
  std::uint64_t failures() const noexcept { return failures_.load(); }
 private:
  void run() {
    for (;;) {
      JointFeedbackSample sample;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_) return;
        sample = queue_.front(); queue_.pop_front();
      }
      try { publish_(sample); }
      catch (...) { ++failures_; }  // Publication failure must never escape into SDK control.
    }
  }
  Publish publish_;
  const std::size_t capacity_;
  std::atomic<std::uint64_t> overwritten_{0}, failures_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<JointFeedbackSample> queue_;
  bool stop_{false};
  std::thread worker_;
};

// Fixed little-endian wire layout: Python Struct('<8sBBHI6Q7d'), 120 bytes.
// Local packets contain measured joints only, never a robot command.
inline std::array<unsigned char, 120> encodeJointFeedback(
    const JointFeedbackSample &sample, std::uint32_t pid, std::uint64_t session,
    std::uint64_t sequence, std::uint64_t send_errors, std::uint64_t ros_overwrites) {
  static_assert(sizeof(double) == 8, "feedback protocol requires 64-bit doubles");
  std::array<unsigned char, 120> packet{};
  std::memcpy(packet.data(), "AIDEJNT1", 8);
  packet[8] = sample.side; packet[9] = sample.source;
  auto u64 = [&](std::size_t at, std::uint64_t value) {
    for (int i=0; i<8; ++i) packet[at+i] = static_cast<unsigned char>(value >> (8*i));
  };
  for (int i=0; i<4; ++i) packet[12+i] = static_cast<unsigned char>(pid >> (8*i));
  u64(16, session); u64(24, sequence); u64(32, sample.stamp_ns);
  u64(40, sample.mono_ns); u64(48, send_errors); u64(56, ros_overwrites);
  for (std::size_t i=0; i<7; ++i) {
    std::uint64_t bits; std::memcpy(&bits, &sample.joints[i], 8); u64(64+8*i,bits);
  }
  return packet;
}

class LocalJointFeedback {
 public:
  static LocalJointFeedback &instance() { static LocalJointFeedback value; return value; }
  bool enabled() const noexcept { return socket_ >= 0; }
  static std::uint64_t monoNow() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void offer(const JointFeedbackSample &sample, std::uint64_t ros_overwrites = 0) noexcept {
    if (!enabled() || sample.side > 1 || sample.source > 1 || !sample.stamp_ns) return;
    for (double v : sample.joints) if (!std::isfinite(v)) return;
    const auto index = 2*sample.side+sample.source;
    const auto sequence = ++sequence_[index];
    const auto packet = encodeJointFeedback(sample, static_cast<std::uint32_t>(::getpid()),
        session_, sequence, errors_[index].load(), ros_overwrites);
    if (::sendto(socket_, packet.data(), packet.size(), MSG_DONTWAIT,
          reinterpret_cast<const sockaddr *>(&destination_), sizeof(destination_)) !=
        static_cast<ssize_t>(packet.size())) ++errors_[index];
  }
  ~LocalJointFeedback() { if (socket_ >= 0) ::close(socket_); }
 private:
  LocalJointFeedback() {
    const char *setting = std::getenv("AIDE_JOINT_FEEDBACK_PORT");
    if (!setting || !*setting) return;  // Enabled only by the acquisition launcher.
    char *end = nullptr; const long port = std::strtol(setting, &end, 10);
    if (*end || port < 1 || port > 65535) throw std::invalid_argument("invalid AIDE_JOINT_FEEDBACK_PORT");
    socket_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (socket_ < 0) throw std::runtime_error("cannot create loopback joint feedback socket");
    destination_.sin_family = AF_INET; destination_.sin_port = htons(static_cast<std::uint16_t>(port));
    destination_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    session_ = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    std::cerr << "AIDE_LOCAL_JOINT_FEEDBACK=127.0.0.1:" << port << "; measured timestamps, protocol v1\n";
  }
  int socket_{-1};
  sockaddr_in destination_{};
  std::uint64_t session_{0};
  std::array<std::atomic<std::uint64_t>,4> sequence_{};
  std::array<std::atomic<std::uint64_t>,4> errors_{};
};
}  // namespace rokae_driver
#endif
