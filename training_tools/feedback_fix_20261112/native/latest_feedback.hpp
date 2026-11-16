#pragma once
// Bounded FIFO draining. This file has no robot or ROS dependencies.
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace latest_feedback {
using Clock = std::chrono::steady_clock;
struct DrainResult {
  std::size_t frames{};
  double duration_ms{};
  std::size_t discarded() const { return frames ? frames - 1 : 0; }
};

// receive() must be nonblocking and return zero when the SDK FIFO is empty.
// Only a drained FIFO may be used for control; exhausting a bound is an error,
// not permission to use the last historical packet read so far.
template <typename Receive, typename Now>
DrainResult drain(Receive receive, Now now, std::chrono::milliseconds budget,
                  std::size_t maximum_frames = 4096) {
  const auto start = now();
  DrainResult result;
  const auto check = [&] {
    const double elapsed = std::chrono::duration<double, std::milli>(now() - start).count();
    if (result.frames >= maximum_frames || elapsed >= budget.count())
      throw std::runtime_error(
          "SDK feedback FIFO did not drain within bounded read budget: limit=" +
          std::string(result.frames >= maximum_frames ? "frames" : "time") +
          ", elapsed_ms=" + std::to_string(elapsed) +
          ", budget_ms=" + std::to_string(budget.count()) +
          ", frames=" + std::to_string(result.frames) +
          ", maximum_frames=" + std::to_string(maximum_frames));
  };
  while (true) {
    check();
    if (receive() == 0) {
      // A nominally nonblocking SDK call may still be delayed by scheduling.
      // An empty response after the deadline must not bypass the same bound.
      check();
      break;
    }
    ++result.frames;
  }
  result.duration_ms = std::chrono::duration<double, std::milli>(now() - start).count();
  return result;
}

inline bool fresh(Clock::time_point last_read, Clock::time_point now,
                  std::chrono::milliseconds maximum_age = std::chrono::milliseconds(30)) {
  return last_read != Clock::time_point{} && now >= last_read && now - last_read <= maximum_age;
}
}  // namespace latest_feedback
