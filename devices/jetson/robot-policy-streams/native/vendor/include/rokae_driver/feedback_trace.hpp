#ifndef ROKAE_DRIVER__FEEDBACK_TRACE_HPP_
#define ROKAE_DRIVER__FEEDBACK_TRACE_HPP_

// Opt-in timing only. Producers never write files. The queue mutex is never
// held during file I/O; queue overflow drops diagnostic rows, not robot data.
// Set AIDE_FEEDBACK_TRACE_DIR before starting the driver; unset means disabled.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>

namespace rokae_driver {
class FeedbackTrace {
 public:
  struct Event {
    std::uint64_t mono_ns, utc_ns, start_ns, id, stamp_ns;
    std::string side, operation, phase;
  };
  static FeedbackTrace &instance() { static FeedbackTrace trace; return trace; }
  bool enabled() const noexcept { return enabled_; }
  static std::uint64_t monoNow() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  std::uint64_t nextId() noexcept { return ++sequence_; }
  void emit(const char *side, const char *operation, const char *phase,
            std::uint64_t id = 0, std::uint64_t start = 0,
            std::uint64_t stamp = 0) noexcept {
    if (!enabled_) return;
    try {
      Event e{monoNow(), static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()),
          start, id, stamp, side, operation, phase};
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.size() >= 16384) { ++dropped_; return; }
      queue_.push_back(std::move(e));
      lock.unlock();
      condition_.notify_one();
    } catch (...) { ++dropped_; }
  }
  ~FeedbackTrace() {
    if (!enabled_) return;
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    condition_.notify_one();
    writer_.join();
  }
 private:
  FeedbackTrace() {
    const char *directory = std::getenv("AIDE_FEEDBACK_TRACE_DIR");
    if (!directory || !*directory) return;
    const std::string path = std::string(directory) + "/sdk_" + std::to_string(::getpid()) + ".jsonl";
    output_.open(path, std::ios::out | std::ios::app);
    if (!output_) { std::cerr << "Cannot open feedback trace: " << path << '\n'; return; }
    try {
      writer_ = std::thread([this] { writeLoop(); });
      enabled_ = true;
    } catch (const std::exception &error) {
      std::cerr << "Cannot start feedback trace writer: " << error.what() << '\n';
      output_.close();
      return;
    }
    std::cerr << "AIDE_FEEDBACK_SDK_TRACE=" << path << '\n';
  }
  void writeLoop() {
    for (;;) {
      std::deque<Event> batch;
      bool done;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(100),
                            [this] { return stop_ || !queue_.empty(); });
        batch.swap(queue_);
        done = stop_;
      }
      for (const auto &e : batch) {
        // Side/operation/phase are internal fixed labels, never user strings.
        output_ << "{\"kind\":\"sdk\",\"mono_ns\":" << e.mono_ns
                << ",\"utc_ns\":" << e.utc_ns << ",\"start_ns\":" << e.start_ns
                << ",\"id\":" << e.id << ",\"stamp_ns\":" << e.stamp_ns
                << ",\"side\":\"" << e.side << "\",\"op\":\"" << e.operation
                << "\",\"phase\":\"" << e.phase << "\"}\n";
      }
      output_.flush();
      if (done) break;
    }
    output_ << "{\"kind\":\"trace_end\",\"dropped\":" << dropped_.load() << "}\n";
    output_.flush();
  }
  bool enabled_{false}, stop_{false};
  std::atomic<std::uint64_t> sequence_{0}, dropped_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Event> queue_;
  std::ofstream output_;
  std::thread writer_;
};

class FeedbackSpan {
 public:
  FeedbackSpan(const char *side, const char *operation)
      : trace_(FeedbackTrace::instance()), side_(side), operation_(operation) {
    if (trace_.enabled()) {
      start_ = FeedbackTrace::monoNow(); id_ = trace_.nextId();
      trace_.emit(side_, operation_, "begin", id_, start_);
    }
  }
  FeedbackSpan(const FeedbackSpan &) = delete;
  ~FeedbackSpan() { finish(); }
  void finish() noexcept {
    if (start_) { trace_.emit(side_, operation_, "end", id_, start_); start_ = 0; }
  }
 private:
  FeedbackTrace &trace_;
  const char *side_, *operation_;
  std::uint64_t start_{0}, id_{0};
};

template <typename Callable>
decltype(auto) traceFeedbackCall(const char *side, const char *operation, Callable &&callable) {
  FeedbackSpan span(side, operation);
  return std::forward<Callable>(callable)();
}
}  // namespace rokae_driver
#endif
