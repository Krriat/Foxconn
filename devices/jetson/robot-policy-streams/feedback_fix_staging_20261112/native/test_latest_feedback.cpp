#include "latest_feedback.hpp"
#include <cassert>
#include <deque>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace std::chrono;
  auto now = latest_feedback::Clock::time_point{} + seconds(1);
  std::deque<int> fifo;
  int selected = -1;
  // 1 kHz production, 100 Hz control: every selection must be the newest frame,
  // including a startup backlog and stationary joints with identical values.
  for (int cycle = 1; cycle <= 100; ++cycle) {
    for (int i = 1; i <= (cycle == 1 ? 500 : 10); ++i) fifo.push_back(cycle*1000+i);
    const int expected = fifo.back();
    auto result = latest_feedback::drain([&] {
      if (fifo.empty()) return 0;
      selected = fifo.front(); fifo.pop_front(); return 64;
    }, [&] { now += microseconds(1); return now; }, milliseconds(5));
    assert(selected == expected && fifo.empty());
    assert(result.frames == (cycle == 1 ? 500U : 10U));
    assert(result.discarded() + 1 == result.frames);
  }
  auto empty = latest_feedback::drain([] { return 0; }, [&] { return now; }, milliseconds(5));
  assert(empty.frames == 0 && empty.discarded() == 0);
  bool count_failed = false, time_failed = false;
  try { latest_feedback::drain([] { return 64; }, [&] { return now; }, milliseconds(5), 20); }
  catch (const std::runtime_error &) { count_failed = true; }
  try { latest_feedback::drain([] { return 64; }, [&] { now += milliseconds(1); return now; }, milliseconds(5)); }
  catch (const std::runtime_error &) { time_failed = true; }
  assert(count_failed && time_failed);
  bool receive_failed = false;
  try { latest_feedback::drain([]() -> int { throw std::runtime_error("transport"); }, [&] { return now; }, milliseconds(5)); }
  catch (const std::runtime_error &) { receive_failed = true; }
  assert(receive_failed);
  assert(!latest_feedback::fresh({}, now));
  assert(latest_feedback::fresh(now, now + milliseconds(30)));
  assert(!latest_feedback::fresh(now, now + milliseconds(31)));
  assert(!latest_feedback::fresh(now, now - milliseconds(1)));
  std::cout << "PASS: newest FIFO selection, startup backlog, empty input, bounded overload, transport error, stale feedback\n";
}
