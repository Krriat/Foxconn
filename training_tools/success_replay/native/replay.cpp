// Offline only: reads numbers from stdin. No ROS, SDK, sockets or robot calls.
#include "timed_policy.hpp"
#include <iomanip>
#include <iostream>
#include <string>

int main() {
  using namespace timed_policy;
  double coefficient, speed, acceleration, response, relative_limit, tracking_limit;
  if (!(std::cin >> coefficient >> speed >> acceleration >> response >> relative_limit >> tracking_limit)) return 6;
  Fusion fusion(coefficient);
  Limiter limiter(speed, acceleration, response);
  std::cout << std::setprecision(17);
  char kind;
  while (std::cin >> kind) {
    if (kind == 'A') {
      Joints anchor{};
      for (auto &x : anchor) std::cin >> x;
      limiter.reset(anchor);
    } else if (kind == 'P') {
      Plan plan;
      std::cin >> plan.sequence >> plan.observation_ns;
      plan.actions.resize(50);
      for (auto &point : plan.actions) for (auto &x : point) std::cin >> x;
      if (!fusion.add(std::move(plan))) return 2;
    } else if (kind == 'T') {
      std::int64_t execution_ns;
      double dt;
      Joints measured{}, desired{};
      std::cin >> execution_ns >> dt;
      for (auto &x : measured) std::cin >> x;
      Fused fused;
      if (!fusion.sample(execution_ns, fused)) return 3;
      for (unsigned j=0; j<14; ++j) {
        if (std::abs(limiter.position()[j]-measured[j]) > tracking_limit) {
          std::cerr << "tracking guard at " << execution_ns << '\n'; return 4;
        }
        desired[j] = std::clamp(fused.target[j],measured[j]-relative_limit,measured[j]+relative_limit);
      }
      const auto command = limiter.step(desired,dt);
      std::cout << fused.contributors;
      for (auto x : fused.target) std::cout << ' ' << x;
      for (auto x : command) std::cout << ' ' << x;
      for (auto x : limiter.velocity()) std::cout << ' ' << x;
      std::cout << '\n';
    } else return 5;
  }
  return 0;
}
