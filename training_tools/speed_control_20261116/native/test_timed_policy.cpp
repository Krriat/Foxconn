#include "timed_policy.hpp"
#include <cassert>
#include <iostream>
#include <limits>

using namespace timed_policy;
Plan line(std::uint64_t sequence, std::int64_t origin, double offset = 0) {
  Plan plan; plan.sequence = sequence; plan.observation_ns = origin;
  for (unsigned i = 0; i < 50; ++i) {
    Action point{};
    for (unsigned joint = 0; joint < 14; ++joint) point[joint] = offset + double(origin) / 1e9 + i * .05;
    point[14] = .8; point[15] = 0.; plan.actions.push_back(point);
  }
  return plan;
}
int main() {
  {
    Fusion fusion(0., 1.5); Fused fused;
    // Both plans describe q(t)=1.5*t, using unmodified 50-ms training steps.
    fusion.add(line(1, 0)); fusion.add(line(2, 500000000, .25));
    assert(fusion.sample(750000000, fused));
    assert(fused.contributors == 2);
    assert(std::abs(fused.target[0] - 1.125) < 1e-12);
    assert(std::abs(fused.oldest_index - 22.5) < 1e-12);
    assert(std::abs(fused.newest_index - 7.5) < 1e-12);
    std::cout << "PASS 1.5x overlapping plans align using each real observation origin\n";
  }
  {
    Fusion fusion(.01, 1.5); Fused fused;
    auto plan = line(1, 1000000000);
    for (unsigned i=0; i<50; ++i) plan.actions[i][14] = double(i)/49.;
    fusion.add(plan);
    assert(!fusion.sample(999999999, fused));
    assert(fusion.sample(1070000000, fused));
    assert(std::abs(fused.newest_index - 2.1) < 1e-12);
    assert(std::abs(fused.target[14] - 2.1/49.) < 1e-12);
    assert(fusion.sample(2600000000, fused));
    assert(!fusion.sample(2634000000, fused));
    fusion.reset(); assert(!fusion.sample(1070000000, fused));
    std::cout << "PASS latency compensation and suction share scaled horizon; real stamps, expiry and resets preserved\n";
  }
  {
    for (double invalid : {0., .999, 1.500001, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected=false;
      try { Fusion bad(.01, invalid); } catch (const std::invalid_argument &) { rejected=true; }
      assert(rejected);
    }
    bool rejected=false;
    try { Limiter bad(.5, 2.250001); } catch (const std::invalid_argument &) { rejected=true; }
    assert(rejected);
    Limiter smoother(.5, 2.25); Joints zero{}, target{}, last_v{}; smoother.reset(zero);
    double max_accel=0.;
    for (unsigned tick=0; tick<500; ++tick) {
      target.fill(tick/25%2 ? -.2 : .2);
      const auto previous=smoother.position(); const auto q=smoother.step(target,.01);
      for (unsigned j=0; j<14; ++j) {
        const double v=smoother.velocity()[j],accel=std::abs(v-last_v[j])/.01;
        assert(std::abs(v)<=.5+1e-12 && accel<=2.25+1e-10);
        assert(std::abs(q[j]-previous[j]-v*.01)<1e-12);
        max_accel=std::max(max_accel,accel);last_v[j]=v;
      }
    }
    assert(max_accel>2.24);
    std::cout << "PASS requested 2.25 acceleration limit, continuous commands, unchanged 0.5 speed cap and invalid-setting rejection\n";
  }
  {
    Fusion fusion(0.); Fused fused;
    fusion.add(line(1, 0)); fusion.add(line(2, 500000000));
    assert(fusion.sample(750000000, fused));
    assert(fused.contributors == 2 && std::abs(fused.target[0] - .75) < 1e-12);
    assert(std::abs(fused.oldest_index - 15.) < 1e-12 && std::abs(fused.newest_index - 5.) < 1e-12);
    assert(!fusion.sample(3000000000, fused));
    assert(!fusion.add(line(2, 500000000)));
    std::cout << "PASS same absolute execution time, interpolation, expiry and duplicate rejection\n";
  }
  {
    Fusion fusion(0.); Fused fused;
    auto first = line(1, 0); auto second = line(2, 500000000, -.1);
    first.actions[15][14] = 1.; second.actions[5][14] = 0.;
    fusion.add(first); fusion.add(second); fusion.sample(750000000, fused);
    assert(std::abs(fused.target[0] - .70) < 1e-12 && fused.target[14] == 0.);
    std::cout << "PASS conflicting plans fuse at aligned time; discrete IO follows newest eligible plan\n";
  }
  {
    Limiter smoother(.5, 1.); Joints zero{}, target{}; smoother.reset(zero);
    Joints last_q{}, last_v{};
    for (unsigned tick = 0; tick < 1000; ++tick) {
      // Simulate alternating new chunks, including the user's rollback case.
      target.fill(tick / 50 % 2 ? -.2 : .2);
      auto q = smoother.step(target, .01); const auto v = smoother.velocity();
      for (unsigned joint = 0; joint < 14; ++joint) {
        assert(std::abs(v[joint]) <= .5 + 1e-12);
        assert(std::abs((v[joint] - last_v[joint]) / .01) <= 1. + 1e-10);
        assert(std::abs((q[joint] - last_q[joint]) / .01 - v[joint]) < 1e-10);
      }
      last_q = q; last_v = v;
    }
    std::cout << "PASS continuous position, bounded velocity and acceleration under repeated target rollback\n";
  }
  {
    Limiter smoother; Joints zero{}, target{}; target.fill(.1); smoother.reset(zero);
    for (unsigned i = 0; i < 200; ++i) smoother.step(target, .01);
    assert(std::abs(smoother.position()[0] - .1) < .001);
    bool rejected = false; try { smoother.step(target, .1); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    std::cout << "PASS convergence and delayed-cycle rejection\n";
  }
}
