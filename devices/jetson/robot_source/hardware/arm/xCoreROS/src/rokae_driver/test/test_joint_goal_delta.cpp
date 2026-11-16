#include <cassert>
#include <cmath>
#include <limits>

#include "rokae_driver/joint_goal_delta.hpp"

int main() {
  using namespace rokae_driver::joint_goal_delta;
  const auto left = makeLimits(2.3, kMaxJ5DeltaRad);
  const auto right = makeLimits(2.3, 2.3);
  assert(std::abs(kMaxJ5DeltaRad - 3.490658503988659) < 1e-14);
  // The reported 174-degree change is accepted only for the configured J5.
  for (std::size_t joint = 0; joint < left.size(); ++joint) {
    assert(withinLimit(0.0, 3.036504, joint, left) == (joint == 4));
    assert(!withinLimit(0.0, 3.036504, joint, right));
    assert(withinLimit(0.0, 2.3, joint, left));
  }
  assert(withinLimit(0.0, kMaxJ5DeltaRad, 4, left));
  assert(withinLimit(0.0, -kMaxJ5DeltaRad, 4, left));
  assert(withinLimit(-2.735798, 0.300706, 4, left));
  assert(!withinLimit(0.0, std::nextafter(kMaxJ5DeltaRad, 4.0), 4, left));
  assert(!withinLimit(kMaxJ5DeltaRad, -0.1, 4, left));
  assert(!withinLimit(0.0, std::nextafter(2.3, 3.0), 3, left));
  assert(!withinLimit(0.0, std::numeric_limits<double>::quiet_NaN(), 4, left));

  for (const double bad : {0.0, -1.0, kMaxGoalDeltaRad + .001,
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    bool rejected = false;
    try { makeLimits(2.3, bad); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
  }
  bool rejected = false;
  try { makeLimits(kMaxGoalDeltaRad + .001, kMaxJ5DeltaRad); }
  catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);  // Reject changes larger than a complete revolution.

  const auto relaxedLeft = makeLimits(2.3, kMaxJ5DeltaRad, 2.5);
  const auto unchangedRight = makeLimits(2.3, kMaxJ5DeltaRad);
  assert(withinLimit(0., 2.333101, 2, relaxedLeft));
  assert(withinLimit(0., -2.333101, 2, relaxedLeft));
  assert(withinLimit(-1., 1.333101, 2, relaxedLeft));
  assert(!withinLimit(0., 2.333101, 2, unchangedRight));
  assert(withinLimit(0., 2.5, 2, relaxedLeft));
  assert(!withinLimit(0., std::nextafter(2.5, 3.), 2, relaxedLeft));
  for (std::size_t joint = 0; joint < relaxedLeft.size(); ++joint) {
    if (joint != 2) assert(relaxedLeft[joint] == left[joint]);
  }
  for (const double bad : {0., -1., kMaxGoalDeltaRad + .001,
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    rejected = false;
    try { makeLimits(2.3, kMaxJ5DeltaRad, bad); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
  }
  const auto all = makeLimits(kMaxGoalDeltaRad, kMaxGoalDeltaRad, kMaxGoalDeltaRad);
  for (std::size_t joint = 0; joint < all.size(); ++joint) {
    assert(withinLimit(0., 2.333102, joint, all));
    assert(withinLimit(-2.95, 2.95, joint, all));
    assert(withinLimit(0., kMaxGoalDeltaRad, joint, all));
    assert(!withinLimit(0., std::nextafter(kMaxGoalDeltaRad, 7.), joint, all));
  }
}
