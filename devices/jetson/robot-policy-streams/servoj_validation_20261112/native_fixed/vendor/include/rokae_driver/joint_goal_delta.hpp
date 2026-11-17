#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace rokae_driver::joint_goal_delta {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kMaxJ5DeltaRad = 200.0 * kPi / 180.0;
inline constexpr double kMaxGoalDeltaRad = 2.0 * kPi;
using Limits = std::array<double, 7>;

inline Limits makeLimits(double defaultDeltaRad, double j5DeltaRad, double j3DeltaRad) {
  if (!std::isfinite(defaultDeltaRad) || defaultDeltaRad <= 0.0 || defaultDeltaRad > kMaxGoalDeltaRad) {
    throw std::invalid_argument("max_goal_delta_rad must be in (0, 2*pi]");
  }
  if (!std::isfinite(j5DeltaRad) || j5DeltaRad <= 0.0 || j5DeltaRad > kMaxGoalDeltaRad) {
    throw std::invalid_argument("j5_max_goal_delta_rad must be in (0, 2*pi]");
  }
  if (!std::isfinite(j3DeltaRad) || j3DeltaRad <= 0.0 || j3DeltaRad > kMaxGoalDeltaRad) {
    throw std::invalid_argument("j3_max_goal_delta_rad must be in (0, 2*pi]");
  }
  Limits limits;
  limits.fill(defaultDeltaRad);
  limits[2] = j3DeltaRad;
  limits[4] = j5DeltaRad;
  return limits;
}

inline Limits makeLimits(double defaultDeltaRad, double j5DeltaRad) {
  return makeLimits(defaultDeltaRad, j5DeltaRad, defaultDeltaRad);
}

inline bool withinLimit(double current, double target, std::size_t joint, const Limits &limits) {
  const double delta = std::abs(target - current);
  return std::isfinite(delta) && delta <= limits.at(joint);
}

}  // namespace rokae_driver::joint_goal_delta
