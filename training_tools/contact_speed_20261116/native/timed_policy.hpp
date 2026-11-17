#pragma once
// Pure trajectory math. No ROS, SDK, device, clock setter or hardware calls.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>

namespace timed_policy {
using Action = std::array<double, 16>;
using Joints = std::array<double, 14>;

struct Plan {
  std::uint64_t sequence{};
  std::int64_t observation_ns{}, inference_start_ns{}, inference_end_ns{}, send_ns{}, receive_ns{};
  std::int64_t step_ns{50000000};
  std::vector<Action> actions;
};

struct Fused {
  Action target{};
  unsigned contributors{};
  double newest_index{}, oldest_index{};
  std::int64_t newest_observation_ns{};
  std::uint64_t newest_sequence{};
  std::int64_t inference_start_ns{}, inference_end_ns{}, send_ns{}, receive_ns{};
};

// Continuous integrated action time; wall stamps are never rewritten. A new
// scale affects only future elapsed time, never rewinds existing predictions.
class ActionClock {
 public:
  struct Knot { std::int64_t wall; double action, scale; };
  void clear() { knots_.clear(); }
  bool initialized() const { return !knots_.empty(); }
  void update(std::int64_t wall, double scale, double initial_scale) {
    if (!std::isfinite(scale) || scale < 1. || scale > 1.5)
      throw std::invalid_argument("invalid action clock scale");
    if (knots_.empty()) {
      before_scale_=initial_scale;knots_.push_back({wall,0.,scale});return;
    }
    if (wall < knots_.back().wall) throw std::runtime_error("action clock moved backwards");
    const double next=value(wall);
    if (wall == knots_.back().wall) knots_.back().scale=scale;
    else knots_.push_back({wall,next,scale});
    // Active fusion retains at most 60 plans (~3 seconds). Keep a longer clock
    // history, retaining absolute accumulated action values when pruning.
    while (knots_.size()>2 && knots_[1].wall<wall-5000000000LL) knots_.pop_front();
  }
  double value(std::int64_t wall) const {
    if (knots_.empty()) throw std::logic_error("action clock not initialized");
    if (wall<knots_.front().wall)
      return knots_.front().action+double(wall-knots_.front().wall)*before_scale_;
    const auto upper=std::upper_bound(knots_.begin(),knots_.end(),wall,
      [](std::int64_t t,const Knot &k){return t<k.wall;});
    const auto &k=*(upper-1);
    return k.action+double(wall-k.wall)*k.scale;
  }
 private:
  double before_scale_{1.};
  std::deque<Knot> knots_;
};

// Pose-based speed regions only: NOT suction-contact or task-success detection.
// Thresholds come from the saved successful trajectories. Once entered, the
// slow region remains latched until BOTH arms complete the manipulation sweep.
class ContactSpeed {
 public:
  void reset(bool near) { near_=near;phase_=0;lift_seen_=false; }
  double request(const Joints &q,double fast) {
    const auto rad=[](double degree){return degree*3.14159265358979323846/180.;};
    if (phase_==0 && (near_ ? q[3]>=rad(114.) : q[1]>=rad(30.))) phase_=1;
    if (phase_==1) {
      if (near_ ? q[10]>=rad(125.) : q[8]>=rad(35.)) lift_seen_=true;
      const bool clear=near_ ? (q[1]<rad(25.) && q[8]<rad(25.) && q[10]<rad(115.))
                            : (q[1]<rad(30.) && q[8]<rad(30.) && q[10]>rad(80.));
      if (lift_seen_ && clear) phase_=2;
    }
    return phase_==1 ? 1. : fast;
  }
  const char *phase() const { return phase_==0 ? "approach" : phase_==1 ? "contact_and_lift" : "retract"; }
 private:
  bool near_{false},lift_seen_{false};int phase_{0};
};

class Fusion {
 public:
  explicit Fusion(double coefficient = .01, double execution_speed_scale = 1.)
      : coefficient_(coefficient), execution_speed_scale_(execution_speed_scale) {
    if (!std::isfinite(coefficient) || coefficient < 0 || coefficient > 1)
      throw std::invalid_argument("invalid ensemble coefficient");
    if (!std::isfinite(execution_speed_scale) || execution_speed_scale < 1. || execution_speed_scale > 1.5)
      throw std::invalid_argument("execution speed scale must be between 1 and 1.5");
  }
  void reset() { plans_.clear();clock_.clear(); }
  void update_speed_scale(std::int64_t wall,double scale) { clock_.update(wall,scale,execution_speed_scale_); }
  double action_age(std::int64_t execution,std::int64_t observation) const {
    return clock_.initialized() ? clock_.value(execution)-clock_.value(observation)
                                : execution_speed_scale_*double(execution-observation);
  }
  bool add(Plan plan) {
    if (plan.actions.size() != 50 || plan.step_ns != 50000000)
      throw std::invalid_argument("plan must contain fifty 20-Hz actions");
    for (const auto &point : plan.actions)
      for (double value : point) if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite plan point");
    if (!plans_.empty() && (plan.sequence <= plans_.back().sequence ||
                           plan.observation_ns <= plans_.back().observation_ns)) return false;
    plans_.push_back(std::move(plan));
    while (plans_.size() > 60) plans_.pop_front();
    return true;
  }
  void discard_before(std::int64_t minimum_ns) {
    while (!plans_.empty() && plans_.front().observation_ns < minimum_ns) plans_.pop_front();
  }
  bool sample(std::int64_t execution_ns, Fused &result) const {
    result = Fused{};
    double weight_sum = 0.;
    for (const auto &plan : plans_) {
      const double index = action_age(execution_ns,plan.observation_ns)/double(plan.step_ns);
      if (index < 0 || index > double(plan.actions.size() - 1)) continue;
      const auto low = static_cast<std::size_t>(std::floor(index));
      const auto high = std::min(low + 1, plan.actions.size() - 1);
      const double fraction = index - low;
      // ACT temporal-ensemble convention: rank zero is the oldest eligible plan.
      const double weight = std::exp(-coefficient_ * result.contributors);
      for (unsigned joint = 0; joint < 14; ++joint) {
        const double target = plan.actions[low][joint] * (1 - fraction) + plan.actions[high][joint] * fraction;
        result.target[joint] += weight * target;
      }
      if (result.contributors == 0) result.oldest_index = index;
      ++result.contributors; weight_sum += weight;
      result.newest_index = index;
      result.newest_observation_ns = plan.observation_ns;
      result.newest_sequence = plan.sequence;
      result.inference_start_ns = plan.inference_start_ns;
      result.inference_end_ns = plan.inference_end_ns;
      result.send_ns = plan.send_ns;
      result.receive_ns = plan.receive_ns;
      // Discrete IO uses the newest eligible plan at this SAME execution time.
      // Averaging old grasp phases would delay an intentional suction release.
      for (unsigned channel = 14; channel < 16; ++channel)
        result.target[channel] = plan.actions[low][channel] * (1 - fraction) + plan.actions[high][channel] * fraction;
    }
    if (!result.contributors) return false;
    for (unsigned joint = 0; joint < 14; ++joint) result.target[joint] /= weight_sum;
    return true;
  }
  std::size_t size() const { return plans_.size(); }
 private:
  double coefficient_, execution_speed_scale_;
  ActionClock clock_;
  std::deque<Plan> plans_;
};

class Limiter {
 public:
  Limiter(double maximum_speed = .5, double maximum_acceleration = 1., double response_seconds = .1)
      : maximum_speed_(maximum_speed), maximum_acceleration_(maximum_acceleration), response_seconds_(response_seconds) {
    if (!(maximum_speed > 0 && maximum_speed <= .5 && maximum_acceleration > 0 &&
          maximum_acceleration <= 2.25 && std::isfinite(response_seconds) && response_seconds >= .05))
      throw std::invalid_argument("invalid trajectory limits");
  }
  void reset(const Joints &position) { position_ = position; velocity_.fill(0.); initialized_ = true; }
  const Joints &step(const Joints &target, double actual_dt) {
    if (!initialized_) throw std::logic_error("trajectory not initialized");
    if (!(actual_dt > 0 && actual_dt <= .05)) throw std::runtime_error("trajectory cycle overdue");
    for (unsigned joint = 0; joint < 14; ++joint) {
      if (!std::isfinite(target[joint])) throw std::invalid_argument("nonfinite trajectory target");
      const double error = target[joint] - position_[joint];
      const double braking_speed = std::sqrt(2 * maximum_acceleration_ * std::abs(error));
      const double desired_speed = std::copysign(
          std::min({maximum_speed_, braking_speed, std::abs(error) / response_seconds_}), error);
      const double maximum_dv = maximum_acceleration_ * actual_dt;
      const double next_velocity = std::clamp(desired_speed, velocity_[joint] - maximum_dv, velocity_[joint] + maximum_dv);
      // Preserve velocity and position across plan updates. Never snap to a new
      // chunk's first point or reset the ramp when another prediction arrives.
      position_[joint] += next_velocity * actual_dt;
      velocity_[joint] = next_velocity;
    }
    return position_;
  }
  const Joints &position() const { return position_; }
  const Joints &velocity() const { return velocity_; }
 private:
  double maximum_speed_, maximum_acceleration_, response_seconds_;
  bool initialized_{false};
  Joints position_{}, velocity_{};
};
}  // namespace timed_policy
