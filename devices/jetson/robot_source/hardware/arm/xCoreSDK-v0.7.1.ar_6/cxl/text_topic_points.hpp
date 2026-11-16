#pragma once
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace text_points {
class NoTarget : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
// Add message adapters in text_topic_points.cpp; the workflow only sees named poses.
struct Source {
  std::string name, arm, adapter, topic, frame, orientationTemplate;
  std::string classTopic;
  double timeout = 10, maxAge = 5;
  std::array<double, 3> rpy{};
};
std::string adapterForType(const std::string &type);
Source parseSource(const std::vector<std::string> &tokens);
struct Sample {
  std::array<double, 3> xyz{};
  std::string frame;
  double stamp = 0;
  int category = 0;
};
Sample parsePartJson(const std::string &json);
void validateSample(const Source &source, const Sample &sample, double now);
class TopicPoints {
 public:
  explicit TopicPoints(const std::map<std::string, Source> &sources);
  ~TopicPoints();
  TopicPoints(const TopicPoints &) = delete;
  TopicPoints &operator=(const TopicPoints &) = delete;
  // First use waits for a new valid sample and freezes it for this workflow run.
  std::array<double, 6> resolve(const std::string &name, const std::function<bool()> &cancelled);
  int category(const std::string &name) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
