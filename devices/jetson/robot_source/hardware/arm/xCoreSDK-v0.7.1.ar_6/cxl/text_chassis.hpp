#pragma once
#include <functional>
#include <memory>
#include <string>
namespace text_chassis {
class Client {
 public:
  explicit Client(std::function<bool()> cancelled, const std::string &service="/robot/chassis/spin");
  ~Client();
  void spin(double turns, double angularSpeed);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
