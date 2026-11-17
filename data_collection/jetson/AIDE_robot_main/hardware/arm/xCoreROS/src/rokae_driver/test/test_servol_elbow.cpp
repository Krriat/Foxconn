#include <cmath>
#include <iostream>
#include <stdexcept>

#include "rokae_driver/servol_math.hpp"

namespace {
void near(double actual, double expected) {
  if (std::abs(actual - expected) > 1e-12) {
    throw std::runtime_error("elbow fixed-speed mismatch");
  }
}
}

int main() {
  using rokae_driver::servol::nextElbowCommand;
  const double pi = std::acos(-1.0);
  const double speed = 5.0 * pi / 180.0;
  for (double direction : {-1.0, 1.0}) {
    for (double magnitude : {0.150001, 0.2, 0.5, 0.8, 1.0}) {
      for (double period : {0.005, 0.01, 0.02, 0.1}) {
        near(nextElbowCommand(0.2, direction * magnitude, period),
             0.2 + direction * speed * period);
      }
      double elbow = 0.2;
      for (int cycle = 0; cycle < 100; ++cycle) {
        elbow = nextElbowCommand(elbow, direction * magnitude, 0.01);
      }
      near(elbow, 0.2 + direction * speed);
    }
  }
  for (double joystick : {-0.15, -0.1, 0.0, 0.1, 0.15}) {
    near(nextElbowCommand(0.2, joystick, 0.01), 0.2);
  }
  near(nextElbowCommand(pi - 0.0001, 1.0, 0.01), pi);
  near(nextElbowCommand(-pi + 0.0001, -1.0, 0.01), -pi);
  near(nextElbowCommand(pi, 1.0, 0.01), pi);
  near(nextElbowCommand(-pi, -1.0, 0.01), -pi);
  near(nextElbowCommand(pi, -1.0, 0.01), pi - speed * 0.01);
  near(nextElbowCommand(-pi, 1.0, 0.01), -pi + speed * 0.01);
  std::cout << "PASS: fixed 5 deg/s for both directions and all amplitudes, "
               "period scaling, one-second integration, deadband, angle bounds\n";
}
