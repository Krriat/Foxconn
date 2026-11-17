#include "rokae_driver/node_factories.hpp"
#include "rokae_driver/shared_arm_hardware.hpp"
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  {
    auto initializer = rokae_driver::makeRobotInitializerService();
    auto move = rokae_driver::makeMoveAbsJService();
    auto servo = rokae_driver::makeServoJSubscriber();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 6);
    executor.add_node(initializer); executor.add_node(move); executor.add_node(servo);
    executor.spin();
  }
  rokae_driver::SharedArmRegistry::instance().clear();
  rclcpp::shutdown();
}
