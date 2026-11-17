#ifndef ROKAE_DRIVER__NODE_FACTORIES_HPP_
#define ROKAE_DRIVER__NODE_FACTORIES_HPP_

#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace rokae_driver {
std::shared_ptr<rclcpp::Node> makeCartesianDragSafeService();

std::shared_ptr<rclcpp::Node> makeHandService();
std::shared_ptr<rclcpp::Node> makeCartesianTeachService();
std::shared_ptr<rclcpp::Node> makeKinematicsService();
std::shared_ptr<rclcpp::Node> makeGoHomeService();
std::shared_ptr<rclcpp::Node> makeMoveAbsJActionServer();
std::shared_ptr<rclcpp::Node> makeMoveAbsJService();
std::shared_ptr<rclcpp::Node> makeMoveLService();
std::shared_ptr<rclcpp::Node> makeMoveJByPathService();
std::vector<std::shared_ptr<rclcpp::Node>> makeStatePublishers();
std::shared_ptr<rclcpp::Node> makeRobotInitializerService();
std::shared_ptr<rclcpp::Node> makeServoJSubscriber();
std::shared_ptr<rclcpp::Node> makeServoLSubscriber();

}  // namespace rokae_driver

#endif  // ROKAE_DRIVER__NODE_FACTORIES_HPP_
