#include <m050/m050.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

#include "m050_ros/msg/io_state.hpp"
#include "m050_ros/srv/get_io.hpp"
#include "m050_ros/srv/set_do.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace m050_ros
{
class M050Node : public rclcpp::Node
{
public:
  M050Node() : Node("m050")
  {
    const auto host = setting<std::string>("host", M050_DEFAULT_HOST);
    const auto port = integer_setting("port", M050_DEFAULT_PORT, 1, 65535);
    const auto unit = integer_setting("unit_id", M050_DEFAULT_UNIT_ID, 1, 247);
    const auto timeout = integer_setting(
      "timeout_ms", M050_DEFAULT_TIMEOUT_MS, 1, std::numeric_limits<int>::max());
    const auto source = setting<std::string>("source_ip", "");
    const auto poll_ms = integer_setting(
      "poll_interval_ms", 500, 0, std::numeric_limits<int>::max());

    client_ = std::make_unique<m050::Client>(
      host, static_cast<unsigned>(port), static_cast<unsigned>(unit),
      static_cast<int>(timeout), source);
    publisher_ = create_publisher<msg::IOState>("~/state", rclcpp::QoS(1));
    get_io_ = create_service<srv::GetIO>(
      "~/get_io", [this](const std::shared_ptr<srv::GetIO::Request>,
        std::shared_ptr<srv::GetIO::Response> response) {
        response->state = read_io();
      });
    set_do_ = create_service<srv::SetDO>(
      "~/set_do", [this](const std::shared_ptr<srv::SetDO::Request> request,
        std::shared_ptr<srv::SetDO::Response> response) {
        set_do(*request, *response);
      });
    if (poll_ms > 0) {
      timer_ = create_wall_timer(std::chrono::milliseconds(poll_ms), [this]() {read_io();});
    }
    RCLCPP_INFO(
      get_logger(), "USR-M050 %s:%ld unit=%ld timeout=%ld ms poll=%ld ms",
      host.c_str(), static_cast<long>(port), static_cast<long>(unit),
      static_cast<long>(timeout), static_cast<long>(poll_ms));
  }

private:
  template<typename T>
  T setting(const std::string & name, const T & default_value)
  {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    descriptor.description = "Startup setting; restart the node to change it.";
    return declare_parameter<T>(name, default_value, descriptor);
  }

  int64_t integer_setting(
    const std::string & name, int64_t default_value, int64_t minimum, int64_t maximum)
  {
    const auto value = setting<int64_t>(name, default_value);
    if (value < minimum || value > maximum) {
      throw std::invalid_argument(name + " is outside the allowed range [" +
        std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    }
    return value;
  }

  msg::IOState read_io()
  {
    msg::IOState state;
    try {
      const auto value = client_->read_io();
      state.valid = true;
      state.do_state = {{value.do_state.ch1, value.do_state.ch2}};
      state.di_state = {{value.di.ch1, value.di.ch2}};
      state.message = "success";
    } catch (const m050::Error & error) {
      state.error_code = error.code();
      state.message = error.what();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "IO read failed: %s", error.what());
    }
    state.stamp = now();
    publisher_->publish(state);
    return state;
  }

  void set_do(const srv::SetDO::Request & request, srv::SetDO::Response & response)
  {
    try {
      const auto value = client_->set_do(request.channel, request.closed);
      response.success = true;
      response.do_state_valid = true;
      response.do_state = {{value.ch1, value.ch2}};
      response.message = "success";
    } catch (const m050::Error & error) {
      response.error_code = error.code();
      response.message = error.what();
      response.write_may_have_applied = error.write_may_have_applied();
      RCLCPP_WARN(get_logger(), "DO%u request failed: %s (write_may_have_applied=%s)",
        static_cast<unsigned>(request.channel), error.what(),
        response.write_may_have_applied ? "true" : "false");
    }
  }

  std::unique_ptr<m050::Client> client_;
  rclcpp::Publisher<msg::IOState>::SharedPtr publisher_;
  rclcpp::Service<srv::GetIO>::SharedPtr get_io_;
  rclcpp::Service<srv::SetDO>::SharedPtr set_do_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace m050_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    // One executor serializes polling and services, including SDK network waits.
    rclcpp::spin(std::make_shared<m050_ros::M050Node>());
  } catch (const std::exception & error) {
    if (rclcpp::ok()) {
      RCLCPP_ERROR(rclcpp::get_logger("m050"), "%s", error.what());
      result = 1;
    }
  }
  rclcpp::shutdown();
  return result;
}
