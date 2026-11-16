#include "text_chassis.hpp"
#include <rclcpp/rclcpp.hpp>
#include <chassis_spin_interfaces/srv/spin.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace text_chassis {
using Clock=std::chrono::steady_clock;
using Spin=chassis_spin_interfaces::srv::Spin;
using Stop=std_srvs::srv::Trigger;
struct Client::Impl {
  std::function<bool()> cancelled;
  std::shared_ptr<rclcpp::Context> context;
  std::shared_ptr<rclcpp::Node> node;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  rclcpp::Client<Spin>::SharedPtr start;
  rclcpp::Client<Stop>::SharedPtr stop;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription;
  boost::property_tree::ptree status;
  Clock::time_point received{};
  std::string statusError;
  uint64_t count=0;
  bool active=false;
  void pump(bool honorCancel=true) {
    if (honorCancel && cancelled()) throw std::runtime_error("Chassis operation cancelled");
    executor->spin_once(std::chrono::milliseconds(50));
    if (honorCancel && cancelled()) throw std::runtime_error("Chassis operation cancelled");
  }
  template<class Future> void wait(Future &f, double seconds, bool honorCancel=true) {
    const auto deadline=Clock::now()+std::chrono::duration<double>(seconds);
    while (f.wait_for(std::chrono::seconds(0))!=std::future_status::ready) {
      if (Clock::now()>=deadline) throw std::runtime_error("Chassis service response timeout");
      pump(honorCancel);
    }
  }
  void halt() noexcept {
    if (!active) return;
    active=false;
    try {
      auto f=stop->async_send_request(std::make_shared<Stop::Request>());
      wait(f,2,false);
      auto response=f.get();
      std::cerr<<"Chassis stop: "<<response->message<<std::endl;
      if (!response->success) std::cerr<<"Chassis stop was rejected"<<std::endl;
    } catch (const std::exception &e) { std::cerr<<"Chassis stop failed: "<<e.what()<<std::endl; }
  }
  explicit Impl(std::function<bool()> c,const std::string &service):cancelled(std::move(c)) {
    context=std::make_shared<rclcpp::Context>();
    rclcpp::InitOptions init;
    unsigned long domain=0;
    if (const char *env=std::getenv("TEXT_DCT_CHASSIS_DOMAIN_ID")) {
      std::string value(env); std::size_t used=0; domain=std::stoul(value,&used);
      if (used!=value.size() || domain>232) throw std::runtime_error("Invalid TEXT_DCT_CHASSIS_DOMAIN_ID");
    }
    init.set_domain_id(domain); context->init(0,nullptr,init);
    rclcpp::NodeOptions options; options.context(context);
    node=std::make_shared<rclcpp::Node>("text_dct_chassis",options);
    rclcpp::ExecutorOptions eo; eo.context=context;
    executor=std::make_unique<rclcpp::executors::SingleThreadedExecutor>(eo);
    executor->add_node(node);
    start=node->create_client<Spin>(service);
    stop=node->create_client<Stop>(service+"/stop");
    subscription=node->create_subscription<std_msgs::msg::String>(service+"/status",rclcpp::QoS(1),
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        ++count; received=Clock::now();
        try {
          boost::property_tree::ptree next; std::istringstream in(msg->data);
          boost::property_tree::read_json(in,next);
          next.get<int64_t>("task_id");
          const auto state=next.get<std::string>("state");
          if (state!="idle" && state!="running" && state!="succeeded" && state!="failed" && state!="canceled")
            throw std::runtime_error("Unknown chassis state");
          status=std::move(next); statusError.clear();
        } catch (const std::exception &e) { statusError=e.what(); }
      });
    const auto deadline=Clock::now()+std::chrono::seconds(5);
    while (!start->service_is_ready() || !stop->service_is_ready() || !count) {
      if (Clock::now()>=deadline) throw std::runtime_error("Chassis preflight: service/status missing on domain "+std::to_string(domain));
      pump();
    }
    if (!statusError.empty()) throw std::runtime_error("Chassis status invalid: "+statusError);
    if (status.get<std::string>("state")=="running") throw std::runtime_error("Chassis already running another task");
  }
  ~Impl() { halt(); executor->remove_node(node); context->shutdown("chassis client finished"); }
};
Client::Client(std::function<bool()> c,const std::string &s):impl_(new Impl(std::move(c),s)) {}
Client::~Client()=default;
void Client::spin(double turns,double speed) {
  auto &p=*impl_;
  if (p.cancelled()) throw std::runtime_error("Chassis operation cancelled");
  if (!std::isfinite(turns) || turns==0 || std::abs(turns)>100 || !std::isfinite(speed) || speed<=0 || speed>0.5)
    throw std::runtime_error("Invalid chassis turns/speed");
  const auto oldCount=p.count;
  const auto readyDeadline=Clock::now()+std::chrono::seconds(2);
  while (p.count==oldCount) {
    if (Clock::now()>readyDeadline) throw std::runtime_error("No fresh chassis state before request");
    p.pump();
  }
  if (!p.statusError.empty()) throw std::runtime_error("Invalid chassis status: "+p.statusError);
  if (p.status.get<std::string>("state")=="running") throw std::runtime_error("Chassis is busy");
  const auto baseline=p.count;
  const auto previousTask=p.status.get<int64_t>("task_id");
  auto request=std::make_shared<Spin::Request>(); request->turns=turns; request->angular_speed=speed;
  p.active=true;
  try {
    auto future=p.start->async_send_request(request);
    p.wait(future,5);
    auto response=future.get();
    if (!response->accepted) { p.active=false; throw std::runtime_error("Chassis request rejected: "+response->message); }
    const auto marker=response->message.find("task_id=");
    if (marker==std::string::npos) throw std::runtime_error("Chassis accepted but response has no task_id");
    const auto id=std::stoll(response->message.substr(marker+8));
    if (id<=previousTask) throw std::runtime_error("Chassis task generation changed; restart/recheck the server");
    std::cout<<"Chassis accepted task="<<id<<" turns="<<turns<<"; waiting for completion"<<std::endl;
    const auto deadline=Clock::now()+std::chrono::duration<double>(std::max(10.0,3*std::abs(turns)*2*3.141592653589793/speed+5)+2);
    const auto acceptedAt=Clock::now();
    while (Clock::now()<deadline) {
      p.pump();
      if (Clock::now()-std::max(p.received,acceptedAt)>std::chrono::seconds(2)) throw std::runtime_error("Chassis status lost");
      if (p.count<=baseline) continue;
      if (!p.statusError.empty()) throw std::runtime_error("Chassis status invalid: "+p.statusError);
      if (p.status.get<int64_t>("task_id")!=id) continue;
      const auto state=p.status.get<std::string>("state");
      if (state=="succeeded") {
        p.active=false;
        // Server sends five final zero-velocity messages at 10 Hz before accepting the next workflow action.
        const auto settle=Clock::now()+std::chrono::milliseconds(600);
        while (Clock::now()<settle) p.pump();
        std::cout<<"Chassis task "<<id<<" completed"<<std::endl; return;
      }
      if (state=="failed" || state=="canceled") throw std::runtime_error("Chassis "+state+": "+p.status.get<std::string>("message"));
    }
    throw std::runtime_error("Chassis completion timeout");
  } catch (...) { p.halt(); throw; }
}
}
