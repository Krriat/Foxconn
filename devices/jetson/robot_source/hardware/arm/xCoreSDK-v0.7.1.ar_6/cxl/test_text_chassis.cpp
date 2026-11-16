// Local fake services only. Does not publish cmd_vel or call real chassis services.
#include "text_chassis.hpp"
#include "text_topic_points.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <chassis_spin_interfaces/srv/spin.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/string.hpp>
#include <atomic>
#include <thread>
#include <sstream>
#include <iostream>
using Spin=chassis_spin_interfaces::srv::Spin;
using Stop=std_srvs::srv::Trigger;
int main() {
  setenv("TEXT_DCT_CHASSIS_DOMAIN_ID","174",1);
  setenv("TEXT_DCT_POINTS_DOMAIN_ID","174",1);
  setenv("ROS_LOCALHOST_ONLY","1",1);
  auto context=std::make_shared<rclcpp::Context>();
  rclcpp::InitOptions init; init.set_domain_id(174); context->init(0,nullptr,init);
  rclcpp::NodeOptions options; options.context(context);
  auto node=std::make_shared<rclcpp::Node>("fake_chassis",options);
  const std::string prefix="/text_chassis_isolated_test/spin";
  std::atomic<int> mode{0},stops{0},requests{0}; std::atomic<bool> cancel{false};
  std::atomic<double> lastTurns{0};
  int id=0,ticks=0; std::string state="idle";
  auto pub=node->create_publisher<std_msgs::msg::String>(prefix+"/status",10);
  auto posePub=node->create_publisher<geometry_msgs::msg::PoseStamped>(prefix+"/pose",10);
  auto classPub=node->create_publisher<std_msgs::msg::String>(prefix+"/class",10);
  std::atomic<int> category{2}; std::atomic<bool> mismatch{false};
  auto start=node->create_service<Spin>(prefix,[&](const Spin::Request::SharedPtr req,Spin::Response::SharedPtr out){
    ++requests; lastTurns=req->turns;
    if(mode==2) { out->accepted=false; out->message="injected rejection"; return; }
    ++id; state="running"; ticks=0;
    out->accepted=true; out->message="spin accepted; task_id="+std::to_string(id);
  });
  auto stop=node->create_service<Stop>(prefix+"/stop",[&](const Stop::Request::SharedPtr,Stop::Response::SharedPtr out){
    ++stops; state="canceled"; out->success=true; out->message="fake stop";
  });
  auto timer=node->create_wall_timer(std::chrono::milliseconds(50),[&]{
    if(state=="running" && ++ticks>=5) {
      if(mode==3) cancel=true;
      else state=mode==1?"failed":"succeeded";
    }
    std_msgs::msg::String msg;
    msg.data="{\"task_id\":"+std::to_string(id)+",\"state\":\""+state+"\",\"message\":\"fake\"}";
    pub->publish(msg);
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp=node->now(); pose.header.frame_id="base_link";
    pose.pose.position.x=.4; pose.pose.position.y=.2; pose.pose.position.z=-.1;
    pose.pose.orientation.w=1;
    std_msgs::msg::String label;
    label.data="{\"detected\":true,\"class\":\"part"+std::to_string(category.load())+
      "\",\"frame_id\":\"base_link\",\"position\":{\"x\":0.4,\"y\":0.2,\"z\":-0.1},\"stamp\":{\"sec\":"+
      std::to_string(pose.header.stamp.sec-(mismatch?1:0))+",\"nanosec\":"+std::to_string(pose.header.stamp.nanosec)+"}}";
    classPub->publish(label); posePub->publish(pose);
  });
  rclcpp::ExecutorOptions eo; eo.context=context;
  rclcpp::executors::SingleThreadedExecutor exec(eo); exec.add_node(node);
  std::thread worker([&]{exec.spin();});
  int result=0;
  try {
    text_chassis::Client client([&]{return cancel.load();},prefix);
    client.spin(.25,.2); client.spin(-.25,.2);
    if(lastTurns!=-0.25 || requests!=2 || stops!=0) throw std::runtime_error("signed/success mismatch");
    auto fails=[&]{bool failed=false; try {client.spin(.25,.2);} catch(const std::exception &){failed=true;}
      if(!failed) throw std::runtime_error("expected failure");};
    mode=1; fails(); if(stops!=1) throw std::runtime_error("failure must stop");
    mode=2; fails(); if(stops!=1) throw std::runtime_error("rejection must not stop another task");
    mode=3; fails(); if(stops!=2) throw std::runtime_error("cancellation must stop");
    std::cout<<"Mock chassis: signed quarter turns, completion, failure, rejection and cancellation PASS\n";
    cancel=false; mode=0;
    auto source=text_points::parseSource({"target","left",prefix+"/pose","base_link","template","5","5"});
    source.classTopic=prefix+"/class";
    {
      text_points::TopicPoints points({{"target",source}});
      const auto pose=points.resolve("target",[]{return false;});
      if(pose[0]!=.4 || points.category("target")!=2) throw std::runtime_error("paired class2 failure");
      category=1;
      points.resolve("target",[]{return false;});
      if(points.category("target")!=2) throw std::runtime_error("category must stay frozen");
    }
    {
      text_points::TopicPoints points({{"target",source}});
      points.resolve("target",[]{return false;});
      if(points.category("target")!=1) throw std::runtime_error("new round must acquire new category");
    }
    mismatch=true;
    {
      text_points::TopicPoints points({{"target",source}});
      bool failed=false;
      try {points.resolve("target",[]{return false;});}
      catch(const text_points::NoTarget &e) {throw std::runtime_error(std::string("class mismatch must not mean empty material: ")+e.what());}
      catch(const std::exception &) {failed=true;}
      if(!failed) throw std::runtime_error("mismatched class timestamp accepted");
    }
    std::cout<<"Mock vision: paired class, per-round freeze/refresh, mismatch rejection PASS\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;result=1;}
  exec.cancel(); worker.join(); exec.remove_node(node); context->shutdown("test finished");
  return result;
}
