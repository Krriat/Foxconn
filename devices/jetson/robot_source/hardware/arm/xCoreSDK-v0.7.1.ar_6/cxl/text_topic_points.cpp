#include "text_topic_points.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <deque>

namespace text_points {
std::string adapterForType(const std::string &type) {
  if (type=="std_msgs/msg/String") return "part_json";
  if (type=="geometry_msgs/msg/PoseStamped") return "pose_stamped";
  if (type=="geometry_msgs/msg/PointStamped") return "point_stamped";
  throw std::runtime_error("Unsupported target message type: "+type+"; add an adapter in text_topic_points.cpp");
}
Source parseSource(const std::vector<std::string> &t) {
  if (t.size() != 5 && t.size() != 7 && t.size() != 6 && t.size() != 8)
    throw std::runtime_error("topic point: NAME left|right TOPIC FRAME RPY_TEMPLATE [TIMEOUT_S MAX_AGE_S]");
  Source s;
  const bool automatic = t.size()==5 || t.size()==7;
  const std::size_t offset = automatic ? 0 : 1;
  s.name=t[0]; s.arm=t[1]; s.adapter=automatic ? "auto" : t[2];
  s.topic=t[2+offset]; s.frame=t[3+offset]; s.orientationTemplate=t[4+offset];
  if (s.arm!="left" && s.arm!="right") throw std::runtime_error("Topic point arm must be left or right.");
  if (s.adapter!="auto" && s.adapter!="part_json" && s.adapter!="pose_stamped" && s.adapter!="point_stamped")
    throw std::runtime_error("Unknown point adapter; use part_json, pose_stamped or point_stamped.");
  if (s.topic.empty() || s.topic[0]!='/') throw std::runtime_error("Topic must be an absolute ROS topic name.");
  if (t.size()==7 || t.size()==8) {
    auto number=[](const std::string &v) {
      std::size_t used=0; double n=std::stod(v,&used);
      if (used!=v.size() || !std::isfinite(n) || n<=0 || n>120)
        throw std::runtime_error("Topic timeout/max age must be >0 and <=120 seconds.");
      return n;
    };
    s.timeout=number(t[5+offset]); s.maxAge=number(t[6+offset]);
  }
  return s;
}
Sample parsePartJson(const std::string &json) {
  boost::property_tree::ptree tree;
  std::istringstream in(json);
  boost::property_tree::read_json(in, tree);
  if (!tree.get<bool>("detected")) throw NoTarget("Detection reports no target.");
  Sample s;
  s.frame=tree.get<std::string>("frame_id");
  s.xyz={tree.get<double>("position.x"),tree.get<double>("position.y"),tree.get<double>("position.z")};
  const auto sec=tree.get<int64_t>("stamp.sec");
  const auto nsec=tree.get<int64_t>("stamp.nanosec");
  if (nsec<0 || nsec>=1000000000) throw std::runtime_error("Invalid detection timestamp.");
  s.stamp=sec+nsec*1e-9;
  if (auto label=tree.get_optional<std::string>("class")) {
    auto value=*label;
    if (value.compare(0,4,"part")==0) value=value.substr(4);
    std::size_t used=0;
    s.category=std::stoi(value,&used);
    if (used!=value.size() || s.category<=0) throw std::runtime_error("Invalid material class: "+*label);
  }
  return s;
}
void validateSample(const Source &source, const Sample &s, double now) {
  if (s.frame!=source.frame) throw std::runtime_error("Frame mismatch: received="+s.frame+", expected="+source.frame);
  for (double v:s.xyz) if (!std::isfinite(v)) throw std::runtime_error("Non-finite target coordinate.");
  if (!std::isfinite(s.stamp) || s.stamp<=0 || now-s.stamp>source.maxAge || s.stamp-now>1)
    throw std::runtime_error("Target timestamp stale/invalid or ROS clocks not synchronized.");
}
struct TopicPoints::Impl {
  struct Entry {
    Source source;
    Sample sample;
    bool valid=false, frozen=false, subscribed=false;
    bool sawFreshPose=false;
    uint64_t sequence=0;
    uint64_t invalidCount=0;
    std::array<double,6> pose{};
    std::string error="No message received";
    std::deque<Sample> classes;
    int frozenCategory=0;
  };
  std::map<std::string,Entry> entries;
  std::shared_ptr<rclcpp::Context> context;
  std::shared_ptr<rclcpp::Node> node;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;
  std::mutex mutex;
  std::condition_variable cv;
  std::thread worker;
  std::string fatal;
  double roundStart=0;
  template<class Decode> void receive(const std::string &name, Decode decode) {
    std::lock_guard<std::mutex> lock(mutex);
    auto &e=entries.at(name);
    ++e.sequence;
    try {
      e.sample=decode();
      validateSample(e.source,e.sample,node->now().seconds());
      if (e.sample.stamp>=roundStart) e.sawFreshPose=true;
      e.valid=true; e.error.clear();
    } catch (const NoTarget &ex) { e.valid=false; e.error=ex.what(); }
    catch (const std::exception &ex) { e.valid=false; ++e.invalidCount; e.error=ex.what(); }
    cv.notify_all();
  }
  explicit Impl(const std::map<std::string,Source> &sources) {
    for (const auto &pair:sources) { Entry e; e.source=pair.second; entries.emplace(pair.first,e); }
    context=std::make_shared<rclcpp::Context>();
    rclcpp::InitOptions init;
    // Allows vision on a different domain from lift/gripper without changing their context.
    if (const char *env=std::getenv("TEXT_DCT_POINTS_DOMAIN_ID")) {
      std::string value(env); std::size_t used=0;
      unsigned long domain=std::stoul(value,&used);
      if (used!=value.size() || domain>232) throw std::runtime_error("TEXT_DCT_POINTS_DOMAIN_ID must be 0..232.");
      init.set_domain_id(domain);
    }
    context->init(0,nullptr,init);
    rclcpp::NodeOptions options; options.context(context);
    node=std::make_shared<rclcpp::Node>("text_dct_topic_points",options);
    roundStart=node->now().seconds();
    rclcpp::ExecutorOptions executorOptions; executorOptions.context=context;
    executor=std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executorOptions);
    executor->add_node(node);
    for (const auto &pair:sources)
      if (pair.second.adapter!="auto") subscribe(pair.first,pair.second);
    for (const auto &pair:sources) {
      if (pair.second.classTopic.empty()) continue;
      const auto name=pair.first;
      subscriptions.push_back(node->create_subscription<std_msgs::msg::String>(pair.second.classTopic,
        rclcpp::QoS(10).best_effort().durability_volatile(),
        [this,name](std_msgs::msg::String::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(mutex);
          auto &e=entries.at(name);
          try {
            auto sample=parsePartJson(msg->data);
            validateSample(e.source,sample,node->now().seconds());
            if (!sample.category) throw std::runtime_error("Missing class in detection JSON");
            e.classes.push_back(sample);
            if (e.classes.size()>30) e.classes.pop_front();
          } catch (const NoTarget &) { e.classes.clear(); }
          catch (const std::exception &ex) { ++e.invalidCount; e.error=ex.what(); }
          cv.notify_all();
        }));
    }
    worker=std::thread([this] {
      try { executor->spin(); }
      catch (const std::exception &e) { std::lock_guard<std::mutex> lock(mutex); fatal=e.what(); cv.notify_all(); }
    });
  }
  void subscribe(const std::string &name, const Source &s) {
    auto qos=rclcpp::QoS(1).best_effort().durability_volatile();
      if (s.adapter=="part_json") {
        subscriptions.push_back(node->create_subscription<std_msgs::msg::String>(s.topic,qos,
          [this,name](std_msgs::msg::String::ConstSharedPtr msg) {
            receive(name,[&] { return parsePartJson(msg->data); });
          }));
      } else if (s.adapter=="pose_stamped") {
        subscriptions.push_back(node->create_subscription<geometry_msgs::msg::PoseStamped>(s.topic,qos,
          [this,name](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
            receive(name,[&] { return Sample{{msg->pose.position.x,msg->pose.position.y,msg->pose.position.z},
              msg->header.frame_id,rclcpp::Time(msg->header.stamp).seconds()}; });
          }));
      } else {
        subscriptions.push_back(node->create_subscription<geometry_msgs::msg::PointStamped>(s.topic,qos,
          [this,name](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
            receive(name,[&] { return Sample{{msg->point.x,msg->point.y,msg->point.z},
              msg->header.frame_id,rclcpp::Time(msg->header.stamp).seconds()}; });
          }));
      }
    entries.at(name).subscribed=true;
  }
  ~Impl() {
    executor->cancel();
    if (worker.joinable()) worker.join();
    executor->remove_node(node);
    context->shutdown("topic points finished");
  }
};
TopicPoints::TopicPoints(const std::map<std::string,Source> &s):impl_(new Impl(s)) {}
TopicPoints::~TopicPoints()=default;
std::array<double,6> TopicPoints::resolve(const std::string &name,const std::function<bool()> &cancelled) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  auto &e=impl_->entries.at(name);
  if (cancelled()) throw std::runtime_error("Topic point wait cancelled.");
  if (e.frozen) return e.pose;
  const auto initial=e.sequence;
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(e.source.timeout);
  while (std::chrono::steady_clock::now()<deadline) {
    if (cancelled()) throw std::runtime_error("Topic point wait cancelled.");
    if (!impl_->fatal.empty()) throw std::runtime_error("Topic executor failed: "+impl_->fatal);
    if (!e.subscribed) {
      const auto publishers=impl_->node->get_publishers_info_by_topic(e.source.topic);
      std::string type;
      for (const auto &publisher:publishers) {
        if (!type.empty() && type!=publisher.topic_type())
          throw std::runtime_error("Multiple message types published on "+e.source.topic);
        type=publisher.topic_type();
      }
      if (!type.empty()) {
        Source selected=e.source;
        selected.adapter=adapterForType(type);
        impl_->subscribe(name,selected);
      } else e.error="No publisher discovered; check topic and ROS domain";
    }
    if (e.sequence>initial && e.valid) {
      if (e.sample.stamp < impl_->roundStart) {
        e.valid=false;
        e.error="Sample was captured before this round; waiting for a new detection";
        continue;
      }
      try {
        validateSample(e.source,e.sample,impl_->node->now().seconds());
        int category=e.sample.category;
        if (!e.source.classTopic.empty()) {
          category=0;
          for (const auto &label:e.classes) {
            // Epoch seconds stored as double lose sub-microsecond precision.
            if (std::abs(label.stamp-e.sample.stamp)>1e-6 || label.frame!=e.sample.frame) continue;
            bool same=true;
            for (int i=0;i<3;++i) if (std::abs(label.xyz[i]-e.sample.xyz[i])>0.00011) same=false;
            if (same) { category=label.category; break; }
          }
          if (!category) {
            e.error="No class JSON matching pose timestamp/position on "+e.source.classTopic;
            impl_->cv.wait_for(lock,std::chrono::milliseconds(50));
            continue;
          }
        }
        e.frozenCategory=category;
        for (int i=0;i<3;++i) { e.pose[i]=e.sample.xyz[i]; e.pose[i+3]=e.source.rpy[i]; }
        e.frozen=true;
        std::ostringstream log;
        log << std::fixed << std::setprecision(6)
            << "Topic point " << name << " frozen for this run:\n"
            << "  XYZ(m)=[" << e.pose[0] << ", " << e.pose[1] << ", " << e.pose[2] << "]\n"
            << "  RPY(rad)=[" << e.pose[3] << ", " << e.pose[4] << ", " << e.pose[5]
            << "] (template=" << e.source.orientationTemplate << ")\n";
        std::cout<<log.str()<<std::flush;
        if (category) std::cout<<"Material class for "<<name<<" = "<<category<<std::endl;
        return e.pose;
      } catch (const std::exception &ex) { e.valid=false; ++e.invalidCount; e.error=ex.what(); }
    }
    impl_->cv.wait_for(lock,std::chrono::milliseconds(50));
  }
  if (cancelled()) throw std::runtime_error("Topic point wait cancelled.");
  const auto reason="Topic point "+name+" timed out on "+e.source.topic+": "+e.error;
  if (e.invalidCount==0 && !(e.sawFreshPose && !e.source.classTopic.empty())) throw NoTarget(reason);
  throw std::runtime_error(reason);
}
int TopicPoints::category(const std::string &name) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto &e=impl_->entries.at(name);
  if (!e.frozen || !e.frozenCategory) throw std::runtime_error("Material category not acquired for "+name);
  return e.frozenCategory;
}
}
