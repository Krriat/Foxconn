#include "text_topic_points.hpp"
#include "text_pose_offsets.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace text_points;
template<class F> void rejects(F f) {
  bool failed=false;
  try { f(); } catch (const std::exception &) { failed=true; }
  if (!failed) throw std::runtime_error("Expected rejection");
}
int main() {
  std::map<std::string,Offset> offsets{{"above",{"target",{0,0,.05}}},
                                      {"higher",{"above",{.01,-.02,.02}}}};
  auto ref=flattenOffset("higher",offsets);
  auto derived=applyOffset({.5,.3,-.2,1,2,3},ref.xyz);
  if (ref.base!="target" || std::abs(derived[0]-.51)>1e-10 ||
      std::abs(derived[1]-.28)>1e-10 || std::abs(derived[2]+.13)>1e-10 ||
      derived[3]!=1 || derived[4]!=2 || derived[5]!=3)
    throw std::runtime_error("Offset coordinate/orientation mismatch");
  offsets["target"]={"higher",{0,0,0}};
  rejects([&]{ flattenOffset("above",offsets); });
  auto automatic=parseSource({"target","left","/vision/part_target","base_link","pose2","10","5"});
  if (automatic.adapter!="auto" || automatic.topic!="/vision/part_target" || automatic.timeout!=10)
    throw std::runtime_error("Automatic source parse mismatch");
  if (adapterForType("std_msgs/msg/String")!="part_json" ||
      adapterForType("geometry_msgs/msg/PoseStamped")!="pose_stamped" ||
      adapterForType("geometry_msgs/msg/PointStamped")!="point_stamped")
    throw std::runtime_error("Automatic adapter selection mismatch");
  rejects([]{ adapterForType("std_msgs/msg/Int32"); });
  auto s=parseSource({"target","left","part_json","/vision/part_target","base_link","pose2","10","5"});
  auto p=parsePartJson(R"({"detected":true,"frame_id":"base_link","position":{"x":0.2,"y":0.3,"z":0.4},"stamp":{"sec":100,"nanosec":500000000}})");
  if (std::abs(p.xyz[0]-.2)>1e-10 || p.stamp!=100.5) throw std::runtime_error("JSON parse mismatch");
  validateSample(s,p,101);
  rejects([&]{ validateSample(s,p,110); });
  rejects([&]{ validateSample(s,p,90); });
  p.frame="camera"; rejects([&]{ validateSample(s,p,101); });
  p.frame="base_link"; p.xyz[0]=NAN; rejects([&]{ validateSample(s,p,101); });
  bool empty=false;
  try { parsePartJson(R"({"detected":false})"); }
  catch (const NoTarget &) { empty=true; }
  if (!empty) throw std::runtime_error("No target must be distinguishable from malformed data");
  rejects([]{ parsePartJson(R"({"detected":true,"frame_id":"base_link"})"); });
  rejects([]{ parseSource({"t","left","unknown","/a","base_link","p"}); });
  rejects([]{ parseSource({"t","left","part_json","/a","base_link","p","nan","5"}); });
  std::cout<<"Topic adapter parse, no detection, missing fields, frame, age and finite checks PASS\n";
}
