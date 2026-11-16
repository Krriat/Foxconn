// 双臂 MoveL/MoveJ/MoveAbsJ + 并行块 + DO2 配置流程。用法：text_dct [--check] [配置文件]
// 坐标取自 cartesian_drag_safe 的 endInRef 输出；保持原工具/参考系配置。
// 位姿和流程读取同目录 text_dct.conf；完整校验后才允许连接硬件。
// DO2 对应 /dev/gpiochip0 line 6，与 text_dct.py 一致；不回放拖动轨迹。
#include <array>
#include <atomic>
#include <set>
#include <algorithm>
#include <fstream>
#include <future>
#include <memory>
#include <sstream>
#include <map>
#include <vector>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/gpio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <Eigen/Geometry>
#include "rokae/robot.h"
#include "rokae/utility.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "text_topic_points.hpp"
#include "text_pose_offsets.hpp"
#include "text_chassis.hpp"
#include "liftmotor_rs485/msg/lift_motor_state.hpp"
#include "liftmotor_rs485/srv/move_absolute.hpp"
#include "liftmotor_rs485/srv/move_relative.hpp"
#include "liftmotor_rs485/srv/stop.hpp"

using namespace rokae;

namespace {
struct Step {
  std::string operation;
  std::string arm;
  std::string argument;
  double number = 0;
  int line = 0;
  int group = 0;
  std::string branch;
  int materialClass = 0;
  double angularSpeed = 0.2;
};
bool armMotion(const Step &s) {
  return s.operation == "move" || s.operation == "movel" ||
         s.operation == "movej" || s.operation == "moveabsj";
}
bool linearMotion(const Step &s) { return s.operation == "move" || s.operation == "movel"; }

struct Workflow {
  double speed = 50;
  double jointSpeed = 0.2;
  double jointTolerance = 0.001;
  double gripperTimeout = 10;
  int gripperTolerance = 20;
  double maxMoveDistance = 0.30;
  std::map<std::string, std::array<double, 6>> poses;
  std::map<std::string, text_points::Source> topicPoints;
  std::map<std::string, std::string> classTopics;
  std::map<std::string, text_points::Offset> offsets;
  std::map<std::string, text_points::Offset> dynamicReferences;
  std::map<std::string, std::array<double, 7>> joints;
  std::vector<Step> steps;
};

Workflow loadWorkflow(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open config: " + path);
  Workflow flow;
  std::string line, section;
  int lineNumber = 0;
  bool speedSeen = false;
  bool jointSpeedSeen = false;
  bool jointToleranceSeen = false;
  std::set<std::string> gripperSettings;
  int group = 0, nextGroup = 0, groupSize = 0;
  std::set<std::string> resources;
  bool concurrent = false;
  std::string branch;
  int branchSize = 0;
  std::set<std::string> branches;
  std::map<std::string, std::string> owners;
  bool maxMoveDistanceSeen = false;
  auto fail = [&](const std::string &message) {
    throw std::runtime_error(path + ":" + std::to_string(lineNumber) + ": " + message);
  };
  auto number = [&](const std::string &text) {
    std::size_t used = 0;
    double value = 0;
    try { value = std::stod(text, &used); }
    catch (...) { fail("Expected a number, got: " + text); }
    if (used != text.size() || !std::isfinite(value)) fail("Invalid finite number: " + text);
    return value;
  };
  while (std::getline(input, line)) {
    ++lineNumber;
    line = line.substr(0, line.find('#'));
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) tokens.push_back(token);
    if (tokens.empty()) continue;
    if (tokens[0][0] == '[') {
      if (tokens.size() != 1 || (tokens[0] != "[settings]" &&
          tokens[0] != "[poses]" && tokens[0] != "[topic_classes]" && tokens[0] != "[topic_poses]" && tokens[0] != "[offset_poses]" && tokens[0] != "[joints]" && tokens[0] != "[steps]")) fail("Unknown section.");
      if (group) fail("Close parallel_end before changing section.");
      section = tokens[0];
    } else if (section == "[settings]") {
      if (tokens.size() != 2)
        fail("Setting needs exactly: NAME VALUE.");
      if (tokens[0] == "speed_mm_s") {
        if (speedSeen) fail("Duplicate speed_mm_s setting.");
        flow.speed = number(tokens[1]);
        if (flow.speed < 1 || flow.speed > 250) fail("speed_mm_s must be 1..250.");
        speedSeen = true;
      } else if (tokens[0] == "joint_speed_ratio") {
        if (jointSpeedSeen) fail("Duplicate joint_speed_ratio.");
        flow.jointSpeed = number(tokens[1]);
        if (flow.jointSpeed <= 0 || flow.jointSpeed > 1) fail("joint_speed_ratio must be >0 and <=1.");
        jointSpeedSeen = true;
      } else if (tokens[0] == "gripper_timeout_s" || tokens[0] == "gripper_tolerance_counts") {
        if (!gripperSettings.insert(tokens[0]).second) fail("Duplicate " + tokens[0]);
        const double value = number(tokens[1]);
        if (tokens[0] == "gripper_timeout_s") {
          if (value <= 0 || value > 120) fail("gripper_timeout_s must be >0 and <=120.");
          flow.gripperTimeout = value;
        } else {
          if (value < 0 || value > 4095 || std::floor(value) != value)
            fail("gripper_tolerance_counts must be an integer 0..4095.");
          flow.gripperTolerance = static_cast<int>(value);
        }
      } else if (tokens[0] == "joint_tolerance_rad") {
        if (jointToleranceSeen) fail("Duplicate joint_tolerance_rad.");
        flow.jointTolerance = number(tokens[1]);
        if (flow.jointTolerance <= 0 || flow.jointTolerance > 0.01)
          fail("joint_tolerance_rad must be >0 and <=0.01 rad.");
        jointToleranceSeen = true;
      } else if (tokens[0] == "max_move_distance_m") {
        if (maxMoveDistanceSeen) fail("Duplicate max_move_distance_m setting.");
        flow.maxMoveDistance = number(tokens[1]);
        if (flow.maxMoveDistance < 0.001 || flow.maxMoveDistance > 2.0)
          fail("max_move_distance_m must be 0.001..2.0.");
        maxMoveDistanceSeen = true;
      } else {
        fail("Unknown setting: " + tokens[0]);
      }
    } else if (section == "[topic_classes]") {
      if (tokens.size()!=2 || tokens[1].empty() || tokens[1][0]!='/') fail("Use: TOPIC_POINT CLASS_JSON_TOPIC");
      if (!flow.classTopics.emplace(tokens[0],tokens[1]).second) fail("Duplicate point class source");
    } else if (section == "[offset_poses]") {
      if (tokens.size() != 5) fail("Offset point needs: NAME BASE_POINT DX DY DZ (metres).");
      text_points::Offset offset{tokens[1], {number(tokens[2]), number(tokens[3]), number(tokens[4])}};
      if (!flow.offsets.emplace(tokens[0], offset).second) fail("Duplicate offset point: " + tokens[0]);
    } else if (section == "[topic_poses]") {
      try {
        auto source = text_points::parseSource(tokens);
        if (!flow.topicPoints.emplace(source.name, source).second) fail("Duplicate topic point: " + source.name);
      } catch (const std::exception &e) { fail(e.what()); }
    } else if (section == "[poses]") {
      if (tokens.size() != 7) fail("Pose needs: name X Y Z RX RY RZ (metres/radians).");
      std::array<double, 6> pose{};
      for (int i = 0; i < 6; ++i) pose[i] = number(tokens[i + 1]);
      if (!flow.poses.emplace(tokens[0], pose).second) fail("Duplicate pose: " + tokens[0]);
    } else if (section == "[joints]") {
      if (tokens.size() != 8) fail("Joint point needs: NAME J1 J2 J3 J4 J5 J6 J7 (rad).");
      std::array<double, 7> q{};
      for (int i = 0; i < 7; ++i) q[i] = number(tokens[i + 1]);
      if (!flow.joints.emplace(tokens[0], q).second) fail("Duplicate joint point: " + tokens[0]);
    } else if (section == "[steps]") {
      if (tokens[0] == "concurrent_begin") {
        if (tokens.size() != 1 || group) fail("concurrent_begin takes no arguments; blocks cannot nest.");
        group = ++nextGroup; groupSize = 0; concurrent = true;
        branches.clear(); owners.clear(); branch.clear();
        continue;
      }
      if (tokens[0] == "branch") {
        if (!concurrent || !branch.empty() || tokens.size() != 2)
          fail("Use branch NAME inside concurrent_begin; close the previous branch_end first.");
        if (!branches.insert(tokens[1]).second) fail("Duplicate branch name: " + tokens[1]);
        branch = tokens[1]; branchSize = 0;
        continue;
      }
      if (tokens[0] == "branch_end") {
        if (tokens.size() != 1 || !concurrent || branch.empty() || !branchSize)
          fail("Unmatched or empty branch.");
        branch.clear();
        continue;
      }
      if (tokens[0] == "concurrent_end") {
        if (tokens.size() != 1 || !concurrent || !branch.empty() || branches.size() < 2)
          fail("Close all branches and provide at least two branches before concurrent_end.");
        concurrent = false; group = 0;
        continue;
      }
      if (tokens[0] == "parallel_begin" || tokens[0] == "parallel_end") {
        if (concurrent) fail("Parallel markers cannot appear inside concurrent blocks.");
        if (tokens.size() != 1) fail("Parallel marker takes no arguments.");
        if (tokens[0] == "parallel_begin") {
          if (group) fail("Nested parallel blocks are not supported.");
          group = ++nextGroup; groupSize = 0; resources.clear();
        } else {
          if (!group || !groupSize) fail("Unmatched or empty parallel block.");
          group = 0;
        }
        continue;
      }
      if (tokens[0]=="spin_if_class") {
        if (group) fail("spin_if_class must be outside parallel/concurrent blocks, after all arm actions have joined.");
        if (tokens.size()!=4 && tokens.size()!=5) fail("Use: spin_if_class POINT CLASS TURNS [RAD_S]");
        Step spin;
        spin.operation=tokens[0]; spin.argument=tokens[1]; spin.line=lineNumber;
        const auto category=number(tokens[2]);
        if (category<=0 || category>65535 || std::floor(category)!=category) fail("Class must be a positive integer");
        spin.materialClass=static_cast<int>(category); spin.number=number(tokens[3]);
        if (spin.number==0 || std::abs(spin.number)>100) fail("Chassis turns must be nonzero and within +/-100");
        if (tokens.size()==5) spin.angularSpeed=number(tokens[4]);
        if (spin.angularSpeed<=0 || spin.angularSpeed>0.5) fail("Chassis speed must be >0 and <=0.5 rad/s");
        flow.steps.push_back(spin);
        if (flow.steps.size()>1000) fail("Too many steps");
        continue;
      }
      if (tokens.size() < 2 || tokens.size() > 4)
        fail("Step must be: move [left|right] NAME [SPEED_MM_S], lift DISTANCE_MM, lift_abs POSITION_MM, do 0|1, or wait SECONDS.");
      Step step;
      if (concurrent && branch.empty()) fail("Put concurrent actions inside branch NAME ... branch_end.");
      step.branch = branch;
      step.operation = tokens[0];
      step.line = lineNumber;
      step.group = group;
      if (step.operation == "gripper" || step.operation == "gripper_read") {
        const bool read = step.operation == "gripper_read";
        if (tokens.size() != (read ? 2u : 3u))
          fail("Use gripper left|right POSITION or gripper_read left|right.");
        step.arm = tokens[1];
        if (step.arm != "left" && step.arm != "right") fail("Gripper side must be left or right.");
        if (!read) {
          step.argument = tokens[2];
          step.number = number(tokens[2]);
          if (step.number < 0 || step.number > 4095 || std::floor(step.number) != step.number)
            fail("Gripper position must be an integer 0..4095.");
        }
      } else if (step.operation == "do") {
        if (tokens.size() != 2) fail("do needs exactly one value.");
        step.argument = tokens[1];
        if (step.argument != "0" && step.argument != "1") fail("do accepts only 0 or 1.");
        step.number = number(step.argument);
      } else if (step.operation == "wait") {
        if (tokens.size() != 2) fail("wait needs exactly one value.");
        step.argument = tokens[1];
        step.number = number(step.argument);
        if (step.number < 0 || step.number > 3600) fail("Wait must be 0..3600 seconds.");
      } else if (step.operation == "lift") {
        if (tokens.size() != 2) fail("lift needs exactly one relative distance in mm.");
        step.argument = tokens[1];
        step.number = number(step.argument);
        if (step.number == 0)
          fail("Lift distance must be nonzero.");
      } else if (step.operation == "lift_abs") {
        if (tokens.size() != 2) fail("lift_abs needs exactly one absolute position in mm.");
        step.argument = tokens[1];
        step.number = number(step.argument);
      } else if (armMotion(step)) {
        std::size_t speedIndex = 0;
        if (tokens[1] == "left" || tokens[1] == "right") {
          if (tokens.size() < 3) fail("move with an arm needs a pose name.");
          step.arm = tokens[1];
          step.argument = tokens[2];
          speedIndex = 3;
        } else {
          // 兼容旧配置：move NAME [SPEED] 默认控制左臂。
          step.arm = "left";
          step.argument = tokens[1];
          speedIndex = 2;
        }
        if (tokens.size() > speedIndex + 1) fail("move has too many values.");
        // 0 表示执行时采用 [settings] 的全局默认速度。
        if (tokens.size() == speedIndex + 1) {
          step.number = number(tokens[speedIndex]);
          if (linearMotion(step)) {
            if (step.number < 1 || step.number > 250) fail("MoveL speed must be 1..250 mm/s.");
          } else if (step.number <= 0 || step.number > 1) {
            fail("MoveJ/MoveAbsJ joint speed ratio must be >0 and <=1.");
          }
        }
      } else fail("Unknown operation: " + step.operation);
      if (group) {
        const std::string resource = armMotion(step) ? step.arm :
            ((step.operation == "gripper" || step.operation == "gripper_read") ? "gripper_" + step.arm :

            ((step.operation == "lift" || step.operation == "lift_abs") ? "lift" : step.operation));
        if (resource != "wait") {
          if (concurrent) {
            const auto owner = owners.emplace(resource, branch);
            if (!owner.second && owner.first->second != branch)
              fail("Concurrent branches cannot share " + resource + "; already owned by " + owner.first->second);
          } else if (!resources.insert(resource).second)
            fail("Parallel block contains multiple commands for " + resource);
        }
        if (concurrent) ++branchSize;
        ++groupSize;
      }
      flow.steps.push_back(step);
      if (flow.steps.size() > 1000) fail("Too many steps (maximum 1000).");
    } else fail("Add [settings], [poses], or [steps] before entries.");
  }
  if (group) fail(concurrent ? "Missing branch_end/concurrent_end." : "Missing parallel_end.");
  if (input.bad()) fail("Config read failed.");
  if (flow.steps.empty()) fail("No steps configured.");
  for (const auto &pair:flow.classTopics) {
    if (!flow.topicPoints.count(pair.first)) fail("Class source must reference a topic point: "+pair.first);
    flow.topicPoints.at(pair.first).classTopic=pair.second;
  }
  for (auto &pair : flow.topicPoints) {
    if (flow.poses.count(pair.first)) fail("Static/topic point name collision: " + pair.first);
    auto found = flow.poses.find(pair.second.orientationTemplate);
    if (found == flow.poses.end()) fail("Topic point requires a static RPY template: " + pair.second.orientationTemplate);
    for (int i = 0; i < 3; ++i) pair.second.rpy[i] = found->second[i+3];
    flow.dynamicReferences.emplace(pair.first, text_points::Offset{pair.first, {0,0,0}});
  }
  for (const auto &pair : flow.offsets) {
    if (flow.poses.count(pair.first) || flow.topicPoints.count(pair.first) || flow.joints.count(pair.first))
      fail("Offset point name collision: " + pair.first);
  }
  for (const auto &pair : flow.offsets) {
    text_points::Offset ref;
    try { ref = text_points::flattenOffset(pair.first, flow.offsets); }
    catch (const std::exception &e) { fail(e.what()); }
    if (flow.topicPoints.count(ref.base)) flow.dynamicReferences.emplace(pair.first, ref);
    else if (flow.poses.count(ref.base))
      flow.poses.emplace(pair.first, text_points::applyOffset(flow.poses.at(ref.base), ref.xyz));
    else fail("Undefined offset base point: " + ref.base);
  }
  std::map<std::string, const std::array<double, 6> *> previous;
  for (const auto &step : flow.steps) {
    lineNumber = step.line;
    if (step.operation=="spin_if_class") {
      if (!flow.dynamicReferences.count(step.argument)) fail("spin_if_class requires a topic/derived point");
      const auto &root=flow.dynamicReferences.at(step.argument).base;
      if (flow.topicPoints.at(root).classTopic.empty()) fail("Configure [topic_classes] for "+root);
      continue;
    }
    if (!armMotion(step)) continue;
    if (step.operation == "moveabsj") {
      if (!flow.joints.count(step.argument)) fail("Undefined joint point: " + step.argument);
      previous.erase(step.arm); // FK is checked online; no Cartesian estimate from config.
      continue;
    }
    auto dynamic = flow.dynamicReferences.find(step.argument);
    if (dynamic != flow.dynamicReferences.end()) {
      const auto &source = flow.topicPoints.at(dynamic->second.base);
      if (source.arm != step.arm) fail("Topic point belongs to " + source.arm + ": " + step.argument);
      previous.erase(step.arm); // Actual endpoint distance is checked after resolving the topic.
      continue;
    }
    auto found = flow.poses.find(step.argument);
    if (found == flow.poses.end()) fail("Undefined pose: " + step.argument + "; fill it in [poses].");
    const auto previousForArm = previous.find(step.arm);
    if (previousForArm != previous.end()) {
      double squared = 0;
      for (int i = 0; i < 3; ++i)
        squared += std::pow((*(previousForArm->second))[i] - found->second[i], 2);
      if (std::sqrt(squared) > flow.maxMoveDistance)
        fail("Consecutive " + step.arm + " targets exceed max_move_distance_m=" +
             std::to_string(flow.maxMoveDistance) + ".");
    }
    previous[step.arm] = &found->second;
  }
  return flow;
}

constexpr double kZone = 0.0;    // 到点停止，不做转弯过渡
volatile std::sig_atomic_t interrupted = 0;
std::atomic<bool> cancelled{false};
bool stopping() { return interrupted || cancelled.load(); }
void onSignal(int) { interrupted = 1; }

void checked(const error_code &ec, const char *operation) {
  if (ec) throw std::runtime_error(std::string(operation) + ": " + ec.message());
}

void printPose(const char *label, const CartesianPosition &p) {
  std::cout << label << " XYZ(m)=[" << p.trans[0] << ", " << p.trans[1]
            << ", " << p.trans[2] << "] RPY(rad)=[" << p.rpy[0] << ", "
            << p.rpy[1] << ", " << p.rpy[2] << "] elbow=" << p.elbow << '\n';
}

double distance(const CartesianPosition &a, const CartesianPosition &b) {
  double sum = 0;
  for (int i = 0; i < 3; ++i) sum += std::pow(a.trans[i] - b.trans[i], 2);
  return std::sqrt(sum);
}

Eigen::Matrix3d rotation(const CartesianPosition &p) {
  std::array<double, 6> pose{p.trans[0], p.trans[1], p.trans[2],
                             p.rpy[0], p.rpy[1], p.rpy[2]};
  std::array<double, 16> transform{};
  Eigen::Matrix3d result;
  Eigen::Vector3d translation;
  Utils::postureToTransArray(pose, transform);
  Utils::arrayToTransMatrix(transform, result, translation);
  return result;
}

// 异常、超时或 Ctrl+C 时停止并清空本次运动；必须比 robot 先析构。
struct MotionGuard {
  ArRobot &robot;
  bool active = false;
  ~MotionGuard() {
    if (!active) return;
    cancelled.store(true);
    error_code ec;
    robot.stop(ec);
    if (ec) std::cerr << "stop failed: " << ec.message() << '\n';
    robot.moveReset(ec);
    if (ec) std::cerr << "moveReset failed: " << ec.message() << '\n';
  }
};

// 使用 Linux GPIO 字符设备接口，无需额外安装 libgpiod 开发包。
// 只申请 line 6；整个循环保持句柄，异常/正常退出前均写入 0。
class Do2Output {
 public:
  Do2Output() {
    const int chip = ::open("/dev/gpiochip0", O_RDONLY | O_CLOEXEC);
    if (chip < 0) fail("open gpiochip0");
    gpiohandle_request request{};
    request.lineoffsets[0] = 6;
    request.flags = GPIOHANDLE_REQUEST_OUTPUT;
    request.default_values[0] = 0;
    request.lines = 1;
    std::strncpy(request.consumer_label, "text-dct-do2", sizeof(request.consumer_label) - 1);
    const int result = ::ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &request);
    const int saved = errno;
    ::close(chip);
    if (result < 0) { errno = saved; fail("request DO2 line 6"); }
    fd_ = request.fd;
    std::cout << "DO2 acquired, initial output=0.\n";
  }
  Do2Output(const Do2Output &) = delete;
  Do2Output &operator=(const Do2Output &) = delete;
  ~Do2Output() {
    gpiohandle_data data{};
    if (::ioctl(fd_, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &data) < 0)
      std::cerr << "Failed to reset DO2 to 0: " << std::strerror(errno) << '\n';
    ::close(fd_);
  }
  void set(int value) {
    if (stopping()) throw std::runtime_error("Interrupted before DO2 change.");
    gpiohandle_data data{};
    data.values[0] = value;
    if (::ioctl(fd_, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &data) < 0) fail("write DO2");
    gpiohandle_data actual{};
    if (::ioctl(fd_, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &actual) < 0) fail("read DO2");
    if (actual.values[0] != value) throw std::runtime_error("DO2 readback mismatch.");
    std::cout << "DO2=" << value << std::endl;
  }
 private:
  int fd_ = -1;
  static void fail(const char *operation) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
  }
};

// Independent ROS context/executor per gripper: can run beside the lift client.
// No stop/effort interface exists in the driver; cancellation stops waiting only.
class GripperClient {
 public:
  explicit GripperClient(const std::string &side) : side_(side) {
    context_ = std::make_shared<rclcpp::Context>();
    context_->init(0, nullptr);
    rclcpp::NodeOptions options;
    options.context(context_);
    node_ = std::make_shared<rclcpp::Node>("text_dct_gripper_" + side, options);
    rclcpp::ExecutorOptions executorOptions;
    executorOptions.context = context_;
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(executorOptions);
    const std::string prefix = "/robot/gripper/" + side;
    publisher_ = node_->create_publisher<std_msgs::msg::Int32>(prefix + "/command/position", 1);
    subscription_ = node_->create_subscription<std_msgs::msg::Int32>(
        prefix + "/state", rclcpp::QoS(1), [this](std_msgs::msg::Int32::SharedPtr msg) {
          position_ = msg->data;
          ++samples_;
        });
    executor_->add_node(node_);
  }
  ~GripperClient() {
    executor_->remove_node(node_);
    context_->shutdown("gripper client finished");
  }
  int read() {
    waitSample(std::chrono::steady_clock::now() + std::chrono::seconds(5));
    std::cout << "Gripper " << side_ << " position=" << position_ << " counts" << std::endl;
    return position_;
  }
  void move(int target, double timeout, int tolerance) {
    auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (publisher_->get_subscription_count() == 0) {
      if (stopping()) throw std::runtime_error("Gripper command cancelled.");
      if (std::chrono::steady_clock::now() >= readyDeadline)
        throw std::runtime_error("No gripper command subscriber; start STS3215 driver/check ROS_DOMAIN_ID.");
      executor_->spin_once(std::chrono::milliseconds(50));
    }
    // Require a recent valid feedback sample before sending exactly one command.
    waitSample(readyDeadline);
    if (stopping()) throw std::runtime_error("Gripper command cancelled.");
    std_msgs::msg::Int32 msg;
    msg.data = target;
    publisher_->publish(msg);
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout));
    int reached = 0;
    try {
      while (std::chrono::steady_clock::now() < deadline) {
        waitSample(std::min(deadline, std::chrono::steady_clock::now() + std::chrono::seconds(2)));
        reached = std::abs(position_ - target) <= tolerance ? reached + 1 : 0;
        if (reached >= 3) {
          std::cout << "Gripper " << side_ << " reached " << position_ << " counts" << std::endl;
          return;
        }
      }
      throw std::runtime_error("Gripper target timeout.");
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string(e.what()) + " target=" + std::to_string(target) +
          " actual=" + std::to_string(position_) +
          "; driver has no stop interface; last position command remains active.");
    }
  }
 private:
  void waitSample(std::chrono::steady_clock::time_point deadline) {
    const auto previous = samples_;
    while (samples_ == previous && std::chrono::steady_clock::now() < deadline) {
      if (stopping()) throw std::runtime_error("Gripper wait cancelled.");
      executor_->spin_once(std::chrono::milliseconds(50));
    }
    if (stopping()) throw std::runtime_error("Gripper wait cancelled.");
    if (samples_ == previous) throw std::runtime_error("No fresh gripper state before deadline.");
    if (position_ < 0 || position_ > 4095) throw std::runtime_error("Gripper state outside 0..4095.");
  }
  std::string side_;
  std::shared_ptr<rclcpp::Context> context_;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  uint64_t samples_ = 0;
  int position_ = 0;
};

// 通过已经启动的 liftmotor_rs485 ROS 2 节点控制升降轴。
// 正毫米下降，负毫米上升；每条命令等待位置反馈连续到位后才返回。
class LiftMotorClient {
 public:
  LiftMotorClient() {
    rclcpp::init(0, nullptr, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
    // Executor 会创建 guard condition，必须在 rclcpp::init() 之后构造。
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    node_ = std::make_shared<rclcpp::Node>("text_dct_lift_client");
    moveClient_ = node_->create_client<Move>("/robot/lift/move_relative");
    absoluteClient_ = node_->create_client<Absolute>("/robot/lift/move_absolute");
    stopClient_ = node_->create_client<Stop>("/robot/lift/stop");
    subscription_ = node_->create_subscription<State>(
        "/robot/lift/state", rclcpp::QoS(1),
        [this](State::SharedPtr message) {
          state_ = std::move(message);
          ++stateCount_;
        });
    executor_->add_node(node_);
    if (!moveClient_->wait_for_service(std::chrono::seconds(5)) ||
        !absoluteClient_->wait_for_service(std::chrono::seconds(5)))
      throw std::runtime_error(
          "Lift service unavailable; start liftmotor_rs485 node first.");
  }

  LiftMotorClient(const LiftMotorClient &) = delete;
  LiftMotorClient &operator=(const LiftMotorClient &) = delete;

  ~LiftMotorClient() {
    if (motionActive_) stopNoThrow();
    executor_->remove_node(node_);
    node_.reset();
    if (rclcpp::ok()) rclcpp::shutdown();
  }

  void moveRelative(double distanceMm) {
    waitForNewState(std::chrono::seconds(5));
    checkStateReady();
    auto request = std::make_shared<Move::Request>();
    request->stop_generation = state_->stop_generation;
    request->distance_mm = distanceMm;
    std::cout << "Lift relative move=" << distanceMm
              << " mm (positive=down, negative=up), generation="
              << request->stop_generation << std::endl;

    motionActive_ = true;
    auto future = moveClient_->async_send_request(request);
    const auto responseResult = spinFuture(future, std::chrono::seconds(15));
    if (responseResult != rclcpp::FutureReturnCode::SUCCESS) {
      throw std::runtime_error(
          "Lift response missing/interrupted; stop requested, do not retry blindly.");
    }
    const auto response = future.get();
    if (!response->success) {
      motionActive_ = false;
      throw std::runtime_error("Lift command rejected: " + response->message);
    }
    std::cout << "Lift accepted: displacement=" << response->displacement_pulses
              << " pulses, target=" << response->target_pulses
              << " pulses. Waiting for completion..." << std::endl;
    waitForTarget(response->target_pulses, "relative");
  }

  void moveAbsolute(double positionMm) {
    waitForNewState(std::chrono::seconds(5));
    checkStateReady();

    auto request = std::make_shared<Absolute::Request>();
    request->stop_generation = state_->stop_generation;
    request->position_mm = positionMm;
    std::cout << "Lift absolute target=" << positionMm
              << " mm, generation=" << request->stop_generation << std::endl;

    motionActive_ = true;
    auto future = absoluteClient_->async_send_request(request);
    const auto responseResult = spinFuture(future, std::chrono::seconds(15));
    if (responseResult != rclcpp::FutureReturnCode::SUCCESS) {
      throw std::runtime_error(
          "Lift absolute response missing/interrupted; stop requested, do not retry blindly.");
    }
    const auto response = future.get();
    if (!response->success) {
      motionActive_ = false;
      throw std::runtime_error("Lift absolute command rejected: " + response->message);
    }
    std::cout << "Lift absolute accepted: target=" << response->target_pulses
              << " pulses. Waiting for completion..." << std::endl;
    waitForTarget(response->target_pulses, "absolute");
  }

  void abortMotion() noexcept {
    if (motionActive_) { stopNoThrow(); motionActive_ = false; }
  }

 private:
  using State = liftmotor_rs485::msg::LiftMotorState;
  using Move = liftmotor_rs485::srv::MoveRelative;
  using Absolute = liftmotor_rs485::srv::MoveAbsolute;
  using Stop = liftmotor_rs485::srv::Stop;

  template<typename FutureT>
  rclcpp::FutureReturnCode spinFuture(FutureT &future, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!stopping() && std::chrono::steady_clock::now() < deadline) {
      if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        return rclcpp::FutureReturnCode::SUCCESS;
      executor_->spin_once(std::chrono::milliseconds(50));
    }
    return stopping() ? rclcpp::FutureReturnCode::INTERRUPTED
                       : rclcpp::FutureReturnCode::TIMEOUT;
  }

  void waitForNewState(std::chrono::seconds timeout) {
    const uint64_t previous = stateCount_;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!stopping() && stateCount_ == previous &&
           std::chrono::steady_clock::now() < deadline)
      executor_->spin_once(std::chrono::milliseconds(100));
    if (stopping()) throw std::runtime_error("Interrupted while waiting for lift state.");
    if (stateCount_ == previous || !state_)
      throw std::runtime_error("No new /robot/lift/state message within 5 seconds.");
  }

  void checkStateReady() const {
    if (!state_->connected || state_->moving || state_->stop_latched ||
        state_->alarm_code != 0 || !state_->error.empty()) {
      std::ostringstream detail;
      detail << "Lift not ready: connected=" << std::boolalpha << state_->connected
             << ", moving=" << state_->moving
             << ", stop_latched=" << state_->stop_latched
             << ", alarm_code=" << state_->alarm_code
             << ", position_mm=" << state_->position_mm
             << ", error='" << state_->error << "'";
      throw std::runtime_error(detail.str());
    }
  }

  void waitForTarget(int64_t targetPulses, const char *motionType) {
    constexpr int64_t kTolerancePulses = 100;  // 当前标定下为 0.05 mm。
    constexpr int kRequiredSamples = 3;
    int reachedSamples = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline) {
      if (stopping()) throw std::runtime_error("Interrupted during lift motion.");
      waitForNewState(std::chrono::seconds(5));
      if (!state_->connected || state_->stop_latched || state_->alarm_code != 0 || !state_->error.empty())
        throw std::runtime_error("Lift fault: " + state_->error);
      const int64_t errorPulses =
          std::llabs(static_cast<int64_t>(state_->position_pulses) - targetPulses);
      // 0x6002 的运行标志可能在电机到位后仍未及时清除；连续三个
      // 高精度反馈样本在目标范围内即可确认到位。
      if (errorPulses <= kTolerancePulses)
        ++reachedSamples;
      else
        reachedSamples = 0;
      if (reachedSamples >= kRequiredSamples) {
        motionActive_ = false;
        std::cout << "Lift " << motionType << " target reached at "
                  << state_->position_mm << " mm; error=" << errorPulses
                  << " pulses." << std::endl;
        return;
      }
    }
    const int64_t current = state_ ? state_->position_pulses : 0;
    throw std::runtime_error(
        std::string("Lift ") + motionType + " target timeout: target=" +
        std::to_string(targetPulses) + ", current=" + std::to_string(current) +
        ", error=" + std::to_string(std::llabs(current - targetPulses)) + " pulses.");
  }

  void stopNoThrow() noexcept {
    try {
      if (!stopClient_->wait_for_service(std::chrono::seconds(1))) {
        std::cerr << "Lift stop service unavailable." << std::endl;
        return;
      }
      auto future = stopClient_->async_send_request(std::make_shared<Stop::Request>());
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
             std::chrono::steady_clock::now() < deadline) {
        // A stop request must still be dispatched after SIGINT/SIGTERM.
        executor_->spin_once(std::chrono::milliseconds(50));
      }
      if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready ||
          !future.get()->success)
        std::cerr << "Lift software stop was not confirmed." << std::endl;
      else
        std::cerr << "Lift software stop requested." << std::endl;
    } catch (const std::exception &error) {
      std::cerr << "Lift software stop failed: " << error.what() << std::endl;
    }
  }

  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Client<Move>::SharedPtr moveClient_;
  rclcpp::Client<Absolute>::SharedPtr absoluteClient_;
  rclcpp::Client<Stop>::SharedPtr stopClient_;
  rclcpp::Subscription<State>::SharedPtr subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  State::SharedPtr state_;
  uint64_t stateCount_ = 0;
  bool motionActive_ = false;
};

void waitSeconds(double seconds) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (stopping()) throw std::runtime_error("Interrupted during dwell.");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

// Each robot is owned by one worker at a time. All SDK calls for it stay on that worker.
void moveTo(ArRobot &robot, const Step &step, const Workflow &flow,
            const std::array<double, 6> *resolved = nullptr) {
  MotionGuard guard{robot};
  error_code ec;
  if (stopping()) throw std::runtime_error("Motion cancelled.");
  auto state = robot.operationState(ec); checked(ec, "read initial state");
  if (state != OperationState::idle) throw std::runtime_error("Robot must be idle.");
  const auto current = robot.cartPosture(CoordinateType::endInRef, ec);
  checked(ec, "read current pose");
  CartesianPosition target = current;
  const bool absolute = step.operation == "moveabsj";
  if (absolute) {
    const auto tool = robot.toolset(ec); checked(ec, "read toolset");
    target = robot.model().calcFk(flow.joints.at(step.argument), tool, ec);
    checked(ec, "joint target FK");
  } else {
    const auto &v = resolved ? *resolved : flow.poses.at(step.argument);
    for (int i = 0; i < 3; ++i) { target.trans[i] = v[i]; target.rpy[i] = v[i+3]; }
    // MoveL retains the current elbow; PTP lets the controller choose a solution.
    target.hasElbow = linearMotion(step);
  }
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(current.trans[i]) || !std::isfinite(current.rpy[i]) ||
        !std::isfinite(target.trans[i]) || !std::isfinite(target.rpy[i]))
      throw std::runtime_error("Invalid pose feedback/FK.");
  }
  if (linearMotion(step) && !std::isfinite(current.elbow))
    throw std::runtime_error("Invalid elbow feedback.");
  if (distance(current, target) > flow.maxMoveDistance)
    throw std::runtime_error("Target exceeds max_move_distance_m=" + std::to_string(flow.maxMoveDistance));
  if (stopping()) throw std::runtime_error("Motion cancelled before setup.");
  robot.setMotionControlMode(MotionControlMode::NrtCommand, ec); checked(ec, "set NrtCommand");
  robot.setOperateMode(OperateMode::automatic, ec); checked(ec, "set automatic mode");
  robot.setPowerState(true, ec); checked(ec, "power on");
  robot.setDefaultConfOpt(false, ec); checked(ec, "set configuration option");
  robot.moveReset(ec); checked(ec, "reset motion queue");
  guard.active = true;
  std::string commandId;
  const double speed = step.number > 0 ? step.number :
      (linearMotion(step) ? flow.speed : flow.jointSpeed);
  if (absolute) {
    const auto &q = flow.joints.at(step.argument);
    MoveAbsJCommand command(JointPosition(std::vector<double>(q.begin(), q.end())), flow.speed, kZone);
    command.jointSpeed = speed;
    robot.moveAppend({command}, commandId, ec);
  } else if (linearMotion(step)) {
    robot.moveAppend({MoveLCommand(target, speed, kZone)}, commandId, ec);
  } else {
    MoveJCommand command(target, flow.speed, kZone);
    command.jointSpeed = speed;
    robot.moveAppend({command}, commandId, ec);
  }
  checked(ec, "append motion");
  if (stopping()) throw std::runtime_error("Motion cancelled before start.");
  robot.moveStart(ec); checked(ec, "start motion");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  auto feedbackDeadline = deadline;
  bool completionSeen = false;
  int reachedSamples = 0;
  std::string feedbackDetail;
  while (std::chrono::steady_clock::now() < deadline) {
    if (stopping()) throw std::runtime_error("Motion cancelled.");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto event = robot.queryEventInfo(Event::moveExecution, ec);
    checked(ec, "read motion event");
    const auto id = event.find(EventInfoKey::MoveExecution::ID);
    bool finished = false;
    if (id != event.end() && std::any_cast<std::string>(id->second) == commandId) {
      const auto e = event.find(EventInfoKey::MoveExecution::Error);
      if (e != event.end()) checked(std::any_cast<error_code>(e->second), "controller motion error");
      const auto done = event.find(EventInfoKey::MoveExecution::ReachTarget);
      finished = done != event.end() && std::any_cast<bool>(done->second);
    }
    state = robot.operationState(ec); checked(ec, "read motion state");
    if (state != OperationState::idle && state != OperationState::moving)
      throw std::runtime_error("Unexpected robot state.");
    if (finished && !completionSeen) {
      completionSeen = true;
      feedbackDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    }
    if (!completionSeen) continue;
    if (state != OperationState::idle) {
      reachedSamples = 0;
      if (std::chrono::steady_clock::now() >= feedbackDeadline)
        throw std::runtime_error("Controller reported completion but arm did not become idle.");
      continue;
    }
    bool reached = false;
    if (absolute) {
      const auto q = robot.jointPos(ec); checked(ec, "read final joints");
      reached = true;
      std::ostringstream detail;
      detail << std::fixed << std::setprecision(9) << "Joint tolerance=" << flow.jointTolerance << " rad;";
      for (int i = 0; i < 7; ++i) {
        const double expected = flow.joints.at(step.argument)[i];
        const double error = std::abs(q[i] - expected);
        reached = reached && std::isfinite(q[i]) && error <= flow.jointTolerance;
        detail << "\nJ" << i + 1 << " target=" << expected
               << " actual=" << q[i] << " error=" << error << " rad";
      }
      feedbackDetail = detail.str();
    } else {
      const auto pose = robot.cartPosture(CoordinateType::endInRef, ec); checked(ec, "read final pose");
      const double angle = std::abs(Eigen::AngleAxisd(rotation(pose) * rotation(target).transpose()).angle());
      reached = distance(pose, target) <= 0.001 && angle <= 0.01;
      feedbackDetail = "position error=" + std::to_string(distance(pose, target)) +
          " m, orientation error=" + std::to_string(angle) + " rad";
    }
    reachedSamples = reached ? reachedSamples + 1 : 0;
    if (reachedSamples < 3) {
      if (std::chrono::steady_clock::now() >= feedbackDeadline)
        throw std::runtime_error("Controller finished but feedback did not settle within 2 s. " + feedbackDetail);
      continue;
    }
    robot.moveReset(ec); checked(ec, "clear completed motion");
    guard.active = false;
    return;
  }
  throw std::runtime_error("Arm motion timed out.");
}

// Launch all workers behind a gate; wait for every worker even on failure.
// This is concurrent execution, not controller-clock synchronization.
template<class Run>
void runParallel(const std::vector<Step> &steps, std::size_t begin, std::size_t end, Run run) {
  std::promise<void> gate;
  auto ready = gate.get_future().share();
  std::vector<std::future<void>> workers;
  workers.reserve(end - begin);
  std::exception_ptr failure;
  try {
    for (auto i = begin; i < end; ++i) {
      workers.push_back(std::async(std::launch::async, [&, i, ready] {
        ready.wait();
        try { if (!stopping()) run(steps[i]); }
        catch (...) { cancelled.store(true); throw; }
      }));
    }
  } catch (...) { cancelled.store(true); failure = std::current_exception(); }
  gate.set_value();
  for (auto &worker : workers) {
    try { worker.get(); } catch (...) { if (!failure) failure = std::current_exception(); }
  }
  if (failure) std::rethrow_exception(failure);
  if (stopping()) throw std::runtime_error("Parallel block interrupted.");
}

// Each branch owns its devices for the entire block and advances independently.
template<class Run>
void runBranches(const std::vector<Step> &steps, std::size_t begin, std::size_t end, Run run) {
  std::vector<Step> heads;
  for (auto i = begin; i < end; ++i)
    if (i == begin || steps[i].branch != steps[i - 1].branch) heads.push_back(steps[i]);
  runParallel(heads, 0, heads.size(), [&](const Step &head) {
    for (auto i = begin; i < end; ++i) {
      if (steps[i].branch != head.branch) continue;
      if (stopping()) throw std::runtime_error("Concurrent branch interrupted: " + head.branch);
      run(steps[i]);
    }
  });
}

}  // namespace

int main(int argc, char **argv) {
  bool checkOnly = false;
  bool loop = false;
  std::string path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--check" && !checkOnly) checkOnly = true;
    else if (arg == "--loop" && !loop) loop = true;
    else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--check] [--loop] [config.conf]\n";
      return 0;
    } else if (path.empty() && arg.compare(0, 2, "--") != 0) path = arg;
    else {
      std::cerr << "Usage: " << argv[0] << " [--check] [--loop] [config.conf]\n";
      return 2;
    }
  }
  if (path.empty()) {
    const std::string source = __FILE__;
    path = source.substr(0, source.find_last_of("/\\") + 1) + "text_dct.conf";
  }
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  std::cout << std::fixed << std::setprecision(6);
  try {
    // 完整解析/校验并保存快照；执行过程中修改文件不会改变本次流程。
    const Workflow flow = loadWorkflow(path);
    if (loop && !std::any_of(flow.steps.begin(), flow.steps.end(), [&](const Step &s) {
          return armMotion(s) && s.operation != "moveabsj" && flow.dynamicReferences.count(s.argument);
        })) throw std::runtime_error("--loop requires at least one topic point used by a motion step.");
    const bool hasLift = std::any_of(
        flow.steps.begin(), flow.steps.end(),
        [](const Step &step) {
          return step.operation == "lift" || step.operation == "lift_abs";
        });
    const bool hasLeft = std::any_of(
        flow.steps.begin(), flow.steps.end(),
        [](const Step &step) {
          return armMotion(step) && step.arm == "left";
        });
    const bool hasRight = std::any_of(
        flow.steps.begin(), flow.steps.end(),
        [](const Step &step) {
          return armMotion(step) && step.arm == "right";
        });
    std::cout << "Config: " << path << '\n';
    if (hasLeft)
      std::cout << "LEFT arm: robot=192.168.1.160, local=192.168.1.10\n";
    if (hasRight)
      std::cout << "RIGHT arm: robot=192.168.2.160, local=192.168.2.10\n";
    std::cout << "Default MoveL speed=" << flow.speed
              << " mm/s; max endpoint distance=" << flow.maxMoveDistance
              << " m; joint speed ratio=" << flow.jointSpeed
              << "; joint tolerance=" << flow.jointTolerance
              << (loop ? " rad; loop until no new target at round preflight.\n" : " rad; runs once.\n");
    for (std::size_t i = 0; i < flow.steps.size(); ++i) {
      const auto &step = flow.steps[i];
      std::cout << i + 1 << ". " << step.operation << " ";
      if (!step.arm.empty()) std::cout << step.arm << " ";
      std::cout << step.argument;
      if (step.operation=="spin_if_class") std::cout << " class=" << step.materialClass
          << " turns=" << step.number << " speed=" << step.angularSpeed << " rad/s";
      if (armMotion(step)) {
        std::cout << " [";
        if (step.operation == "moveabsj") {
          for (double value : flow.joints.at(step.argument)) std::cout << value << ' ';
        } else if (flow.dynamicReferences.count(step.argument)) {
          const auto &ref = flow.dynamicReferences.at(step.argument);
          const auto &source = flow.topicPoints.at(ref.base);
          std::cout << "runtime topic=" << source.topic << " adapter=" << source.adapter
                    << " frame=" << source.frame << " RPY=" << source.orientationTemplate;
          if (flow.offsets.count(step.argument))
            std::cout << " base=" << ref.base << " offset(m)=" << ref.xyz[0] << "," << ref.xyz[1] << "," << ref.xyz[2];
        } else {
          for (double value : flow.poses.at(step.argument)) std::cout << value << ' ';
        }
        std::cout << "] speed=" << (step.number > 0 ? step.number :
            (linearMotion(step) ? flow.speed : flow.jointSpeed))
            << (linearMotion(step) ? " mm/s" : " joint ratio");
      }
      if (step.group) {
        if (step.branch.empty()) std::cout << " [parallel " << step.group << "]";
        else std::cout << " [concurrent " << step.group << ", branch " << step.branch << "]";
      }
      std::cout << '\n';
    }
    if (checkOnly) {
      std::cout << "Config valid. No hardware accessed; reachability/path not verified.\n";
      return 0;
    }
    std::cout << "DO2 initialized to 0 before steps and reset to 0 on exit.\n"
              << (hasLift ? "Liftmotor ROS 2 node must already be running.\n" : "")
              << "Verify tool load, TCP/reference frame and clear motion paths (PTP paths are not straight).\n"
              << "Program switches to automatic mode and powers on.\n"
              << "Type MOVE to execute: " << std::flush;
    std::string answer;
    if (!std::getline(std::cin, answer) || answer != "MOVE" || stopping()) {
      std::cout << "Cancelled.\n";
      return 0;
    }
    for (std::size_t round = 1; ; ++round) {
    if (stopping()) throw std::runtime_error("Workflow cancelled before next round.");
    if (loop) std::cout << "Starting round " << round << ": acquiring new targets." << std::endl;
    std::unique_ptr<text_points::TopicPoints> topicPoints;
    std::map<std::string, text_points::Source> usedSources;
    for (const auto &step : flow.steps)
      if ((armMotion(step) || step.operation=="spin_if_class") && step.operation != "moveabsj" && flow.dynamicReferences.count(step.argument)) {
        const auto &root = flow.dynamicReferences.at(step.argument).base;
        usedSources.emplace(root, flow.topicPoints.at(root));
      }
    std::map<std::string, std::array<double, 6>> resolvedTopicPoses;
    std::map<std::string, int> materialClasses;
    if (!usedSources.empty()) {
      std::cout << "Preflight: acquiring all " << usedSources.size()
                << " topic points before any device initialization or workflow step." << std::endl;
      topicPoints = std::make_unique<text_points::TopicPoints>(usedSources);
      for (const auto &source : usedSources) {
        std::cout << "Preflight: waiting for " << source.first << " on "
                  << source.second.topic << std::endl;
        try {
          resolvedTopicPoses.emplace(source.first,
              topicPoints->resolve(source.first, [] { return stopping(); }));
          if (!source.second.classTopic.empty()) materialClasses.emplace(source.first,topicPoints->category(source.first));
        } catch (const text_points::NoTarget &e) {
          if (loop) {
            std::cout << "Loop finished: no new target for round " << round
                      << "; completed rounds=" << (round - 1)
                      << ". No steps started for this round.\n" << e.what() << std::endl;
            return 0;
          }
          throw std::runtime_error(std::string("Topic preflight failed; no workflow steps executed: ") + e.what());
        } catch (const std::exception &e) {
          throw std::runtime_error(std::string("Topic preflight failed; no workflow steps executed: ") + e.what());
        }
      }
      // The complete immutable snapshot is ready before GPIO, lift, gripper or arm setup.
      topicPoints.reset();
      for (const auto &step : flow.steps) {
        if (!armMotion(step) || step.operation == "moveabsj" || !flow.dynamicReferences.count(step.argument) ||
            resolvedTopicPoses.count(step.argument)) continue;
        const auto &ref = flow.dynamicReferences.at(step.argument);
        const auto pose = text_points::applyOffset(resolvedTopicPoses.at(ref.base), ref.xyz);
        resolvedTopicPoses.emplace(step.argument, pose);
        std::cout << "Derived point " << step.argument << " from " << ref.base << ": [";
        for (double value : pose) std::cout << value << ' ';
        std::cout << "] (m/rad)" << std::endl;
      }
      std::cout << "Preflight passed: all topic points acquired and frozen." << std::endl;
    }
    if (stopping()) throw std::runtime_error("Workflow cancelled before device initialization.");
    std::unique_ptr<text_chassis::Client> chassis;
    for (const auto &step:flow.steps) {
      if (step.operation!="spin_if_class") continue;
      const auto &root=flow.dynamicReferences.at(step.argument).base;
      if (materialClasses.at(root)==step.materialClass && !chassis)
        chassis=std::make_unique<text_chassis::Client>([] { return stopping(); });
    }
    std::unique_ptr<GripperClient> leftGripper, rightGripper;
    for (const auto &step : flow.steps) {
      if (step.operation != "gripper" && step.operation != "gripper_read") continue;
      auto &client = step.arm == "left" ? leftGripper : rightGripper;
      if (!client) client = std::make_unique<GripperClient>(step.arm);
    }
    std::unique_ptr<LiftMotorClient> lift;
    if (hasLift) lift = std::make_unique<LiftMotorClient>();
    Do2Output output;
    std::unique_ptr<ArRobot> leftRobot;
    std::unique_ptr<ArRobot> rightRobot;
    if (hasLeft)
      leftRobot = std::make_unique<ArRobot>("192.168.1.160", "192.168.1.10");
    if (hasRight)
      rightRobot = std::make_unique<ArRobot>("192.168.2.160", "192.168.2.10");
    auto run = [&](const Step &step) {
      if (stopping()) throw std::runtime_error("Workflow interrupted.");
      std::cout << "Executing line " << step.line << ": " << step.operation
                << " " << step.arm << " " << step.argument << std::endl;
      try {
        if (armMotion(step)) {
          ArRobot &robot = step.arm == "left" ? *leftRobot : *rightRobot;
          if (step.operation != "moveabsj" && flow.dynamicReferences.count(step.argument)) {
            moveTo(robot, step, flow, &resolvedTopicPoses.at(step.argument));
          } else moveTo(robot, step, flow);
        } else if (step.operation=="spin_if_class") {
          const auto &root=flow.dynamicReferences.at(step.argument).base;
          if (materialClasses.at(root)==step.materialClass) chassis->spin(step.number,step.angularSpeed);
          else std::cout<<"Skip chassis spin: "<<root<<" class="<<materialClasses.at(root)<<std::endl;
        } else if (step.operation == "gripper" || step.operation == "gripper_read") {
          auto &client = step.arm == "left" ? leftGripper : rightGripper;
          if (step.operation == "gripper_read") client->read();
          else client->move(static_cast<int>(step.number), flow.gripperTimeout, flow.gripperTolerance);
        } else if (step.operation == "do") output.set(static_cast<int>(step.number));
        else if (step.operation == "lift") lift->moveRelative(step.number);
        else if (step.operation == "lift_abs") lift->moveAbsolute(step.number);
        else waitSeconds(step.number);
      } catch (const std::exception &e) {
        cancelled.store(true);
        if (step.operation == "lift" || step.operation == "lift_abs") lift->abortMotion();
        throw std::runtime_error("Config line " + std::to_string(step.line) + " " +
            step.operation + " " + step.arm + ": " + e.what());
      }
    };
    for (std::size_t i = 0; i < flow.steps.size();) {
      const auto &step = flow.steps[i];
      if (!step.group) { run(step); ++i; continue; }
      auto end = i + 1;
      while (end < flow.steps.size() && flow.steps[end].group == step.group) ++end;
      std::cout << "Starting " << (step.branch.empty() ? "parallel" : "concurrent")
                << " block " << step.group << std::endl;
      if (step.branch.empty()) runParallel(flow.steps, i, end, run);
      else runBranches(flow.steps, i, end, run);
      i = end;
    }
    output.set(0);
    std::cout << "Workflow completed; DO2=0. Robot remains powered on.\n";
    if (!loop) return 0;
    std::cout << "Round " << round << " completed. Rechecking targets for the next round." << std::endl;
    } // Per-round devices and target snapshots are released before the next preflight.
  } catch (const std::exception &e) {
    std::cerr << "text_dct: " << e.what() << '\n';
    return 1;
  }
}
