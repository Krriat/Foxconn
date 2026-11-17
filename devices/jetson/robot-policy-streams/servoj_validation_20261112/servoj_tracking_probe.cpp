// Explicitly authorized, bounded ServoJ diagnostic. No model, ROS or suction IO.
// Moves only J6 by +0.004 rad and back, at 100 Hz, then stops the session.
#include "latest_feedback.hpp"
#include <rokae/robot.h>
#include <rokae/motion_control_rt.h>
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Joints = std::array<double, 7>;
volatile std::sig_atomic_t interrupted = 0;
void signalHandler(int) { interrupted = 1; }
void check(const std::error_code &ec) { if (ec) throw std::runtime_error(ec.message()); }

double offset(double t) {
  constexpr double pi = 3.14159265358979323846;
  if (t < 1.) return 0.;
  if (t < 3.) return .002 * (1. - std::cos(pi * (t - 1.) / 2.));
  if (t < 5.) return .004;
  if (t < 7.) return .002 * (1. + std::cos(pi * (t - 5.) / 2.));
  return 0.;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    double prev=offset(0), maxSpeed=0, maxAccel=0, prevSpeed=0;
    for (int i=1;i<=800;++i) {
      const auto x=offset(i*.01); const auto speed=(x-prev)/.01;
      if (x<0 || x>.004000001) return 1;
      maxSpeed=std::max(maxSpeed,std::abs(speed));
      maxAccel=std::max(maxAccel,std::abs(speed-prevSpeed)/.01);
      prev=x;prevSpeed=speed;
    }
    if (offset(0)!=0 || offset(8)!=0 || maxSpeed>.0032 || maxAccel>.005) return 1;
    std::cout << "Trajectory checks passed; max_speed="<<maxSpeed<<" max_accel="<<maxAccel<<'\n';
    return 0;
  }
  if (argc!=5 || (std::string(argv[1])!="left" && std::string(argv[1])!="right") ||
      (std::string(argv[2])!="0.01" && std::string(argv[2])!="0.02") ||
      (std::string(argv[3])!="--inspect" && std::string(argv[3])!="--run")) return 2;
  std::signal(SIGINT,signalHandler);std::signal(SIGTERM,signalHandler);
  const bool left=std::string(argv[1])=="left", execute=std::string(argv[3])=="--run";
  const double lookahead=std::stod(argv[2]);
  Json log={{"side",argv[1]},{"execute",execute},{"period_s",.01},{"lookahead_s",lookahead},
            {"gain",0.},{"joint_1_based",6},{"amplitude_rad",.004},{"duration_s",8.},
            {"samples",Json::array()},{"cleanup_errors",Json::array()}};
  bool receiving=false, realtime=false, configured=false, started=false, poweredByProbe=false;
  bool passed=false;
  std::shared_ptr<rokae::RtMotionControlCobot<7>> controller;
  std::unique_ptr<rokae::ArRobot> robot;
  auto save=[&] { std::ofstream f(argv[4]);f<<log.dump(2)<<'\n'; };
  try {
    robot=std::make_unique<rokae::ArRobot>(left?"192.168.1.160":"192.168.2.160",left?"192.168.1.10":"192.168.2.10");
    std::error_code ec;
    const auto initialPower=robot->powerState(ec);check(ec);
    const auto state=robot->operationState(ec);check(ec);
    Joints anchor=robot->jointPos(ec);check(ec);
    std::array<double[2],7> limits{};
    const bool limitsEnabled=robot->getSoftLimit(limits,ec);check(ec);
    log["initial_power_state"]=static_cast<int>(initialPower);
    log["initial_operation_state"]=static_cast<int>(state);
    log["initial_joints"]=anchor;log["soft_limits_enabled"]=limitsEnabled;
    if (state!=rokae::OperationState::idle) throw std::runtime_error("Arm is not idle");
    if (initialPower!=rokae::PowerState::on && initialPower!=rokae::PowerState::off)
      throw std::runtime_error("Controller is estopped, guarded, or unknown; cannot run probe");
    if (!limitsEnabled) throw std::runtime_error("Soft limits disabled");
    for (unsigned j=0;j<7;++j) {
      if (!std::isfinite(anchor[j]) || anchor[j]-.21<limits[j][0] || anchor[j]+.21+(j==5?.004:0)>limits[j][1])
        throw std::runtime_error("Insufficient joint soft-limit margin");
    }
    save();
    if (execute) {
      robot->setOperateMode(rokae::OperateMode::automatic,ec);check(ec);
      robot->setMotionControlMode(rokae::MotionControlMode::NrtCommand,ec);check(ec);
      if (initialPower==rokae::PowerState::off) {
        robot->setPowerState(true,ec);check(ec);poweredByProbe=true;
        const auto power = robot->powerState(ec);check(ec);
        if (power!=rokae::PowerState::on) throw std::runtime_error("Power-on not confirmed");
      }
      robot->setMotionControlMode(rokae::MotionControlMode::RtCommand,ec);check(ec);realtime=true;
      controller=robot->getRtMotionController().lock();
      if (!controller) throw std::runtime_error("No realtime controller");
      robot->startReceiveRobotState(std::chrono::milliseconds(1),{rokae::RtSupportedFields::jointPos_m});receiving=true;
      controller->setServoJoint(.01,lookahead,0.,ec);check(ec);configured=true;
      const auto drain=[&](int budget) {
        return latest_feedback::drain([&]{return robot->updateRobotState(Clock::duration::zero());},
                                     []{return Clock::now();},std::chrono::milliseconds(budget));
      };
      drain(50);
      if (robot->getStateData(rokae::RtSupportedFields::jointPos_m,anchor)!=0) throw std::runtime_error("Missing anchor feedback");
      log["anchor"]=anchor;
      controller->startMove(rokae::RtControllerMode::jointPosition);started=true;
      auto next=Clock::now(), start=next, previous=next, lastRead=next;
      Joints lastCommand=anchor;
      for (int tick=0;tick<=800;++tick) {
        if (interrupted) throw std::runtime_error("Probe interrupted");
        std::this_thread::sleep_until(next);
        const auto now=Clock::now();
        const double actualDt=std::chrono::duration<double>(now-previous).count();previous=now;
        if (tick && (actualDt>.03 || actualDt<.001)) throw std::runtime_error("Unsafe probe scheduling interval");
        const auto frames=drain(tick==0?50:5);
        if (frames.frames) lastRead=Clock::now();
        if (!latest_feedback::fresh(lastRead,Clock::now())) throw std::runtime_error("Stale feedback");
        Joints measured;
        if (robot->getStateData(rokae::RtSupportedFields::jointPos_m,measured)!=0) throw std::runtime_error("Missing joint feedback");
        Joints target=anchor;target[5]+=offset(tick*.01);
        for (unsigned j=0;j<7;++j) {
          if (!std::isfinite(measured[j]) || std::abs(measured[j]-anchor[j])>.008 ||
              std::abs(target[j]-lastCommand[j])>.00005 || std::abs(target[j]-measured[j])>.01 ||
              target[j]<limits[j][0]+.21 || target[j]>limits[j][1]-.21)
            throw std::runtime_error("Probe displacement, tracking or soft-limit guard exceeded");
        }
        controller->sendCommand(rokae::JointPosition(std::vector<double>(target.begin(),target.end())));
        lastCommand=target;
        log["samples"].push_back({{"tick",tick},{"time_s",std::chrono::duration<double>(now-start).count()},
              {"dt_s",actualDt},{"command",target},{"measured",measured},{"frames_drained",frames.frames}});
        next+=std::chrono::milliseconds(10);
      }
      log["completed_trajectory"]=true;
    }
    passed=true;
  } catch (const std::exception &e) { log["error"]=e.what(); }
  const auto cleanup=[&](const char *name, auto fn) {
    try { fn(); } catch(const std::exception &e) { passed=false;log["cleanup_errors"].push_back(std::string(name)+": "+e.what()); }
  };
  if (started) cleanup("stopMove",[&]{controller->stopMove();std::this_thread::sleep_for(std::chrono::milliseconds(100));});
  if (configured) cleanup("stopServoJoint",[&]{controller->stopServoJoint();});
  if (receiving) cleanup("stopReceive",[&]{robot->stopReceiveRobotState();});
  if (realtime) cleanup("restore_NRT",[&]{std::error_code ec;robot->setMotionControlMode(rokae::MotionControlMode::NrtCommand,ec);check(ec);});
  if (robot) cleanup("final_state",[&]{
    std::error_code ec;
    const auto state=robot->operationState(ec);check(ec);
    const auto power=robot->powerState(ec);check(ec);
    log["power_after_servo_stop"]=static_cast<int>(power);
    if (execute && power!=rokae::PowerState::on) throw std::runtime_error("Unexpected power loss during probe");
    if (state!=rokae::OperationState::idle) throw std::runtime_error("Arm not idle after stop");
    if (poweredByProbe) { robot->setPowerState(false,ec);check(ec); }
    log["final_power_state"]=static_cast<int>(robot->powerState(ec));check(ec);
    log["final_operation_state"]=static_cast<int>(robot->operationState(ec));check(ec);
    log["final_joints"]=robot->jointPos(ec);check(ec);
  });
  log["passed"]=passed;save();
  std::cout << Json({{"passed",passed},{"side",argv[1]},{"lookahead_s",lookahead},
                    {"samples",log["samples"].size()},{"error",log.value("error",std::string())},
                    {"cleanup_errors",log["cleanup_errors"]}}).dump()<<'\n';
  return passed?0:1;
}
