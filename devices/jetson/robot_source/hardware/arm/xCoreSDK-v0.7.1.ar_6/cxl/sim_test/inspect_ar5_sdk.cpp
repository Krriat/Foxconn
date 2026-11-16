// Read-only SDK collector. No mode/power/motion/limit/protection setters.
// Use only while idle and with other control clients exited: robot.h explicitly
// documents that disconnectFromRobot stops motion; SDK destructor effects are opaque.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "rokae/robot.h"

using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void onSignal(int) { interrupted = 1; }
long long ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
void check(const error_code &ec, const char *op) {
  if(ec) throw std::runtime_error(std::string(op)+": "+ec.message());
}
std::string quote(const std::string &s) {
  std::ostringstream o; o << '"';
  for(unsigned char c:s) {
    if(c=='"' || c=='\\') o << '\\' << c;
    else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else o << c;
  }
  o << '"'; return o.str();
}
template<class T> std::string arrayJson(const T &a) {
  std::ostringstream o; o << std::setprecision(17) << '[';
  bool first=true;
  for(const auto &v:a) { if(!first)o<<','; first=false; if(std::isfinite(v))o<<v; else o<<"null"; }
  o<<']'; return o.str();
}
struct Frame {
  long long seq=0, rx=0;
  unsigned bytes=0;
  std::array<double,7> q{}, v{}, tau{};
  bool identical=false;
};
// Single producer/single consumer ring. Disk activity cannot block the receiver.
class Writer {
  static constexpr size_t capacity=4096;
  std::vector<Frame> ring{capacity};
  std::atomic<size_t> head{0},tail{0};
  std::atomic<bool> done{false}, failed{false};
  std::ofstream file;
  std::thread worker;
public:
  explicit Writer(const std::string &path):file(path) {
    if(!file) throw std::runtime_error("Cannot create native frame log");
    worker=std::thread([this] {
      try {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        while(!done.load() || tail.load()!=head.load()) {
          auto t=tail.load();
          if(t==head.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
          const Frame f=ring[t]; tail.store((t+1)%capacity);
          file<<"{\"kind\":\"sample\",\"sample_seq\":"<<f.seq<<",\"host_rx_ns\":"<<f.rx
              <<",\"received_bytes\":"<<f.bytes<<",\"identical_payload\":"<<(f.identical?"true":"false")
              <<",\"q\":"<<arrayJson(f.q)<<",\"qd\":"<<arrayJson(f.v)<<",\"tau\":"<<arrayJson(f.tau)<<"}\n";
        }
        file.flush();
      } catch(...) { failed.store(true); }
    });
  }
  bool push(const Frame &f) {
    auto h=head.load(), next=(h+1)%capacity;
    if(failed.load() || next==tail.load())return false;
    ring[h]=f; head.store(next); return true;
  }
  bool finish() { done.store(true); if(worker.joinable())worker.join(); return !failed.load(); }
  ~Writer() { finish(); }
};

int main(int argc,char **argv) {
  if(argc!=8 || std::string(argv[1])!="--connect-idle-read-only") {
    std::cerr<<"Usage: inspect_ar5_sdk --connect-idle-read-only ROBOT_IP LOCAL_IP SECONDS INTERVAL_MS OUT_DIR qv|qvt\n";
    return 2;
  }
  std::signal(SIGINT,onSignal); std::signal(SIGTERM,onSignal);
  try {
    size_t used=0;
    const std::string secondsArg=argv[4], intervalArg=argv[5];
    const double seconds=std::stod(secondsArg,&used);
    if(used!=secondsArg.size() || !std::isfinite(seconds) || seconds<10 || seconds>600) throw std::runtime_error("SECONDS must be 10..600");
    const int interval=std::stoi(intervalArg,&used);
    if(used!=intervalArg.size() || (interval!=1 && interval!=2 && interval!=4 && interval!=8 && interval!=1000)) throw std::runtime_error("Unsupported packet interval");
    const std::string fieldsArg=argv[7];
    if(fieldsArg!="qv" && fieldsArg!="qvt") throw std::runtime_error("Use qv or qvt");
    const bool torque=fieldsArg=="qvt";
    const std::string out=argv[6];
    for(const char *name:{"robot_info.json","native_events.jsonl","frames.jsonl"})
      if(std::filesystem::exists(out+"/"+name)) throw std::runtime_error("Refusing to overwrite existing acquisition: "+out);
    // Output directory is fresh and created by the Python wrapper.
    std::ofstream info(out+"/robot_info.json"), events(out+"/native_events.jsonl");
    if(!info || !events) throw std::runtime_error("Cannot create outputs");
    auto event=[&](const std::string &type,const std::string &reason) {
      events<<"{\"host_ns\":"<<ns()<<",\"type\":"<<quote(type)<<",\"reason\":"<<quote(reason)<<"}\n"; events.flush();
    };
    Writer writer(out+"/frames.jsonl");
    rokae::ArRobot robot;
    bool receiving=false;
    try {
      robot.connectToRobot(argv[2],argv[3]);
      error_code ec;
      auto state=robot.operationState(ec); check(ec,"operationState");
      if(state!=rokae::OperationState::idle) throw std::runtime_error("Inspection requires idle robot; no attempt to stop or take control");
      auto ri=robot.robotInfo(ec); check(ec,"robotInfo");
      if(ri.joint_num!=7) throw std::runtime_error("Expected 7 joints");
      auto mode=robot.operateMode(ec); check(ec,"operateMode");
      auto power=robot.powerState(ec); check(ec,"powerState");
      info<<std::setprecision(17)<<"{\"type\":"<<quote(ri.type)<<",\"controller_version\":"<<quote(ri.version)
          <<",\"sdk_version\":"<<quote(rokae::BaseRobot::sdkVersion())<<",\"joint_num\":"<<ri.joint_num
          <<",\"operation_state\":"<<int(state)<<",\"operate_mode\":"<<int(mode)<<",\"power_state\":"<<int(power);
      std::array<double[2],7> limits{};
      bool enabled=robot.getSoftLimit(limits,ec);
      info<<",\"soft_limits\":";
      if(ec) { info<<"null"; event("field_unavailable","getSoftLimit: "+ec.message()); }
      else {
        info<<"{\"enabled\":"<<(enabled?"true":"false")<<",\"lower_rad\":";
        std::array<double,7> lo{},hi{};
        for(size_t i=0;i<7;++i){lo[i]=limits[i][0];hi[i]=limits[i][1];}
        info<<arrayJson(lo)<<",\"upper_rad\":"<<arrayJson(hi)<<'}';
      }
      ec.clear(); auto tool=robot.toolset(ec);
      info<<",\"tool\":";
      if(ec) { info<<"null"; event("field_unavailable","toolset: "+ec.message()); }
      else {
        info<<"{\"mass_kg\":";
        if(std::isfinite(tool.load.mass)) info<<tool.load.mass; else info<<"null";
        info<<",\"cog_m\":"<<arrayJson(tool.load.cog)<<",\"inertia_kg_m2\":"<<arrayJson(tool.load.inertia)
            <<",\"end_trans_m\":"<<arrayJson(tool.end.trans)<<",\"end_rpy_rad\":"<<arrayJson(tool.end.rpy)
            <<",\"ref_trans_m\":"<<arrayJson(tool.ref.trans)<<",\"ref_rpy_rad\":"<<arrayJson(tool.ref.rpy)<<'}';
      }
      info<<"}\n"; info.flush();
      std::vector<std::string> fields{rokae::RtSupportedFields::jointPos_m,rokae::RtSupportedFields::jointVel_m};
      if(torque)fields.push_back(rokae::RtSupportedFields::tau_m);
      robot.startReceiveRobotState(std::chrono::milliseconds(interval),fields); receiving=true;
      auto start=Clock::now(), lastMode=start;
      Frame previous{}; long long seq=0;
      while(std::chrono::duration<double>(Clock::now()-start).count()<seconds) {
        if(interrupted) throw std::runtime_error("operator_interrupt");
        Frame f; f.bytes=robot.updateRobotState(std::chrono::milliseconds(20)); f.rx=ns();
        if(!f.bytes) { event("receive_timeout","No packet in 20ms; no cached sample logged"); continue; }
        f.seq=seq++;
        if(robot.getStateData(rokae::RtSupportedFields::jointPos_m,f.q)!=0 ||
           robot.getStateData(rokae::RtSupportedFields::jointVel_m,f.v)!=0)
          throw std::runtime_error("Position/velocity field unavailable");
        f.tau.fill(NAN);
        if(torque && robot.getStateData(rokae::RtSupportedFields::tau_m,f.tau)!=0) throw std::runtime_error("Torque field unavailable: rerun explicitly with --without-torque");
        f.identical=f.seq>0 && f.q==previous.q && f.v==previous.v && (!torque || f.tau==previous.tau);
        bool valid=true;
        for(size_t i=0;i<7;++i) valid=valid && std::isfinite(f.q[i]) && std::isfinite(f.v[i]) && (!torque || std::isfinite(f.tau[i]));
        if(!writer.push(f)) throw std::runtime_error("Log queue overflow or disk write failure; acquisition aborted");
        if(!valid) throw std::runtime_error("Nonfinite feedback; raw sample preserved as null");
        previous=f;
        if(Clock::now()-lastMode>std::chrono::milliseconds(250)) {
          state=robot.operationState(ec); check(ec,"operationState");
          if(state!=rokae::OperationState::idle) throw std::runtime_error("Robot left idle; inspection aborted");
          lastMode=Clock::now();
        }
      }
      robot.stopReceiveRobotState(); receiving=false;
      if(!writer.finish()) throw std::runtime_error("Disk write failure");
      if(seq==0) throw std::runtime_error("No feedback packets collected");
      event("acquisition_completed","No motion commands issued; device sample identity remains unavailable");
      return 0;
    } catch(const std::exception &e) {
      event("acquisition_aborted",e.what());
      if(receiving)robot.stopReceiveRobotState();
      writer.finish();
      throw;
    }
  } catch(const std::exception &e) {
    std::cerr<<e.what()<<'\n'; return 1;
  }
}
