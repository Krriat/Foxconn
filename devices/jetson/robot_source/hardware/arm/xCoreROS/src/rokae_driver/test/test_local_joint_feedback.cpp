// Offline-only test and simulator. This file never includes or creates an SDK robot.
#include "rokae_driver/local_joint_feedback.hpp"
#include <fstream>
#include <future>
#include <vector>

using namespace rokae_driver;
static void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }

static void testSlowPublisher() {
  std::promise<void> entered;
  auto first=entered.get_future();
  std::atomic<std::uint64_t> last{0};
  std::atomic<unsigned> calls{0};
  AsyncJointPublisher worker([&](const JointFeedbackSample &s) {
    if(++calls==1) { entered.set_value(); std::this_thread::sleep_for(std::chrono::milliseconds(1100)); }
    last.store(s.stamp_ns);
  });
  JointFeedbackSample s; s.stamp_ns=1; worker.offer(s);
  require(first.wait_for(std::chrono::seconds(1))==std::future_status::ready,"worker never entered publish");
  auto started=std::chrono::steady_clock::now();
  for(std::uint64_t i=2;i<=1001;++i) {s.stamp_ns=i;worker.offer(s);}
  const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  require(elapsed<.5,"blocked DDS publisher stalled the sampler queue");
  require(worker.overwritten()>=984,"ROS queue is not bounded");
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
  while(last.load()!=1001&&std::chrono::steady_clock::now()<deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  require(last.load()==1001,"worker changed or lost the latest measured timestamp");
  std::cout<<"Injected 1100 ms publication block; offered 1000 measured samples in "
           <<elapsed*1000<<" ms; ROS queue overwrites="<<worker.overwritten()<<"\n";
}

static int simulate(double seconds) {
  std::atomic<unsigned> published{0};
  AsyncJointPublisher worker([&](const JointFeedbackSample &) {
    if(published++%150==0) std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  });
  auto next=std::chrono::steady_clock::now();
  const auto deadline=next+std::chrono::duration<double>(seconds);
  unsigned samples=0;
  while(std::chrono::steady_clock::now()<deadline) {
    for(int side=0;side<2;++side) {
      JointFeedbackSample sample;sample.side=side;
      sample.stamp_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      sample.mono_ns=LocalJointFeedback::monoNow();
      for(int j=0;j<7;++j) sample.joints[j]=.1+.01*std::sin(sample.mono_ns/1e9+j);
      LocalJointFeedback::instance().offer(sample,worker.overwritten());
      worker.offer(sample);++samples;
    }
    next+=std::chrono::milliseconds(10);
    next=std::max(next,std::chrono::steady_clock::now());
    std::this_thread::sleep_until(next);
  }
  std::cout<<"SIMULATED_ONLY samples="<<samples<<" ROS_queue_overwrites="<<worker.overwritten()<<"\n";
  return 0;
}

int main(int argc,char **argv) {
  try {
    if(argc==3&&std::string(argv[1])=="--simulate") return simulate(std::stod(argv[2]));
    if(argc==3&&std::string(argv[1])=="--fixture") {
      JointFeedbackSample s{1,0,123456789012345ULL,98765432101234ULL,{.1,.2,.3,.4,.5,.6,.7}};
      const auto packet=encodeJointFeedback(s,123,456,789,2,9);
      std::ofstream f(argv[2],std::ios::binary);f.write(reinterpret_cast<const char *>(packet.data()),packet.size());
      return f?0:1;
    }
    testSlowPublisher();return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
