// Run the actual session/guard code against an in-memory robot; no SDK or ROS
// node is instantiated and no network connection or motion is possible.
#include "../src/ros_cartesian_drag_safe_service.cpp"

namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
struct FakeForce {
  std::chrono::milliseconds latency{0};
  void getEndTorque(FrameType, std::array<double, 7> &joint,
                    std::array<double, 7> &external, std::array<double, 3> &torque,
                    std::array<double, 3> &force, error_code &ec) {
    std::this_thread::sleep_for(latency);
    joint.fill(0); external.fill(0); torque.fill(0); force.fill(0); ec.clear();
  }
};
struct FakeRobot {
  Toolset tool{};
  OperationState state{OperationState::idle};
  std::atomic<bool> *stopOnLoad{nullptr}, *stopOnEnable{nullptr};
  bool enableError{false}, disableError{false};
  int enabled{0}, disabled{0}, powerChanges{0}, modeChanges{0}, reads{0};
  double poseStep{0}, jointStep{0};
  OperationState operationState(error_code &ec) { ec.clear(); return state; }
  Toolset toolset(error_code &ec) { ec.clear(); return tool; }
  void setMotionControlMode(MotionControlMode, error_code &ec) { ++modeChanges; ec.clear(); }
  void setOperateMode(OperateMode, error_code &ec) { ec.clear(); }
  void setPowerState(bool value, error_code &ec) {
    require(!value, "drag must not power on automatically"); ++powerChanges; ec.clear();
  }
  void setToolset(const Toolset &value, error_code &ec) {
    tool = value; ec.clear(); if (stopOnLoad) stopOnLoad->store(true);
  }
  std::array<double, 6> posture(CoordinateType, error_code &ec) {
    ec.clear(); std::array<double, 6> result{}; result[0] = reads++ * poseStep; return result;
  }
  std::array<double, 7> jointPos(error_code &ec) {
    ec.clear(); std::array<double, 7> result{}; result[0] = reads * jointStep; return result;
  }
  void enableDrag(DragParameter::Space, DragParameter::Type, error_code &ec,
                  bool directDrag) {
    require(!directDrag, "physical end button must remain required");
    require(modeChanges > 0 && powerChanges > 0, "SDK drag preparation was skipped");
    ++enabled; state = OperationState::drag;
    ec = enableError ? std::make_error_code(std::errc::io_error) : error_code{};
    if (stopOnEnable) stopOnEnable->store(true);
  }
  void disableDrag(error_code &ec) {
    ++disabled;
    ec = disableError ? std::make_error_code(std::errc::io_error) : error_code{};
    if (!disableError) state = OperationState::idle;
  }
  FakeForce forceControl() { return {}; }
};
const auto running = [] { return true; };
const auto noReady = [] {};
void testCancellationAndCleanup() {
  {
    FakeRobot robot; std::atomic<bool> stop{true};
    auto result = runDrag(robot, 60., stop, noReady, running);
    require(result.success && result.cleaned, "early cancel must be clean");
    require(robot.enabled == 0 && robot.powerChanges == 0, "early cancel changed robot");
  }
  {
    FakeRobot robot; std::atomic<bool> stop{false}; robot.stopOnLoad = &stop;
    auto result = runDrag(robot, 60., stop, noReady, running);
    require(result.success && robot.enabled == 0, "load-stage cancel enabled drag");
    require(robot.tool.load.mass == 0., "SDK default load changed");
  }
  for (bool failure : {false, true}) {
    FakeRobot robot; std::atomic<bool> stop{false}; robot.stopOnEnable = &stop;
    robot.disableError = failure;
    auto result = runDrag(robot, 60., stop, noReady, running);
    require(robot.enabled == 1 && robot.disabled == 1, "enable/cancel skipped disableDrag");
    require(result.cleaned != failure && result.success != failure, "cleanup result was hidden");
  }
  {
    FakeRobot robot; std::atomic<bool> stop{false}; robot.enableError = true;
    auto result = runDrag(robot, 60., stop, noReady, running);
    require(!result.success && result.cleaned, "enable failure did not report clean rejection");
    require(robot.disabled == 1, "failed enable skipped cleanup");
  }
  {
    FakeRobot robot; std::atomic<bool> stop{false}; robot.state = OperationState::drag;
    auto result = runDrag(robot, 60., stop, noReady, running);
    require(result.success && robot.disabled == 1 && robot.enabled == 0,
            "existing drag must stop before a new session");
  }
}
void testReadyAndOriginalGuards() {
  {
    FakeRobot robot; FakeForce force; std::atomic<bool> stop{false}; int samples = 0;
    const auto started = std::chrono::steady_clock::now();
    auto result = monitorDrag(robot, force, std::array<double, 6>{}, .25, stop, running,
        [&](const std::array<double, 7> &) { ++samples; });
    require(result.safe && samples >= 12, "recording feedback must exceed 30 Hz");
    require(robot.reads <= 6, "recording changed the original 50 ms guard cadence");
    require(std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(240),
            "recording changed session duration");
  }
  {
    FakeRobot robot; robot.jointStep = .2; FakeForce force; std::atomic<bool> stop{false};
    int samples = 0;
    const auto started = std::chrono::steady_clock::now();
    auto result = monitorDrag(robot, force, std::array<double, 6>{}, .3, stop, running,
        [&](const std::array<double, 7> &) { ++samples; });
    require(!result.safe && samples >= 7, "high-rate telemetry bypassed the joint-speed guard");
    require(std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(95),
            "telemetry shortened the original two-monitor-sample speed debounce");
  }
  {
    FakeRobot robot; FakeForce force; std::atomic<bool> stop{false}; int samples = 0;
    auto result = monitorDrag(robot, force, std::array<double, 6>{}, .2, stop, running,
        [&](const std::array<double, 7> &joints) {
          require(joints == std::array<double, 7>{}, "recorded joints differ from SDK feedback");
          ++samples;
          if (samples == 2) stop.store(true);
        });
    require(result.safe && samples == 2, "drag telemetry must continue inside the session lock");
  }
  {
    FakeRobot robot; std::atomic<bool> stop{false}; bool ready = false;
    auto result = runDrag(robot, 60., stop, [&] { ready = true; stop.store(true); }, running);
    require(ready && result.success && result.cleaned && robot.disabled == 1,
            "ready/normal stop did not complete cleanup");
  }
  for (double step : {.6, .07}) {
    FakeRobot robot; robot.poseStep = step; FakeForce force; std::atomic<bool> stop{false};
    auto result = monitorDrag(robot, force, std::array<double, 6>{}, .3, stop, running);
    require(!result.safe, "original translation/speed guard failed to stop");
  }
  {
    FakeRobot robot; robot.jointStep = .2; FakeForce force; std::atomic<bool> stop{false};
    auto result = monitorDrag(robot, force, std::array<double, 6>{}, .3, stop, running);
    require(!result.safe, "original joint-speed guard failed to stop");
  }
}
void testSlowWrenchReadInterruptsJointPublication() {
  FakeRobot robot;
  FakeForce force;
  force.latency = std::chrono::milliseconds(120);
  std::atomic<bool> stop{false};
  std::vector<std::chrono::steady_clock::time_point> samples;
  const auto result = monitorDrag(robot, force, std::array<double, 6>{}, .24,
      stop, running, [&](const std::array<double, 7> &) {
        samples.push_back(std::chrono::steady_clock::now());
      });
  require(result.safe && samples.size() >= 5, "latency fixture did not produce telemetry");
  double maxGap = 0.;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    maxGap = std::max(maxGap, std::chrono::duration<double>(samples[i]-samples[i-1]).count());
  }
  require(maxGap >= .12, "slow wrench query should expose serial telemetry blocking");
  std::cout << "Injected 120 ms wrench read: measured maximum joint sample gap "
            << maxGap * 1000. << " ms (in-memory fixture only)\n";
}
}  // namespace
int main() {
  try {
    testCancellationAndCleanup(); testReadyAndOriginalGuards();
    testSlowWrenchReadInterruptsJointPublication();
  }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  return 0;
}
