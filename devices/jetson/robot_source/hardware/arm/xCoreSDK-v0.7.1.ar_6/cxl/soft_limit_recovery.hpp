#pragma once

#include <algorithm>
#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "rokae/robot.h"

namespace soft_limit_recovery {
using Joints = std::array<double, 7>;
using Limits = std::array<double[2], 7>;

struct Options {
  double recoveryMargin = 0.08; // SDK 官方恢复示例使用的限位内余量，rad。
  double maxOverrun = 0.20;     // 更大越界需现场检查，不能关闭限位后长距离运动。
  double jogRate = 0.05;
  double jointSpeed = 0.10;
  std::chrono::milliseconds poll{50};
  std::chrono::milliseconds stopTimeout{5000};
  std::chrono::milliseconds jogTimeout{30000};
  std::chrono::milliseconds moveTimeout{120000};
};

// Robot 模板参数使恢复流程可以用离线假控制器验证，不连接真机。
template<class Robot>
class Recovery {
 public:
  Recovery(Robot &robot, std::ostream &output,
           std::function<bool()> cancelled = [] { return false; },
           Options options = {})
      : robot_(robot), out_(output), cancelled_(std::move(cancelled)), options_(options) {}

  void recoverAndMoveAbsJ(const Joints &target) {
    preflight(target);
    try {
      checkCancelled();
      touched_ = true;
      powerOff();
      robot_.setOperateMode(rokae::OperateMode::manual, ec_); check("切换手动模式");
      robot_.setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec_); check("切换非实时模式");
      robot_.moveReset(ec_); check("清空运动指令");

      if(needsRecovery_) {
        // 即使控制器返回错误，也按可能已关闭处理，异常清理会重新读取并恢复。
        limitsMayBeDisabled_ = true;
        robot_.setSoftLimit(false, ec_); check("暂时关闭软限位");
        powerOn();
        for(size_t axis = 0; axis < 7; ++axis) {
          checkCancelled();
          const auto current = joints();
          double goal = current[axis];
          if(goal < savedLimits_[axis][0]) goal = savedLimits_[axis][0] + options_.recoveryMargin;
          else if(goal > savedLimits_[axis][1]) goal = savedLimits_[axis][1] - options_.recoveryMargin;
          else continue;
          if(std::abs(goal-current[axis]) > options_.maxOverrun+options_.recoveryMargin+1e-6)
            throw std::runtime_error("恢复过程中关节读数变化过大");
          const double degrees = std::abs(goal-current[axis]) * 180.0 / 3.14159265358979323846;
          out_ << "恢复 J" << axis+1 << ": " << current[axis] << " -> " << goal
               << " rad，Jog " << degrees << "°" << std::endl;
          robot_.startJog(rokae::JogOpt::jointSpace, options_.jogRate,
                          degrees, axis, goal > current[axis], ec_); check("开始关节 Jog");
          waitJog(axis, current[axis], goal);
          // stop() 返回后仍可能处于停止过程；必须等 idle 后再下电或开始下一轴。
          stopAndWait();
          if(std::abs(joints()[axis]-goal) > 0.01)
            throw std::runtime_error("Jog 未到达恢复角度");
        }
        powerOff();
        restoreLimits();
      }
      verifyLimits();
      requireInside(joints());
      checkCancelled();
      robot_.setOperateMode(rokae::OperateMode::automatic, ec_); check("切换自动模式");
      powerOn();
      const auto current = joints();
      if(atTarget(current, target)) {
        out_ << "已在目标点，无需重复发送 MoveAbsJ" << std::endl;
        return;
      }
      robot_.moveReset(ec_); check("回位前清空指令");
      rokae::MoveAbsJCommand command(
          rokae::JointPosition(std::vector<double>(target.begin(), target.end())), 200, 0);
      command.jointSpeed = options_.jointSpeed;
      std::string id;
      robot_.moveAppend({command}, id, ec_); check("添加 MoveAbsJ");
      if(id.empty()) throw std::runtime_error("控制器未返回 MoveAbsJ 指令 ID");
      checkCancelled();
      robot_.moveStart(ec_); check("开始 MoveAbsJ");
      waitMove(id, target);
      out_ << "回位完成：已到达目标关节角，软限位开启，自动模式、上电、idle" << std::endl;
    } catch(...) {
      if(touched_) cleanup();
      throw;
    }
  }

 private:
  Robot &robot_;
  std::ostream &out_;
  std::function<bool()> cancelled_;
  Options options_;
  std::error_code ec_;
  Limits savedLimits_{};
  bool needsRecovery_ = false;
  bool touched_ = false;
  bool limitsMayBeDisabled_ = false;

  void check(const char *action) {
    if(ec_) throw std::runtime_error(std::string(action)+": "+ec_.message()+" ("+std::to_string(ec_.value())+")");
  }
  void checkCancelled() {
    if(cancelled_()) throw std::runtime_error("用户请求停止");
  }
  Joints joints() {
    auto q = robot_.jointPos(ec_); check("读取关节角");
    for(double value : q) if(!std::isfinite(value)) throw std::runtime_error("控制器返回非有限关节角");
    return q;
  }
  void requireInside(const Joints &q) {
    for(size_t i=0; i<7; ++i)
      if(q[i] < savedLimits_[i][0] || q[i] > savedLimits_[i][1])
        throw std::runtime_error("J"+std::to_string(i+1)+" 仍在原软限位外");
  }
  static bool atTarget(const Joints &q, const Joints &target) {
    for(size_t i=0; i<7; ++i) if(std::abs(q[i]-target[i]) > 0.01) return false;
    return true;
  }
  void preflight(const Joints &target) {
    auto state = robot_.operationState(ec_); check("读取运行状态");
    if(state != rokae::OperationState::idle) throw std::runtime_error("机械臂不是 idle，请先结束其他运动或拖动程序");
    auto power = robot_.powerState(ec_); check("读取上电状态");
    if(power != rokae::PowerState::on && power != rokae::PowerState::off)
      throw std::runtime_error("急停或安全门状态未恢复");
    auto mode = robot_.operateMode(ec_); check("读取手自动模式");
    if(mode != rokae::OperateMode::manual && mode != rokae::OperateMode::automatic)
      throw std::runtime_error("手自动模式未知");
    waitIdle();
    const bool enabled = robot_.getSoftLimit(savedLimits_, ec_); check("读取软限位");
    if(!enabled) throw std::runtime_error("软限位原本关闭，请先在控制器检查并恢复");
    const auto q = joints();
    for(size_t i=0; i<7; ++i) {
      const double lo = savedLimits_[i][0], hi = savedLimits_[i][1];
      if(!std::isfinite(lo) || !std::isfinite(hi) || hi-lo <= 2*options_.recoveryMargin)
        throw std::runtime_error("J"+std::to_string(i+1)+" 软限位设置无效");
      if(!std::isfinite(target[i]) || target[i] < lo+0.01 || target[i] > hi-0.01)
        throw std::runtime_error("J"+std::to_string(i+1)+" 回位目标不在软限位内或过于靠近边界");
      if(q[i] < lo-options_.maxOverrun || q[i] > hi+options_.maxOverrun)
        throw std::runtime_error("J"+std::to_string(i+1)+" 越界超过 0.20 rad，需现场处理");
      needsRecovery_ = needsRecovery_ || q[i] < lo || q[i] > hi;
      out_ << "J" << i+1 << " 当前=" << q[i] << " 目标=" << target[i]
           << " 限位=[" << lo << ", " << hi << "] rad" << std::endl;
    }
  }
  void powerOff() {
    robot_.setPowerState(false, ec_); check("下电");
    auto power = robot_.powerState(ec_); check("确认下电");
    if(power != rokae::PowerState::off) throw std::runtime_error("控制器未确认下电");
  }
  void powerOn() {
    checkCancelled();
    robot_.setPowerState(true, ec_); check("上电（若有外接使能开关，手动 Jog 期间需按住）");
    auto power = robot_.powerState(ec_); check("确认上电");
    if(power != rokae::PowerState::on) throw std::runtime_error("控制器未确认上电");
  }
  void waitIdle(bool respectCancellation = true) {
    const auto deadline = std::chrono::steady_clock::now()+options_.stopTimeout;
    unsigned stable = 0;
    while(std::chrono::steady_clock::now() < deadline) {
      if(respectCancellation) checkCancelled();
      auto state = robot_.operationState(ec_); check("等待 idle");
      auto velocity = robot_.jointVel(ec_); check("读取关节速度");
      bool stopped = state == rokae::OperationState::idle;
      for(double v : velocity) stopped = stopped && std::isfinite(v) && std::abs(v) < 0.01;
      stable = stopped ? stable+1 : 0;
      if(stable >= 2) return;
      std::this_thread::sleep_for(options_.poll);
    }
    throw std::runtime_error("停止后未在超时内确认 idle 和低关节速度");
  }
  void stopAndWait() {
    robot_.stop(ec_); check("停止 Jog");
    waitIdle();
  }
  void waitJog(size_t axis, double start, double goal) {
    const auto deadline = std::chrono::steady_clock::now()+options_.jogTimeout;
    while(std::chrono::steady_clock::now() < deadline) {
      checkCancelled();
      const auto q = joints();
      const double sign = goal > start ? 1.0 : -1.0;
      if(sign*(q[axis]-start) < -0.01 || sign*(q[axis]-goal) > 0.02)
        throw std::runtime_error("Jog 读回方向或角度异常");
      auto state = robot_.operationState(ec_); check("等待 Jog 完成");
      if((state == rokae::OperationState::jog || state == rokae::OperationState::idle)
          && std::abs(q[axis]-goal) <= 0.005) return;
      if(state != rokae::OperationState::jog && state != rokae::OperationState::jogging
          && state != rokae::OperationState::idle)
        throw std::runtime_error("控制器异常退出 Jog 状态");
      std::this_thread::sleep_for(options_.poll);
    }
    throw std::runtime_error("Jog 超时或未到达恢复角度");
  }
  void verifyLimits() {
    Limits actual{};
    const bool enabled = robot_.getSoftLimit(actual, ec_); check("核验软限位");
    if(!enabled) throw std::runtime_error("软限位仍然关闭");
    for(size_t i=0; i<7; ++i) for(size_t j=0; j<2; ++j)
      if(!std::isfinite(actual[i][j]) || std::abs(actual[i][j]-savedLimits_[i][j]) > 1e-9)
        throw std::runtime_error("软限位数值与运行前不一致");
  }
  void restoreLimits() {
    requireInside(joints());
    robot_.setSoftLimit(true, ec_, savedLimits_); check("恢复原软限位");
    verifyLimits();
    limitsMayBeDisabled_ = false;
    out_ << "原软限位已恢复并读回确认" << std::endl;
  }
  void waitMove(const std::string &id, const Joints &target) {
    using namespace rokae::EventInfoKey::MoveExecution;
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started+options_.moveTimeout;
    while(std::chrono::steady_clock::now() < deadline) {
      checkCancelled();
      auto info = robot_.queryEventInfo(rokae::Event::moveExecution, ec_); check("读取 MoveAbsJ 执行信息");
      auto found = info.find(ID);
      if(found != info.end() && std::any_cast<std::string>(found->second) == id) {
        auto error = info.find(Error);
        if(error != info.end()) {
          const auto motionError = std::any_cast<std::error_code>(error->second);
          if(motionError) throw std::runtime_error("MoveAbsJ 执行失败: "+motionError.message()+" ("+std::to_string(motionError.value())+")");
        }
        auto reach = info.find(ReachTarget);
        if(reach != info.end() && std::any_cast<bool>(reach->second)) {
          waitIdle();
          const auto final = joints();
          requireInside(final);
          if(!atTarget(final, target)) throw std::runtime_error("MoveAbsJ 回报完成，但读回角度未到目标");
          verifyLimits();
          return;
        }
      }
      auto state = robot_.operationState(ec_); check("读取回位运行状态");
      if(state != rokae::OperationState::idle && state != rokae::OperationState::moving)
        throw std::runtime_error("控制器异常退出 MoveAbsJ 运动状态");
      if(state == rokae::OperationState::idle
          && std::chrono::steady_clock::now()-started > std::chrono::seconds(3))
        throw std::runtime_error("机械臂已停止，但未确认 MoveAbsJ 到达目标");
      requireInside(joints());
      std::this_thread::sleep_for(options_.poll);
    }
    throw std::runtime_error("MoveAbsJ 超时，未确认到达目标");
  }
  void cleanup() noexcept {
    // 清理必须忽略 Ctrl+C，完成停止、下电、恢复限位后才退出。
    try {
      robot_.stop(ec_);
      if(ec_) out_ << "异常清理停止返回：" << ec_.message() << std::endl;
      try { waitIdle(false); }
      catch(const std::exception &e) { out_ << "停止确认失败：" << e.what() << std::endl; }
      powerOff();
      robot_.setOperateMode(rokae::OperateMode::manual, ec_); check("异常清理切换手动模式");
      robot_.moveReset(ec_); check("异常清理清空指令");
      if(limitsMayBeDisabled_) restoreLimits();
      else verifyLimits();
      out_ << "已停止并下电，原软限位开启；请检查本次失败原因" << std::endl;
    } catch(const std::exception &e) {
      out_ << "异常清理未完成：" << e.what()
           << "；请在控制器确认停止、下电及软限位状态" << std::endl;
    }
  }
};
} // namespace soft_limit_recovery
