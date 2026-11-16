#!/usr/bin/env python3
"""Offline parser and concurrent scheduler checks; never access hardware."""
from pathlib import Path
import tempfile, subprocess
root=Path(__file__).resolve().parent.parent
exe=root/'build/bin/text_dct'
base='[settings]\nspeed_mm_s 50\njoint_speed_ratio 0.2\n[poses]\np 0 0 0 0 0 0\nr 1 0 0 0 0 0\n[joints]\nq 0 0 0 0 0 0 0\n[steps]\n'
cases=[('all', 'movel left p 30\nmovej left p 0.2\nmoveabsj right q 0.1\nparallel_begin\nmove left p\nmovej right r\nlift -1\ndo 1\nwait 0.1\nparallel_end\nwait 1\n',True),('same_arm','parallel_begin\nmove left p\nmovej left p\nparallel_end\n',False),('same_lift','parallel_begin\nlift 1\nlift_abs 1\nparallel_end\n',False),('same_do','parallel_begin\ndo 0\ndo 1\nparallel_end\n',False),('nested','parallel_begin\nparallel_begin\n',False),('unclosed','parallel_begin\nwait 1\n',False),('unmatched','parallel_end\n',False),('empty','parallel_begin\nparallel_end\n',False),('unknown_joint','moveabsj right missing\n',False),('ratio','movej left p 30\n',False),('zero_ratio','moveabsj left q 0\n',False),('negative_ratio','movej left p -1\n',False),('linear_speed','movel left p 0.2\n',False),('distance','movej left p\nmovej left r\n',False),('after_joints','move left p\nmoveabsj left q\nmove left r\n',True),('legacy','move p 30\n',True)]
cases += [
 ('grippers', 'gripper_read left\ngripper_read right\ngripper left 0\ngripper right 4095\n', True),
 ('gripper_parallel', 'parallel_begin\nmove left p\ngripper left 1000\ngripper right 2000\nparallel_end\n', True),
 ('gripper_conflict', 'parallel_begin\ngripper left 1000\ngripper_read left\nparallel_end\n', False),
 ('gripper_side', 'gripper middle 1000\n', False),
 ('gripper_fraction', 'gripper left 1.5\n', False),
 ('gripper_range', 'gripper left 4096\n', False),
 ('gripper_negative', 'gripper right -1\n', False),
 ('gripper_missing', 'gripper left\n', False),
 ('gripper_read_extra', 'gripper_read right 1\n', False),
]
concurrent = 'concurrent_begin\nbranch a\n{}\nbranch_end\nbranch b\n{}\nbranch_end\nconcurrent_end\n'
cases += [
 ('branches_repeat', concurrent.format('move left p\nwait 1\nmove left p', 'gripper right 1000\ngripper right 2000'), True),
 ('branches_conflict', concurrent.format('move left p', 'movej left p'), False),
 ('branches_gripper_conflict', concurrent.format('gripper_read left', 'gripper left 1000'), False),
 ('branches_lift_conflict', concurrent.format('lift 1', 'lift_abs 1'), False),
 ('branches_do_conflict', concurrent.format('do 0', 'do 1'), False),
 ('branches_empty', concurrent.format('', 'wait 1'), False),
 ('branches_distance', concurrent.format('move left p\nmove left r', 'wait 1'), False),
 ('branches_unclosed', 'concurrent_begin\nbranch a\nwait 1\n', False),
 ('branches_outside', 'branch a\nwait 1\nbranch_end\n', False),
 ('branches_missing_wrapper', 'concurrent_begin\nwait 1\n', False),
 ('branches_duplicate', concurrent.format('wait 1','wait 1').replace('branch b','branch a'), False),
 ('branches_nested', concurrent.format('parallel_begin\nwait 1\nparallel_end','wait 1'), False),
 ('branches_followed_by_parallel', concurrent.format('wait 0','wait 0')+'parallel_begin\nmove left p\nmove right p\nparallel_end\n', True),
]
with tempfile.TemporaryDirectory() as d:
 for name,steps,ok in cases:
  path=Path(d)/f'{name}.conf';path.write_text(base+steps)
  r=subprocess.run([str(exe),'--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(name,r.stdout,r.stderr)
  print(name,'PASS')
 for value,ok in [('0.002',True),('0',False),('-0.001',False),('0.011',False),('nan',False),('0.002\njoint_tolerance_rad 0.003',False)]:
  path=Path(d)/'tolerance.conf'
  path.write_text(base.replace('[settings]','[settings]\njoint_tolerance_rad '+value)+'moveabsj right q\n')
  r=subprocess.run([str(exe),'--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(value,r.stdout,r.stderr)
 print('joint tolerance config PASS')
 for settings,ok in [
  ('gripper_timeout_s 10\ngripper_tolerance_counts 20', True),
  ('gripper_timeout_s 0', False), ('gripper_timeout_s nan', False),
  ('gripper_timeout_s 121', False), ('gripper_tolerance_counts 0.5', False),
  ('gripper_tolerance_counts -1', False), ('gripper_tolerance_counts 4096', False),
  ('gripper_timeout_s 10\ngripper_timeout_s 20', False),
 ]:
  path=Path(d)/'gripper_settings.conf'
  path.write_text(base.replace('[settings]', '[settings]\n'+settings)+'gripper_read left\n')
  r=subprocess.run([str(exe),'--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(settings,r.stdout,r.stderr)
 print('gripper settings PASS')
 for name,definition,steps,ok in [
  ('topic_auto','target left /vision/part_target base_link p','move left target',True),
  ('topic_auto_timeout','target left /vision/part_target base_link p 10 5','move left target',True),
  ('topic_valid','target left part_json /vision/part_target base_link p','move left target\nmovej left target',True),
  ('topic_wrong_arm','target left part_json /vision/part_target base_link p','move right target',False),
  ('topic_template','target left pose_stamped /vision/material_coord base_link missing','move left target',False),
  ('topic_collision','p left point_stamped /point base_link p','move left p',False),
  ('topic_adapter','target left unknown /point base_link p','move left target',False),
  ('topic_joint','target left part_json /vision/part_target base_link p','moveabsj left target',False),
  ('topic_timeout','target left part_json /vision/part_target base_link p 0 5','move left target',False),
  ('topic_concurrent','target left part_json /vision/part_target base_link p','concurrent_begin\nbranch a\nmove left target\nbranch_end\nbranch b\nmove right p\nbranch_end\nconcurrent_end',True),
 ]:
  path=Path(d)/(name+'.conf')
  path.write_text(base.replace('[steps]', '[topic_poses]\n'+definition+'\n[steps]')+steps+'\n')
  r=subprocess.run([str(exe),'--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(name,r.stdout,r.stderr)
  print(name,'PASS')
 for name,offsets,steps,ok in [
  ('offset_static','above p 0 0 0.05','move left above',True),
  ('offset_dynamic','above target 0 0 0.05','move left above\nmove left target',True),
  ('offset_chain','higher above 0.01 0 0.02\nabove target 0 0 0.05','move left higher',True),
  ('offset_wrong_arm','above target 0 0 0.05','move right above',False),
  ('offset_cycle','a b 0 0 0\nb a 0 0 0','move left a',False),
  ('offset_missing','above missing 0 0 0','move left above',False),
  ('offset_collision','p target 0 0 0','move left p',False),
  ('offset_nan','above p nan 0 0','move left above',False),
  ('offset_distance','above p 1 0 0','move left p\nmove left above',False),
 ]:
  path=Path(d)/(name+'.conf')
  path.write_text(base.replace('[steps]', '[topic_poses]\ntarget left /vision/material_coord base_link p\n[offset_poses]\n'+offsets+'\n[steps]')+steps+'\n')
  r=subprocess.run([str(exe),'--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(name,r.stdout,r.stderr)
  if name=='offset_static' and ok: assert '0.050000' in r.stdout,r.stdout
  print(name,'PASS')
 r=subprocess.run([str(exe),'--check'],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 print('current config PASS')
 for steps,ok in [('move left p',False), ('move left target',True)]:
  path=Path(d)/'loop.conf'
  path.write_text(base.replace('[steps]', '[topic_poses]\ntarget left /vision/material_coord base_link p\n[steps]')+steps+'\n')
  r=subprocess.run([str(exe),'--loop','--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(r.stdout,r.stderr)
  if ok: assert 'loop until no new target' in r.stdout
 print('loop requires a used topic target PASS')
 for step,ok in [('spin_if_class target 2 0.25 0.2',True),
                 ('spin_if_class target 2 -0.25',True),
                 ('spin_if_class target 2 0',False),
                 ('spin_if_class target 2 0.25 1',False),
                 ('spin_if_class p 2 0.25',False),
                 ('parallel_begin\nspin_if_class target 2 0.25\nwait 1\nparallel_end',False),
                 ('spin_if_class target 2.5 0.25',False)]:
  path=Path(d)/'chassis.conf'
  path.write_text(base.replace('[steps]', '[topic_poses]\ntarget left /vision/material_coord base_link p\n[topic_classes]\ntarget /vision/part_target\n[steps]')+step+'\n')
  r=subprocess.run([str(exe),'--check',str(path)],capture_output=True,text=True)
  assert (r.returncode==0)==ok,(step,r.stdout,r.stderr)
 print('conditional chassis parser PASS')


# Compile scheduler as a standalone unit, without SDK/hardware.
s=(root/'cxl/text_dct.cpp').read_text();a=s.index('template<class Run>\nvoid runParallel');b=s.index('\n}  // namespace',a)
h='''#include <future>
#include <vector>
#include <atomic>
#include <thread>
#include <stdexcept>
#include <cassert>
#include <iostream>
struct Step { int id; std::string branch; };
std::atomic<bool> cancelled{false};
bool stopping() { return cancelled.load(); }
'''+s[a:b]+'''
int main() {
 std::vector<Step> steps{{0},{1}};
 std::atomic<int> entered{0},done{0};
 auto start=std::chrono::steady_clock::now();
 runParallel(steps,0,2,[&](const Step &) {
   ++entered;
   while(entered.load()!=2) {
     if(std::chrono::steady_clock::now()-start>std::chrono::seconds(2)) throw std::runtime_error("not concurrent");
     std::this_thread::yield();
   }
   ++done;
 });
 assert(done==2);
 entered=0; done=0;
 bool failed=false;
 try { runParallel(steps,0,2,[&](const Step &s) {
   ++entered;
   while(entered.load()!=2) std::this_thread::yield();
   if(s.id==0) throw std::runtime_error("injected failure");
   while(!stopping()) std::this_thread::yield();
   ++done;
 }); } catch(const std::exception &) { failed=true; }
 assert(failed && stopping() && done==1);
 std::cout<<"scheduler overlap, join and peer cancellation PASS\\n";
 cancelled=false;
 std::vector<Step> branches{{0,"a"},{1,"a"},{2,"b"},{3,"b"},{4,"b"}};
 std::atomic<int> a{0},b{0};
 start=std::chrono::steady_clock::now();
 runBranches(branches,0,branches.size(),[&](const Step &s) {
   if(s.id==0) {
     a=1;
     while(b.load()!=3) {
       if(std::chrono::steady_clock::now()-start>std::chrono::seconds(2)) throw std::runtime_error("branches locked in step");
       std::this_thread::yield();
     }
   } else if(s.id==1) { assert(a==1 && b==3); a=2; }
   else { assert(b==s.id-2); ++b; }
 });
 assert(a==2 && b==3);
 cancelled=false;
 bool branchFailed=false;
 try { runBranches(branches,0,branches.size(),[&](const Step &s) {
   if(s.id==0) throw std::runtime_error("branch failure");
   if(s.id==1) throw std::logic_error("must not run next step");
   while(!stopping()) std::this_thread::yield();
 }); } catch(const std::exception &) { branchFailed=true; }
 assert(branchFailed && stopping());
 std::cout<<"independent branch progress, ordering, join and cancellation PASS\\n";
}
'''
with tempfile.TemporaryDirectory() as d:
 source=Path(d)/'scheduler.cpp'; binary=Path(d)/'scheduler'
 source.write_text(h)
 subprocess.run(['c++','-std=c++17','-pthread',str(source),'-o',str(binary)],check=True)
 subprocess.run([str(binary)],check=True,timeout=8)
