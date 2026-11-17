"""Read-only readiness check. Creates subscriptions only, no service clients."""
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String,Bool
p=argparse.ArgumentParser();p.add_argument('--timeout',type=float,default=60);p.add_argument('--output',type=Path);a=p.parse_args()
rclpy.init();n=rclpy.create_node('jetson_act_brain_readiness');values={};times={};subs=[]
def receive(key,msg):
 values[key]=json.loads(msg.data) if isinstance(msg,String) else bool(msg.data);times[key]=time.monotonic()
for key in ('status','inference_status','brain_status'):
 subs.append(n.create_subscription(String,'/robot/policy/'+key,lambda m,key=key:receive(key,m),qos_profile_sensor_data))
for side in ('left','right'):
 subs.append(n.create_subscription(Bool,f'/robot/suction/{side}/state',lambda m,key=side+'_suction':receive(key,m),qos_profile_sensor_data))
end=time.monotonic()+a.timeout;stable=None;ready=False;reason='waiting for five status topics'
try:
 while time.monotonic()<end:
  rclpy.spin_once(n,timeout_sec=.02);now=time.monotonic();s=values.get('status',{});i=values.get('inference_status',{});b=values.get('brain_status',{})
  ages=i.get('input_receive_age_ms',{})
  checks=dict(idle=s.get('phase')=='IDLE',original_speed=s.get('execution_speed_scale')==i.get('execution_speed_scale')==1.0 and s.get('execution_speed_profile')==i.get('execution_speed_profile')=='uniform' and s.get('max_joint_acceleration_rad_s2')==1.0,jetson_ACT=i.get('ready') and not i.get('error') and i.get('host')=='tegra-ubuntu' and i.get('runtime')=='jetson_act' and i.get('backend')=='tensorrt',Qwen_LoRA=b.get('ready') and b.get('transitions_enabled') and b.get('adapter_sha256')==s.get('brain_adapter_sha256'),fresh_status=all(now-times.get(k,0)<limit for k,limit in [('status',.4),('brain_status',.4),('inference_status',1.5)]),fresh_inputs=len(ages)==7 and all(v is not None and v<200 for v in ages.values()),suction_off=all(values.get(k) is False and now-times.get(k,0)<.3 for k in ('left_suction','right_suction')))
  valid=all(checks.values());stable=(stable or now) if valid else None;reason='ready' if valid else ', '.join(k for k,v in checks.items() if not v)
  if stable is not None and now-stable>=12.:ready=True;break
 report=dict(ready=ready,reason=reason,robot_motion_requested=False,execution_speed_scale=values.get('status',{}).get('execution_speed_scale'),ACT=values.get('inference_status'),brain=values.get('brain_status'),phase=values.get('status',{}).get('phase'),suction=[values.get(k) for k in ('left_suction','right_suction')])
 if a.output:a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2))
finally:n.destroy_node();rclpy.shutdown()
raise SystemExit(0 if ready else 1)
