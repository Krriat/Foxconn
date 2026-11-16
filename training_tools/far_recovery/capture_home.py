"""Bounded home-only diagnostic on Jetson; no ACT or suction commands.

Waits for GO after writing READY and camera previews. Uses production home
parameters and verifies measured joints. Requires an idle policy throughout.
"""
import argparse
import json
import signal
import time
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rokae_interfaces.srv import MoveAbsJ
import yaml

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--without-cameras',action='store_true',help='Use a separate PC camera observer; avoid an extra remote image subscriber')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
cfg=yaml.safe_load((Path.home()/'.local/share/robot-policy-streams/config.yaml').read_text())
rclpy.init();n=rclpy.create_node('bounded_home_capture');values={};times={};subs=[]
for sig in (signal.SIGINT,signal.SIGTERM):
    signal.signal(sig,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
def callback(key,msg):
    times[key]=time.monotonic()
    if isinstance(msg,CompressedImage):
        values[key]=bytes(msg.data)
        (a.output/(key+'.jpg')).write_bytes(values[key])
    elif isinstance(msg,JointState):
        m=dict(zip(msg.name,msg.position));values[key]=[m[f'{key}_joint_{i}'] for i in range(1,8)]
    else:values[key]=json.loads(msg.data) if isinstance(msg,String) else bool(msg.data)
for key in (() if a.without_cameras else ('head','left_wrist','right_wrist')):
    subs.append(n.create_subscription(CompressedImage,f'/robot/sensors/cameras/{key}/color/image_raw/compressed',lambda m,k=key:callback(k,m),qos_profile_sensor_data))
for side in ('left','right'):
    subs.append(n.create_subscription(JointState,f'/robot/upperlimb/joint_states/{side}_arm',lambda m,k=side:callback(k,m),qos_profile_sensor_data))
    subs.append(n.create_subscription(Bool,f'/robot/suction/{side}/state',lambda m,k=side+'_suction':callback(k,m),qos_profile_sensor_data))
subs.append(n.create_subscription(String,'/robot/policy/status',lambda m:callback('status',m),1))
init=n.create_client(Trigger,'/robot/upperlimb/initialize/dual_arm')
move={s:n.create_client(MoveAbsJ,f'/robot/upperlimb/move_absj/{s}_arm') for s in ('left','right')}
stop={s:n.create_client(Trigger,f'/robot/upperlimb/stop/{s}_arm') for s in ('left','right')}
def check():
    if len(values)!=(5 if a.without_cameras else 8):raise RuntimeError('Missing inputs: '+str(list(values)))
    now=time.monotonic()
    stale={k:round(now-t,3) for k,t in times.items() if now-t>(1.5 if k=='status' else .3)}
    if stale:raise RuntimeError('Stale inputs: '+str(stale))
    if values['status']['phase']!='IDLE':raise RuntimeError('Policy must remain IDLE')
    if values['left_suction'] or values['right_suction']:raise RuntimeError('Suction must stay OFF')
    if (a.output/'STOP').exists():raise RuntimeError('Observer stop')
def wait(futures,timeout):
    end=time.monotonic()+timeout
    while not all(f.done() for f in futures):
        rclpy.spin_once(n,timeout_sec=.01);check()
        if time.monotonic()>end:raise TimeoutError('Home service timeout')
    for f in futures:
        r=f.result()
        if not r or not r.success:raise RuntimeError(str(r))
started=False;result={}
try:
    end=time.monotonic()+60;stable=None
    while time.monotonic()<end:
        rclpy.spin_once(n,timeout_sec=.01)
        try:check()
        except RuntimeError as e:stable=None;last_preflight_error=str(e);continue
        stable=stable or time.monotonic()
        if time.monotonic()-stable>3:break
    else:raise RuntimeError('No stable idle preflight: '+str(locals().get('last_preflight_error')))
    for c in [init,*move.values(),*stop.values()]:
        if not c.wait_for_service(timeout_sec=3):raise RuntimeError('Driver service unavailable')
    (a.output/'READY').write_text('Home only; GO starts the verified dataset home move.\n')
    end=time.monotonic()+180
    while not (a.output/'GO').exists():
        rclpy.spin_once(n,timeout_sec=.01)
        if time.monotonic()>end:raise TimeoutError('GO wait timeout')
    check();started=True;wait([init.call_async(Trigger.Request())],15)
    futures=[]
    for side,c in move.items():
        q=MoveAbsJ.Request();q.joint_positions=cfg['home_'+side]
        q.speed_mm_s=cfg['home_speed_mm_s'];q.zone_mm=0.;q.joint_speed_scale=cfg['home_joint_speed_scale']
        q.joint_tolerance_rad=cfg['home_tolerance_rad'];q.stable_samples=3;q.poll_interval_ms=50;q.timeout_s=60.
        futures.append(c.call_async(q))
    wait(futures,65)
    end=time.monotonic()+2
    while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01);check()
    error=max(abs(x-y) for s in ('left','right') for x,y in zip(values[s],cfg['home_'+s]))
    if error>cfg['home_tolerance_rad']:raise RuntimeError('Measured home verification failed')
    result=dict(success=True,home_max_error_rad=error,state=values['left']+values['right']+[0.,0.],time_ns=time.time_ns())
except BaseException as e:
    result=dict(success=False,error=repr(e))
    if started:
        futures=[c.call_async(Trigger.Request()) for c in stop.values()];end=time.monotonic()+10
        while time.monotonic()<end and not all(f.done() for f in futures):rclpy.spin_once(n,timeout_sec=.01)
        result['stop_results']=[str(f.result()) if f.done() else 'timeout' for f in futures]
finally:
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    n.destroy_node();rclpy.shutdown()
