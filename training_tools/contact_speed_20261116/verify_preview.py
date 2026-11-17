"""Inspect or preview the deployed pipeline. No motion-start services exist here.

Preview computes live model/trajectory targets but must send zero hardware
commands and leave both suction channels OFF. The existing stop endpoint ends
the preview; idle checks alone issue no service calls.
"""
import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--preview-seconds',type=float,default=0.)
p.add_argument('--expected-scale',type=float,default=1.5)
p.add_argument('--expected-acceleration',type=float,default=2.25)
a=p.parse_args()
if not 0.<=a.preview_seconds<=30.:raise ValueError('Preview interval must be between 0 and 30 seconds')
a.output.mkdir(parents=True,exist_ok=True)
rclpy.init();n=rclpy.create_node('verify_speed_preview_only');latest={};times={};subs=[];chunks=[];traces=[]
log=(a.output/'events.jsonl').open('w',buffering=1)
def receive(key,m):
    value=json.loads(m.data) if isinstance(m,String) else bool(m.data)
    latest[key]=value;times[key]=time.monotonic()
    if key=='action_chunk':chunks.append(value)
    if key=='servo_trace':traces.append(value)
    log.write(json.dumps(dict(event=key,time_ns=time.time_ns(),value=value))+'\n')
for key in ('status','inference_status','trajectory_status','action_chunk','servo_trace'):
    subs.append(n.create_subscription(String,'/robot/policy/'+key,lambda m,k=key:receive(k,m),qos_profile_sensor_data))
for side in ('left','right'):
    subs.append(n.create_subscription(Bool,f'/robot/suction/{side}/state',lambda m,k=side+'_suction':receive(k,m),qos_profile_sensor_data))
clients={key:n.create_client(Trigger,'/robot/policy/'+key) for key in ('preview','stop')}
def wait(predicate,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        rclpy.spin_once(n,timeout_sec=.01)
        if predicate():return
    raise TimeoutError('Timed out waiting for pipeline state')
def call(key):
    if not clients[key].wait_for_service(timeout_sec=5):raise RuntimeError(key+' unavailable')
    f=clients[key].call_async(Trigger.Request());wait(f.done,8);r=f.result()
    if not r or not r.success:raise RuntimeError(str(r))
    return r.message
def idle():
    return (latest.get('status',{}).get('phase')=='IDLE' and
            all(k in latest and latest[k] is False and time.monotonic()-times[k]<.3 for k in ('left_suction','right_suction')))
started=False;report={}
try:
    wait(lambda:idle() and latest.get('inference_status',{}).get('ready') and latest.get('trajectory_status',{}).get('ready'),60)
    # DDS discovery has produced brief wrist-camera pauses on this installation.
    # Require sustained fresh input BEFORE asking for preview; keep runtime
    # watchdogs unchanged rather than accepting stale inputs on startup.
    stable=None;deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        rclpy.spin_once(n,timeout_sec=.01)
        ages=latest.get('inference_status',{}).get('input_receive_age_ms',{})
        fresh=(idle() and len(ages)==7 and all(v is not None and v<200. for v in ages.values())
               and time.monotonic()-times.get('inference_status',0)<1.5)
        stable=(stable or time.monotonic()) if fresh else None
        if stable is not None and time.monotonic()-stable>=12.:break
    else:raise RuntimeError('No sustained fresh-input interval before preview')
    s=latest['status'];inf=latest['inference_status'];driver=latest['trajectory_status']
    assert s.get('execution_speed_scale')==inf.get('execution_speed_scale')==a.expected_scale,(s,inf)
    assert s['max_joint_acceleration_rad_s2']==a.expected_acceleration,s
    assert driver['execution_time_scaling']=='per_observation_v1' and driver.get('continuous_action_clock_v1'),driver
    assert s.get('execution_speed_profile')==inf.get('execution_speed_profile')=='contact_slow_v1'
    assert driver['maximum_joint_acceleration_rad_s2']>=a.expected_acceleration,driver
    report.update(before_status=s,before_inference=inf,driver=driver)
    if a.preview_seconds:
        # Mark before calling: cleanup must still stop an accepted request whose
        # response was lost. This endpoint cannot initiate live motion.
        started=True;report['preview_response']=call('preview')
        wait(lambda:latest.get('status',{}).get('phase')=='PREVIEW',5)
        run=latest['status']['run_id'];end=time.monotonic()+a.preview_seconds
        while time.monotonic()<end:
            rclpy.spin_once(n,timeout_sec=.01)
            if latest['status']['phase']!='PREVIEW':raise RuntimeError(str(latest['status']))
            if any(latest.get(k) is not False for k in ('left_suction','right_suction')):raise RuntimeError('Unexpected suction state')
        selected=[c for c in chunks if c.get('run_id')==run]
        selected_traces=[t for t in traces if t.get('run_id')==run and t.get('active')]
        assert len(selected)>=20 and len(selected_traces)>=20,(len(selected),len(selected_traces))
        assert all(c['execution_speed_scale']==a.expected_scale and c['step_ns']==50_000_000 and c['execution_speed_profile']=='contact_slow_v1' for c in selected)
        assert all(t['mode']=='preview' and not t.get('hardware_commands_sent') for t in selected_traces)
        errors=[abs(t['newest_index']-a.expected_scale*(t['requested_execution_time_ns']-t['observation_time_ns'])/50_000_000) for t in selected_traces]
        assert max(errors)<1e-9,max(errors)
        report.update(run_id=run,preview_chunks=len(selected),preview_active_traces=len(selected_traces),
                      max_sampling_index_error=max(errors),hardware_commands_sent=0,
                      source_action_step_ns=50_000_000,effective_action_step_ms=50./a.expected_scale)
    report['success']=True
except BaseException as e:
    report.update(success=False,error=repr(e))
finally:
    if started:
        try:report['stop_response']=call('stop');wait(idle,15)
        except BaseException as e:report.update(success=False,cleanup_error=repr(e))
    report.update(final_status=latest.get('status'),final_suction=[latest.get(k) for k in ('left_suction','right_suction')],robot_live_motion_requested=False)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before_status','before_inference','driver','final_status')},indent=2),flush=True)
    log.close();n.destroy_node();rclpy.shutdown()
if not report.get('success'):raise SystemExit(1)
