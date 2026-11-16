#!/usr/bin/python3
"""One authorized manual sequence, with bounded supervision and stop cleanup."""
import json, time, signal, argparse
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage, JointState
from std_srvs.srv import Trigger
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=75)
p.add_argument('--expected-runtime',default='lerobot')
p.add_argument('--start-service',choices=('start_sequence','start_near'),default='start_sequence')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
n=rclpy.create_node('far_recovery_supervisor');values={};received={};inputs={};saved={};subs=[]
f=(a.output/'events.jsonl').open('w',buffering=1)
def log(event,**kw):f.write(json.dumps(dict(event=event,time_ns=time.time_ns(),**kw))+'\n')
def receive(k,m):
    v=json.loads(m.data)
    if k=='action_chunk':
        log(k,**{key:v.get(key) for key in ('run_id','sequence','observation_time_ns','inference_start_ns','inference_end_ns','send_time_ns','inference_ms','source_stamps_ns','max_input_receive_age_ms')})
    else:
        values[k]=v;received[k]=time.monotonic();log(k,value=v)
for k in ('status','inference_status','trajectory_status','action_chunk'):
    subs.append(n.create_subscription(String,'/robot/policy/'+k,lambda m,k=k:receive(k,m),qos_profile_sensor_data))
def inp(k,m):
    now=time.monotonic();stamp=None
    if hasattr(m,'header'):stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec
    data=bool(m.data) if isinstance(m,Bool) else list(m.position) if isinstance(m,JointState) else None
    previous=inputs.get(k)
    inputs[k]=(now,stamp,data)
    log('input',key=k,source_ns=stamp,gap_ms=(now-previous[0])*1000 if previous else None,data=data)
    if isinstance(m,CompressedImage) and now-saved.get(k,0)>.25:
        tmp=a.output/(k+'.tmp');tmp.write_bytes(bytes(m.data));tmp.replace(a.output/(k+'.jpg'));saved[k]=now
for k in ('head','left_wrist','right_wrist'):
    subs.append(n.create_subscription(CompressedImage,f'/robot/sensors/cameras/{k}/color/image_raw/compressed',lambda m,k=k:inp(k,m),qos_profile_sensor_data))
for side in ('left','right'):
    subs.append(n.create_subscription(JointState,f'/robot/upperlimb/joint_states/{side}_arm',lambda m,k=side+'_arm':inp(k,m),qos_profile_sensor_data))
    subs.append(n.create_subscription(Bool,f'/robot/suction/{side}/state',lambda m,k=side+'_suction':inp(k,m),qos_profile_sensor_data))
clients={k:n.create_client(Trigger,'/robot/policy/'+k) for k in ('start_sequence','start_near','finish_far','finish_near','stop')}
def wait(pred,sec):
    end=time.monotonic()+sec
    while not pred() and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
    if not pred():raise RuntimeError('Timeout')
def call(k):
    if not clients[k].wait_for_service(timeout_sec=4):raise RuntimeError(k+' unavailable')
    future=clients[k].call_async(Trigger.Request());wait(future.done,6)
    r=future.result();print(json.dumps(dict(service=k,success=r.success,message=r.message)),flush=True);log('service',name=k,success=r.success,message=r.message)
    if not r.success:raise RuntimeError(r.message)
started=False;reason='not_started';fault=None
try:
    wait(lambda:len(inputs)==7 and all(k in values for k in ('status','inference_status','trajectory_status')) and values['inference_status'].get('ready'),45)
    assert values['status']['phase']=='IDLE',values['status']['phase']
    assert all(not inputs[k][2] for k in ('left_suction','right_suction'))
    assert values['inference_status'].get('runtime')==a.expected_runtime,values['inference_status']
    (a.output/'READY').write_text('Ready; no motion until START file appears.\n')
    print('READY: waiting for operator visual preflight',flush=True)
    wait(lambda:(a.output/'START').exists(),300)
    assert all(time.monotonic()-v[0]<.3 for v in inputs.values()),'stale input'
    started=True;call(a.start_service)
    begin=time.monotonic();running=None;next_print=0;active_stage=None;transition=None;completed=False
    while time.monotonic()-begin<280:
        rclpy.spin_once(n,timeout_sec=.01);now=time.monotonic();s=values['status'];phase=s['phase']
        job=s.get('task_sequence') or {};stage=job.get('stage')
        if phase=='RUNNING' and stage != active_stage:
            active_stage=stage;running=now;transition=None
        if phase=='IDLE' and job.get('result')=='complete':
            completed=True;reason='Requested manual stages completed';break
        if phase=='FAULT' or (phase=='STOPPING' and transition is None):
            fault=s['reason'];reason='Executor stopped: '+fault;break
        if now-received['status']>1:raise RuntimeError('Executor heartbeat lost')
        if any(now-inputs[k][0]>.6 for k in ('head','left_wrist','right_wrist')):raise RuntimeError('Camera stream lost')
        if running and transition is None and now-running>=a.seconds:reason='Stage bounded test deadline';break
        if transition and now-transition[1]>95:raise RuntimeError('Stage transition deadline')
        if (a.output/'STOP').exists():reason='Observer stop';break
        for target, service in [('far','finish_far'),('near','finish_near')]:
            trigger=a.output/service.upper()
            if trigger.exists() and transition is None and phase=='RUNNING' and stage==target:
                transition=(service,now);call(service);trigger.rename(trigger.with_suffix('.accepted'))
        if now>=next_print:
            print(json.dumps(dict(phase=phase,stage=stage,pending=job.get('pending'),running_seconds=round(now-running,1) if running else None,actions=s.get('accepted_actions'),reason=s['reason'],inference_ms=values['inference_status'].get('last_inference_ms'))),flush=True);next_print=now+2
    else:raise RuntimeError('Sequence total deadline')
except BaseException as e:
    reason=repr(e);print(reason,flush=True)
finally:
    cleanup_error=''
    if started:
        try:
            call('stop');wait(lambda:values['status']['phase'] in ('IDLE','FAULT'),15)
            if values['status']['phase']=='FAULT':call('stop')
            wait(lambda:values['status']['phase']=='IDLE' and all(not inputs[k][2] and time.monotonic()-inputs[k][0]<.3 for k in ('left_suction','right_suction')),10)
        except BaseException as e:cleanup_error=repr(e)
    result=dict(reason=reason,fault=fault,robot_motion_requested=started,stages_completed=bool(locals().get('completed',False)),cleanup_error=cleanup_error,final_status=values.get('status'),final_inference=values.get('inference_status'),final_suction=[inputs.get(k,(None,None,None))[2] for k in ('left_suction','right_suction')])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='finished',reason=reason,fault=fault,cleanup_error=cleanup_error,phase=values.get('status',{}).get('phase'),suctions=result['final_suction'])),flush=True)
    f.close();n.destroy_node();rclpy.shutdown()
