#!/usr/bin/python3
"""Record and supervise ONE authorized real sequence; --execute is required."""
import argparse
import json
from pathlib import Path
import time
import signal

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-gate',type=Path,help='Wait without motion for this file after saving current camera views')
    args=p.parse_args()
    if not args.execute:p.error('Use --execute only for an authorized real robot test')
    args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(signum, frame):raise KeyboardInterrupt(f'Signal {signum}')
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(signum,interrupt)
    node=rclpy.create_node('supervised_jetson_dual_act_once')
    values={};received={};images={};image_stamps={};suctions={};subs=[]
    log=(args.output/'ros_events.jsonl').open('w',buffering=1)
    def update(key,msg):
        value=json.loads(msg.data);values[key]=value;received[key]=time.monotonic()
        log.write(json.dumps(dict(received_ns=time.time_ns(),topic=key,value=value))+'\n')
    for key in ('status','inference_status','brain_status','action_chunk','servo_trace'):
        subs.append(node.create_subscription(String,'/robot/policy/'+key,
            lambda m,key=key:update(key,m),qos_profile_sensor_data))
    for side in ('left','right'):
        subs.append(node.create_subscription(Bool,f'/robot/suction/{side}/state',
            lambda m,side=side:suctions.update({side:(m.data,time.monotonic())}),qos_profile_sensor_data))
    def camera_update(camera,msg):
        images[camera]=(cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR),time.monotonic())
        image_stamps[camera]=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
    for camera in ('head','left_wrist','right_wrist'):
        subs.append(node.create_subscription(CompressedImage,
            f'/robot/sensors/cameras/{camera}/color/image_raw/compressed',
            lambda m,camera=camera:camera_update(camera,m),qos_profile_sensor_data))
    clients={key:node.create_client(Trigger,'/robot/policy/'+key) for key in ('start_sequence_brain','stop')}
    def wait(predicate,timeout=10,label=None):
        end=time.monotonic()+timeout
        next_log=0.
        while not predicate() and time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.02)
            if label and time.monotonic()>=next_log:
                print(json.dumps(dict(event=label,phase=values.get('status',{}).get('phase'),
                    trajectory_ready=values.get('status',{}).get('trajectory_ready'),
                    act_ready=values.get('inference_status',{}).get('ready'),
                    act_error=values.get('inference_status',{}).get('error'),
                    brain_ready=values.get('brain_status',{}).get('ready'),
                    brain_reason=values.get('brain_status',{}).get('reason'),
                    brain_error=values.get('brain_status',{}).get('error'))),flush=True)
                next_log=time.monotonic()+5
        if not predicate():raise RuntimeError('Condition timeout: '+json.dumps(values.get('status',{})))
    def call(name):
        if not clients[name].wait_for_service(timeout_sec=5):raise RuntimeError(name+' unavailable')
        future=clients[name].call_async(Trigger.Request());wait(future.done)
        if not future.result().success:raise RuntimeError(future.result().message)
        return future.result().message
    started=False;outcome='not_started';error='';video=None;video_log=None;frame_number=0
    try:
        wait(lambda:all(k in values for k in ('status','inference_status','brain_status')) and
             values['inference_status'].get('ready') and values['brain_status'].get('ready') and
             values['status'].get('trajectory_ready') and len(images)==3 and len(suctions)==2,
             60,label='waiting_for_ready')
        status,inference,brain=(values[k] for k in ('status','inference_status','brain_status'))
        assert status['phase']=='IDLE' and all(not state[0] for state in suctions.values())
        assert status.get('execution_speed_scale')==inference.get('execution_speed_scale')==1.0
        assert status.get('execution_speed_profile')==inference.get('execution_speed_profile')=='uniform'
        assert status['max_joint_acceleration_rad_s2']==1.0
        assert (inference.get('ready') and inference.get('runtime')=='jetson_act' and
                inference.get('host')=='tegra-ubuntu' and inference.get('backend')=='tensorrt'),inference
        assert brain.get('ready') and brain.get('transitions_enabled') and not brain.get('error'),brain
        wait(lambda:all(time.monotonic()-entry[1]<.3 for entry in images.values()),
             20,label='waiting_for_fresh_camera_views')
        if args.start_gate:
            if args.start_gate.exists():raise RuntimeError('Start gate must not already exist')
            print(json.dumps(dict(event='waiting_for_visual_preflight',gate=str(args.start_gate))),flush=True)
            deadline=time.monotonic()+600;last_image=0.
            while not args.start_gate.exists():
                if time.monotonic()>deadline:raise RuntimeError('Visual preflight timed out; no motion started')
                rclpy.spin_once(node,timeout_sec=.02)
                if time.monotonic()-last_image>=.25:
                    montage=np.hstack([images[c][0] for c in ('head','left_wrist','right_wrist')])
                    cv2.imwrite(str(args.output/'preflight.jpg'),montage);last_image=time.monotonic()
        # New DDS subscribers have briefly paused wrist delivery on this setup.
        # Settle discovery while IDLE; retain all runtime watchdog thresholds.
        stable=None;deadline=time.monotonic()+90;next_report=0.
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02);now=time.monotonic()
            ages=values['inference_status'].get('input_receive_age_ms',{})
            fresh=(values['status']['phase']=='IDLE' and values['inference_status'].get('ready')
                   and values['brain_status'].get('ready') and len(ages)==7
                   and all(age is not None and age<200. for age in ages.values())
                   and now-received['status']<.4 and now-received['brain_status']<.4
                   and all(now-entry[1]<.3 for entry in images.values())
                   and all(not entry[0] and now-entry[1]<.3 for entry in suctions.values()))
            stable=(stable or now) if fresh else None
            if now>=next_report:
                print(json.dumps(dict(event='waiting_for_stable_inputs',
                    stable_seconds=round(now-stable,1) if stable else 0.,input_ages_ms=ages)),flush=True)
                next_report=now+3
            if stable is not None and now-stable>=12.:break
        else:raise RuntimeError('Inputs did not remain fresh for 12 seconds; no motion started')
        montage=np.hstack([images[c][0] for c in ('head','left_wrist','right_wrist')])
        cv2.imwrite(str(args.output/'before.jpg'),montage)
        video=cv2.VideoWriter(str(args.output/'cameras.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10,(1920,480))
        if not video.isOpened():raise RuntimeError('Camera recorder could not be opened; no motion started')
        video_log=(args.output/'video_frames.jsonl').open('w',buffering=1)
        started=True  # A lost service reply must still enter the stop/cleanup path.
        result=call('start_sequence_brain')
        print(json.dumps({'event':'started','message':result}),flush=True)
        start=time.monotonic();last_snapshot=0.;last_phase=None;last_log=0.
        while time.monotonic()-start<240:
            rclpy.spin_once(node,timeout_sec=.01)
            now=time.monotonic();status=values['status'];job=status.get('task_sequence') or {}
            phase=(status['phase'],job.get('stage'),job.get('result'))
            if now-last_snapshot>=.1 and len(images)==3:
                montage=np.hstack([images[c][0] for c in ('head','left_wrist','right_wrist')])
                video.write(montage);last_snapshot=now
                video_log.write(json.dumps(dict(frame=frame_number,received_ns=time.time_ns(),
                    elapsed_seconds=now-start,phase=phase,image_stamps_ns=dict(image_stamps),
                    receive_ages_ms={c:(now-images[c][1])*1000 for c in images}))+'\n')
                frame_number+=1
                cv2.imwrite(str(args.output/'latest.jpg'),montage)
            if phase!=last_phase or now-last_log>=3:
                if phase!=last_phase:
                    cv2.imwrite(str(args.output/f'{int(now-start):03d}_{status["phase"]}_{job.get("stage")}.jpg'),
                        np.hstack([images[c][0] for c in ('head','left_wrist','right_wrist')]))
                print(json.dumps(dict(elapsed=round(now-start,2),phase=phase,reason=status['reason'],
                    inference_ms=values['inference_status'].get('last_inference_ms'),
                    brain=values['brain_status'].get('result',{}).get('value'),
                    brain_error=values['brain_status'].get('error'),
                    progress=status.get('manipulation_progress'),
                    far_coordination=status.get('far_arm_coordination'))),flush=True)
                last_phase=phase;last_log=now
            if status['phase']=='FAULT':raise RuntimeError(status['reason'])
            if job.get('result')=='complete' and status['phase']=='IDLE':outcome='complete';break
            if job.get('result') in ('fault','cancelled'):raise RuntimeError(job['result']+': '+status['reason'])
            if now-received['status']>2:raise RuntimeError('Executor status stale')
            if status['phase']=='RUNNING' and now-received['brain_status']>5:raise RuntimeError('Brain heartbeat stale')
        else:raise RuntimeError('Supervised sequence time limit')
    except BaseException as e:
        error=repr(e);outcome='failed';print(json.dumps({'event':'error','error':error}),flush=True)
    finally:
        if started and outcome!='complete':
            try:
                call('stop')
                wait(lambda:values['status']['phase'] in ('IDLE','FAULT'),15)
                if values['status']['phase']=='FAULT':call('stop')
                wait(lambda:values['status']['phase']=='IDLE' and all(not s[0] for s in suctions.values()),10)
            except Exception as e:error+='; stop: '+repr(e)
        if video:video.release()
        if video_log:video_log.close()
        result=dict(outcome=outcome,error=error,robot_motion_requested=started,
            final_status=values.get('status'),final_brain=values.get('brain_status'),
            final_inference=values.get('inference_status'),suctions={k:v[0] for k,v in suctions.items()})
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        log.close();node.destroy_node();rclpy.shutdown()
    if outcome!='complete':raise SystemExit(1)


if __name__=='__main__':main()
