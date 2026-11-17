"""Real saved-input replay through native fusion/limiter, without hardware.

Changed output is conditional on the recorded feedback, not a simulated robot
rollout. Baseline must reproduce the actual successful commands numerically.
"""
import argparse
import io
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
here=Path(__file__).parent;binary=a.output/'native_replay'
subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra',str(here/'native/replay.cpp'),'-o',str(binary)],check=True)
events=[json.loads(l) for l in (a.recording/'executor_events.jsonl').open()]
chunks=[r for r in events if r['event']=='chunk'];run=chunks[0]['run_id']
db=sqlite3.connect('file:'+str(next((a.recording/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
traces=[];all_traces=[]
for raw, in db.execute('select m.data from messages m join topics t on t.id=m.topic_id where t.name=? order by m.timestamp',('/robot/policy/servo_trace',)):
    r=json.loads(deserialize_message(raw,String).data)
    if r.get('run_id')==run and r.get('hardware_commands_sent'):
        all_traces.append(r)
        if r.get('active'):traces.append(r)
db.close()
cfg=yaml.safe_load((here/'config.yaml').read_text())
original=np.array([t['command'] for t in traces]);original_fused=np.array([t['fused_target'] for t in traces])
reports=[]
for scale,accel in ((1.,1.),(1.5,2.25)):
    text=io.StringIO();text.write(f"{cfg['temporal_ensemble_coefficient']} {cfg['max_joint_speed_rad_s']} {accel} {cfg['trajectory_response_seconds']} {cfg['max_relative_target_rad']} {cfg['max_tracking_error_rad']}\n")
    text.write('A '+' '.join(map(repr,all_traces[0]['command']))+'\n')
    plans=[c for c in chunks if c['sequence']>=traces[0]['chunk_sequence']];cursor=0
    for tr in traces:
        while cursor<len(plans) and plans[cursor]['sequence']<=tr['chunk_sequence']:
            c=plans[cursor];text.write(f"P {c['sequence']} {c['observation_time_ns']} "+' '.join(map(repr,np.array(c['actions']).ravel().tolist()))+'\n');cursor+=1
        text.write(f"T {tr['requested_execution_time_ns']} {tr['dt_s']!r} "+' '.join(map(repr,tr['left_measured']+tr['right_measured']))+'\n')
    result=subprocess.run([str(binary),str(scale)],input=text.getvalue(),text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(f'Guard/trajectory replay failure scale={scale}: '+result.stderr)
    values=np.loadtxt(io.StringIO(result.stdout));assert len(values)==len(traces)
    fused,command,velocity=values[:,1:17],values[:,17:31],values[:,31:45]
    delta=float(abs(command-original).max());dt=np.array([t['dt_s'] for t in traces])
    acceleration=np.diff(velocity,axis=0)/dt[1:,None]
    assert np.isfinite(values).all()
    assert abs(velocity).max()<=cfg['max_joint_speed_rad_s']+1e-9
    assert abs(acceleration).max()<=accel+1e-8
    if scale==1.:
        assert delta<1e-9,delta
        assert abs(fused-original_fused).max()<1e-9
    reports.append(dict(execution_speed_scale=scale,max_acceleration_rad_s2=accel,servo_samples=len(traces),
        command_difference_from_recorded_max_deg=float(np.rad2deg(delta)),
        max_speed_rad_s=float(abs(velocity).max()),max_acceleration_observed_rad_s2=float(abs(acceleration).max()),
        unchanged_tracking_guard_passed=True,suction_target_samples_different=int(((fused[:,14:]>=.6)!=(original_fused[:,14:]>=.6)).any(1).sum()),
        source_stamps_and_action_step_unchanged=True))
    np.savez_compressed(a.output/f'replay_{scale:g}.npz',command=command,fused=fused,velocity=velocity)
report=dict(recorded_run_id=run,scope='Historical feedback and observations fixed; validates software and limits, NOT physical success or task duration',results=reports)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
