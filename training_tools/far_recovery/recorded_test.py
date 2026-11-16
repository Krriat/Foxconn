"""Run the supervised round and flush its recorder even on test failure."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=float,default=70)
p.add_argument('--expected-runtime',default='lerobot')
p.add_argument('--start-service',choices=('start_sequence','start_near'),default='start_sequence')
a = p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
topics=[f'/robot/sensors/cameras/{c}/color/image_raw/compressed' for c in ('head','left_wrist','right_wrist')]
topics += [f'/robot/upperlimb/joint_states/{s}_arm' for s in ('left','right')]
topics += [f'/robot/suction/{s}/state' for s in ('left','right')]
topics += [f'/robot/policy/{s}' for s in ('status','action_chunk','servo_trace','inference_status','trajectory_status')]
with (a.output/'bag.log').open('w') as baglog,(a.output/'supervisor.log').open('w') as supervisorlog:
    bag=subprocess.Popen(['ros2','bag','record','-o',str(a.output/'rosbag'),*topics],stdout=baglog,stderr=subprocess.STDOUT)
    try:
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('supervise.py')),
            '--output',str(a.output),'--seconds',str(a.seconds),'--expected-runtime',a.expected_runtime,
            '--start-service',a.start_service],stdout=supervisorlog,stderr=subprocess.STDOUT,check=True)
    finally:
        if bag.poll() is None:
            bag.send_signal(signal.SIGINT)
            bag.wait(timeout=30)
