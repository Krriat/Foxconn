#!/usr/bin/env python3
"""Plot recorded no-motion trajectory traces; never connects to ROS or hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
traces=[json.loads(line) for line in (args.directory/'servo_trace.jsonl').read_text().splitlines()]
traces=[t for t in traces if t.get('active')]
t=np.array([r['selection_time_ns'] for r in traces],dtype=np.int64);elapsed=(t-t[0])/1e9
q=np.array([r['command'] for r in traces]);v=np.array([r['velocity'] for r in traces]);dt=np.array([r['dt_s'] for r in traces])
fused=np.array([r['fused_target'][:14] for r in traces]);limited=np.array([r['limited_target'] for r in traces])
acc=np.diff(v,axis=0)/dt[1:,None]
fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
joint=2;deg=180/np.pi;baseline=q[0,joint]
axes[0].plot(elapsed,(fused[:,joint]-baseline)*deg,label='Time-aligned fused target',alpha=.75)
axes[0].plot(elapsed,(limited[:,joint]-baseline)*deg,label='Target after feedback-relative limit',linestyle='--')
axes[0].plot(elapsed,(q[:,joint]-baseline)*deg,label='Continuous ServoJ preview command',linewidth=2)
axes[0].set_ylabel('Left J3 change (degree)');axes[0].legend(loc='best')
axes[1].plot(elapsed,v[:,joint],label='Left J3 velocity');axes[1].axhline(.5,color='gray',linestyle=':');axes[1].axhline(-.5,color='gray',linestyle=':')
axes[1].set_ylabel('Velocity (rad/s)');axes[1].legend(loc='best')
axes[2].plot(elapsed[1:],acc[:,joint],label='Left J3 acceleration');axes[2].axhline(1,color='gray',linestyle=':');axes[2].axhline(-1,color='gray',linestyle=':')
axes[2].set_ylabel('Acceleration (rad/s²)');axes[2].set_xlabel('Elapsed time (s)');axes[2].legend(loc='best')
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('Real-camera ACT preview: aligned chunk fusion and continuous trajectory\nNo physical ServoJ or suction commands were sent')
fig.savefig(args.directory/'trajectory.png',dpi=160)
print(args.directory/'trajectory.png')
