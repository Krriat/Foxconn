"""Summarize a saved physical round without ROS services or hardware access."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from std_msgs.msg import String
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
db=sqlite3.connect('file:'+str(next((a.directory/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
def read(topic):
    return [json.loads(deserialize_message(raw,String).data) for raw, in db.execute(
        'select m.data from messages m join topics t on t.id=m.topic_id where t.name=? order by m.timestamp',(topic,))]
chunks=read('/robot/policy/action_chunk')
traces=[x for x in read('/robot/policy/servo_trace') if x.get('active') and x.get('hardware_commands_sent')]
times=np.array([x['selection_time_ns'] for x in traces],dtype=np.int64);times=(times-times[0])/1e9
measured=np.array([x['left_measured']+x['right_measured'] for x in traces]);cmd=np.array([x['command'] for x in traces])
actions=np.array([x['actions'] for x in chunks]);states=np.array([x['observation_state'] for x in chunks]);ct=np.array([x['observation_time_ns'] for x in chunks],dtype=np.int64);ct=(ct-traces[0]['selection_time_ns'])/1e9
np.savez_compressed(a.directory/'analysis.npz',time=times,measured=measured,command=cmd,chunk_time=ct,actions=actions,observation_state=states)
success=np.load('training_verification/success_2105_comparison_20261113/success_far_analysis.npz')
rows=[]
for t in [0,3,5,10,15,20,25,30,40,60]:
    if t>times[-1]:continue
    i=np.argmin(abs(times-t));k=np.argmin(abs(ct-t));j=np.argmin(abs(success['time']-t))
    distances=np.rad2deg(abs(success['measured'][:,:7]-measured[i,:7])).mean(1);nearest=int(distances.argmin())
    rows.append(dict(seconds=t,current_deg=np.rad2deg(measured[i]).tolist(),reference_deg=np.rad2deg(success['measured'][j]).tolist(),
        nearest_success_left_time=float(success['time'][nearest]),nearest_left_mae_deg=float(distances[nearest]),
        suction_actual=states[k,14:].tolist(),suction_prediction=actions[k,2,14:].tolist(),
        next_2s_joint_motion_deg=np.rad2deg(actions[k,42,:14]-actions[k,2,:14]).tolist()))
result=dict(run_id=chunks[0]['run_id'],chunks=len(chunks),seconds=float(times[-1]),
    tracking_p95_deg=float(np.percentile(np.rad2deg(abs(measured-cmd)),95)),
    inference_median_ms=float(np.median([x['inference_ms'] for x in chunks])),
    max_send_gap_ms=float(np.diff([x['send_time_ns'] for x in chunks]).max()/1e6),
    last_20s_measured_ptp_deg=np.rad2deg(np.ptp(measured[times>times[-1]-20],axis=0)).tolist(),rows=rows)
(a.directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
