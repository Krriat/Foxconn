import json,sqlite3
from pathlib import Path
import cv2,numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
ROOT=Path('/home/jj/code/ACT')
SRC=ROOT/'training_verification/far_recovery_20261113/live_500_wifi_off'
db=sqlite3.connect('file:'+str(next((SRC/'rosbag').glob('*.db3')))+'?mode=ro',uri=True)
chunks=[json.loads(deserialize_message(raw,String).data) for raw, in db.execute(
    'select m.data from messages m join topics t on t.id=m.topic_id where t.name=? order by m.timestamp',('/robot/policy/action_chunk',))]
ct=np.load(SRC/'analysis.npz')['chunk_time']; wanted=[0,5,10,15,25,40,60]
selected=[chunks[np.argmin(abs(ct-t))] for t in wanted]
cameras=('head','left_wrist','right_wrist');found={k:{} for k in cameras}
for k in cameras:
    need={c['source_stamps_ns'][k] for c in selected}
    for raw, in db.execute('select m.data from messages m join topics t on t.id=m.topic_id where t.name=?',
            (f'/robot/sensors/cameras/{k}/color/image_raw/compressed',)):
        msg=deserialize_message(raw,CompressedImage);stamp=int((msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)*1e9)
        if stamp in need:found[k][stamp]=cv2.cvtColor(cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
images=np.array([[found[k][c['source_stamps_ns'][k]] for k in cameras] for c in selected])
np.savez_compressed(SRC/'selected_inputs.npz',images=images,state=np.array([c['observation_state'] for c in selected],np.float32),seconds=np.array(wanted),prediction=np.array([c['actions'] for c in selected],np.float32))
print('Prepared seven exact recorded cases; no physical action')
