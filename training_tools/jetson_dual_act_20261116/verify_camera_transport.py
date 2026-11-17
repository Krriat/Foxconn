"""Read-only camera arrival and publisher-stamp audit. No service clients."""
import argparse
import json
import time
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=90);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=rclpy.create_node('read_only_camera_transport_audit');subs=[]
    latest={};gaps={c:[] for c in ('head','left_wrist','right_wrist')};stamps={c:[] for c in gaps}
    start=time.monotonic();last={};phases=set();health={}
    def receive(camera,msg):
        now=time.monotonic();stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if camera in last and now-start>10:
            gaps[camera].append((now-last[camera][0])*1000)
            stamps[camera].append((stamp-last[camera][1])*1000)
        last[camera]=(now,stamp)
    for c in gaps:
        subs.append(node.create_subscription(CompressedImage,
            f'/robot/sensors/cameras/{c}/color/image_raw/compressed',lambda m,c=c:receive(c,m),qos_profile_sensor_data))
    subs.append(node.create_subscription(String,'/robot/policy/status',
        lambda m:phases.add(json.loads(m.data).get('phase')),qos_profile_sensor_data))
    for camera_node in ('robot_head_camera','robot_wrist_cameras'):
        subs.append(node.create_subscription(String,f'/robot/sensors/cameras/status/{camera_node}',
            lambda m,c=camera_node:health.update({c:json.loads(m.data)}),qos_profile_sensor_data))
    next_report=start
    try:
        while time.monotonic()-start<a.seconds:
            rclpy.spin_once(node,timeout_sec=.02)
            if time.monotonic()>=next_report:
                print(json.dumps({'elapsed':round(time.monotonic()-start,1),
                    'ages_ms':{k:round((time.monotonic()-v[0])*1000,1) for k,v in last.items()},
                    'phases':sorted(phases)}),flush=True);next_report=time.monotonic()+10
    finally:
        def stats(values):
            v=sorted(values)
            return None if not v else {'n':len(v),'median_ms':v[len(v)//2],
                'p99_ms':v[min(len(v)-1,int(.99*len(v)))],'max_ms':v[-1],
                'gaps_over_250ms':sum(x>250 for x in v),'gaps_over_300ms':sum(x>300 for x in v)}
        result={'robot_motion_requested':False,'phases':sorted(phases),
            'receive_gaps':{c:stats(v) for c,v in gaps.items()},
            'source_stamp_gaps':{c:stats(v) for c,v in stamps.items()},'publisher_health':health}
        result['passed']=(phases=={'IDLE'} and all(gaps[c] and max(gaps[c])<250 for c in gaps))
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);node.destroy_node();rclpy.shutdown()
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
