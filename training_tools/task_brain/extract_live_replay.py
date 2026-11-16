"""Extract archived successful live observations read-only; no ROS publishers."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--stage",choices=("far","near"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--interval",type=float,help="Dense read-only sampling interval in seconds")
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    events=[json.loads(l) for l in (args.source/"run_events.jsonl").read_text().splitlines()]
    traces=[e["trace"] for e in events if e["event"]=="servo_trace" and e["trace"].get("active")]
    start,end=traces[0]["selection_time_ns"],traces[-1]["selection_time_ns"]
    conn=sqlite3.connect(f'file:{next((args.source/"rosbag").glob("*.db3"))}?mode=ro',uri=True)
    records={};stamps={}
    for camera in ("head","left_wrist","right_wrist"):
        rows=conn.execute('SELECT m.timestamp,m.data FROM messages m JOIN topics t ON t.id=m.topic_id '
            'WHERE t.name=? AND m.timestamp BETWEEN ? AND ? ORDER BY m.timestamp',
            (f'/robot/sensors/cameras/{camera}/color/image_raw/compressed',start-1_000_000_000,end+1_000_000_000)).fetchall()
        records[camera]=rows;stamps[camera]=[r[0] for r in rows]
    manifest=[];last_head_stamp=None
    points=(np.append(np.arange(start,end,int(args.interval*1e9),dtype=np.int64),end)
            if args.interval else np.linspace(start,end,16,dtype=np.int64))
    for i,stamp in enumerate(points):
        paths=[];head_stamp=None
        for camera in records:
            index=bisect_right(stamps[camera],int(stamp))-1
            if index<0 or int(stamp)-stamps[camera][index]>300_000_000:raise ValueError("Stale archive camera")
            msg=deserialize_message(records[camera][index][1],CompressedImage)
            if camera=="head":head_stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            image=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
            path=args.output/f'{i:02d}_{camera}.jpg';cv2.imwrite(str(path),image);paths.append(str(path.resolve()))
        if head_stamp==last_head_stamp:continue
        last_head_stamp=head_stamp
        manifest.append({"index":i,"elapsed_seconds":(int(stamp)-start)/1e9,"stage":args.stage,
                         "head_image_time_ns":head_stamp,"images":paths})
    conn.close()
    (args.output/"observations.json").write_text(json.dumps(manifest,indent=2)+"\n")
    heads=[]
    for r in manifest:
        image=cv2.resize(cv2.imread(r["images"][0]),(256,192))
        cv2.putText(image,f'{r["index"]} {r["elapsed_seconds"]:.1f}s',(5,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);heads.append(image)
    while len(heads)%4:heads.append(np.zeros_like(heads[0]))
    cv2.imwrite(str(args.output/"overview.jpg"),np.concatenate([np.concatenate(heads[i:i+4],axis=1) for i in range(0,len(heads),4)],axis=0))
    print(json.dumps({"read_only":True,"samples":len(manifest),"duration_seconds":(end-start)/1e9}),flush=True)


if __name__=="__main__":main()
