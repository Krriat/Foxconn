#!/usr/bin/env python3
"""Replay all episodes against the completion gate; never controls hardware."""
import json
from pathlib import Path
import cv2
import numpy as np
from task_sequence import SceneDetector, CompletionGate
from build_detector import RAW, OUT, HELD_OUT


def main():
    detector = SceneDetector(Path(__file__).with_name("task_detector.npz"))
    results = []
    tiles = []
    for p in sorted(RAW.glob("episode_*")):
        ep = int(p.name.split("_")[-1]); state = np.load(p/"trajectory.npz")["state"]
        stage = "far" if ep%2 == 0 else "near"
        gate = CompletionGate(stage, 0.)
        first = None
        for i in range(0, len(state), 2):
            bgr = cv2.imread(str(p/f"images/head/{i:06d}.png"))
            result = detector.classify(bgr)
            if gate.update(i/20, i+1, result["scene"], state[i,:14], state[i,14:]>.5,
                           detector.endpoint_distance(stage, state[i,:14]), not any(state[i,14:]>.5)):
                first = i
                tiles.append((ep, i, bgr))
                break
        # Episodes often end during arm retreat. Holding their final observation
        # tests the completion criterion separately from demonstration length.
        for j in range(1, 32):
            if first is not None: break
            bgr = cv2.imread(str(p/f"images/head/{len(state)-1:06d}.png"))
            result = detector.classify(bgr); i = len(state)-1+j*2
            if gate.update(i/20, i+1, result["scene"], state[-1,:14], state[-1,14:]>.5,
                           detector.endpoint_distance(stage, state[-1,:14]), not any(state[-1,14:]>.5)):
                first = i; tiles.append((ep, i, bgr))
        released = (np.flatnonzero(np.diff((state[:,14]>.5).astype(int)) < 0)+1).tolist()
        results.append({"episode":ep, "held_out":ep in HELD_OUT, "stage":stage,
                        "first_completion_frame":first, "recorded_frames":len(state),
                        "requires_endpoint_hold":first is not None and first >= len(state),
                        "release_frames": released})
    for k in range(0,len(tiles),30):
        part=tiles[k:k+30]; canvas=np.full((((len(part)+3)//4)*200, 1000,3),255,np.uint8)
        for n,(ep,frame,bgr) in enumerate(part):
            x=(n%4)*250;y=(n//4)*200
            canvas[y:y+180,x:x+240]=cv2.resize(bgr,(240,180))
            cv2.putText(canvas,f'ep{ep:02} f{frame}',(x,y+195),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        cv2.imwrite(str(OUT/f'completion_frames_{k//30}.jpg'),canvas)
    report={"episodes":results,"detected":sum(x['first_completion_frame'] is not None for x in results),
            "total":len(results), "note":"Replay uses true observations; artificial endpoint holds are identified. Not closed-loop success."}
    (OUT/'completion_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
