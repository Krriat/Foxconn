#!/usr/bin/env python3
"""Build reproducible fixed-camera prototypes; split by whole raw episode."""
import json
from pathlib import Path
import cv2
import numpy as np
from task_sequence import feature, SceneDetector

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz"
OUT = ROOT / "training_verification/two_stage_20261112"
HELD_OUT = {18, 20, 21, 35, 39, 47}  # same raw episodes as policy validation


def main():
    features, labels, end_q, end_stage, manifest = [], [], [], [], []
    for p in sorted(RAW.glob("episode_*")):
        ep = int(p.name.split("_")[-1]); state = np.load(p/"trajectory.npz")["state"]
        n = len(state); stage = ep % 2
        manifest.append({"episode": ep, "stage": "far" if stage == 0 else "near", "frames": n,
                         "split": "validation" if ep in HELD_OUT else "train",
                         "label_basis": "visually reviewed first/last head frames; alternating tasks"})
        if ep in HELD_OUT:
            continue
        for frame, label in ((0, stage), (n-1, stage+1)):
            im = cv2.imread(str(p/f"images/head/{frame:06d}.png"))
            features.append(feature(im)); labels.append(label)
        end_q.append(state[-1, :14]); end_stage.append(stage)
    path = Path(__file__).with_name("task_detector.npz")
    np.savez_compressed(path, features=np.stack(features), labels=labels, end_q=end_q,
                        end_stage=end_stage, max_distance=.05, min_margin=.015)
    detector = SceneDetector(path)
    cases = []
    for ep in sorted(HELD_OUT):
        p = RAW/f"episode_{ep:06d}"; state = np.load(p/"trajectory.npz")["state"]
        for frame, expected in ((0, ("closed", "far_open")[ep%2]),
                                (len(state)-1, ("far_open", "both_open")[ep%2])):
            result = detector.classify(cv2.imread(str(p/f"images/head/{frame:06d}.png")))
            cases.append({"episode": ep, "frame": frame, "expected": expected, **result})
    current = detector.classify(cv2.imread(str(ROOT/"training_verification/task_mix_audit_20261112/current_head.jpg")))
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT/"dataset_labels.json").write_text(json.dumps(manifest, indent=2)+"\n")
    report = {"validation_endpoint_cases": cases, "correct": sum(x["scene"]==x["expected"] for x in cases),
              "total": len(cases), "current_scene": current,
              "limits": "Endpoint labels only; this is not a full-trajectory or real-world success guarantee."}
    (OUT/"detector_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
