"""Conservative endpoint-window supervision, split by whole demonstrations.

All 58 first/last head frames were visually checked. Only early and terminal
windows are supervised; intermediate opening frames are NOT labeled completed
from a percentage-of-time heuristic. Images from other cameras are not labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/"AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz"
VALIDATION = {18, 21, 38, 41, 48, 49}
TEST = {20, 35, 39, 47, 54, 55, 56, 57}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, episodes, arrays = [], [], []
    for ep in sorted(RAW.glob("episode_*")):
        number = int(ep.name.split("_")[-1])
        state = np.load(ep/"trajectory.npz")["state"]
        n = len(state)
        split = "test" if number in TEST else "validation" if number in VALIDATION else "train"
        stage = "far" if number % 2 == 0 else "near"
        # Narrow windows are visible before any lid motion and after the
        # demonstrated opening/retreat. No endpoint-joint memorization.
        windows = [(0, min(39, n-1), number % 2, "initial_visually_reviewed"),
                   (max(0, n-40), n-1, number % 2+1, "terminal_visually_reviewed")]
        episodes.append({"episode": number, "stage": stage, "split": split, "frames": n,
                         "windows": windows})
        for begin, end, label, basis in windows:
            for frame in sorted(set(np.linspace(begin, end, 16, dtype=int).tolist())):
                path = ep/f"images/head/{frame:06d}.png"
                bgr = cv2.imread(str(path))
                if bgr is None or bgr.shape != (480, 640, 3):
                    raise ValueError(str(path))
                rgb = cv2.cvtColor(cv2.resize(bgr[80:480, 80:600], (192, 192)), cv2.COLOR_BGR2RGB)
                rows.append({"index": len(rows), "episode": number, "stage": stage, "split": split,
                             "frame": frame, "label": label, "label_basis": basis,
                             "image": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                arrays.append(rgb)
    splits = {split: sorted({r["episode"] for r in rows if r["split"] == split})
              for split in ("train", "validation", "test")}
    assert not (set(splits["train"]) & set(splits["validation"]) or
                set(splits["train"]) & set(splits["test"]) or
                set(splits["validation"]) & set(splits["test"]))
    (args.output/"samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    np.savez_compressed(args.output/"images.npz", images=np.stack(arrays))
    (args.output/"dataset.json").write_text(json.dumps({"episodes": episodes, "split_episodes": splits,
        "samples": len(rows), "labels": ["closed", "far_open", "both_open", "unknown"],
        "label_limits": "Only visually reviewed endpoint windows are ground truth; middle frames remain unlabeled. Unknown training images are explicitly synthetic corruptions, not real failure demonstrations."}, indent=2)+"\n")
    print(json.dumps({"samples":len(rows),"split_episode_counts":{k:len(v) for k,v in splits.items()}}))


if __name__ == "__main__":
    main()
