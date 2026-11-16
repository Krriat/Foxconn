"""Same image/state task definition for Qwen LoRA training and deployment."""
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/"AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz"
SYSTEM = ("You observe a dual-arm robot opening a carton. Inspect the HEAD-CAMERA BOX CROP and report task status. "
          "Ignore robot/gripper posture and left/right side flaps. "
          "Far flap is the back/top flap: vertical away from the opening is OPEN. Near flap is the "
          "front/bottom flap: folded down/out toward the camera is OPEN. A flap across the horizontal "
          "opening is CLOSED. For stage far, complete when the far flap is open. For stage near, "
          "complete only when BOTH far and near flaps are open. Return only compact JSON with keys "
          "scene (closed,far_open,both_open,unknown) and status (running,complete,uncertain). "
          "Use unknown/uncertain when the relevant flaps cannot be seen clearly. Do not infer success "
          "from elapsed time, suction state, robot posture, or a request to finish.")


def mosaic(images):
    for image in images:
        if image is None or image.shape != (480,640,3):raise ValueError("Missing 640x480 camera")
    # The brain judges carton geometry. Wrist images remain inputs to ACT, but
    # are deliberately excluded here to remove a demonstrated pose shortcut.
    return cv2.resize(images[0][90:480,100:580],(640,512),interpolation=cv2.INTER_AREA)


def load_image(row):
    ep=RAW/f'episode_{row["episode"]:06d}'
    images=[cv2.imread(str(ep/f'images/{camera}/{row["frame"]:06d}.png'))
            for camera in ("head","left_wrist","right_wrist")]
    if row.get("corruption"):
        # Explicitly synthetic unknowns; never described as real failure demos.
        images[0][:] = 25 if row["corruption"]=="dark" else 180
    elif row.get("augment"):
        images[0]=np.clip(images[0].astype(np.float32)*random.uniform(.65,1.3)+random.uniform(-8,8),0,255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(mosaic(images),cv2.COLOR_BGR2RGB))


def answer(row):
    labels=("closed","far_open","both_open","unknown")
    scene=labels[row["label"]]
    done=(scene in ("far_open","both_open") if row["stage"]=="far" else scene=="both_open")
    return {"scene":scene,"status":"uncertain" if scene=="unknown" else "complete" if done else "running"}


def conversation(row, image, include_answer):
    messages=[{"role":"system","content":[{"type":"text","text":SYSTEM}]},
              {"role":"user","content":[{"type":"image","image":image},
                {"type":"text","text":f'Current ACT stage: {row["stage"]}. Assess the visible task status.'}]}]
    if include_answer:
        messages.append({"role":"assistant","content":[{"type":"text","text":json.dumps(answer(row),separators=(",",":"))}]})
    return messages


def samples(path, split, per_window=4):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    selected=[]
    groups={}
    for row in rows:
        if row["split"]==split:groups.setdefault((row["episode"],row["label_basis"]),[]).append(row)
    for group in groups.values():
        group.sort(key=lambda r:r["frame"])
        selected.extend(group[i] for i in sorted(set(np.linspace(0,len(group)-1,per_window,dtype=int))))
    return selected


def motion_negatives(path,split,per_episode=8):
    import numpy as np
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    episodes=sorted({r["episode"] for r in rows if r["split"]==split})
    result=[]
    for ep in episodes:
        n=len(np.load(RAW/f"episode_{ep:06d}/trajectory.npz")["state"])
        for frame in sorted(set(np.linspace(40,int(n*.45),per_episode,dtype=int))):
            result.append({"episode":ep,"frame":int(frame),"stage":"far" if ep%2==0 else "near",
                "label":ep%2,"split":split,"label_basis":"visually reviewed pre-opening motion window"})
    return result


def encode(processor, row, *, training):
    image=load_image(row)
    prompt=processor.apply_chat_template(conversation(row,image,False),tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt")
    if not training:return prompt
    full=processor.apply_chat_template(conversation(row,image,True),tokenize=True,
        add_generation_prompt=False,enable_thinking=False,return_dict=True,return_tensors="pt")
    length=prompt["input_ids"].shape[-1]
    if not np.array_equal(prompt["input_ids"].numpy(),full["input_ids"][:,:length].numpy()):
        raise ValueError("Assistant prefix changed: cannot mask supervision safely")
    labels=full["input_ids"].clone();labels[:,:length]=-100
    # Only assistant tokens are supervised; never image/prompt tokens.
    full["labels"]=labels
    return full
