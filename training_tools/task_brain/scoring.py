"""Constrained task readout from Qwen token likelihoods, not another network."""
import json
import time
import numpy as np

import torch
from torch.nn import functional as F

from training_tools.task_brain.lora_data import conversation,answer
from training_tools.task_brain.lora_model import to_device


def visibility(image):
    gray=np.asarray(image.convert("L"),dtype=np.float32)
    p05,p95=np.percentile(gray,[5,95]);mean=float(gray.mean())
    good=mean>=12. and p95>=24. and p95-p05>=8. and p05<245.
    return {"ok":bool(good),"mean_luma":mean,"p05_luma":float(p05),"p95_luma":float(p95),
            "reason":"visible" if good else "image too dark, saturated, or without usable contrast"}


@torch.inference_mode()
def classify(model,processor,image,stage,minimum_margin=.05):
    started=time.monotonic();rows=[];lengths=[];answers=[]
    quality=visibility(image)
    if not quality["ok"]:
        return {"value":{"scene":"unknown","status":"uncertain"},"visibility_ok":False,
                "quality":quality,"latency_ms":(time.monotonic()-started)*1000,
                "method":"input visibility check; no task completion inferred"}
    prompt=processor.apply_chat_template(conversation({"stage":stage},image,False),tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt")
    prefix=prompt["input_ids"].shape[-1]
    for label in range(4):
        row={"stage":stage,"label":label};answers.append(answer(row))
        encoded=processor.apply_chat_template(conversation(row,image,True),tokenize=True,
            add_generation_prompt=False,enable_thinking=False,return_dict=True,return_tensors="pt")
        if not torch.equal(prompt["input_ids"],encoded["input_ids"][:,:prefix]):
            raise ValueError("Inference prefix mismatch")
        labels=encoded["input_ids"].clone();labels[:,:prefix]=-100
        encoded["labels"]=labels;rows.append(encoded);lengths.append(labels.shape[-1])
    maximum=max(lengths);batch={}
    for key in ("input_ids","attention_mask","mm_token_type_ids","labels"):
        padding=-100 if key=="labels" else processor.tokenizer.pad_token_id if key=="input_ids" else 0
        batch[key]=torch.cat([F.pad(row[key],(maximum-row[key].shape[-1],0),value=padding) for row in rows])
    for key in ("pixel_values","image_grid_thw"):
        batch[key]=torch.cat([row[key] for row in rows])
    batch=to_device(batch);labels=batch.pop("labels")
    keep=max(int((row["labels"]!=-100).sum()) for row in rows)+1
    result=model(**batch,logits_to_keep=keep,use_cache=False)
    targets=labels[:,-keep+1:];logits=result.logits[:,:-1,:].float()
    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),
        ignore_index=-100,reduction="none").view(targets.shape)
    scores=(-(losses.sum(-1)/(targets!=-100).sum(-1))).cpu().tolist()
    order=sorted(range(4),key=lambda i:scores[i],reverse=True)
    margin=scores[order[0]]-scores[order[1]]
    selected=answers[order[0]]
    if margin<minimum_margin:selected={"scene":"unknown","status":"uncertain"}
    return {"value":selected,"unthresholded":answers[order[0]],"margin_nats_per_token":margin,
            "visibility_ok":True,"quality":quality,
            "candidate_mean_log_likelihood":scores,"latency_ms":(time.monotonic()-started)*1000,
            "method":"LoRA Qwen constrained response likelihood; margin is not a calibrated probability"}
