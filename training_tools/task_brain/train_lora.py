"""Supervised image-to-JSON LoRA task completion training. No ROS imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from training_tools.task_brain.lora_data import samples,encode,motion_negatives
from training_tools.task_brain.lora_model import load,processor,to_device,BASE


def append(path,value):
    with path.open("a") as f:f.write(json.dumps(value)+"\n")
    print(json.dumps(value),flush=True)


def supervised_loss(model,batch,tokenizer):
    labels=batch.pop("labels")
    # Compute only logits preceding assistant tokens. Avoid a giant
    # (all image/prompt tokens x 248K vocabulary) loss tensor.
    positions=torch.nonzero(labels[0]!=-100).flatten()
    if not len(positions) or int(positions[0])<1:raise ValueError("Invalid response mask")
    logits=model(**batch,logits_to_keep=positions-1,use_cache=False).logits
    targets=labels[0,positions]
    losses=F.cross_entropy(logits[0].float(),targets,reduction="none")
    keywords=("closed","far","both","unknown","running","complete","uncertain")
    weights=torch.tensor([6. if any(w in tokenizer.decode([int(t)]) for w in keywords) else 1.
                          for t in targets],device=losses.device)
    return (losses*weights).sum()/weights.sum()


@torch.no_grad()
def validation_loss(model,processor_,rows):
    model.eval();losses=[]
    for row in rows:
        batch=to_device(encode(processor_,row,training=True))
        losses.append(float(supervised_loss(model,batch,processor_.tokenizer)))
    model.train()
    return sum(losses)/len(losses)


def main():
    parser_=argparse.ArgumentParser(description=__doc__)
    parser_.add_argument("--data",type=Path,required=True)
    parser_.add_argument("--base",type=Path,default=BASE)
    parser_.add_argument("--output",type=Path,required=True)
    parser_.add_argument("--steps",type=int,default=100)
    parser_.add_argument("--accumulation",type=int,default=4)
    parser_.add_argument("--rank",type=int,default=8)
    parser_.add_argument("--smoke",action="store_true")
    parser_.add_argument("--resume-adapter",type=Path)
    args=parser_.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    random.seed(147);torch.manual_seed(147)
    proc=processor(args.base);model=load(base=args.base,adapter=args.resume_adapter,train=True,rank=args.rank)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise ValueError("No trainable LoRA weights")
    if any("lora_" not in name for name,p in model.named_parameters() if p.requires_grad):
        raise ValueError("Base weights unexpectedly trainable")
    train=samples(args.data,"train",per_window=8)
    train+=motion_negatives(args.data,"train",per_episode=8)
    # Include explicit uncertainty supervision, independent of episode time.
    train += [dict(r,label=3,corruption="dark" if i%2 else "covered") for i,r in enumerate(train[::12])]
    validation=samples(args.data,"validation",per_window=1)+motion_negatives(args.data,"validation",per_episode=2)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    started=time.monotonic();best=float("inf");manifest={
        "base":str(args.base),"method":"true LoRA on NF4 base (QLoRA)","rank":args.rank,
        "alpha":args.rank*2,"dropout":.05,"trainable_layers":"language layers 24-31 attention/MLP and visual merger LoRA",
        "vision_encoder":"backbone frozen BF16; merger LoRA trainable", "max_pixels":640*480,
        "input":"head-camera box crop only; no wrist-pose shortcut", "resume_adapter":str(args.resume_adapter),
        "training_augmentation":"brightness .65-1.3, offset -8..8",
        "key_token_loss_weight":6.,"trainable_parameters":sum(p.numel() for p in params),
        "train_samples":len(train),"validation_samples":len(validation),"batch_size":1,
        "gradient_accumulation":args.accumulation,"learning_rate":1e-4,"seed":147,
        "no_robot_commands":True,"qwen_base_modified":False,"steps":args.steps}
    (args.output/"training_config.json").write_text(json.dumps(manifest,indent=2)+"\n")
    append(args.output/"training.jsonl",{"event":"loaded",**manifest,
        "gpu_allocated_bytes":torch.cuda.memory_allocated()})
    cursor=0;random.shuffle(train)
    for step in range(1,(1 if args.smoke else args.steps)+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True);loss_sum=0
        before=params[1].detach().clone() if step==1 else None
        for micro in range(args.accumulation):
            if cursor==len(train):random.shuffle(train);cursor=0
            row=dict(train[cursor],augment=True);cursor+=1
            batch=to_device(encode(proc,row,training=True))
            loss=supervised_loss(model,batch,proc.tokenizer)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite LoRA loss")
            (loss/args.accumulation).backward();loss_sum+=float(loss)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not math.isfinite(norm):raise FloatingPointError("Nonfinite LoRA gradient")
        optimizer.step()
        updated=bool((params[1].detach()!=before).any()) if before is not None else None
        if step==1 and not updated:raise RuntimeError("LoRA weights did not update")
        append(args.output/"training.jsonl",{"step":step,"loss":loss_sum/args.accumulation,
            "gradient_norm":norm,"seconds":time.monotonic()-tick,"elapsed_seconds":time.monotonic()-started,
            "gpu_peak_bytes":torch.cuda.max_memory_allocated(),"adapter_changed":updated})
        if args.smoke or step%20==0 or step==args.steps:
            val=None if args.smoke else validation_loss(model,proc,validation)
            append(args.output/"training.jsonl",{"event":"validation","step":step,"loss":val})
            model.save_pretrained(args.output/"last_adapter");proc.save_pretrained(args.output/"last_adapter")
            if val is None or val<best:
                best=val if val is not None else best
                model.save_pretrained(args.output/"adapter");proc.save_pretrained(args.output/"adapter")
                (args.output/"best.json").write_text(json.dumps({"step":step,"validation_loss":val})+"\n")
    append(args.output/"training.jsonl",{"event":"training_complete","smoke_only":args.smoke,
        "elapsed_seconds":time.monotonic()-started,"best_validation_loss":None if args.smoke else best})


if __name__=="__main__":main()
