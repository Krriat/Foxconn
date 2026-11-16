"""Show true LoRA modules and ordinary greedy image-to-JSON examples; no ROS."""
import argparse
import json
from pathlib import Path
import time

import torch
from training_tools.task_brain.lora_data import load_image,conversation,RAW
from training_tools.task_brain.lora_model import load,processor,to_device


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    model=load(adapter=args.adapter);proc=processor()
    adapters=[n for n,_ in model.named_parameters() if "lora_" in n]
    assert len(adapters)>80 and any("visual.merger" in n for n in adapters)
    result={"LoRA_tensors":len(adapters),"example_names":adapters[:4]+[n for n in adapters if "visual.merger" in n],
            "optimized_base_weights":False,"examples":[],"robot_commands":0}
    cases=[(54,0,"far"),(54,381,"far"),(55,0,"near"),(55,392,"near"),(54,0,"near"),(55,392,"far")]
    with torch.inference_mode():
        for ep,frame,stage in cases:
            row={"episode":ep,"frame":frame,"stage":stage};image=load_image(row)
            batch=proc.apply_chat_template(conversation(row,image,False),tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt")
            batch=to_device(batch);length=batch["input_ids"].shape[-1];start=time.monotonic()
            output=model.generate(**batch,max_new_tokens=32,do_sample=False,use_cache=True)
            text=proc.tokenizer.decode(output[0,length:],skip_special_tokens=True).strip()
            item={**row,"greedy_output":text,"latency_seconds":time.monotonic()-start}
            result["examples"].append(item);print(json.dumps(item),flush=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
