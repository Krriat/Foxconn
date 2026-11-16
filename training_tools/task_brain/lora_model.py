"""Qwen3.5-9B NF4 base + true trainable LoRA; no second image classifier."""
from pathlib import Path
import re

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"model_outputs/base/Qwen3.5-9B"


def processor(base=BASE):
    p=AutoProcessor.from_pretrained(str(base),local_files_only=True,
        min_pixels=128*128,max_pixels=640*480)
    p.tokenizer.padding_side="left"
    return p


def load(*, base=BASE, adapter=None, train=False, rank=8):
    torch.set_num_threads(6)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual","lm_head"])
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(base),local_files_only=True,
        quantization_config=quant,dtype=torch.bfloat16,device_map={"":"cuda:0"},
        attn_implementation="sdpa")
    model.config.use_cache=not train
    # Keep frozen embedding/vision/lm_head in bf16; casting these 9B components
    # to fp32 would waste several GB. Only LoRA adapter tensors are optimized.
    for parameter in model.parameters():parameter.requires_grad_(False)
    if adapter:
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=train)
    elif train:
        layers="(?:"+"|".join(str(i) for i in range(24,32))+")"
        modules="(?:self_attn\\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\\.(?:in_proj_qkv|in_proj_z|out_proj)|mlp\\.(?:gate_proj|up_proj|down_proj))"
        target=rf"(?:.*language_model\.layers\.{layers}\.{modules}|.*visual\.merger\.(?:linear_fc1|linear_fc2))"
        config=LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,bias="none",
                          task_type="CAUSAL_LM",target_modules=target)
        model=get_peft_model(model,config)
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.train()
    else:model.eval()
    return model


def to_device(batch):
    return {k:v.to("cuda",dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k,v in batch.items()}
