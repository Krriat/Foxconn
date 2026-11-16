"""Direct image-to-status inference with the trained Qwen LoRA adapter."""
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import torch

from training_tools.task_brain.decision import parse_answer
from training_tools.task_brain.lora_data import conversation
from training_tools.task_brain.lora_model import load,processor,to_device,BASE
from training_tools.task_brain.scoring import classify


class Brain:
    def __init__(self, adapter, *, base=BASE):
        self.adapter=Path(adapter)
        self.sha=hashlib.sha256((self.adapter/"adapter_model.safetensors").read_bytes()).hexdigest()
        self.processor=processor(base);self.model=load(base=base,adapter=self.adapter)

    @torch.inference_mode()
    def predict(self,image,stage):
        if stage not in ("far","near"):raise ValueError("Unknown stage")
        result=classify(self.model,self.processor,image,stage)
        result["adapter_sha256"]=self.sha
        parse_answer(result["value"],stage)
        return result
