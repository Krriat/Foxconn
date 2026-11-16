"""Export the original deterministic ACT forward; no image resizing or retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch
from torch import nn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.bundle))
    from jetson_policy_runtime import ActEngine
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    class Forward(nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,head,left,right,state):
            return self.model({'observation.images':[head,left,right],'observation.state':state})[0]
    report={}
    for stage in ('far','near'):
        engine=ActEngine(args.bundle/'models'/stage,args.bundle,device='cpu')
        wrapper=Forward(engine.model).eval()
        dummy=[torch.zeros(1,3,480,640) for _ in range(3)]+[torch.zeros(1,16)]
        path=args.bundle/(stage+'.onnx')
        with torch.inference_mode():
            torch.onnx.export(wrapper,tuple(dummy),path,input_names=['head','left_wrist','right_wrist','state'],
                output_names=['actions'],opset_version=17,dynamo=False,do_constant_folding=True)
        onnx.checker.check_model(str(path))
        report[stage]={'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'weights_sha256':hashlib.sha256((args.bundle/'models'/stage/'model.safetensors').read_bytes()).hexdigest(),
            'inputs':'Three normalized FP32 1x3x480x640 RGB tensors and normalized FP32 1x16 state',
            'output':'FP32 normalized 1x50x16 actions; zero VAE latent; unchanged network'}
    (args.bundle/'onnx_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
