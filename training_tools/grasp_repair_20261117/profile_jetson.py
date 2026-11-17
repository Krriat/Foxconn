"""Profile equivalent FP32 preprocessing and CUDA graph; no ROS or hardware IO."""
import argparse,json,sys,time
from pathlib import Path
sys.path.append("/usr/lib/python3/dist-packages")
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=80)
    a=p.parse_args();sys.path.insert(0,str(a.bundle.parent))
    from jetson_policy_runtime import ActEngine
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
    data=np.load(a.bundle/'far_validation.npz');obs=dict(images=list(data['images'][0]),state=data['states'][0])
    engine=ActEngine(a.bundle/'models/far',a.bundle,backend='tensorrt')
    mean=torch.stack([engine.stats[k+'.mean'] for k in engine.keys]);std=torch.stack([engine.stats[k+'.std'] for k in engine.keys])
    if mean.ndim!=4:raise ValueError(str(mean.shape))
    def gpu_prepare(obs):
        pixels=torch.from_numpy(np.stack(obs['images'])).to('cuda')
        pixels=pixels.permute(0,3,1,2).contiguous().float().div_(255)
        pixels=(pixels-mean)/(std+1e-8)
        state=torch.from_numpy(obs['state']).float().unsqueeze(0).to('cuda')
        state=(state-engine.stats['observation.state.mean'])/(engine.stats['observation.state.std']+1e-8)
        return {'observation.images':[im.unsqueeze(0) for im in pixels.unbind(0)],'observation.state':state}
    def fast_predict(obs):
        batch=gpu_prepare(obs);pred=engine.model(batch)[0].float()
        return (pred*(engine.out['action.std']+1e-8)+engine.out['action.mean']).cpu().numpy()[0]
    report={'hardware_commands':0,'variants':{}}
    for label,fn in (('original_cpu_preprocess',engine.predict),('gpu_uint8_preprocess',fast_predict)):
        for _ in range(12):fn(obs)
        durations=[]
        for _ in range(a.samples):
            t=time.perf_counter();fn(obs);durations.append((time.perf_counter()-t)*1000)
        report['variants'][label]={'median_ms':float(np.median(durations)),'p95_ms':float(np.percentile(durations,95)),'max_ms':max(durations)}
    deltas=[];flips=0
    for images,state in zip(data['images'],data['states']):
        obs=dict(images=list(images),state=state);base=engine.predict(obs);fast=fast_predict(obs);deltas.append(abs(base-fast))
        flips+=sum(int(((base[:,14:]>=x)!=(fast[:,14:]>=x)).sum()) for x in (.4,.6))
    delta=np.stack(deltas);report['equivalence']={'max_joint_delta_rad':float(delta[:,:,:14].max()),'max_IO_delta':float(delta[:,:,14:].max()),'threshold_crossings':flips}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
