#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.utils import prepare_observation_for_inference

ROOT = Path('/home/jj/code/ACT')
BASE = ROOT / 'AIDE_ACT_dataset_20261111_195100/act'
sys.path[:0] = [str(BASE / 'xcore_act/src'), str(BASE / 'xcroe_interACT/src'), str(BASE / 'xcore_act/scripts/auditing')]
from audit_xcore_act_checkpoints import load_policy, make_raw_observation
from xcore_interact.data import NormalizationStats, preprocess_observation, denormalize_actions
from xcore_interact.policy import InterACT3DPolicy

os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
root = ROOT / 'prepared_data/suction_20hz_v1/val'
ds = LeRobotDataset('local/dual_arm_suction_val', root=root, video_backend='pyav')
item = ds[0]
actpath = ROOT / 'model_outputs/act_suction_20261111_213453/checkpoints/best/pretrained_model'
intpath = ROOT / 'model_outputs/interact_suction_20261111_234920/checkpoints/best/pretrained_model'
act, pre, post, keys = load_policy(actpath, ds.meta, 'cuda')
raw = make_raw_observation(item, keys)
images = [raw[k] for k in ('observation.images.head', 'observation.images.left_wrist', 'observation.images.right_wrist')]
state = raw['observation.state']
interact = InterACT3DPolicy.from_pretrained(intpath).cuda().eval()
stats = NormalizationStats.load(intpath / 'normalization.json')

def act_pre():
    return pre(prepare_observation_for_inference(dict(raw), torch.device('cuda'), task='', robot_type=''))
def interact_pre():
    return preprocess_observation(*images, state=state, stats=stats, image_size=interact.config.image_size, device='cuda')
def summarize(values):
    return {'mean_ms': float(np.mean(values)), 'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values,95)), 'p99_ms': float(np.percentile(values,99)), 'max_ms': float(np.max(values))}
def measure(fn, warm=30, n=200):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    out=[]
    for _ in range(n):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
        out.append((time.perf_counter()-start)*1000)
    return summarize(out)

report={'device':torch.cuda.get_device_name(), 'torch':torch.__version__, 'sample':'validation frame 0, three RGB cameras and 16-dim state', 'batch_size':1, 'warmup':30, 'iterations':200, 'torch_cpu_threads':4, 'chunk_size':50, 'timing':'wall clock synchronized CUDA; end-to-end excludes camera capture, ROS and robot command', 'results':{}}
with torch.inference_mode():
  for name in ('act','interact'):
    for precision in ('fp32','bf16'):
      def autocast(): return torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16')
      if name=='act':
        batch=act_pre()
        def forward():
          with autocast(): return act.predict_action_chunk(batch)
        def end():
          b=act_pre()
          with autocast(): y=act.predict_action_chunk(b)
          return post(y.float()).detach().cpu().numpy()
      else:
        inputs=interact_pre()
        def forward():
          with autocast(): return interact.predict_action_chunk(inputs[0],inputs[1],inputs[2])
        def end():
          i=interact_pre()
          with autocast(): y=interact.predict_action_chunk(i[0],i[1],i[2])
          return denormalize_actions(y[0].float(),stats).cpu().numpy()
      y=forward(); actual=end()
      assert tuple(y.shape)==(1,50,16) and np.isfinite(actual).all(), (name, y.shape)
      entry={'forward':measure(forward), 'preprocess_forward_postprocess_cpu':measure(end), 'output_shape':list(y.shape)}
      report['results'][f'{name}_{precision}']=entry
      print(name,precision,json.dumps(entry,ensure_ascii=False),flush=True)
out=ROOT/'training_verification/inference_latency_20261112.json'
out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('report:',out,flush=True)
