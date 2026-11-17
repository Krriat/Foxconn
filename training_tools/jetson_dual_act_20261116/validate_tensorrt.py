"""Compare TensorRT native-unit actions to original FP32 on held-out fixtures."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.append('/usr/lib/python3/dist-packages')
import numpy as np
import torch
from jetson_policy_runtime import ActEngine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--variant',default='canonical')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    result=dict(variant=args.variant,primary_reference='Original successful desktop ACT FP32 outputs',
                stages={},hardware_commands=0)
    fixtures={}
    for stage in ('far','near'):
        with np.load(args.bundle/(stage+'_validation.npz')) as a:
            fixtures[stage]=[dict(images=list(im),state=st) for im,st in zip(a['images'],a['states'])]
        reference=ActEngine(args.bundle/'models'/stage,args.bundle)
        plan=args.bundle/(stage+'.engine' if args.variant=='canonical' else stage+'_'+args.variant+'.engine')
        candidate=ActEngine(args.bundle/'models'/stage,args.bundle,backend='tensorrt',engine_path=plan)
        differences=[];hysteresis_flips=0;references=[];candidates=[]
        for obs in fixtures[stage]:
            actual=candidate.predict(obs);expected=reference.predict(obs)
            references.append(expected);candidates.append(actual)
            differences.append(np.abs(actual-expected))
            for threshold in (.4,.6):
                hysteresis_flips+=int(((actual[:,14:]>=threshold)!=(expected[:,14:]>=threshold)).sum())
        runtime_delta=np.stack(differences)
        desktop=np.load(args.bundle/(stage+'_desktop_validation.npy'))
        delta=np.abs(np.stack(candidates)-desktop)
        desktop_flips=sum(int(((np.stack(candidates)[:,:,14:]>=t)!=(desktop[:,:,14:]>=t)).sum()) for t in (.4,.6))
        result['stages'][stage]=dict(cases=len(differences),max_joint_delta_rad=float(delta[:,:,:14].max()),
            mean_joint_delta_rad=float(delta[:,:,:14].mean()),max_suction_delta=float(delta[:,:,14:].max()),
            first10_max_joint_delta_rad=float(delta[:,:10,:14].max()),
            passed=bool(np.isfinite(delta).all() and delta[:,:,:14].max()<.001 and delta[:,:,14:].max()<.005),
            hysteresis_threshold_crossings=desktop_flips,
            crossings_vs_jetson_pytorch=hysteresis_flips,
            max_joint_delta_vs_jetson_pytorch=float(runtime_delta[:,:,:14].max()),
            precision=json.loads(plan.with_suffix('.engine.json').read_text())['precision'])
        desktop_path=args.bundle/(stage+'_desktop_validation.npy')
        if desktop_path.exists():
            desktop=np.load(desktop_path)
            for name,predictions in (('pytorch_vs_desktop',references),('tensorrt_vs_desktop',candidates)):
                d=np.abs(np.stack(predictions)-desktop)
                result['stages'][stage][name]=dict(max_joint_delta_rad=float(d[:,:,:14].max()),
                    mean_joint_delta_rad=float(d[:,:,:14].mean()),max_suction_delta=float(d[:,:,14:].max()))
        del reference,candidate;gc.collect();torch.cuda.empty_cache()
    models={s:ActEngine(args.bundle/'models'/s,args.bundle,backend='tensorrt',
        engine_path=args.bundle/(s+'.engine' if args.variant=='canonical' else s+'_'+args.variant+'.engine')) for s in ('far','near')}
    for stage,engine in models.items():
        obs=fixtures[stage][0]
        for _ in range(20):engine.predict(obs)
        durations=[]
        for i in range(200):
            obs=fixtures[stage][i%len(fixtures[stage])]
            start=time.perf_counter();engine.predict(obs);durations.append((time.perf_counter()-start)*1000)
        result['stages'][stage]['timing']=dict(median_ms=float(np.median(durations)),
            p95_ms=float(np.percentile(durations,95)),max_ms=max(durations),samples=len(durations))
    result.update(passed=all(s['passed'] for s in result['stages'].values()),both_models_resident=True,
        timing='RGB CPU preprocessing, GPU inference, native-unit postprocess, CPU result; excludes JPEG/ROS')
    output='tensorrt_validation.json' if args.variant=='canonical' else 'validation_'+args.variant+'.json'
    (args.bundle/output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
