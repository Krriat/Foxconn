"""Jetson two-model fixture equivalence/timing. No ROS or robot clients."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.append('/usr/lib/python3/dist-packages')
import numpy as np
import torch
import yaml
from jetson_policy_runtime import ActEngine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    with np.load(args.bundle/'sample.npz') as a:
        observation = dict(images=list(a['images']), state=a['state'])
    engines = {s: ActEngine(args.bundle/'models'/s, args.bundle) for s in ('far', 'near')}
    report = {}
    for stage, engine in engines.items():
        for _ in range(10):
            prediction = engine.predict(observation)
        reference = np.load(args.bundle/f'{stage}_reference.npy')
        np.testing.assert_allclose(prediction, reference, atol=3e-4, rtol=1e-4)
        durations = []
        for _ in range(60):
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.predict(observation)
            torch.cuda.synchronize()
            durations.append((time.perf_counter()-start)*1000)
        report[stage] = dict(shape=list(prediction.shape), passed=True,
            max_abs_delta_from_desktop=float(np.max(np.abs(prediction-reference))),
            median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations,95)),
            maximum_ms=max(durations))
    result = dict(passed=True, both_models_resident=True, precision='fp32', torch=torch.__version__,
        cuda=torch.version.cuda, results=report, hardware_commands=0,
        timing='CPU RGB preprocessing + H2D + forward + native-unit postprocess + D2H; excludes JPEG/ROS')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
