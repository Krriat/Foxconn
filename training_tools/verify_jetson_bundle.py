#!/usr/bin/env python3
"""Compare inference-only package predictions against the original trainers."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
bundle = ROOT / "training_verification/jetson_bundle_20261112"
sys.path[:0] = [str(bundle), str(ROOT / "AIDE_ACT_dataset_20261111_195100/act/xcore_act/scripts/auditing")]
from jetson_inference_runtime import Engine
from audit_xcore_act_checkpoints import load_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.utils import prepare_observation_for_inference
from xcore_interact.policy import InterACT3DPolicy
from xcore_interact.data import preprocess_observation, NormalizationStats, denormalize_actions

sample = np.load(bundle / "sample.npz")
images, state = list(sample["images"]), sample["state"]
torch.set_num_threads(4)
checks = {}
with torch.inference_mode():
    for name in ("act", "interact"):
        engine = Engine(bundle, name)
        got = engine.postprocess(engine.forward(engine.prepare(images, state)))
        if name == "act":
            meta = LeRobotDatasetMetadata("local/dual_arm_suction_train", root=ROOT / "prepared_data/suction_20hz_v1/train")
            model, pre, post, keys = load_policy(bundle / name, meta, "cuda")
            raw = {"observation.state": state, **dict(zip(keys, images))}
            batch = pre(prepare_observation_for_inference(raw, torch.device("cuda"), task="", robot_type=""))
            expected = post(model.predict_action_chunk(batch)).float().cpu().numpy()
        else:
            model = InterACT3DPolicy.from_pretrained(bundle / name).cuda().eval()
            stats = NormalizationStats.load(bundle / name / "normalization.json")
            inputs = preprocess_observation(*images, state=state, stats=stats, image_size=224, device="cuda")
            expected = denormalize_actions(model.predict_action_chunk(*inputs), stats).float().cpu().numpy()
        difference = float(abs(got - expected).max())
        assert np.allclose(got, expected, atol=1e-6, rtol=1e-5), (name, difference)
        checks[name] = {"max_abs_difference": difference, "shape": list(got.shape)}
        np.save(bundle / (name + "_reference_fp32.npy"), expected)
        del engine, model
        torch.cuda.empty_cache()
(bundle / "local_equivalence.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks))
