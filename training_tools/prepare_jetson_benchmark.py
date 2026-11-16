#!/usr/bin/env python3
"""Package saved models and exact network source for an isolated Jetson benchmark."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "training_verification/jetson_bundle_20261112"
DEST.mkdir(parents=True, exist_ok=True)
tools = ROOT / "training_tools"
source = ROOT / ".venv-act/lib/python3.10/site-packages/lerobot/policies/act/modeling_act.py"
code = source.read_text()
tree = ast.parse(code)
definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
               and node.name not in ("ACTPolicy", "ACTTemporalEnsembler")]
header = code[:code.index('"""')]
imports = '''from __future__ import annotations
import math
from collections.abc import Callable
from itertools import chain
from types import SimpleNamespace as ACTConfig
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
ACTION = "action"
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGES = "observation.images"
'''
(DEST / "act_core.py").write_text(header + imports + "\n\n" + "\n\n".join(ast.get_source_segment(code, node) for node in definitions) + "\n")
src = ROOT / "AIDE_ACT_dataset_20261111_195100/act/xcroe_interACT/src/xcore_interact"
shutil.copytree(src, DEST / "xcore_interact", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name, run in (("act", "act_suction_20261111_213453"), ("interact", "interact_suction_20261111_234920")):
    checkpoint = ROOT / "model_outputs" / run / "checkpoints/best/pretrained_model"
    shutil.copytree(checkpoint, DEST / name, dirs_exist_ok=True)
for name in ("jetson_inference_runtime.py", "benchmark_jetson.py"):
    shutil.copy2(tools / name, DEST / name)
live = ROOT / "training_verification/ros_live_models_20261112_01"
images = [cv2.cvtColor(cv2.imread(str(live / f"first_{name}.jpg")), cv2.COLOR_BGR2RGB) for name in ("head", "left_wrist", "right_wrist")]
with np.load(live / "predictions.npz") as data:
    state = data["states"][0]
np.savez_compressed(DEST / "sample.npz", images=np.stack(images), state=state)
manifest = {"source": str(source), "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "act_network_definitions": [node.name for node in definitions],
            "sample": "three saved live ROS RGB snapshots (JPEG decoded), first measured 16D robot state; timing fixture, not task validation",
            "models": {name: hashlib.sha256((DEST/name/"model.safetensors").read_bytes()).hexdigest() for name in ("act", "interact")}}
(DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(DEST)
