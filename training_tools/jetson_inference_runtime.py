"""Inference-only loaders using the unchanged ACT/InterACT network definitions."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_file
import torch


class Engine:
    def __init__(self, root, name, device="cuda"):
        self.name = name
        self.device = device
        checkpoint = Path(root) / name
        self.config = json.loads((checkpoint / "config.json").read_text())
        if name == "act":
            from act_core import ACT
            cfg = SimpleNamespace(**self.config)
            cfg.pretrained_backbone_weights = None
            cfg.image_features = {k: SimpleNamespace(**v) for k, v in cfg.input_features.items() if v["type"] == "VISUAL"}
            cfg.robot_state_feature = SimpleNamespace(**cfg.input_features["observation.state"])
            cfg.env_state_feature = None
            cfg.action_feature = SimpleNamespace(**cfg.output_features["action"])
            self.model = ACT(cfg)
            weights = load_file(checkpoint / "model.safetensors")
            self.model.load_state_dict({k.removeprefix("model."): v for k, v in weights.items()}, strict=True)
            self.stats = {k: v.to(device) for k, v in load_file(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors").items()}
            self.out_stats = {k: v.to(device) for k, v in load_file(checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").items()}
            self.keys = list(cfg.image_features)
        else:
            from xcore_interact.policy import InterACT3DPolicy
            from xcore_interact.data import NormalizationStats
            self.model = InterACT3DPolicy.from_pretrained(checkpoint)
            self.normalization = NormalizationStats.load(checkpoint / "normalization.json")
            self.stats = self.normalization.tensors(device)
        self.model.to(device).eval()

    def prepare(self, images, state):
        if self.name == "act":
            result = {}
            for key, image in zip(self.keys, images):
                tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255).unsqueeze(0).to(self.device)
                result[key] = (tensor - self.stats[key + ".mean"]) / (self.stats[key + ".std"] + 1e-8)
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            result["observation.state"] = (tensor - self.stats["observation.state.mean"]) / (self.stats["observation.state.std"] + 1e-8)
            result["observation.images"] = [result[k] for k in self.keys]
            return result
        from xcore_interact.data import preprocess_observation
        return preprocess_observation(*images, state=state, stats=self.normalization,
                                      image_size=self.config["image_size"], device=self.device)

    def forward(self, prepared):
        if self.name == "act":
            return self.model(prepared)[0]
        return self.model.predict_action_chunk(prepared[0], prepared[1], prepared[2])

    def postprocess(self, action):
        action = action.float()
        if self.name == "act":
            action = action * (self.out_stats["action.std"] + 1e-8) + self.out_stats["action.mean"]
        else:
            action = action * self.stats["action_std"] + self.stats["action_mean"]
        return action.cpu().numpy()
