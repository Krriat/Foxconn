"""Reviewed stage-to-checkpoint mapping; no ROS or hardware access."""
import re


def stage_specs(config):
    specs = config.get("act_stage_checkpoints")
    if specs is None:
        return {}
    if config["model"] != "act" or not isinstance(specs, dict) or set(specs) != {"far", "near"}:
        raise ValueError("Stage checkpoints require ACT and exactly far/near entries")
    for spec in specs.values():
        if (not isinstance(spec, dict) or not isinstance(spec.get("path"), str)
                or not spec["path"].startswith("/")
                or not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", "")))):
            raise ValueError("Stage checkpoint needs an absolute path and SHA-256")
    if specs["far"]["sha256"] != config["checkpoint_sha256"]["act"]:
        raise ValueError("Default ACT must be the reviewed far checkpoint")
    return specs


def checkpoint_map(config):
    return {"default": config["checkpoint_sha256"][config["model"]],
            **{key: spec["sha256"] for key, spec in stage_specs(config).items()}}


def selected_policy(config, sequence):
    key = "default"
    if stage_specs(config) and sequence and sequence.get("result") == "running":
        key = sequence.get("stage")
        if key not in ("far", "near"):
            raise ValueError("Invalid active task stage")
    return key, checkpoint_map(config)[key]


def action_config(config, sequence):
    key, sha = selected_policy(config, sequence)
    return key, dict(config, checkpoint_sha256={**config["checkpoint_sha256"], config["model"]: sha})
