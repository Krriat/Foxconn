#!/usr/bin/env python3
"""Run the verified ACT and RGB InterACT recipes on this workstation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

WORKSPACE = Path(__file__).resolve().parents[1]
BASE = WORKSPACE / "AIDE_ACT_dataset_20261111_195100"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("act", "interact", "both"))
    parser.add_argument("--smoke", action="store_true", help="20 updates, validation and checkpoint checks")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-base", type=Path, default=WORKSPACE / "model_outputs")
    parser.add_argument("--eval-batches", type=int, help="Override validation sampling; 0 uses all frames")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.model == "both" and (args.output_dir or args.resume_from):
        parser.error("output-dir/resume-from require a single model")
    recipe = json.loads((Path(__file__).parent / "validated_config.json").read_text())
    manifest_path = Path(recipe["data_root"]) / "split_manifest.json"
    if not json.loads(manifest_path.read_text())["complete"]:
        raise RuntimeError("dataset export is incomplete")
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{BASE}/act/xcore_act/src:{BASE}/act/xcroe_interACT/src",
               PYTHONNOUSERSITE="1", TORCH_HOME=str(WORKSPACE / ".cache/torch"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    python = WORKSPACE / ".venv-act/bin/python"
    models = ("act", "interact") if args.model == "both" else (args.model,)
    for model in models:
        settings = dict(recipe["models"][model]["parameters"])
        tag = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output = (args.output_dir.resolve() if args.output_dir else
                  args.resume_from.resolve().parent.parent if args.resume_from else
                  args.output_base.resolve() / f"{'smoke_' if args.smoke else ''}{model}_suction_{tag}")
        if args.smoke:
            settings.update(steps=20, save_freq=10, log_freq=1, eval_freq=10, eval_batches=4)
        if args.steps is not None:
            settings["steps"] = args.steps
        if args.eval_batches is not None:
            if args.eval_batches < 0:
                parser.error("eval-batches must be nonnegative")
            settings["eval_batches"] = args.eval_batches
        if settings["steps"] <= 0:
            parser.error("steps must be positive")
        command = [str(python), str(BASE / recipe["models"][model]["script"]),
                   "--dataset-root", str(Path(recipe["data_root"]) / "train"),
                   "--dataset-repo-id", "local/dual_arm_suction_train",
                   "--validation-root", str(Path(recipe["data_root"]) / "val"),
                   "--validation-repo-id", "local/dual_arm_suction_val",
                   "--output-dir", str(output)]
        for key, value in settings.items():
            command += ["--"+key.replace("_", "-"), str(value)]
        if args.resume_from:
            command += ["--resume-from", str(args.resume_from.resolve())]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, cwd=WORKSPACE, check=True)


if __name__ == "__main__":
    main()
