"""Promote only the evaluated LoRA artifact; no ROS or hardware operations."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/"training_verification/task_brain_20261112"
MODEL=ROOT/"model_outputs/task_brain_lora_20261112"


def main():
    endpoint=json.loads((EVIDENCE/"lora_test_03/report.json").read_text())
    replay=json.loads((EVIDENCE/"dense_replay_03/report.json").read_text())
    if not replay["passed"] or endpoint["adapter"]["false_completion_count"]:
        raise RuntimeError("Final offline qualification failed")
    for name in ("decision_tests.log","observer_ros_tests.log","executor_brain_ros_tests.log","executor_manual_regression.log"):
        text=(EVIDENCE/name).read_text()
        if "Traceback" in text or "AssertionError" in text or "FAILED" in text:raise RuntimeError(name)
    src=MODEL/"run_03/adapter";dst=MODEL/"adapter"
    sha=hashlib.sha256((src/"adapter_model.safetensors").read_bytes()).hexdigest()
    if dst.exists():
        if hashlib.sha256((dst/"adapter_model.safetensors").read_bytes()).hexdigest()!=sha:
            raise RuntimeError("Refusing to replace a different released adapter")
    else:shutil.copytree(src,dst)
    config=json.loads((MODEL/"run_03/training_config.json").read_text())
    (MODEL/"brain_config.json").write_text(json.dumps(config,indent=2)+"\n")
    sources=MODEL/"released_source";sources.mkdir(exist_ok=True)
    hashes={}
    for path in (ROOT/"training_tools/task_brain").iterdir():
        if path.is_file():
            shutil.copyfile(path,sources/path.name);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={"base_model":"Qwen/Qwen3.5-9B","adapter_sha256":sha,
        "method":"NF4 base with true language/vision-merger LoRA, no separate classifier",
        "chosen_adapter":"run_03/adapter","training":config,
        "warm_median_inference_ms":endpoint["adapter"]["median_ms"],
        "endpoint_checks":endpoint,"temporal_replay_checks":replay,
        "offline_checks_passed":True,"live_motion_tested":False,
        "transition_test_domain":77,"default_deployment_mode":"observe (no robot service clients)",
        "source_sha256":hashes,
        "limits":"Development replay after iterative error analysis; not an independent live success rate."}
    for path in (EVIDENCE/"release_manifest.json",MODEL/"release_manifest.json"):
        path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"released_adapter":str(dst),"sha256":sha,"replay_passed":replay["passed"]}),flush=True)


if __name__=="__main__":main()
