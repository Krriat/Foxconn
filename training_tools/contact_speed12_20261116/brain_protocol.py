"""Validate keyed brain completions before the existing stage stop/home logic."""
import re

FIELDS={"run_id","stage","sequence_id","adapter_sha256","image_time_ns","decision_time_ns","scene"}


def validate_brain_completion(value,context,config,*,now_ns,elapsed_seconds,home_after_ns):
    if not isinstance(value,dict) or set(value)!=FIELDS:raise ValueError("Unexpected brain completion fields")
    job=context.get("task_sequence") or {}
    if (context.get("phase")!="RUNNING" or job.get("result")!="running" or
            job.get("completion_mode")!="qwen_lora" or job.get("pending")):
        raise ValueError("No active automatic brain sequence stage")
    for field,expected in (("run_id",context.get("run_id")),("stage",job.get("stage")),("sequence_id",job.get("id"))):
        if value[field]!=expected:raise ValueError("Stale brain "+field)
    expected=config.get("brain_adapter_sha256","")
    if not re.fullmatch(r"[0-9a-f]{64}",expected) or value["adapter_sha256"]!=expected:
        raise ValueError("Unreviewed brain adapter")
    if elapsed_seconds<5.:raise ValueError("Brain stage minimum duration not reached")
    for field in ("image_time_ns","decision_time_ns"):
        if type(value[field]) is not int or not 0<=now_ns-value[field]<=2_500_000_000:
            raise ValueError("Stale or future brain "+field)
    if value["image_time_ns"]<=home_after_ns or value["decision_time_ns"]<value["image_time_ns"]:
        raise ValueError("Brain observation predates verified home")
    if value["scene"] not in (("far_open","both_open") if value["stage"]=="far" else ("both_open",)):
        raise ValueError("Brain scene contradicts stage completion")
    return value["stage"]
