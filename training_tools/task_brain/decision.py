"""Pure stage transition logic. No ROS, camera, model, SDK, or IO writes."""
from dataclasses import dataclass
import math


def parse_answer(value, stage):
    if not isinstance(value,dict) or set(value)!={"scene","status"}:
        raise ValueError("Unexpected model response schema")
    if value["scene"] not in ("closed","far_open","both_open","unknown"):
        raise ValueError("Unknown scene value")
    if value["status"] not in ("running","complete","uncertain"):
        raise ValueError("Unknown completion value")
    if stage not in ("far","near"):raise ValueError("Invalid ACT stage")
    if value["status"]=="complete":
        allowed=("far_open","both_open") if stage=="far" else ("both_open",)
        if value["scene"] not in allowed:raise ValueError("Completion contradicts scene")
    if value["scene"]=="unknown" and value["status"]!="uncertain":
        raise ValueError("Unknown scene must abstain")
    return dict(value)


@dataclass
class Evidence:
    run_id: str
    stage: str
    image_stamp_ns: int
    captured_mono: float
    value: dict


class Gate:
    def __init__(self, *, min_stage_seconds=5., hold_seconds=1.5,
                 required=3, max_result_age=2.5, max_gap=1.5):
        self.min_stage_seconds=min_stage_seconds;self.hold_seconds=hold_seconds
        self.required=required;self.max_result_age=max_result_age;self.max_gap=max_gap
        self.key=None;self.started=0.;self.samples=[];self.last_stamp=0
        self.dispatched=set();self.reason="waiting for active stage"

    @staticmethod
    def context_key(context):
        job=context.get("task_sequence") or {}
        if (context.get("phase")!="RUNNING" or context.get("mode")!="live" or
            job.get("result")!="running" or job.get("stage") not in ("far","near") or job.get("pending")):
            return None
        return (context.get("run_id"),job["stage"],job.get("id"))

    def observe_context(self,context,now):
        key=self.context_key(context)
        if key!=self.key:
            self.key=key;self.started=now;self.samples=[];self.last_stamp=0
        return key

    def invalidate(self,reason):
        self.samples=[];self.reason=reason

    def update(self,evidence,context,*,now,context_age):
        key=self.observe_context(context,now)
        if key is None:
            self.invalidate("executor is not running a sequence stage");return None
        if (not math.isfinite(context_age) or context_age<0 or context_age>.4):
            self.invalidate("executor status stale");return None
        if key in self.dispatched:
            self.reason="transition already dispatched for this stage";return None
        if (evidence.run_id,evidence.stage)!=key[:2]:
            self.invalidate("late result from another run/stage");return None
        age=now-evidence.captured_mono
        if not math.isfinite(age) or age<0 or age>self.max_result_age:
            self.invalidate("model result too old");return None
        if evidence.image_stamp_ns<=self.last_stamp:
            self.invalidate("duplicate or reordered observation");return None
        self.last_stamp=evidence.image_stamp_ns
        try:value=parse_answer(evidence.value,evidence.stage)
        except ValueError as e:
            self.invalidate(str(e));return None
        if value["status"]!="complete":
            self.invalidate("model has not confirmed visual completion");return None
        if now-self.started<self.min_stage_seconds:
            self.invalidate("stage start grace period");return None
        if self.samples and evidence.captured_mono-self.samples[-1]>self.max_gap:
            self.samples=[]
        self.samples.append(evidence.captured_mono)
        # Bounded memory and a sustained sequence of fresh, distinct images.
        self.samples=self.samples[-20:]
        if len(self.samples)<self.required or self.samples[-1]-self.samples[0]<self.hold_seconds:
            self.reason="confirming completion on successive images";return None
        self.reason="LoRA model confirmed stable stage completion"
        return {"service":"finish_far" if evidence.stage=="far" else "finish_near",
                "run_id":evidence.run_id,"stage":evidence.stage,"sequence_id":key[2]}

    def mark_dispatched(self,decision):
        key=(decision["run_id"],decision["stage"],decision["sequence_id"])
        if key!=self.key or key in self.dispatched:raise ValueError("Stale or duplicate dispatch")
        self.dispatched.add(key)
