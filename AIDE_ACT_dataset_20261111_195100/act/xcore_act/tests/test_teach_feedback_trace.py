"""Causal attribution fixtures; no ROS, SDK connections, robot or camera access."""
import importlib.util
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("analyze_teach_feedback", SCRIPTS/"diagnostics/analyze_teach_feedback.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
sys.path.insert(0, str(SCRIPTS.parent/"src"))
from xcore_teach_recording.feedback_trace import FeedbackTrace


def fixture(tmp_path, sdk_stall=False, callback_stall=False):
    sdk, receive = [], []
    for side in ("left", "right"):
        sdk += [dict(kind="sdk",side=side,op="drag.status",phase="teaching",mono_ns=1),
                dict(kind="sdk",side=side,op="drag.status",phase="stopped",mono_ns=12000000000)]
        for k in range(1,1200):
            mono=k*10000000
            if sdk_stall and side=="left" and 100<=k<120:
                continue
            stamp=100000000000+mono
            sdk.append(dict(kind="sdk",side=side,op="drag.publish",phase="sample",mono_ns=mono,stamp_ns=stamp))
            arrival=mono+1000000
            if callback_stall and 100<=k<120:
                arrival=1201000000
            receive.append(dict(kind="receive",stream=side+".drag",mono_ns=arrival,done_ns=arrival+10000,stamp_ns=stamp))
    for c in ("head","left_wrist","right_wrist"):
        for k in range(1,360):
            mono=k*33333333
            receive.append(dict(kind="receive",stream=c+".camera",mono_ns=mono,done_ns=mono+10000,stamp_ns=100000000000+mono))
    for k in range(1,240):
        if callback_stall and 20<=k<24:
            continue
        mono=k*50000000
        receive.append(dict(kind="receive",stream="executor.heartbeat",mono_ns=mono,done_ns=mono,stamp_ns=0))
    if sdk_stall:
        sdk += [dict(kind="sdk",side="left",op="drag.getEndTorque",phase="begin",id=1,start_ns=991000000,mono_ns=991000000),
                dict(kind="sdk",side="left",op="drag.getEndTorque",phase="end",id=1,start_ns=991000000,mono_ns=1199000000)]
    # This expected lock wait must not be blamed for a pause in the dedicated drag stream.
    sdk.append(dict(kind="sdk",side="right",op="sdk_lock.wait",phase="end",id=2,start_ns=100000000,mono_ns=11000000000))
    for name,rows in (("sdk_1.jsonl",sdk),("receive_2.jsonl",receive)):
        rows.append(dict(kind="trace_end",dropped=0))
        (tmp_path/name).write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    return analyzer.analyze(tmp_path)


def test_sdk_query_stall_is_distinguished_from_ros_delay(tmp_path):
    r=fixture(tmp_path,sdk_stall=True)
    gaps=r["teaching_publication_gaps_over_80ms"]
    assert len(gaps)==1
    assert gaps[0]["slow_operations"][0]["op"]=="drag.getEndTorque"
    assert gaps[0]["slow_operations"][0]["duration_ms"]==208
    assert r["status"]=="needs_review"


def test_continuous_publication_with_delayed_executor(tmp_path):
    r=fixture(tmp_path,callback_stall=True)
    assert not r["teaching_publication_gaps_over_80ms"]
    assert any(g["executor_heartbeat_delayed"] for g in r["receive_gaps"])
    assert r["delivery"]["left.drag"]["source_to_callback_latency"]["max_ms"]>190


def test_expected_sdk_lock_wait_is_not_a_drag_stream_failure(tmp_path):
    r=fixture(tmp_path)
    assert not r["slow_calls_over_20ms"]
    assert r["status"]=="timing_checks_passed"


def test_no_live_teaching_evidence_cannot_pass(tmp_path):
    assert analyzer.analyze(tmp_path)["status"]=="needs_live_teaching_trace"


def test_receiver_metadata_is_drained_on_exit(tmp_path):
    t=FeedbackTrace(tmp_path)
    for i in range(250):
        t.emit("left.drag",1000000000+i,accepted=bool(i%2))
    t.close()
    rows=[json.loads(s) for s in t.path.read_text().splitlines()]
    assert len(rows)==251 and rows[-1]==dict(kind="trace_end",dropped=0)
    assert not t.error
    assert all(r["done_ns"]>=r["mono_ns"] for r in rows[:-1])
