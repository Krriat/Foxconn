#!/usr/bin/env python3
"""Read saved timing traces; never imports ROS, connects to SDK, or sends commands."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def intervals(rows, field="mono_ns"):
    return [(b[field] - a[field]) / 1e6 for a, b in zip(rows, rows[1:])]


def summary(values):
    values = sorted(values)
    return dict(count=len(values), p50_ms=statistics.median(values),
                p95_ms=values[min(len(values)-1, int(.95*len(values)))], max_ms=max(values)) if values else dict(count=0)


def analyze(directory):
    directory = Path(directory)
    sdk, receive, warnings, files = [], [], [], []
    for pattern, target in (("sdk_*.jsonl", sdk), ("receive_*.jsonl", receive)):
        for path in sorted(directory.glob(pattern)):
            rows = []
            for i, line in enumerate(path.read_text().splitlines()):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    warnings.append(f"{path.name}:{i+1}: incomplete/invalid JSON")
            footers = [r for r in rows if r.get("kind") == "trace_end"]
            if not footers:
                warnings.append(path.name + ": trace not closed; process may still be running")
            if any(r.get("dropped", 0) for r in footers):
                warnings.append(path.name + ": diagnostic rows dropped; causal conclusions incomplete")
            target.extend(r for r in rows if "mono_ns" in r)
            files.append(str(path))
    sdk.sort(key=lambda r: r["mono_ns"])
    receive.sort(key=lambda r: r["mono_ns"])
    ended = [r for r in sdk if r.get("phase") == "end"]
    for r in ended:
        r["duration_ms"] = (r["mono_ns"]-r["start_ns"])/1e6
    spans = defaultdict(list)
    for r in ended:
        spans[r["side"] + ":" + r["op"]].append(r["duration_ms"])
    operation_stats = {key: summary(values) for key, values in spans.items()}
    slow_calls = [r for r in ended if r["duration_ms"] > 20 and
                  r["op"] not in ("sdk_lock.wait", "sdk_lock.hold", "drag.session")]
    begin_ids = {r["id"]: r for r in sdk if r.get("phase") == "begin"}
    end_ids = {r["id"] for r in ended}
    unfinished = [r for k,r in begin_ids.items() if k not in end_ids]
    windows = defaultdict(list)
    for side in ("left", "right"):
        start = None
        for r in sdk:
            if r.get("side") != side or r.get("op") != "drag.status":
                continue
            if r["phase"] == "teaching":
                if start is None:
                    start = r["mono_ns"]
            elif start is not None:
                windows[side].append((start, r["mono_ns"]))
                start = None
        if start is not None:
            windows[side].append((start, sdk[-1]["mono_ns"]))
            warnings.append(side + ": teaching trace has no closing status")
    def same_window(side, a, b):
        return any(start <= a <= b <= end for start,end in windows[side])
    published = defaultdict(list)
    for r in sdk:
        if r.get("phase") == "sample" and r["op"].endswith((".publish",".local_send")):
            prefix = "local_" if r["op"].endswith(".local_send") else ""
            published[r["side"] + "." + prefix + r["op"].split(".")[0]].append(r)
    publication_gaps = []
    for key, rows in published.items():
        side = key.split(".")[0]
        if not key.endswith((".drag",".local_drag")):
            continue  # Ordinary/return streams intentionally pause while drag owns the SDK.
        for a,b in zip(rows, rows[1:]):
            if not same_window(side, a["mono_ns"], b["mono_ns"]):
                continue
            gap = (b["mono_ns"]-a["mono_ns"])/1e6
            if gap <= 80:
                continue
            overlapping = [r for r in slow_calls if r["side"] == side and
                r["start_ns"] < b["mono_ns"] and r["mono_ns"] > a["mono_ns"]]
            publication_gaps.append(dict(stream=key, start_ns=a["mono_ns"], end_ns=b["mono_ns"],
                interval_ms=gap, slow_operations=[dict(op=r["op"], duration_ms=r["duration_ms"])
                                                for r in overlapping]))
    received = defaultdict(list)
    for r in receive:
        received[r["stream"]].append(r)
    receipt_stats, receipt_gaps, delivery = {}, [], {}
    source_time_errors = []
    heartbeats = received.get("executor.heartbeat", [])
    heartbeat_gaps = [(a["mono_ns"], b["mono_ns"]) for a,b in zip(heartbeats, heartbeats[1:])
                      if b["mono_ns"]-a["mono_ns"] > 80000000]
    for key, rows in received.items():
        ordered = sorted(rows, key=lambda r:r["mono_ns"])
        callback = [(r["done_ns"]-r["mono_ns"])/1e6 for r in ordered if "done_ns" in r]
        receipt_stats[key] = dict(received=len(rows), intervals=summary(intervals(ordered)),
                                 callback_duration=summary(callback))
        data = [r for r in ordered if r.get("stamp_ns",0) > 0]
        if data:
            receipt_stats[key]["source_intervals"] = summary(intervals(data,"stamp_ns"))
            bad = [dict(stream=key, previous_stamp_ns=a["stamp_ns"], stamp_ns=b["stamp_ns"])
                   for a,b in zip(data,data[1:]) if b["stamp_ns"] <= a["stamp_ns"]]
            receipt_stats[key]["non_increasing_source_timestamps"] = len(bad)
            source_time_errors.extend(bad)
        for a,b in zip(data, data[1:]):
            if key.endswith((".drag",".local_drag")) and not same_window(key.split(".")[0], a["mono_ns"], b["mono_ns"]):
                continue
            if not key.endswith((".drag",".local_drag",".camera")):
                continue
            if key.endswith(".camera") and not all(same_window(s,a["mono_ns"],b["mono_ns"])
                                                    for s in ("left","right")):
                continue
            receipt_ms = (b["mono_ns"]-a["mono_ns"])/1e6
            source_ms = (b["stamp_ns"]-a["stamp_ns"])/1e6
            limit = 55. if key.endswith(".camera") else 80.
            if max(receipt_ms,source_ms) > limit:
                receipt_gaps.append(dict(stream=key, start_ns=a["mono_ns"], end_ns=b["mono_ns"],
                    receipt_interval_ms=receipt_ms, source_interval_ms=source_ms,
                    executor_heartbeat_delayed=any(x<b["mono_ns"] and y>a["mono_ns"] for x,y in heartbeat_gaps)))
        if key in published and data:
            by_stamp = {r["stamp_ns"]:r for r in data}
            pairs = [(r,by_stamp[r["stamp_ns"]]) for r in published[key] if r["stamp_ns"] in by_stamp]
            # Ignore discovery/shutdown edges; don't treat expected mode changes as network loss.
            interior = [r for r in published[key] if data[0]["mono_ns"]+1000000000 <=
                        r["mono_ns"] <= data[-1]["mono_ns"]-1000000000]
            delivery[key] = dict(matched_messages=len(pairs),
                source_to_callback_latency=summary([(b["mono_ns"]-a["mono_ns"])/1e6 for a,b in pairs]),
                unmatched_published_interior_messages=sum(r["stamp_ns"] not in by_stamp for r in interior))
    coverage = {s: sum((b-a)/1e9 for a,b in windows[s]) for s in ("left","right")}
    active_keys = {s:s+(".local_drag" if received.get(s+".local_drag") else ".drag") for s in ("left","right")}
    tested = all(coverage[s] >= 10 and len(received.get(active_keys[s],[])) >= 100 and
                 published.get(active_keys[s]) for s in ("left","right")) and all(
                     len(received.get(c+".camera",[])) >= 100 for c in ("head","left_wrist","right_wrist"))
    status = "needs_review" if publication_gaps or receipt_gaps or warnings or source_time_errors else "timing_checks_passed"
    if not tested:
        status = "needs_live_teaching_trace"
    return dict(status=status, trace_dir=str(directory.resolve()), files=files, warnings=warnings,
        teaching_windows_ns=dict(windows), teaching_seconds=coverage, sdk_operation_durations=operation_stats,
        slow_calls_over_20ms=slow_calls, unfinished_spans=unfinished,
        teaching_publication_gaps_over_80ms=[g for g in publication_gaps if ".local_" not in g["stream"]],
        teaching_local_sample_gaps_over_80ms=[g for g in publication_gaps if ".local_" in g["stream"]],
        recording_feedback_streams=active_keys,
        receive=receipt_stats, receive_gaps=receipt_gaps, source_time_errors=source_time_errors, delivery=delivery,
        limitations=["Opt-in diagnostics add overhead; compare with normal capture after a fix.",
                     "SDK lock waits and missing ordinary-state messages during teaching are expected by the current ownership design.",
                     "Camera source gaps alone do not separate camera hardware, PC capture, network bridge and ROS; correlate PC counters/kernel logs.",
                     "No robot commands are sent by this analyzer. Timing checks do not prove task or suction success."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir",type=Path,required=True)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    result=analyze(args.trace_dir)
    output=args.report or args.trace_dir/"feedback_report.json"
    with output.open("x") as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print("诊断结果："+result["status"])
    print("示教期间 >80ms 发布间断："+str(len(result["teaching_publication_gaps_over_80ms"])))
    for r in sorted(result["slow_calls_over_20ms"],key=lambda r:r["duration_ms"],reverse=True)[:10]:
        print(f"慢调用 {r['side']} {r['op']}: {r['duration_ms']:.2f} ms")
    print("报告："+str(output))
    return 0 if result["status"]=="timing_checks_passed" else 2


if __name__=="__main__":
    raise SystemExit(main())
