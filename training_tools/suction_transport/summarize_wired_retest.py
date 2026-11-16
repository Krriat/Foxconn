"""Summarize a passive wired relay retest without connecting to hardware."""
import argparse
import collections
import datetime
import json
import re
import statistics
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(count=len(values), min_ms=values[0], mean_ms=statistics.mean(values),
                median_ms=statistics.median(values), p95_ms=percentile(.95),
                p99_ms=percentile(.99), max_ms=values[-1],
                over_300ms=sum(v > 300 for v in values))


def ros_summary(path):
    records = read_lines(path)
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["topic"]].append(r)
    result = {"logger_finished": bool(groups.get("logger_finished")), "topics": {}}
    for topic, rows in groups.items():
        if not topic.startswith("/robot/"):
            continue
        duration = (rows[-1]["monotonic_ns"] - rows[0]["monotonic_ns"]) / 1e9
        item = dict(messages=len(rows), duration_sec=duration,
                    rate_hz=(len(rows)-1)/duration if duration else None,
                    gaps=stats([r["gap_ms"] for r in rows if r.get("gap_ms") is not None]))
        if "/suction/" in topic:
            item["values"] = dict(collections.Counter(str(r["data"]) for r in rows))
        if topic == "/robot/policy/status":
            item["phases"] = dict(collections.Counter(r.get("phase") for r in rows))
            item["last"] = rows[-1]
        if topic == "/robot/hardware_state/diagnostics":
            item["first"] = rows[0]["status"]
            item["last"] = rows[-1]["status"]
            suction = [s for r in rows for s in r["status"] if s["name"] == "suction_output_feedback"]
            if suction:
                item["suction_failed_samples_delta"] = (int(suction[-1]["values"]["failures"]) -
                                                         int(suction[0]["values"]["failures"]))
                item["suction_diagnostic_levels"] = sorted({s["level"] for s in suction})
        result["topics"][topic] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    capture = read_lines(root / "tcp_final.jsonl")
    analysis = json.loads((root / "tcp_final_analysis.json").read_text())
    packets = [p for p in capture if "src" in p]
    requests = analysis["all_requests"]
    outgoing = [p for p in packets if p["dst_port"] == 20108 and p["payload_len"]]
    incoming = [p for p in packets if p["src_port"] == 20108 and p["payload_len"]]
    unexpected_requests = [p for p in outgoing if p["payload_len"] != 12 or
                           bytes.fromhex(p["payload_hex"])[2:] != bytes.fromhex("00000006010100000002")]
    unexpected_responses = [p for p in incoming if p["payload_len"] != 10 or
                            bytes.fromhex(p["payload_hex"])[2:9] != bytes.fromhex("00000004010101")]
    duplicate_responses = len(incoming) - len({(p["src_port"], p["dst_port"], p["seq"], p["payload_hex"]) for p in incoming})
    first, last = capture[0]["time_ns"], capture[-1]["time_ns"]
    timezone = datetime.timezone(datetime.timedelta(hours=8))
    local_time = lambda ns: datetime.datetime.fromtimestamp(ns/1e9, timezone).isoformat()
    timing = []
    for line in (root / "state_runtime.log").read_text().splitlines():
        match = re.search(r"\[([0-9]+\.[0-9]+)\].*m050_timing attempt=(\d+) result=(-?\d+) "
                          r"lock_ms=([\d.]+) read_ms=([\d.]+) publish_ms=([\d.]+) gap_ms=([-\d.]+)", line)
        if match and first/1e9 <= float(match[1]) <= last/1e9:
            timing.append(dict(attempt=int(match[2]), result=int(match[3]),
                               lock_ms=float(match[4]), read_ms=float(match[5]),
                               publish_ms=float(match[6]), gap_ms=float(match[7])))
    report = dict(
        saved_at=datetime.datetime.now(timezone).isoformat(),
        scope="Communication only; no arm motion, suction writes, or policy start requested.",
        configuration=dict(relay="192.168.1.7:20108", interface="enP8p1s0",
                           jetson_source="192.168.1.10", production_read_hz=10,
                           persistent_tcp=True, tcp_nodelay=True,
                           freshness_limit_ms=300, right_arm="Disconnected/off by user; not tested"),
        capture=dict(start=local_time(first), end=local_time(last), duration_sec=(last-first)/1e9,
                     final_event=capture[-1].get("event"), packets=len(packets),
                     connections_observed=analysis["connections"], requests=analysis["requests"],
                     responses=analysis["responses"], missing_replies=[r for r in requests if r.get("reply_not_seen")],
                     request_retransmissions=analysis["retransmitted_request_packets"],
                     duplicate_response_packets=duplicate_responses,
                     tcp_reset_packets=sum(bool(p["flags"] & 4) for p in packets),
                     unexpected_requests=unexpected_requests, unexpected_responses=unexpected_responses,
                     response_time=stats([r["response_ms"] for r in requests if "response_ms" in r]),
                     kernel_capture_scheduling_max_ms=analysis["capture_scheduling_max_ms"]),
        ros={host: ros_summary(root / f"{host}_receipts.jsonl") for host in ("jetson", "pc")},
        source_timing=dict(logged_rows=len(timing), error_rows=sum(r["result"] != 0 for r in timing),
                           read_over120ms=sum(r["read_ms"] > 120 for r in timing),
                           lock_over20ms=sum(r["lock_ms"] > 20 for r in timing),
                           publish_gap_over200ms=sum(r["gap_ms"] > 200 for r in timing),
                           note="Normal reads logged every tenth attempt; all reads >120ms, lock waits >20ms, publish gaps >200ms and failures logged. These rows are not an unbiased latency sample."),
        ping_summary="\n".join((root/"ping_relay.txt").read_text().splitlines()[-3:]),
        limitations=[
            "Finite idle polling test; suction output transitions and combined arm motion load were not exercised.",
            "The current deployment combines wired transport and persistent TCP; this is not a cable-only controlled comparison.",
            "Packet capture measures TCP traffic at Jetson; it cannot establish loss-free operation for all future conditions.",
            "PC receives ROS over the existing intercomputer network; wired relay transport does not remove that network's delays.",
            "A reply absent only at the capture boundary must not be interpreted as proven packet loss."
        ],
        evidence=["tcp_final.jsonl", "tcp_final_analysis.json", "jetson_receipts.jsonl",
                  "pc_receipts.jsonl", "ping_relay.txt", "deployed_config.txt", "state_runtime.log"])
    final_status = root / "final_status.json"
    if final_status.exists():
        report["final_status"] = json.loads(final_status.read_text())
        report["evidence"].append(final_status.name)
    path = root / "wired_communication_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"report": str(path), "capture": report["capture"],
                      "suction_ros": {host: {topic: value for topic, value in data["topics"].items()
                                             if "/suction/" in topic}
                                      for host, data in report["ros"].items()}}, indent=2))


if __name__ == "__main__":
    main()
