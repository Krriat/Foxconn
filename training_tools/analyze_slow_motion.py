#!/usr/bin/python3
"""Analyze saved timed-policy traces. No ROS, SSH, or hardware operations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def describe(values):
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def replay_limiter(targets, dt, anchor, acceleration=1., response=.1):
    position = np.array(anchor, dtype=float)
    velocity = np.zeros(14)
    output = []
    for wanted, interval in zip(targets, dt):
        error = wanted-position
        desired_speed = np.copysign(np.minimum.reduce([
            np.full(14, .5), np.sqrt(2*acceleration*np.abs(error)), np.abs(error)/response]), error)
        velocity = np.clip(desired_speed, velocity-acceleration*interval, velocity+acceleration*interval)
        position = position+velocity*interval
        output.append(position.copy())
    return np.stack(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    events = [json.loads(s) for s in (out/"run_events.jsonl").read_text().splitlines()]
    chunks = [s for s in events if s["event"] == "chunk"]
    all_traces = [s["trace"] for s in events if s["event"] == "servo_trace" and s["trace"].get("hardware_commands_sent")]
    traces = [s for s in all_traces if s.get("active")]
    times = np.array([s["selection_time_ns"] for s in traces], dtype=np.int64)
    t = (times-times[0])/1e9
    command = np.array([s["command"] for s in traces])
    fused = np.array([s["fused_target"][:14] for s in traces])
    limited = np.array([s["limited_target"] for s in traces])
    velocity = np.array([s["velocity"] for s in traces])
    measured = np.array([s["left_measured"]+s["right_measured"] for s in traces])
    dt = np.array([s["dt_s"] for s in traces])
    acceleration = np.diff(velocity, axis=0)/dt[1:, None]
    clip = np.abs(fused-limited) > 1e-8
    original = replay_limiter(limited, dt, all_traces[0]["command"])
    replay_difference = float(np.max(np.abs(command-original)))
    assert replay_difference < 1e-10, replay_difference

    # Reconstruct native fusion from the recorded plans, using its actual
    # newest sequence and execution time. Startup plans were discarded by C++.
    candidates = [s for s in chunks if s["sequence"] >= traces[0]["chunk_sequence"]]
    sequence = np.array([s["sequence"] for s in candidates])
    observation_ns = np.array([s["observation_time_ns"] for s in candidates], dtype=np.int64)
    actions = np.array([s["actions"] for s in candidates])
    reconstructed, newest, short_window = [], [], []
    for trace in traces:
        index = (np.int64(trace["requested_execution_time_ns"])-observation_ns)/50_000_000
        valid = np.flatnonzero((sequence <= trace["chunk_sequence"]) & (index >= 0) & (index <= 49))
        h = index[valid]
        low = np.floor(h).astype(int)
        high = np.minimum(low+1, 49)
        fraction = h-low
        points = actions[valid, low, :14]*(1-fraction[:, None])+actions[valid, high, :14]*fraction[:, None]
        weights = np.exp(-.01*np.arange(len(valid)))
        reconstructed.append(np.average(points, axis=0, weights=weights))
        newest.append(points[-1])
        recent = h <= h[-1]+5  # Same execution time, observations within 0.25 s of newest.
        short_window.append(np.average(points[recent], axis=0, weights=weights[recent]))
    reconstructed, newest, short_window = map(np.array, (reconstructed, newest, short_window))
    fusion_difference = float(np.max(np.abs(fused-reconstructed)))
    assert fusion_difference < 1e-10, fusion_difference
    variants = {}
    for name, accel, response in (("recorded", 1., .1), ("acceleration_2", 2., .1),
                                   ("response_0_05", 1., .05), ("no_acceleration_bound_diagnostic", 1e6, .1)):
        value = replay_limiter(limited, dt, all_traces[0]["command"], accel, response)
        variants[name] = {"max_difference_from_recorded_command_rad": float(np.max(np.abs(value-command))),
                          "final_difference_from_recorded_command_rad": float(np.max(np.abs(value[-1]-command[-1]))),
                          "per_joint_range_rad": np.ptp(value, axis=0).tolist()}
    report = {
        "run_id": traces[0]["run_id"], "mode": "offline_recorded_trace_analysis_no_hardware",
        "duration_s": float(t[-1]), "chunk_count": len(chunks), "active_control_samples": len(traces),
        "chunk_rate_hz": (len(chunks)-1)*1e9/(chunks[-1]["time_ns"]-chunks[0]["time_ns"]),
        "control_interval_ms": describe(dt*1000),
        "timing": {
            "inference_ms_per_chunk": describe([s["inference_ms"] for s in chunks]),
            "send_to_jetson_receive_ms_per_chunk": describe([(s["jetson_receive_ns"]-s["send_time_ns"])/1e6 for s in chunks]),
            "observation_to_selection_ms_per_control_cycle": describe([s["observation_age_ms"] for s in traces]),
            "left_sdk_send_interval_ms": describe(np.diff([s["left_sdk_send_start_ns"] for s in traces])/1e6),
            "right_sdk_send_interval_ms": describe(np.diff([s["right_sdk_send_start_ns"] for s in traces])/1e6)},
        "limits": {
            "configured_speed_rad_s": .5, "configured_acceleration_rad_s2": 1., "response_seconds": .1,
            "maximum_command_speed_rad_s": float(np.max(np.abs(velocity))),
            "speed_bound_active_cycle_fraction": float(np.mean(np.max(np.abs(velocity), axis=1) >= .499)),
            "acceleration_bound_active_cycle_fraction": float(np.mean(np.max(np.abs(acceleration), axis=1) >= .99)),
            "relative_position_clip_cycle_fraction": float(np.mean(np.any(clip, axis=1))),
            "relative_position_clip_fraction_per_joint": np.mean(clip, axis=0).tolist(),
            "smoother_position_lag_rad": describe(np.abs(limited-command)),
            "reported_command_minus_RT_feedback_rad": describe(np.abs(command-measured)),
            "feedback_caveat": "RT feedback is read from a FIFO at a lower frequency than it is produced. This difference must NOT be interpreted as fresh physical tracking error."},
        "fusion": {"contributors": describe([s["contributors"] for s in traces]),
                   "native_reconstruction_max_difference_rad": fusion_difference,
                   "fused_range_per_joint_rad": np.ptp(fused, axis=0).tolist(),
                   "newest_only_same_time_range_per_joint_rad": np.ptp(newest, axis=0).tolist(),
                   "last_0_25_s_same_time_range_per_joint_rad": np.ptp(short_window, axis=0).tolist(),
                   "caveat": "Alternative fusions use the same recorded observations; they are not predictions of closed-loop behavior after a controller change."},
        "smoother_replay": {"native_reconstruction_max_difference_rad": replay_difference,
                            "variants": variants,
                            "caveat": "Fixed recorded clipped targets only; changing the live controller also changes feedback and future model observations."},
        "motion": {"command_range_per_joint_rad": np.ptp(command, axis=0).tolist(),
                   "reported_RT_feedback_range_per_joint_rad": np.ptp(measured, axis=0).tolist()},
        "code_defect": {"location": "ros_servoj_subscriber.cpp: startArm()/sendArm()",
                        "producer_interval_ms": 1, "consumer_interval_ms": 10,
                        "receiver_reads_per_control_cycle": 1,
                        "SDK_evidence": "sdk_read_robot_state_example.cpp explicitly states that the receiving queue does not overwrite old frames; its example drains the queue in a loop.",
                        "impact": "Old joint feedback is used for the +/-0.035 rad relative target clamp and tracking checks. This can constrain motion around historical positions and makes logged tracking error unreliable.",
                        "missing_measurement": "No per-packet controller timestamp, queue age or queue depth is recorded, so actual feedback delay cannot be recovered exactly from this run."},
        "source_sha256": {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                          ("run_events.jsonl", "ros_servoj_subscriber.cpp", "timed_policy.hpp", "sdk_read_robot_state_example.cpp")}}
    (out/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    np.savez_compressed(out/"analysis.npz", time=t, command=command, fused=fused, limited=limited,
                        reported_RT_feedback=measured, newest=newest, short_window=short_window,
                        velocity=velocity, acceleration=acceleration)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax = axs[0, 0]
    for values, label in ((newest, "newest prediction at execution time"), (fused, "fused target"),
                          (command, "sent command"), (measured, "reported RT feedback (possibly stale)")):
        ax.plot(t, values[:, 2], label=label, lw=1)
    ax.set(title="Left J3: target, command, reported feedback", xlabel="Run time (s)", ylabel="Joint angle (rad)")
    ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.plot(t, command[:, 2]-measured[:, 2], label="command minus reported RT feedback")
    ax.plot(t, limited[:, 2]-command[:, 2], label="clipped target minus smooth command")
    ax.axhline(-.035, color="r", ls="--", label="relative target bound")
    ax.set(title="RT feedback freshness must be fixed before judging tracking", xlabel="Run time (s)", ylabel="Difference (rad)")
    ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.plot(t, np.max(np.abs(velocity), axis=1), label="max commanded joint speed")
    ax.axhline(.5, color="r", ls="--", label="speed bound")
    ax.set(title="Speed bound is never reached", xlabel="Run time (s)", ylabel="Speed (rad/s)")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    labels = ["model inference", "send to Jetson", "observation to selection"]
    values = [report["timing"][k] for k in ("inference_ms_per_chunk", "send_to_jetson_receive_ms_per_chunk", "observation_to_selection_ms_per_control_cycle")]
    x = np.arange(3)
    ax.bar(x-.15, [v["median"] for v in values], .3, label="median")
    ax.bar(x+.15, [v["p95"] for v in values], .3, label="P95")
    ax.set(xticks=x, xticklabels=labels, ylabel="Time (ms)", title="Timing: no multi-second communication backlog")
    ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Recorded live run cc335440: RT feedback FIFO defect identified\nReported RT feedback age is unknown; it is not a verified current encoder trace")
    fig.tight_layout()
    fig.savefig(out/"summary.png", dpi=145)
    plt.close(fig)
    print(json.dumps({"timing": report["timing"], "limits": report["limits"], "smoother_replay": report["smoother_replay"]}, indent=2))


if __name__ == "__main__":
    main()
