#!/usr/bin/env python3
"""Offline analysis of an archived live run; no ROS or hardware operations."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_slow_motion import replay_limiter


def describe(values):
    x = np.asarray(values)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    out = parser.parse_args().directory
    events = [json.loads(line) for line in (out / "run_events.jsonl").read_text().splitlines()]
    chunks = [e for e in events if e["event"] == "chunk"]
    all_traces = [e["trace"] for e in events if e["event"] == "servo_trace"
                  and e["trace"].get("hardware_commands_sent")]
    traces = [e for e in all_traces if e.get("active")]
    assert np.all(np.diff([e["trace_sequence"] for e in all_traces]) == 1), "Missing trace samples"
    ns = np.array([e["selection_time_ns"] for e in traces], dtype=np.int64)
    t = (ns - ns[0]) / 1e9
    dt = np.array([e["dt_s"] for e in traces])
    command = np.array([e["command"] for e in traces])
    measured = np.array([e["left_measured"] + e["right_measured"] for e in traces])
    fused = np.array([e["fused_target"][:14] for e in traces])
    limited = np.array([e["limited_target"] for e in traces])
    velocity = np.array([e["velocity"] for e in traces])
    anchor = np.array(all_traces[0]["command"])
    acceleration = np.diff(velocity, axis=0) / dt[1:, None]
    native_replay = replay_limiter(limited, dt, anchor)
    limiter_error = float(np.max(abs(native_replay - command)))
    assert limiter_error < 1e-10, limiter_error

    # Reconstruct native interpolation/weights from the exact plans that survived startup.
    eligible = [e for e in chunks if e["sequence"] >= traces[0]["chunk_sequence"]]
    seq = np.array([e["sequence"] for e in eligible])
    obs_ns = np.array([e["observation_time_ns"] for e in eligible], dtype=np.int64)
    plans = np.array([e["actions"] for e in eligible])
    reconstructed, newest, recent, disagreement = [], [], [], []
    for trace, q in zip(traces, measured):
        index = (np.int64(trace["requested_execution_time_ns"]) - obs_ns) / 50_000_000
        keep = (seq <= trace["chunk_sequence"]) & (index >= 0) & (index <= 49)
        h = index[keep]
        lo = np.floor(h).astype(int)
        hi = np.minimum(lo + 1, 49)
        fraction = (h - lo)[:, None]
        p = plans[keep]
        points = p[np.arange(len(p)), lo, :14] * (1 - fraction) + p[np.arange(len(p)), hi, :14] * fraction
        weights = np.exp(-.01 * np.arange(len(p)))
        reconstructed.append(np.average(points, axis=0, weights=weights))
        newest.append(points[-1])
        short = h <= h[-1] + 5
        recent.append(np.average(points[short], axis=0, weights=weights[short]))
        disagreement.append((points[-1] - q) * (points[0] - q) < 0)
    reconstructed, newest, recent = map(np.array, (reconstructed, newest, recent))
    fusion_error = float(np.max(abs(reconstructed - fused)))
    assert fusion_error < 1e-10, fusion_error

    variants = {}
    for name, accel, response in [("recorded", 1., .1), ("acceleration_2", 2., .1),
                                  ("response_0_05", 1., .05)]:
        value = replay_limiter(limited, dt, anchor, accel, response)
        variants[name] = {"max_command_difference_rad": float(np.max(abs(value - command))),
                          "final_command_difference_rad": float(np.max(abs(value[-1] - command[-1])))}

    # Fit measured displacement against sent displacement about the actual session anchor.
    # A latency-only explanation must preserve gain=1; allow a lag and estimate gain separately.
    fit_mask = t >= 1
    fits = []
    for lag in np.arange(0, .201, .001):
        delayed = np.stack([np.interp(t - lag, t, command[:, j]) for j in range(14)], axis=1)
        x, y = delayed[fit_mask] - anchor, measured[fit_mask] - anchor
        gain = np.sum(x * y, axis=0) / np.sum(x * x, axis=0)
        rmse = float(np.sqrt(np.mean((y - gain * x) ** 2)))
        fits.append((rmse, float(lag), gain, delayed))
    rmse, lag, gain, delayed = min(fits, key=lambda v: v[0])
    per_joint_r2 = 1 - np.sum((measured[fit_mask] - anchor - gain * (delayed[fit_mask] - anchor)) ** 2, axis=0) / np.sum((measured[fit_mask] - measured[fit_mask].mean(axis=0)) ** 2, axis=0)

    state = np.array([e["observation_state"] for e in chunks])
    state_ns = np.array([e["source_stamps_ns"]["left_arm"] for e in chunks], dtype=np.int64)
    state_time = (state_ns - ns[0]) / 1e9
    aligned = np.stack([np.interp(state_time, t, measured[:, j]) for j in range(14)], axis=1)
    state_mask = (state_ns >= ns[0]) & (state_ns <= ns[-1])
    # Right arm source timestamp differs by less than a microsecond in this run.
    horizons = np.array([e["actions"] for e in chunks])
    chunk_ns = np.array([e["observation_time_ns"] for e in chunks], dtype=np.int64)
    late_chunks = chunk_ns > ns[0] + 30_000_000_000
    windows = {}
    for start, end in [(0, 10), (10, 30), (30, 60)]:
        m = (t >= start) & (t < end)
        windows[f"{start}_{end}_s"] = {
            "measured_range_deg": np.rad2deg(np.ptp(measured[m], axis=0)).tolist(),
            "command_range_deg": np.rad2deg(np.ptp(command[m], axis=0)).tolist(),
            "fused_range_deg": np.rad2deg(np.ptp(fused[m], axis=0)).tolist(),
            "newest_prediction_range_deg": np.rad2deg(np.ptp(newest[m], axis=0)).tolist(),
            "command_abs_velocity_rad_s": describe(abs(velocity[m])),
        }
    report = {
        "run_id": traces[0]["run_id"], "mode": "offline_analysis_no_hardware_operations",
        "first_event_local": datetime.fromtimestamp(events[0]["time_ns"] / 1e9).astimezone().isoformat(),
        "active_duration_seconds": float(t[-1]), "chunk_count": len(chunks), "active_trace_count": len(traces),
        "trace_sequence_contiguous": True,
        "timing": {
            "chunk_rate_hz": (len(chunks) - 1) * 1e9 / (chunks[-1]["send_time_ns"] - chunks[0]["send_time_ns"]),
            "control_rate_hz": (len(traces) - 1) / t[-1],
            "inference_ms": describe([e["inference_ms"] for e in chunks]),
            "network_to_jetson_ms": describe([(e["jetson_receive_ns"] - e["send_time_ns"]) / 1e6 for e in chunks]),
            "observation_to_selection_ms": describe([e["observation_age_ms"] for e in traces]),
            "source_skew_ms": describe([e["observation_source_skew_ms"] for e in chunks]),
            "control_dt_ms": describe(dt * 1000),
            "max_chunk_send_gap_ms": float(max(np.diff([e["send_time_ns"] for e in chunks])) / 1e6),
            "clock_caveat": "Cross-host timing includes clock offset; local control intervals do not.",
        },
        "feedback": {
            arm: {"all_fresh": all(e[arm + "_feedback"]["fresh"] for e in traces),
                  "frames_per_drain": describe([e[arm + "_feedback"]["frames_drained"] for e in traces]),
                  "host_read_age_ms": describe([e[arm + "_feedback"]["host_read_age_ms"] for e in traces]),
                  "age_caveat": "Age since latest SDK read, not a controller capture timestamp."}
            for arm in ("left", "right")
        },
        "model_input_vs_interpolated_native_feedback_rad": describe(abs(state[state_mask, :14] - aligned[state_mask])),
        "limiter": {
            "speed_bound_rad_s": .5, "acceleration_bound_rad_s2": 1.,
            "max_command_speed_rad_s": float(abs(velocity).max()),
            "speed_bound_cycle_fraction": float(np.mean(np.any(abs(velocity) >= .499, axis=1))),
            "acceleration_bound_cycle_fraction": float(np.mean(np.any(abs(acceleration) >= .999, axis=1))),
            "relative_clip_cycle_fraction": float(np.mean(np.any(abs(fused - limited) > 1e-8, axis=1))),
            "fused_minus_command_abs_rad": describe(abs(fused - command)),
            "reconstruction_error_rad": limiter_error,
            "fixed_target_replay": variants,
            "replay_caveat": "Recorded targets held fixed; not a closed-loop prediction after changing the controller.",
        },
        "physical_tracking": {
            "command_displacement_deg": np.rad2deg(command[-1] - anchor).tolist(),
            "measured_displacement_deg": np.rad2deg(measured[-1] - anchor).tolist(),
            "measured_range_deg": np.rad2deg(np.ptp(measured, axis=0)).tolist(),
            "command_minus_measured_abs_rad": describe(abs(command - measured)),
            "fitted_gain_per_joint": gain.tolist(), "fitted_common_delay_seconds": lag,
            "gain_fit_rmse_rad": rmse, "gain_fit_r2_per_joint": per_joint_r2.tolist(),
            "gain_one_at_same_delay_rmse_rad": float(np.sqrt(np.mean((measured[fit_mask] - delayed[fit_mask]) ** 2))),
            "interpretation": "Measured displacement is approximately half of the sent displacement about session start; persistent gain loss is not explained by latency alone.",
            "cause_status": "Controller/ServoJ path suspected. Exact internal cause not established by this log.",
            "current_ServoJ_parameters": {"period_s": .01, "lookahead_s": .02, "gain": 0.},
            "SDK_example": "servoj_demo.cpp uses equal period and lookahead. This differs from deployment, but causality needs an isolated tracking test.",
        },
        "fusion": {
            "contributors": describe([e["contributors"] for e in traces]),
            "reconstruction_error_rad": fusion_error,
            "newest_vs_fused_abs_rad": describe(abs(newest - fused)),
            "oldest_newest_direction_disagreement_fraction_after_10s": np.mean(np.array(disagreement)[t > 10], axis=0).tolist(),
            "windows": windows,
            "late_chunk_median_delta_from_observation_deg": {
                str(h): np.rad2deg(np.median(horizons[late_chunks, h, :14] - state[late_chunks, :14], axis=0)).tolist()
                for h in (0, 1, 10, 25, 49)
            },
            "caveat": "Model/fusion stagnation was observed with faulty physical tracking; it is not proof of an independent model failure or proof that disabling fusion solves the task.",
        },
        "round_transitions": [e for e in events if e["event"] in ("phase", "stop_retry", "round_end")],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (out / "run_events.jsonl", out / "ros_servoj_subscriber.cpp", out / "timed_policy.hpp", out / "config.yaml")},
    }
    for split in ("train", "val"):
        path = out / f"{split}_motion_reference.json"
        if path.exists():
            rows = json.loads(path.read_text())
            report[f"dataset_{split}_reference"] = {
                "episode_count": len(rows), "median_duration_seconds": float(np.median([r["duration"] for r in rows])),
                "median_per_joint_range_deg": np.median([r["range_deg"] for r in rows], axis=0).tolist(),
                "caveat": "Demonstration range reference, not a task-success score or time-aligned comparison.",
            }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(out / "analysis.npz", time=t, command=command, measured=measured,
                        fused=fused, limited=limited, newest=newest, recent=recent, velocity=velocity,
                        anchor=anchor, delayed_command=delayed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for j, ax, label in [(2, axs[0, 0], "Left J3"), (10, axs[0, 1], "Right J4")]:
        for value, name, style in [(fused, "Fused model target", "-"), (command, "Sent target", "-"),
                                   (measured, "Measured joints", "-"), ((delayed + anchor) / 2, "50% sent displacement (delay fitted)", "--")]:
            ax.plot(t, np.rad2deg(value[:, j] - anchor[j]), style, lw=1.1, label=name)
        ax.set(title=label + ": persistent half-amplitude response", xlabel="Time from active control (s)", ylabel="Displacement from start (degrees)")
        ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.scatter(np.rad2deg(delayed[fit_mask, 2] - anchor[2]), np.rad2deg(measured[fit_mask, 2] - anchor[2]), s=3, alpha=.3)
    x = np.linspace(np.rad2deg((delayed[:, 2] - anchor[2]).min()), np.rad2deg((delayed[:, 2] - anchor[2]).max()), 100)
    ax.plot(x, x, "k--", label="Expected gain 1")
    ax.plot(x, x * .5, "r-", label="Observed gain ~0.5")
    ax.set(title="Left J3: following gain", xlabel="Sent displacement (degrees)", ylabel="Measured displacement (degrees)")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.plot(t, np.max(abs(velocity), axis=1), label="Largest commanded joint speed")
    ax.axhline(.5, color="r", ls="--", label="Configured speed bound")
    ax.set(title="Speed bound never reached", xlabel="Time from active control (s)", ylabel="Speed (rad/s)")
    ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.25)
    fig.suptitle("2026-11-12 16:43 live run: fresh feedback, ~50% motion gain\nExact controller cause requires isolated tracking validation")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("timing", "limiter", "physical_tracking")}, indent=2))


if __name__ == "__main__":
    main()
