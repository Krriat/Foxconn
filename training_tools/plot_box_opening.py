#!/usr/bin/python3
"""Plot saved offline audit arrays; requires only NumPy and Matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def stitch(data, model, episode):
    rows = np.flatnonzero(data["episode_index"] == episode)
    start, end = int(rows[0]), int(rows[-1] + 1)
    lookup = {int(i): k for k, i in enumerate(data["indices"])}
    predicted = np.concatenate([data[model][lookup[i], :min(10, end-i)] for i in range(start, end, 10)])
    return np.arange(end-start) / 20, data["action"][start:end], predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.output / "report.json").read_text())
    data = np.load(args.output / "val_predictions.npz")
    for episode in np.unique(data["episode_index"]):
        fig, axs = plt.subplots(5, 3, figsize=(17, 13))
        for model in ("act", "interact"):
            t, truth, predicted = stitch(data, model, episode)
            for j in range(15):
                axs.flat[j].plot(t, predicted[:, j], label=model, alpha=.8, lw=.9)
        for j in range(15):
            ax = axs.flat[j]
            ax.plot(t, truth[:, j], "k--", label="demonstration", lw=1)
            ax.set_title(f"{'Left' if j<7 else 'Right'} joint {j%7+1} (rad)" if j<14 else "Left suction output")
            ax.grid(alpha=.2)
        axs.flat[14].axhline(.6, color="r", alpha=.3)
        axs.flat[14].legend()
        fig.suptitle(f"Validation segment {episode}: true observation replay, replan every 10 frames (0.5 s)")
        fig.supxlabel("Demonstration time (s)")
        fig.tight_layout()
        fig.savefig(args.output / f"val_episode_{episode:02d}_trajectories.png", dpi=130)
        plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    t, truth, pred = stitch(data, "act", 0)
    ax = axs[0, 0]
    ax.plot(t, truth[:, 1], "k--", label="demonstration")
    ax.plot(t, pred[:, 1], label="ACT, true observations every 0.5 s", lw=1)
    ax.set(title="Offline replay follows the broad motion (left joint 2)", xlabel="Demonstration time (s)", ylabel="Joint angle (rad)")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    steps = [0, 1, 5, 9, 19, 49]
    for model in ("act", "interact"):
        m = report["splits"]["val"]["models"][model]["horizons"]
        ax.plot(np.array(steps)/20, [m[str(h)]["joint_mae_rad"] for h in steps], "o-", label=model)
    ax.plot(np.array(steps)/20, [m[str(h)]["hold_current_joint_mae_rad"] for h in steps], "k--", label="hold current joint angles")
    ax.axvline(.45, alpha=.4, color="r", label="last deployed target (index 9)")
    ax.set(title="Validation forecast error vs. holding still", xlabel="Prediction horizon (s)", ylabel="Mean absolute joint error (rad)")
    ax.legend(fontsize=8)
    events_path = args.output / "jetson_executor_events.jsonl"
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        homes = [r for r in events if r["event"] == "home_verified"]
        if homes:
            home = homes[-1]
            rows = [r for r in events if r["run_id"] == home["run_id"] and r["event"] == "action"]
            raw = np.array([r["raw"] for r in rows])
            live_t = (np.array([r["time_ns"] for r in rows]) - rows[0]["time_ns"]) / 1e9
            ax = axs[1, 0]
            ax.plot(live_t, raw[:, 2], lw=.9, label="published target, left joint 3")
            ax.axhline(home["measured"][2], ls="--", color="k", label="verified initial angle")
            ax.set(title="Recorded real run: targets stay near initial pose", xlabel="Time since first action (s)", ylabel="Target angle (rad)")
            ax.legend(fontsize=8)
            ax = axs[1, 1]
            early = live_t <= 4
            ax.plot(live_t[early], raw[early, 1], lw=1, marker=".", ms=3, label="published target, left joint 2")
            step_indices = np.array([r["prediction_step_index"] for r in rows])
            boundaries = np.flatnonzero(np.diff(step_indices) <= 0) + 1
            for k in boundaries[live_t[boundaries] <= 4]:
                ax.axvline(live_t[k], color="r", alpha=.2)
            ax.set(title="Recorded target jumps at new chunks (red lines)", xlabel="Time since first action (s)", ylabel="Target angle (rad)")
            ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Current ACT audit: offline forecast is not closed-loop task success\nBottom panels show logged targets, not measured executed trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output / "summary.png", dpi=150)
    plt.close(fig)
    print("Saved summary and seven validation trajectory figures")


if __name__ == "__main__":
    main()
