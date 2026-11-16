#!/usr/bin/env python3
"""Compare archived 8-second ServoJ tracking probes. No hardware calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
root = parser.parse_args().directory
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
results = {}
for ax, side in zip(axs, ("left", "right")):
    for lookahead_ms in (20, 10):
        path = root / f"{side}_{lookahead_ms}ms.json"
        data = json.loads(path.read_text())
        assert data["passed"] and data["completed_trajectory"] and not data["cleanup_errors"]
        assert data["final_operation_state"] == 0
        assert data["initial_power_state"] == data["final_power_state"]
        samples = data["samples"]
        ticks = np.array([s["tick"] for s in samples])
        assert np.array_equal(ticks, np.arange(801))
        time = np.array([s["time_s"] for s in samples])
        command = np.array([s["command"] for s in samples]) - data["anchor"]
        measured = np.array([s["measured"] for s in samples]) - data["anchor"]
        plateau = (ticks >= 400) & (ticks <= 490)
        delta = float(np.median(measured[plateau, 5]))
        gain = delta / data["amplitude_rad"]
        expected = .5 if lookahead_ms == 20 else 1.
        assert abs(gain - expected) < .01, (side, lookahead_ms, gain)
        return_error = float(np.max(abs(measured[-1])))
        assert return_error < .0001
        results[path.stem] = {
            "period_ms": 10, "lookahead_ms": lookahead_ms, "samples": len(samples),
            "command_displacement_rad": data["amplitude_rad"],
            "measured_plateau_displacement_rad": delta, "displacement_gain": gain,
            "max_other_joint_displacement_rad": float(np.max(abs(np.delete(measured, 5, axis=1)))),
            "return_error_rad": return_error,
            "dt_ms_p95": float(np.percentile([s["dt_s"] * 1000 for s in samples[1:]], 95)),
            "dt_ms_max": float(max(s["dt_s"] * 1000 for s in samples[1:])),
            "initial_power_state": data["initial_power_state"], "final_power_state": data["final_power_state"],
            "final_operation_state": data["final_operation_state"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if lookahead_ms == 20:
            ax.plot(time, np.rad2deg(command[:, 5]), "k--", label="Command: 0.004 rad")
        ax.plot(time, np.rad2deg(measured[:, 5]), label=f"Lookahead {lookahead_ms} ms: {gain*100:.2f}%")
    ax.set(title=f"{side.title()} arm J6", xlabel="Time (s)", ylabel="Displacement (degrees)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
fig.suptitle("100 Hz ServoJ: equal 10 ms period and lookahead restores full displacement")
fig.tight_layout()
fig.savefig(root / "tracking_comparison.png", dpi=160)
plt.close(fig)
report = {"tests_passed": True, "results": results,
          "conclusion": "On both tested controllers, 10 ms period / 20 ms lookahead / gain 0 yields half displacement. Equal 10 ms period and lookahead yields unity displacement.",
          "scope": "Controlled J6 excursion of +0.004 rad and return; no model or suction control. Full task trajectory has not been retested.",
          "final_hardware_state": "Both probes returned to their start, stopped ServoJ, restored NRT, and restored initial power-off state."}
(root / "tracking_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
