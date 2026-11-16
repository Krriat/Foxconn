#!/usr/bin/env python3
"""Compare two recorded runs offline; does not connect to ROS or the robot."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_run(path):
    events = [json.loads(line) for line in path.open()]
    chunks = [e for e in events if e['event'] == 'chunk']
    traces = [e['trace'] for e in events if e['event'] == 'servo_trace'
              and e['trace'].get('active') and e['trace'].get('hardware_commands_sent')]
    start = traces[0]['selection_time_ns']
    t = (np.array([e['selection_time_ns'] for e in traces], dtype=np.int64) - start) / 1e9
    ct = (np.array([e['observation_time_ns'] for e in chunks], dtype=np.int64) - start) / 1e9
    q = np.array([e['left_measured'] + e['right_measured'] for e in traces])
    command = np.array([e['command'] for e in traces])
    fused = np.array([e['fused_target'] for e in traces])
    states = np.array([e['observation_state'] for e in chunks])
    plans = np.array([e['actions'] for e in chunks])
    home = np.array(next(e['measured'] for e in events if e['event'] == 'home_verified'))
    on = states[:, 14] >= .5
    switches = [{'seconds': float(ct[i]), 'on': bool(on[i])}
                for i in np.flatnonzero(np.diff(on.astype(int))) + 1]
    newest, oldest, reconstructed = [], [], []
    eligible = np.array([e['sequence'] >= traces[0]['chunk_sequence'] for e in chunks])
    sequences = np.array([e['sequence'] for e in chunks])
    observations = np.array([e['observation_time_ns'] for e in chunks], dtype=np.int64)
    for trace in traces:
        h = (np.int64(trace['requested_execution_time_ns']) - observations) / 50_000_000
        valid = np.flatnonzero(eligible & (sequences <= trace['chunk_sequence']) & (h >= 0) & (h <= 49))
        lo = np.floor(h[valid]).astype(int)
        hi = np.minimum(lo + 1, 49)
        fraction = (h[valid] - lo)[:, None]
        points = plans[valid, lo, :14] * (1 - fraction) + plans[valid, hi, :14] * fraction
        newest.append(points[-1])
        oldest.append(points[0])
        reconstructed.append(np.average(points, axis=0, weights=np.exp(-.01 * np.arange(len(valid)))))
    newest, oldest = np.array(newest), np.array(oldest)
    reconstruction_error = float(np.max(abs(np.array(reconstructed) - fused[:, :14])))
    assert reconstruction_error < 1e-10, reconstruction_error
    late = (t >= 20) & (t <= 40)
    selected = {}
    for j in (3, 8, 11):
        selected[f'{"left" if j < 7 else "right"}_J{j % 7 + 1}'] = {
            'newest_offset_median_deg': float(np.rad2deg(np.median(newest[late, j] - q[late, j]))),
            'oldest_offset_median_deg': float(np.rad2deg(np.median(oldest[late, j] - q[late, j]))),
            'fused_offset_median_deg': float(np.rad2deg(np.median(fused[late, j] - q[late, j]))),
            'opposite_sign_fraction': float(np.mean((newest[late, j] - q[late, j]) * (oldest[late, j] - q[late, j]) < 0)),
        }
    samples = {str(sec): np.rad2deg(q[np.argmin(abs(t - sec))]).tolist() for sec in (0, 5, 10, 20, 40)}
    report = {
        'source': str(path), 'run_id': events[0]['run_id'],
        'checkpoint_sha256': sorted({c['checkpoint_sha256'] for c in chunks}),
        'publisher_id': sorted({c['publisher_id'] for c in chunks}),
        'recorded_active_seconds': float(t[-1]), 'home_measured_rad': home.tolist(),
        'home_max_error_deg': float(np.rad2deg(np.max(abs(home - events[0]['home'])))),
        'suction_DO_switches': switches,
        'tracking_p95_deg': float(np.percentile(np.rad2deg(abs(command - q)), 95)),
        'prediction_hz': float((len(ct)-1) / (ct[-1]-ct[0])),
        'servo_hz': float((len(t)-1) / (t[-1]-t[0])),
        'observation_age_p95_ms': float(np.percentile([e['observation_age_ms'] for e in traces], 95)),
        'fusion_contributors_median': float(np.median([e['contributors'] for e in traces])),
        'fusion_0_01_reconstruction_error_rad': reconstruction_error,
        'measured_joints_deg_by_second': samples,
        'fusion_20_to_40s': selected,
    }
    return report, dict(t=t, ct=ct, q=q, fused=fused, states=states, plans=plans, home=home,
                        newest=newest, oldest=oldest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('success', type=Path)
    parser.add_argument('failure', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sr, s = read_run(args.success)
    fr, f = read_run(args.failure)
    initial_difference = np.rad2deg(f['plans'][0, :, :14] - s['plans'][0, :, :14])
    largest = np.unravel_index(np.argmax(abs(initial_difference)), initial_difference.shape)
    comparison = {
        'home_between_runs_max_difference_deg': float(np.rad2deg(np.max(abs(s['home']-f['home'])))),
        'first_observation_joints_max_difference_deg': float(np.rad2deg(np.max(abs(s['states'][0,:14]-f['states'][0,:14])))),
        'first_plan_max_joint_difference_deg': float(np.max(abs(initial_difference))),
        'first_plan_largest_difference_step_and_joint_zero_based': [int(x) for x in largest],
        'first_plan_mean_abs_joint_difference_deg': float(np.mean(abs(initial_difference))),
        'first_plan_left_suction_start_end': {
            'success': s['plans'][0, [0, -1], 14].tolist(),
            'failure': f['plans'][0, [0, -1], 14].tolist(),
        },
        'measured_between_runs_max_difference_deg_by_second': {
            str(sec): float(np.max(abs(np.array(fr['measured_joints_deg_by_second'][str(sec)]) -
                                      sr['measured_joints_deg_by_second'][str(sec)])))
            for sec in (0, 5, 10, 20, 40)
        },
    }
    report = dict(success=sr, failure=fr, comparison=comparison,
                  caveats=['No historical input images or vacuum pressure: exact visual/contact cause is unverified.',
                           'Successful run is a first-60-seconds extract; success outcome is from user report and later image.',
                           'Cross-host observation age includes clock offset.',
                           'Recorded-target fusion comparison is not a closed-loop intervention test.'])
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for data, name in ((s, '17:11 success'), (f, '18:09 failure')):
        axes[0].plot(data['t'], np.rad2deg(data['q'][:, 3]), label=name)
        axes[1].plot(data['t'], np.rad2deg(data['q'][:, 8]), label=name)
        axes[2].step(data['ct'], data['states'][:, 14], where='post', label=name)
    axes[0].set_ylabel('Left J4 (deg)')
    axes[1].set_ylabel('Right J2 (deg)')
    axes[2].set_ylabel('Left suction DO')
    axes[2].set_xlabel('Time from active ServoJ (s)')
    axes[2].set_xlim(0, 41)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle('Same home, same ACT weights: recorded successful and failed runs')
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png', dpi=140)
    print(json.dumps(dict(comparison=comparison,
                         success={k:sr[k] for k in ('suction_DO_switches','tracking_p95_deg','fusion_20_to_40s')},
                         failure={k:fr[k] for k in ('suction_DO_switches','tracking_p95_deg','fusion_20_to_40s')}), indent=2))


if __name__ == '__main__':
    main()
