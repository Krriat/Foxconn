#!/usr/bin/env python3
"""Generate a reviewable relative/absolute plan; never imports a hardware adapter."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from common import (ROOT, load_config, write_json, object_hash, finite_positive,
                    derivative_bounds, trajectory, site_blockers)

def build_plan(c, arm, joint, frequencies=None):
    if arm not in ('left', 'right') or joint not in range(1, 8):
        raise ValueError('Select left/right and joint 1..7')
    cfg = c['trial']
    period = finite_positive(c['interface']['command_period_s'], 'command_period_s')
    ramp = finite_positive(cfg['ramp_s'], 'ramp_s')
    before = finite_positive(cfg['stationary_before_s'], 'stationary_before_s')
    after = finite_positive(cfg['stationary_after_s'], 'stationary_after_s')
    if before < 10: raise ValueError('At least 10 seconds of stationary baseline are required')
    amp = math.radians(finite_positive(cfg['amplitude_deg'], 'amplitude_deg'))
    reps = cfg['repetitions']
    if type(reps) is not int or not 1 <= reps <= 10: raise ValueError('repetitions must be 1..10')
    freqs = frequencies if frequencies is not None else cfg['frequencies_hz']
    if not freqs or len(freqs) > 20: raise ValueError('Select 1..20 frequencies')
    limits = c['limits']
    stage = [finite_positive(limits[k], k) for k in
             ('stage_velocity_rad_s', 'stage_acceleration_rad_s2', 'stage_jerk_rad_s3')]
    # This tool implements only the initial stage. Expansion requires a new reviewed implementation/plan.
    if any(x > ceiling for x, ceiling in zip(stage, (.1, .3, 1.))):
        raise ValueError('Initial-stage limits cannot exceed 0.10/0.30/1.0')
    for n, key in enumerate(('velocity_rad_s', 'acceleration_rad_s2', 'jerk_rad_s3')):
        arr = limits.get(key)
        if arr is not None:
            if len(arr) != 7: raise ValueError(key + ' requires seven values')
            for v in arr: finite_positive(v, key)
            stage[n] = min(stage[n], arr[joint-1])
    center = c['center_q_rad'][arm]
    if center is not None and (len(center) != 7 or not np.isfinite(center).all()):
        raise ValueError('center_q_rad requires seven finite controller angles')
    trials = []
    conditions = [(f, 'identification', r) for f in freqs for r in range(1, reps+1)]
    holdout = cfg.get('holdout_frequency_hz')
    if holdout is not None:
        if holdout in freqs: raise ValueError('Holdout frequency must differ from identification frequencies')
        conditions.append((holdout, 'holdout', 1))
    for f, role, repeat in conditions:
        f = finite_positive(f, 'frequency_hz')
        if 1/(f*period) < 10-1e-9: raise ValueError('Less than 10 command points/cycle')
        bounds = derivative_bounds(f, ramp)
        a = min([amp] + [v/b for v, b in zip(stage, bounds[1:])]) * (1-1e-9)
        cycles = 3 if f < 1 else 5
        duration = before + 2*ramp + cycles/f + after
        if duration > 1800: raise ValueError('Trial exceeds 30 minutes; narrow the frequency selection')
        trial = dict(trial_id=f'{arm}_J{joint}_{role}_{f:g}Hz_r{repeat}', arm=arm,
                     joint=joint, role=role, repeat=repeat, frequency_hz=f,
                     requested_amplitude_rad=amp, amplitude_rad=a, stable_cycles=cycles,
                     ramp_s=ramp, stationary_before_s=before, stationary_after_s=after,
                     duration_s=duration, stable_start_s=before+ramp,
                     stable_end_s=before+ramp+cycles/f,
                     conservative_abs_bounds=dict(zip(('q_delta_rad', 'v_rad_s', 'a_rad_s2', 'jerk_rad_s3'),
                                                      [a*b for b in bounds])))
        if center is not None:
            lo, hi, margin = (limits.get(k) for k in ('joint_lower_rad', 'joint_upper_rad', 'joint_stop_margin_rad'))
            if lo is not None and hi is not None and margin is not None:
                if len(lo) != 7 or len(hi) != 7 or not np.isfinite(lo+hi).all():
                    raise ValueError('Joint limits require seven finite values')
                finite_positive(margin, 'joint_stop_margin_rad')
                excursion = np.zeros(7); excursion[joint-1] = a
                if np.any(np.array(center)-excursion-margin <= np.array(lo)) or np.any(np.array(center)+excursion+margin >= np.array(hi)):
                    raise ValueError('Trajectory/holding joints violate angle limit or stop margin')
        trials.append(trial)
    plan = dict(schema_version=1, config=c, config_hash=object_hash(c), arm=arm, joint=joint,
                coordinates='controller_absolute' if center is not None else 'relative_preview_only',
                center_q_rad=center, command_period_s=period, mode='ServoJ_position_only_candidate',
                effective_stage_limits=stage, hardware_execution_allowed=False,
                blockers=site_blockers(c, arm), trials=trials,
                scope='joint-space analytic preview; TCP/collision/stopping trajectory not verified',
                derivative_method='analytic C3 septic envelope; continuous-time conservative product-rule bounds')
    plan['plan_hash'] = object_hash(plan)
    return plan

def validate_plan(plan):
    body = {k:v for k,v in plan.items() if k != 'plan_hash'}
    if object_hash(body) != plan.get('plan_hash'): raise ValueError('Plan hash mismatch: regenerate plan')
    if object_hash(plan['config']) != plan['config_hash']: raise ValueError('Config hash mismatch')
    fs = list(dict.fromkeys(t['frequency_hz'] for t in plan['trials'] if t['role'] == 'identification'))
    expected = build_plan(plan['config'], plan['arm'], plan['joint'], fs)
    if expected != plan: raise ValueError('Plan is not reproducible from its configuration; regenerate')
    return plan

def save_preview(plan, out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'plan.json', plan)
    with (out/'trajectory_preview.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_id', 't_s', 'segment', 'coordinate_kind'] + [f'q{j}_rad' for j in range(1,8)] + ['active_qd_rad_s', 'active_qdd_rad_s2', 'active_jerk_rad_s3'])
        for tr in plan['trials']:
            t = np.arange(0, tr['duration_s']+plan['command_period_s']/2, plan['command_period_s'])
            q, v, a, j, seg = trajectory(tr, t)
            for i in range(len(t)):
                values = list(plan['center_q_rad'] or [0.]*7)
                values[plan['joint']-1] += q[i]
                writer.writerow([tr['trial_id'], t[i], seg[i], plan['coordinates']] + values + [v[i],a[i],j[i]])
    # A standalone figure; matplotlib is imported only by offline preview/analysis.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for tr in plan['trials']:
            if tr['repeat'] != 1: continue
            t = np.linspace(0, tr['duration_s'], 4000)
            for ax, y, name in zip(axes, trajectory(tr,t)[:4], ('delta q (rad)', 'v (rad/s)', 'a (rad/s²)', 'jerk (rad/s³)')):
                ax.plot(t,y,label=f"{tr['frequency_hz']:g} Hz / {tr['role']}"); ax.set_ylabel(name); ax.grid(True)
        axes[0].legend(); axes[-1].set_xlabel('Trial time (s)')
        fig.suptitle('Joint-only preview — TCP/collision/stop review outstanding')
        fig.tight_layout(); fig.savefig(out/'trajectory_preview.png'); plt.close(fig)
    except ImportError:
        pass
    lines = ['# 轨迹计划预览', '', f"臂/轴：{plan['arm']} / J{plan['joint']}；坐标：{plan['coordinates']}", '',
             '真机运动未开放。以下界限包括进入/退出包络，不代表TCP/全臂碰撞检查通过。', '',
             '| 段 | 单边振幅 rad | 速度界 | 加速度界 | jerk界 |', '|---|---:|---:|---:|---:|']
    for tr in plan['trials']:
        b=tr['conservative_abs_bounds']
        lines.append(f"| {tr['trial_id']} | {tr['amplitude_rad']:.7g} | {b['v_rad_s']:.6g} | {b['a_rad_s2']:.6g} | {b['jerk_rad_s3']:.6g} |")
    lines += ['', '待核验：', ''] + ['- '+b for b in plan['blockers']]
    (out/'README_plan.md').write_text('\n'.join(lines)+'\n')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=ROOT/'measurement_config.yaml', type=Path)
    p.add_argument('--arm', choices=['left','right'], required=True)
    p.add_argument('--joint', type=int, choices=range(1,8), required=True)
    p.add_argument('--frequencies', type=float, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    try:
        plan=build_plan(load_config(a.config),a.arm,a.joint,a.frequencies)
        print(save_preview(plan,a.out))
        print(f"{len(plan['trials'])} trials; hardware execution blocked ({len(plan['blockers'])} items)")
    except (ValueError, OSError, KeyError, TypeError) as e: p.exit(2, str(e)+'\n')

if __name__ == '__main__': main()
