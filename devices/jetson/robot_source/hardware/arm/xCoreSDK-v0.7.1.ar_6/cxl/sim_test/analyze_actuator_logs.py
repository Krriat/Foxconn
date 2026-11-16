#!/usr/bin/env python3
"""Fully offline analysis. This module never imports or starts the SDK bridge."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import tarfile
import numpy as np
from common import read_json, write_json, digest, envelope

def num(s):
    try: return float(s) if s not in ('',None) else math.nan
    except (ValueError,TypeError): return math.nan

def integer(s):
    # Parse timestamps without first rounding through floating point.
    return int(s) if s not in ('',None) else None

def distribution(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return dict(count=0,min=None,max=None,mean=None,std=None,p95=None)
    return dict(count=len(a),min=float(a.min()),max=float(a.max()),mean=float(a.mean()),
                std=float(a.std()),p95=float(np.percentile(a,95)))

def harmonic(t,y,f):
    design=np.column_stack([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones(len(t))])
    coef=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y-design@coef
    return float(np.hypot(coef[0],coef[1])),float(np.arctan2(coef[1],coef[0])),float(coef[2]),float(np.std(residual))

def load_samples(path):
    groups=defaultdict(list)
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            key=(r['trial_id'],r['arm'],r['joint_name'])
            groups[key].append((integer(r['host_rx_ns']),integer(r['sample_seq']),integer(r['device_sample_ns']),
                num(r['q_rad']),num(r['qd_rad_s']),num(r['torque_nm']),r['torque_kind'],num(r['current_a']),
                num(r['motor_temperature_c']),r.get('identity_kind','unknown'),r.get('duplicate',''),
                r.get('valid','1'),r.get('segment','unknown'),r.get('identical_payload',''),
                r['fault_flags'],r['limit_flags']))
    return groups

def load_commands(path):
    groups=defaultdict(list)
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            groups[(r['trial_id'],r['arm'],r['joint_name'])].append((integer(r['host_tx_start_ns']),
                integer(r['host_tx_done_ns']),num(r['q_cmd_rad']),num(r.get('qd_reference_rad_s')),
                num(r.get('trajectory_t_s')),integer(r['command_seq'])))
    return groups

def model_fit(estimates):
    """Multi-frequency K/(1+s*T) exp(-s*d), grid bounds reported explicitly.
    Fit identification segments only; never include the held-out frequency.
    """
    valid=[e for e in estimates if e.get('identifiable') and e['role']=='identification' and e['excited']]
    freqs=sorted(set(e['frequency_hz'] for e in valid))
    if len(freqs)<3:
        return dict(status='not_identifiable',reason='At least three informative identification frequencies required to separate first-order response and delay; single-frequency phase is ambiguous')
    w=np.array([2*np.pi*f for f in freqs])
    H=np.array([np.mean([e['gain']*np.exp(1j*e['phase_rad']) for e in valid if e['frequency_hz']==f]) for f in freqs])
    candidates=[]
    for tau in np.geomspace(.002,1.,100):
        for delay in np.linspace(0.,.5,126):
            shape=np.exp(-1j*w*delay)/(1+1j*w*tau)
            gain=float(np.clip(np.real(np.vdot(shape,H))/np.real(np.vdot(shape,shape)),0.,2.))
            loss=float(np.mean(np.abs(H-gain*shape)**2))
            candidates.append((loss,float(tau),float(delay),gain))
    candidates.sort(); loss,tau,delay,gain=candidates[0]
    near=[r for r in candidates if r[0]<=loss+max(1e-5,loss*.10)]
    drange=[min(r[2] for r in near),max(r[2] for r in near)]
    boundary=(tau <= .00201 or tau >= .9999 or delay in (0.,.5) or gain in (0.,2.))
    return dict(status='candidate_ambiguous' if boundary or drange[1]-drange[0]>.04 else 'candidate',
                model='K/(1+s*T) exp(-s*delay)',tau_s=tau,effective_delay_s=delay,gain=gain,
                loss=loss,near_optimal_delay_range_s=drange,range_is_confidence_interval=False,
                search_bounds=dict(tau_s=[.002,1.],delay_s=[0.,.5],gain=[0.,2.]),
                boundary_solution=boundary,fit_frequencies_hz=freqs,
                interpretation='Host-clock effective response only; cannot isolate network latency or internal PID',
                holdout=[])

def predict(t,ct,cq,model,q0):
    # Zero-order hold of actually logged position commands. No velocity feedforward.
    idx=np.maximum(0,np.searchsorted(ct,t-model['effective_delay_s'],side='right')-1)
    target=q0+model['gain']*(cq[idx]-q0)
    y=np.empty(len(t)); y[0]=q0
    for i in range(1,len(t)):
        alpha=math.exp(-(t[i]-t[i-1])/model['tau_s'])
        y[i]=alpha*y[i-1]+(1-alpha)*target[i]
    return y

def analyze(run, plots=True):
    run=Path(run).resolve(); meta=read_json(run/'metadata.json'); plan=read_json(run/'executed_plan.json')
    mismatches=[name for name,h in meta.get('raw_log_sha256',{}).items()
                if not (run/name).is_file() or digest(run/name)!=h]
    synthetic=meta.get('synthetic',False); complete=meta.get('complete',False)
    summary=dict(synthetic=synthetic,run_complete=complete,invalid_reasons=meta.get('invalid_reasons',[]),
                 raw_hash_mismatches=mismatches,groups=[],repeats=[],models={},
                 available_torque_envelope_status='unavailable',
                 time_basis='host receive vs host command-send clock; effective total lag, not network delay',
                 native_sampling_rate_hz=None if not meta.get('device_sample_identity_available') else 'see per-group device intervals')
    samples=load_samples(run/'samples.csv'); commands=load_commands(run/'commands.csv')
    trmap={t['trial_id']:t for t in plan.get('trials',[])}
    estimates=[]; plotdata=[]; validationdata=[]; torque_groups=defaultdict(list)
    cfg=meta['config']; condition=cfg.get('condition',{})
    tf=(run/'torque_vs_speed.csv').open('w',newline='')
    tw=csv.DictWriter(tf,fieldnames=['trial_id','arm','joint_name','direction','posture_id','load_id',
        'motor_temperature_c','segment','qd_rad_s','torque_nm','raw_sdk_tau_nm','torque_kind','current_a',
        'dataset_kind','interpretation'])
    tw.writeheader()
    try:
        for key,rows in samples.items():
            trial_id,arm,joint=key; tr=trmap.get(trial_id)
            rx=np.array([r[0] for r in rows],dtype=np.int64)
            q=np.array([r[3] for r in rows]); v=np.array([r[4] for r in rows])
            torque=np.array([r[5] for r in rows]); current=np.array([r[7] for r in rows])
            device_known=bool(meta.get('device_sample_identity_available'))
            keep=np.isfinite(q)&np.array([r[11] not in ('0','false','False') for r in rows])
            duplicate=np.array([r[10] in ('1','true','True') for r in rows])
            rollback=np.zeros(len(rows),dtype=bool); missing=0
            seen=set(); prev_seq=None; prev_dev=None
            if device_known:
                for i,r in enumerate(rows):
                    identity=(r[1],r[2])
                    # Use either known device ID; identical q alone never means duplicate.
                    if identity in seen: duplicate[i]=True
                    seen.add(identity)
                    if prev_seq is not None and r[1] is not None:
                        if r[1]<prev_seq: rollback[i]=True
                        missing+=max(0,r[1]-prev_seq-1)
                    if prev_dev is not None and r[2] is not None and r[2]<prev_dev: rollback[i]=True
                    prev_seq=r[1]; prev_dev=r[2]
            host_rollback=np.r_[False,np.diff(rx)<=0]
            keep &= ~duplicate & ~rollback & ~host_rollback
            first_rx=int(rx[0]); t=np.array([(int(n)-first_rx)/1e9 for n in rx])
            baseline_end=tr['stationary_before_s'] if tr else 10.
            baseline=keep & (t<baseline_end)
            noise=float(np.std(q[baseline])) if baseline.sum()>2 else None
            actual_duration=float(t[-1]-t[0]) if len(t)>1 else 0.
            packet_intervals=np.diff(rx)/1e6
            dev_intervals=[]
            if device_known:
                devices=[rows[i][2] for i in np.flatnonzero(keep) if rows[i][2] is not None]
                dev_intervals=[(b-a)/1e6 for a,b in zip(devices,devices[1:])]
            group=dict(trial_id=trial_id,arm=arm,joint_name=joint,total_rows=len(rows),
                       valid_rows=int(keep.sum()),device_identity_known=device_known,
                       duplicate_samples=int(duplicate.sum()) if device_known else None,
                       identical_payload_rows=sum(r[13] in ('1','true','True') for r in rows),
                       missing_device_samples=missing if device_known else None,
                       sequence_or_device_clock_rollback=int(rollback.sum()),
                       host_clock_nonincreasing=int(host_rollback.sum()),
                       host_packet_interval_ms=distribution(packet_intervals),
                       device_sample_interval_ms=distribution(dev_intervals),
                       sample_rate_interpretation='synthetic device samples' if synthetic else 'SDK received packets; native device sampling rate UNKNOWN',
                       baseline_duration_s=min(actual_duration,baseline_end),baseline_noise_rad=noise,
                       speed_abs_rad_s=distribution(np.abs(v[keep])),
                       fault_or_limit_rows=sum(bool(r[14] or r[15]) for r in rows))
            summary['groups'].append(group)
            for i in np.flatnonzero(keep):
                r=rows[i]; kind=r[6]; direction='positive' if v[i]>0 else 'negative' if v[i]<0 else 'zero'
                is_demand=kind in ('joint_output_sensor','current_calibrated_estimate','synthetic')
                temp=r[8]; tempbin=None if not math.isfinite(temp) else math.floor(temp/5)*5
                interp='observed_torque_demand' if is_demand else kind if kind in ('commanded','external_torque_estimate') else 'raw_feedback_provenance_unverified'
                tw.writerow(dict(trial_id=trial_id,arm=arm,joint_name=joint,direction=direction,
                    posture_id=condition.get('posture_id'),load_id=condition.get('load_id'),
                    motor_temperature_c=temp if math.isfinite(temp) else '',segment=r[12],qd_rad_s=v[i],
                    torque_nm=torque[i] if is_demand and math.isfinite(torque[i]) else '',
                    raw_sdk_tau_nm=torque[i] if math.isfinite(torque[i]) else '',torque_kind=kind,
                    current_a=current[i] if math.isfinite(current[i]) else '',
                    dataset_kind='synthetic' if synthetic else 'robot_feedback',interpretation=interp))
                torque_groups[(arm,joint,direction,kind,tempbin,r[12])].append((v[i],torque[i],current[i]))
            cr=commands.get(key,[])
            if not tr or not cr or keep.sum()<10: continue
            origin=cr[0][0]
            ct=np.array([(r[0]-origin)/1e9 for r in cr]); cq=np.array([r[2] for r in cr]); cv=np.array([r[3] for r in cr])
            st=np.array([(int(n)-origin)/1e9 for n in rx])
            timing_ok=bool(np.all(np.diff(ct)>0)) and np.isfinite(cq).all()
            stable=keep&(st>=tr['stable_start_s'])&(st<=tr['stable_end_s'])
            cs=(ct>=tr['stable_start_s'])&(ct<=tr['stable_end_s'])
            excited=joint.endswith('joint'+str(tr['joint']))
            identifiable=bool(complete and not mismatches and timing_ok and not rollback.any()
                              and not host_rollback.any() and missing==0
                              and stable.sum()>10 and cs.sum()>10)
            e=dict(trial_id=trial_id,arm=arm,joint_name=joint,role=tr['role'],repeat=tr['repeat'],excited=excited,
                   frequency_hz=tr['frequency_hz'],amplitude_rad=tr['amplitude_rad'],identifiable=False,
                   gain=None,phase_rad=None,effective_phase_lag_s=None,position_rmse_rad=None,
                   velocity_rmse_rad_s=None,max_tracking_error_rad=None,steady_bias_rad=None,
                   response_amplitude_rad=None,signal_to_noise=None,
                   lag_interpretation='wrapped phase-equivalent total lag; ambiguous modulo one period; no communication-delay claim')
            if timing_ok:
                # No resampling written back to raw logs; ZOH command comparison is analysis-only.
                ci=np.maximum(0,np.searchsorted(ct,st,side='right')-1)
                err=q-cq[ci]; valid=keep&(st>=ct[0])&(st<=ct[-1])
                if valid.any():
                    e['position_rmse_rad']=float(np.sqrt(np.mean(err[valid]**2)))
                    e['max_tracking_error_rad']=float(np.max(np.abs(err[valid])))
                    e['steady_bias_rad']=float(np.mean(err[stable])) if stable.any() else None
                    ve=valid&np.isfinite(v)&np.isfinite(cv[ci])
                    if ve.any(): e['velocity_rmse_rad_s']=float(np.sqrt(np.mean((v[ve]-cv[ci[ve]])**2)))
                group['command_interval_ms']=distribution(np.diff(ct)*1000)
                group['command_call_duration_us']=distribution([(r[1]-r[0])/1000 for r in cr if r[1] is not None])
                group['min_actual_command_points_per_cycle']=float(1/(tr['frequency_hz']*np.max(np.diff(ct)))) if len(ct)>1 else None
                if group['min_actual_command_points_per_cycle'] is None or group['min_actual_command_points_per_cycle']<10-1e-6:
                    identifiable=False
            if identifiable:
                a,phase,_,resid=harmonic(st[stable],q[stable],tr['frequency_hz'])
                ca,cp,_,_=harmonic(ct[cs],cq[cs],tr['frequency_hz'])
                snr=a/max(noise or 0,resid,1e-12)
                e.update(response_amplitude_rad=a,signal_to_noise=snr)
                # Other axes' amplitude reports coupling without dividing by a zero command amplitude.
                if excited and ca>1e-12 and snr>=cfg['trial']['minimum_signal_to_noise']:
                    delta=float(np.angle(np.exp(1j*(phase-cp))))
                    e.update(identifiable=True,gain=a/ca,phase_rad=delta,
                             effective_phase_lag_s=-delta/(2*np.pi*tr['frequency_hz']))
            estimates.append(e)
            if excited:
                if tr['role']=='holdout': validationdata.append((arm,joint,tr,st[keep],q[keep],ct,cq))
                if tr['repeat']==1: plotdata.append((trial_id,st[keep],q[keep],ct,cq))
    finally: tf.close()
    for arm,joint in sorted(set((e['arm'],e['joint_name']) for e in estimates if e['excited'])):
        subset=[e for e in estimates if e['arm']==arm and e['joint_name']==joint]
        model=model_fit(subset)
        for a,j,tr,t,q,ct,cq in validationdata:
            if (a,j)!=(arm,joint) or 'tau_s' not in model: continue
            prediction=predict(t,ct,cq,model,float(np.mean(q[t<min(10,t[-1])])) )
            model['holdout'].append(dict(trial_id=tr['trial_id'],whole_segment_rmse_rad=float(np.sqrt(np.mean((q-prediction)**2))),
                                         used_for_fitting=False))
        model['applicability']=dict(condition=condition,frequencies_hz=sorted(set(e['frequency_hz'] for e in subset)),
                                   maximum_amplitude_rad=max(e['amplitude_rad'] for e in subset),
                                   warning='Low-speed local posture/load only; no full-speed extrapolation or PID interpretation')
        summary['models'][arm+'/'+joint]=model
        for f in sorted(set(e['frequency_hz'] for e in subset)):
            valid=[e for e in subset if e['frequency_hz']==f and e['identifiable'] and e['role']=='identification']
            summary['repeats'].append(dict(arm=arm,joint_name=joint,frequency_hz=f,repeats=len(valid),
                gain=distribution([e['gain'] for e in valid]),effective_phase_lag_s=distribution([e['effective_phase_lag_s'] for e in valid]),
                spread_is_confidence_interval=False))
    stats=[]
    for (arm,joint,direction,kind,tempbin,segment),values in torque_groups.items():
        arr=np.array(values)
        stats.append(dict(arm=arm,joint_name=joint,direction=direction,torque_kind=kind,temperature_bin_c=tempbin,
                          posture_id=condition.get('posture_id'),load_id=condition.get('load_id'),segment=segment,
                          speed_rad_s=distribution(arr[:,0]),raw_torque_nm=distribution(arr[:,1]),current_a=distribution(arr[:,2]),
                          interpretation='observed demand only when provenance is established; never available capacity',synthetic=synthetic))
    write_json(run/'torque_statistics.json',stats)
    write_json(run/'summary.json',summary)
    write_json(run/'available_torque_envelope.json',envelope())
    estimate_fields=list(estimates[0]) if estimates else ['trial_id','arm','joint_name','identifiable','effective_phase_lag_s']
    with (run/'servo_estimates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=estimate_fields);w.writeheader();w.writerows(estimates)
    if plots and plotdata:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (run/'plots').mkdir(exist_ok=True)
        for trial_id,t,q,ct,cq in plotdata:
            fig,ax=plt.subplots(figsize=(11,4)); ax.plot(ct,cq,label='Command (ZOH)',drawstyle='steps-post')
            stride=max(1,len(t)//12000); ax.plot(t[::stride],q[::stride],label='Feedback',alpha=.8)
            ax.set(xlabel='Host-relative time (s)',ylabel='Position (rad)',title=('SYNTHETIC: ' if synthetic else '')+trial_id)
            ax.legend();fig.tight_layout();fig.savefig(run/'plots'/f'{trial_id}.png');plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        for key,rows in samples.items():
            selected=rows[::max(1,len(rows)//1000)]
            ax.scatter([r[4] for r in selected],[r[5] for r in selected],s=2,alpha=.4)
        ax.set(xlabel='Joint speed (rad/s)',ylabel='Raw SDK torque (Nm; provenance in CSV)',
               title=('SYNTHETIC; ' if synthetic else '')+'Observed feedback, NOT available torque envelope')
        fig.tight_layout();fig.savefig(run/'plots/torque_vs_speed.png');plt.close(fig)
    lines=['# 测量结果（离线）','', '**合成数据，仅用于流程验证，不能作为机器人测量结论。**' if synthetic else '机器人只读数据；没有命令时不能辨识伺服响应。', '',
           f"采集完成：{complete}；原始日志哈希异常：{mismatches or '无'}。",'',
           'SDK不提供设备序号/时间戳时，原生采样率、丢帧和重复设备采样保持未知；报告仅统计收包间隔。',
           '相位换算的是有效总滞后，包含机械、滤波和传输影响，存在整周期歧义。未进行设备与主机时钟直接相减。','',
           '| 臂/关节 | 有效总滞后 s（包裹相位） | 测试 Hz / 振幅 rad | 位置 RMSE rad | 力矩反馈类型 | 已覆盖最大绝对转速 rad/s | 限幅/异常 |',
           '|---|---:|---|---:|---|---:|---|']
    for g in summary['groups']:
        es=[e for e in estimates if (e['trial_id'],e['arm'],e['joint_name'])==(g['trial_id'],g['arm'],g['joint_name'])]
        e=es[0] if es else {}; k=(g['trial_id'],g['arm'],g['joint_name'])
        kinds=','.join(sorted(set(r[6] for r in samples[k])))
        lines.append(f"| {g['arm']}/{g['joint_name']} ({g['trial_id']}) | {e.get('effective_phase_lag_s')} | {e.get('frequency_hz')} / {e.get('amplitude_rad')} | {e.get('position_rmse_rad')} | {kinds} | {g['speed_abs_rad_s']['max']} | {g['fault_or_limit_rows']}; complete={complete} |")
    lines += ['', '多频模型、整段留出误差及重复试验范围见 summary.json；单频不强行分离机械响应与纯延迟。',
              '速度误差对比轨迹速度参考；位置接口并未发送速度前馈。全轴谐波幅值用于查看耦合运动。',
              '力矩来源未核实的 tau_m 保存在 raw_sdk_tau_nm；torque_nm 留空。缺失电流/温度/电压不以零代替。',
              'available_torque_envelope.status = unavailable。厂家需提供输出端连续/峰值曲线、峰值时长、占空比、温度/电压降额、正反向/驱动制动差异、减速比/效率、母线多轴限制及反馈标定资料。',
              '没有进行整机极限搜索。观测最大值、95%分位数及限流状态均不能替代能力包络。']
    (run/'README_results.md').write_text('\n'.join(lines)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path); p.add_argument('--no-plots',action='store_true')
    p.add_argument('--archive',action='store_true',help='Create a sibling .tar.gz; never overwrite an existing archive')
    a=p.parse_args(); summary=analyze(a.run,not a.no_plots)
    if a.archive:
        archive=a.run.resolve().with_suffix('.tar.gz')
        with archive.open('xb') as f:
            with tarfile.open(fileobj=f,mode='w:gz') as tar: tar.add(a.run,arcname=a.run.name)
        print(archive)
    print(a.run/'README_results.md')
    return 2 if summary['raw_hash_mismatches'] else 0

if __name__=='__main__': raise SystemExit(main())
