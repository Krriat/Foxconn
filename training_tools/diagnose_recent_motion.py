#!/usr/bin/env python3
"""Audit recorded policy decisions and hardware feedback without robot access."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import numpy as np
from analyze_slow_motion import replay_limiter


def stats(values):
    x=np.asarray(values)
    return {"median":float(np.median(x)),"p95":float(np.percentile(x,95)),"max":float(np.max(x))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();out=args.directory
    events=[json.loads(x) for x in (out/'run_events.jsonl').open()]
    chunks=[x for x in events if x['event']=='chunk']
    all_traces=[x['trace'] for x in events if x['event']=='servo_trace' and x['trace'].get('hardware_commands_sent')]
    traces=[x for x in all_traces if x.get('active')]
    times=np.array([x['selection_time_ns'] for x in traces],dtype=np.int64);t=(times-times[0])/1e9
    q=np.array([x['left_measured']+x['right_measured'] for x in traces])
    command=np.array([x['command'] for x in traces]);fused=np.array([x['fused_target'] for x in traces])
    limited=np.array([x['limited_target'] for x in traces]);dt=np.array([x['dt_s'] for x in traces])
    velocity=np.array([x['velocity'] for x in traces]);acceleration=np.diff(velocity,axis=0)/dt[1:,None]
    candidates=[x for x in chunks if x['sequence']>=traces[0]['chunk_sequence']]
    seq=np.array([x['sequence'] for x in candidates]);obs=np.array([x['observation_time_ns'] for x in candidates],dtype=np.int64)
    plans=np.array([x['actions'] for x in candidates]);reconstructed=[];newest=[];oldest=[];short=[]
    for tr in traces:
        ix=(np.int64(tr['requested_execution_time_ns'])-obs)/50_000_000
        valid=np.flatnonzero((seq<=tr['chunk_sequence'])&(ix>=0)&(ix<=49));h=ix[valid]
        lo=np.floor(h).astype(int);hi=np.minimum(lo+1,49);frac=(h-lo)[:,None]
        points=plans[valid,lo,:14]*(1-frac)+plans[valid,hi,:14]*frac
        weights=np.exp(-.01*np.arange(len(valid)))
        reconstructed.append(np.average(points,axis=0,weights=weights));newest.append(points[-1]);oldest.append(points[0])
        keep=h<=h[-1]+5;short.append(np.average(points[keep],axis=0,weights=weights[keep]))
    reconstructed,newest,oldest,short=map(np.asarray,(reconstructed,newest,oldest,short))
    reconstruction_error=float(np.max(abs(reconstructed-fused[:,:14])))
    assert reconstruction_error<1e-10,reconstruction_error
    anchor=all_traces[0]['command'];replay=replay_limiter(limited,dt,anchor)
    replay_error=float(np.max(abs(replay-command)));assert replay_error<1e-10,replay_error
    windows={}
    for a,b in ((0,5),(5,10),(10,20),(20,float(t[-1])+.01)):
        m=(t>=a)&(t<b)
        windows[f'{a:g}-{b:g}s']={
            'measured_range_deg':np.rad2deg(np.ptp(q[m],axis=0)).tolist(),
            'max_joint_speed_rad_s':stats(np.max(abs(velocity[m]),axis=1)),
            'newest_target_offset_deg':stats(np.rad2deg(abs(newest[m]-q[m]))),
            'fused_target_offset_deg':stats(np.rad2deg(abs(fused[m,:14]-q[m]))),
            'newest_vs_oldest_opposite_sign_fraction':float(np.mean((newest[m]-q[m])*(oldest[m]-q[m])<0)),
            'suction_target':stats(fused[m,14])}
    ctimes=(np.array([x['observation_time_ns'] for x in chunks],dtype=np.int64)-times[0])/1e9
    states=np.array([x['observation_state'] for x in chunks]);on=states[:,14]>=.5
    switches=[{'seconds_from_servo_start':float(ctimes[i]),'on':bool(on[i])} for i in np.flatnonzero(np.diff(on.astype(int)))+1]
    actual_switches=[];prev=False
    for i,x in enumerate(fused[:,14]):
        v=True if x>=.6 else False if x<=.4 else prev
        if v!=prev:actual_switches.append({'seconds':float(t[i]),'target_on':v})
        prev=v
    late=t>=20
    selected={}
    for j in (3,8,11):
        selected[f'{"left" if j<7 else "right"}_J{j%7+1}']={
            'late_newest_offset_deg':float(np.rad2deg(np.median(newest[late,j]-q[late,j]))),
            'late_oldest_offset_deg':float(np.rad2deg(np.median(oldest[late,j]-q[late,j]))),
            'late_fused_offset_deg':float(np.rad2deg(np.median(fused[late,j]-q[late,j]))),
            'late_short_window_offset_deg':float(np.rad2deg(np.median(short[late,j]-q[late,j]))),
            'late_opposite_sign_fraction':float(np.mean((newest[late,j]-q[late,j])*(oldest[late,j]-q[late,j])<0))}
    home=next(x for x in events if x['event']=='home_verified')
    reference=events[0]['home']
    report={'run_id':events[0]['run_id'],'duration_seconds':float(t[-1]),'chunks':len(chunks),'active_servo_samples':len(traces),
        'timeline':[dict(x,local_time=datetime.fromtimestamp(x['time_ns']/1e9,timezone(timedelta(hours=8))).isoformat())
                    for x in events if x['event'] in ('phase','home_verified','stage_complete','round_end')],
        'home_max_error_deg':float(np.rad2deg(np.max(abs(np.array(home['measured'])-reference)))),
        'timing':{'prediction_hz':float((len(chunks)-1)/(ctimes[-1]-ctimes[0])),
                  'servo_hz':float((len(t)-1)/t[-1]),
                  'inference_ms':stats([x['inference_ms'] for x in chunks]),
                  'network_ms':stats([(x['jetson_receive_ns']-x['send_time_ns'])/1e6 for x in chunks]),
                  'observation_age_ms':stats([x['observation_age_ms'] for x in traces]),
                  'cross_host_clock_caveat':'Network/observation measurements include clock offset.'},
        'tracking_error_deg':stats(np.rad2deg(abs(command-q))),
        'feedback_fresh':all(x[s+'_feedback']['fresh'] and x[s+'_feedback']['reader']=='drain_latest_v1' for x in traces for s in ('left','right')),
        'speed_limit_fraction':float(np.mean(np.max(abs(velocity),axis=1)>=.499)),
        'max_speed_rad_s':float(np.max(abs(velocity))),
        'acceleration_limit_fraction':float(np.mean(np.max(abs(acceleration),axis=1)>=.999)),
        'relative_clip_fraction':float(np.mean(np.any(abs(fused[:,:14]-limited)>1e-8,axis=1))),
        'suction_feedback_switches':switches,'suction_target_switches':actual_switches,
        'windows':windows,'fusion_contributors':stats([x['contributors'] for x in traces]),
        'selected_joint_fusion':selected,'fusion_reconstruction_error_rad':reconstruction_error,
        'limiter_reconstruction_error_rad':replay_error,
        'fixed_target_accel_2_final_difference_deg':float(np.rad2deg(np.max(abs(replay_limiter(limited,dt,anchor,2.)[-1]-command[-1])))),
        'caveats':['Alternative fusion uses recorded observations only and does not establish a closed-loop improvement.',
                   ('Camera streams are archived in the rosbag; there is no suction pressure measurement. '
                    'DO feedback alone cannot prove contact or vacuum attachment.' if (out/'rosbag/metadata.yaml').exists() else
                    'No recorded camera video or suction pressure measurement: DO feedback cannot prove contact or vacuum attachment.')]}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(out/'analysis.npz',time=t,measured=q,command=command,fused=fused,newest=newest,oldest=oldest,short=short)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for ax,j in zip(axs[:2],(3,8)):
        for data,name in ((newest,'Newest prediction'),(fused[:,:14],'Fused target'),(command,'Command'),(q,'Measured')):
            ax.plot(t,np.rad2deg(data[:,j]),label=name,lw=1)
        ax.set_ylabel('Joint angle (deg)');ax.set_title(('Left J4','Right J2')[j==8]);ax.legend(ncol=4,fontsize=8)
    axs[2].plot(t,fused[:,14],label='Fused left suction target');axs[2].step(ctimes,states[:,14],where='post',label='Left suction DO feedback')
    axs[2].axhline(.6,color='gray',ls='--',label='ON threshold');axs[2].axhline(.4,color='gray',ls=':');axs[2].legend();axs[2].set_xlabel('Time from active ServoJ (s)')
    for ax in axs:ax.grid(alpha=.25)
    started=datetime.fromtimestamp(times[0]/1e9,timezone(timedelta(hours=8)))
    fig.suptitle(f'{started:%Y-%m-%d %H:%M:%S}: tracking and policy/fusion targets')
    fig.tight_layout();fig.savefig(out/'diagnosis.png',dpi=140)
    print(json.dumps({k:report[k] for k in ('timing','tracking_error_deg','speed_limit_fraction','max_speed_rad_s','relative_clip_fraction','suction_feedback_switches','suction_target_switches','fusion_contributors','selected_joint_fusion','fixed_target_accel_2_final_difference_deg')},indent=2))


if __name__=='__main__':main()
