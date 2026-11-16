#!/usr/bin/env python3
"""Preview by default; explicit mock execution. Hardware motion is not implemented."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import (ROOT, COMMAND_FIELDS, SAMPLE_FIELDS, read_json, write_json, new_run,
                    metadata, finish_metadata, envelope, capability_report, trajectory)
from plan_servo_identification import validate_plan
from protection import Protection

FAULTS=['none','duplicate','dropout','nan','sequence_rollback','clock_rollback',
        'mode_change','drive_fault','command_timeout','log_overflow','operator_stop','stop_failure']

def run_mock(plan, out, fault='none', seed=20260909):
    """Virtual time only. Writes synchronously because there is no robot/control thread.
    Device sample clock and sequence are synthetic; never label this as measurement.
    """
    validate_plan(plan)
    run=new_run(out); c=plan['config']; meta=metadata(c,'mock')
    meta.update(sample_sequence_source='synthetic_device_sequence',device_sample_identity_available=True,
                mock_model=dict(time_constant_s=.04,delay_s=.032,position_noise_std_rad=1e-5,seed=seed),
                clocks=dict(host='synthetic_host_monotonic_ns',device='synthetic_device_ns',
                            synchronization='exact synthetic offset only',synchronization_error_ns=0))
    write_json(run/'capabilities.json',dict(**capability_report(),synthetic=True))
    write_json(run/'executed_plan.json',plan); write_json(run/'available_torque_envelope.json',envelope())
    rng=np.random.default_rng(seed); period=plan['command_period_s']; dt=.004
    center=np.array(plan['center_q_rad'] or [0.]*7); axis=plan['joint']-1
    guard=Protection(); base=1_000_000_000_000_000; device_offset=4_000_000_000_000
    sample_seq=0; command_seq=0; offset=0.; injected=False
    with (run/'commands.csv').open('w',newline='') as cf, (run/'samples.csv').open('w',newline='') as sf:
        cw=csv.DictWriter(cf,fieldnames=COMMAND_FIELDS); cw.writeheader()
        sw=csv.DictWriter(sf,fieldnames=SAMPLE_FIELDS); sw.writeheader()
        for tr in plan['trials']:
            ct=np.arange(0,tr['duration_s']+period/2,period)
            cq,cv,_,_,cseg=trajectory(tr,ct)
            ts=np.arange(0,ct[-1]+dt/2,dt)
            _,_,_,_,sseg=trajectory(tr,ts)
            state=np.zeros(7); ci=0; fresh_seq=None; fresh_dev=None
            for idx,t in enumerate(ts):
                now=base+round((offset+t)*1e9)
                delayed=int(np.searchsorted(ct,t-.032,side='right')-1)
                target=np.zeros(7)
                if delayed>=0:
                    target.fill(.02*cq[delayed]); target[axis]=cq[delayed]
                old=state.copy(); state=target+(state-target)*np.exp(-dt/.04)
                v=(state-old)/dt; q=center+state+rng.normal(0,1e-5,7)
                seq=sample_seq; dev=now-device_offset; mode='normal'; vendor_fault=False
                active_fault=fault!='none' and t>=12 and tr==plan['trials'][0]
                if active_fault:
                    if fault=='dropout':
                        guard.tick(now)
                        if guard.state!='active': break
                        continue
                    if fault=='duplicate': seq=fresh_seq; dev=fresh_dev
                    elif not injected:
                        if fault=='nan': q[axis]=np.nan
                        elif fault=='sequence_rollback': seq-=5
                        elif fault=='clock_rollback': dev-=100_000_000
                        elif fault=='mode_change': mode='drag'
                        elif fault=='drive_fault': vendor_fault=True
                        elif fault in ('operator_stop','stop_failure'): guard.stop('operator_stop',now)
                        injected=True
                duplicate=guard.feedback(seq,dev,now,q.tolist(),v.tolist(),mode,vendor_fault)
                if not duplicate: fresh_seq=seq; fresh_dev=dev
                for j in range(7):
                    sw.writerow(dict(trial_id=tr['trial_id'],sample_seq=seq,arm=plan['arm'],
                        joint_name=f"{plan['arm']}_joint{j+1}",device_sample_ns=dev,host_rx_ns=now,
                        q_rad=q[j],qd_rad_s=v[j],torque_nm=2*np.sin(center[j]+state[j])+.1*v[j],
                        torque_kind='synthetic',mode=mode,fault_flags='synthetic_fault' if vendor_fault else '',
                        identity_kind='synthetic_device_sequence',duplicate=int(duplicate),
                        valid=int(guard.state=='active'),invalid_reason=guard.reason or '',segment=sseg[idx]))
                sample_seq+=1
                if guard.state!='active': break
                while ci<len(ct) and ct[ci]<=t+1e-9:
                    due=base+round((offset+ct[ci])*1e9)
                    # Virtual command events occur exactly at ct; 50Hz aligns with 250Hz feedback.
                    tx=now; done=now+1000
                    if active_fault and fault=='command_timeout': done=now+20_000_000
                    log_ok=not(active_fault and fault=='log_overflow')
                    if not guard.can_send(due,tx,done,log_ok): break
                    for j in range(7):
                        cw.writerow(dict(trial_id=tr['trial_id'],command_seq=command_seq,arm=plan['arm'],
                            joint_name=f"{plan['arm']}_joint{j+1}",host_tx_start_ns=tx,host_tx_done_ns=done,
                            q_cmd_rad=center[j]+(cq[ci] if j==axis else 0),
                            mode='synthetic_position_only',trajectory_t_s=ct[ci],
                            qd_reference_rad_s=cv[ci] if j==axis else 0,segment=cseg[ci]))
                    ci+=1; command_seq+=1
                if guard.state!='active': break
            if guard.state!='active': break
            offset+=float(ts[-1])+dt
    completed=guard.state=='active'
    if completed: guard.stop('plan_completed',now)
    if fault=='stop_failure': guard.tick(now+2_000_000_000)
    else: guard.confirm_stop(now+100_000_000,True,[0.]*7)
    meta['complete']=completed and guard.state=='stopped'
    meta['stop_state']=guard.state
    meta['invalid_reasons']=[] if meta['complete'] else [guard.reason or 'unknown',guard.state]
    with (run/'events.jsonl').open('w') as f:
        f.write(json.dumps(dict(type='synthetic_run',note='VIRTUAL TIME; NOT ROBOT DATA'))+'\n')
        for e in guard.events: f.write(json.dumps(e)+'\n')
    finish_metadata(run,meta)
    return run,meta['complete']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--backend',choices=['mock','sdk'],default='mock')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'runs')
    p.add_argument('--fault',choices=FAULTS,default='none')
    a=p.parse_args()
    try: plan=validate_plan(read_json(a.plan))
    except (ValueError,OSError,KeyError,TypeError) as e: p.exit(2,str(e)+'\n')
    if not a.execute:
        print(json.dumps(dict(preview_only=True,arm=plan['arm'],joint=plan['joint'],
                              trials=len(plan['trials']),blockers=plan['blockers']),ensure_ascii=False,indent=2))
        return 0
    if a.backend=='sdk':
        p.exit(2,'真机运动后端尚未实现，未加载SDK、未连接机器人。\n'+'\n'.join(plan['blockers'])+'\n')
    # Virtual simulator assumes these two clocks; other periods are still previewable.
    if abs(plan['command_period_s']/.004-round(plan['command_period_s']/.004))>1e-8:
        p.exit(2,'Mock requires command period to be an integer multiple of 4ms\n')
    run,complete=run_mock(plan,a.out,a.fault)
    print(run); print('SYNTHETIC ONLY; complete='+str(complete))
    return 0 if complete else 1

if __name__=='__main__': raise SystemExit(main())
