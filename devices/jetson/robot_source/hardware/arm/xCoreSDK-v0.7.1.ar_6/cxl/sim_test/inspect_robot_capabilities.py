#!/usr/bin/env python3
"""Default: local SDK survey only. --connect: idle-only feedback collection."""
import argparse
import csv
import ipaddress
import json
from pathlib import Path
import subprocess
import time
from common import (ROOT, SAMPLE_FIELDS, COMMAND_FIELDS, capability_report, load_config,
                    write_json, new_run, metadata, finish_metadata, envelope)

def normalize_frames(run, arm, torque):
    raw=run/'raw_vendor'/'frames.jsonl'
    with (run/'commands.csv').open('w',newline='') as f:
        csv.writer(f).writerow(COMMAND_FIELDS)
    with (run/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=SAMPLE_FIELDS); w.writeheader()
        if not raw.exists(): return
        with raw.open() as src:
            for line in src:
                try: r=json.loads(line)
                except json.JSONDecodeError: continue  # A killed writer's partial line remains in raw_vendor.
                for j in range(7):
                    valid=r['q'][j] is not None and r['qd'][j] is not None and (not torque or r['tau'][j] is not None)
                    w.writerow(dict(trial_id='stationary_inspection',sample_seq=r['sample_seq'],arm=arm,
                        joint_name=f'{arm}_joint{j+1}',host_rx_ns=r['host_rx_ns'],
                        q_rad=r['q'][j],qd_rad_s=r['qd'][j],torque_nm=r['tau'][j],
                        torque_kind='sdk_tau_m_unverified' if torque else 'unavailable',
                        mode='read_only_idle_checked_4Hz',identity_kind='host_transport_counter',
                        identical_payload=int(r['identical_payload']),valid=int(valid),
                        invalid_reason='' if valid else 'nonfinite_or_missing',segment='stationary'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'measurement_config.yaml')
    p.add_argument('--out',type=Path,default=ROOT/'runs')
    p.add_argument('--arm',choices=['left','right'])
    p.add_argument('--connect',action='store_true',help='Connect only when robot is idle and other clients have exited; SDK disconnect can stop motion')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--without-torque',action='store_true',help='Explicit q/dq-only collection if tau_m unsupported; no automatic reconnect')
    p.add_argument('--bridge',type=Path,default=ROOT/'build/inspect_ar5_sdk')
    a=p.parse_args()
    if not 10 <= a.seconds <= 600: p.error('--seconds must be 10..600')
    c=load_config(a.config)
    if a.connect:
        if not a.arm: p.error('--arm is required for --connect')
        if not a.bridge.is_file(): p.error('Build the read-only bridge first (see README.md)')
        for key in ('robot_ip','local_ip'): ipaddress.IPv4Address(c['connection'][a.arm][key])
        if c['interface']['feedback_interval_ms'] not in [1,2,4,8,1000]: p.error('Unsupported feedback interval')
    run=new_run(a.out); caps=capability_report(); meta=metadata(c,'sdk_read_only' if a.connect else 'local_survey')
    write_json(run/'capabilities.json',caps); write_json(run/'available_torque_envelope.json',envelope())
    write_json(run/'executed_plan.json',dict(kind='read_only_inspection',arm=a.arm,duration_s=a.seconds if a.connect else 0))
    rc=0; reasons=[]
    if a.connect:
        addr=c['connection'][a.arm]
        command=[str(a.bridge.resolve()),'--connect-idle-read-only',addr['robot_ip'],addr['local_ip'],
                 str(a.seconds),str(c['interface']['feedback_interval_ms']),str(run/'raw_vendor'),
                 'qv' if a.without_torque else 'qvt']
        print('只读连接：必须已退出其他控制程序且机械臂静止；SDK断开可能停止运动。',flush=True)
        with (run/'raw_vendor'/'bridge_stdout.log').open('w') as out, (run/'raw_vendor'/'bridge_stderr.log').open('w') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,cwd=run/'raw_vendor')
            try: rc=child.wait(timeout=a.seconds+20)
            except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
                reasons.append(type(exc).__name__+': process termination is NOT physical stop confirmation')
                child.terminate()
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                rc=1
        native_events=run/'raw_vendor'/'native_events.jsonl'
        (run/'events.jsonl').write_text(native_events.read_text() if native_events.exists() else '')
        normalize_frames(run,a.arm,not a.without_torque)
        info=run/'raw_vendor'/'robot_info.json'
        if info.exists():
            try:
                runtime=json.loads(info.read_text()); caps['runtime']=runtime; caps['connected']=True
                caps['controller_version']['value']=runtime['controller_version']
                caps['connection']['value']=dict(arm=a.arm,**addr)
            except json.JSONDecodeError: reasons.append('Incomplete robot_info.json; original preserved')
        if rc: reasons.append('SDK acquisition failed; see raw_vendor/bridge_stderr.log and events.jsonl')
    else:
        (run/'events.jsonl').write_text(json.dumps(dict(type='local_survey_only',host_ns=time.monotonic_ns()))+'\n')
        for name,fields in [('samples.csv',SAMPLE_FIELDS),('commands.csv',COMMAND_FIELDS)]:
            with (run/name).open('w',newline='') as f: csv.writer(f).writerow(fields)
    meta['complete']=rc==0 and not reasons; meta['invalid_reasons']=reasons
    write_json(run/'capabilities.json',caps); finish_metadata(run,meta)
    print(run)
    if a.connect:
        from analyze_actuator_logs import analyze
        analyze(run)
    else:
        (run/'README_results.md').write_text('# 本地只读能力调查\n\n未连接机器人，未取得实测数据。详见 capabilities.json；真机运动后端未开放。\n')
    return rc if rc else (1 if reasons else 0)

if __name__=='__main__': raise SystemExit(main())
