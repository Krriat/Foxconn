#!/usr/bin/env python3
"""Fixed-log regression of production callbacks and native trajectory math.

No ROS initialization, publishers, service clients, SDK or hardware access.
Time, feedback and plan arrival order come from the recorded run. Alternative
outputs are conditional on recorded feedback, not a simulated robot rollout.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, MethodType

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'training_tools/policy_recovery'))
sys.path.insert(0,str(ROOT/'training_tools/two_stage_policy'))
from joint_anchor import anchor_joint_chunk


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_callbacks(source, protocol, config, run_id):
    from stage_policy import action_config
    # Extract and execute the actual callback bodies without importing the ROS
    # Node or constructing any communication/hardware objects.
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Executor')
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('on_action','on_trace')]
    clock=SimpleNamespace(ns=0)
    scope=dict(vars(protocol),json=json,String=SimpleNamespace,action_config=action_config,
               time=SimpleNamespace(time_ns=lambda:clock.ns,monotonic=lambda:clock.ns/1e9))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
    emitted=[]; faults=[]
    obj=SimpleNamespace(phase='RUNNING',run_id=run_id,publisher_id=None,last_sequence=0,
        mode='live',home_verified_run_id=run_id,sequence_job=None,
        inference_status={},config=config,heartbeats={},accepted=0,rejected=0,limited_ticks=0,
        shadow_suction=[False,False],native_trace={},native_received=0.,previous=None,
        validated_pub=SimpleNamespace(publish=lambda m:emitted.append(json.loads(m.data))),
        event=lambda *a,**kw:None,request_stop=lambda reason,**kw:faults.append(reason))
    for name in ('on_action','on_trace'):setattr(obj,name,MethodType(scope[name],obj))
    return obj,clock,emitted,faults


def native_replay(events,chunks,transformed,binary,config):
    all_traces=[e['trace'] for e in events if e['event']=='servo_trace' and e['trace'].get('hardware_commands_sent')]
    traces=[t for t in all_traces if t.get('active')]
    inputs=io.StringIO()
    inputs.write(' '.join(str(config[key]) for key in ('temporal_ensemble_coefficient',
        'max_joint_speed_rad_s','max_joint_acceleration_rad_s2','trajectory_response_seconds',
        'max_relative_target_rad','max_tracking_error_rad'))+'\n')
    inputs.write('A '+' '.join(map(repr,all_traces[0]['command']))+'\n')
    plans=[c for c in chunks if c['sequence']>=traces[0]['chunk_sequence']]
    cursor=0
    for tr in traces:
        while cursor<len(plans) and plans[cursor]['sequence']<=tr['chunk_sequence']:
            p=plans[cursor];a=transformed[p['sequence']]
            inputs.write(f"P {p['sequence']} {p['observation_time_ns']} "+' '.join(map(repr,a.ravel().tolist()))+'\n')
            cursor+=1
        inputs.write(f"T {tr['requested_execution_time_ns']} {tr['dt_s']!r} "+
                     ' '.join(map(repr,tr['left_measured']+tr['right_measured']))+'\n')
    result=subprocess.run([str(binary)],input=inputs.getvalue(),text=True,capture_output=True,check=True)
    values=np.loadtxt(io.StringIO(result.stdout))
    assert len(values)==len(traces)
    return traces,values[:,0],values[:,1:17],values[:,17:31],values[:,31:45]


def evaluate(source,events,protocol,config,mode,binary):
    chunks=[e for e in events if e['event']=='chunk']
    obj,clock,emitted,faults=load_callbacks(source,protocol,config,events[0]['run_id'])
    transformed={};bias_failures=[]
    for c in chunks:
        actions=np.asarray(c['actions'])
        if mode=='anchor':
            try:actions=anchor_joint_chunk(actions.astype(np.float32),np.asarray(c['observation_state'],np.float32))
            except ValueError as exc:
                bias_failures.append(dict(sequence=c['sequence'],error=str(exc)));continue
        transformed[c['sequence']]=actions
        value={k:v for k,v in c.items() if k not in ('time_ns','event','phase','jetson_receive_ns')}
        value['actions']=actions.tolist()
        # This recording predates stage tags. Model it as a default far round;
        # retain the recorded checkpoint hash and every action/timestamp.
        if config.get('act_stage_checkpoints') and 'policy_key' not in value:
            value['policy_key']='default'
        clock.ns=c['jetson_receive_ns']
        obj.phase=c['phase'];obj.inference_status={'publisher_id':c['publisher_id']}
        # Original heartbeat issue times are absent from this log. Reuse token
        # identity and explicitly model it as fresh; do not claim age coverage.
        obj.heartbeats[c['executor_heartbeat']]=clock.ns/1e9
        obj.on_action(SimpleNamespace(data=json.dumps(value)))
    report=dict(callback_source=str(source),callback_sha256=sha(source),mode=mode,
        chunks=len(chunks),accepted=obj.accepted,rejected=obj.rejected,callback_faults=faults.copy(),
        anchor_guard_rejections=bias_failures)
    if faults or bias_failures or len(emitted)!=len(chunks):return report,None
    # Feed the callback's actual validated output into the native layer, so a
    # later callback modification cannot be silently bypassed by this regression.
    validated={c['sequence']:np.asarray(c['actions']) for c in emitted}
    traces,counts,fused,command,velocity=native_replay(events,emitted,validated,binary,config)
    original_command=np.asarray([t['command'] for t in traces]);original_fused=np.asarray([t['fused_target'] for t in traces])
    t=(np.array([tr['selection_time_ns'] for tr in traces],dtype=np.int64)-traces[0]['selection_time_ns'])/1e9
    dt=np.array([tr['dt_s'] for tr in traces]);switches=[];last=[False,False]
    for i,tr in enumerate(traces):
        clock.ns=tr['selection_time_ns'];obj.phase='RUNNING'
        value=dict(tr,run_id=events[0]['run_id'],command=command[i].tolist(),fused_target=fused[i].tolist())
        obj.on_trace(SimpleNamespace(data=json.dumps(value)))
        if obj.shadow_suction!=last:
            switches.append(dict(seconds=float(t[i]),target=obj.shadow_suction.copy()));last=obj.shadow_suction.copy()
    report.update(active_servo_samples=len(traces),duration_seconds=float(t[-1]),
        fused_target_max_difference_rad=float(np.max(abs(fused-original_fused))),
        command_max_difference_rad=float(np.max(abs(command-original_command))),
        command_max_difference_deg=float(np.rad2deg(abs(command-original_command)).max()),
        command_p95_difference_deg=float(np.percentile(np.rad2deg(abs(command-original_command)),95)),
        contributor_count_mismatch=int(np.count_nonzero(counts!=[tr['contributors'] for tr in traces])),
        max_speed_rad_s=float(np.max(abs(velocity))),
        max_acceleration_rad_s2=float(np.max(abs(np.diff(velocity,axis=0)/dt[1:,None]))),
        suction_target_switches=switches,trace_faults=faults,
        outcome='exact_recorded_execution_reproduced' if np.max(abs(command-original_command))<1e-9 and not faults
                else 'counterfactual_execution_differs_from_success')
    return report,dict(time=t,command=command,fused=fused,recorded=original_command,velocity=velocity)


def guard_checks(source,events,protocol,config):
    chunk=next(e for e in events if e['event']=='chunk')
    passed=[]
    for kind in ('old_run','repeated_sequence','stale_transport','expired_heartbeat','idle_ignores_chunk'):
        obj,clock,emitted,faults=load_callbacks(source,protocol,config,chunk['run_id'])
        value={k:v for k,v in chunk.items() if k not in ('time_ns','event','phase','jetson_receive_ns')}
        if config.get('act_stage_checkpoints') and 'policy_key' not in value:
            value['policy_key']='default'
        clock.ns=chunk['jetson_receive_ns'];obj.heartbeats[value['executor_heartbeat']]=clock.ns/1e9
        if kind=='old_run':value['run_id']='another_round'
        if kind=='repeated_sequence':obj.last_sequence=value['sequence']
        if kind=='stale_transport':clock.ns+=400_000_000
        if kind=='expired_heartbeat':obj.heartbeats[value['executor_heartbeat']]-=1.
        if kind=='idle_ignores_chunk':obj.phase='IDLE'
        obj.on_action(SimpleNamespace(data=json.dumps(value)))
        assert not emitted and obj.accepted==0,kind
        if kind in ('repeated_sequence','stale_transport','expired_heartbeat'):assert faults,kind
        passed.append(kind)
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',type=Path,default=ROOT/'training_verification/success_failure_20261112/success_first60.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'training_verification/success_replay_20261112')
    parser.add_argument('--config',type=Path,default=ROOT/'training_verification/slow_motion_latest_20261112/config.yaml',
                        help='Executor configuration snapshot; copy the deployed config here to test parameter changes.')
    parser.add_argument('--protocol',type=Path,default=Path('/home/jj/.local/share/robot-policy-streams/protocol.py'),
                        help='Protocol implementation under test, without importing ROS or hardware code.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(args.config.read_text())
    spec=importlib.util.spec_from_file_location('replayed_protocol',args.protocol)
    protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    events=[json.loads(l) for l in args.events.open()]
    fixture=json.loads((HERE/'fixture.json').read_text())
    assert sha(args.events)==fixture['sha256'],'Recorded fixture changed; review its provenance before updating the expected hash'
    assert events[0]['run_id']==fixture['run_id']
    binary=args.output/'native_replay'
    subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra',str(HERE/'native/replay.cpp'),'-o',str(binary)],check=True)
    sources={'success_version':ROOT/'training_verification/restore_far_20261112/baseline/executor_node.py',
             'current_version':ROOT/'training_tools/two_stage_policy/executor_node.py'}
    reports={};arrays={}
    for name,path,mode in [('success_version',sources['success_version'],'absolute'),
                           ('current_version',sources['current_version'],'absolute'),
                           ('anchor_candidate',sources['current_version'],'anchor')]:
        report,data=evaluate(path,events,protocol,config,mode,binary);reports[name]=report
        if data is not None:
            arrays[name]=data;np.savez_compressed(args.output/f'{name}.npz',**data)
        print(json.dumps({k:v for k,v in report.items() if k not in ('callback_source','callback_sha256')},indent=2),flush=True)
    np.testing.assert_allclose(arrays['success_version']['command'],arrays['current_version']['command'],atol=1e-12,rtol=0)
    assert reports['current_version']['outcome']=='exact_recorded_execution_reproduced'
    manifest=dict(source=str(args.events),sha256=sha(args.events),run_id=events[0]['run_id'],
        label='User-reported far-flap success; later head image confirms far-open scene',
        config_source=str(args.config),config_sha256=sha(args.config),
        native_header_sha256=sha(HERE/'native/timed_policy.hpp'),protocol_sha256=sha(Path(protocol.__file__)),
        cases=reports,guard_checks=guard_checks(sources['current_version'],events,protocol,config),limitations=[
            'No continuous three-camera images in the successful recording: image-to-action inference is not covered.',
            'Uses recorded measured feedback, timing, and consumed chunk sequence; not an interactive robot simulation.',
            'Heartbeat issue times are absent and modeled fresh; real DDS transport and SDK calls are not replayed.',
            'Legacy chunks without policy_key are tagged default for current stage-aware validation; original hashes/actions/timing are preserved.',
            'Old successful round was single-stage; visual two-stage completion cannot be reconstructed without its images.'])
    (args.output/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    base=arrays['current_version'];time=base['time']
    for ax,j in zip(axs[:2],(3,8)):
        ax.plot(time,np.rad2deg(base['recorded'][:,j]),label='Recorded successful command',lw=2)
        ax.plot(time,np.rad2deg(base['command'][:,j]),'--',label='Current pipeline replay',lw=1)
        if 'anchor_candidate' in arrays:ax.plot(time,np.rad2deg(arrays['anchor_candidate']['command'][:,j]),label='Anchor candidate',lw=1)
        ax.set_ylabel(('Left J4' if j==3 else 'Right J2')+' (deg)');ax.legend()
    axs[2].plot(time,base['fused'][:,14],label='Replayed suction prediction')
    axs[2].axhline(.6,ls='--',color='grey');axs[2].axhline(.4,ls=':',color='grey')
    axs[2].set_ylabel('Suction score');axs[2].set_xlabel('Seconds from active ServoJ');axs[2].legend()
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle('Fixed successful-run regression: production C++ math and Python callbacks')
    fig.tight_layout();fig.savefig(args.output/'replay.png',dpi=130)


if __name__=='__main__':main()
