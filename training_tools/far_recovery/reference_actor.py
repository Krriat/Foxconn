"""Explicit recorded-trajectory diagnostic. NOT learned-policy inference.

Uses fresh observations, current run IDs, home verification, and the unchanged
production executor/driver guards. Source ACT hashes identify the archived runs.
Every emitted packet is additionally marked as a recorded reference.
"""
import argparse,gc,hashlib,json,signal,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
import yaml
ROOT=Path('/home/jj/code/ACT');APP=Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0,str(APP))
from inference_node import Inference
from protocol import validate_config,validate_joints,encode

def joint_vector(msg,side):
    if len(msg.name)!=len(msg.position) or len(set(msg.name))!=len(msg.name):
        raise ValueError('Invalid JointState names')
    lookup=dict(zip(msg.name,msg.position))
    value=np.array([lookup[f'{side}_joint_{i}'] for i in range(1,8)],np.float32)
    if not np.isfinite(value).all():raise ValueError('Nonfinite joint state')
    return value

class Reference:
    def __init__(self,stage,cfg):
        root=ROOT/'training_verification/success_2105_comparison_20261113'
        self.path=root/f'success_{stage}_analysis.npz'
        data=np.load(self.path)
        self.times=data['time'].copy();self.targets=data['fused'].copy()
        self.commands=data['command'].copy()
        self.stage=stage;self.origin=None;self.node=None
        self.speed_scale=float(cfg.get('execution_speed_scale',1.))
        if not 1.<=self.speed_scale<=1.5:raise ValueError('Invalid reference speed scale')
        self.home=np.array(cfg['home_left']+cfg['home_right']+[0,0])
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        assert self.targets.shape==(len(self.times),16)
        assert np.all(np.diff(self.times)>0) and np.isfinite(self.targets).all()
        for q in self.targets:validate_joints(q[:14].tolist(),cfg['joint_limit_margin_rad'],cfg.get('joint_upper_limits_rad'))
        assert np.max(abs(self.commands[0]-self.home[:14]))<.01
        assert np.max(abs(self.targets[:,14:]))<=1.25
    def reset(self):self.origin=None
    def sample(self,elapsed):
        times=elapsed+np.arange(50)*.05
        q=np.stack([np.interp(times,self.times,self.targets[:,j]) for j in range(14)],axis=1)
        idx=np.clip(np.searchsorted(self.times,times,side='right')-1,0,len(self.times)-1)
        return np.concatenate([q,self.targets[idx,14:]],axis=1)
    def __call__(self,observation):
        phase=self.node.executor_context.get('phase') if self.node else 'IDLE'
        if phase not in ('RUNNING','PREVIEW'):
            return np.tile(self.home,(50,1))
        if self.origin is None:self.origin=observation['source_stamp']
        # The archived trajectory also has an episode clock. Advance its base
        # at the requested speed; native fusion separately maps the local
        # 50-ms action horizon. Raw observation stamps remain real host time.
        elapsed=(observation['source_stamp']-self.origin)*self.speed_scale
        if elapsed < -.001 or elapsed>self.times[-1]+3:
            raise RuntimeError('Recorded reference duration exceeded')
        return self.sample(elapsed)

class MarkedPublisher:
    def __init__(self,publisher,references):self.publisher=publisher;self.references=references
    def publish(self,message):
        data=json.loads(message.data);key=data.get('policy_key','far')
        ref=self.references['far' if key=='default' else key]
        data.update(is_model_inference=False,diagnostic_mode='recorded_success_reference',
            source_reference_sha256=ref.sha,source_reference_path=str(ref.path),
            reference_elapsed_seconds=(data['observation_time_ns']/1e9-ref.origin)*ref.speed_scale if ref.origin is not None else None,
            reference_wall_elapsed_seconds=(data['observation_time_ns']/1e9-ref.origin) if ref.origin is not None else None)
        self.publisher.publish(String(data=encode(data)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    cfg=validate_config(yaml.safe_load((APP/'config.yaml').read_text()))
    expected={'far':'0d71f2e37a180b2e829bc7bf423bbe8171945c3afa327f8d08f2a72297d00d93',
        'near':'ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8'}
    assert {k:v['sha256'] for k,v in cfg['act_stage_checkpoints'].items()}==expected,'Restore source checkpoints before reference test'
    references={k:Reference(k,cfg) for k in expected}
    report={k:dict(artifact_sha256=r.sha,duration_seconds=float(r.times[-1]),
        execution_speed_scale=r.speed_scale,scaled_duration_seconds=float(r.times[-1])/r.speed_scale,
        max_recorded_command_speed=float(np.max(abs(np.diff(r.commands,axis=0)/np.diff(r.times)[:,None]))),
        scaled_max_recorded_command_speed=float(np.max(abs(np.diff(r.commands,axis=0)/np.diff(r.times)[:,None])))*r.speed_scale) for k,r in references.items()}
    print(json.dumps(dict(diagnostic_mode='recorded_success_reference',is_model_inference=False,references=report)),flush=True)
    if a.validate_only:return
    cfg=dict(cfg,inference_runtime='recorded_reference',act_backend='archived_fused_targets')
    policies={k:(r,r.reset,expected[k]) for k,r in references.items()};policies['default']=policies['far']
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=Inference(cfg,SimpleNamespace(joint_vector=joint_vector),*policies['default'],policies=policies)
    for r in references.values():r.node=node
    node.actions=MarkedPublisher(node.actions,references)
    node.status.update(is_model_inference=False,diagnostic_mode='recorded_success_reference')
    end=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:end.set())
    try:
        while rclpy.ok() and not end.is_set():rclpy.spin_once(node,timeout_sec=.05)
    finally:
        node.stop_event.set();node.thread.join(timeout=5)
        if node.gc_suspended and node.gc_initially_enabled:gc.enable()
        gc.callbacks.remove(node.on_gc);node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
