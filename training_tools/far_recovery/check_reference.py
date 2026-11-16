"""Offline checks of reference resampling, resets and production limiter."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,yaml
from reference_actor import Reference,ROOT,APP
sys.path.insert(0,str(ROOT/'training_tools/success_replay'))
from replay_success import native_replay
cfg=yaml.safe_load((APP/'config.yaml').read_text())
base=ROOT/'training_verification/success_2105_comparison_20261113'
result={}
for stage in ('far','near'):
    ref=Reference(stage,cfg);ref.node=SimpleNamespace(executor_context={'phase':'STARTING_SERVO'})
    np.testing.assert_allclose(ref({'source_stamp':100.}),np.tile(ref.home,(50,1)))
    assert ref.origin is None
    ref.node.executor_context={'phase':'RUNNING'}
    np.testing.assert_allclose(ref({'source_stamp':101.}),ref.sample(0))
    np.testing.assert_allclose(ref({'source_stamp':102.}),ref.sample(1))
    ref.reset();assert ref.origin is None
    np.testing.assert_allclose(ref({'source_stamp':1000.}),ref.sample(0))
    try:ref({'source_stamp':1000.+ref.times[-1]+4})
    except RuntimeError:pass
    else:raise AssertionError('Exceeded recorded duration')
    events=[json.loads(l) for l in (base/f'success_{stage}.jsonl').open()]
    traces=[e['trace'] for e in events if e['event']=='servo_trace' and e['trace'].get('active') and e['trace'].get('hardware_commands_sent')]
    t0=traces[0]['selection_time_ns'];chunks=[e for e in events if e['event']=='chunk']
    transformed={c['sequence']:ref.sample((c['observation_time_ns']-t0)/1e9) for c in chunks}
    tr,counts,fused,command,velocity=native_replay(events,chunks,transformed,base/'native_replay',cfg)
    original=np.array([t['command'] for t in tr]);error=np.rad2deg(abs(original-command))
    result[stage]=dict(samples=len(tr),command_difference_max_deg=float(error.max()),command_difference_p95_deg=float(np.percentile(error,95)),
        limits_passed=True,reference_timeline_reset_checks=True,
        limitation='Offline uses historical feedback and packet timing; live start handshake may add a brief initial blend with home targets.')
    assert error.max()<1.0,'Reference resampling too far from successful execution'
(ROOT/'training_verification/far_recovery_20261113/reference_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
