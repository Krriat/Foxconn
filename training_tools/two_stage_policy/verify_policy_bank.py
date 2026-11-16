"""Compare both production loaders on frozen inputs. No ROS init or services."""
import json
import importlib.util
from pathlib import Path
import numpy as np
import yaml
from inference_node import load_policy_bank

root=Path('/home/jj/code/ACT')
config=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
_,bank=load_policy_bank(config)
data=np.load(root/'training_verification/restore_far_20261112/near_contact_audit/live_inputs.npz')
expected=np.load(root/'training_verification/restore_far_20261112/near_contact_audit/zero_latent_live.npz')
report={'scope':'No ROS initialization, publishers, services, or hardware; fixed six input replay','models':{}}
for key,name in (('far','baseline'),('near','candidate')):
    predict,reset,sha=bank[key]
    obs=lambda i:dict(state=data['state'][i].astype(np.float32),images=[data[c][i] for c in ('head','left_wrist','right_wrist')])
    for _ in range(5):predict(obs(0))
    reset()
    actual=np.stack([predict(obs(i)) for i in range(len(data['state']))])
    delta=float(np.abs(actual-expected[name]).max())
    # Compare against the exact original live loader in this same GPU process;
    # cuDNN/TF32 can select different kernels from a previous recorded process.
    source=root/'model_outputs/act_near_success_20261112/evidence'/('inference_before.py' if key=='far' else 'inference_candidate.py')
    spec=importlib.util.spec_from_file_location('archived_'+key,source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=dict(config,checkpoint_sha256={**config['checkpoint_sha256'],'act':sha})
    _,old_predict,old_reset,old_sha=module.load_runtime(cfg)
    assert old_sha==sha
    for _ in range(5):old_predict(obs(0))
    old_reset()
    reference=np.stack([old_predict(obs(i)) for i in range(len(data['state']))])
    same_process_delta=float(np.abs(actual-reference).max())
    assert same_process_delta<1e-6,(key,same_process_delta)
    report['models'][key]=dict(checkpoint_sha256=sha,max_absolute_difference_from_previous_process=delta,
        max_absolute_difference_from_archived_live_loader=same_process_delta)
assert bank['default'] is bank['far']
p=root/'training_verification/dual_act_deployment_20261112/policy_bank_check.json'
p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
