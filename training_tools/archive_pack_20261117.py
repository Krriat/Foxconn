"""Package explicit successful-run/model/runtime roots without robot calls."""
import hashlib,json,os,shutil,tarfile,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('/home/jj/code/ACT');REPO=Path('/home/jj/code/Foxconn-jetson-act-qwen-20261117')
OUT=Path(Path('/tmp/foxconn_export_path').read_text());TAG='snapshot-20261117-jetson-act-qwen'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(8*1024*1024):h.update(data)
    return h.hexdigest()


def index(folder):
    return {str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in sorted(folder.rglob('*')) if p.is_file() and not p.is_symlink()}


def clean(member):
    parts=Path(member.name).parts
    if '__pycache__' in parts or '.pytest_cache' in parts:return None
    if member.name.endswith(('.pyc','.pyo')):return None
    if 'model_comparison/models' in member.name:return None
    if member.issym() and (member.linkname.startswith('/') or '..' in Path(member.linkname).parts):
        raise ValueError('Unexpected external archive link: '+member.name)
    return member


def main():
    assets=OUT/'assets';assets.mkdir(exist_ok=True)
    models=OUT/'assets_staging/models';runtime=OUT/'assets_staging/runtime'
    for folder in (models,runtime):
        for cache in list(folder.rglob('__pycache__')):
            if cache.is_dir():shutil.rmtree(cache)
    # Every binary component is verified independently of the enclosing tar.
    for label,folder in (('models',models),('runtime',runtime)):
        manifest=index(folder);(REPO/'manifests'/f'{label}-files-20261117.json').write_text(json.dumps(manifest,indent=2)+'\n')
    expected={'far':'af60ccd4cfca37c420d1107e440c8fcb78904d22a28e802d39b4b5925a9055f9',
              'near':'ec6e21a2a4e8b1813f5e16926f1e1b15f27468627dd7e3841056f4c016174ad8'}
    bank=models/'devices/jetson/robot-policy-streams/jetson_act_runtime_20261116/models'
    for stage,value in expected.items():assert sha(bank/stage/'model.safetensors')==value
    assert sha(models/'model_outputs/task_brain_lora_20261112/adapter/adapter_model.safetensors')=='63e882d82b66a58da2269ce32f22dedc41b42959b2165d080a131ddeeabf0ade'
    specs=[('foxconn-successful-runs-20261117.tar.gz',ROOT,[Path('training_verification/no_right_hold_20261117_102134'),Path('training_verification/grasp_repair_20261117_095228')]),
           ('foxconn-models-20261117.tar.gz',models,None),
           ('foxconn-jetson-runtime-assets-20261117.tar.gz',runtime,None)]
    reports=[]
    for name,folder,paths in specs:
        started=time.monotonic();target=assets/name
        print(json.dumps({'event':'packing','asset':name}),flush=True)
        with tarfile.open(target,'w:gz',compresslevel=1) as tar:
            for relative in paths if paths is not None else sorted(p.relative_to(folder) for p in folder.iterdir()):
                tar.add(folder/relative,arcname=str(relative),filter=clean)
        names=[]
        with tarfile.open(target,'r:gz') as tar:
            for member in tar:
                if member.name.startswith('/') or '..' in Path(member.name).parts:raise ValueError('Unsafe archive path')
                if member.isfile():names.append(member.name)
        row={'name':name,'bytes':target.stat().st_size,'sha256':sha(target),'release':TAG,'members':len(names)}
        reports.append(row);print(json.dumps({'event':'packed',**row,'seconds':time.monotonic()-started}),flush=True)
    (REPO/'manifests/assets.json').write_text(json.dumps(reports,indent=2)+'\n')
    (REPO/'manifests/assets.sha256').write_text(''.join(f'{r["sha256"]}  {r["name"]}\n' for r in reports))
    (OUT/'archives.json').write_text(json.dumps(reports,indent=2)+'\n')
    original=json.loads((OUT/'collection.json').read_text())
    original.update(repository='Krriat/Foxconn',visibility='private',release_tag=TAG,
        capture_scope='Successful policy code/configuration; exported services may already be stopped for a new teaching session. No service was started/stopped by export.',
        current_control={'coordinate_far_right_arm':False,'require_manipulation_sweep_for_brain':True,'guard_far_suction_release':True,'execution_speed_scale':1.,'max_joint_acceleration_rad_s2':1.},
        test_results={'current_no_right_hold':{'far_seconds':43.88694167137146,'near_seconds':30.674965620040894,'passed':True},
                      'comparison_with_right_hold':{'far_seconds':62.168,'near_seconds':30.986,'passed':True}},
        inference_median_ms=108.442041,
        hardware_environment='Jetson Orin Nano 8GB, current JetPack/TensorRT 10.3/PyTorch environment; model engines are target-specific',
        ACT_weights_unchanged=True,LoRA_weights_unchanged=True)
    (REPO/'manifests/snapshot.json').write_text(json.dumps(original,indent=2)+'\n')
    print(json.dumps({'complete':True,'assets':reports}),flush=True)


if __name__=='__main__':main()
