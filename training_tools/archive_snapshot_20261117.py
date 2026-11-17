"""Read-only device export for the authorized Foxconn branch; never drives IO.

SSH authentication is supplied by existing ControlMaster sessions. No login
passwords, GitHub tokens, virtual environments, or host secrets are collected.
"""
import hashlib,json,shutil,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('/home/jj/code/ACT')
REPO=Path('/home/jj/code/Foxconn-jetson-act-qwen-20261117')
OUT=Path(Path('/tmp/foxconn_export_path').read_text())
JET='jetson@172.16.12.95';UMIE='umie@172.16.12.45'
JS='/tmp/act_jetson_resume_20261117.sock';US='/tmp/act_umie_archive_20261117.sock'
EXCLUDE=['.git/','__pycache__/','.pytest_cache/','.venv*/','backup*/','*backup*/','*before*/','*before*',
         'build/','install/','logs/','log/','evidence/','_rokae_log_/','*.log','*.jsonl','*.db3*',
         '*.avi','*.mp4','*.npz','*.npy','*.safetensors','*.engine','*.onnx','*.cache',
         '*.so','*.so.*','*.a','*.o','*.whl','*.deb','CONTINUATION.md','REMOTE_EDIT_GUIDE.md',
         'jetson_act_runtime_20261113/','grasp_candidate_20261117/']


def command(argv,**kw):
    return subprocess.run(argv,check=True,text=True,capture_output=True,**kw).stdout


def sync(source,dest,host=None,sock=None,filtered=True,dereference=False):
    dest.mkdir(parents=True,exist_ok=True)
    cmd=['rsync','-a'+('L' if dereference else '')]
    if filtered:
        for pattern in EXCLUDE:cmd+=['--exclude='+pattern]
    if host:cmd+=['-e',f'ssh -S {sock} -o BatchMode=yes'];source=host+':'+source
    command(cmd+[source,str(dest)+'/'])


def reset(path):
    assert path.is_relative_to(REPO/'devices') or path==REPO/'training_tools'
    if path.exists():shutil.rmtree(path)
    path.mkdir(parents=True)


def main():
    assert command(['git','-C',str(REPO),'branch','--show-current']).strip()=='jetson-act-qwen-20261117'
    (OUT/'assets_staging').mkdir(exist_ok=True)
    legacy=REPO/'manifests/baseline-20261116';legacy.mkdir(exist_ok=True)
    for f in (REPO/'manifests').iterdir():
        if f.is_file():shutil.copy2(f,legacy/f.name)
    (REPO/'docs').mkdir(exist_ok=True);shutil.copy2(REPO/'README.md',REPO/'docs/SNAPSHOT_20261116_BASELINE.md')
    apps={'pc':['robot-policy-streams'],'jetson':['robot-policy-streams','robot-camera-streams','robot-state-streams'],
          'umie':['robot-camera-streams','robot-relay-bridge','robot-policy-monitor']}
    for device,names in apps.items():
        for name in names:
            dest=REPO/'devices'/device/name;reset(dest)
            if device=='pc':sync(str(Path.home()/'.local/share'/name)+'/',dest)
            else:sync('/home/'+device+'/.local/share/'+name+'/',dest,JET if device=='jetson' else UMIE,JS if device=='jetson' else US)
            print('source',device,name,flush=True)
    reset(REPO/'training_tools');sync(str(ROOT/'training_tools')+'/',REPO/'training_tools')
    for relative in ('hardware/arm/xCoreROS/src','hardware/USR-M050'):
        sync('/home/jetson/AIDE_robot_main/'+relative+'/',REPO/'devices/jetson/robot_source'/relative,JET,JS)
    # The inherited SDK headers/docs are preserved. Current compiled-driver
    # source is additionally captured from the installed native/vendor tree.
    runtime=OUT/'assets_staging/runtime';models=OUT/'assets_staging/models'
    rt=runtime/'devices/jetson'
    for folder in (runtime,models):folder.mkdir(parents=True,exist_ok=True)
    # Move any extensionless executable that survived source filters to assets.
    for f in list((REPO/'devices').rglob('*')):
        if not f.is_file() or f.is_symlink():continue
        with f.open('rb') as stream:magic=stream.read(4)
        if magic==b'\x7fELF':
            dest=runtime/f.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);f.unlink()
    remote_app='/home/jetson/.local/share/robot-policy-streams/'
    sync(remote_app+'native/build/robot_timed_policy_driver',rt/'robot-policy-streams/native/build',JET,JS,False)
    sync('/home/jetson/.local/share/robot-state-streams/build/robot_hardware_state',rt/'robot-state-streams/build',JET,JS,False)
    for name in ('far.engine','far.engine.json','near.engine','near.engine.json'):
        sync(remote_app+'jetson_act_runtime_20261116/'+name,rt/'robot-policy-streams/jetson_act_runtime_20261116',JET,JS,False)
    for relative in ('hardware/USR-M050/build/libm050.so','hardware/arm/xCoreSDK-v0.7.1.ar_6/lib/'):
        src='/home/jetson/AIDE_robot_main/'+relative
        target=rt/'robot_source'/relative
        sync(src,target if relative.endswith('/') else target.parent,JET,JS,False,True)
    sync('/home/jetson/AIDE_robot_main/hardware/arm/xCoreROS/install/',rt/'robot_source/hardware/arm/xCoreROS/install',JET,JS,False,True)
    sync('/home/jetson/act_model_benchmark_20261112/runtime_libs/',rt/'act_model_benchmark_20261112/runtime_libs',JET,JS,False,True)
    for stage in ('far','near'):
        sync(remote_app+'jetson_act_runtime_20261116/models/'+stage+'/',models/'devices/jetson/robot-policy-streams/jetson_act_runtime_20261116/models'/stage,JET,JS,False)
    sync(str(ROOT/'model_outputs/task_brain_lora_20261112/adapter')+'/',models/'model_outputs/task_brain_lora_20261112/adapter',filtered=False)
    print('models/runtime collected',flush=True)
    # Full successful-run evidence is archived as-is; temporary model symlinks
    # used by the offline comparison are omitted, actual weights are above.
    evidence=REPO/'evidence/20261117';evidence.mkdir(parents=True,exist_ok=True)
    cases={'current_no_right_hold':ROOT/'training_verification/no_right_hold_20261117_102134/trial02_no_right_hold',
           'comparison_with_right_hold':ROOT/'training_verification/grasp_repair_20261117_095228/coordination01'}
    for label,case in cases.items():
        dest=evidence/label;dest.mkdir(exist_ok=True)
        for name in ('test_report.json','waiting_breakdown.json','final_idle_status.json','final_status.json'):
            if (case/name).exists():shutil.copy2(case/name,dest/name)
        for name in ('before.jpg','latest.jpg'):
            if (case/'live01'/name).exists():shutil.copy2(case/'live01'/name,dest/name)
    for file in ('comparison.json','test_setup.json','jetson_before_config.yaml','jetson_after_config.yaml'):
        src=ROOT/'training_verification/no_right_hold_20261117_102134'/file
        if src.exists():shutil.copy2(src,evidence/src.name)
    for file in ('progress_replay.json','regression_checks.json','final_review.json'):
        src=ROOT/'training_verification/completion_contact_fix_20261117_090845'/file
        if src.exists():shutil.copy2(src,evidence/('earlier_'+src.name))
    report={'captured_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'branch':'jetson-act-qwen-20261117','parent_commit':'a7e15b19a74f064e8d9fde2bf03753845b68cf04',
            'sources':apps,'robot_motion_requested_by_export':False,
            'full_data_roots':['training_verification/no_right_hold_20261117_102134','training_verification/grasp_repair_20261117_095228'],
            'data_omitted':['temporary model_comparison/models symlinks; model weights are in a separate asset',
                            'full original 28GB dataset','Qwen base weights; public base identifier/config are recorded separately',
                            'SSH/GitHub secrets, virtual environments, caches and unrelated old logs']}
    (OUT/'collection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
