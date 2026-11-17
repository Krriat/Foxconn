"""Install only from fresh IDLE/OFF; no motion, home, or IO service calls."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from training_tools.near_limit_fix.run_candidate_once import status


def run(argv,**kw):
    return subprocess.run(argv,check=True,capture_output=True,text=True,timeout=45,**kw).stdout


def remote(host,sock,code):
    return run(['ssh','-S',sock,'-o','BatchMode=yes',host,
                'python3 -c '+shlex.quote(code)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--keep-brain',action='store_true',help='No brain source/model change in this deployment')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    before=status()
    assert before['phase']=='IDLE' and before['readback_suction']==[False,False],before
    jet=('jetson@172.16.12.95','/tmp/act_jetson_resume_20261117.sock')
    umie=('umie@172.16.12.45','/tmp/act_umie_resume_20261117.sock')
    stamp=args.output.parent.name+'_'+args.output.name
    jet_backup=f'/home/jetson/.local/share/robot-policy-streams/backup_{stamp}'
    umie_backup=f'/home/umie/.local/share/robot-camera-streams/backup_{stamp}'
    prepare='''
from pathlib import Path
import shutil,json
root=Path.home()/'.local/share/robot-policy-streams';backup=Path(BACKUP)
backup.mkdir(exist_ok=False)
for name in ('config.yaml','executor_node.py','manipulation_progress.py','suction_phase_guard.py','far_arm_coordination.py'):
 if (root/name).exists():shutil.copy2(root/name,backup/name)
print((root/'config.yaml').read_text())
'''.replace('BACKUP',repr(jet_backup))
    (args.output/'jetson_before.yaml').write_text(remote(*jet,prepare))
    xml=base64.b64encode((Path(__file__).with_name('fastdds_wrist_transport.xml')).read_bytes()).decode()
    camera_changed=remote(*umie,f'''
from pathlib import Path
import shutil,base64
root=Path.home()/'.local/share/robot-camera-streams';backup=Path({umie_backup!r});backup.mkdir(exist_ok=False)
for name in ('run_camera.sh','fastdds_wrist_transport.xml'):
 if (root/name).exists():shutil.copy2(root/name,backup/name)
profile=root/'fastdds_wrist_transport.xml';data=base64.b64decode({xml!r})
changed=not profile.exists() or profile.read_bytes()!=data
profile.write_bytes(data)
path=root/'run_camera.sh';text=path.read_text()
original=text
if 'export FASTRTPS_DEFAULT_PROFILES_FILE="$camera_dir/fastdds_wrist_transport.xml"' not in text:
 text=text.replace('source /opt/ros/humble/setup.bash','export FASTRTPS_DEFAULT_PROFILES_FILE="$camera_dir/fastdds_wrist_transport.xml"\\nsource /opt/ros/humble/setup.bash')
path.write_text(text)
print(int(changed or original!=text))
''').strip()=='1'
    try:
        run(['ssh','-S',jet[1],'-o','BatchMode=yes',jet[0],'systemctl --user stop robot-policy-executor.service'])
        remote(*jet,'''
from pathlib import Path
import os,shutil,yaml
root=Path.home()/'.local/share/robot-policy-streams';stage=Path('/tmp/act_completion_guard_20261117')
for name in ('executor_node.py','manipulation_progress.py','suction_phase_guard.py','far_arm_coordination.py'):
 temp=root/(name+'.pending');shutil.copy2(stage/name,temp);os.replace(temp,root/name)
p=root/'config.yaml';cfg=yaml.safe_load(p.read_text())
assert cfg['execution_speed_scale']==1. and cfg['execution_speed_profile']=='uniform' and cfg['max_joint_acceleration_rad_s2']==1.
cfg['require_manipulation_sweep_for_brain']=True
cfg['guard_far_suction_release']=True
cfg.setdefault('coordinate_far_right_arm',False)
temp=p.with_suffix('.pending');temp.write_text(yaml.safe_dump(cfg,sort_keys=False));os.replace(temp,p)
''')
        run(['ssh','-S',jet[1],'-o','BatchMode=yes',jet[0],'~/.local/share/robot-policy-streams/start.sh executor'])
        if camera_changed:
            run(['ssh','-S',umie[1],'-o','BatchMode=yes',umie[0],'systemctl --user restart robot-wrist-cameras.service'])
        if not args.keep_brain:
            run(['systemctl','--user','restart','robot-policy-brain.service'])
    except BaseException:
        run(['ssh','-S',jet[1],'-o','BatchMode=yes',jet[0],'systemctl --user stop robot-policy-executor.service'])
        for host,sock,folder,backup,unit in ((*jet,'robot-policy-streams',jet_backup,'robot-policy-executor'),(*umie,'robot-camera-streams',umie_backup,'robot-wrist-cameras')):
            remote(host,sock,f'from pathlib import Path;import shutil;root=Path.home()/".local/share/{folder}";backup=Path({backup!r});[shutil.copy2(f,root/f.name) for f in backup.iterdir() if f.is_file()]')
            command=('~/.local/share/robot-policy-streams/start.sh executor' if unit=='robot-policy-executor' else 'systemctl --user restart '+unit+'.service')
            run(['ssh','-S',sock,'-o','BatchMode=yes',host,command])
        raise
    report=dict(installed=True,robot_motion_requested=False,execution_speed_scale=1.,
                ACT_weights_changed=False,LoRA_weights_changed=False,limits_changed=False,
                guard='LoRA visual completion AND measured manipulation sweep finished',
                jetson_backup=jet_backup,umie_backup=umie_backup)
    (args.output/'deployment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
