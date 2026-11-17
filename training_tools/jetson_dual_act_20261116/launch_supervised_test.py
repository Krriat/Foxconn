"""Run an authorized recorder under systemd; stop robot if supervision exits.

The service survives the lifetime of an interactive tool/terminal. A start
gate can keep it read-only until the latest camera views have been checked.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-gate',type=Path)
    a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required for physical test')
    a.output.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent;unit='act-supervised-'+a.output.name
    argv=['/usr/bin/python3','-u',str(here/'run_live_once.py'),'--execute','--output',str(a.output/'live01')]
    if a.start_gate:argv+=['--start-gate',str(a.start_gate)]
    command=('source /opt/ros/humble/setup.bash\n'
        'export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n'
        'exec '+shlex.join(argv))
    subprocess.run(['systemd-run','--user','--collect','--unit',unit,
        '--property=Restart=no','--property=KillSignal=SIGINT','--property=TimeoutStopSec=25s',
        '--property=StandardOutput=append:'+str(a.output/'live01.log'),
        '--property=StandardError=inherit',
        '--property=ExecStopPost=/usr/bin/timeout 12s /bin/bash '+str(here/'stop_sequence.sh'),
        '--working-directory='+str(here.parents[1]),'/bin/bash','-c',command],check=True)
    (a.output/'systemd_unit.txt').write_text(unit+'.service\n')
    print(unit,flush=True)


if __name__=='__main__':main()
