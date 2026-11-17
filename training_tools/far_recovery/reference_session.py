"""Bounded reference diagnostic; restore real ACT inference on exit."""
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/jj/code/ACT');APP=Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0,str(ROOT))
from training_tools.near_limit_fix.run_candidate_once import status,request_stop
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--start-service',choices=('start_sequence','start_near'),default='start_sequence')
p.add_argument('--seconds',type=float,default=None);a=p.parse_args()
if a.seconds is not None and not 0.<a.seconds<=75.:raise ValueError('Invalid bounded reference test duration')
before=status();assert before['phase']=='IDLE' and before['readback_suction']==[False,False]
assert not a.output.exists()
log=a.output.with_suffix('.actor.log').open('w')
subprocess.run(['systemctl','--user','stop','robot-policy-inference.service'],check=True)
actor=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('reference_actor.py'))],stdout=log,stderr=subprocess.STDOUT)
try:
    subprocess.run([sys.executable,str(Path(__file__).with_name('recorded_test.py')),'--output',str(a.output),
        '--seconds',str(a.seconds) if a.seconds is not None else ('38' if a.start_service=='start_near' else '45'),'--expected-runtime','recorded_reference',
        '--start-service',a.start_service],check=True)
finally:
    try:
        s=status()
        if s['phase']!='IDLE' or s['readback_suction']!=[False,False]:request_stop()
    finally:
        if actor.poll() is None:actor.send_signal(signal.SIGTERM)
        actor.wait(timeout=10);log.close()
    deadline=time.monotonic()+12
    while True:
        s=status()
        if s['phase']=='IDLE' and s['readback_suction']==[False,False]:break
        if time.monotonic()>deadline:raise RuntimeError('Unable to confirm stop; ACT not restarted')
        if s['phase']=='FAULT':request_stop()
    subprocess.run([str(APP/'start.sh'),'inference'],check=True)
    print(json.dumps(dict(reference_actor_stopped=True,actual_act_service_restarted=True,phase=s['phase'],suction=s['readback_suction'])),flush=True)
