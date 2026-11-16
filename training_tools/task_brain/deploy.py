"""Install keyed LoRA brain endpoints while IDLE. Never starts a robot task."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import yaml
from training_tools.near_limit_fix.run_candidate_once import status

ROOT=Path(__file__).resolve().parents[2]
APP=Path.home()/".local/share/robot-policy-streams"
REMOTE="/home/jetson/.local/share/robot-policy-streams"
EVIDENCE=ROOT/"training_verification/task_brain_20261112"
SSH=["ssh","-o","ControlPath=/tmp/act_restore_far.sock","-o","BatchMode=yes","jetson@172.16.12.95"]


def remote(command,**kw):
    return subprocess.run(SSH+[command],check=True,capture_output=True,text=True,timeout=30,**kw).stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--install",action="store_true",required=True)
    parser.parse_args()
    release=json.loads((EVIDENCE/"release_manifest.json").read_text())
    if not release["offline_checks_passed"]:raise ValueError("Offline qualification incomplete")
    adapter=ROOT/"model_outputs/task_brain_lora_20261112/adapter/adapter_model.safetensors"
    sha=hashlib.sha256(adapter.read_bytes()).hexdigest()
    if sha!=release["adapter_sha256"]:raise ValueError("Adapter changed after evaluation")
    before=status()
    if before["phase"]!="IDLE" or before["readback_suction"]!=[False,False]:
        raise ValueError("Executor must already be IDLE with both suction channels off")
    source=ROOT/"training_tools/two_stage_policy"
    tag="backup_before_lora_brain_"+time.strftime("%Y%m%d_%H%M%S")
    backup=APP/tag;backup.mkdir();remote("mkdir "+shlex.quote(REMOTE+"/"+tag))
    names=("executor_node.py","brain_protocol.py","config.yaml")
    local_old={n:(APP/n).read_text() if (APP/n).exists() else None for n in names}
    remote_old={}
    for n,value in local_old.items():
        if value is not None:(backup/n).write_text(value)
    for n in names:
        path=REMOTE+"/"+n
        present=remote("test ! -f "+shlex.quote(path)+" || echo exists").strip()
        remote_old[n]=remote("cat "+shlex.quote(path)) if present else None
        if present:remote("cp "+shlex.quote(path)+" "+shlex.quote(REMOTE+"/"+tag+"/"+n))
    local_config=yaml.safe_load(local_old["config.yaml"]);remote_config=yaml.safe_load(remote_old["config.yaml"])
    for config in (local_config,remote_config):config["brain_adapter_sha256"]=sha
    try:
        remote("systemctl --user stop robot-policy-executor.service")
        for n in ("executor_node.py","brain_protocol.py"):
            shutil.copyfile(source/n,APP/n)
            remote("cat > "+shlex.quote(REMOTE+"/"+n),input=(source/n).read_text())
        (APP/"config.yaml").write_text(yaml.safe_dump(local_config,sort_keys=False))
        remote("cat > "+shlex.quote(REMOTE+"/config.yaml"),input=yaml.safe_dump(remote_config,sort_keys=False))
        remote("python3 -m py_compile "+shlex.quote(REMOTE+"/executor_node.py")+" "+shlex.quote(REMOTE+"/brain_protocol.py"))
        remote(REMOTE+"/start.sh executor")
        after=status()
        if after["phase"]!="IDLE" or after["readback_suction"]!=[False,False]:
            raise RuntimeError("Unexpected post-install executor state")
        assert after.get("brain_adapter_sha256")==sha
    except BaseException:
        remote("systemctl --user stop robot-policy-executor.service")
        for n,value in local_old.items():
            if value is None:(APP/n).unlink(missing_ok=True)
            else:(APP/n).write_text(value)
        for n,value in remote_old.items():
            if value is None:remote("rm -f "+shlex.quote(REMOTE+"/"+n))
            else:remote("cat > "+shlex.quote(REMOTE+"/"+n),input=value)
        remote(REMOTE+"/start.sh executor")
        raise
    result={"local_backup":str(backup),"remote_backup":REMOTE+"/"+tag,
        "adapter_sha256":sha,"before":before,"after":after,
        "motion_services_called":[],"robot_task_started":False,"native_driver_changed":False}
    (EVIDENCE/"deployment.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("before","after")},indent=2))


if __name__=="__main__":main()
