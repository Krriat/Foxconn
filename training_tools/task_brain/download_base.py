"""Download the official Qwen 9B Safetensors checkpoint; resume and verify SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/"model_outputs/base/Qwen3.5-9B"
CHUNK = 64*1024*1024


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    manifest=DEST/"source_manifest.json"
    if manifest.exists():data=json.loads(manifest.read_text())
    else:
        url="https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-9B/repo/files?Revision=master&Recursive=true"
        original=json.load(urllib.request.urlopen(url,timeout=30))["Data"]
        data={"model":"Qwen/Qwen3.5-9B","revision":original["LatestCommitter"]["Id"] or "460979c3",
              "files":[r for r in original["Files"] if r["Type"]=="blob" and not r["Name"].startswith(".")]}
        # The API may omit the head Id. Freeze the revision associated with the
        # reviewed file manifest instead of using a mutable branch for downloads.
        if not data["revision"]:raise RuntimeError("Missing fixed source revision")
        manifest.write_text(json.dumps(data,indent=2)+"\n")
    tasks=[]
    for r in data["files"]:
        target=DEST/r["Path"]
        if target.exists() and target.stat().st_size==r["Size"] and sha(target)==r["Sha256"]:
            continue
        for i,start in enumerate(range(0,r["Size"],CHUNK)):
            end=min(start+CHUNK,r["Size"])-1
            tasks.append((r,i,start,end))
    chunks=DEST/".download_chunks";chunks.mkdir(exist_ok=True)
    def part(task):
        r,i,start,end=task;p=chunks/(r["Name"]+f".{i:04d}")
        if not p.exists() or p.stat().st_size!=end-start+1:
            url=f'https://modelscope.cn/models/Qwen/Qwen3.5-9B/resolve/{data["revision"]}/{r["Path"]}'
            temporary=p.with_suffix(p.suffix+".partial")
            subprocess.run(["curl","-fsSL","--retry","4","--retry-all-errors","--connect-timeout","20",
                "--max-time","600","--range",f"{start}-{end}","-o",str(temporary),url],check=True)
            if temporary.stat().st_size!=end-start+1:raise RuntimeError("Invalid byte range: "+str(p))
            temporary.replace(p)
        return end-start+1
    started=time.monotonic();last=0.;done=0;total=sum(t[3]-t[2]+1 for t in tasks)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(part,t) for t in tasks]):
            done+=f.result();now=time.monotonic()
            if now-last>10 or done==total:
                print(json.dumps({"bytes":done,"total":total,"percent":round(done/max(total,1)*100,1),
                                  "MB_s":round(done/(now-started)/1e6,1)}),flush=True);last=now
    for r in data["files"]:
        parts=[t for t in tasks if t[0]["Path"]==r["Path"]]
        if not parts:continue
        target=DEST/r["Path"];temp=target.with_suffix(target.suffix+".assembling")
        with temp.open("wb") as f:
            for _,i,_,_ in parts:
                with (chunks/(r["Name"]+f".{i:04d}")).open("rb") as source:shutil.copyfileobj(source,f)
        if temp.stat().st_size!=r["Size"] or sha(temp)!=r["Sha256"]:
            raise RuntimeError("Source checksum mismatch: "+r["Path"])
        temp.replace(target)
        for _,i,_,_ in parts:(chunks/(r["Name"]+f".{i:04d}")).unlink()
        print("Verified "+r["Path"],flush=True)
    print(json.dumps({"complete":True,"model":data["model"],"revision":data["revision"]}),flush=True)


if __name__=="__main__":main()
