"""Loopback protocol/causality and blocked-DDS tests, with no robot access."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from xcore_teach_recording.local_feedback import PACKET,decode_packet,LocalFeedbackReceiver
RECORDING=Path(__file__).resolve().parents[1]/"scripts/recording"
sys.path.insert(0,str(RECORDING))
import record_xcore_teach_suction as rec

BINARY=ROOT/"hardware/arm/xCoreROS/build/rokae_driver/test_local_joint_feedback"


def packet(seq=1,side=0,source=0,session=123,stamp=None,mono=None,joints=None):
    return PACKET.pack(b"AIDEJNT1",side,source,0,os.getpid(),session,seq,
        time.time_ns() if stamp is None else stamp,
        time.monotonic_ns() if mono is None else mono,0,0,*(joints or [.1]*7))


def wait_until(test,seconds=1.):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if test():return
        time.sleep(.005)
    assert test()


def test_cpp_and_python_wire_layout(tmp_path):
    if not BINARY.exists():pytest.skip("build test_local_joint_feedback for cross-language fixture")
    path=tmp_path/"packet.bin"
    subprocess.run([str(BINARY),"--fixture",str(path)],check=True)
    p=decode_packet(path.read_bytes())
    assert (p.side,p.source,p.pid,p.session,p.sequence)==("right","drag",123,456,789)
    assert p.stamp_ns==123456789012345 and p.mono_ns==98765432101234
    assert p.send_errors==2 and p.ros_overwrites==9
    assert p.joints==pytest.approx([.1,.2,.3,.4,.5,.6,.7])


def test_rejects_bad_or_non_measured_packets():
    for b in [b"bad",packet(joints=[math.nan]*7),packet(joints=[10.]*7),packet(side=2),packet(seq=0)]:
        with pytest.raises(ValueError):decode_packet(b)


def test_loss_reordering_and_producer_restart_are_reported():
    rows=[];r=LocalFeedbackReceiver(lambda p,at:rows.append(p),port=0)
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        def send(data):s.sendto(data,("127.0.0.1",r.port))
        send(packet(seq=1));send(packet(seq=3));send(packet(seq=2))
        wait_until(lambda:len(rows)==2)
        stats=r.diagnostics()["streams"]["left.drag"]
        assert stats["missing_sequences"]==1
        wait_until(lambda:r.diagnostics()["streams"]["left.drag"]["out_of_order"]==1)
        send(packet(seq=4,session=999))
        wait_until(lambda:r.error is not None)
        assert len(rows)==2
    finally:r.close();s.close()


def test_stale_packets_are_not_accepted_as_fresh():
    rows=[];r=LocalFeedbackReceiver(lambda p,at:rows.append(p),port=0,max_age=.05)
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        s.sendto(packet(mono=time.monotonic_ns()-1000000000),("127.0.0.1",r.port))
        wait_until(lambda:r.stale_packets==1)
        assert not rows
        with pytest.raises(OSError):LocalFeedbackReceiver(lambda p,at:None,port=r.port)
    finally:r.close();s.close()


def test_mode_selection_preserves_sdk_sampling_time():
    args=rec.parse_args([]);inputs=rec.KeyboardInputs(rec.SuctionOutputs(None,report=lambda _:None),args)
    p=decode_packet(packet(mono=10000000000,stamp=20000000000))
    assert inputs.push_local_joint_packet(p,10.2,20.2)
    sample=inputs.buffers["left"][-1]
    assert sample.stamp==20. and sample.received==10.
    inputs.set_return_recording(True)
    assert not inputs.push_local_joint_packet(p,10.2,20.2)
    returned=decode_packet(packet(source=1,mono=10000000000,stamp=20000000000))
    assert inputs.push_local_joint_packet(returned,10.2,20.2)


def test_local_stream_survives_1100ms_blocked_publisher():
    if not BINARY.exists():pytest.skip("build test_local_joint_feedback for blocked publisher integration")
    samples={"left":[],"right":[]}
    r=LocalFeedbackReceiver(lambda p,at:samples[p.side].append(p),port=0)
    try:
        process=subprocess.run([str(BINARY),"--simulate","3"],
            env=dict(os.environ,AIDE_JOINT_FEEDBACK_PORT=str(r.port)),capture_output=True,text=True,timeout=8)
        assert process.returncode==0,process.stderr
        stats=r.diagnostics()
        for side in samples:
            values=samples[side]
            assert len(values)>=280
            assert max(b.mono_ns-a.mono_ns for a,b in zip(values,values[1:]))<80000000
            assert stats["streams"][side+".drag"]["missing_sequences"]==0
        assert any(v["ros_queue_overwrites"]>0 for v in stats["streams"].values())
        assert not r.error
    finally:r.close()
