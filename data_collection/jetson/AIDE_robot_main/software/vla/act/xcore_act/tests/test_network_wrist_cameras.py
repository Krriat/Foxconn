"""Network timing, image integrity and delayed pairing; no robot/USB access."""
from pathlib import Path
import socket
import struct
import sys
import json
import tempfile
import threading
from unittest.mock import patch
import unittest

import numpy as np

RECORDING=Path(__file__).resolve().parents[1]/'scripts/recording'
sys.path.insert(0,str(RECORDING/'network_cameras'))
sys.path.insert(0,str(RECORDING))
from protocol import MAGIC, MAX_PAYLOAD, PREFIX, receive_packet, send_packet
from clock_sync import CaptureTimestampGuard, ClockEstimator
from pc_camera_server import FrameHub, encode_rgb
from jetson_camera_bridge import decode_image
from capture_trace import CaptureTrace
from pc_camera_server import CameraServer, parse_args as parse_pc_args
import record_xcore_teach_suction as rec


class NetworkTimingTest(unittest.TestCase):
    def exchange(self, clock, i=0, offset=250_000_000, forward=3_000_000, reverse=5_000_000, epoch='pc1'):
        t1=1_700_000_000_000_000_000+i*1_000_000_000
        t2=t1+offset+forward;t3=t2+200_000;t4=t1+forward+200_000+reverse
        return clock.add(t1,t2,t3,t4,t4-t1,epoch,at=i*.2)

    def test_clock_correction_retains_network_delay_and_bounds_asymmetry(self):
        clock=ClockEstimator()
        for i in range(4):self.assertTrue(self.exchange(clock,i))
        estimate=clock.estimate(.6)
        self.assertEqual(estimate.offset_ns,249_000_000)
        self.assertLessEqual(abs(estimate.offset_ns-250_000_000),estimate.uncertainty_ns)
        capture_jetson=1_700_000_000_000_000_000
        corrected=capture_jetson+250_000_000-estimate.offset_ns
        arrival=capture_jetson+120_000_000
        self.assertEqual(arrival-corrected,119_000_000)

    def test_clock_staleness_epoch_change_and_time_steps_stop_alignment(self):
        clock=ClockEstimator()
        for i in range(4):self.exchange(clock,i)
        with self.assertRaises(ValueError):clock.estimate(4.)
        self.exchange(clock,4,epoch='pc2')
        with self.assertRaises(ValueError):clock.estimate(.8)
        for i in range(5,8):self.exchange(clock,i,epoch='pc2')
        self.assertEqual(clock.estimate(1.4).epoch,'pc2')
        self.exchange(clock,8,offset=800_000_000,epoch='pc2')
        with self.assertRaises(ValueError):clock.estimate(1.6)

    def test_high_rtt_and_local_clock_step_are_not_valid_samples(self):
        clock=ClockEstimator()
        self.assertFalse(self.exchange(clock,forward=30_000_000))
        self.assertFalse(clock.add(100_000_000,101_000_000,102_000_000,112_000_000,1_000_000,'pc'))
        self.assertEqual(len(clock.samples),0)

    def test_camera_clock_regression_requires_fresh_stable_history(self):
        guard=CaptureTimestampGuard(.1)
        base=1_700_000_000_000_000_000
        for i in range(10):
            accepted=guard.accept(base+i*33_333_333,base+i*33_333_333+8_000_000,i/30,'epoch')
        self.assertTrue(accepted)
        self.assertFalse(guard.accept(base,base+350_000_000,.35,'epoch'))
        self.assertFalse(guard.accept(base+400_000_000,base+408_000_000,.4,'epoch'))
        self.assertFalse(guard.accept(base+2_000_000_000,base+500_000_000,.5,'epoch'))

    def test_fragmented_packet_and_eof(self):
        a,b=socket.socketpair()
        self.addCleanup(a.close);self.addCleanup(b.close)
        header=b'{"type":"frame"}';data=PREFIX.pack(MAGIC,len(header),3)+header+b'abc'
        thread=threading.Thread(target=lambda:[a.sendall(bytes([value])) for value in data])
        thread.start()
        self.assertEqual(receive_packet(b),({'type':'frame'},b'abc'))
        thread.join();a.close()
        with self.assertRaises(EOFError):receive_packet(b)

    def test_oversized_packet_is_rejected_before_allocation(self):
        a,b=socket.socketpair()
        try:
            a.sendall(PREFIX.pack(MAGIC,2,MAX_PAYLOAD+1))
            with self.assertRaises(ValueError):receive_packet(b)
        finally:a.close();b.close()

    def test_lossless_codecs_preserve_pixels_and_bad_shape_is_rejected(self):
        rgb=np.random.default_rng(8).integers(0,256,(24,32,3),dtype=np.uint8)
        for codec in ('raw','png','jpeg'):
            payload=encode_rgb(rgb,codec,95)
            image=decode_image(dict(width=32,height=24,encoding='rgb8',codec=codec),payload,32,24)
            if codec!='jpeg':np.testing.assert_array_equal(image,rgb)
            with self.assertRaises(ValueError):
                decode_image(dict(width=16,height=48,encoding='rgb8',codec=codec),payload,32,24)

    def test_slow_client_cannot_grow_capture_queue(self):
        hub=FrameHub()
        for _ in range(20):hub.offer({},b'rgb')
        self.assertEqual(len(hub.frames),4)
        self.assertEqual(hub.after(0,threading.Event())[0]['sequence'],17)

    def test_network_head_profile_preserves_existing_head_without_usb_wrists(self):
        import yaml
        root=RECORDING.parents[5]
        original=yaml.safe_load((root/'hardware/robot_vision_ws/src/robot_vision_bringup/config/cameras_30hz.yaml').read_text())
        network=yaml.safe_load((RECORDING/'network_cameras/cameras_head_network.yaml').read_text())
        self.assertEqual(network['cameras'],{'head':original['cameras']['head']})


class CaptureTraceTest(unittest.TestCase):
    def test_trace_drains_all_records_and_reports_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = CaptureTrace(directory)
            for number in range(20):
                self.assertTrue(trace.event('capture', camera='left_wrist', camera_frame_number=number))
            trace.close()
            records = [json.loads(v) for v in trace.path.read_text().splitlines()]
            self.assertEqual([v['camera_frame_number'] for v in records[:-1]], list(range(20)))
            self.assertEqual(records[-1], dict(stage='trace_summary', written=20, dropped=0))
            self.assertFalse(trace.event('capture'))

    def test_full_diagnostic_queue_drops_metadata_without_blocking_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(threading.Thread, 'start'):
                trace = CaptureTrace(directory)
            try:
                for _ in range(trace.jobs.maxsize):
                    self.assertTrue(trace.event('capture'))
                self.assertFalse(trace.event('capture'))
                self.assertEqual(trace.dropped, 1)
                self.assertEqual(trace.jobs.qsize(), 4096)
            finally:
                trace.output.close()

    def test_synthetic_capture_exposes_stage_timings_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            for extra in ([], ['--diagnostics-dir', directory]):
                server = CameraServer(parse_pc_args(['--synthetic', '--warm-seconds', '0'] + extra))
                worker = threading.Thread(target=server.capture, args=('left_wrist', 'test-only'), daemon=True)
                worker.start()
                try:
                    frame = server.hubs['left_wrist'].after(0, server.stopping)
                    self.assertIsNotNone(frame)
                    self.assertEqual('pc_capture_trace' in frame[0], bool(extra))
                    if extra:
                        self.assertGreaterEqual(frame[0]['pc_capture_trace']['encode_ms'], 0)
                        self.assertIn('capture_session', frame[0]['pc_capture_trace'])
                finally:
                    server.stopping.set()
                    worker.join(3.)
                    if server.trace:
                        server.trace.close()
                self.assertFalse(worker.is_alive())


class DelayedCameraPairingTest(unittest.TestCase):
    def test_pairing_uses_capture_time_after_120ms_transport_delay(self):
        args=rec.parse_args(['--width','4','--height','3','--camera-lookahead','0.20'])
        outputs=rec.SuctionOutputs(None,report=lambda _:None)
        inputs=rec.KeyboardInputs(outputs,args);inputs.teaching=dict.fromkeys(rec.SIDES,True)
        saved=[];previous=None;frame=-1
        rgb=np.zeros((3,4,3),dtype=np.uint8)
        for tick in range(401):
            t=tick*.005;wall=100+t;mono=10+t
            outputs.record_readback((False,False),mono-.001)
            if tick%2==0:
                for side in rec.SIDES:inputs.push(side,wall,mono,np.full(7,t))
            current=int((t+1e-8)*30)
            if current>frame:
                inputs.push('head',100+current/30,mono,rgb);frame=current
            delayed=int((t-.12+1e-8)*30)
            if t>=.12 and (not saved or delayed>=0):
                stamp=100+delayed/30
                for side in ('left_wrist','right_wrist'):
                    if not inputs.buffers[side] or stamp>inputs.buffers[side][-1].stamp:
                        inputs.push(side,stamp,mono,rgb)
            try:
                bundle=inputs.snapshot(wall,mono,previous_sources=previous)
            except rec.NotReady:
                continue
            previous=dict(zip(rec.STREAMS,bundle.source_stamps));saved.append(bundle)
        self.assertGreater(len(saved),30)
        for bundle in saved:
            camera_stamps=bundle.source_stamps[2:]
            self.assertLess(np.ptp(camera_stamps),1e-6)
            self.assertLess(abs(float(bundle.state[0])-(bundle.source_stamps[0]-100)),1e-5)
        # Historical pairing may be older, but current joint heartbeat is still mandatory.
        with self.assertRaisesRegex(rec.NotReady,'left 反馈停止'):
            outputs.record_readback((False,False),12.5)
            inputs.snapshot(102.5,12.5,previous_sources=previous)


if __name__=='__main__':unittest.main()
