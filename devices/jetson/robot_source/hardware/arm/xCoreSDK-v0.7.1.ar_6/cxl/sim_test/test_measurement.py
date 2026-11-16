"""Offline regression tests for trajectory mathematics, refusal gates and fault handling."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from common import ROOT, load_config, trajectory, derivative_bounds
from plan_servo_identification import build_plan, validate_plan
from protection import Protection
from run_servo_identification import run_mock
from analyze_actuator_logs import analyze, harmonic, model_fit
from inspect_robot_capabilities import normalize_frames
from replay_actuator_logs import replay
from common import metadata, finish_metadata, write_json

class TrajectoryTests(unittest.TestCase):
    def setUp(self): self.c=load_config(ROOT/'measurement_config.yaml')

    def test_full_envelope_derivatives_and_limits(self):
        plan=build_plan(self.c,'left',4,[.1,.25,.5,1.,2.,4.])
        for tr in plan['trials']:
            t=np.linspace(0,tr['duration_s'],100001)
            q,v,a,j,_=trajectory(tr,t)
            limits=plan['effective_stage_limits']
            for y,bound in zip((v,a,j),limits): self.assertLessEqual(float(np.max(np.abs(y))),bound)
            for values in (q,v,a,j):
                self.assertAlmostEqual(float(values[0]),0.,places=10)
                self.assertAlmostEqual(float(values[-1]),0.,places=10)
            # Differentiate interior points independently to catch envelope-sign errors.
            dt=t[1]-t[0]
            for original,derivative in ((q,v),(v,a),(a,j)):
                error=np.max(np.abs(np.gradient(original,dt)[2:-2]-derivative[2:-2]))
                self.assertLess(error,2e-4)

    def test_corruption_limits_and_no_auto_center(self):
        plan=build_plan(self.c,'right',1)
        self.assertEqual(plan['coordinates'],'relative_preview_only')
        self.assertFalse(plan['hardware_execution_allowed'])
        plan['trials'][0]['amplitude_rad']=1
        with self.assertRaises(ValueError): validate_plan(plan)
        with self.assertRaises(ValueError): build_plan(self.c,'left',1,[6])
        self.c['center_q_rad']['left']=[0.]*7
        self.c['limits'].update(joint_lower_rad=[-.001]*7,joint_upper_rad=[.001]*7,joint_stop_margin_rad=.0001)
        with self.assertRaises(ValueError): build_plan(self.c,'left',1)

    def test_harmonic_known_phase(self):
        t=np.linspace(0,30,7500); lag=.073
        amp,phase,bias,_=harmonic(t,.02*np.sin(2*np.pi*.5*(t-lag))+.4,.5)
        self.assertAlmostEqual(amp,.02);self.assertAlmostEqual(bias,.4)
        self.assertAlmostEqual(-phase/np.pi,lag)

    def test_multifrequency_model_excludes_holdout(self):
        es=[]
        for f in [.1,.25,.5,1,2]:
            w=2*np.pi*f; h=.9*np.exp(-1j*w*.08)/(1+1j*w*.05)
            es.append(dict(identifiable=True,role='identification',excited=True,frequency_hz=f,gain=abs(h),phase_rad=np.angle(h)))
        es.append(dict(identifiable=True,role='holdout',excited=True,frequency_hz=3,gain=900,phase_rad=0))
        m=model_fit(es)
        self.assertNotIn(3,m['fit_frequencies_hz']);self.assertAlmostEqual(m['effective_delay_s'],.08,delta=.005)
        self.assertAlmostEqual(m['tau_s'],.05,delta=.006)

class ProtectionTests(unittest.TestCase):
    def ready(self):
        g=Protection();g.feedback(10,100,1000,[0.]*7,[0.]*7);return g

    def assert_stopped_latch(self,g):
        self.assertNotEqual(g.state,'active')
        self.assertFalse(g.can_send(1,1));g.feedback(1000,1000,1000,[0.]*7,[0.]*7)
        self.assertNotEqual(g.state,'active')

    def test_repeat_is_not_fresh(self):
        g=self.ready()
        self.assertTrue(g.feedback(10,100,20_000_000,[0.]*7,[0.]*7))
        g.tick(60_000_000);self.assertEqual(g.reason,'stale_feedback');self.assert_stopped_latch(g)

    def test_drop_rollback_nan_mode_fault(self):
        cases=[('drop',lambda g:g.tick(60_000_000)),
               ('sequence',lambda g:g.feedback(9,200,2000,[0.]*7,[0.]*7)),
               ('clock',lambda g:g.feedback(11,99,2000,[0.]*7,[0.]*7)),
               ('host',lambda g:g.feedback(11,101,999,[0.]*7,[0.]*7)),
               ('nan',lambda g:g.feedback(11,101,2000,[float('nan')]*7,[0.]*7)),
               ('mode',lambda g:g.feedback(11,101,2000,[0.]*7,[0.]*7,'drag')),
               ('fault',lambda g:g.feedback(11,101,2000,[0.]*7,[0.]*7,fault=True)),
               ('late_send',lambda g:g.can_send(0,20_000_000)),
               ('blocked_send',lambda g:g.can_send(2000,2000,30_000_000)),
               ('log_overflow',lambda g:g.can_send(2000,2000,log_ok=False))]
        for name,inject in cases:
            with self.subTest(name=name):
                g=self.ready();inject(g);self.assert_stopped_latch(g)

    def test_stop_readback_and_failure(self):
        g=self.ready();g.stop('operator',2000)
        self.assertFalse(g.confirm_stop(3000,True,[.2]*7))
        self.assertFalse(g.confirm_stop(3000,False,[0.]*7))
        self.assertTrue(g.confirm_stop(3000,True,[0.]*7));self.assert_stopped_latch(g)
        g=self.ready();g.stop('operator',2000);g.tick(2_000_000_000)
        self.assertEqual(g.state,'stop_unconfirmed')
        self.assertFalse(g.confirm_stop(2_000_000_001,True,[0.]*7))

class PipelineTests(unittest.TestCase):
    def test_mock_analysis_hashes_and_sdk_refusal(self):
        c=load_config(ROOT/'measurement_config.yaml');c['trial']['repetitions']=1;c['trial']['holdout_frequency_hz']=None
        plan=build_plan(c,'left',1)
        with tempfile.TemporaryDirectory() as folder:
            run,complete=run_mock(plan,folder)
            self.assertTrue(complete)
            summary=analyze(run,plots=False)
            self.assertTrue(summary['synthetic']);self.assertFalse(summary['raw_hash_mismatches'])
            self.assertEqual(len(summary['groups']),7)
            with (run/'servo_estimates.csv').open() as f: es=list(csv.DictReader(f))
            e=next(e for e in es if e['excited']=='True')
            self.assertEqual(e['identifiable'],'True')
            self.assertGreater(float(e['effective_phase_lag_s']),.04)
            self.assertLess(float(e['effective_phase_lag_s']),.10)
            self.assertEqual(summary['models']['left/left_joint1']['status'],'not_identifiable')
            report=replay(run)
            self.assertIsNone(report['reason']);self.assertFalse(report['raw_hash_mismatches'])
            result=subprocess.run([sys.executable,str(ROOT/'run_servo_identification.py'),'--plan',str(run/'executed_plan.json'),'--backend','sdk','--execute'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertIn('未连接',result.stderr)
            with (run/'events.jsonl').open('a') as f:f.write('{}\n')
            changed=analyze(run,plots=False)
            self.assertIn('events.jsonl',changed['raw_hash_mismatches'])

    def test_native_identical_payload_not_device_duplicate_and_empty_failure(self):
        c=load_config(ROOT/'measurement_config.yaml')
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'raw_vendor').mkdir()
            # A failed connection with no frame file must still be analyzable.
            normalize_frames(run,'left',True)
            self.assertTrue((run/'commands.csv').is_file())
            records=[dict(sample_seq=i,host_rx_ns=10**18+i*4_000_000,identical_payload=bool(i),
                          q=[0.]*7,qd=[0.]*7,tau=[1.]*7) for i in range(3)]
            (run/'raw_vendor/frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            normalize_frames(run,'left',True)
            (run/'events.jsonl').write_text('')
            write_json(run/'executed_plan.json',dict(kind='read_only_inspection'))
            meta=metadata(c,'sdk_read_only');meta['complete']=True;finish_metadata(run,meta)
            summary=analyze(run,plots=False)
            for group in summary['groups']:
                self.assertIsNone(group['duplicate_samples']);self.assertEqual(group['valid_rows'],3)
                self.assertAlmostEqual(group['host_packet_interval_ms']['mean'],4.)
                self.assertEqual(group['identical_payload_rows'],2)
            with (run/'torque_vs_speed.csv').open() as f:
                for row in csv.DictReader(f):
                    self.assertEqual(row['torque_nm'],'');self.assertEqual(float(row['raw_sdk_tau_nm']),1.)

    def test_aborted_mock_never_resumes(self):
        c=load_config(ROOT/'measurement_config.yaml');c['trial']['repetitions']=1;c['trial']['holdout_frequency_hz']=None
        plan=build_plan(c,'right',2)
        with tempfile.TemporaryDirectory() as folder:
            for fault in ('duplicate','dropout','nan','sequence_rollback','clock_rollback','mode_change',
                          'drive_fault','command_timeout','log_overflow','operator_stop','stop_failure'):
                with self.subTest(fault=fault):
                    run,complete=run_mock(plan,folder,fault)
                    self.assertFalse(complete)
                    meta=json.loads((run/'metadata.json').read_text())
                    events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
                    stop=next(e['host_ns'] for e in events if e['type']=='controlled_stop_requested')
                    with (run/'commands.csv').open() as f:
                        self.assertTrue(all(int(r['host_tx_start_ns'])<stop for r in csv.DictReader(f)))
                    self.assertEqual(meta['stop_state'],'stop_unconfirmed' if fault=='stop_failure' else 'stopped')

if __name__=='__main__': unittest.main()
