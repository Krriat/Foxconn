"""Capture-to-export regressions and training diagnostics; no hardware."""
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/recording"))
import record_xcore_teach_suction as r
from xcore_teach_recording.quality import episode_quality

spec = importlib.util.spec_from_file_location("suction_quality_checker", ROOT/"scripts/diagnostics/check_teach_training.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
spec = importlib.util.spec_from_file_location("suction_generic_exporter", ROOT/"scripts/dataset_tools/export_teach_lerobot.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = r.dataset_schema(r.parse_args(["--fps", "30", "--width", "4", "--height", "3"]))
        r.prepare_root(self.root, self.settings)

    def record(self, *, split=False, lost_boundary=False):
        writer = r.EpisodeWriter(self.root, "双臂吸盘测试", self.settings)
        try:
            writer.begin_segment()
            for i in range(60):
                if split and i == 30:
                    writer.end_segment("mode transition")
                    writer.begin_segment()
                t = 100.+i/30+(.2 if lost_boundary and i >= 30 else 0.)
                q = np.arange(1,15,dtype=np.float32)/100+i/1000
                suction = [int(20 <= i < 40), int(30 <= i < 50)]
                state = np.r_[q, suction].astype(np.float32)
                writer.add(r.Bundle(t, state, state.copy(),
                    {k: np.full((3,4,3), [17,33,201], np.uint8) for k in r.CAMERAS},
                    np.full(5,t), t-.01, np.full(2,t-.02)))
                writer.jobs.join()
            writer.end_segment("save")
            return writer.save()
        finally:
            if not writer.closed:
                writer.discard()

    def rewrite_arrays(self, path, edit):
        with np.load(path/"trajectory.npz", allow_pickle=False) as archive:
            arrays = {k:archive[k] for k in archive.files}
        edit(arrays)
        np.savez_compressed(path/"trajectory.npz", **arrays)

    def test_complete_demonstration_has_both_suction_values_and_context(self):
        path = self.record()
        meta, arrays = r.validate_episode(path,self.settings)
        quality = episode_quality(meta,arrays)
        self.assertEqual(quality["warnings"], [])
        self.assertEqual(quality["full_chunk_start_frames"], 11)
        for side in r.SIDES:
            self.assertEqual(quality["suction"][side]["action_values"], [0,1])
        self.assertEqual(meta["training_quality"]["status"], "checks_passed")
        self.assertEqual(checker.check_dataset(self.root)["status"], "checks_passed")

    def test_ros_queue_overwrites_do_not_invalidate_independent_local_samples(self):
        meta,arrays = r.validate_episode(self.record(),self.settings)
        meta["capture_diagnostics"] = dict(local_joint_feedback=dict(at_record_start={},
            current=dict(error=None,streams={"left.drag":dict(missing_sequences=0,ros_queue_overwrites=500)})))
        self.assertEqual(episode_quality(meta,arrays)["warnings"],[])

    def test_local_packet_loss_is_reported_relative_to_recording_start(self):
        meta,arrays = r.validate_episode(self.record(),self.settings)
        meta["capture_diagnostics"] = dict(local_joint_feedback=dict(
            at_record_start=dict(streams={"left.drag":dict(missing_sequences=3)}),
            current=dict(error=None,streams={"left.drag":dict(missing_sequences=4)})))
        warnings=episode_quality(meta,arrays)["warnings"]
        self.assertTrue(any("序号缺失1个" in w for w in warnings))

    def test_unsplit_missing_sources_cannot_be_promoted_to_completed_episode(self):
        with self.assertRaisesRegex(ValueError,"未分段"):
            self.record(lost_boundary=True)
        self.assertEqual(list(self.root.glob("episode_*")), [])

    def test_future_or_unrelated_image_anchor_is_rejected(self):
        path = self.record()
        self.rewrite_arrays(path, lambda a:a["ros_timestamp"].__setitem__(10,a["ros_timestamp"][10]+.2))
        with self.assertRaisesRegex(ValueError,"源时间戳不匹配"):
            r.validate_episode(path,self.settings)

    def test_future_suction_command_is_rejected(self):
        path = self.record()
        self.rewrite_arrays(path, lambda a:a["suction_command_timestamp"].__setitem__((10,0),a["ros_timestamp"][10]+.01))
        with self.assertRaisesRegex(ValueError,"未来"):
            r.validate_episode(path,self.settings)

    def test_nonbinary_suction_label_is_rejected(self):
        path = self.record()
        self.rewrite_arrays(path, lambda a:a["action"].__setitem__((10,15),.5))
        with self.assertRaisesRegex(ValueError,"0或1"):
            r.validate_episode(path,self.settings)

    def test_wrong_png_bit_depth_is_not_silently_converted_to_rgb8(self):
        path = self.record()
        cv2.imwrite(str(path/"images/head/000010.png"), np.full((3,4,3),1000,np.uint16))
        with self.assertRaisesRegex(ValueError,"图像"):
            r.validate_episode(path,self.settings)
        self.assertEqual(checker.check_dataset(self.root)["status"], "invalid")

    def test_generic_export_preserves_suction_features_rgb_and_segment_boundaries(self):
        self.record(split=True)
        episodes, pending, arguments = [], [], {}
        destination = self.root/"export"
        dataset = SimpleNamespace(image_writer=None, finalize=lambda:None, stop_image_writer=lambda:None)
        dataset.add_frame = pending.append
        def save():
            episodes.append(list(pending));pending.clear()
        dataset.save_episode = save
        def factory(**kwargs):
            arguments.update(kwargs)
            (destination/"meta").mkdir(parents=True)
            return dataset
        count = exporter.export(self.root,destination,"local/quality_test",factory)
        self.assertEqual(count,2)
        self.assertEqual([len(e) for e in episodes],[30,30])
        self.assertEqual(arguments["features"]["action"]["names"][-2:],list(r.STATE_NAMES[-2:]))
        self.assertEqual(arguments["robot_type"],"xcore_dual_arm_suction_keyboard")
        self.assertEqual(arguments["vcodec"],"h264")
        np.testing.assert_array_equal(episodes[0][20]["action"][14:],[1,0])
        np.testing.assert_array_equal(episodes[1][0]["action"][14:],[1,1])
        np.testing.assert_array_equal(episodes[1][20]["action"][14:],[0,0])
        np.testing.assert_array_equal(episodes[0][0]["observation.images.head"][0,0],[17,33,201])

    def test_both_validation_cli_entries_accept_the_suction_schema(self):
        self.record()
        with patch.object(r,"notify") as notify:
            self.assertEqual(r.main(["--root",str(self.root),"--validate-only"]),0)
            self.assertTrue(any("格式/时序校验通过" in str(c) for c in notify.call_args_list))
        process=subprocess.run([sys.executable,str(ROOT/"scripts/dataset_tools/export_teach_lerobot.py"),
                                "--root",str(self.root),"--validate-only"],
                               text=True,capture_output=True,timeout=10)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertIn("structure/timing OK",process.stdout)

    def test_fractional_fps_is_rejected_before_creating_export(self):
        settings = dict(self.settings,fps=1.5)
        (self.root/"schema.json").write_text(json.dumps(settings))
        factory=Mock()
        with self.assertRaisesRegex(ValueError,"整数采样率"):
            r.export_dataset(self.root,self.root/"export","local/test",factory)
        factory.assert_not_called()

    def test_stopped_camera_and_unrecorded_return_require_review(self):
        path = self.record()
        meta,arrays=r.validate_episode(path,self.settings)
        meta["recording_summary"]={"elapsed_seconds":60.}
        meta["capture_gaps"]=[dict(after_frame=60,start_elapsed_seconds=2.,end_elapsed_seconds=60.)]
        meta["control_events"]=[dict(key="b",after_frame=60)]
        quality=episode_quality(meta,arrays)
        self.assertEqual(quality["status"],"needs_review")
        self.assertEqual(quality["trailing_no_frame_seconds"],58.)
        self.assertTrue(any("回位" in s for s in quality["warnings"]))

    def test_line_input_accepts_b_without_triggering_other_commands(self):
        read_fd,write_fd=os.pipe()
        events=queue.Queue()
        with os.fdopen(read_fd) as stream,patch.object(sys,"stdin",stream):
            try:
                with r.Keyboard(events,line_input=True):
                    os.write(write_fd,b"b\n")
                    self.assertEqual(events.get(timeout=2),("key","b"))
            finally:
                os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
