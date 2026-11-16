"""Policy boundary tests: stale/replayed commands and bounded physical targets."""
import copy
from pathlib import Path
import unittest
import yaml
from protocol import bounded_target, suction_target, validate_action, validate_chunk, validate_config, round_time_expired


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
        self.pose = self.config["home_left"] + self.config["home_right"]
        self.message = {"schema": "dual_arm_suction_action_v1", "run_id": "current",
                        "publisher_id": "expected", "sequence": 1, "model": "act",
                        "checkpoint_sha256": self.config["checkpoint_sha256"]["act"],
                        "action_hz": 20, "max_input_receive_age_ms": 50,
                        "action": self.pose + [0.7, 0.1]}

    def validate(self, message):
        return validate_action(message, "current", "expected", 0, self.config)

    def test_valid_absolute_joint_and_suction_order(self):
        self.assertEqual(self.validate(self.message), self.pose + [0.7, 0.1])

    def test_configuration_cannot_silently_increase_reviewed_speed_or_watchdog(self):
        validate_config(self.config)
        for key, value in [("max_round_seconds", 0), ("max_joint_speed_rad_s", 2),
                           ("command_timeout_seconds", -1), ("action_hz", 15),
                           ("max_relative_target_rad", float("nan"))]:
            config = copy.deepcopy(self.config); config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_explicit_manual_stop_only_or_positive_duration_is_preserved(self):
        for duration in (None, 30, 60):
            config = copy.deepcopy(self.config)
            config["max_round_seconds"] = duration
            validate_config(config)
        config["max_round_seconds"] = None
        self.assertFalse(round_time_expired(10**8, config))
        config["max_round_seconds"] = 60
        self.assertFalse(round_time_expired(59.9, config))
        self.assertTrue(round_time_expired(60, config))

    def test_stale_round_publisher_weights_and_sequence_rejected(self):
        for key, value in [("run_id", "old"), ("publisher_id", "other"), ("sequence", 0),
                           ("sequence", True), ("model", "interact"), ("checkpoint_sha256", "wrong"),
                           ("action_hz", 15), ("max_input_receive_age_ms", 301)]:
            with self.subTest(key=key, value=value):
                message = copy.deepcopy(self.message); message[key] = value
                with self.assertRaises(ValueError):
                    self.validate(message)

    def test_nonfinite_wrong_shape_and_limits_rejected(self):
        for action in [self.pose[:13], [float("nan")] + self.pose[1:] + [0, 0],
                       [float("inf")] + self.pose[1:] + [0, 0],
                       [10] + self.pose[1:] + [0, 0], self.pose + [2, 0]]:
            with self.subTest(action=action):
                message = copy.deepcopy(self.message); message["action"] = action
                with self.assertRaises(ValueError):
                    self.validate(message)

    def test_velocity_bound_survives_delayed_servo_timer(self):
        action = [x + 0.1 for x in self.pose] + [0, 0]
        target, limited = bounded_target(action, self.pose, self.pose, 0.5, self.config)
        self.assertTrue(limited)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(target,self.pose)), 0.005000001)

    def test_tracking_loss_stops_instead_of_approaching_target(self):
        last = list(self.pose); last[0] += 0.2
        with self.assertRaises(ValueError):
            bounded_target(self.pose + [0, 0], self.pose, last, 0.01, self.config)

    def test_suction_hysteresis_preserves_state_in_middle(self):
        self.assertTrue(suction_target(0.5, True, self.config))
        self.assertFalse(suction_target(0.5, False, self.config))
        self.assertTrue(suction_target(0.8, False, self.config))
        self.assertFalse(suction_target(0.2, True, self.config))

    def test_timed_chunk_clocks_and_full_horizon(self):
        now = 1_000_000_000
        message = dict(self.message, schema="dual_arm_suction_chunk_v2", step_ns=50_000_000,
                       actions=[self.message["action"][:] for _ in range(50)],
                       observation_time_ns=now-80_000_000, inference_start_ns=now-50_000_000,
                       inference_end_ns=now-10_000_000, send_time_ns=now-5_000_000)
        self.assertEqual(len(validate_chunk(message,"current","expected",0,self.config,now)),50)
        for key,value in (("actions",message["actions"][:10]),("observation_time_ns",now-301_000_000),
                          ("inference_end_ns",now+1),("send_time_ns",now+11_000_000),
                          ("step_ns",100_000_000)):
            corrupt=dict(message);corrupt[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_chunk(corrupt,"current","expected",0,self.config,now)
        corrupt=copy.deepcopy(message);corrupt["actions"][49][0]=100.
        with self.assertRaises(ValueError):
            validate_chunk(corrupt,"current","expected",0,self.config,now)


if __name__ == "__main__":
    unittest.main()
