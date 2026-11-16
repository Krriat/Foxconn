from __future__ import annotations

import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import record_with_torque_retry as fusion


class _FakeACTPolicy:
    def __init__(self, action_steps: int = 10) -> None:
        self.config = SimpleNamespace(n_action_steps=action_steps)

    def predict_action_chunk(self, _observation):
        values = torch.arange(50, dtype=torch.float32)
        return values.reshape(1, 50, 1)


def _controller(
    action_stride: int = 3,
    replan_interval: int | None = None,
) -> fusion.ACTChunkFusionController:
    return fusion.ACTChunkFusionController(
        policy=_FakeACTPolicy(),
        preprocessor=lambda observation: observation,
        postprocessor=lambda action: action,
        device=torch.device("cpu"),
        use_amp=False,
        threshold=0.8,
        old_weight=0.3,
        action_stride=action_stride,
        replan_interval=replan_interval,
    )


class ACTChunkFusionTest(unittest.TestCase):
    def test_fixed_replan_interval_starts_after_two_selected_actions(self) -> None:
        controller = _controller(replan_interval=2)
        with (
            patch.object(
                fusion,
                "prepare_observation_for_inference",
                lambda observation, device, task, robot_type: observation,
            ),
            patch.object(controller, "_start_worker") as start_worker,
        ):
            controller.select_action({}, None, None)
            start_worker.assert_not_called()
            controller.select_action({}, None, None)
            start_worker.assert_called_once()

    def test_default_camera_args_use_two_act_ros_topics(self) -> None:
        with patch.object(sys, "argv", ["record_with_torque_retry.py"]):
            fusion._prepare_camera_args()
            argument = next(
                value for value in sys.argv if value.startswith("--robot.cameras=")
            )

        cameras = json.loads(argument.split("=", 1)[1])
        self.assertEqual(
            {name: config["topic"] for name, config in cameras.items()},
            {
                "head": "/robot/sensors/cameras/head/color/image_raw",
                "right_wrist": "/robot/sensors/cameras/right_wrist/color/image_raw",
            },
        )

    def test_explicit_camera_args_are_preserved(self) -> None:
        explicit = '--robot.cameras={"head":{"type":"mock"}}'
        with patch.object(sys, "argv", ["record_with_torque_retry.py", explicit]):
            fusion._prepare_camera_args()
            self.assertEqual(sys.argv, ["record_with_torque_retry.py", explicit])

    def test_left_camera_configuration_drives_preview_camera_selection(self) -> None:
        explicit = (
            '--robot.cameras={"head":{"type":"mock"},'
            '"left_wrist":{"type":"mock"}}'
        )
        with patch.object(sys, "argv", ["record_with_torque_retry.py", explicit]):
            self.assertEqual(
                fusion._configured_camera_names(),
                ["head", "left_wrist"],
            )

    def test_stride_three_selects_ten_actions_covering_raw_index_27(self) -> None:
        with patch.object(
            fusion,
            "prepare_observation_for_inference",
            lambda observation, device, task, robot_type: observation,
        ):
            actions = _controller(action_stride=3)._predict_chunk({}, None, None)

        self.assertEqual(len(actions), 10)
        self.assertEqual(
            [action.item() for action in actions],
            list(range(0, 28, 3)),
        )

    def test_overlapping_actions_use_old_new_weights_03_07(self) -> None:
        controller = _controller()
        controller._actions = {
            0: torch.tensor([[10.0]]),
            1: torch.tensor([[20.0]]),
        }

        controller._merge_chunk(
            0,
            [torch.tensor([[30.0]]), torch.tensor([[40.0]])],
        )

        self.assertTrue(
            torch.allclose(controller._actions[0], torch.tensor([[24.0]]))
        )
        self.assertTrue(
            torch.allclose(controller._actions[1], torch.tensor([[34.0]]))
        )

    def test_fusion_does_not_reinsert_consumed_actions(self) -> None:
        controller = _controller()
        controller._step = 2
        controller._actions = {2: torch.tensor([[10.0]])}

        controller._merge_chunk(
            0,
            [
                torch.tensor([[100.0]]),
                torch.tensor([[100.0]]),
                torch.tensor([[20.0]]),
            ],
        )

        self.assertEqual(set(controller._actions), {2})
        self.assertTrue(
            torch.allclose(controller._actions[2], torch.tensor([[17.0]]))
        )

    def test_stride_must_fit_inside_predicted_chunk(self) -> None:
        with patch.object(
            fusion,
            "prepare_observation_for_inference",
            lambda observation, device, task, robot_type: observation,
        ):
            with self.assertRaisesRegex(ValueError, "need 55 raw steps"):
                _controller(action_stride=6)._predict_chunk({}, None, None)


if __name__ == "__main__":
    unittest.main()
