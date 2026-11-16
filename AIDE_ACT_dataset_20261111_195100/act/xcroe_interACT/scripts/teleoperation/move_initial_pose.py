#!/usr/bin/env python3
"""Move the left arm, right arm, or both arms to saved reset joints.

This is a one-shot hardware command used by the Edge API.  Dual-arm motion is
performed by one isolated xCoreSDK process per arm so that the two controllers
start together and never share an SDK client instance.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import signal
import sys
import time
import traceback
from typing import Any

from xcore_interact.robot import _load_sdk
from xcore_interact.schema import HardwareConfig


MOTION_CONFIRMATION = "I_UNDERSTAND_INITIAL_POSE_MOTION"
MOVE_SPEED = 1000
MOVE_ZONE = 10
MOVE_TIMEOUT_S = 60.0
POWER_TIMEOUT_S = 3.0
JOINT_TOLERANCE_RAD = 0.02
JOINT_MARGIN_RAD = math.radians(12.0)

HARD_LOWER = (-3.10669, -2.0944, -3.10669, 0.0, -3.10669, -1.0472, -1.0472)
HARD_UPPER = (3.10669, 2.0944, 3.10669, 2.53073, 3.10669, 1.0472, 1.0472)
DEFAULT_LEFT_JOINTS = (
    1.66969,
    1.365856,
    -1.272963,
    1.476893,
    -1.735811,
    -0.026802,
    -0.165558,
)
DEFAULT_RIGHT_JOINTS = (
    -1.969919,
    1.365856,
    1.272963,
    1.476893,
    1.735811,
    -0.026802,
    0.165558,
)


def _check_ec(ec: dict[str, Any], operation: str) -> None:
    if int(ec.get("ec", 0)):
        raise RuntimeError(
            f"{operation} failed: code={ec.get('ec')} message={ec.get('message', '')}"
        )


def _validated_joints(values: Any, side: str) -> list[float]:
    try:
        joints = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"--{side}-joints must contain 7 numbers") from error
    if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
        raise ValueError(f"--{side}-joints must contain 7 finite values")
    for index, (value, lower, upper) in enumerate(
        zip(joints, HARD_LOWER, HARD_UPPER), start=1
    ):
        if not lower <= value <= upper:
            raise ValueError(
                f"--{side}-joints J{index}={value:.6f} is outside "
                f"[{lower:.6f}, {upper:.6f}]"
            )
    return joints


def _switch_to_nrt(robot: Any, sdk: Any, side: str) -> None:
    last_ec: dict[str, Any] = {}
    for _ in range(20):
        last_ec = {}
        robot.setMotionControlMode(sdk.MotionControlMode.NrtCommandMode, last_ec)
        if not int(last_ec.get("ec", 0)):
            return
        time.sleep(0.1)
    _check_ec(last_ec, f"{side} setMotionControlMode(NrtCommandMode)")


def _restore_ready(robot: Any, sdk: Any, side: str) -> None:
    _switch_to_nrt(robot, sdk, side)
    ec: dict[str, Any] = {}
    robot.setOperateMode(sdk.OperateMode.automatic, ec)
    _check_ec(ec, f"{side} setOperateMode(automatic)")
    ec = {}
    robot.setPowerState(True, ec)
    _check_ec(ec, f"{side} setPowerState(True)")

    deadline = time.monotonic() + POWER_TIMEOUT_S
    while time.monotonic() < deadline:
        ec = {}
        state = robot.powerState(ec)
        _check_ec(ec, f"{side} powerState")
        if state == sdk.PowerState.on:
            return
        if state in (sdk.PowerState.estop, sdk.PowerState.gstop):
            raise RuntimeError(f"{side} safety stop is active")
        time.sleep(0.1)
    raise TimeoutError(f"{side} power-on timeout")


def _controller_limits(robot: Any, sdk: Any, side: str) -> tuple[list[float], list[float]]:
    limits_out = sdk.PyTypeVectorArrayDouble2()
    ec: dict[str, Any] = {}
    enabled = bool(robot.getSoftLimit(limits_out, ec))
    _check_ec(ec, f"{side} getSoftLimit")
    if not enabled:
        raise RuntimeError(f"{side} controller soft limits are disabled")
    pairs = list(limits_out.content())[:7]
    if len(pairs) != 7:
        raise RuntimeError(f"{side} returned fewer than seven soft limits")
    lower: list[float] = []
    upper: list[float] = []
    for hard_lower, hard_upper, pair in zip(HARD_LOWER, HARD_UPPER, pairs):
        controller_lower, controller_upper = [float(value) for value in pair]
        lower.append(max(hard_lower, controller_lower) + JOINT_MARGIN_RAD)
        upper.append(min(hard_upper, controller_upper) - JOINT_MARGIN_RAD)
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise RuntimeError(f"{side} has no usable range after joint safety margin")
    return lower, upper


def _validate_controller_joints(
    joints: list[float],
    lower: list[float],
    upper: list[float],
    side: str,
    label: str,
) -> None:
    for index, (value, minimum, maximum) in enumerate(
        zip(joints, lower, upper), start=1
    ):
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{side} {label} J{index}={value:.6f} is outside controller "
                f"safety range [{minimum:.6f}, {maximum:.6f}]"
            )


def _stop_motion(robot: Any, side: str) -> None:
    if robot is None:
        return
    ec: dict[str, Any] = {}
    try:
        robot.stop(ec)
        _check_ec(ec, f"{side} stop")
    except Exception as error:
        print(f"[{side.upper()}] stop failed: {error}", file=sys.stderr)
    ec = {}
    try:
        robot.moveReset(ec)
        _check_ec(ec, f"{side} moveReset(after stop)")
    except Exception as error:
        print(f"[{side.upper()}] moveReset failed: {error}", file=sys.stderr)


def _move_arm(side: str, joints_values: Any, cancel_event: Any) -> dict[str, Any]:
    joints = _validated_joints(joints_values, side)
    hardware = HardwareConfig()
    config = hardware.left if side == "left" else hardware.right
    sdk = _load_sdk(hardware.sdk_python_path)
    robot: Any | None = None
    started = False
    try:
        print(
            f"[{side.upper()}] connecting robot={config.robot_ip} "
            f"local={config.local_ip}",
            flush=True,
        )
        robot = sdk.ArRobot(config.robot_ip, config.local_ip)
        _restore_ready(robot, sdk, side)
        lower, upper = _controller_limits(robot, sdk, side)
        ec: dict[str, Any] = {}
        current = [float(value) for value in list(robot.jointPos(ec))[:7]]
        _check_ec(ec, f"{side} jointPos(before initial pose)")
        if len(current) != 7:
            raise RuntimeError(f"{side} returned invalid pre-motion joint state")
        _validate_controller_joints(current, lower, upper, side, "measured")
        _validate_controller_joints(joints, lower, upper, side, "target")
        if cancel_event.is_set():
            raise InterruptedError(f"{side} initial-pose motion cancelled before start")

        ec = {}
        robot.moveReset(ec)
        _check_ec(ec, f"{side} moveReset")
        command = sdk.MoveAbsJCommand(joints, MOVE_SPEED, MOVE_ZONE)
        command_id = sdk.PyString()
        ec = {}
        robot.moveAppend([command], command_id, ec)
        _check_ec(ec, f"{side} moveAppend(initial pose)")
        if cancel_event.is_set():
            raise InterruptedError(f"{side} initial-pose motion cancelled before moveStart")
        ec = {}
        robot.moveStart(ec)
        _check_ec(ec, f"{side} moveStart(initial pose)")
        started = True

        deadline = time.monotonic() + MOVE_TIMEOUT_S
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                raise InterruptedError(f"{side} initial-pose motion cancelled")
            ec = {}
            state = robot.operationState(ec)
            _check_ec(ec, f"{side} operationState(initial pose)")
            if state == sdk.OperationState.idle:
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(f"{side} initial-pose motion timed out")

        ec = {}
        actual = [float(value) for value in list(robot.jointPos(ec))[:7]]
        _check_ec(ec, f"{side} jointPos(after initial pose)")
        if len(actual) != 7:
            raise RuntimeError(f"{side} returned invalid post-motion joint state")
        maximum_error = max(abs(value - target) for value, target in zip(actual, joints))
        if maximum_error > JOINT_TOLERANCE_RAD:
            raise RuntimeError(
                f"{side} did not reach initial pose; maximum joint error="
                f"{maximum_error:.6f} rad"
            )
        print(
            f"[{side.upper()}] initial pose reached; "
            f"maximum_joint_error={maximum_error:.6f} rad",
            flush=True,
        )
        return {"side": side, "success": True, "maximum_joint_error": maximum_error}
    except BaseException:
        if started or cancel_event.is_set():
            _stop_motion(robot, side)
        raise


def _worker(side: str, joints: list[float], cancel_event: Any, result_queue: Any) -> None:
    signal.signal(signal.SIGINT, lambda *_: cancel_event.set())
    signal.signal(signal.SIGTERM, lambda *_: cancel_event.set())
    try:
        result_queue.put(_move_arm(side, joints, cancel_event))
    except BaseException as error:
        detail = f"{error.__class__.__name__}: {error}"
        print(f"[{side.upper()}] initial-pose motion failed: {detail}", file=sys.stderr)
        traceback.print_exc()
        cancel_event.set()
        result_queue.put({"side": side, "success": False, "error": detail})


def _move_dual(left: list[float], right: list[float]) -> None:
    context = mp.get_context("spawn")
    cancel_event = context.Event()
    result_queue = context.Queue(maxsize=4)
    workers = {
        side: context.Process(
            target=_worker,
            args=(side, joints, cancel_event, result_queue),
            name=f"initial-pose-{side}",
        )
        for side, joints in (("left", left), ("right", right))
    }

    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    for sig in previous_handlers:
        signal.signal(sig, lambda *_: cancel_event.set())
    results: dict[str, dict[str, Any]] = {}
    try:
        for worker in workers.values():
            worker.start()
        deadline = time.monotonic() + MOVE_TIMEOUT_S + 15.0
        while len(results) < 2 and time.monotonic() < deadline:
            try:
                result = result_queue.get(timeout=0.2)
            except queue.Empty:
                for side, worker in workers.items():
                    if worker.exitcode not in (None, 0) and side not in results:
                        results[side] = {
                            "side": side,
                            "success": False,
                            "error": f"worker exited with code {worker.exitcode}",
                        }
                        cancel_event.set()
                continue
            if isinstance(result, dict) and result.get("side") in workers:
                results[str(result["side"])] = result
                if not result.get("success"):
                    cancel_event.set()
        if len(results) < 2:
            cancel_event.set()
            missing = sorted(set(workers) - set(results))
            for side in missing:
                results[side] = {
                    "side": side,
                    "success": False,
                    "error": "worker completion timed out",
                }
    finally:
        cancel_event.set()
        for worker in workers.values():
            worker.join(timeout=5.0)
        for worker in workers.values():
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    failures = [
        f"{side}: {result.get('error', 'unknown error')}"
        for side, result in sorted(results.items())
        if not result.get("success")
    ]
    if failures:
        raise RuntimeError("dual initial-pose motion failed; " + "; ".join(failures))


def run_self_test() -> None:
    assert _validated_joints(DEFAULT_LEFT_JOINTS, "left") == list(DEFAULT_LEFT_JOINTS)
    assert _validated_joints(DEFAULT_RIGHT_JOINTS, "right") == list(DEFAULT_RIGHT_JOINTS)
    try:
        _validated_joints([0.0] * 6, "right")
    except ValueError:
        pass
    else:
        raise AssertionError("six-joint input was accepted")
    invalid = list(DEFAULT_RIGHT_JOINTS)
    invalid[3] = -0.01
    try:
        _validated_joints(invalid, "right")
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range J4 was accepted")
    print("PASS: initial-pose argument validation (no robot connection or motion)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("left", "right", "dual"), default="right")
    parser.add_argument(
        "--left-joints",
        type=float,
        nargs=7,
        default=list(DEFAULT_LEFT_JOINTS),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument(
        "--right-joints",
        type=float,
        nargs=7,
        default=list(DEFAULT_RIGHT_JOINTS),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument("--confirm-motion", default="")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.confirm_motion != MOTION_CONFIRMATION:
        raise PermissionError(
            f"real initial-pose motion requires --confirm-motion={MOTION_CONFIRMATION}"
        )
    left = _validated_joints(args.left_joints, "left")
    right = _validated_joints(args.right_joints, "right")
    if args.mode == "dual":
        _move_dual(left, right)
        print("Both arms reached their saved teleoperation initial poses")
    else:
        side = args.mode
        joints = left if side == "left" else right
        cancel_event = mp.get_context("spawn").Event()
        signal.signal(signal.SIGINT, lambda *_: cancel_event.set())
        signal.signal(signal.SIGTERM, lambda *_: cancel_event.set())
        _move_arm(side, joints, cancel_event)
        print(f"{side.title()} arm reached its saved teleoperation initial pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
