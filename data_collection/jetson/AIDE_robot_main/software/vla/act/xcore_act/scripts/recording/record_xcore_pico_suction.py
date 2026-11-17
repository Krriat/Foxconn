#!/usr/bin/env python3
r"""PICO single-arm recording with independent left/right USR-M050 suction cups.

Use the same ROS/LeRobot environment and recorder options as record_xcore_pico.py.
Left Trigger controls DO1 (left cup); right Trigger controls DO2 (right cup).
Pressed = suction ON / relay closed, released = suction OFF / relay open.
Grip, A/down, B/up and subtask handling are inherited unchanged.

Two terminals (cameras must already be running). Configure BOTH PICO controllers
to send to the robot computer's IP, UDP 5051 (not the legacy port 5050):

  python record_xcore_pico_suction.py --teleop-only
  # In another terminal, using the original recorder's ROS/LeRobot environment:
  python record_xcore_pico_suction.py \
      --repo-id local/xcore_right_suction --task "pick and place" \
      --root /path/to/a/new/suction_dataset

Teleop mode receives both hands at UDP 5051 and forwards the unchanged packets
to the original PICO process at UDP 5050. It mirrors packets to a local-only
recording input (default 15051) for independent suction control. Teleoperation
continues if the recorder exits, just as it does with the original recorder.
The ROS remaps isolate its old gripper command and transport the selected arm's
actual DO readback through its legacy gripper-feedback input. Datasets contain
9 features: the selected arm's 7 joints + left_suction.on + right_suction.on.
Both suction values are 0/1. Arm selection and all other buttons stay unchanged;
this does not add left-arm motion to the original right-arm-only executable.

Only this new file is needed; neither the old recorder nor pico_servol is edited.
Recording mode leaves arm motion to the external PICO process; --teleop-only
launches that original process with ROS remaps. Arm gains/initial pose and
GPIO/DO1 services are unchanged. M050 SDK calls run in a worker thread.
Startup only reads IO until controller data arrives. Shutdown does not write an
extra relay command; the device retains its last output as documented by M050.
No automatic write retries. A failed write is reconciled by reading DO state.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as C
import importlib.util
import ipaddress
import json
import math
from pathlib import Path
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace


WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
M050_ROOT = WORKSPACE_ROOT / "hardware" / "USR-M050"
JOINT_NAMES = tuple(f"joint_{index}.pos" for index in range(1, 8))
SUCTION_NAMES = {"left": "left_suction.on", "right": "right_suction.on"}
STATE_NAMES = JOINT_NAMES + tuple(SUCTION_NAMES.values())
CHANNELS = {"left": 1, "right": 2}
ACTION_SOURCE = "controller_jointPos_m_same_sample_and_both_pico_trigger_suctions"


class M050Config(C.Structure):
    _fields_ = [("host", C.c_char_p), ("port", C.c_uint16),
                ("unit_id", C.c_uint8), ("timeout_ms", C.c_int),
                ("source_ip", C.c_char_p)]


class M050Channels(C.Structure):
    _fields_ = [("ch1", C.c_uint8), ("ch2", C.c_uint8)]


class M050Error(C.Structure):
    _fields_ = [("code", C.c_int), ("system_errno", C.c_int),
                ("modbus_exception", C.c_uint8),
                ("write_may_have_applied", C.c_int), ("message", C.c_char * 192)]


class M050Client:
    """ctypes binding to the existing SDK, including its write/readback checks."""

    def __init__(self, args, channel):
        if channel not in (1, 2):
            raise ValueError("M050 channel must be 1 or 2")
        self.channel = channel
        self.library = C.CDLL(str(args.m050_library))
        self.handle = C.c_void_p()
        self.library.m050_config_default.argtypes = [C.POINTER(M050Config)]
        self.library.m050_config_default.restype = None
        self.library.m050_create.argtypes = [C.POINTER(M050Config),
                                             C.POINTER(C.c_void_p), C.POINTER(M050Error)]
        self.library.m050_create.restype = C.c_int
        self.library.m050_destroy.argtypes = [C.c_void_p]
        self.library.m050_destroy.restype = None
        self.library.m050_read_do.argtypes = [C.c_void_p, C.POINTER(M050Channels),
                                              C.POINTER(M050Error)]
        self.library.m050_read_do.restype = C.c_int
        self.library.m050_set_do.argtypes = [C.c_void_p, C.c_uint, C.c_int,
                                             C.POINTER(M050Channels), C.POINTER(M050Error)]
        self.library.m050_set_do.restype = C.c_int
        config = M050Config()
        self.library.m050_config_default(C.byref(config))
        config.host = args.m050_host.encode("ascii")
        config.port, config.unit_id = args.m050_port, args.m050_unit
        config.timeout_ms = args.m050_timeout_ms
        config.source_ip = args.m050_source_ip.encode("ascii") if args.m050_source_ip else None
        error = M050Error()
        self._check(self.library.m050_create(C.byref(config), C.byref(self.handle),
                                             C.byref(error)), error)

    @staticmethod
    def _check(result, error):
        if result:
            message = bytes(error.message).decode("utf-8", errors="replace")
            uncertain = "; write may already have applied" if error.write_may_have_applied else ""
            raise RuntimeError(f"M050 error {result}: {message}{uncertain}")

    def read(self):
        state, error = M050Channels(), M050Error()
        self._check(self.library.m050_read_do(self.handle, C.byref(state), C.byref(error)), error)
        return bool(state.ch1 if self.channel == 1 else state.ch2)

    def write(self, enabled):
        state, error = M050Channels(), M050Error()
        self._check(self.library.m050_set_do(self.handle, self.channel, int(enabled),
                                            C.byref(state), C.byref(error)), error)
        return bool(state.ch1 if self.channel == 1 else state.ch2)

    def close(self):
        if self.handle:
            self.library.m050_destroy(self.handle)
            self.handle = C.c_void_p()


@dataclass
class SuctionState:
    desired: bool | None = None
    confirmed: bool | None = None
    verified_at: float = 0.0
    buttons_at: float = 0.0
    sequence: int = 0
    applied_sequence: int = -1
    error: str | None = None


class SuctionController:
    """Serialize IO off the ROS executor; only Trigger level changes write DO."""

    def __init__(self, client, *, poll_seconds=.1, report=print):
        self.client, self.poll_seconds, self.report = client, poll_seconds, report
        self.condition = threading.Condition()
        self.state = SuctionState()
        self.stopping = False
        self.last_report = None
        self.thread = threading.Thread(target=self._run, name="m050-suction", daemon=True)

    def start(self):
        self.thread.start()

    def buttons(self, buttons):
        desired = bool(int(buttons) & 1)  # Trigger is bit 0; all other bits pass through.
        with self.condition:
            self.state.buttons_at = time.monotonic()
            if desired != self.state.desired:
                self.state.desired = desired
                self.state.sequence += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return replace(self.state)

    def _report(self, message):
        if message != self.last_report:
            self.report(message)
            self.last_report = message

    def _run(self):
        attempted_sequence = 0
        next_poll = 0.0
        while True:
            with self.condition:
                request = replace(self.state)
                write = request.desired is not None and request.sequence != attempted_sequence
                if self.stopping and not write:
                    return
                if not write and time.monotonic() < next_poll:
                    self.condition.wait(timeout=max(0., next_poll - time.monotonic()))
                    continue
            try:
                if write:
                    # Mark attempted before IO: failed writes are never blindly retried.
                    attempted_sequence = request.sequence
                    confirmed = self.client.write(request.desired)
                    if confirmed != request.desired:
                        raise RuntimeError("M050 DO readback does not match Trigger")
                else:
                    confirmed = self.client.read()
                with self.condition:
                    self.state.confirmed = confirmed
                    self.state.verified_at = time.monotonic()
                    if self.state.desired is None or confirmed == self.state.desired:
                        self.state.applied_sequence = self.state.sequence
                        self.state.error = None
                    else:
                        self.state.error = "M050 DO differs from the current Trigger; waiting for a new Trigger edge"
                    error = self.state.error
                self._report(error or f"[SUCTION] DO{self.client.channel}={'ON' if confirmed else 'OFF'} (verified)")
            except Exception as exc:
                with self.condition:
                    self.state.confirmed = None
                    self.state.error = str(exc)
                self._report(f"[SUCTION] {exc}; recording waits for valid DO readback")
            next_poll = time.monotonic() + self.poll_seconds

    def close(self, timeout):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError("M050 worker has not stopped; SDK handle was left intact")
        self.client.close()


class PicoUdpRelay:
    """Read each hand's Trigger and forward its exact packet to legacy teleop."""

    def __init__(self, on_buttons=None, *, host="0.0.0.0", port=5051,
                 forward_port=5050, mirror_port=None):
        if port and port in (forward_port, mirror_port):
            raise ValueError("PICO input port must differ from the legacy teleop port")
        self.on_buttons = on_buttons
        self.destinations = [("127.0.0.1", value) for value in (forward_port, mirror_port)
                             if value is not None]
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((host, port))
            self.socket.settimeout(.1)
        except BaseException:
            self.socket.close()
            raise
        self.port = self.socket.getsockname()[1]
        self.stopping = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="pico-dual-trigger-relay", daemon=True)

    def start(self):
        self.thread.start()

    @staticmethod
    def parse_buttons(packet):
        if len(packet) != 38 or packet[0] not in (0, 1):
            return None
        values = struct.unpack("<9f", packet[1:37])
        if not all(math.isfinite(value) for value in values):
            return None
        if sum(value * value for value in values[3:7]) < 1e-16:
            return None
        return ("left" if packet[0] == 0 else "right", packet[37])

    def _run(self):
        while not self.stopping.is_set():
            try:
                packet, _ = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stopping.is_set():
                    self.error = str(exc)
                return
            parsed = self.parse_buttons(packet)
            if parsed is None:
                continue
            if self.on_buttons is not None:
                self.on_buttons(*parsed)
            try:
                for destination in self.destinations:
                    self.socket.sendto(packet, destination)
            except OSError as exc:
                self.error = str(exc)
                return

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=1.)
        self.socket.close()
        if self.thread.is_alive():
            raise RuntimeError("PICO UDP relay has not stopped")


def telemetry_topics(side):
    prefix = f"/xcore/{side}/suction_source"
    return {key: f"{prefix}/{value}" for key, value in (
        ("state", "record_state"), ("action", "record_action"),
        ("active", "teleop_active"), ("buttons", "pico_buttons"),
        ("stop", "record_stop_teleop"))}


def feedback_topic(side):
    return f"/xcore/{side}/suction_source/relay_feedback"


def teleop_command(args):
    """ROS remapping prevents the legacy publisher from reaching the gripper."""
    side = args.arm_side
    remaps = {
        "/sts3215/position": f"/xcore/{side}/suction_source/unused_gripper_command",
        "/sts3215/state": feedback_topic(side),
    }
    names = {"state": "record_state", "action": "record_action", "active": "teleop_active",
             "buttons": "pico_buttons", "stop": "record_stop_teleop"}
    remaps.update({f"/xcore/{side}/{names[key]}": value
                   for key, value in telemetry_topics(side).items()})
    command = [str(args.pico_executable), "--ros-args"]
    for source, destination in remaps.items():
        command.extend(["-r", f"{source}:={destination}"])
    return command


def run_teleop(args):
    """Keep the UDP relay with the original teleop, independently of recording."""
    stopped = threading.Event()

    def request_stop(_signum, _frame):
        stopped.set()

    def stop_process(process):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)  # Original program's SDK cleanup handler.
            try:
                process.wait(timeout=15.)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"PICO process {process.pid} has not confirmed exit; "
                                   "the UDP relay is being closed") from exc

    with ExitStack() as cleanup:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, request_stop)
            cleanup.callback(signal.signal, signum, previous)
        relay = PicoUdpRelay(port=args.pico_input_port, mirror_port=args.pico_trigger_port)
        try:
            relay.start()
        except BaseException:
            relay.socket.close()
            raise
        cleanup.callback(relay.close)
        process = subprocess.Popen(teleop_command(args), start_new_session=True)
        cleanup.callback(stop_process, process)
        print(f"[PICO] Both hands -> robot IP:UDP {relay.port}; "
              "original teleop receives unchanged packets at UDP 5050.", flush=True)
        while not stopped.is_set():
            result = process.poll()
            if result is not None:
                return result
            if relay.error is not None:
                raise RuntimeError(f"PICO UDP relay failed: {relay.error}")
            stopped.wait(.1)
        return 0


def parse_suction_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All remaining options are passed unchanged to record_xcore_pico.py "
               "(including --repo-id, --root, --task, --fps and --segment-title).",
        allow_abbrev=False)
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument("--m050-host", default="192.168.1.7")
    parser.add_argument("--m050-port", type=int, default=20108)
    parser.add_argument("--m050-unit", type=int, default=1)
    parser.add_argument("--pico-input-port", type=int, default=5051,
                        help="both PICO hands send here; packets are forwarded unchanged to UDP 5050")
    parser.add_argument("--pico-trigger-port", type=int, default=15051,
                        help="local-only trigger mirror; use the same value in both terminals")
    parser.add_argument("--m050-timeout-ms", type=int, default=3000)
    parser.add_argument("--m050-source-ip", default=None)
    parser.add_argument("--m050-poll-seconds", type=float, default=.1)
    parser.add_argument("--m050-state-timeout", type=float, default=1.)
    parser.add_argument("--m050-library", type=Path, default=M050_ROOT / "build" / "libm050.so")
    parser.add_argument("--pico-executable", type=Path, default=WORKSPACE_ROOT /
                        "hardware/arm/xCoreSDK-v0.7.1.ar_6/build/bin/pico_servol",
                        help="original PICO executable for --teleop-only / --print-teleop-command")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-teleop-command", action="store_true")
    mode.add_argument("--teleop-only", action="store_true",
                      help="run the original PICO executable with the suction remaps; no recorder")
    mode.add_argument("--self-test", action="store_true", help="pure software test; no hardware IO")
    args, remaining = parser.parse_known_args(argv)
    if any(value == "--m050-channel" or value.startswith("--m050-channel=") for value in remaining):
        parser.error("Both channels are fixed: left Trigger=DO1, right Trigger=DO2; remove --m050-channel")
    try:
        ipaddress.IPv4Address(args.m050_host)
        if args.m050_source_ip:
            ipaddress.IPv4Address(args.m050_source_ip)
    except ipaddress.AddressValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.m050_port <= 65535 or not 1 <= args.m050_unit <= 247:
        parser.error("M050 port must be 1..65535 and unit must be 1..247")
    if not 1 <= args.pico_input_port <= 65535 or args.pico_input_port == 5050:
        parser.error("PICO input port must be 1..65535 and differ from legacy UDP 5050")
    if (not 1 <= args.pico_trigger_port <= 65535 or
            args.pico_trigger_port in (5050, args.pico_input_port)):
        parser.error("PICO trigger port must be 1..65535 and differ from both PICO input and legacy ports")
    if not 1 <= args.m050_timeout_ms <= 2147483647:
        parser.error("M050 timeout must be a positive signed 32-bit millisecond value")
    if not all(math.isfinite(value) and value > 0 for value in
               (args.m050_poll_seconds, args.m050_state_timeout)):
        parser.error("M050 poll and state timeout must be finite and positive")
    return args, remaining


def load_recorder():
    source = Path(__file__).with_name("record_xcore_pico.py")
    sys.path.insert(0, str(source.parents[2] / "src"))
    spec = importlib.util.spec_from_file_location("_xcore_pico_suction_base", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing recording dependency {exc.name!r}; use the same "
                           "ROS/LeRobot environment as record_xcore_pico.py") from exc
    return module


def configure_recorder(base, args, controllers):
    """Reuse the original lifecycle and buttons; adapt only the end effector."""
    from std_msgs.msg import Int32

    base.STATE_NAMES = base.ACTION_NAMES = STATE_NAMES
    base.JOINT_NAMES = JOINT_NAMES
    base.ACTION_SOURCE = ACTION_SOURCE
    base.TELEMETRY_TOPICS = {side: telemetry_topics(side) for side in ("left", "right")}

    def validate_action(action):
        values = [float(action[name]) for name in STATE_NAMES]
        if not all(math.isfinite(value) for value in values):
            return False, "joint/suction action contains NaN or infinity"
        if any(abs(value) > base.MAX_ABS_JOINT_POSITION_RAD for value in values[:7]):
            return False, "joint action is outside the 3.2 rad sanity bound"
        return ((True, "") if all(value in (0., 1.) for value in values[7:])
                else (False, "both suction values must be 0 or 1"))

    base.validate_joint_action = validate_action
    binding = {"host": args.m050_host, "port": args.m050_port, "unit": args.m050_unit,
               "channels": CHANNELS, "pressed_closed": True}
    original_metadata = base.write_action_source_metadata

    def write_metadata(root):
        original_metadata(root)
        path = root / "meta" / base.ACTION_SOURCE_FILE
        metadata = json.loads(path.read_text())
        metadata.update(schema_version=3, state_names=list(STATE_NAMES), m050=binding,
                        observation_source="controller_jointPos_m_and_latest_verified_m050_do_readback",
                        suction_observation_source="latest_verified_m050_do_readback",
                        suction_action_source="left_and_right_raw_pico_udp_trigger_bit_0",
                        suction_feedback_timing="asynchronous; not a simultaneous joint/IO measurement",
                        suction_feedback_max_age_seconds=args.m050_state_timeout,
                        suction_feedback_is_vacuum_detection=False)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(path)

    base.write_action_source_metadata = write_metadata
    original_make_dataset = base.make_dataset

    def make_dataset(robot, recorder_args):
        root = Path(recorder_args.root)
        if (root / "meta" / "info.json").exists():
            info = json.loads((root / "meta" / "info.json").read_text())
            if tuple(info["features"]["observation.state"]["names"]) != STATE_NAMES:
                raise RuntimeError("Use a new --root for suction data; the existing state schema differs")
            metadata = json.loads((root / "meta" / base.ACTION_SOURCE_FILE).read_text())
            if metadata.get("m050") != binding:
                raise RuntimeError("Use a new --root: existing M050 channel/connection semantics differ")
        return original_make_dataset(robot, recorder_args)

    base.make_dataset = make_dataset
    original_parse_args = base.parse_args

    def parse_args():
        parsed = original_parse_args()
        if not any(value == "--root" or value.startswith("--root=") for value in sys.argv[1:]):
            parsed.root = str(base.REPO_ROOT / "data/datasets" / f"xcore_{args.arm_side}_arm_dual_suction_qm_v1")
        return parsed

    base.parse_args = parse_args
    original_bridge = base.PicoDatasetBridge

    class SuctionDatasetBridge(original_bridge):
        def __init__(self, arm_side):
            super().__init__(arm_side)
            self._arm_side = arm_side
            self._relay_feedback = self.create_publisher(Int32, feedback_topic(arm_side), 10)
            self._relay_timer = self.create_timer(.05, self._publish_relay_feedback)

        def _publish_relay_feedback(self):
            state = controllers[self._arm_side].snapshot()
            if state.confirmed is not None and time.monotonic() - state.verified_at <= args.m050_state_timeout:
                # Legacy transport only: this 0/1 is actual relay readback, not a servo position.
                self._relay_feedback.publish(Int32(data=int(state.confirmed)))

        def _parse_named_vector(self, message, label, expected_names):
            legacy = super()._parse_named_vector(message, label, JOINT_NAMES + ("gripper.pos",))
            if legacy is None:
                return None
            value = legacy.pop("gripper.pos")
            states = {side: controller.snapshot() for side, controller in controllers.items()}
            if label == "record_action":
                if value not in (900., 1500.):
                    self.get_logger().error("Unexpected legacy Trigger target; expected 900/1500")
                    return None
                if any(state.desired is None for state in states.values()):
                    return None
                if bool(value == 1500.) != states[self._arm_side].desired:
                    return None  # Wait for a joint sample matching this hand's latest Trigger.
                values = {side: state.desired for side, state in states.items()}
            elif value not in (0., 1.):
                self.get_logger().error("Expected remapped M050 DO feedback, not physical gripper counts")
                return None
            else:
                if any(state.confirmed is None for state in states.values()):
                    return None
                values = {side: state.confirmed for side, state in states.items()}
            legacy.update({SUCTION_NAMES[side]: float(value) for side, value in values.items()})
            return legacy

        def snapshot(self):
            snapshot = super().snapshot()
            relays = {side: controller.snapshot() for side, controller in controllers.items()}
            now = time.monotonic()
            ready = all(relay.desired is not None and relay.confirmed is not None and
                        relay.error is None and relay.applied_sequence == relay.sequence and
                        now - relay.verified_at <= args.m050_state_timeout and
                        now - relay.buttons_at <= args.m050_state_timeout
                        for relay in relays.values())
            if not ready or (snapshot.action is not None and any(
                    snapshot.action[SUCTION_NAMES[side]] != float(relay.desired)
                    for side, relay in relays.items())):
                snapshot.state = snapshot.action = None
            elif snapshot.state is not None:
                snapshot.state.update({SUCTION_NAMES[side]: float(relay.confirmed)
                                       for side, relay in relays.items()})
            return snapshot

    base.PicoDatasetBridge = SuctionDatasetBridge


def self_test():
    """Exercise level mapping, no repeat writes and failure reconciliation in RAM."""
    class FakeClient:
        channel = 2

        def __init__(self, channel=2):
            self.channel = channel
            self.value = False
            self.writes = []
            self.fail_write = False
            self.closed = False

        def read(self):
            return self.value

        def write(self, value):
            self.writes.append(value)
            self.value = value
            if self.fail_write:
                raise RuntimeError("write response lost; output may have changed")
            return value

        def close(self):
            self.closed = True

    def wait_until(predicate):
        deadline = time.monotonic() + 2.
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("worker did not reach the expected state")
            time.sleep(.002)

    client = FakeClient()
    controller = SuctionController(client, poll_seconds=.005, report=lambda text: None)
    controller.start()
    try:
        wait_until(lambda: controller.snapshot().verified_at > 0)
        assert client.writes == []  # Startup queries must not write DO.
        controller.buttons(1)
        wait_until(lambda: controller.snapshot().confirmed is True)
        for buttons in (3, 7, 15, 1):  # Other buttons and a held Trigger do not write.
            controller.buttons(buttons)
        time.sleep(.02)
        assert client.writes == [True]
        controller.buttons(14)
        wait_until(lambda: controller.snapshot().confirmed is False)
        assert client.writes == [True, False]
        client.fail_write = True
        controller.buttons(1)
        wait_until(lambda: len(client.writes) == 3 and controller.snapshot().error is None)
        assert controller.snapshot().confirmed is True  # Recovered by read, without retry.
    finally:
        controller.close(timeout=2.)
    assert client.closed and client.writes == [True, False, True]  # No exit write.
    # Both cups must work for every combination, independently of Grip/A/B.
    clients = {side: FakeClient(channel) for side, channel in CHANNELS.items()}
    controllers = {side: SuctionController(client, poll_seconds=.005, report=lambda text: None)
                   for side, client in clients.items()}
    for controller in controllers.values():
        controller.start()
    try:
        for left, right in ((False, False), (True, False), (True, True),
                            (False, True), (False, False)):
            for index, enabled in enumerate((left, right)):
                packet = bytes([index]) + struct.pack("<9f", 0, 0, 0, 0, 0, 0, 1, 0, 0) + bytes([14 | enabled])
                side, buttons = PicoUdpRelay.parse_buttons(packet)
                controllers[side].buttons(buttons)
            wait_until(lambda: all(controller.snapshot().confirmed == expected and
                                   controller.snapshot().applied_sequence == controller.snapshot().sequence
                                   for controller, expected in zip(controllers.values(), (left, right))))
        assert clients["left"].writes == [False, True, False]
        assert clients["right"].writes == [False, True, False]
        assert PicoUdpRelay.parse_buttons(b"bad") is None
    finally:
        for controller in controllers.values():
            controller.close(timeout=2.)
    args, _ = parse_suction_args([])
    command = teleop_command(args)
    assert "/sts3215/position:=/xcore/right/suction_source/unused_gripper_command" in command
    assert "/sts3215/state:=/xcore/right/suction_source/relay_feedback" in command
    assert "/xcore/right/pico_buttons:=/xcore/right/suction_source/pico_buttons" in command
    print("Dual suction: left Trigger=DO1, right Trigger=DO2; self-test passed (no hardware IO)")


def main(argv=None):
    args, remaining = parse_suction_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.print_teleop_command or args.teleop_only:
        if args.arm_side == "left" and args.pico_executable.name == "pico_servol":
            raise SystemExit("The bundled pico_servol is right-arm only; supply a matching left-arm "
                             "--pico-executable publishing /xcore/left/* for --arm-side left")
        command = teleop_command(args)
        if args.teleop_only:
            if remaining:
                raise SystemExit("Recorder options belong in the recording terminal, without --teleop-only")
            return run_teleop(args)
        print(shlex.join(command))
        return 0
    base = load_recorder()
    with ExitStack() as cleanup:
        controllers = {}
        for side, channel in CHANNELS.items():
            client = M050Client(args, channel)
            controller = SuctionController(client, poll_seconds=args.m050_poll_seconds)
            try:
                controller.start()
            except BaseException:
                client.close()
                raise
            cleanup.callback(controller.close, timeout=2 * args.m050_timeout_ms / 1000. + 1.)
            controllers[side] = controller
        relay = PicoUdpRelay(lambda side, buttons: controllers[side].buttons(buttons),
                             host="127.0.0.1", port=args.pico_trigger_port, forward_port=None)
        try:
            relay.start()
        except BaseException:
            relay.socket.close()
            raise
        cleanup.callback(relay.close)
        old_argv = sys.argv
        try:
            configure_recorder(base, args, controllers)
            sys.argv = [str(Path(__file__)), "--arm-side", args.arm_side, *remaining]
            print(f"[SUCTION] {args.m050_host}:{args.m050_port}, unit={args.m050_unit}; "
                  "left Trigger=DO1, right Trigger=DO2; pressed=ON/closed, released=OFF/open")
            print(f"[PICO] Local Trigger mirror: 127.0.0.1:UDP {relay.port}; "
                  f"both controllers must target robot IP:UDP {args.pico_input_port}")
            print("Start this script with --teleop-only in the other terminal, "
                  "replacing the old PICO launch.")
            return base.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
