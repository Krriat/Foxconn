"""Exercise the compiled ROS node against a loopback-only Modbus device."""
import itertools
import os
import signal
import socket
import socketserver
import struct
import subprocess
import tempfile
import threading
import time

os.environ["ROS_DOMAIN_ID"] = str(80 + os.getpid() % 20)
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

import pytest
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from m050_ros.msg import IOState
from m050_ros.srv import GetIO, SetDO

EXECUTABLE = os.environ["M050_NODE_EXECUTABLE"]
COUNTER = itertools.count()


def exact(sock, size):
    result = b""
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            return None
        result += part
    return result


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            self.request.settimeout(1)
            while True:
                header = exact(self.request, 7)
                if header is None:
                    return
                tid, protocol, length, unit = struct.unpack(">HHHB", header)
                assert protocol == 0 and length == 6 and unit == 1
                pdu = exact(self.request, length - 1)
                if pdu is None:
                    return
                fc, address, value = struct.unpack(">BHH", pdu)
                with self.server.guard:
                    mode = self.server.mode
                    self.server.requests.append((fc, address, value))
                    if fc in (1, 2):
                        assert address == 0 and value == 2
                        bits = self.server.outputs if fc == 1 else self.server.inputs
                        response = bytes([fc, 1, bits])
                    else:
                        assert fc == 5 and address in (0, 1) and value in (0, 0xFF00)
                        if mode != "readback_bad":
                            bit = 1 << address
                            self.server.outputs = ((self.server.outputs | bit) if value
                                                   else (self.server.outputs & ~bit))
                        response = pdu
                if mode == "silent" or (mode == "readback_timeout" and fc == 1):
                    self.server.stop.wait(0.5)
                    return
                if mode == "di_exception" and fc == 2:
                    response = bytes([fc | 0x80, 2])
                if mode == "echo_bad" and fc == 5:
                    response = pdu[:-1] + b"\x01"
                self.request.sendall(
                    struct.pack(">HHHB", tid, 0, len(response) + 1, unit) + response)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as error:
            self.server.errors.append(error)


class Simulator(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.guard = threading.Lock()
        self.stop = threading.Event()
        self.mode = "normal"
        self.outputs, self.inputs = 0, 2
        self.requests, self.errors = [], []

    def configure(self, **values):
        with self.guard:
            for key, value in values.items():
                setattr(self, key, value)


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def device():
    server = Simulator()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    yield server
    server.stop.set()
    server.shutdown()
    server.server_close()
    thread.join(2)
    assert not server.errors


class NodeHarness:
    def __init__(self, device, poll_ms=0, use_launch=False):
        self.namespace = f"/m050_test_{os.getpid()}_{next(COUNTER)}"
        self.log = tempfile.TemporaryFile(mode="w+t")
        command = [
            EXECUTABLE, "--ros-args", "-r", f"__ns:={self.namespace}",
            "-p", "host:=127.0.0.1", "-p", f"port:={device.server_address[1]}",
            "-p", "timeout_ms:=120", "-p", f"poll_interval_ms:={poll_ms}",
        ]
        if use_launch:
            command = [
                "ros2", "launch", "m050_ros", "m050.launch.py", "host:=127.0.0.1",
                f"port:={device.server_address[1]}", "timeout_ms:=120",
                f"poll_interval_ms:={poll_ms}", f"namespace:={self.namespace}",
            ]
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        self.node = rclpy.create_node(f"client_{next(COUNTER)}")
        self.get_io = self.node.create_client(GetIO, self.namespace + "/m050/get_io")
        self.set_do = self.node.create_client(SetDO, self.namespace + "/m050/set_do")
        self.states = []
        self.subscription = self.node.create_subscription(
            IOState, self.namespace + "/m050/state", self.states.append, 10)

    def __enter__(self):
        if not self.get_io.wait_for_service(timeout_sec=6):
            self.__exit__(None, None, None)
            pytest.fail("Node did not expose get_io")
        assert self.set_do.wait_for_service(timeout_sec=3)
        return self

    def __exit__(self, *args):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.node.destroy_node()
        self.log.seek(0)
        output = self.log.read()
        self.log.close()
        if args[0] is None:
            assert self.process.returncode == 0, output

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=4)
        assert future.done(), "Service did not respond"
        return future.result()

    def wait_until(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self.node, timeout_sec=0.02)
        assert predicate(), "Expected state was not published"


def test_start_stop_and_disabled_poll_do_not_access_device(device):
    with NodeHarness(device):
        time.sleep(0.15)
        assert device.requests == []
    assert device.requests == []


def test_launch_file_parameters_namespace_and_read_only_startup(device):
    with NodeHarness(device, use_launch=True) as ros:
        assert not device.requests
        state = ros.call(ros.get_io, GetIO.Request()).state
        assert state.valid
    assert [r[0] for r in device.requests] == [1, 2]


def test_fresh_read_and_state_topic_all_bit_patterns(device):
    with NodeHarness(device) as ros:
        ros.wait_until(lambda: ros.node.count_publishers(ros.namespace + "/m050/state") == 1)
        for bits in range(4):
            device.configure(outputs=bits, inputs=3 - bits)
            state = ros.call(ros.get_io, GetIO.Request()).state
            assert state.valid and state.error_code == 0
            assert list(state.do_state) == [bool(bits & 1), bool(bits & 2)]
            assert list(state.di_state) == [bool((3 - bits) & 1), bool((3 - bits) & 2)]
            ros.wait_until(lambda: any(s.stamp == state.stamp for s in ros.states))
            assert state.stamp.sec > 0
        assert [r[0] for r in device.requests] == [1, 2] * 4


def test_both_outputs_and_preserve_other_channel(device):
    with NodeHarness(device) as ros:
        for channel, closed, expected in [
            (1, True, [True, False]), (2, True, [True, True]),
            (1, False, [False, True]), (2, False, [False, False]),
        ]:
            response = ros.call(ros.set_do, SetDO.Request(channel=channel, closed=closed))
            assert response.success and response.do_state_valid
            assert response.error_code == 0 and not response.write_may_have_applied
            assert list(response.do_state) == expected
    assert [r[0] for r in device.requests] == [5, 1] * 4


def test_invalid_channels_never_reach_modbus(device):
    with NodeHarness(device) as ros:
        for channel in (0, 3, 255):
            response = ros.call(ros.set_do, SetDO.Request(channel=channel, closed=True))
            assert not response.success and not response.do_state_valid
            assert response.error_code == -1 and not response.write_may_have_applied
        assert not device.requests


@pytest.mark.parametrize("mode,code,fcs", [
    ("readback_timeout", -5, [5, 1]),
    ("readback_bad", -9, [5, 1]),
    ("echo_bad", -7, [5]),
])
def test_failed_writes_preserve_uncertainty_without_retry(device, mode, code, fcs):
    device.configure(mode=mode)
    with NodeHarness(device) as ros:
        response = ros.call(ros.set_do, SetDO.Request(channel=1, closed=True))
        assert not response.success and not response.do_state_valid
        assert response.error_code == code and response.write_may_have_applied
        time.sleep(0.15)
        assert [r[0] for r in device.requests] == fcs
        device.configure(mode="normal")
        state = ros.call(ros.get_io, GetIO.Request()).state
        assert state.valid
        assert bool(state.do_state[0]) == (mode != "readback_bad")


def test_partial_read_failure_invalidates_arrays_and_recovers(device):
    device.configure(outputs=3, inputs=3)
    with NodeHarness(device) as ros:
        assert ros.call(ros.get_io, GetIO.Request()).state.valid
        device.configure(mode="di_exception")
        failed = ros.call(ros.get_io, GetIO.Request()).state
        assert not failed.valid and failed.error_code == -8
        assert list(failed.do_state) == [False, False]
        assert list(failed.di_state) == [False, False]
        device.configure(mode="normal")
        recovered = ros.call(ros.get_io, GetIO.Request()).state
        assert recovered.valid and list(recovered.do_state) == [True, True]
        assert all(r[0] in (1, 2) for r in device.requests)


def test_polling_reports_failure_and_recovers_without_writes(device):
    with NodeHarness(device, poll_ms=50) as ros:
        ros.wait_until(lambda: any(s.valid for s in ros.states))
        ros.states.clear()
        device.configure(mode="silent")
        ros.wait_until(lambda: any(not s.valid and s.error_code == -5 for s in ros.states))
        ros.states.clear()
        device.configure(mode="normal", outputs=3, inputs=1)
        ros.wait_until(lambda: any(s.valid and list(s.do_state) == [True, True]
                                  and list(s.di_state) == [True, False] for s in ros.states))
    assert all(r[0] in (1, 2) for r in device.requests)


def test_runtime_connection_parameter_change_is_rejected(device):
    with NodeHarness(device) as ros:
        client = ros.node.create_client(SetParameters, ros.namespace + "/m050/set_parameters")
        assert client.wait_for_service(timeout_sec=3)
        response = ros.call(client, SetParameters.Request(parameters=[Parameter(
            name="host", value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                               string_value="127.0.0.2"))]))
        assert not response.results[0].successful
        assert ros.call(ros.get_io, GetIO.Request()).state.valid


@pytest.mark.parametrize("parameter", [
    "port:=0", "unit_id:=248", "timeout_ms:=0", "poll_interval_ms:=-1",
    "host:=invalid", "source_ip:=invalid",
])
def test_invalid_startup_parameters_fail_before_network(device, parameter):
    result = subprocess.run([
        EXECUTABLE, "--ros-args", "-p", "host:=127.0.0.1",
        "-p", f"port:={device.server_address[1]}", "-p", parameter,
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=4)
    assert result.returncode != 0, result.stdout
    assert not device.requests
