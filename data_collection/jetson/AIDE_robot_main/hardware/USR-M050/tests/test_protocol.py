#!/usr/bin/env python3
"""Protocol tests use only a loopback simulator; never write to real hardware."""
import concurrent.futures
import ctypes as C
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

LIBRARY, CLI, STATIC, CC, INCLUDE = sys.argv[1:6]
del sys.argv[1:6]


class Config(C.Structure):
    _fields_ = [("host", C.c_char_p), ("port", C.c_uint16),
                ("unit_id", C.c_uint8), ("timeout_ms", C.c_int), ("source_ip", C.c_char_p)]


class Channels(C.Structure):
    _fields_ = [("ch1", C.c_uint8), ("ch2", C.c_uint8)]


class State(C.Structure):
    _fields_ = [("di", Channels), ("do_state", Channels)]


class Error(C.Structure):
    _fields_ = [("code", C.c_int), ("system_errno", C.c_int),
                ("modbus_exception", C.c_uint8), ("write_may_have_applied", C.c_int),
                ("message", C.c_char * 192)]


lib = C.CDLL(LIBRARY)
lib.m050_config_default.argtypes = [C.POINTER(Config)]
lib.m050_config_default.restype = None
lib.m050_create.argtypes = [C.POINTER(Config), C.POINTER(C.c_void_p), C.POINTER(Error)]
lib.m050_create.restype = C.c_int
lib.m050_destroy.argtypes = [C.c_void_p]
lib.m050_destroy.restype = None
lib.m050_set_connection_reuse.argtypes = [C.c_void_p, C.c_int, C.POINTER(Error)]
lib.m050_set_connection_reuse.restype = C.c_int
for name in ("m050_read_do", "m050_read_di"):
    getattr(lib, name).argtypes = [C.c_void_p, C.POINTER(Channels), C.POINTER(Error)]
    getattr(lib, name).restype = C.c_int
lib.m050_read_io.argtypes = [C.c_void_p, C.POINTER(State), C.POINTER(Error)]
lib.m050_read_io.restype = C.c_int
lib.m050_set_do.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.POINTER(Channels), C.POINTER(Error)]
lib.m050_set_do.restype = C.c_int


def exact(sock, size):
    result = b""
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            return None
        result += part
    return result


class Handler(socketserver.BaseRequestHandler):
    def setup(self):
        with self.server.guard:
            self.server.connections.append(self.client_address)

    def handle(self):
        try:
            self.request.settimeout(2)
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
                    self.server.requests.append((tid, fc, address, value))
                    if fc in (1, 2):
                        assert address == 0 and value == 2
                        bits = self.server.outputs if fc == 1 else self.server.inputs
                        response = bytes([fc, 1, bits])
                    elif fc == 5:
                        assert address in (0, 1) and value in (0, 0xFF00)
                        if self.server.mode != "readback_bad":
                            bit = 1 << address
                            self.server.outputs = ((self.server.outputs | bit) if value
                                                   else (self.server.outputs & ~bit))
                        response = pdu
                    else:
                        raise AssertionError("unexpected request")
                mode = self.server.mode
                if mode == "silent" or (mode == "readback_timeout" and fc == 1):
                    self.server.stop.wait(1)
                    return
                if mode == "slow_drip":
                    packet = struct.pack(">HHHB", tid, 0, len(response) + 1, unit) + response
                    for byte in packet:
                        self.request.sendall(bytes([byte]))
                        if self.server.stop.wait(0.035):
                            return
                    continue
                if mode == "close_header":
                    self.request.sendall(header[:3])
                    return
                if mode in ("exception", "bad_exception") or (mode == "di_exception" and fc == 2):
                    response = bytes([fc | 0x80, 2]) + (b"\x00" if mode == "bad_exception" else b"")
                if mode == "wrong_fc":
                    response = bytes([3]) + response[1:]
                if mode == "bad_count" and fc in (1, 2):
                    response = bytes([fc, 2, 0])
                if mode == "short_status" and fc in (1, 2):
                    response = bytes([fc, 1])
                if mode == "long_status" and fc in (1, 2):
                    response += b"\x00"
                if mode == "echo_bad" and fc == 5:
                    response = pdu[:-1] + b"\x01"
                response_header = struct.pack(">HHHB", tid ^ (1 if mode == "bad_tid" else 0),
                                              1 if mode == "bad_protocol" else 0,
                                              len(response) + 1,
                                              unit + (1 if mode == "bad_unit" else 0))
                if mode.startswith("length_"):
                    response_header = response_header[:4] + struct.pack(">H", int(mode[7:])) + response_header[6:]
                    self.request.sendall(response_header)
                    return
                if mode == "close_body":
                    self.request.sendall(response_header + response[:1])
                    return
                packet = response_header + response
                if mode == "fragment":
                    for byte in packet:
                        self.request.sendall(bytes([byte]))
                        time.sleep(0.001)
                else:
                    self.request.sendall(packet)
                if mode == "close_after_response":
                    return
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as error:
            self.server.errors.append(error)


class Simulator(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, mode="normal", outputs=0, inputs=2):
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode, self.outputs, self.inputs = mode, outputs, inputs
        self.requests, self.errors, self.connections = [], [], []
        self.guard = threading.Lock()
        self.stop = threading.Event()
        self.connection_closed = threading.Event()

    def shutdown_request(self, request):
        super().shutdown_request(request)
        self.connection_closed.set()

    def __enter__(self):
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.shutdown()
        self.server_close()
        self.thread.join(2)
        if self.errors and args[0] is None:
            raise self.errors[0]

    def cli(self, *args, timeout_ms=500):
        return subprocess.run([CLI, "--host", "127.0.0.1", "--port", str(self.server_address[1]),
                               "--timeout", str(timeout_ms), *args],
                              capture_output=True, text=True, timeout=3)


class ProtocolTest(unittest.TestCase):
    def client(self, server, timeout=500, source=None):
        config = Config()
        lib.m050_config_default(C.byref(config))
        config.host = b"127.0.0.1"
        config.port = server.server_address[1]
        config.timeout_ms = timeout
        config.source_ip = source
        handle, error = C.c_void_p(), Error()
        self.assertEqual(lib.m050_create(C.byref(config), C.byref(handle), C.byref(error)), 0)
        self.addCleanup(lib.m050_destroy, handle)
        return handle

    def test_defaults(self):
        config = Config()
        lib.m050_config_default(C.byref(config))
        self.assertEqual((config.host, config.port, config.unit_id, config.timeout_ms),
                         (b"192.168.1.7", 20108, 1, 3000))
        lib.m050_config_default(None)
        lib.m050_destroy(None)

    def test_connection_reuse_is_lazy_and_retains_successful_socket(self):
        with Simulator() as server:
            handle = self.client(server)
            error, state = Error(), Channels()
            self.assertEqual(lib.m050_set_connection_reuse(handle, 1, C.byref(error)), 0)
            self.assertEqual(server.connections, [])
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), 0)
            self.assertEqual(lib.m050_set_do(handle, 1, 1, C.byref(state), C.byref(error)), 0)
            self.assertEqual(lib.m050_read_di(handle, C.byref(state), C.byref(error)), 0)
            self.assertEqual(len(server.connections), 1)
            self.assertEqual([r[1] for r in server.requests], [1, 5, 1, 2])
            self.assertEqual([r[0] for r in server.requests], [1, 2, 3, 4])

    def test_disabling_reuse_closes_retained_socket(self):
        with Simulator() as server:
            handle = self.client(server)
            state = Channels()
            self.assertEqual(lib.m050_set_connection_reuse(handle, 1, None), 0)
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), None), 0)
            self.assertEqual(lib.m050_set_connection_reuse(handle, 0, None), 0)
            self.assertTrue(server.connection_closed.wait(1))
            self.assertEqual(lib.m050_read_di(handle, C.byref(state), None), 0)
            self.assertEqual(len(server.connections), 2)

    def test_reuse_reconnects_idle_closed_peer_before_sending(self):
        with Simulator("close_after_response") as server:
            handle = self.client(server)
            state = Channels()
            self.assertEqual(lib.m050_set_connection_reuse(handle, 1, None), 0)
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), None), 0)
            self.assertTrue(server.connection_closed.wait(1))
            server.mode = "normal"
            self.assertEqual(lib.m050_read_di(handle, C.byref(state), None), 0)
            self.assertEqual(len(server.connections), 2)
            self.assertEqual([r[1] for r in server.requests], [1, 2])

    def test_reuse_drops_failed_socket_without_replaying_write(self):
        with Simulator() as server:
            handle = self.client(server)
            state, error = Channels(), Error()
            self.assertEqual(lib.m050_set_connection_reuse(handle, 1, None), 0)
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), None), 0)
            server.mode = "readback_bad"
            self.assertEqual(lib.m050_set_do(handle, 1, 1, C.byref(state), C.byref(error)), -9)
            self.assertEqual(error.write_may_have_applied, 1)
            self.assertEqual([r[1] for r in server.requests], [1, 5, 1])
            server.mode = "normal"
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), 0)
            self.assertEqual(len(server.connections), 2)
            self.assertEqual([r[1] for r in server.requests], [1, 5, 1, 1])
            self.assertEqual(bytes(error), bytes(Error()))

    def test_connection_reuse_rejects_invalid_settings(self):
        with Simulator() as server:
            handle = self.client(server)
            self.assertEqual(lib.m050_set_connection_reuse(handle, 2, None), -1)
            self.assertEqual(lib.m050_set_connection_reuse(None, 1, None), -1)
            self.assertEqual(server.connections, [])

    def test_reads_all_bit_patterns(self):
        for bits in range(4):
            with self.subTest(bits=bits), Simulator(outputs=bits, inputs=3-bits) as server:
                handle = self.client(server)
                state, error = State(), Error()
                self.assertEqual(lib.m050_read_io(handle, C.byref(state), C.byref(error)), 0)
                self.assertEqual((state.do_state.ch1, state.do_state.ch2), (bits & 1, bits >> 1))
                self.assertEqual((state.di.ch1, state.di.ch2), ((3-bits) & 1, (3-bits) >> 1))
                self.assertEqual([r[1] for r in server.requests], [1, 2])

    def test_both_relays_on_off_and_preserve_other_channel(self):
        with Simulator() as server:
            handle = self.client(server)
            state, error = Channels(), Error()
            for channel, closed, expected in ((1, 1, (1, 0)), (2, 1, (1, 1)),
                                               (1, 0, (0, 1)), (2, 0, (0, 0))):
                self.assertEqual(lib.m050_set_do(handle, channel, closed, C.byref(state), C.byref(error)), 0)
                self.assertEqual((state.ch1, state.ch2), expected)
                self.assertEqual(error.write_may_have_applied, 0)
            self.assertEqual([r[1] for r in server.requests], [5, 1] * 4)

    def test_fragmented_responses(self):
        with Simulator("fragment") as server:
            handle = self.client(server, timeout=1500)
            state, error = State(), Error()
            self.assertEqual(lib.m050_read_io(handle, C.byref(state), C.byref(error)), 0)
            self.assertEqual(lib.m050_set_do(handle, 2, 1, None, C.byref(error)), 0)

    def test_reject_invalid_frames(self):
        for mode in ("bad_tid", "bad_protocol", "bad_unit", "length_0", "length_1", "length_255",
                     "wrong_fc", "bad_count", "short_status", "long_status", "bad_exception"):
            with self.subTest(mode=mode), Simulator(mode) as server:
                handle = self.client(server)
                state, error = Channels(9, 8), Error()
                self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), -7)
                self.assertEqual((state.ch1, state.ch2), (9, 8))
                self.assertEqual((error.code, error.write_may_have_applied), (-7, 0))

    def test_exception_details(self):
        with Simulator("exception") as server:
            handle = self.client(server)
            state, error = Channels(), Error()
            self.assertEqual(lib.m050_read_di(handle, C.byref(state), C.byref(error)), -8)
            self.assertEqual(error.modbus_exception, 2)

    def test_truncated_response(self):
        for mode in ("close_header", "close_body"):
            with self.subTest(mode=mode), Simulator(mode) as server:
                handle = self.client(server)
                state, error = Channels(), Error()
                self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), -6)

    def test_read_timeout(self):
        with Simulator("silent") as server:
            handle = self.client(server, timeout=80)
            state, error = Channels(), Error()
            start = time.monotonic()
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), -5)
            self.assertLess(time.monotonic() - start, 0.6)
            self.assertEqual(error.write_may_have_applied, 0)

    def test_deadline_bounds_slow_fragment_stream(self):
        with Simulator("slow_drip") as server:
            handle = self.client(server, timeout=90)
            state, error = Channels(), Error()
            start = time.monotonic()
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), -5)
            self.assertLess(time.monotonic() - start, 0.3)

    def test_write_failures_report_uncertainty_without_retry(self):
        for mode, expected, count in (("silent", -5, 1), ("readback_timeout", -5, 2),
                                       ("readback_bad", -9, 2), ("echo_bad", -7, 1),
                                       ("exception", -8, 1), ("close_body", -6, 1)):
            with self.subTest(mode=mode), Simulator(mode) as server:
                handle = self.client(server, timeout=90)
                state, error = Channels(9, 8), Error()
                self.assertEqual(lib.m050_set_do(handle, 1, 1, C.byref(state), C.byref(error)), expected)
                self.assertEqual(error.write_may_have_applied, 1)
                self.assertEqual((state.ch1, state.ch2), (9, 8))
                self.assertEqual(len(server.requests), count)

    def test_read_io_does_not_publish_partial_state(self):
        with Simulator("di_exception") as server:
            handle = self.client(server)
            state, error = State(Channels(9, 8), Channels(7, 6)), Error()
            self.assertEqual(lib.m050_read_io(handle, C.byref(state), C.byref(error)), -8)
            self.assertEqual(bytes(state), bytes(State(Channels(9, 8), Channels(7, 6))))
            self.assertEqual([r[1] for r in server.requests], [1, 2])

    def test_invalid_config(self):
        for field, value in (("host", b"localhost"), ("host", None), ("port", 0),
                             ("unit_id", 0), ("unit_id", 248), ("timeout_ms", 0),
                             ("timeout_ms", -1), ("source_ip", b"not-an-ip")):
            with self.subTest(field=field, value=value):
                config, handle, error = Config(), C.c_void_p(), Error()
                lib.m050_config_default(C.byref(config))
                setattr(config, field, value)
                self.assertEqual(lib.m050_create(C.byref(config), C.byref(handle), C.byref(error)), -1)
                self.assertFalse(handle.value)
        self.assertEqual(lib.m050_create(None, None, None), -1)
        handle = C.c_void_p()
        self.assertEqual(lib.m050_create(None, C.byref(handle), None), 0)
        lib.m050_destroy(handle)

    def test_invalid_call_does_not_connect(self):
        with Simulator() as server:
            handle = self.client(server)
            error, state = Error(), Channels()
            for channel, value in ((0, 1), (3, 1), (1, -1), (2, 2)):
                self.assertEqual(lib.m050_set_do(handle, channel, value, None, C.byref(error)), -1)
            self.assertEqual(lib.m050_read_do(handle, None, C.byref(error)), -1)
            self.assertEqual(lib.m050_read_do(None, C.byref(state), C.byref(error)), -1)
            self.assertEqual(server.requests, [])

    def test_connection_refused_before_write(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            class Endpoint:
                server_address = reserved.getsockname()
            handle = self.client(Endpoint())
            error = Error()
            self.assertEqual(lib.m050_set_do(handle, 1, 1, None, C.byref(error)), -4)
            self.assertNotEqual(error.system_errno, 0)
            self.assertEqual(error.write_may_have_applied, 0)

    def test_bind_local_source(self):
        with Simulator() as server:
            handle = self.client(server, source=b"127.0.0.1")
            state = Channels()
            self.assertEqual(lib.m050_read_di(handle, C.byref(state), None), 0)

    def test_same_handle_concurrency(self):
        with Simulator() as server:
            handle = self.client(server)
            def read(_):
                state, error = State(), Error()
                return lib.m050_read_io(handle, C.byref(state), C.byref(error))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                self.assertEqual(list(pool.map(read, range(24))), [0] * 24)
            self.assertEqual([r[1] for r in server.requests], [1, 2] * 24)
            tids = [r[0] for r in server.requests]
            self.assertEqual(tids, list(range(1, 49)))

    def test_error_is_cleared_after_success(self):
        with Simulator("exception") as server:
            handle = self.client(server)
            state, error = Channels(), Error()
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), -8)
            server.mode = "normal"
            self.assertEqual(lib.m050_read_do(handle, C.byref(state), C.byref(error)), 0)
            self.assertEqual(bytes(error), bytes(Error()))

    def test_cpp_cli_default_and_writes(self):
        with Simulator(inputs=1) as server:
            result = server.cli()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "DO1=0 DO2=0\nDI1=1 DI2=0\n")
            for channel in (1, 2):
                for value in ("on", "off"):
                    result = server.cli("set-do", str(channel), value)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpp_cli_exception(self):
        with Simulator("readback_bad") as server:
            result = server.cli("set-do", "1", "on")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Write may already have applied", result.stderr)

    def test_cli_rejects_bad_arguments_without_io(self):
        with Simulator() as server:
            for args in (("set-do", "3", "on"), ("set-do", "1", "bad"),
                         ("status", "extra"), ("set-do", "1"), ("--unit", "256"),
                         ("--port", "-1"), ("--timeout", "0"), ("--bad",)):
                with self.subTest(args=args):
                    self.assertEqual(server.cli(*args).returncode, 2)
            self.assertEqual(server.requests, [])

    def test_static_c_consumer(self):
        source = r'''
#include <m050/m050.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    m050_config config;
    m050_client *client = NULL;
    m050_io_state state;
    m050_error error;
    int rc;
    if (argc != 2) return 10;
    m050_config_default(&config);
    config.host = "127.0.0.1";
    config.port = (uint16_t)atoi(argv[1]);
    rc = m050_create(&config, &client, &error);
    if (!rc) rc = m050_read_io(client, &state, &error);
    if (!rc) rc = m050_set_do(client, 2, 1, NULL, &error);
    m050_destroy(client);
    return rc ? 11 : 0;
}
'''
        with tempfile.TemporaryDirectory() as temp, Simulator() as server:
            src, exe = Path(temp) / "consumer.c", Path(temp) / "consumer"
            src.write_text(source)
            subprocess.run([CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-I", INCLUDE,
                            str(src), STATIC, "-pthread", "-o", str(exe)], check=True, capture_output=True)
            result = subprocess.run([str(exe), str(server.server_address[1])], timeout=3)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(server.outputs, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
