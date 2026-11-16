"""USR-M050 output worker; no network writes until a live round requests one."""
import ctypes as C
import fcntl
import os
from pathlib import Path
import threading


class Config(C.Structure):
    _fields_ = [("host", C.c_char_p), ("port", C.c_uint16), ("unit_id", C.c_uint8),
                ("timeout_ms", C.c_int), ("source_ip", C.c_char_p)]


class Channels(C.Structure):
    _fields_ = [("ch1", C.c_uint8), ("ch2", C.c_uint8)]


class Error(C.Structure):
    _fields_ = [("code", C.c_int), ("system_errno", C.c_int),
                ("modbus_exception", C.c_uint8), ("write_may_have_applied", C.c_int),
                ("message", C.c_char * 192)]


class Suction:
    def __init__(self, config, initial_verified, *, lock_path=None):
        if len(initial_verified) != 2 or any(type(value) is not bool for value in initial_verified):
            raise ValueError("Two fresh verified ROS suction states are required")
        lib = Path(config["robot_root"]) / "hardware/USR-M050/build/libm050.so"
        self.lib = C.CDLL(str(lib))
        self.lib.m050_config_default.argtypes = [C.POINTER(Config)]
        self.lib.m050_create.argtypes = [C.POINTER(Config), C.POINTER(C.c_void_p), C.POINTER(Error)]
        self.lib.m050_read_do.argtypes = [C.c_void_p, C.POINTER(Channels), C.POINTER(Error)]
        self.lib.m050_set_do.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.POINTER(Channels), C.POINTER(Error)]
        self.lib.m050_destroy.argtypes = [C.c_void_p]
        self.client = C.c_void_p()
        settings, error = Config(), Error()
        self.lib.m050_config_default(C.byref(settings))
        settings.timeout_ms = config["io_timeout_ms"]
        self.check(self.lib.m050_create(C.byref(settings), C.byref(self.client), C.byref(error)), error)
        lock_path = lock_path or f"/run/user/{os.getuid()}/robot-m050.lock"
        try:
            self.device_lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        except Exception:
            self.lib.m050_destroy(self.client)
            raise
        self.lock, self.wake, self.stop_event = threading.Lock(), threading.Event(), threading.Event()
        # Reuse the fresh hardware readback already checked by the executor.
        # Opening a second reader here used to collide with the telemetry reader.
        self.verified = list(initial_verified)
        self.run_id, self.target, self.error, self.writes = None, None, "", 0
        self.pending_operation = False
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    @staticmethod
    def check(result, error):
        if result:
            raise RuntimeError(f"M050: {bytes(error.message).decode(errors='replace')}; "
                               f"write_may_have_applied={error.write_may_have_applied}")

    def request(self, run_id, target):
        with self.lock:
            self.run_id, self.target = run_id, list(target)
        self.wake.set()

    def suspend(self):
        # Keep the last physical output, including suction holding an object.
        with self.lock:
            self.run_id, self.target = None, None

    def release_left(self, run_id):
        # Explicit manual stop: cancel queued policy writes, close left DO1,
        # and leave right DO2 alone. A fault/timeout still calls suspend().
        with self.lock:
            self.run_id, self.target, self.error = run_id + ":manual_release", [False, None], ""
        self.wake.set()

    def state(self):
        with self.lock:
            settled = self.target is None or all(target is None or target == actual
                                                 for target, actual in zip(self.target, self.verified))
            return {"verified": list(self.verified), "error": self.error, "writes": self.writes,
                    "pending_operation": self.pending_operation, "settled": settled}

    def worker(self):
        while not self.stop_event.is_set():
            self.wake.wait(0.05)
            self.wake.clear()
            for channel in range(2):
                with self.lock:
                    run_id = self.run_id
                    target = None if self.target is None else self.target[channel]
                    previous = self.verified[channel]
                    if run_id is None or target is None or target == previous:
                        continue
                    self.pending_operation = True
                error, values = Error(), Channels()
                try:
                    fcntl.flock(self.device_lock_fd, fcntl.LOCK_EX)
                    try:
                        # A manual/fault stop can arrive while waiting for the
                        # reader. Never apply a queued target after that stop.
                        with self.lock:
                            if self.run_id != run_id or self.target is None or self.target[channel] != target:
                                continue
                        self.check(self.lib.m050_set_do(self.client, channel + 1, int(target),
                                                       C.byref(values), C.byref(error)), error)
                    finally:
                        fcntl.flock(self.device_lock_fd, fcntl.LOCK_UN)
                    with self.lock:
                        self.verified = [bool(values.ch1), bool(values.ch2)]
                        self.writes += 1
                except Exception as exception:
                    with self.lock:
                        self.error = str(exception)
                        self.run_id, self.target = None, None
                    break
                finally:
                    with self.lock:
                        self.pending_operation = False

    def close(self):
        self.suspend()
        self.stop_event.set()
        self.wake.set()
        self.thread.join()
        self.lib.m050_destroy(self.client)
        os.close(self.device_lock_fd)
