"""Measured 7-joint feedback over loopback UDP; no robot or IO commands."""
from dataclasses import dataclass
import math
import socket
import struct
import threading
import time

DEFAULT_PORT = 29701
PACKET = struct.Struct("<8sBBHI6Q7d")


@dataclass(frozen=True)
class JointPacket:
    side: str
    source: str
    pid: int
    session: int
    sequence: int
    stamp_ns: int
    mono_ns: int
    send_errors: int
    ros_overwrites: int
    joints: tuple


def decode_packet(data):
    if len(data) != PACKET.size:
        raise ValueError("invalid joint packet length")
    magic, side, source, reserved, pid, session, seq, stamp, mono, errors, overwritten, *joints = PACKET.unpack(data)
    if (magic != b"AIDEJNT1" or side not in (0,1) or source not in (0,1) or reserved or
            not all(v > 0 for v in (pid,session,seq,stamp,mono)) or
            not all(math.isfinite(v) and abs(v) <= 2*math.pi for v in joints)):
        raise ValueError("invalid measured joint packet")
    return JointPacket(("left","right")[side], ("drag","recording")[source],
                       pid,session,seq,stamp,mono,errors,overwritten,tuple(joints))


class LocalFeedbackReceiver:
    def __init__(self, callback, port=DEFAULT_PORT, max_age=.5):
        self.callback, self.max_age = callback, max_age
        self.lock = threading.Lock()
        self.producer = None
        self.error = None
        self.invalid_packets = self.stale_packets = 0
        self.streams = {}
        self._stop = threading.Event()
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024*1024)
            # No SO_REUSEADDR: one collector owns this port, preventing split delivery.
            self.socket.bind(("127.0.0.1",port))
            self.socket.settimeout(.1)
        except BaseException:
            self.socket.close()
            raise
        self.port = self.socket.getsockname()[1]
        self.thread = threading.Thread(target=self._receive,name="local-arm-feedback",daemon=True)
        self.thread.start()

    def _receive(self):
        while not self._stop.is_set():
            try:
                data,address = self.socket.recvfrom(4096)
                now = time.monotonic_ns()
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self.error = str(exc)
                return
            try:
                if address[0] != "127.0.0.1":
                    raise ValueError("joint feedback is local only")
                p = decode_packet(data)
            except ValueError:
                with self.lock:
                    self.invalid_packets += 1
                continue
            latency = (now-p.mono_ns)/1e9
            if not -.001 <= latency <= self.max_age:
                with self.lock:
                    self.stale_packets += 1
                continue
            key = p.side+"."+p.source
            with self.lock:
                identity = p.pid,p.session
                if self.producer is None:
                    self.producer = identity
                if self.producer != identity:
                    self.error = "multiple/restarted local joint producers; restart the collector"
                    continue
                previous = self.streams.get(key)
                if previous and (p.sequence <= previous["last_sequence"] or
                                 p.stamp_ns <= previous["last_stamp_ns"]):
                    previous["out_of_order"] += 1
                    continue
                if previous is None:
                    previous = self.streams[key] = dict(received=0,missing_sequences=0,
                        out_of_order=0,max_latency_ms=0.,last_sequence=p.sequence-1,
                        initial_sender_send_errors=p.send_errors)
                previous["missing_sequences"] += p.sequence-previous["last_sequence"]-1
                previous.update(received=previous["received"]+1,last_sequence=p.sequence,
                    last_stamp_ns=p.stamp_ns,last_received_ns=now,
                    max_latency_ms=max(previous["max_latency_ms"],latency*1000),
                    sender_send_errors=p.send_errors,ros_queue_overwrites=p.ros_overwrites)
                previous["sender_send_errors_since_first"] = p.send_errors-previous["initial_sender_send_errors"]
            if self.error:
                continue
            try:
                self.callback(p,now/1e9)
            except Exception as exc:
                self.error = "joint receive callback failed: "+str(exc)

    def diagnostics(self):
        with self.lock:
            return dict(transport="loopback_udp_v1",port=self.port,producer=self.producer,
                error=self.error,invalid_packets=self.invalid_packets,stale_packets=self.stale_packets,
                streams={k:dict(v) for k,v in self.streams.items()})

    def ready(self):
        now = time.monotonic_ns()
        with self.lock:
            return not self.error and all(any(k.startswith(side+".") and
                now-v["last_received_ns"] < 250000000 for k,v in self.streams.items())
                for side in ("left","right"))

    def close(self):
        self._stop.set()
        self.thread.join(1.)
        self.socket.close()
