"""Optional callback timestamps; bounded asynchronous metadata writes, no hardware."""
import json
import os
from pathlib import Path
import queue
import threading
import time


class FeedbackTrace:
    def __init__(self, directory=None):
        self.enabled = False
        self.dropped = 0
        self.error = None
        self._queue = queue.Queue(maxsize=16384)
        self._stop = threading.Event()
        self._thread = None
        self.path = None
        if not directory:
            return
        self.path = Path(directory) / f"receive_{os.getpid()}.jsonl"
        try:
            self._output = self.path.open("x", buffering=1024 * 1024)
        except OSError as exc:
            self.error = str(exc)
            return
        self.enabled = True
        self._thread = threading.Thread(target=self._write, name="feedback-trace", daemon=True)
        self._thread.start()

    def emit(self, stream, stamp_ns=0, callback_ns=None, **fields):
        if not self.enabled:
            return
        try:
            now = time.monotonic_ns()
            self._queue.put_nowait(dict(kind="receive", stream=stream, stamp_ns=int(stamp_ns),
                mono_ns=now if callback_ns is None else callback_ns, done_ns=now,
                utc_ns=time.time_ns(), **fields))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        flushed = time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=.1)
                except queue.Empty:
                    row = None
                if row is not None:
                    self._output.write(json.dumps(row, separators=(",", ":")) + "\n")
                if time.monotonic() - flushed > 1.:
                    self._output.flush()
                    flushed = time.monotonic()
            self._output.write(json.dumps(dict(kind="trace_end", dropped=self.dropped)) + "\n")
        except OSError as exc:
            self.error = str(exc)
        finally:
            self._output.close()

    def close(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(5.)
            if self._thread.is_alive():
                self.error = "trace writer did not finish within 5 seconds"
        self.enabled = False
