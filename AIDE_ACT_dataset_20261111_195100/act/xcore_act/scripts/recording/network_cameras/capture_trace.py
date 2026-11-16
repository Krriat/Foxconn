"""Optional bounded background trace; records metadata, never camera pixels."""
from pathlib import Path
import json
import queue
import threading
import time
import uuid


class CaptureTrace:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('capture_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
                                  + '_' + uuid.uuid4().hex[:8] + '.jsonl')
        self.output = self.path.open('x', buffering=1024 * 1024)
        self.jobs = queue.Queue(maxsize=4096)
        self.stopping = threading.Event()
        self.dropped = self.written = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, name='camera-metadata-trace', daemon=True)
        self.thread.start()
        print('PC_CAPTURE_TRACE=' + str(self.path), flush=True)

    def event(self, stage, **fields):
        if self.stopping.is_set() or self.error:
            return False
        record = dict(stage=stage, trace_utc_ns=time.time_ns(),
                      trace_monotonic_ns=time.monotonic_ns(), **fields)
        try:
            self.jobs.put_nowait(record)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def run(self):
        last_flush = time.monotonic()
        try:
            while not self.stopping.is_set() or not self.jobs.empty():
                try:
                    record = self.jobs.get(timeout=.1)
                except queue.Empty:
                    record = None
                if record is not None:
                    self.output.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
                    self.written += 1
                if time.monotonic() - last_flush >= 1:
                    self.output.flush()
                    last_flush = time.monotonic()
            self.output.write(json.dumps(dict(stage='trace_summary', written=self.written,
                                              dropped=self.dropped)) + '\n')
        except Exception as exc:
            self.error = str(exc)
            print('PC_CAPTURE_TRACE_ERROR=' + self.error, flush=True)
        finally:
            self.output.close()

    def close(self):
        self.stopping.set()
        self.thread.join(5.)
        if self.thread.is_alive():
            print('PC_CAPTURE_TRACE_INCOMPLETE: writer did not stop within 5 seconds', flush=True)
        elif self.dropped:
            print(f'PC_CAPTURE_TRACE_INCOMPLETE: {self.dropped} metadata records dropped', flush=True)
