from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np


COMMANDS = {
    "start",
    "pause",
    "resume",
    "save_episode",
    "discard_episode",
    "stop_session",
    "abort_session",
}


class ExternalControlServer:
    """Receive non-blocking local commands from xcore_edge_api."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(str(self.path))
        self.socket.setblocking(False)

    def drain(self) -> list[str]:
        commands: list[str] = []
        while True:
            try:
                payload = self.socket.recv(4096)
            except BlockingIOError:
                return commands
            try:
                command = json.loads(payload.decode("utf-8")).get("command")
            except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if command in COMMANDS:
                commands.append(command)

    def close(self) -> None:
        self.socket.close()
        self.path.unlink(missing_ok=True)


class AtomicStatusWriter:
    def __init__(self, path: Path, minimum_interval_s: float = 0.15):
        self.path = path
        self.minimum_interval_s = minimum_interval_s
        self.last_write = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, payload: dict, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_write < self.minimum_interval_s:
            return
        value = {
            **payload,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        self.last_write = now


class WebPreviewPublisher:
    """Atomically publish the same BGR frames used by the recorder."""

    def __init__(
        self,
        directory: Path,
        *,
        width: int = 960,
        jpeg_quality: int = 70,
        publish_every: int = 1,
    ):
        self.directory = directory
        self.width = width
        self.jpeg_quality = jpeg_quality
        self.publish_every = max(1, publish_every)
        self.frame_count = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish(self, frames: dict[str, np.ndarray]) -> None:
        self.frame_count += 1
        if self.frame_count % self.publish_every:
            return
        for camera, frame_bgr in frames.items():
            image = frame_bgr
            height, width = image.shape[:2]
            if width > self.width:
                target_height = max(1, round(height * self.width / width))
                image = cv2.resize(
                    image,
                    (self.width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not ok:
                continue
            destination = self.directory / f"{camera}.jpg"
            temporary = self.directory / f".{camera}.jpg.tmp"
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(destination)
