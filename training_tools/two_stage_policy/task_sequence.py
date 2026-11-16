"""Conservative, fixed-camera carton state recognition and completion gates.

This module has no ROS, motion clients, or output writes. Image distances are
template distances, not calibrated probabilities. Unknown scenes fail closed.
"""
from collections import deque
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

SCENES = ("closed", "far_open", "both_open")


def feature(bgr):
    if bgr is None or bgr.shape != (480, 640, 3):
        raise ValueError("Task detector requires the original 640x480 head camera")
    # Box and both flaps, excluding most of the robot/workbench background.
    crop = bgr[100:480, 120:560]
    return (cv2.resize(crop, (44, 38), interpolation=cv2.INTER_AREA)
            .astype(np.float32).reshape(-1) / 255.0)


class SceneDetector:
    def __init__(self, path):
        self.path = Path(path)
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as data:
            self.templates = data["features"].copy()
            self.labels = data["labels"].copy()
            self.end_q = data["end_q"].copy()
            self.end_stage = data["end_stage"].copy()
            self.max_distance = float(data["max_distance"])
            self.min_margin = float(data["min_margin"])

    def classify(self, bgr):
        distances = np.sqrt(np.mean((self.templates - feature(bgr))**2, axis=1))
        scores = [float(distances[self.labels == i].min()) for i in range(3)]
        order = np.argsort(scores)
        distance, margin = scores[order[0]], scores[order[1]] - scores[order[0]]
        known = distance <= self.max_distance and margin >= self.min_margin
        return {"scene": SCENES[order[0]] if known else "unknown",
                "distance": distance, "margin": margin,
                "distances": dict(zip(SCENES, scores)), "detector_sha256": self.sha}

    def endpoint_distance(self, stage, q):
        rows = self.end_q[self.end_stage == (0 if stage == "far" else 1)]
        return float(np.min(np.max(np.abs(rows - np.asarray(q)), axis=1)))


class CompletionGate:
    """Requires a fresh visual change, suction cycle, retreat and steady joints."""
    def __init__(self, stage, now, *, hold=1.0, min_seconds=5.0,
                 max_joint_range=.015, max_endpoint_distance=.12):
        if stage not in ("far", "near"):
            raise ValueError("stage must be far or near")
        self.stage, self.started = stage, now
        self.hold, self.min_seconds = hold, min_seconds
        self.max_joint_range, self.max_endpoint_distance = max_joint_range, max_endpoint_distance
        self.saw_suction = False
        self.samples = deque()
        self.last_stamp = 0
        self.reason = "waiting for stage motion"

    def update(self, now, stamp, scene, q, suction, endpoint_distance, target_off):
        self.saw_suction |= bool(suction[0])
        if stamp <= self.last_stamp:
            self.samples.clear()
            self.reason = "duplicate or reordered head image"
            return False
        self.last_stamp = stamp
        expected = "far_open" if self.stage == "far" else "both_open"
        if (not self.saw_suction or now-self.started < self.min_seconds or scene != expected
                or any(suction) or not target_off or
                (self.stage == "far" and endpoint_distance > self.max_endpoint_distance)):
            self.samples.clear()
            self.reason = "waiting for open flap, released suction and demonstrated retreat pose"
            return False
        if self.samples and now-self.samples[-1][0] > .3:
            self.samples.clear()
        self.samples.append((now, np.asarray(q, dtype=float)))
        while len(self.samples) > 1 and now-self.samples[1][0] >= self.hold:
            self.samples.popleft()
        span = now-self.samples[0][0]
        stable = np.max(np.ptp(np.stack([x[1] for x in self.samples]), axis=0)) <= self.max_joint_range
        self.reason = "holding completion evidence" if stable else "waiting for arm retreat to settle"
        return span >= self.hold and stable

    def invalidate(self, reason):
        self.samples.clear()
        self.reason = reason
