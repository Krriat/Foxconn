"""Arm-motion interlock for LoRA completion, never a lid/suction detector.

The milestone regions match the reviewed successful demonstration sweeps.
Visual completion remains necessary; a lifted upright lid alone cannot abort
the right arm while it is still performing the manipulation.
"""
import math


class ManipulationProgress:
    def __init__(self):
        self.reset(None)

    def reset(self, stage):
        self.stage = stage
        self.contact_seen = self.sweep_seen = self.ready = False
        self.clear_since = None

    def update(self, joints, now):
        if self.stage not in ("far", "near"):
            return
        if len(joints) != 14 or any(not math.isfinite(x) for x in joints):
            raise ValueError("Progress requires fourteen finite measured joints")
        q = [math.degrees(x) for x in joints]
        near = self.stage == "near"
        self.contact_seen |= q[3] >= 114. if near else q[1] >= 30.
        if self.contact_seen:
            self.sweep_seen |= q[10] >= 125. if near else q[8] >= 35.
        clear = (q[1] < 25. and q[8] < 25. and q[10] < 115.) if near else (
            q[1] < 30. and q[8] < 30. and q[10] > 80.)
        if self.sweep_seen and clear:
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= .5:
                self.ready = True
        else:
            self.clear_since = None

    def status(self):
        phase = ("retract" if self.ready else "manipulation" if self.sweep_seen
                 else "contact" if self.contact_seen else "approach")
        return {"stage": self.stage, "phase": phase, "contact_seen": self.contact_seen,
                "sweep_seen": self.sweep_seen, "completion_ready": self.ready,
                "is_lid_or_vacuum_detection": False}
