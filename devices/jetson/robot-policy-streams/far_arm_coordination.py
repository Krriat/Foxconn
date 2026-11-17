"""Hold far-stage right arm at VERIFIED home until left ACT reaches lift.

Only delays right-arm ACT targets; never supplies a replayed manipulation.
Fresh ACT targets resume through the existing chunk fusion and motion limiter.
"""
import math


class FarArmCoordination:
    def __init__(self):
        self.run_id=None;self.hold_target=None;self.released=False;self.enabled=False

    def apply(self, actions, *, run_id, verified_anchor, enabled, left_lift_seen):
        self.enabled=bool(enabled)
        if not enabled:
            self.run_id=run_id;self.hold_target=None;self.released=False
            return actions
        if self.run_id!=run_id or self.hold_target is None:
            if len(verified_anchor)!=14 or any(not math.isfinite(q) for q in verified_anchor):
                raise ValueError('Right-arm coordination requires verified finite measured home')
            self.run_id=run_id;self.hold_target=list(verified_anchor[7:14]);self.released=False
        self.released |= bool(left_lift_seen)
        if self.released:return actions
        return [list(row[:7])+list(self.hold_target)+list(row[14:]) for row in actions]

    def status(self):
        return dict(enabled=self.enabled,right_held=self.enabled and not self.released,
                    left_lift_seen=self.released,hold_target=self.hold_target,
                    release_condition='measured left-arm lift during suction ON',
                    is_vacuum_detection=False)
