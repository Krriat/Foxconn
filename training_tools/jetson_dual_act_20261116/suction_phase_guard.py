"""Bound premature ACT vacuum release; not a vacuum-pressure detector.

Automatic FAR stage only. An OFF prediction after
ON without the demonstrated left-arm lifting motion gets a bounded grace
interval, then a controlled fault instead of continuing an empty manipulation.
The pose region occurs before suction release in all 29 original far demos
(one episode initially has suction OFF). It is not proof of actual attachment.
Explicit stop/release bypasses this filter.
"""
import math


class FarReleaseGuard:
    def __init__(self, grace_seconds=1.5):
        self.grace_seconds=grace_seconds
        self.reset()

    def reset(self):
        self.on_seen=False
        self.released=False
        self.left_lift_seen=False
        self.waiting_since=None

    def update_measured(self, joints):
        if len(joints)!=14 or any(not math.isfinite(x) for x in joints):
            raise ValueError('Release interlock requires fourteen finite measured joints')
        if self.on_seen:
            self.left_lift_seen |= math.degrees(joints[1])>=47. and math.degrees(joints[0])<=28.

    def filter(self, target, *, now, enabled):
        result=list(target)
        if not enabled:
            self.reset()
            return result
        if target[0]:
            self.on_seen=True;self.released=False;self.waiting_since=None
        elif self.on_seen and not self.released:
            if self.left_lift_seen:
                self.released=True;self.waiting_since=None
            else:
                if self.waiting_since is None:self.waiting_since=now
                if now-self.waiting_since>=self.grace_seconds:
                    raise RuntimeError('ACT requested suction release before left-arm lifting motion; '
                                       'grasp unconfirmed, cancelling instead of empty flipping')
                result[0]=True
        return result

    def status(self):
        return dict(on_seen=self.on_seen,released=self.released,left_lift_seen=self.left_lift_seen,
                    holding_pending_lift=self.waiting_since is not None,
                    grace_seconds=self.grace_seconds,is_vacuum_detection=False)
