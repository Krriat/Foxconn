"""Replay/mock protection state machine, NOT a validated hardware safety controller.

The SDK motion backend is intentionally absent. A stop request and confirmed stop
are separate states. After any stop, send permission is permanently revoked.
"""
import math

class Protection:
    def __init__(self, *, feedback_timeout_s=.05, command_deadline_s=.01,
                 stopped_velocity_rad_s=.001, stop_timeout_s=1.):
        self.feedback_ns=int(feedback_timeout_s*1e9)
        self.deadline_ns=int(command_deadline_s*1e9)
        self.stop_timeout_ns=int(stop_timeout_s*1e9)
        self.stop_velocity=stopped_velocity_rad_s
        self.state='active'; self.reason=None; self.last_seq=None
        self.last_device=None; self.last_fresh=None; self.last_host=None
        self.stop_requested_ns=None; self.events=[]

    def stop(self, reason, now):
        if self.state != 'active': return
        self.state='stop_requested'; self.reason=reason; self.stop_requested_ns=now
        self.events.append(dict(type='controlled_stop_requested',reason=reason,host_ns=now))

    def feedback(self, seq, device_ns, host_ns, q, v, mode='normal', fault=False):
        duplicate=False
        if self.state!='active': return duplicate
        if self.last_host is not None and host_ns < self.last_host:
            self.stop('host_clock_rollback',host_ns); return duplicate
        self.last_host=host_ns
        if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in list(q)+list(v)):
            self.stop('nonfinite_feedback',host_ns); return duplicate
        if mode!='normal' or fault:
            self.stop('mode_or_fault',host_ns); return duplicate
        if seq is not None and self.last_seq is not None:
            if seq<self.last_seq: self.stop('sequence_rollback',host_ns)
            duplicate=seq==self.last_seq
        if device_ns is not None and self.last_device is not None:
            if device_ns<self.last_device: self.stop('device_clock_rollback',host_ns)
            duplicate=duplicate or device_ns==self.last_device
        if self.state!='active': return duplicate
        if not duplicate:
            self.last_seq=seq; self.last_device=device_ns; self.last_fresh=host_ns
        self.tick(host_ns)
        return duplicate

    def tick(self, now):
        if self.state=='active' and self.last_fresh is not None and now-self.last_fresh>self.feedback_ns:
            self.stop('stale_feedback',now)
        if self.state=='stop_requested' and now-self.stop_requested_ns>self.stop_timeout_ns:
            self.state='stop_unconfirmed'
            self.events.append(dict(type='stop_unconfirmed',reason='stop_timeout; watchdog/physical state not established',host_ns=now))

    def can_send(self, due_ns, start_ns, done_ns=None, log_ok=True):
        if self.state!='active': return False
        if self.last_fresh is None:
            self.stop('no_feedback',start_ns)
        elif start_ns-due_ns>self.deadline_ns or (done_ns is not None and done_ns-start_ns>self.deadline_ns):
            self.stop('command_deadline',start_ns)
        elif not log_ok:
            self.stop('log_queue_overflow',start_ns)
        else: self.tick(start_ns)
        return self.state=='active'

    def confirm_stop(self, now, operation_idle, velocities):
        if self.state!='stop_requested': return False
        self.tick(now)
        if self.state!='stop_requested': return False
        if operation_idle and len(velocities)==7 and all(math.isfinite(x) and abs(x)<=self.stop_velocity for x in velocities):
            self.state='stopped'
            self.events.append(dict(type='stop_confirmed',host_ns=now))
            return True
        return False
