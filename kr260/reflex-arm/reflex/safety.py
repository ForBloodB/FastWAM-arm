"""Independent host-side limits; these cannot replace a verified hardware interlock."""
from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class Envelope:
    residual_scale: tuple = (.015, .015, .010, .15, 3.0)  # m, m, m, rad, N
    low: tuple = (.12, -.25, .025)
    high: tuple = (.48, .25, .40)
    force_range: tuple = (0., 12.)

    def project(self, base, action):
        import numpy as np
        base, a = np.asarray(base, float), np.asarray(action, float)
        if base.shape != (5,) or a.shape != (5,) or not np.isfinite(np.r_[base, a]).all():
            raise ValueError('Expected finite [x,y,z,yaw,force] and five residuals')
        if np.any(base[:3]<self.low) or np.any(base[:3]>self.high) or not self.force_range[0]<=base[4]<=self.force_range[1]:
            raise ValueError('Base plan outside envelope; refusing an unbounded correction')
        bounded = np.clip(a, -1, 1)
        target = base + bounded * self.residual_scale
        target[:3] = np.clip(target[:3], self.low, self.high)
        target[4] = np.clip(target[4], *self.force_range)
        return target


class State(IntEnum):
    DISARMED = 0
    ARMED = 1
    FAULT = 2


class SafetyFSM:
    """Monotonic-clock reference model. Faults latch until a safe manual reset."""
    def __init__(self, timeout_ns=100_000_000):
        self.timeout_ns = timeout_ns
        self.state = State.DISARMED
        self.last_rx_ns = None
        self.reason = None

    def tick(self, now_ns, *, heartbeat=False, arm=False, reset=False,
             estop=False, limit=False, sensor_ok=True, command_ok=True):
        bad = 'estop' if estop else 'limit' if limit else 'sensor' if not sensor_ok else 'command' if not command_ok else None
        if heartbeat:
            self.last_rx_ns = now_ns
        fresh = self.last_rx_ns is not None and 0 <= now_ns-self.last_rx_ns < self.timeout_ns
        if bad or (self.state == State.ARMED and not fresh):
            self.state, self.reason = State.FAULT, bad or 'timeout'
        elif self.state == State.FAULT:
            if reset and fresh and not arm:
                self.state, self.reason = State.DISARMED, None
        elif self.state == State.DISARMED and arm and fresh:
            self.state = State.ARMED
        return self.state

    @property
    def drive_enabled(self):
        return self.state == State.ARMED
