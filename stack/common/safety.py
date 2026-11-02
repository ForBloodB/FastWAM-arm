from __future__ import annotations

from dataclasses import dataclass
import math
import time


JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


class SafetyError(ValueError):
    pass


@dataclass(frozen=True)
class JointLimit:
    min_rad: float
    max_rad: float
    max_velocity_rad_s: float
    max_step_rad: float

    def validate(self) -> None:
        values = (self.min_rad, self.max_rad, self.max_velocity_rad_s, self.max_step_rad)
        if not all(math.isfinite(v) for v in values):
            raise SafetyError("joint limits must be finite")
        if self.min_rad >= self.max_rad or self.max_velocity_rad_s <= 0 or self.max_step_rad <= 0:
            raise SafetyError("invalid joint limits")


def validate_trajectory(
    *,
    sequence: int,
    previous_sequence: int,
    issued_unix_ns: int,
    deadline_unix_ns: int,
    duration_ms: int,
    positions: dict[str, float],
    velocities: dict[str, float],
    current: dict[str, float],
    limits: dict[str, JointLimit],
    allowed_joints: set[str],
    now_unix_ns: int | None = None,
) -> None:
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if sequence <= previous_sequence:
        raise SafetyError("sequence must increase")
    if issued_unix_ns > now + 100_000_000:
        raise SafetyError("command timestamp is in the future")
    if deadline_unix_ns < now or deadline_unix_ns <= issued_unix_ns:
        raise SafetyError("command is expired")
    if deadline_unix_ns - issued_unix_ns > 500_000_000:
        raise SafetyError("command lease exceeds 500 ms")
    if not 20 <= duration_ms <= 200:
        raise SafetyError("trajectory duration must be 20..200 ms")
    names = set(positions)
    if not names or names != set(velocities):
        raise SafetyError("position and velocity joint sets must match and be non-empty")
    if not names <= allowed_joints:
        raise SafetyError("command contains a joint outside the enabled test set")
    if not names <= set(limits) or not names <= set(current):
        raise SafetyError("unknown joint or missing current feedback")
    for name in names:
        p, v, q = positions[name], velocities[name], current[name]
        limit = limits[name]
        limit.validate()
        if not all(math.isfinite(x) for x in (p, v, q)):
            raise SafetyError(f"non-finite target or feedback for {name}")
        if not limit.min_rad <= p <= limit.max_rad:
            raise SafetyError(f"position limit exceeded for {name}")
        if not 0 < v <= limit.max_velocity_rad_s:
            raise SafetyError(f"velocity limit exceeded for {name}")
        if abs(p - q) > limit.max_step_rad:
            raise SafetyError(f"target step exceeded for {name}")
        if abs(p - q) / (duration_ms / 1000) > v:
            raise SafetyError(f"trajectory duration exceeds velocity limit for {name}")


def interpolate(start: float, target: float, elapsed_s: float, duration_s: float) -> float:
    if duration_s <= 0:
        raise SafetyError("duration must be positive")
    alpha = min(1.0, max(0.0, elapsed_s / duration_s))
    return start + (target - start) * alpha
