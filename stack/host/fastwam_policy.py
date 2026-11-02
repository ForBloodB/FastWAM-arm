from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol


class FastWAMBackend(Protocol):
    """Implemented by the pinned upstream FastWAM repository adapter."""

    def reset(self, task: str) -> None: ...
    def predict(self, observation: dict[str, Any]) -> dict[str, float]: ...


@dataclass
class FastWAMPolicy:
    backend: FastWAMBackend
    max_action_delta_deg: float = 1.0

    def reset(self, task: str) -> None:
        self.backend.reset(task)

    def predict(self, observation: dict[str, Any]) -> dict[str, float]:
        action = self.backend.predict(observation)
        if not action:
            raise RuntimeError("FastWAM returned an empty action")
        for name, value in action.items():
            if (
                name not in {f"{joint}.pos" for joint in (
                    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                    "wrist_yaw", "wrist_roll", "gripper",
                )}
                or isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise RuntimeError(f"FastWAM action must use LeRobot joint positions: {name}")
            current_key = name
            if current_key not in observation:
                raise RuntimeError(f"observation is missing {current_key}")
            if abs(float(value) - float(observation[current_key])) > self.max_action_delta_deg:
                raise RuntimeError(f"FastWAM action delta exceeds limit for {name}")
        return action
