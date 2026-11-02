from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence


class FastWAMBackend(Protocol):
    """Adapter around the pinned upstream checkpoint and inference processor."""

    def reset(self, task: str) -> None: ...
    def predict(self, observation: dict[str, Any]) -> Sequence[Sequence[float]]: ...


class B601ActionAdapter(Protocol):
    """Converts one model end-effector action to bounded B601 joint positions."""

    def to_joint_positions(
        self, observation: dict[str, Any], model_action: Sequence[float]
    ) -> dict[str, float]: ...


@dataclass
class FastWAMPolicy:
    backend: FastWAMBackend
    action_adapter: B601ActionAdapter
    action_dim: int = 7
    max_joint_delta_deg: float = 1.0

    def reset(self, task: str) -> None:
        self.backend.reset(task)

    def predict(self, observation: dict[str, Any]) -> dict[str, float]:
        chunk = self.backend.predict(observation)
        if not chunk:
            raise RuntimeError("FastWAM returned an empty action chunk")
        action = chunk[0]
        if len(action) != self.action_dim or not all(math.isfinite(float(v)) for v in action):
            raise RuntimeError("FastWAM action must be a finite 7D end-effector/gripper vector")

        # FastWAM's released actions are not B601 joint targets; a verified IK,
        # frame, unit, and normalization adapter must be supplied explicitly.
        joints = self.action_adapter.to_joint_positions(observation, action)
        if not joints:
            raise RuntimeError("B601 action adapter returned no joint targets")
        for name, value in joints.items():
            key = f"{name}.pos"
            if key not in observation or isinstance(value, bool) or not math.isfinite(float(value)):
                raise RuntimeError(f"invalid adapted joint action: {name}")
            if abs(float(value) - float(observation[key])) > self.max_joint_delta_deg:
                raise RuntimeError(f"B601 joint action delta exceeds limit for {name}")
        return {f"{name}.pos": float(value) for name, value in joints.items()}
