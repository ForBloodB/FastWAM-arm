from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


class RemoteRobot(Protocol):
    def connect(self, calibrate: bool = True) -> bool: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...
    def disconnect(self) -> None: ...


class CameraRig(Protocol):
    def connect(self) -> None: ...
    def read(self) -> dict[str, dict[str, Any]]: ...
    def disconnect(self) -> None: ...


class Policy(Protocol):
    def predict(self, observation: dict[str, Any]) -> dict[str, float]: ...


ObservationBuilder = Callable[
    [Mapping[str, Any], Mapping[str, Mapping[str, Any]]], dict[str, Any]
]


@dataclass(frozen=True)
class BrainStep:
    observation: dict[str, Any]
    action: dict[str, float]
    dispatched: bool


class BrainRuntime:
    """395-side LeRobot brain; all robot IO is delegated to the KR260 adapter."""

    def __init__(
        self,
        robot: RemoteRobot,
        cameras: CameraRig,
        policy: Policy,
        observation_builder: ObservationBuilder,
        *,
        allow_action_dispatch: bool = False,
    ) -> None:
        self.robot = robot
        self.cameras = cameras
        self.policy = policy
        self.observation_builder = observation_builder
        self.allow_action_dispatch = allow_action_dispatch
        self.connected = False

    def connect(self) -> None:
        self.robot.connect(calibrate=False)
        try:
            self.cameras.connect()
        except Exception:
            self.robot.disconnect()
            raise
        self.connected = True

    def step(self) -> BrainStep:
        if not self.connected:
            raise RuntimeError("brain runtime is not connected")
        robot_state = self.robot.get_observation()
        camera_frames = self.cameras.read()
        observation = self.observation_builder(robot_state, camera_frames)
        action = self.policy.predict(observation)
        dispatched = False
        if self.allow_action_dispatch:
            self.robot.send_action(action)
            dispatched = True
        return BrainStep(observation, action, dispatched)

    def disconnect(self) -> None:
        try:
            self.cameras.disconnect()
        finally:
            self.robot.disconnect()
            self.connected = False
