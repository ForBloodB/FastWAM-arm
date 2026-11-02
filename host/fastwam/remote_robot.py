from __future__ import annotations

import time
import grpc
import math
import threading
from dataclasses import dataclass

import robot_service_pb2 as pb
import robot_service_pb2_grpc as rpc


JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_yaw", "wrist_roll", "gripper",
)


from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot


@RobotConfig.register_subclass("remote_b601")
@dataclass(kw_only=True)
class RemoteB601RobotConfig(RobotConfig):
    target: str = "172.21.16.91:50051"
    timeout_s: float = 0.25


class RemoteB601Robot(Robot):
    """LeRobot Robot adapter that reaches hardware only through KR260 gRPC."""

    config_class = RemoteB601RobotConfig
    name = "remote_b601"

    def __init__(self, config: RemoteB601RobotConfig):
        super().__init__(config)
        self.config = config
        self.target = config.target
        self.timeout_s = config.timeout_s
        self.channel = grpc.insecure_channel(
            self.target,
            options=(
                ("grpc.keepalive_time_ms", 100),
                ("grpc.keepalive_timeout_ms", 100),
                ("grpc.max_reconnect_backoff_ms", 250),
            ),
        )
        self.stub = rpc.RobotServiceStub(self.channel)
        self.sequence = 0
        self.calibrated = False
        self._connected = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

    @property
    def is_connected(self):
        return self._connected and self.channel is not None and self.stub is not None

    @property
    def is_calibrated(self):
        return self.calibrated

    @property
    def observation_features(self):
        return {
            **{f"{n}.pos": float for n in JOINTS},
            **{f"{n}.vel": float for n in JOINTS},
            **{f"{n}.torque": float for n in JOINTS},
            **{f"{n}.temperature": float for n in JOINTS},
            **{f"leader.{n}.pos": float for n in JOINTS},
        }

    @property
    def action_features(self):
        return {f"{n}.pos": float for n in JOINTS}

    def connect(self, calibrate=True):
        if self.is_connected:
            return
        state = self.stub.GetState(pb.GetStateRequest(), timeout=self.timeout_s)
        names = [joint.name for joint in state.joints]
        if len(names) != len(JOINTS) or set(names) != set(JOINTS):
            raise RuntimeError("KR260 returned an incomplete or unexpected joint set")
        self._check_state_freshness(state.sampled_unix_ns)
        for joint in state.joints:
            if not joint.feedback_fresh or joint.fault_bits:
                raise RuntimeError(f"invalid feedback for {joint.name}")
        self.calibrated = state.calibrated
        if not self.calibrated:
            if calibrate:
                raise RuntimeError("remote B601 connect will not initialize hardware; calibration is missing")
            raise RuntimeError("KR260 reports missing or incomplete calibration")
        self._connected = True
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def calibrate(self):
        return None

    def configure(self):
        self._require_connected()
        self.heartbeat()

    def get_observation(self) -> dict[str, float | int | bool]:
        self._require_connected()
        state = self.stub.GetState(pb.GetStateRequest(), timeout=self.timeout_s)
        self._check_state_freshness(state.sampled_unix_ns)
        names = [joint.name for joint in state.joints]
        if len(names) != len(JOINTS) or set(names) != set(JOINTS):
            raise RuntimeError("KR260 returned an incomplete or unexpected joint set")
        result = {}
        for joint in state.joints:
            if not joint.feedback_fresh or joint.fault_bits:
                raise RuntimeError(f"invalid feedback for {joint.name}")
            # LeRobot's B601 driver uses degrees for public joint positions.
            result[f"{joint.name}.pos"] = math.degrees(joint.position_rad)
            result[f"{joint.name}.vel"] = math.degrees(joint.velocity_rad_s)
            result[f"{joint.name}.torque"] = joint.torque_nm
            result[f"{joint.name}.temperature"] = joint.temperature_c
            result[f"leader.{joint.name}.pos"] = math.degrees(joint.leader_position_rad)
        return result

    @staticmethod
    def _check_state_freshness(sampled_unix_ns: int) -> None:
        if abs(time.time_ns() - sampled_unix_ns) > 200_000_000:
            raise RuntimeError("stale KR260 state or unsynchronized host clocks")

    def send_action(self, action: dict[str, float], max_velocity_rad_s=0.10, duration_ms=200):
        self._require_connected()
        if not action or not math.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0:
            raise ValueError("action and positive finite velocity are required")
        expected = {f"{name}.pos" for name in JOINTS}
        if set(action) != expected:
            raise ValueError(f"action keys must exactly match the B601 joints: {sorted(expected)}")
        if not 20 <= duration_ms <= 200:
            raise ValueError("trajectory duration must be 20..200 ms")
        if not all(math.isfinite(float(value)) for value in action.values()):
            raise ValueError("action positions must be finite")
        self.sequence += 1
        self.heartbeat()
        now = time.time_ns()
        joints = []
        for name in JOINTS:
            key = f"{name}.pos"
            if key not in action:
                continue
            joints.append(pb.JointValue(
                name=name,
                position_rad=math.radians(float(action[key])),
                max_velocity_rad_s=max_velocity_rad_s,
            ))
        command = pb.JointTrajectory(
            sequence=self.sequence,
            issued_unix_ns=now,
            deadline_unix_ns=now + min(500_000_000, (duration_ms + 100) * 1_000_000),
            duration_ms=duration_ms,
            joints=joints,
        )
        result = self.stub.ExecuteJointTrajectory(command, timeout=self.timeout_s)
        if not result.accepted:
            raise RuntimeError(result.reason)
        return action

    def heartbeat(self):
        self._require_connected()
        now = time.time_ns()
        return self.stub.Heartbeat(
            pb.HeartbeatRequest(sequence=self.sequence, issued_unix_ns=now), timeout=self.timeout_s
        )

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(0.05):
            try:
                self.heartbeat()
            except (grpc.RpcError, RuntimeError):
                self._connected = False
                self._heartbeat_stop.set()
                break

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("remote B601 robot is not connected")

    def stop(self, reason="host stop"):
        return self.stub.Cancel(pb.CancelRequest(reason=reason), timeout=self.timeout_s)

    def close(self):
        self._heartbeat_stop.set()
        self._connected = False
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=0.2)
            self._heartbeat_thread = None
        if self.channel is not None:
            self.channel.close()
            self.channel = None

    def disconnect(self):
        try:
            if self.channel is not None and self.is_connected:
                try:
                    self.stop("LeRobot disconnect")
                except grpc.RpcError:
                    pass
        finally:
            self.close()
