from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
import math
import os
import threading
import time
import socket
import struct

import grpc
from motorbridge import Controller, Mode
from motorbridge_smart_servo import FashionStarServo

import robot_service_pb2 as pb
import robot_service_pb2_grpc as rpc
from safety import JOINTS, JointLimit, SafetyError, interpolate, validate_trajectory


LIMITS = {
    "shoulder_pan": JointLimit(-2.53, 2.53, 0.15, 0.052),
    "shoulder_lift": JointLimit(-2.97, 0.02, 0.15, 0.052),
    "elbow_flex": JointLimit(-3.49, 0.02, 0.15, 0.052),
    "wrist_flex": JointLimit(-1.57, 1.40, 0.15, 0.052),
    "wrist_yaw": JointLimit(-1.57, 1.57, 0.15, 0.052),
    "wrist_roll": JointLimit(-1.57, 1.57, 0.15, 0.052),
    "gripper": JointLimit(-4.71, 0.02, 0.10, 0.052),
}

# Known telemetry anomaly remains an unconditional movement inhibit.
BLOCKED_JOINTS = {"wrist_roll"}
TAU_LIMIT_NM = 0.298
KP = {name: 0.5 for name in JOINTS}
KD = {name: 0.05 for name in JOINTS}
IDLE_TIMEOUT_S = 0.25
FEEDBACK_TIMEOUT_S = 0.15


class Hardware:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.can = Controller(channel="can0")
        self.motors = {
            name: self.can.add_robstride_motor(index + 1, 0xFD, "rs-06" if index < 3 else "rs-00")
            for index, name in enumerate(JOINTS)
        }
        self.leader = FashionStarServo("/dev/ttyUSB0", baudrate=1_000_000)
        self.leader.set_loss_threshold(1)
        self.last_feedback: dict[str, tuple[float, object]] = {}
        self.last_position: dict[str, tuple[float, float]] = {}
        self.can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.can_socket.bind(("can0",))
        self.can_socket.setblocking(False)

    def read(self) -> dict[str, dict[str, float | int | bool]]:
        with self.lock:
            return self._read()

    def _read(self) -> dict[str, dict[str, float | int | bool]]:
        now = time.monotonic()
        while True:
            try:
                frame = self.can_socket.recv(16)
            except BlockingIOError:
                break
            cid, length, data = struct.unpack("=IB3x8s", frame)
            cid &= 0x1FFFFFFF
            kind, motor_id = (cid >> 24) & 31, (cid >> 8) & 255
            if kind not in (2, 24) or not 1 <= motor_id <= 7 or length != 8:
                continue
            name = JOINTS[motor_id - 1]
            state = self.motors[name].get_state()
            if state is not None:
                self.last_feedback[name] = (now, state)
        result = {}
        for name in JOINTS:
            entry = self.last_feedback.get(name)
            if entry is None or now - entry[0] > FEEDBACK_TIMEOUT_S:
                raise SafetyError(f"stale follower feedback: {name}")
            _, state = entry
            fault_bits = int(state.status_code)
            if fault_bits:
                raise SafetyError(f"motor fault {name}: 0x{fault_bits:02x}")
            if not all(math.isfinite(v) for v in (state.pos, state.vel, state.torq, state.t_mos)):
                raise SafetyError(f"invalid motor telemetry: {name}")
            if state.t_mos >= 50:
                raise SafetyError(f"motor temperature limit: {name}")
            # RobStride B601 public coordinates reverse raw motor coordinates.
            result[name] = {
                "position_rad": -float(state.pos),
                "velocity_rad_s": -float(state.vel),
                "torque_nm": float(state.torq),
                "temperature_c": float(state.t_mos),
                "fault_bits": fault_bits,
                "feedback_fresh": True,
            }
        leaders = self.leader.sync_monitor(list(range(7)))
        for index, name in enumerate(JOINTS):
            sample = leaders.get(index)
            if sample is None or not sample.reliable:
                # Zero filtering is a known SDK condition. Read raw angle to
                # distinguish it from a missing response; threshold=1 makes
                # missing replies raise rather than return cached data.
                raw = self.leader.read_raw_angle(index, multi_turn=False)
            else:
                raw = sample.angle_deg
            result[name]["leader_position_rad"] = math.radians(raw * (-6 if name == "gripper" else -1))
        return result

    def move(self, name: str, target: float, velocity: float) -> None:
        with self.lock:
            state = self.motors[name].get_state()
            if state is None:
                raise SafetyError(f"missing state for {name}")
            q = -float(state.pos)
            dq = -float(state.vel)
            now = time.monotonic()
            previous = self.last_position.get(name)
            if previous is not None:
                dt = now - previous[0]
                if dt > 0:
                    estimated_velocity = (q - previous[1]) / dt
                    if abs(estimated_velocity) > velocity * 1.5:
                        raise SafetyError(f"position-derived velocity limit: {name}")
            self.last_position[name] = (now, q)
            if abs(dq) > velocity * 1.5:
                raise SafetyError(f"reported velocity limit: {name}")
            if abs(target - q) > LIMITS[name].max_step_rad:
                raise SafetyError(f"following error limit: {name}")
            # MotorBridge coordinates are the inverse of the public B601 frame.
            raw_target = -target
            raw_velocity = -velocity
            tau = KP[name] * (raw_target - float(state.pos)) + KD[name] * (
                raw_velocity - float(state.vel)
            )
            tau = max(-TAU_LIMIT_NM, min(TAU_LIMIT_NM, tau))
            self.motors[name].send_mit(0.0, 0.0, 0.0, 0.0, tau)

    def disable(self) -> None:
        with self.lock:
            try:
                self.can.disable_all()
            finally:
                for motor in self.motors.values():
                    try:
                        motor.robstride_set_active_report(False)
                    except Exception:
                        pass

    def close(self) -> None:
        self.disable()
        self.leader.close()
        self.can_socket.close()
        self.can.close()


class Service(rpc.RobotServiceServicer):
    def __init__(self, hardware: Hardware) -> None:
        self.hw = hardware
        self.lock = threading.RLock()
        self.last_seq = 0
        self.last_heartbeat = time.monotonic()
        self.last_command = time.monotonic()
        self.armed = False
        self.stop_reason = "startup: actuation disarmed"
        self.allowed = set(filter(None, os.getenv("KRIA_ALLOWED_JOINTS", "").split(",")))
        self.allowed -= BLOCKED_JOINTS
        self.allowed &= set(JOINTS)
        self.active_joints: set[str] = set()
        self.targets: dict[str, float] = {}
        self.start_positions: dict[str, float] = {}
        self.target_velocities: dict[str, float] = {}
        self.target_started = time.monotonic()
        self.target_duration_s = 0.08
        self.runner = threading.Thread(target=self._run, daemon=True)
        self.runner.start()

    @staticmethod
    def _authorize(context) -> None:
        peer = context.peer()
        if not peer.startswith("ipv4:172.21.16.96:"):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "only 395 may call the KR260 executor")

    def _stop(self, reason: str) -> None:
        self.armed = False
        self.stop_reason = reason
        self.active_joints.clear()
        self.hw.disable()

    def GetState(self, request, context):
        self._authorize(context)
        try:
            with self.lock:
                measured = self.hw.read()
                return pb.RobotState(
                    sampled_unix_ns=time.time_ns(), last_command_sequence=self.last_seq,
                    armed=self.armed, estop=False, stop_reason=self.stop_reason,
                    joints=[pb.JointState(name=n, **measured[n]) for n in JOINTS],
                )
        except Exception as exc:
            self._stop(f"state read failed: {exc}")
            context.abort(grpc.StatusCode.UNAVAILABLE, self.stop_reason)

    def Heartbeat(self, request, context):
        self._authorize(context)
        if abs(time.time_ns() - request.issued_unix_ns) > 500_000_000:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "heartbeat timestamp invalid")
        with self.lock:
            self.last_heartbeat = time.monotonic()
        return pb.HeartbeatReply(alive=True, received_unix_ns=time.time_ns())

    def Cancel(self, request, context):
        self._authorize(context)
        with self.lock:
            self._stop(request.reason or "cancel requested")
        return pb.CommandResult(accepted=True, sequence=self.last_seq, reason=self.stop_reason)

    def ExecuteJointTrajectory(self, request, context):
        self._authorize(context)
        with self.lock:
            try:
                if not self.allowed:
                    raise SafetyError("no joints enabled in KRIA_ALLOWED_JOINTS")
                if time.monotonic() - self.last_heartbeat > IDLE_TIMEOUT_S:
                    raise SafetyError("fresh host heartbeat required before trajectory")
                measured = self.hw.read()
                positions = {v.name: v.position_rad for v in request.joints}
                velocities = {v.name: v.max_velocity_rad_s for v in request.joints}
                if len(positions) != len(request.joints):
                    raise SafetyError("duplicate joint names")
                current = {n: float(measured[n]["position_rad"]) for n in positions}
                validate_trajectory(
                    sequence=request.sequence, previous_sequence=self.last_seq,
                    issued_unix_ns=request.issued_unix_ns, deadline_unix_ns=request.deadline_unix_ns,
                    duration_ms=request.duration_ms, positions=positions, velocities=velocities,
                    current=current, limits=LIMITS, allowed_joints=self.allowed,
                )
                if not self.armed:
                    for name in positions:
                        self.hw.motors[name].ensure_mode(Mode.MIT)
                        self.hw.motors[name].robstride_write_param_u32(0x7028, 4000)
                        if self.hw.motors[name].robstride_get_param_u32(0x7028) != 4000:
                            raise SafetyError(f"CAN timeout setting mismatch: {name}")
                        self.hw.motors[name].robstride_set_active_report(True)
                    for name in positions:
                        self.hw.motors[name].enable()
                    self.armed = True
                    self.stop_reason = ""
                else:
                    for name in positions.keys() - self.active_joints:
                        self.hw.motors[name].ensure_mode(Mode.MIT)
                        self.hw.motors[name].robstride_write_param_u32(0x7028, 4000)
                        if self.hw.motors[name].robstride_get_param_u32(0x7028) != 4000:
                            raise SafetyError(f"CAN timeout setting mismatch: {name}")
                        self.hw.motors[name].robstride_set_active_report(True)
                        self.hw.motors[name].enable()
                for name, target in positions.items():
                    self.start_positions[name] = current[name]
                    self.targets[name] = target
                    self.target_velocities[name] = velocities[name]
                self.active_joints.update(positions)
                self.target_started = time.monotonic()
                self.target_duration_s = request.duration_ms / 1000
                self.last_seq = request.sequence
                self.last_command = time.monotonic()
                return pb.CommandResult(accepted=True, sequence=self.last_seq, reason="accepted")
            except Exception as exc:
                self._stop(f"trajectory rejected: {exc}")
                return pb.CommandResult(accepted=False, sequence=self.last_seq, reason=self.stop_reason)

    def _run(self) -> None:
        while True:
            if not self.armed:
                time.sleep(0.02)
                continue
            try:
                now = time.monotonic()
                if now - self.last_command > IDLE_TIMEOUT_S or now - self.last_heartbeat > IDLE_TIMEOUT_S:
                    raise SafetyError("command or heartbeat timeout")
                with self.lock:
                    measured = self.hw.read()
                    elapsed = now - self.target_started
                    active = tuple(self.active_joints)
                    for name in active:
                        q = float(measured[name]["position_rad"])
                        desired = interpolate(
                            self.start_positions[name], self.targets[name], elapsed,
                            self.target_duration_s,
                        )
                        if abs(self.targets[name] - q) > LIMITS[name].max_step_rad:
                            raise SafetyError(f"following error: {name}")
                        self.hw.move(name, desired, self.target_velocities[name])
                time.sleep(0.02)
            except Exception as exc:
                with self.lock:
                    self._stop(f"execution stopped: {exc}")
                time.sleep(0.02)


def serve() -> None:
    hw = Hardware()
    service = Service(hw)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    rpc.add_RobotServiceServicer_to_server(service, server)
    bind = os.getenv("KRIA_GRPC_BIND", "172.21.16.91:50051")
    if server.add_insecure_port(bind) == 0:
        raise RuntimeError(f"cannot bind {bind}")
    hw.can.disable_all()
    for motor in hw.motors.values():
        motor.robstride_set_active_report(True)
    server.start()
    try:
        server.wait_for_termination()
    finally:
        server.stop(0)
        hw.close()


if __name__ == "__main__":
    serve()
