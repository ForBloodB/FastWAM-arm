from __future__ import annotations

import math

import numpy as np


def _axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def adapt_libero_action_to_b601(
    state: np.ndarray | list[float],
    action: np.ndarray | list[float],
    *,
    max_translation_step_m: float = 0.04,
    max_rotation_step_rad: float = 0.20,
) -> dict[str, object]:
    """Integrate a LIBERO 6D EEF delta into a base_link pose with bounded steps.

    The state layout and gripper inversion match FastWAM's LIBERO evaluator.
    Translation and rotation deltas are applied in the world/base frame.
    """
    state = np.asarray(state, dtype=float).reshape(-1)
    action = np.asarray(action, dtype=float).reshape(-1)
    if state.shape != (8,) or action.shape != (7,):
        raise ValueError("expected LIBERO state (8,) and action (7,)")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state and action must be finite")
    if max_translation_step_m <= 0 or max_rotation_step_rad <= 0:
        raise ValueError("step limits must be positive")

    translation = action[:3].copy()
    translation_norm = float(np.linalg.norm(translation))
    translation_scale = min(1.0, max_translation_step_m / max(translation_norm, 1e-12))
    translation *= translation_scale

    rotation = action[3:6].copy()
    rotation_norm = float(np.linalg.norm(rotation))
    rotation_scale = min(1.0, max_rotation_step_rad / max(rotation_norm, 1e-12))
    rotation *= rotation_scale

    current_rotation = _axis_angle_to_matrix(state[3:6])
    delta_rotation = _axis_angle_to_matrix(rotation)
    # LIBERO's OSC pose deltas are expressed in the base/world frame.
    target_rotation = delta_rotation @ current_rotation
    target_position = state[:3] + translation

    # FastWAM's evaluator maps dataset [0, 1] open values to robosuite [-1, 1]
    # and flips sign: positive means close, negative means open.
    gripper_command = -(2.0 * float(action[6]) - 1.0)
    return {
        "position_m": target_position.tolist(),
        "axis_angle_rad": _matrix_to_axis_angle(target_rotation).tolist(),
        "gripper_command": gripper_command,
        "gripper_intent": "close" if gripper_command >= 0.0 else "open",
        "raw_translation_delta_m": action[:3].tolist(),
        "applied_translation_delta_m": translation.tolist(),
        "translation_scale": translation_scale,
        "raw_rotation_delta_rad": action[3:6].tolist(),
        "applied_rotation_delta_rad": rotation.tolist(),
        "rotation_scale": rotation_scale,
    }
