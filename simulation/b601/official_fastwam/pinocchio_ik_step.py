import json
import sys

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model, pad_q_for_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik, solve_ik_with_retry

request = json.load(sys.stdin)
current_q = np.asarray(request["current_q"], dtype=float)
target_position = np.asarray(request["target_position_m"], dtype=float)
target_rotation = np.asarray(request["target_rotation"], dtype=float)
if current_q.shape != (6,) or target_position.shape != (3,) or target_rotation.shape != (3, 3):
    raise ValueError("Expected six joints, a 3D position and a 3x3 rotation")
model = load_robot_model()
data = model.createData()
frame_id = get_end_effector_frame_id(model)
target = pos_rot_to_se3(target_position, target_rotation)
params = IKParams(max_iter=1000, tolerance=1e-4)
result = solve_ik(model, data, frame_id, target, current_q, params,
                  controlled_joints=6, position_only=False)
if not result.success:
    seed = pad_q_for_model(model, current_q, 6)
    retried = solve_ik_with_retry(model, data, frame_id, target, seed, params, max_retries=4)
    if retried.error < result.error:
        result = retried
q = np.zeros(model.nq)
q[:min(6, len(result.q))] = result.q[:6]
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
pose = data.oMf[frame_id]
position_error = float(np.linalg.norm(pose.translation - target_position))
rotation_error = float(np.linalg.norm(pin.log3(pose.rotation.T @ target_rotation)))
print(json.dumps({
    "q": np.asarray(result.q[:6]).tolist(),
    "converged": bool(result.success),
    "error": float(result.error),
    "iterations": int(result.iterations),
    "position_residual_m": position_error,
    "rotation_residual_rad": rotation_error,
    "position_only": False,
    "solver": "Seeed Pinocchio full 6D IK",
}))
