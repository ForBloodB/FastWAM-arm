import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import get_end_effector_frame, get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.robot_model import _hw_config, _resolve_urdf

report = json.loads(Path(__file__).with_name("initial_pose.json").read_text())
model = load_robot_model()
data = model.createData()
q = np.zeros(model.nq)
q[:6] = report["initial_q"]
frame_id = get_end_effector_frame_id(model)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
pose = data.oMf[frame_id]
target_position = np.asarray(report["target_position_m"])
target_rotation = np.asarray(report["target_rotation"])
target = pin.SE3(target_rotation, target_position)
print(json.dumps({
    "hardware_config": _hw_config(),
    "urdf": _resolve_urdf()[0],
    "end_effector_frame": get_end_effector_frame(),
    "nq": model.nq,
    "joint_names": list(model.names),
    "lower_limits": model.lowerPositionLimit.tolist(),
    "upper_limits": model.upperPositionLimit.tolist(),
    "q": q.tolist(),
    "actual_position": pose.translation.tolist(),
    "target_position": target_position.tolist(),
    "position_error_m": float(np.linalg.norm(target_position - pose.translation)),
    "actual_rotation": pose.rotation.tolist(),
    "target_rotation": target_rotation.tolist(),
    "local_6d_error": pin.log6(pose.inverse() * target).vector.tolist(),
    "local_6d_error_norm": float(np.linalg.norm(pin.log6(pose.inverse() * target).vector)),
}, indent=2, default=str))
