import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model

model = load_robot_model()
data = model.createData()
initial_q = np.array([0.0, 1.3, 0.7, 1.0, -0.8, 0.0])
if np.any(initial_q < model.lowerPositionLimit[:6]) or np.any(initial_q > model.upperPositionLimit[:6]):
    raise RuntimeError("Initial configuration violates official joint limits")
q = np.zeros(model.nq)
q[:6] = initial_q
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
pose = data.oMf[get_end_effector_frame_id(model)]
report = {
    "solver": "Official Seeed Pinocchio FK of previously stable six-joint configuration",
    "method": "fixed_configuration_fk_not_pregrasp_ik",
    "target_position_m": pose.translation.tolist(),
    "target_rotation": pose.rotation.tolist(),
    "seed_q": initial_q.tolist(),
    "initial_q": initial_q.tolist(),
    "success": True,
    "error": 0.0,
    "iterations": 0,
    "position_only": False,
    "controlled_joints": 6,
    "note": "Unreachable strict vertical pregrasp attempts are recorded separately in initial_pose.json",
}
Path(__file__).with_name("initial_pose_seed.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
