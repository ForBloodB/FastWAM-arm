import json
from pathlib import Path

import numpy as np
from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik

pose = json.loads(Path(__file__).with_name("reachable_grasp_pose.json").read_text())
rotation = np.asarray(pose["target_rotation"])
seed = np.asarray(pose["initial_q"])
model = load_robot_model()
data = model.createData()
frame_id = get_end_effector_frame_id(model)
params = IKParams(max_iter=1000, tolerance=1e-4)
rows = []
for x, y in ((0.35, -0.08), (0.50, 0.09)):
    for z in np.arange(0.155, 0.301, 0.005):
        position = np.array([x, y, float(z)])
        result = solve_ik(model, data, frame_id, pos_rot_to_se3(position, rotation), seed, params,
                          controlled_joints=6, position_only=False)
        row = {"xy": [x, y], "z": float(z), "success": bool(result.success),
               "error": float(result.error), "q": np.asarray(result.q[:6]).tolist()}
        rows.append(row)
        print(json.dumps(row), flush=True)
Path(__file__).with_name("pose_height_probe.json").write_text(json.dumps(rows, indent=2) + "\n")
