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
for x, y in ((0.42, 0.02), (0.43, 0.03), (0.40, 0.04), (0.38, 0.04), (0.44, 0.0), (0.40, 0.0), (0.38, 0.02)):
    candidate = {"xy": [x, y], "solutions": []}
    for z in (0.192, 0.205, 0.23, 0.245):
        result = solve_ik(model, data, frame_id, pos_rot_to_se3(np.array([x, y, z]), rotation), seed, params,
                          controlled_joints=6, position_only=False)
        row = {"z": z, "success": bool(result.success), "error": float(result.error),
               "q": np.asarray(result.q[:6]).tolist()}
        candidate["solutions"].append(row)
    candidate["all_reachable"] = all(row["success"] for row in candidate["solutions"])
    rows.append(candidate)
    print(json.dumps(candidate), flush=True)
Path(__file__).with_name("destination_probe.json").write_text(json.dumps(rows, indent=2) + "\n")
