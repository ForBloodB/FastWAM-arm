import json
from pathlib import Path

import numpy as np

from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik

model = load_robot_model()
data = model.createData()
frame_id = get_end_effector_frame_id(model)
rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
params = IKParams(max_iter=1000, tolerance=1e-4)
best = None
trials = 0
for height in (0.28, 0.32, 0.36):
    position = np.array([0.35, -0.08, height])
    target = pos_rot_to_se3(position, rotation)
    for q2 in (0.6, 1.0, 1.4):
        for q3 in (0.3, 0.7, 1.1, 1.5):
            for q4 in (-1.2, 0.0, 1.2):
                for q5 in (-1.0, 0.0, 1.0):
                    seed = np.array([0.225, q2, q3, q4, q5, 0.0])
                    result = solve_ik(model, data, frame_id, target, seed, params,
                                      controlled_joints=6, position_only=False)
                    trials += 1
                    if best is None or result.error < best[0].error:
                        best = (result, position.copy(), seed.copy())
                        print(json.dumps({"trial": trials, "height": height, "error": result.error,
                                          "success": result.success, "q": result.q.tolist()}), flush=True)
                    if result.success:
                        break
                if result.success:
                    break
            if result.success:
                break
        if result.success:
            break
    if result.success:
        break
result, position, seed = best
report = {
    "solver": "Seeed Pinocchio solve_ik with structured full-pose seeds",
    "target_position_m": position.tolist(),
    "target_rotation": rotation.tolist(),
    "seed_q": seed.tolist(),
    "initial_q": np.asarray(result.q).tolist(),
    "success": bool(result.success),
    "error": float(result.error),
    "iterations": int(result.iterations),
    "structured_trials": trials,
    "position_only": False,
    "controlled_joints": 6,
}
Path(__file__).with_name("initial_pose.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not result.success:
    raise RuntimeError(f"No converged structured full-pose solution; best error={result.error}")
