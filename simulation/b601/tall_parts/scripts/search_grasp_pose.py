import json
from pathlib import Path

import numpy as np
from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik

model = load_robot_model()
data = model.createData()
frame_id = get_end_effector_frame_id(model)
params = IKParams(max_iter=500, tolerance=1e-4)
seeds = [
    np.array([0.0, 1.3, 0.7, 1.0, -0.8, 0.0]),
    np.array([0.0, 0.8, 1.2, -1.2, 0.8, 0.0]),
    np.array([0.0, 1.8, 0.4, 1.2, -0.8, 0.0]),
    np.array([0.0, 0.6, 0.6, -1.0, -0.8, 0.0]),
    np.array([0.0, 1.2, 1.2, 0.0, 0.0, 1.57]),
    np.array([0.0, 1.2, 1.2, 0.0, 0.0, -1.57]),
]
positions = [
    np.array([0.35, -0.08, 0.155]),
    np.array([0.38, -0.05, 0.155]),
    np.array([0.40, 0.00, 0.155]),
    np.array([0.35, -0.08, 0.200]),
]
best = None
trials = 0
for position in positions:
    for yaw_deg in (0, 90, 180, 270):
        yaw = np.radians(yaw_deg)
        closing = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        for tilt_deg in (0, 20, -20, 40, -40):
            tilt = np.radians(tilt_deg)
            approach = -np.cos(tilt) * np.array([0.0, 0.0, 1.0]) + np.sin(tilt) * horizontal
            third = np.cross(approach, closing)
            rotation = np.column_stack((approach, closing, third))
            target = pos_rot_to_se3(position, rotation)
            for seed in seeds:
                result = solve_ik(model, data, frame_id, target, seed, params,
                                  controlled_joints=6, position_only=False)
                trials += 1
                candidate = {"position_m": position.tolist(), "yaw_deg": yaw_deg,
                             "tilt_deg": tilt_deg, "error": float(result.error),
                             "success": bool(result.success), "q": np.asarray(result.q[:6]).tolist(),
                             "rotation": rotation.tolist(), "seed_q": seed.tolist()}
                if best is None or result.error < best["error"]:
                    best = candidate
                    print(json.dumps({"best": best, "trials": trials}), flush=True)
                if result.success:
                    report = {"solver": "Seeed Pinocchio structured reachable-pose search",
                              "success": True, "position_only": False, "trials": trials,
                              "target_position_m": position.tolist(), "target_rotation": rotation.tolist(),
                              "initial_q": np.asarray(result.q[:6]).tolist(), "error": float(result.error),
                              "iterations": int(result.iterations), "yaw_deg": yaw_deg, "tilt_deg": tilt_deg}
                    Path(__file__).with_name("reachable_grasp_pose.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(report, indent=2), flush=True)
                    raise SystemExit(0)
report = {"success": False, "trials": trials, "best": best}
Path(__file__).with_name("reachable_grasp_pose.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(1)
