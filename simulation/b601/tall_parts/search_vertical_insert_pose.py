import argparse
import json
from pathlib import Path

import numpy as np
from reBotArm_control_py.kinematics import get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, pos_rot_to_se3, solve_ik


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
    best = None
    trials = 0
    for height in (.169, .175, .180, .185, .190, .195, .200):
        position = np.array([.35, -.08, height])
        for yaw_deg in (0, 90, 180, 270):
            yaw = np.radians(yaw_deg)
            closing = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            for tilt_deg in (-20, -40, 40):
                tilt = np.radians(tilt_deg)
                approach = -np.cos(tilt) * np.array([0.0, 0.0, 1.0]) + np.sin(tilt) * horizontal
                third = np.cross(approach, closing)
                rotation = np.column_stack((approach, closing, third))
                target = pos_rot_to_se3(position, rotation)
                for seed in seeds:
                    result = solve_ik(model, data, frame_id, target, seed, params,
                                      controlled_joints=6, position_only=False)
                    trials += 1
                    candidate = {"height_m": height, "yaw_deg": yaw_deg, "tilt_deg": tilt_deg,
                                 "error": float(result.error), "success": bool(result.success),
                                 "q": np.asarray(result.q[:6]).tolist()}
                    if best is None or result.error < best["error"]:
                        best = candidate
                    if result.success:
                        report = {
                            "solver": "Seeed Pinocchio alternate-approach insert-grasp search",
                            "success": True, "position_only": False, "trials": trials,
                            "target_position_m": position.tolist(),
                            "target_rotation": rotation.tolist(),
                            "initial_q": np.asarray(result.q[:6]).tolist(),
                            "error": float(result.error), "iterations": int(result.iterations),
                            "yaw_deg": yaw_deg, "tilt_deg": tilt_deg,
                        }
                        args.out.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(report, indent=2), flush=True)
                        return
    report = {"success": False, "trials": trials, "best": best}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
