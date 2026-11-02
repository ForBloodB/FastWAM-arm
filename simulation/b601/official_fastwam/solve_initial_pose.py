from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reBotArm_control_py.kinematics import compute_ik, load_robot_model, pad_q_for_model


TARGET_POSITION = np.array([0.35, -0.08, 0.28], dtype=float)
TARGET_ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
SEED = np.array([0.0, 1.3, 0.7, 1.0, -0.8, 0.0])


def main() -> None:
    from reBotArm_control_py.kinematics import get_end_effector_frame_id, pos_rot_to_se3, solve_ik_with_retry

    model = load_robot_model()
    data = model.createData()
    frame_id = get_end_effector_frame_id(model)
    candidates = []
    for height in (0.28, 0.32, 0.36, 0.40, 0.44):
        target_position = np.array([TARGET_POSITION[0], TARGET_POSITION[1], height])
        result = solve_ik_with_retry(
            model,
            data,
            frame_id,
            pos_rot_to_se3(target_position, TARGET_ROTATION),
            pad_q_for_model(model, SEED, 6),
            max_retries=8,
        )
        candidates.append({"target_position_m": target_position.tolist(), "success": bool(result.success),
                           "error": float(result.error), "iterations": int(result.iterations)})
        print(json.dumps(candidates[-1]), flush=True)
        if result.success:
            break
    full_q = np.asarray(result.q, dtype=float)
    if full_q.shape[0] < 6 or not np.isfinite(full_q).all():
        raise RuntimeError(f"Unexpected IK result: {full_q}")
    # The official URDF appends two passive gripper joints to the six arm joints.
    initial_q = full_q[:6]
    report = {
        "solver": "Seeed reBotArm_control_py Pinocchio full 6D solve_ik_with_retry",
        "target_position_m": target_position.tolist(),
        "reachability_trials": candidates,
        "target_rotation": TARGET_ROTATION.tolist(),
        "seed_q": SEED.tolist(),
        "initial_q": initial_q.tolist(),
        "success": bool(result.success),
        "error": float(result.error),
        "iterations": int(result.iterations),
        "position_only": False,
        "controlled_joints": 6,
    }
    output = Path(__file__).with_name("initial_pose.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not result.success:
        raise RuntimeError(f"Strict full-pose initial IK failed; error={result.error:.12g}")


if __name__ == "__main__":
    main()
