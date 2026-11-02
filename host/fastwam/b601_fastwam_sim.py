from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def rotation_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w, x, y, z = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif idx == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w, x, y, z = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w, x, y, z = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    return [w, x, y, z]


def look_at_quaternion(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None) -> list[float]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0]) if up_hint is None else up_hint
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    return matrix_to_quaternion(rotation)


def build_scene(source_xml: Path, output_xml: Path) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    root.set("model", "B601 FastWAM simulation")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="224", offheight="224")
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.65 0.65 0.65", specular="0.18 0.18 0.18")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("B601 model is missing worldbody")

    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        pos="0 0 -0.08",
        size="2 2 0.1",
        rgba="0.55 0.58 0.57 1",
        friction="1.2 0.02 0.002",
    )
    ET.SubElement(world, "light", name="key_light", pos="0.3 -0.5 1.2", dir="0 0 -1", directional="true", diffuse="0.8 0.8 0.8", specular="0.2 0.2 0.2")
    ET.SubElement(
        world,
        "body",
        name="table",
        pos="0.49 0 -0.04",
    )
    table = world.find("body[@name='table']")
    ET.SubElement(
        table,
        "geom",
        name="tabletop",
        type="box",
        size="0.30 0.38 0.04",
        rgba="0.48 0.34 0.22 1",
        friction="1.2 0.02 0.002",
    )

    block = ET.SubElement(world, "body", name="red_block", pos="0.34 -0.10 0.03")
    ET.SubElement(block, "freejoint", name="red_block_free")
    ET.SubElement(
        block,
        "geom",
        name="red_block_geom",
        type="box",
        size="0.025 0.025 0.03",
        mass="0.08",
        rgba="0.85 0.08 0.05 1",
        friction="1.5 0.03 0.003",
    )
    target = ET.SubElement(world, "body", name="target_zone", pos="0.59 0.10 0.003")
    ET.SubElement(
        target,
        "geom",
        name="target_zone_geom",
        type="cylinder",
        size="0.07 0.002",
        rgba="0.08 0.40 0.85 0.42",
        contype="0",
        conaffinity="0",
    )

    eye = np.array([0.88, -0.90, 0.72])
    target_point = np.array([0.39, 0.0, 0.10])
    ET.SubElement(
        world,
        "camera",
        name="global",
        pos=" ".join(f"{v:.8f}" for v in eye),
        quat=" ".join(f"{v:.9f}" for v in look_at_quaternion(eye, target_point)),
        fovy="58",
    )

    link6 = world.find(".//body[@name='link6']")
    if link6 is None:
        raise RuntimeError("B601 model is missing link6")
    ET.SubElement(link6, "site", name="b601_eef", pos="0 0 0.290481", size="0.002", rgba="0 1 0 0")
    wrist_eye = np.array([0.0, 0.06, 0.16])
    link6_position = np.array([0.1355, 0.0, 0.217746])
    link6_rotation = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    block_position = np.array([0.34, -0.10, 0.03])
    wrist_target = link6_rotation.T @ (block_position - link6_position)
    ET.SubElement(
        link6,
        "camera",
        name="wrist",
        pos=" ".join(f"{v:.6f}" for v in wrist_eye),
        quat=" ".join(f"{v:.9f}" for v in look_at_quaternion(wrist_eye, wrist_target, np.array([1.0, 0.0, 0.0]))),
        fovy="62",
    )

    actuator = ET.SubElement(root, "actuator")
    for idx, (name, low, high, kp) in enumerate(
        [
            ("joint1", -2.8, 2.8, 150),
            ("joint2", 0.0, 3.14, 150),
            ("joint3", 0.0, 3.14, 150),
            ("joint4", -1.57, 1.57, 60),
            ("joint5", -1.57, 1.57, 60),
            ("joint6", -3.14, 3.14, 40),
            ("gripper_joint1", 0.0, 0.05, 500),
            ("gripper_joint2", 0.0, 0.0715, 500),
        ]
    ):
        ET.SubElement(
            actuator,
            "position",
            name=f"{name}_position",
            joint=name,
            kp=str(kp),
            ctrlrange=f"{low} {high}",
            ctrllimited="true",
        )

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def get_pose(mujoco, model, data, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def solve_ik(mujoco, model, data, site_id: int, target_pos: np.ndarray, target_rot: np.ndarray,
             joint_qpos: list[int], joint_dofs: list[int], joint_limits: np.ndarray) -> tuple[np.ndarray, dict]:
    saved_qpos = data.qpos.copy()
    q = np.array([data.qpos[index] for index in joint_qpos], dtype=np.float64)
    best_error = math.inf
    for iteration in range(80):
        for j, index in enumerate(joint_qpos):
            data.qpos[index] = q[j]
        mujoco.mj_forward(model, data)
        position, rotation = get_pose(mujoco, model, data, site_id)
        position_error = target_pos - position
        rotation_error = rotation_to_axis_angle(target_rot @ rotation.T)
        error = position_error
        error_norm = float(np.linalg.norm(error))
        best_error = min(best_error, error_norm)
        if np.linalg.norm(position_error) < 0.003:
            break
        jac_pos = np.zeros((3, model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        jacobian = jac_pos[:, joint_dofs]
        damping = 0.01
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(3), error)
        delta = np.clip(delta, -0.06, 0.06)
        q = np.clip(q + delta, joint_limits[:, 0] + 1e-4, joint_limits[:, 1] - 1e-4)

    for index, value in zip(joint_qpos, q):
        data.qpos[index] = value
    mujoco.mj_forward(model, data)
    position, rotation = get_pose(mujoco, model, data, site_id)
    pos_residual = float(np.linalg.norm(target_pos - position))
    rot_residual = float(np.linalg.norm(rotation_to_axis_angle(target_rot @ rotation.T)))
    q_solution = q.copy()
    data.qpos[:] = saved_qpos
    mujoco.mj_forward(model, data)
    return q_solution, {
        "iterations": iteration + 1,
        "best_position_error_m": best_error,
        "position_residual_m": pos_residual,
        "rotation_residual_rad": rot_residual,
        "orientation_control": "held_at_observed_pose",
        "converged": pos_residual < 0.005,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/aup/lerobot-fastwam"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output or root / "evaluate_results/b601_fastwam_sim").resolve()
    source_xml = root / "sim_b601/B601_from_urdf.xml"
    scene_xml = root / "sim_b601/B601_fastwam_scene.xml"
    build_scene(source_xml, scene_xml)

    sys.path.insert(0, str(root / "host"))
    sys.path.insert(0, str(root / "vendor/FastWAM-main/src"))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import torch
    from PIL import Image
    from fastwam_inference_smoke import _load_policy
    from fastwam_b601_adapter import adapt_libero_action_to_b601

    checkpoint = root / "checkpoints/fastwam_release/libero_optional_idm_2cam224.pt"
    stats = root / "checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json"
    context_path = root / "checkpoints/fastwam_release/task_context.pt"
    cfg, policy_model, processor = _load_policy(root, checkpoint, stats, args.inference_steps)
    context_blob = torch.load(context_path, map_location="cpu", weights_only=True)
    context = context_blob["context"].to("cuda")
    context_mask = context_blob["context_mask"].to("cuda")
    prompt = str(context_blob.get("prompt", "Pick up the red block and place it on the table."))

    mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=224, width=224)
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "b601_eef")
    block_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    joint_names = [f"joint{i}" for i in range(1, 7)]
    joint_qpos = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
    joint_dofs = [mj_model.jnt_dofadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
    joint_limits = np.array([mj_model.jnt_range[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names])
    gripper_joint_names = ["gripper_joint1", "gripper_joint2"]
    gripper_qpos = [mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in gripper_joint_names]
    actuator_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position") for name in joint_names + gripper_joint_names]
    actuator_ids_by_joint = dict(zip(joint_names + gripper_joint_names, actuator_ids))
    low = np.array([0.0, 0.0])
    high = np.array([0.05, 0.0715])
    state_key = processor.shape_meta["state"][0]["key"]
    action_key = processor.shape_meta["action"][0]["key"]
    action_normalizer = processor.normalizer.normalizers["action"][action_key]
    state_meta = processor.shape_meta["state"][0]
    image_metas = processor.shape_meta["images"]
    image_h, image_w = int(image_metas[0]["shape"][1]), int(image_metas[0]["shape"][2])
    concat_mode = cfg.data.train.get("concat_multi_camera", "horizontal")
    if concat_mode != "horizontal" or image_h != 224 or image_w != 224:
        raise RuntimeError(f"unexpected FastWAM camera layout: {concat_mode}, {image_h}x{image_w}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_episode_metrics = []
    for episode in range(args.episodes):
        mujoco.mj_resetData(mj_model, data)
        data.qpos[joint_qpos] = np.zeros(6)
        data.qpos[gripper_qpos] = high
        object_joint = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_free")
        object_qpos = mj_model.jnt_qposadr[object_joint]
        data.qpos[object_qpos : object_qpos + 3] = [0.34 + 0.025 * episode, -0.10 + 0.02 * episode, 0.03]
        for index, value in zip(joint_names, data.qpos[joint_qpos]):
            data.ctrl[actuator_ids_by_joint[f"{index}"]] = value
        for name, index in zip(gripper_joint_names, gripper_qpos):
            data.ctrl[actuator_ids_by_joint[name]] = data.qpos[index]
        mujoco.mj_forward(mj_model, data)
        episode_dir = output_dir / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        inference_times = []
        executed = 0
        stop_reason = "max_actions"

        for step in range(args.max_actions):
            mujoco.mj_forward(mj_model, data)
            renderer.update_scene(data, camera="global")
            external_rgb = renderer.render().copy()
            renderer.update_scene(data, camera="wrist")
            wrist_rgb = renderer.render().copy()
            site_pos, site_rot = get_pose(mujoco, mj_model, data, site_id)
            raw_state = np.concatenate((site_pos, rotation_to_axis_angle(site_rot), data.qpos[gripper_qpos])).astype(np.float32)
            state_batch = {"state": {state_key: torch.from_numpy(raw_state).unsqueeze(0)}}
            state_batch = processor.action_state_transform(state_batch)
            normalized_state = processor.normalizer.forward(state_batch)["state"][state_key]
            image = np.concatenate((external_rgb, wrist_rgb), axis=1)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=policy_model.torch_dtype)
            image_tensor = image_tensor * (2.0 / 255.0) - 1.0

            started = time.perf_counter()
            with torch.inference_mode():
                prediction = policy_model.infer_action(
                    prompt=None,
                    input_image=image_tensor,
                    action_horizon=4,
                    context=context,
                    context_mask=context_mask,
                    proprio=normalized_state.to("cuda"),
                    num_inference_steps=args.inference_steps,
                    seed=1234 + episode * 100 + step,
                    rand_device="cpu",
                    compile_action_infer=False,
                    action_infer_mode="first_frame",
                    negative_prompt="",
                    text_cfg_scale=1.0,
                    tiled=False,
                )
            inference_s = time.perf_counter() - started
            inference_times.append(inference_s)
            denormalized = action_normalizer.backward(prediction["action"].detach().float().cpu()).numpy()
            if denormalized.ndim == 3 and denormalized.shape[0] == 1:
                denormalized = denormalized[0]
            elif denormalized.ndim == 1:
                denormalized = denormalized[None, :]
            if denormalized.ndim != 2 or denormalized.shape[1] != 7 or not np.isfinite(denormalized).all():
                print("Rejected FastWAM action:", {
                    "shape": tuple(denormalized.shape),
                    "finite": bool(np.isfinite(denormalized).all()),
                    "sample": np.nan_to_num(denormalized.reshape(-1)[:14]).tolist(),
                })
                stop_reason = "invalid_model_action"
                break
            denormalized[:, -1] = -(denormalized[:, -1] * 2.0 - 1.0)
            model_action = denormalized[0]
            adapted = adapt_libero_action_to_b601(raw_state, model_action)
            target_position = np.asarray(adapted["position_m"], dtype=np.float64)
            target_rotation = site_rot
            solution, ik = solve_ik(mujoco, mj_model, data, site_id, target_position, target_rotation,
                                    joint_qpos, joint_dofs, joint_limits)
            if not np.isfinite(solution).all():
                stop_reason = "ik_nonfinite"
                break
            current_joints = data.qpos[joint_qpos].copy()
            delta = solution - current_joints
            commanded_joints = current_joints + np.clip(delta, -math.radians(1.0), math.radians(1.0))
            for name, value in zip(joint_names, commanded_joints):
                data.ctrl[actuator_ids_by_joint[name]] = value
            grip_target = low if adapted["gripper_intent"] == "close" else high
            for name, value in zip(gripper_joint_names, grip_target):
                data.ctrl[actuator_ids_by_joint[name]] = value
            block_start = data.xpos[block_body].copy()
            for _ in range(25):
                mujoco.mj_step(mj_model, data)
            mujoco.mj_forward(mj_model, data)
            actual_pos, _ = get_pose(mujoco, mj_model, data, site_id)
            block_end = data.xpos[block_body].copy()
            frame_prefix = f"step_{step:04d}"
            Image.fromarray(external_rgb).save(episode_dir / f"{frame_prefix}_external.png")
            Image.fromarray(wrist_rgb).save(episode_dir / f"{frame_prefix}_wrist.png")
            rows.append({
                "step": step,
                "instruction": prompt,
                "observation": {
                    "eef_position_m": site_pos.tolist(),
                    "eef_axis_angle_rad": rotation_to_axis_angle(site_rot).tolist(),
                    "gripper_qpos_m": data.qpos[gripper_qpos].tolist(),
                    "joint_positions_rad": current_joints.tolist(),
                    "block_position_m": block_start.tolist(),
                    "external_image": f"{frame_prefix}_external.png",
                    "wrist_image": f"{frame_prefix}_wrist.png",
                },
                "fastwam": {
                    "checkpoint": checkpoint.name,
                    "device": torch.cuda.get_device_name(0),
                    "hip": torch.version.hip,
                    "inference_seconds": inference_s,
                    "predicted_chunk_dataset_scale": denormalized.tolist(),
                    "executed_action": model_action.tolist(),
                },
                "adaptation": adapted,
                "rotation_action_applied": False,
                "rotation_action_note": "FastWAM/Panda wrist rotation is logged but held at the B601 observed orientation until the cross-embodiment tool-frame mapping is calibrated.",
                "ik": ik,
                "commanded_joint_positions_rad": commanded_joints.tolist(),
                "result": {
                    "eef_position_m": actual_pos.tolist(),
                    "block_position_m": block_end.tolist(),
                    "joint_positions_rad": data.qpos[joint_qpos].tolist(),
                    "simulated_step_count": 25,
                },
                "actuator_dispatch": "MuJoCo simulation only",
            })
            executed += 1
            if not ik["converged"] and ik["position_residual_m"] > 0.06:
                stop_reason = "ik_unreachable_or_residual_high"
                break

        with (episode_dir / "transitions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        final_block = data.xpos[block_body].copy()
        final_eef = data.site_xpos[site_id].copy()
        metrics = {
            "episode": episode,
            "executed_actions": executed,
            "stop_reason": stop_reason,
            "mean_inference_seconds": float(np.mean(inference_times)) if inference_times else None,
            "max_inference_seconds": float(np.max(inference_times)) if inference_times else None,
            "final_eef_position_m": final_eef.tolist(),
            "final_block_position_m": final_block.tolist(),
            "block_displacement_m": float(np.linalg.norm(final_block - rows[0]["observation"]["block_position_m"])) if rows else None,
            "task_success": None,
            "task_success_note": "No B601-specific success was asserted; this is a policy rollout, not a validated successful pick-and-place.",
        }
        (episode_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        all_episode_metrics.append(metrics)

    summary = {
        "system": "395 host + MuJoCo B601-RS simulation",
        "checkpoint": str(checkpoint),
        "checkpoint_embodiment": "FastWAM LIBERO checkpoint (Panda training data); not B601-finetuned",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "instruction": prompt,
        "scene": str(scene_xml),
        "cameras": ["global", "wrist"],
        "joint_control": "bounded 1 degree per policy step; translation-only damped-least-squares IK; gripper intent mapped to B601 open/close; MuJoCo position actuators",
        "physical_hardware_connected": False,
        "episodes": all_episode_metrics,
        "dataset_format": "per-episode PNG camera frames + transitions.jsonl + metrics.json",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


if __name__ == "__main__":
    main()
