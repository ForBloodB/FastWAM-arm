from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/aup/lerobot-fastwam"))
    parser.add_argument("--task", choices=["grasp", "insert"], required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    scene = root / f"sim_b601/tall_parts/{args.task}.xml"
    pose = json.loads(args.pose.read_text())
    if not pose.get("success"):
        raise RuntimeError("Reachable pose search did not succeed")
    rotation = np.asarray(pose["target_rotation"], dtype=float)
    seed_q = np.asarray(pose["initial_q"], dtype=float)

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=224, width=224)
    names = [f"joint{i}" for i in range(1, 7)]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qpos_addr = [model.jnt_qposadr[i] for i in ids]
    dof_addr = [model.jnt_dofadr[i] for i in ids]
    actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor") for name in names]
    grip_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")]
    grip_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")]
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint7_motor")
    left_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_left")]
    right_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_right")]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "b601_eef")
    object_name = "tall_block" if args.task == "grasp" else "tall_peg"
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    kp = np.array([80., 100., 100., 35., 25., 18.])
    kd = np.array([8., 10., 10., 4., 3., 2.5])
    tau_limit = np.array([36., 36., 36., 14., 14., 14.])
    target_q = seed_q.copy()
    target_grip = 0.05
    rows = []
    frame_index = 0
    global_video = cv2.VideoWriter(str(out / "global.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (224, 224))
    wrist_video = cv2.VideoWriter(str(out / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (224, 224))

    def physics(count):
        for _ in range(count):
            q = data.qpos[qpos_addr]
            qd = data.qvel[dof_addr]
            tau = data.qfrc_bias[dof_addr] + kp * (target_q - q) - kd * qd
            data.ctrl[actuators] = np.clip(tau, -tau_limit, tau_limit)
            data.ctrl[grip_act] = np.clip(300. * (target_grip - data.qpos[grip_q]) - 120. * data.qvel[grip_dof], -150., 150.)
            mujoco.mj_step(model, data)

    def render(camera):
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    def contacts():
        found = []
        for index in range(data.ncon):
            contact = data.contact[index]
            first = model.geom(contact.geom1).name
            second = model.geom(contact.geom2).name
            pair = {first, second}
            object_contact = any(name == object_name or name.startswith(object_name + "_") for name in pair)
            if object_contact and any("gripper" in name for name in pair):
                found.append([first, second])
        return found

    def record(phase, target_position=None, ik=None):
        nonlocal frame_index
        external = render("global")
        wrist = render("wrist")
        Image.fromarray(external).save(out / f"frame_{frame_index:03d}_global.png")
        Image.fromarray(wrist).save(out / f"frame_{frame_index:03d}_wrist.png")
        global_video.write(cv2.cvtColor(external, cv2.COLOR_RGB2BGR))
        wrist_video.write(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        row = {"frame": frame_index, "phase": phase,
               "target_tcp_m": None if target_position is None else np.asarray(target_position).tolist(),
               "actual_tcp_m": data.site_xpos[site].tolist(),
               "object_base_m": data.xpos[object_body].tolist(),
               "object_quaternion_wxyz": data.xquat[object_body].tolist(),
               "actual_q_rad": data.qpos[qpos_addr].tolist(),
               "target_q_rad": target_q.tolist(),
               "target_gripper": target_grip,
               "actual_gripper_m": data.qpos[[left_q, right_q]].tolist(),
               "gripper_object_contacts": contacts(), "ik": ik}
        rows.append(row)
        (out / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps({"frame": frame_index, "phase": phase, "tcp": row["actual_tcp_m"],
                          "object": row["object_base_m"], "contacts": len(row["gripper_object_contacts"])}), flush=True)
        frame_index += 1

    def official_ik(current_q, target_position):
        request = {"current_q": np.asarray(current_q).tolist(),
                   "target_position_m": np.asarray(target_position).tolist(),
                   "target_rotation": rotation.tolist()}
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/aup/Workspace/reBotArm_control_py"
        process = subprocess.run(
            ["/home/aup/Workspace/reBotArm_control_py/.venv/bin/python",
             str(root / "sim_b601/official_fastwam/pinocchio_ik_step.py")],
            input=json.dumps(request), capture_output=True, text=True, env=env, timeout=30)
        if process.returncode != 0:
            raise RuntimeError(f"Official IK process failed: {process.stderr}")
        result = json.loads(process.stdout)
        if (not result["converged"] or result["position_residual_m"] > .002
                or result["rotation_residual_rad"] > .025):
            raise RuntimeError(f"Official IK rejected: {result}")
        return np.asarray(result["q"], dtype=float), result

    def move_to(target_position, phase):
        nonlocal target_q
        start = data.site_xpos[site].copy()
        target_position = np.asarray(target_position, dtype=float)
        count = max(1, int(math.ceil(np.linalg.norm(target_position - start) / .005)))
        for position in np.linspace(start, target_position, count + 1)[1:]:
            solution, ik = official_ik(data.qpos[qpos_addr], position)
            current_q = data.qpos[qpos_addr].copy()
            delta = solution - current_q
            max_delta = float(np.max(np.abs(delta)))
            if max_delta > math.radians(8.):
                raise RuntimeError(f"IK branch jump rejected: {np.degrees(delta).tolist()} deg")
            # Preserve the coordinated six-axis solution. Independent per-joint clipping
            # can reverse Cartesian motion near a limit; bound duration instead.
            target_q = solution
            control_steps = max(240, int(math.ceil(max_delta / math.radians(4.) / .001)))
            physics(control_steps)
            record(phase, position, ik)

    def correct_peg_axis(phase, max_steps=12):
        nonlocal rotation
        for _ in range(max_steps):
            object_axis = data.xmat[object_body].reshape(3, 3)[:, 2]
            correction_axis = np.cross(object_axis, np.array([0., 0., 1.]))
            sine = float(np.linalg.norm(correction_axis))
            angle = math.atan2(sine, float(object_axis[2]))
            if angle < math.radians(.3):
                break
            correction_axis /= max(sine, 1e-12)
            correction, _ = cv2.Rodrigues(
                correction_axis * min(.4 * angle, math.radians(1.5)))
            rotation = correction @ data.site_xmat[site].reshape(3, 3)
            move_to(data.site_xpos[site].copy(), phase)
        physics(1200)

    def servo_insert():
        target_xy = np.array([.40, .04])
        target_base_z = .132
        # Correct the measured peg-base error in a few bounded moves. Repeated
        # sub-millimetre dithering excites the round head inside the jaws.
        for _ in range(6):
            if not contacts():
                raise RuntimeError("Physical grasp contact lost during alignment; stopping")
            object_position = data.xpos[object_body].copy()
            if object_position[2] < .17:
                raise RuntimeError("Peg bottom is too low for safe lateral alignment")
            error = target_xy - object_position[:2]
            if np.linalg.norm(error) < .0001:
                break
            correction = error.copy()
            correction *= min(1., .003 / max(np.linalg.norm(correction), 1e-12))
            tcp = data.site_xpos[site].copy()
            desired = tcp.copy()
            desired[:2] += correction
            move_to(desired, "align_peg_above_hole")
            physics(120)
        correct_peg_axis("upright_peg_above_hole")
        if data.xmat[object_body].reshape(3, 3)[2, 2] < .9998:
            raise RuntimeError("Peg axis is not vertical enough before descent; keeping gripper closed")
        # Servo from the measured peg-base height. A fixed TCP height cannot prove insertion
        # because the grasp offset changes when the peg is rotated upright.
        for _ in range(90):
            object_position = data.xpos[object_body].copy()
            height_error = object_position[2] - target_base_z
            if height_error <= .0005:
                break
            error = target_xy - object_position[:2]
            correction = .2 * error
            correction *= min(1., .0005 / max(np.linalg.norm(correction), 1e-12))
            tcp = data.site_xpos[site].copy()
            desired = tcp.copy()
            desired[:2] += correction
            desired[2] = tcp[2] - min(.001, max(.0005, height_error))
            move_to(desired, "servo_insert_by_peg_base")
            if not contacts():
                raise RuntimeError("Physical grasp contact lost during insertion; stopping")
            if np.linalg.norm(target_xy - data.xpos[object_body][:2]) > .003:
                raise RuntimeError("Peg lateral error exceeded 3 mm; keeping gripper closed")
            if data.xmat[object_body].reshape(3, 3)[2, 2] < .998:
                raise RuntimeError("Peg tilt exceeded the insertion safety gate; keeping gripper closed")
        else:
            raise RuntimeError("Peg did not reach insertion depth within 90 servo steps; keeping gripper closed")

    try:
        grasp_z = .155 if args.task == "grasp" else float(pose["target_position_m"][2])
        pregrasp = np.array([.35, -.08, .245 if args.task == "grasp" else .282])
        initial_q, initial_ik = official_ik(seed_q, pregrasp)
        data.qpos[qpos_addr] = initial_q
        data.qpos[[grip_q, left_q, right_q]] = .05
        data.qvel[:] = 0.
        target_q = initial_q.copy()
        mujoco.mj_forward(model, data)
        physics(500)
        initial_object = data.xpos[object_body].copy()
        max_object_z = float(initial_object[2])
        record("pregrasp", pregrasp, initial_ik)

        move_to(np.array([.35, -.08, grasp_z]), "descend_to_grip")
        # The 22 mm fitted head needs a position hold near its measured diameter.
        # Commanding zero here over-closes the round head and ejects it during settling.
        target_grip = 0.0 if args.task == "grasp" else .003
        physics(1000)
        record("close_gripper")
        contact_after_close = len(contacts())
        move_to(pregrasp, "lift_to_safe_height")
        physics(500)
        record("lift_hold")
        max_object_z = max(max_object_z, float(data.xpos[object_body][2]))
        lifted_for_transfer = bool(data.xpos[object_body][2] >= initial_object[2] + .06)
        if not lifted_for_transfer:
            raise RuntimeError("Object was not lifted at least 60 mm; transfer stopped")

        if args.task == "insert":
            correct_peg_axis("upright_peg_axis_feedback")
            record("upright_peg_hold")

        # Keep the peg bottom above the 50 mm socket during lateral alignment.
        if args.task == "insert":
            destination = data.site_xpos[site].copy()
            destination[:2] += np.array([.40, .04]) - data.xpos[object_body][:2]
            destination[2] = .317
        else:
            destination = np.array([.389, .04, .23])
        move_to(destination, "transfer_above_target")
        if args.task == "insert":
            physics(120)
            record("transfer_settled")
            servo_insert()
        else:
            move_to(np.array([.380, .04, .192]), "place_or_insert")
        physics(500)
        record("insertion_hold" if args.task == "insert" else "place_hold")
        max_object_z = max(max_object_z, float(data.xpos[object_body][2]))
        if args.task == "insert":
            held_object = data.xpos[object_body].copy()
            held_rotation = data.xmat[object_body].reshape(3, 3)
            held_xy_error = float(np.linalg.norm(held_object[:2] - np.array([.40, .04])))
            if (held_xy_error > .0005 or not .128 < held_object[2] < .145
                    or held_rotation[2, 2] <= .995):
                raise RuntimeError(
                    f"Insertion gate rejected before release: xy={held_xy_error:.6f} m, "
                    f"base_z={held_object[2]:.6f} m, axis_z={held_rotation[2, 2]:.6f}; "
                    "keeping gripper closed")
        target_grip = .05
        physics(1000)
        record("release")
        move_to(destination, "retreat")
        physics(500)
        record("final")

        final_object = data.xpos[object_body].copy()
        object_rotation = data.xmat[object_body].reshape(3, 3)
        upright = bool(object_rotation[2, 2] > .995)
        lifted = bool(max_object_z >= initial_object[2] + .06)
        if args.task == "grasp":
            target_ok = bool(abs(final_object[0] - .40) < .008 and abs(final_object[1] - .04) < .008
                             and .125 < final_object[2] < .155)
            task_success = bool(lifted and target_ok and upright)
        else:
            target_ok = bool(np.linalg.norm(final_object[:2] - np.array([.40, .04])) < .0005
                             and .128 < final_object[2] < .145)
            task_success = bool(lifted and target_ok and upright)
        summary = {
            "system": "Official B601-RS MuJoCo dynamics; high block/socket/tray use local STL, new peg uses RGB-D fitted flange-shaft-head primitives",
            "geometry_fidelity": "High block uses local tall STL; new peg is an engineering surrogate, not exact user CAD",
            "task": args.task, "controller": "Scripted Cartesian waypoints + Seeed official Pinocchio full 6D IK + gravity-compensated PD",
            "policy_used": False, "physical_hardware_connected": False,
            "initial_object_base_m": initial_object.tolist(), "final_object_base_m": final_object.tolist(),
            "max_object_base_z_m": max_object_z, "lifted_60mm": lifted,
            "upright": upright, "target_geometry_satisfied": target_ok,
            "contact_after_close": contact_after_close, "task_success": task_success,
            "grasp_tcp_height_above_table_mm": 155 if args.task == "grasp" else round(grasp_z * 1000),
            "approach_tilt_deg": 20 if args.task == "grasp" else float(pose.get("tilt_deg", 40)), "strict_vertical_approach_used": False,
            "success_criteria": "physical contacts only; object lifted >=60 mm; final base and upright orientation satisfy tray/insertion geometry; no attachment or teleport",
            "frames": len(rows), "trajectory": "trajectory.jsonl", "videos": ["global.mp4", "wrist.mp4"],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        failure = {"task": args.task, "task_success": False, "error": repr(exc),
                   "last_object_base_m": data.xpos[object_body].tolist(), "frames": len(rows)}
        (out / "failure.json").write_text(json.dumps(failure, indent=2))
        print(json.dumps(failure, indent=2), flush=True)
    finally:
        global_video.release()
        wrist_video.release()
        renderer.close()


if __name__ == "__main__":
    main()
