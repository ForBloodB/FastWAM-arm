from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import numpy as np


def matrix_to_quaternion(matrix):
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale]
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        return [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale]
    if index == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        return [(matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale]
    scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
    return [(matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]


def look_at(eye, target, up_hint):
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up_hint, dtype=float))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return matrix_to_quaternion(np.column_stack((right, up, -forward)))


def fmt(values):
    return " ".join(f"{float(value):.10g}" for value in values)


def build(arm_path, output_path, task):
    root = ET.Element("mujoco", model=f"B601_RS_tall_PLA_{task}")
    ET.SubElement(root, "compiler", angle="radian", meshdir="../")
    ET.SubElement(root, "option", timestep="0.001", iterations="100", solver="Newton", integrator="Euler")
    contact = ET.SubElement(root, "contact")
    for first, second in [("world", "link1"), ("base_link", "link1"), ("link1", "link2"),
                          ("link2", "link3"), ("link2", "link5"), ("link3", "link4"),
                          ("link3", "link5"), ("link4", "link5"), ("link5", "link6"),
                          ("gripper_left", "gripper_right")]:
        ET.SubElement(contact, "exclude", body1=first, body2=second)
    ET.SubElement(root, "statistic", center="0.36 0 0.25", extent="0.72")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="224", offheight="224", azimuth="125", elevation="-28")
    ET.SubElement(visual, "headlight", diffuse="0.68 0.68 0.68", ambient="0.42 0.42 0.42", specular="0.1 0.1 0.1")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.30 0.48 0.65", rgb2="0.04 0.06 0.08", width="512", height="3072")
    for name, rgba in [("table", "0.34 0.37 0.36 1"), ("pla_blue", "0.035 0.32 0.72 1"),
                       ("pla_orange", "0.95 0.27 0.045 1"), ("pla_gray", "0.56 0.62 0.68 1"),
                       ("pla_green", "0.06 0.48 0.30 1")]:
        ET.SubElement(asset, "material", name=name, rgba=rgba)
    meshes = {
        "tall_block_mesh": "01_grip_block_36x30x70mm.stl",
        "tall_peg_mesh": "02_peg_d16_h80mm.stl",
        "tall_socket_mesh": "03_socket_50x50x50mm_hole16p4.stl",
        "tall_tray_mesh": "04_sorting_tray_80x65x40mm.stl",
    }
    for name, filename in meshes.items():
        ET.SubElement(asset, "mesh", name=name, file=f"tall_parts/meshes/{filename}", scale="0.001 0.001 0.001")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.25 -0.15 1.3", dir="0.05 0.1 -1", directional="true")
    global_eye = np.array([0.82, -0.68, 0.72])
    global_target = np.array([0.38, 0.0, 0.16])
    ET.SubElement(world, "camera", name="global", mode="fixed", pos=fmt(global_eye),
                  quat=fmt(look_at(global_eye, global_target, [0, 0, 1])), fovy="55")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 0.04", rgba="0.12 0.16 0.20 1", friction="1.2 0.01 0.001")
    table = ET.SubElement(world, "body", name="task_table", pos="0.36 0 0.05")
    ET.SubElement(table, "geom", name="task_table_geom", type="box", size="0.23 0.23 0.05", material="table",
                  friction="1.8 0.02 0.002", solref="0.003 1", solimp="0.98 0.995 0.0005")

    if task == "grasp":
        obj = ET.SubElement(world, "body", name="tall_block", pos="0.35 -0.08 0.10")
        ET.SubElement(obj, "freejoint", name="tall_block_freejoint")
        ET.SubElement(obj, "geom", name="tall_block_geom", type="mesh", mesh="tall_block_mesh", material="pla_blue",
                      mass="0.060", friction="2.0 0.03 0.003", solref="0.002 1", solimp="0.99 0.995 0.0005")
        target = ET.SubElement(world, "body", name="tall_tray", pos="0.40 0.04 0.10")
        ET.SubElement(target, "geom", name="tall_tray_geom", type="mesh", mesh="tall_tray_mesh", material="pla_green",
                      friction="1.5 0.02 0.002")
    else:
        obj = ET.SubElement(world, "body", name="tall_peg", pos="0.35 -0.08 0.10")
        ET.SubElement(obj, "freejoint", name="tall_peg_freejoint")
        ET.SubElement(obj, "geom", name="tall_peg_geom", type="mesh", mesh="tall_peg_mesh", material="pla_orange",
                      mass="0.025", friction="1.5 0.02 0.002", solref="0.002 1", solimp="0.99 0.995 0.0005")
        target = ET.SubElement(world, "body", name="tall_socket", pos="0.40 0.04 0.10")
        ET.SubElement(target, "geom", name="tall_socket_geom", type="mesh", mesh="tall_socket_mesh", material="pla_gray",
                      friction="1.2 0.02 0.002", solref="0.002 1", solimp="0.995 0.999 0.0002")

    arm = ET.parse(arm_path).getroot()
    gripper = arm.find(".//body[@name='gripper_end']")
    if gripper is None:
        raise RuntimeError("Official model gripper_end missing")
    ET.SubElement(gripper, "site", name="b601_eef", pos="0 0 0", size="0.003", rgba="0 1 0 0")
    eye = np.array([-0.130, -0.110, 0.100])
    # At the official seed pose, aim the eye-in-hand camera at the actual task object.
    # This is a fixed camera extrinsic transform after construction, not a per-frame tracker.
    initial_tcp = np.array([0.39633521599019916, -0.14541435008686618, 0.3925795185865059])
    initial_rotation = np.array([
        [0.641708348123984, 0.6607268498718665, -0.3894231962870058],
        [-0.7173550943022867, 0.6967077354779413, 0.0000019381502287813292],
        [0.2713154338155969, 0.2793534699687811, 0.9210589417566375],
    ])
    object_focus = np.array([0.35, -0.08, 0.145 if task == "grasp" else 0.155])
    wrist_target = initial_rotation.T @ (object_focus - initial_tcp)
    wrist_up = initial_rotation.T @ np.array([0.0, 0.0, 1.0])
    ET.SubElement(gripper, "camera", name="wrist", pos=fmt(eye),
                  quat=fmt(look_at(eye, wrist_target, wrist_up)), fovy="75")
    original_world = arm.find("worldbody")
    for child in list(original_world):
        world.append(child)
    root.append(arm.find("asset"))
    for tag in ("equality", "actuator"):
        section = arm.find(tag)
        if section is not None:
            root.append(section)
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=["grasp", "insert"], required=True)
    args = parser.parse_args()
    build(args.arm.resolve(), args.out.resolve(), args.task)
    print(args.out.resolve())
