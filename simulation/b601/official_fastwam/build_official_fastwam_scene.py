from pathlib import Path
import argparse
import numpy as np
import xml.etree.ElementTree as ET


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
    rotation = np.column_stack((right, up, -forward))
    return matrix_to_quaternion(rotation)


def fmt(values):
    return " ".join(f"{float(value):.10g}" for value in values)


def build(arm_path, output_path):
    root = ET.Element("mujoco", model="B601_RS_FastWAM_paper_box_basket")
    ET.SubElement(root, "compiler", angle="radian", meshdir="../")
    ET.SubElement(root, "option", timestep="0.001", iterations="100", solver="Newton", integrator="Euler")
    contact = ET.SubElement(root, "contact")
    for first, second in [("world", "link1"), ("base_link", "link1"), ("link1", "link2"),
                          ("link2", "link3"), ("link2", "link5"), ("link3", "link4"),
                          ("link3", "link5"), ("link4", "link5"), ("link5", "link6"),
                          ("gripper_left", "gripper_right")]:
        ET.SubElement(contact, "exclude", body1=first, body2=second)
    ET.SubElement(root, "statistic", center="0.32 0 0.22", extent="0.72")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="224", offheight="224", azimuth="125", elevation="-28")
    ET.SubElement(visual, "headlight", diffuse="0.68 0.68 0.68", ambient="0.42 0.42 0.42", specular="0.1 0.1 0.1")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.30 0.48 0.65", rgb2="0.04 0.06 0.08", width="512", height="3072")
    ET.SubElement(asset, "material", name="table", rgba="0.34 0.37 0.36 1")
    ET.SubElement(asset, "material", name="paper", rgba="0.90 0.83 0.64 1")
    ET.SubElement(asset, "material", name="basket", rgba="0.08 0.40 0.85 1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.25 -0.15 1.3", dir="0.05 0.1 -1", directional="true")
    global_eye = np.array([0.82, -0.68, 0.72])
    global_target = np.array([0.36, 0.0, 0.10])
    ET.SubElement(world, "camera", name="global", mode="fixed", pos=fmt(global_eye), quat=fmt(look_at(global_eye, global_target, [0, 0, 1])), fovy="55")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 0.04", rgba="0.12 0.16 0.20 1", friction="1.2 0.01 0.001")
    table = ET.SubElement(world, "body", name="task_table", pos="0.36 0 0.05")
    ET.SubElement(table, "geom", name="task_table_geom", type="box", size="0.23 0.23 0.05", material="table", friction="1.8 0.02 0.002", solref="0.003 1", solimp="0.98 0.995 0.0005")
    box = ET.SubElement(world, "body", name="paper_box", pos="0.35 -0.08 0.151")
    ET.SubElement(box, "freejoint", name="paper_box_freejoint")
    ET.SubElement(box, "geom", name="paper_box_geom", type="box", size="0.04 0.03 0.05", material="paper", mass="0.035", friction="2.2 0.05 0.005", solref="0.002 1", solimp="0.99 0.995 0.0005")
    basket = ET.SubElement(world, "body", name="basket", pos="0.50 0.09 0.10")
    ET.SubElement(basket, "geom", name="basket_bottom", type="box", pos="0 0 0.003", size="0.075 0.065 0.003", material="basket", friction="1.5 0.02 0.002")
    for name, pos, size in [
        ("basket_left", "0 -0.065 0.022", "0.075 0.003 0.022"),
        ("basket_right", "0 0.065 0.022", "0.075 0.003 0.022"),
        ("basket_back", "0.075 0 0.022", "0.003 0.065 0.022"),
        ("basket_front", "-0.075 0 0.022", "0.003 0.065 0.022"),
    ]:
        ET.SubElement(basket, "geom", name=name, type="box", pos=pos, size=size, material="basket", friction="1.5 0.02 0.002")
    arm = ET.parse(arm_path).getroot()
    gripper = arm.find(".//body[@name='gripper_end']")
    if gripper is None:
        raise RuntimeError("Official model gripper_end missing")
    # TCP is the official gripper_end origin; fingers extend from negative X to the origin along local +X.
    ET.SubElement(gripper, "site", name="b601_eef", pos="0 0 0", size="0.003", rgba="0 1 0 0")
    eye = np.array([-0.130, 0.0, 0.070])
    target = np.array([0.020, 0.0, -0.005])
    ET.SubElement(gripper, "camera", name="wrist", pos=fmt(eye), quat=fmt(look_at(eye, target, [0, 0, 1])), fovy="65")
    # Inline the official arm after adding the simulated wrist camera.
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
    args = parser.parse_args()
    build(args.arm.resolve(), args.out.resolve())
    print(args.out.resolve())
