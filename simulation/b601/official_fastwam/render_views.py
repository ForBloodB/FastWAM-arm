import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--initial-q", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
import json
report = json.loads(args.initial_q.read_text())
model = mujoco.MjModel.from_xml_path(str(args.scene))
data = mujoco.MjData(model)
for index in range(1, 7):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
    data.qpos[model.jnt_qposadr[joint_id]] = report["initial_q"][index - 1]
for name in ("joint7", "joint_left", "joint_right"):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = 0.05
mujoco.mj_forward(model, data)
args.out.mkdir(parents=True, exist_ok=False)
renderer = mujoco.Renderer(model, height=224, width=224)
for camera in ("global", "wrist"):
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(args.out / f"{camera}.png")
renderer.close()
print(args.out)
