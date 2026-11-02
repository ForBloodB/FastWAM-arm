"""Offline-only official RS model loader. No robot or networking interfaces.

Vendor files are immutable. The derived MJCF keeps official visual/contact
meshes, gripper coupling and scene, and aligns link inertias and limits to the
official URDF. Controller parameters are experimental, not hardware settings.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6c2bb32b3f7f430c58a37cf93d459d07ee99d736"
VENDOR = ROOT / "assets/vendor/seeed_b601_rs" / COMMIT
RS = VENDOR / "Rebot_Arm_description/RS"
URDF = RS / "urdf/ReBot_Arm_RS.urdf"
SCENE = RS / "mjcf/rs_grasp_scene.xml"
DERIVED = ROOT / "assets/sim_official_b601_rs"
MODEL = DERIVED / "rs_urdf_aligned_scene.xml"


def verify_vendor():
    manifest = json.loads((VENDOR / "VENDOR_MANIFEST.json").read_text())
    for item in manifest["files"]:
        data = (VENDOR / item["path"]).read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Official model source changed: " + item["path"])
    return manifest


def _nums(value):
    return np.fromstring(value, sep=" ")


def build_model():
    """Generate a transparent URDF-dynamics-aligned adapter, never edit vendor."""
    verify_vendor()
    urdf = ET.parse(URDF).getroot()
    arm = ET.parse(RS / "mjcf/rs_arm.xml").getroot()
    changes = []
    links = {x.get("name"): x for x in urdf.findall("link")}
    for body in arm.findall(".//body"):
        name = body.get("name")
        if name not in links:
            continue
        src = links[name].find("inertial")
        dst = body.find("inertial")
        if src is None or dst is None:
            raise ValueError("Missing explicit inertia " + name)
        origin = src.find("origin")
        if not np.allclose(_nums(origin.get("rpy", "0 0 0")), 0):
            raise ValueError("Nonzero inertia orientation requires explicit tensor rotation")
        inertia = src.find("inertia")
        new = {"pos": origin.get("xyz"), "mass": src.find("mass").get("value"),
               "fullinertia": " ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))}
        changes.append({"body": name, "field": "inertial", "vendor_mjcf": dict(dst.attrib), "official_urdf": new})
        dst.attrib.clear()
        dst.attrib.update(new)
    joint_map = {**{f"joint{i}": f"joint{i}" for i in range(1, 7)},
                 "joint_left": "gripper_joint1", "joint_right": "gripper_joint2"}
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    for dst in arm.findall(".//joint"):
        name = dst.get("name")
        if name not in joint_map:
            continue
        src = joints[joint_map[name]].find("limit")
        new_range = src.get("lower") + " " + src.get("upper")
        changes.append({"joint": name, "field": "range", "vendor_mjcf": dst.get("range"), "official_urdf": new_range})
        dst.set("range", new_range)
    # Six arm actuator efforts already equal the URDF; fail on disagreement.
    for i in range(1, 7):
        motor = arm.find(f"actuator/motor[@name='joint{i}_motor']")
        effort = float(joints[f"joint{i}"].find("limit").get("effort"))
        if not np.allclose(_nums(motor.get("ctrlrange")), [-effort, effort]):
            raise ValueError("Actuator effort mismatch")
    import os
    meshdir = Path(os.path.relpath(RS / "meshes", DERIVED)).as_posix()
    arm.find("compiler").set("meshdir", meshdir)
    scene = ET.parse(SCENE).getroot()
    scene.find("include").set("file", "rs_urdf_aligned_arm.xml")
    scene.find("compiler").set("meshdir", meshdir)
    DERIVED.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(arm).write(DERIVED / "rs_urdf_aligned_arm.xml", encoding="utf-8", xml_declaration=True)
    ET.ElementTree(scene).write(MODEL, encoding="utf-8", xml_declaration=True)
    audit = {"official_commit": COMMIT, "official_urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
             "changes": changes, "visual_and_collision_geometries": "Unchanged from official MJCF",
             "unchanged_simulator_assumptions": ["Official contact friction/compliance and exclusions",
                 "Official simplified gripper drive coupling and 0.001 kg virtual coupler",
                 "Convex mesh contact per official segmented geometry; not exact nonconvex triangle contact",
                 "Default D405 camera visual; not the user's Gemini 305 calibration or sensor model"],
             "simulation_only": True, "hardware_commands": 0}
    (DERIVED / "CONVERSION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def load_urdf_reference():
    """Native MuJoCo URDF import with path-only and importer settings adapter."""
    tree = ET.parse(URDF)
    for mesh in tree.findall(".//mesh"):
        mesh.set("filename", str((URDF.parent / mesh.get("filename")).resolve()))
    extra = ET.SubElement(tree.getroot(), "mujoco")
    ET.SubElement(extra, "compiler", discardvisual="false", fusestatic="false", strippath="false")
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


class OfficialB601Simulation:
    """Torque-controlled simulation adapter; joint7 is metres, NOT motor radians."""
    KP = np.array([180., 220., 180., 90., 55., 30.])
    KD = np.array([18., 22., 18., 7., 4., 2.])

    def __init__(self):
        if not MODEL.exists():
            raise FileNotFoundError("Run build_model() first")
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.jids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 7)])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.dids = self.model.jnt_dofadr[self.jids]
        self.aids = np.array([self.model.actuator(f"joint{i}_motor").id for i in range(1, 7)])
        self.gjid = self.model.joint("joint7").id
        self.gqid = self.model.jnt_qposadr[self.gjid]
        self.gdid = self.model.jnt_dofadr[self.gjid]
        self.gaid = self.model.actuator("joint7_motor").id
        self.ranges = self.model.jnt_range[self.jids].copy()

    def reset(self, q=(0, .4, .8, -.4, 0, 0), opening=.035):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < self.ranges[:, 0]) or np.any(q > self.ranges[:, 1]):
            raise ValueError("Reset outside official URDF limits")
        if not np.isfinite(opening) or not 0 <= opening <= .05:
            raise ValueError("Reset outside simulation gripper limits")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = q
        for name in ("joint7", "joint_left", "joint_right"):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = opening
        mujoco.mj_forward(self.model, self.data)
        return self.data.site("tcp").xpos.copy()

    def step(self, target, opening=.035, grip_force_cap=12.):
        target = np.asarray(target, float)
        if target.shape != (6,) or not np.isfinite(target).all() or np.any(target < self.ranges[:, 0]) or np.any(target > self.ranges[:, 1]):
            raise ValueError("Joint command outside official URDF limits")
        if not np.isfinite([opening, grip_force_cap]).all() or not 0 <= opening <= .05 or not 0 < grip_force_cap <= 30:
            raise ValueError("Simulation gripper envelope exceeded")
        # Explicit ideal model-based compensation; dynamics are not disabled.
        torque = self.KP * (target - self.data.qpos[self.qids]) - self.KD * self.data.qvel[self.dids]
        torque += self.data.qfrc_bias[self.dids]
        self.data.ctrl[self.aids] = np.clip(torque, self.model.actuator_ctrlrange[self.aids, 0], self.model.actuator_ctrlrange[self.aids, 1])
        force = 1800 * (opening-self.data.qpos[self.gqid]) - 18 * self.data.qvel[self.gdid]
        self.data.ctrl[self.gaid] = np.clip(force, -grip_force_cap, grip_force_cap)
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")

    def contact_pairs(self):
        return sorted({tuple(sorted((self.model.body(int(self.model.geom_bodyid[c.geom1])).name,
                                     self.model.body(int(self.model.geom_bodyid[c.geom2])).name)))
                       for c in self.data.contact if c.dist < 0})
