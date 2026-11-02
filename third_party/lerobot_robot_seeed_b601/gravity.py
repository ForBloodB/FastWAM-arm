"""最简重力补偿计算模块（纯 Python，不依赖 Pinocchio）。

只保留 demo ``9b_gravity_compensation_purepy.py`` 控制环真正用到的三件事：
解析 URDF、向量化快速路径计算 g(q)、暴露同名接口。递归版 / 势能 /
有限差分 / selftest 全部裁掉，单文件可独立使用，速度与 demo 一致
（向量化快速路径，~0.5 ms/call，可支撑 500 Hz）。

对外接口（与 ``gravity_purepy`` 同名同参，可直接替换 import）::

    load_dynamics_model(urdf_path=None) -> PurePyModel
    compute_generalized_gravity(model=None, q=None, gravity=None) -> np.ndarray
    get_default_gravity() -> np.ndarray

原理：重力是纯力，关节轴上力矩只取决于各连杆质心位置与质量，与转动惯量
无关。对每个可动关节 i（世界轴向 a_i、轴心点 p_i），下游子树连杆集合
subtree(i) 的重力平衡力矩为

    g_i(q) = -a_i · Σ_{j∈subtree(i)} (c_j - p_i) × (m_j · g_vec)

等价于势能 V = Σ m_j·9.81·z_j 对 q 的梯度。向量化实现：首次调用预计算
拓扑/常量到 ``model._fast``，之后每帧只做 numpy 批量 FK + 向量力矩累加。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── 重力常量 ──────────────────────────────────────────────────────────────────
EARTH_GRAVITY: tuple[float, float, float] = (0.0, 0.0, -9.81)

# ── URDF 路径解析（与 gravity_purepy 一致，yaml 缺失时回退默认 URDF）─────────
_cfg_dir = Path(__file__).resolve().parents[2] / "config"
_project_root = _cfg_dir.parent
_DEFAULT_URDF = "urdf/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf"
_hw_cfg_cache: dict | None = None
_MODEL_CACHE: dict[str, "PurePyModel"] = {}


def _hw_config() -> dict:
    global _hw_cfg_cache
    if _hw_cfg_cache is not None:
        return _hw_cfg_cache
    try:
        import yaml
    except ImportError:
        return {}
    hw_yaml = ""
    global_cfg = _cfg_dir / "rebotarm.yaml"
    if global_cfg.exists():
        d = yaml.safe_load(global_cfg.read_text()) or {}
        hw_yaml = d.get("hardware_yaml", hw_yaml)
    hw_path = _cfg_dir / hw_yaml
    if not hw_path.exists():
        return {}
    _hw_cfg_cache = yaml.safe_load(hw_path.read_text()) or {}
    return _hw_cfg_cache


def _resolve_urdf(urdf_path: Optional[str] = None) -> str:
    if urdf_path:
        p = urdf_path
        return p if Path(p).is_absolute() else str(_project_root / p)
    try:
        p = (_hw_config() or {}).get("urdf_path", "")
        if p:
            return p if Path(p).is_absolute() else str(_project_root / p)
    except Exception:
        pass
    return str(_project_root / _DEFAULT_URDF)


# ── 旋转工具 ──────────────────────────────────────────────────────────────────
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rpy_to_R(roll, pitch, yaw):
    """URDF rpy -> 旋转矩阵，R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def _axis_angle_R(axis, angle):
    """Rodrigues 公式。"""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    x, y, z = a
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


# ── URDF 解析 ─────────────────────────────────────────────────────────────────
@dataclass
class _LinkInfo:
    name: str
    mass: float
    com: np.ndarray  # 质心在连杆坐标系下位置 (3,)


@dataclass
class _JointInfo:
    name: str
    jtype: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    parent: str
    child: str
    axis: np.ndarray
    q_idx: int = -1  # 可动关节广义坐标下标；fixed 为 -1


@dataclass
class PurePyModel:
    """纯 Python 机器人模型。"""

    nq: int
    nv: int
    links: dict[str, _LinkInfo] = field(default_factory=dict)
    joints: list[_JointInfo] = field(default_factory=list)
    root: str = ""
    children: dict[str, list[int]] = field(default_factory=dict)


def _parse_floats(text, n):
    vals = [float(v) for v in text.replace(",", " ").split()]
    if len(vals) != n:
        raise ValueError(f"期望 {n} 个数，实际 {len(vals)}: {text!r}")
    return np.array(vals, dtype=float)


def _parse_urdf(path: str) -> PurePyModel:
    tree = ET.parse(path)
    root = tree.getroot()

    links: dict[str, _LinkInfo] = {}
    for link_el in root.findall("link"):
        name = link_el.get("name", "")
        mass = 0.0
        com = np.zeros(3)
        inertial = link_el.find("inertial")
        if inertial is not None:
            m_el = inertial.find("mass")
            if m_el is not None:
                mass = float(m_el.get("value", "0"))
            o_el = inertial.find("origin")
            if o_el is not None and o_el.get("xyz"):
                com = _parse_floats(o_el.get("xyz", "0 0 0"), 3)
        links[name] = _LinkInfo(name=name, mass=mass, com=com)

    joints: list[_JointInfo] = []
    for j_el in root.findall("joint"):
        o_el = j_el.find("origin")
        xyz = _parse_floats(o_el.get("xyz", "0 0 0"), 3) if o_el is not None else np.zeros(3)
        rpy = _parse_floats(o_el.get("rpy", "0 0 0"), 3) if o_el is not None else np.zeros(3)
        parent = j_el.find("parent").get("link", "")
        child = j_el.find("child").get("link", "")
        a_el = j_el.find("axis")
        axis = _parse_floats(a_el.get("xyz", "0 0 1"), 3) if a_el is not None else np.array([0, 0, 1.0])
        nrm = float(np.linalg.norm(axis))
        axis = axis / nrm if nrm > 1e-12 else np.array([0, 0, 1.0])
        joints.append(
            _JointInfo(
                name=j_el.get("name", ""),
                jtype=j_el.get("type", "fixed"),
                origin_xyz=xyz,
                origin_rpy=rpy,
                parent=parent,
                child=child,
                axis=axis,
            )
        )

    # 根连杆 = 从未作为任何 joint child 的连杆
    child_links = {j.child for j in joints}
    root_candidates = [name for name in links if name not in child_links]
    if not root_candidates:
        raise ValueError("URDF 未找到根连杆")
    root = root_candidates[0]

    # 从基座到末端 DFS 序，给可动关节编号
    children_traverse: dict[str, list[int]] = {name: [] for name in links}
    for i, j in enumerate(joints):
        children_traverse.setdefault(j.parent, []).append(i)

    ordered: list[_JointInfo] = []

    def _dfs(link: str) -> None:
        for ji in children_traverse.get(link, []):
            j = joints[ji]
            ordered.append(j)
            _dfs(j.child)

    _dfs(root)

    movable = [j for j in ordered if j.jtype in ("revolute", "prismatic", "continuous")]
    for idx, j in enumerate(movable):
        j.q_idx = idx

    children: dict[str, list[int]] = {name: [] for name in links}
    for idx, j in enumerate(ordered):
        children[j.parent].append(idx)

    return PurePyModel(
        nq=len(movable),
        nv=len(movable),
        links=links,
        joints=ordered,
        root=root,
        children=children,
    )


# ── 向量化快速路径：预计算扁平数组 ───────────────────────────────────────────
def _build_fast_cache(model: PurePyModel) -> dict:
    """把拓扑/常量预计算成 numpy 数组，控制环每帧只做批量 FK + 力矩累加。"""
    names = list(model.links.keys())
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    mass = np.array([model.links[n].mass for n in names], dtype=float)
    com = np.array([model.links[n].com for n in names], dtype=float)
    J = len(model.joints)
    parent_link = np.zeros(J, dtype=int)
    child_link = np.zeros(J, dtype=int)
    R_origin = np.zeros((J, 3, 3))
    origin_xyz = np.zeros((J, 3))
    axis = np.zeros((J, 3))
    q_idx = np.full(J, -1, dtype=int)
    is_rev = np.zeros(J, dtype=bool)
    is_pris = np.zeros(J, dtype=bool)
    for j, ji in enumerate(model.joints):
        parent_link[j] = idx[ji.parent]
        child_link[j] = idx[ji.child]
        R_origin[j] = _rpy_to_R(*ji.origin_rpy)
        origin_xyz[j] = ji.origin_xyz
        axis[j] = ji.axis
        q_idx[j] = ji.q_idx
        is_rev[j] = ji.jtype in ("revolute", "continuous")
        is_pris[j] = ji.jtype == "prismatic"

    children_of_link: dict[str, list[int]] = {n: [] for n in names}
    for ji in model.joints:
        children_of_link[ji.parent].append(idx[ji.child])

    def _subtree_idx(root: int) -> np.ndarray:
        out: list[int] = []
        stack = [root]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(children_of_link[names[x]])
        return np.array(out, dtype=int)

    sub_idx = [_subtree_idx(int(child_link[j])) for j in range(J)]
    nq = int((q_idx >= 0).sum())
    return {
        "N": N, "J": J, "nq": nq,
        "mass": mass, "com": com,
        "parent_link": parent_link, "child_link": child_link,
        "R_origin": R_origin, "origin_xyz": origin_xyz,
        "axis": axis, "q_idx": q_idx,
        "is_rev": is_rev, "is_pris": is_pris,
        "sub_idx": sub_idx, "root_idx": idx[model.root],
    }


def _get_fast_cache(model: PurePyModel) -> dict:
    """惰性构建并缓存 model._fast。"""
    fc = getattr(model, "_fast", None)
    if fc is None:
        fc = _build_fast_cache(model)
        object.__setattr__(model, "_fast", fc)
    return fc


def _pad_q(model: PurePyModel, q) -> np.ndarray:
    if q is None:
        return np.zeros(model.nq)
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] == model.nq:
        return q
    padded = np.zeros(model.nq)
    padded[: min(q.shape[0], model.nq)] = q[: min(q.shape[0], model.nq)]
    return padded


# ── 对外 API ──────────────────────────────────────────────────────────────────
def load_dynamics_model(urdf_path: Optional[str] = None) -> PurePyModel:
    """加载模型（解析 URDF），按解析后的路径缓存，返回前预热向量化快速路径。

    重复调用同一 URDF 零开销返回已预热的模型。首次调用解析 URDF + 构建
    ``model._fast``，之后每帧只做批量 FK + 力矩累加。
    """
    path = _resolve_urdf(urdf_path)
    cached = _MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    model = _parse_urdf(path)
    # 预热：首次调用构建 _fast 缓存，避免控制环首帧延迟。
    compute_generalized_gravity(model=model, q=np.zeros(model.nq))
    _MODEL_CACHE[path] = model
    return model


def get_default_gravity() -> np.ndarray:
    """默认重力加速度 [0, 0, -9.81] m/s²。"""
    return np.array(EARTH_GRAVITY)


def compute_generalized_gravity(
    model: Optional[PurePyModel] = None,
    q: Optional[np.ndarray] = None,
    data: Optional[object] = None,  # 对齐签名；纯 Python 无需 data
    gravity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """计算广义重力向量 g(q)，返回 shape=(nq,) 力矩 (N·m)，可直接作 MIT 模式 tau 前馈。

    向量化快速路径：首次调用构建 model._fast，之后每帧批量 FK + 向量力矩累加。
    """
    if model is None:
        model = load_dynamics_model()
    q = _pad_q(model, q)
    g_vec = np.asarray(gravity, dtype=float) if gravity is not None else get_default_gravity()

    fc = _get_fast_cache(model)
    N, J = fc["N"], fc["J"]
    parent_link = fc["parent_link"]
    child_link = fc["child_link"]
    R_origin = fc["R_origin"]
    origin_xyz = fc["origin_xyz"]
    axis = fc["axis"]
    q_idx = fc["q_idx"]
    is_rev = fc["is_rev"]
    is_pris = fc["is_pris"]

    # ── 正运动学：每连杆世界位姿 (R_world, t_world)，DFS 序保证父先于子 ──
    R_world = np.zeros((N, 3, 3))
    R_world[fc["root_idx"]] = np.eye(3)
    t_world = np.zeros((N, 3))
    for j in range(J):
        p = parent_link[j]
        Rp = R_world[p]
        tp = t_world[p]
        R_fixed = Rp @ R_origin[j]
        t_fixed = tp + Rp @ origin_xyz[j]
        qi = q_idx[j]
        if is_rev[j]:
            R_child = R_fixed @ _axis_angle_R(axis[j], q[qi])
            t_child = t_fixed
        elif is_pris[j]:
            R_child = R_fixed
            t_child = t_fixed + R_fixed @ (axis[j] * q[qi])
        else:  # fixed
            R_child = R_fixed
            t_child = t_fixed
        cl = child_link[j]
        R_world[cl] = R_child
        t_world[cl] = t_child

    # 各连杆质心世界坐标 (N,3)
    com_world = np.einsum("nij,nj->ni", R_world, fc["com"]) + t_world
    mass = fc["mass"]
    sub_idx = fc["sub_idx"]

    tau = np.zeros(fc["nq"])
    for j in range(J):
        qi = q_idx[j]
        if qi < 0:
            continue
        p = parent_link[j]
        Rp = R_world[p]
        tp = t_world[p]
        R_fixed = Rp @ R_origin[j]
        axis_world = R_fixed @ axis[j]
        p_i = tp + Rp @ origin_xyz[j]
        s = sub_idx[j]
        if is_rev[j]:
            # revolute: 力矩 = -axis · Σ (c-p) × (m·g)
            rel = com_world[s] - p_i                  # (k,3)
            fm = mass[s][:, None] * g_vec             # (k,3)
            moment = np.cross(rel, fm).sum(axis=0)    # (3,)
            tau[qi] = -float(axis_world @ moment)
        else:  # prismatic: 广义力 = -axis · Σ m·g
            force = (mass[s][:, None] * g_vec).sum(axis=0)
            tau[qi] = -float(axis_world @ force)
    return tau


if __name__ == "__main__":
    # 冒烟测试：加载模型并打印若干构型的 g(q)
    m = load_dynamics_model()
    print(f"nq={m.nq}, nv={m.nv}, root={m.root}")
    for q in (np.zeros(m.nq), np.array([1.2, -1.2, -1.0, 0.8, 0.5, -0.6, 0.1, -0.1])[: m.nq]):
        g = compute_generalized_gravity(m, q)
        print(f"q={np.array2string(q, precision=3)}  g(q)={np.array2string(g, precision=4)}  N·m")
