"""Localize the plain 36x30x70 mm printed block from real wrist RGB-D."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--robot-state", type=Path, required=True)
parser.add_argument("--handeye", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)

image = cv2.imread(str(args.capture / "color.jpg"))
depth = np.load(args.capture / "aligned_depth.npy")
factory = json.loads((args.capture / "summary.json").read_text())
robot = json.loads(args.robot_state.read_text())
handeye = json.loads(args.handeye.read_text())
if image is None or depth.shape != image.shape[:2]:
    raise RuntimeError("Color/aligned depth capture is invalid")
if not handeye.get("quality_gate_passed"):
    raise RuntimeError("Wrist hand-eye quality gate did not pass")
scale = float(factory["aligned_depth_scale_m"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([75, 45, 55]), np.array([115, 255, 255]))
mask[: image.shape[0] // 2] = 0
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
candidates = []
for label in range(1, count):
    x, y, w, h, area = stats[label]
    if area < 500:
        continue
    component = labels == label
    values = depth[component & (depth > 0)] * scale
    candidates.append({
        "label": label,
        "area_px": int(area),
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "centroid_px": centroids[label].tolist(),
        "valid_depth_pixels": int(values.size),
        "depth_median_m": float(np.median(values)) if values.size else None,
    })
workspace = [
    c for c in candidates
    if c["bbox_xywh"][1] > image.shape[0] * 0.55
    and c["depth_median_m"] is not None
    and c["depth_median_m"] < 0.6
]
if not workspace:
    raise RuntimeError(f"No near lower-workspace printed object: {candidates}")
# The plain high block is the largest near cyan component. The socket, peg and
# notched block are retained in evidence but are not eligible for this task.
candidate = max(workspace, key=lambda c: c["area_px"])
component = labels == candidate["label"]
depth_mask = cv2.erode(component.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
values = depth[depth_mask & (depth > 0)] * scale
if values.size < 100:
    raise RuntimeError("Insufficient high-block depth")
ys, xs = np.nonzero(depth_mask & (depth > 0))
z = depth[ys, xs].astype(np.float64) * scale
intrinsic = factory["native_factory"]["color_intrinsic"]
distortion = factory["native_factory"]["color_distortion"]
k = np.array([
    [intrinsic["fx"], 0, intrinsic["cx"]],
    [0, intrinsic["fy"], intrinsic["cy"]],
    [0, 0, 1.0],
])
dist = np.array([distortion.get(s, 0.0) for s in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
normalized = cv2.undistortPoints(np.stack([xs, ys], axis=1).astype(np.float64).reshape(-1, 1, 2), k, dist).reshape(-1, 2)
points_camera_all = np.column_stack([normalized * z[:, None], z, np.ones(len(z))])
t_base_gripper = np.asarray(robot["T_base_gripper"], dtype=np.float64)
t_gripper_camera = np.asarray(handeye["T_gripper_camera"], dtype=np.float64)
t_base_camera = t_base_gripper @ t_gripper_camera
points_base_all = (t_base_camera @ points_camera_all.T).T[:, :3]
# A camera-depth band is not a horizontal plane with this oblique wrist view.
# Select the horizontal top after transforming the whole component to base.
z_top = float(np.percentile(points_base_all[:, 2], 85))
top_band = np.abs(points_base_all[:, 2] - z_top) < 0.0025
if np.count_nonzero(top_band) < 80:
    raise RuntimeError("High-block horizontal top plane is too sparse")
points_camera = points_camera_all[top_band]
points_base = points_base_all[top_band]
target = np.median(points_base, axis=0)
spread = np.percentile(points_base, [10, 90], axis=0)
base_z_std_mm = float(np.std(points_base[:, 2]) * 1000)
top_widths_mm = (spread[1] - spread[0]) * 1000
if base_z_std_mm > 2.0 or np.any(top_widths_mm[:2] < 12.0) or np.any(top_widths_mm[:2] > 60.0):
    raise RuntimeError(f"Top-plane geometry rejected: target={target.tolist()}, z_std_mm={base_z_std_mm}, widths_mm={top_widths_mm.tolist()}")
result = {
    "source": "real Gemini 305 RGB-D + measured KR260 TF + quality-gated hand-eye",
    "object": "plain printed high block 36x30x70 mm",
    "candidate": candidate,
    "all_candidates": candidates,
    "top_plane": {
        "base_height_p85_m": z_top,
        "band_pixels": int(np.count_nonzero(top_band)),
        "base_p10_m": spread[0].tolist(),
        "base_p90_m": spread[1].tolist(),
        "base_z_std_mm": float(np.std(points_base[:, 2]) * 1000),
    },
    "target_base_m": target.tolist(),
    "target_is_top_surface": True,
    "current_gripper_base_m": t_base_gripper[:3, 3].tolist(),
    "translation_target_minus_gripper_m": (target - t_base_gripper[:3, 3]).tolist(),
    "hardware_execution_enabled": False,
}
(args.out / "localization.json").write_text(json.dumps(result, indent=2))
annotated = image.copy()
x, y, w, h = candidate["bbox_xywh"]
cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
center_camera = np.median(points_camera[:, :3], axis=0)
center_px = cv2.projectPoints(center_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3), k, dist)[0].reshape(2)
cv2.circle(annotated, tuple(np.round(center_px).astype(int)), 9, (0, 0, 255), -1)
cv2.putText(annotated, "plain high block top", (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imwrite(str(args.out / "annotated.jpg"), annotated)
cv2.imwrite(str(args.out / "mask.png"), mask)
print(json.dumps(result, indent=2))
