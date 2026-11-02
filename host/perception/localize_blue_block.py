import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--robot-state', type=Path, required=True)
parser.add_argument('--handeye', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)

image = cv2.imread(str(args.capture / 'color.jpg'))
depth = np.load(args.capture / 'aligned_depth.npy')
factory = json.loads((args.capture / 'summary.json').read_text())
robot = json.loads(args.robot_state.read_text())
handeye = json.loads(args.handeye.read_text())
if image is None or depth.shape != image.shape[:2]:
    raise RuntimeError('Color/aligned depth capture is invalid')
scale = float(factory['aligned_depth_scale_m'])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Cyan/blue target only; restrict to the lower image half to reject background.
mask = cv2.inRange(hsv, np.array([75, 45, 55]), np.array([115, 255, 255]))
mask[:image.shape[0] // 2] = 0
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
        'label': label, 'area_px': int(area), 'bbox_xywh': [int(x), int(y), int(w), int(h)],
        'centroid_px': centroids[label].tolist(), 'valid_depth_pixels': int(values.size),
        'depth_median_m': float(np.median(values)) if values.size else None,
        'depth_p10_m': float(np.percentile(values, 10)) if values.size else None,
        'depth_p90_m': float(np.percentile(values, 90)) if values.size else None,
    })
if not candidates:
    raise RuntimeError('No cyan/blue object candidate')
# Reject the cyan leader arm and distant background: the tabletop target is
# a near component whose bounding box starts in the lower workspace.
workspace_candidates = [
    c for c in candidates
    if c['bbox_xywh'][1] > image.shape[0] * 0.55
    and c['depth_median_m'] is not None and c['depth_median_m'] < 0.6
]
if not workspace_candidates:
    raise RuntimeError(f'No near lower-workspace target: {candidates}')
candidate = max(workspace_candidates, key=lambda c: c['area_px'])
if candidate['valid_depth_pixels'] < 100:
    raise RuntimeError(f'Insufficient target depth: {candidate}')
label = candidate['label']
component = labels == label
# Erode before depth pooling so edges and the background do not bias the target.
depth_mask = cv2.erode(component.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
values = depth[depth_mask & (depth > 0)] * scale
if values.size < 50:
    depth_mask = component
    values = depth[depth_mask & (depth > 0)] * scale
# The colored mask also contains the nearer vertical side face. Use the
# lower depth quintile and a narrow band to estimate the horizontal top rim.
z = float(np.percentile(values, 20))
ys, xs = np.nonzero(depth_mask & (depth > 0))
valid = np.abs(depth[ys, xs] * scale - z) < .003
if np.count_nonzero(valid) < 50:
    raise RuntimeError('Top-surface depth band is too sparse')
# The central through-hole gives a less biased XY center than the colored
# silhouette, which includes the vertical side face.
x0, y0, w0, h0 = candidate['bbox_xywh']
roi = hsv[y0:y0 + h0, x0:x0 + w0]
dark = cv2.inRange(roi, np.array([0, 0, 0]), np.array([179, 255, 75]))
dark[:12] = 0
dark[-12:] = 0
dark[:, :12] = 0
dark[:, -12:] = 0
hole_count, hole_labels, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(dark)
hole_candidates = [label for label in range(1, hole_count) if hole_stats[label, cv2.CC_STAT_AREA] >= 100]
if not hole_candidates:
    raise RuntimeError('Central target hole was not detected')
local_center = np.asarray(candidate['centroid_px']) - np.array([x0, y0])
hole = min(hole_candidates, key=lambda label: np.linalg.norm(hole_centroids[label] - local_center))
u = float(x0 + hole_centroids[hole][0])
v = float(y0 + hole_centroids[hole][1])
i = factory['native_factory']['color_intrinsic']
d = factory['native_factory']['color_distortion']
k = np.array([[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1.]], dtype=np.float64)
dist = np.array([d.get(s, 0.) for s in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')], dtype=np.float64)
normalized = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), k, dist).reshape(2)
point_camera = np.array([normalized[0] * z, normalized[1] * z, z, 1.])
t_base_gripper = np.asarray(robot['T_base_gripper'], dtype=np.float64)
t_gripper_camera = np.asarray(handeye['T_gripper_camera'], dtype=np.float64)
point_base = t_base_gripper @ t_gripper_camera @ point_camera
result = {
    'source': 'real Gemini 305 color + aligned depth + measured KR260 TF + calibrated hand-eye',
    'candidate': candidate,
    'all_candidates': candidates,
    'depth_pool': {'pixels': int(values.size), 'top_surface_p20_m': z, 'top_band_pixels': int(np.count_nonzero(valid)), 'p10_m': float(np.percentile(values, 10)), 'p90_m': float(np.percentile(values, 90))},
    'target_pixel_uv': [u, v],
    'target_camera_m': point_camera[:3].tolist(),
    'target_base_m': point_base[:3].tolist(),
    'current_gripper_base_m': t_base_gripper[:3, 3].tolist(),
    'translation_target_minus_gripper_m': (point_base[:3] - t_base_gripper[:3, 3]).tolist(),
    'target_is_top_surface_not_object_center': True,
    'cross_embodiment_mapping_validated': False,
}
(args.out / 'localization.json').write_text(json.dumps(result, indent=2))
annotated = image.copy()
x, y, w, h = candidate['bbox_xywh']
cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
cv2.circle(annotated, (round(u), round(v)), 8, (0, 0, 255), -1)
cv2.putText(annotated, f'cyan target z={z:.3f}m', (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
cv2.imwrite(str(args.out / 'annotated.jpg'), annotated)
cv2.imwrite(str(args.out / 'mask.png'), mask)
print(json.dumps(result, indent=2))
