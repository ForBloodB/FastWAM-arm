"""Board-anchored RGB-D localization of the rearranged block and tray.

This produces a no-contact observation proposal, not permission to grasp.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--capture', type=Path, required=True)
p.add_argument('--robot-state', type=Path, required=True)
p.add_argument('--handeye', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
image = cv2.imread(str(a.capture / 'color.jpg'))
depth = np.load(a.capture / 'aligned_depth.npy')
factory = json.loads((a.capture / 'summary.json').read_text())
robot = json.loads(a.robot_state.read_text())
handeye = json.loads(a.handeye.read_text())
intr = factory['native_factory']['color_intrinsic']
d = factory['native_factory']['color_distortion']
K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
D = np.array([d.get(k, 0.) for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')])
board = cv2.aruco.CharucoBoard((10, 6), .015, .011, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
board.setLegacyPattern(True)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
dp = cv2.aruco.DetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dp.adaptiveThreshWinSizeMin = 3
dp.adaptiveThreshWinSizeMax = 53
dp.adaptiveThreshWinSizeStep = 4
dp.minMarkerPerimeterRate = .012
dp.polygonalApproxAccuracyRate = .05
dp.minCornerDistanceRate = .02
cp = cv2.aruco.CharucoParameters()
cp.tryRefineMarkers = True
cp.minMarkers = 1
corners, ids, _, _ = cv2.aruco.CharucoDetector(board, cp, dp).detectBoard(enhanced)
if ids is None or len(ids) < 20:
    raise RuntimeError('Insufficient stationary board corners')
obj = board.getChessboardCorners()[ids.flatten()]
pixels = np.asarray(corners).reshape(-1, 2)
undistorted = cv2.undistortPoints(pixels[:, None, :], K, D, P=K).reshape(-1, 2)
ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, undistorted, K, None, reprojectionError=3., iterationsCount=500)
if not ok:
    raise RuntimeError('Board PnP failed')
rvec, tvec = cv2.solvePnPRefineLM(obj, undistorted, K, None, rvec, tvec)
projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels)**2, axis=1))))
if rms > 1.5:
    raise RuntimeError(f'Board reprojection error too high: {rms}')
R, _ = cv2.Rodrigues(rvec)
Tcb = np.eye(4)
Tcb[:3, :3] = R
Tcb[:3, 3] = tvec.ravel()
Tbg = np.asarray(robot['T_base_gripper'])
Tbb = np.asarray(handeye['T_base_board'])
Tbc = Tbb @ np.linalg.inv(Tcb)
Tgc = np.linalg.inv(Tbg) @ Tbc

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([75, 45, 55]), np.array([115, 255, 255]))
mask[:220] = 0
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
candidates = []
for label in range(1, count):
    x, y, w, h, area = stats[label]
    if area < 700 or x < 200 or x > 1100 or y < 230:
        continue
    component = labels == label
    eroded = cv2.erode(component.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ys, xs = np.nonzero(eroded & (depth > 0))
    if len(z := depth[ys, xs]) < 100:
        continue
    z = z.astype(float) * factory['aligned_depth_scale_m']
    if np.median(z) > .8:
        continue
    norm = cv2.undistortPoints(np.stack([xs, ys], 1).astype(float).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    pc = np.column_stack([norm * z[:, None], z, np.ones(len(z))])
    pb = (Tbc @ pc.T).T[:, :3]
    bands = []
    for percentile in (75, 80, 85, 90):
        ztop = float(np.percentile(pb[:, 2], percentile))
        sel = np.abs(pb[:, 2] - ztop) < .0025
        if np.count_nonzero(sel) < 50:
            continue
        points = pb[sel]
        spread = np.percentile(points, [10, 90], axis=0)
        widths = (spread[1] - spread[0]) * 1000
        bands.append({'percentile': percentile, 'pixels': int(np.count_nonzero(sel)),
                      'z_m': float(np.median(points[:, 2])), 'std_mm': float(np.std(points[:, 2]) * 1000),
                      'widths_mm': widths.tolist(), 'target_base_m': np.median(points, axis=0).tolist()})
    candidates.append({'label': label, 'area_px': int(area), 'bbox_xywh': [int(x), int(y), int(w), int(h)],
                       'centroid_px': centroids[label].tolist(), 'depth_median_m': float(np.median(z)),
                       'valid_depth_pixels': len(z), 'bands': bands})
block_options = []
for c in candidates:
    for b in c['bands']:
        wx, wy = b['widths_mm'][:2]
        if b['std_mm'] <= 2. and 12 <= wx <= 48 and 12 <= wy <= 48 and c['centroid_px'][1] < 500:
            block_options.append((c, b))
if not block_options:
    (a.out / 'diagnostic.json').write_text(json.dumps({'board_rms': rms, 'candidates': candidates}, indent=2))
    cv2.imwrite(str(a.out / 'mask.png'), mask)
    raise RuntimeError(f'No valid high-block top; candidates={candidates}')
block_options.sort(key=lambda cb: (abs(cb[1]['widths_mm'][0] - 30) + abs(cb[1]['widths_mm'][1] - 25), cb[1]['std_mm']))
block, top = block_options[0]
tray_options = [c for c in candidates if c['label'] != block['label'] and c['centroid_px'][1] > block['centroid_px'][1]]
tray = max(tray_options, key=lambda c: c['centroid_px'][1]) if tray_options else None
Tgo_old = Tbg @ np.asarray(handeye['T_gripper_camera'])
result = {
    'source': 'fresh Gemini RGB-D plus stationary board anchor; for no-contact observation only',
    'block_model_m': [.036, .030, .070], 'tray_model_m': [.080, .065, .040],
    'board': {'corners': len(ids), 'reprojection_rms_px': rms, 'T_camera_board': Tcb.tolist()},
    'board_assumption': 'Printed board stayed at the hand-eye T_base_board location',
    'T_base_camera_board_anchored': Tbc.tolist(),
    'T_gripper_camera_board_anchored': Tgc.tolist(),
    'T_base_camera_previous': Tgo_old.tolist(),
    'previous_extrinsic_board_position_error_m': float(np.linalg.norm((Tbg @ np.asarray(handeye['T_gripper_camera']) @ Tcb)[:3, 3] - Tbb[:3, 3])),
    'target_base_m': top['target_base_m'], 'block': block, 'block_top': top,
    'tray_candidate': tray, 'candidates': candidates,
    'hardware_execution_enabled': False, 'direct_contact_allowed': False,
    'required_next_step': 'Move only to a collision-checked elevated pregrasp, then acquire a fresh RGB-D pose before contact'
}
(a.out / 'localization.json').write_text(json.dumps(result, indent=2))
cv2.imwrite(str(a.out / 'mask.png'), mask)
annotated = image.copy()
for c, color, text in [(block, (0, 0, 255), 'HIGH BLOCK')] + ([(tray, (255, 0, 0), 'TRAY')] if tray else []):
    x, y, w, h = c['bbox_xywh']
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
    cv2.putText(annotated, text, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids, (0, 255, 255))
cv2.imwrite(str(a.out / 'annotated.jpg'), annotated)
print(json.dumps(result, indent=2))
