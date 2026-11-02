"""Board-anchored wrist PnP for the visible 80x65 mm tray; read-only."""
import json
from pathlib import Path
import cv2
import numpy as np

root = Path('/home/aup')
capture = root / 'final_observe_rgbd'
image = cv2.imread(str(capture / 'color.jpg'))
depth = np.load(capture / 'aligned_depth.npy')
factory = json.loads((capture / 'summary.json').read_text())
handeye = json.loads((root / 'handeye_used.json').read_text())
intr = factory['native_factory']['color_intrinsic']
ds = factory['native_factory']['color_distortion']
K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
D = np.array([ds.get(k, 0.) for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')])
board = cv2.aruco.CharucoBoard((10, 6), .015, .011, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
board.setLegacyPattern(True)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
dp = cv2.aruco.DetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
cp = cv2.aruco.CharucoParameters()
cp.tryRefineMarkers = True
cp.minMarkers = 1
corners, ids, _, _ = cv2.aruco.CharucoDetector(board, cp, dp).detectBoard(enhanced)
if ids is None or len(ids) < 20:
    raise RuntimeError('Insufficient board corners')
obj_board = board.getChessboardCorners()[ids.flatten()]
pix_board = np.asarray(corners).reshape(-1, 2)
und_board = cv2.undistortPoints(pix_board[:, None, :], K, D, P=K).reshape(-1, 2)
ok, rvec, tvec, _ = cv2.solvePnPRansac(obj_board, und_board, K, None, reprojectionError=3., iterationsCount=500)
if not ok:
    raise RuntimeError('Board PnP failed')
rvec, tvec = cv2.solvePnPRefineLM(obj_board, und_board, K, None, rvec, tvec)
Rb, _ = cv2.Rodrigues(rvec)
Tcb = np.eye(4)
Tcb[:3, :3] = Rb
Tcb[:3, 3] = tvec.ravel()
Tbc = np.asarray(handeye['T_base_board']) @ np.linalg.inv(Tcb)
pixels = np.array([[501., 574.], [675., 594.], [675., 742.], [467., 724.]])
obj = np.array([[-.04, -.0325, 0], [.04, -.0325, 0], [.04, .0325, 0], [-.04, .0325, 0]])
und = cv2.undistortPoints(pixels[:, None, :], K, D, P=K)
ok, rvec_o, tvec_o = cv2.solvePnP(obj, und, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
if not ok:
    raise RuntimeError('Object PnP failed')
Ro, _ = cv2.Rodrigues(rvec_o)
Tco = np.eye(4)
Tco[:3, :3] = Ro
Tco[:3, 3] = tvec_o.ravel()
Tbo = Tbc @ Tco
projected, _ = cv2.projectPoints(obj, rvec_o, tvec_o, K, D)
rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
values = depth[630:690, 540:610]
values = values[values > 0] * factory['aligned_depth_scale_m']
depth_z = float(np.median(values))
result = {'source': 'fresh wrist PnP anchored to stationary Charuco board; independent of current arm TF',
          'board_corners': int(len(ids)), 'target_corners_px': pixels.tolist(),
          'reprojection_rms_px': rms, 'pnp_camera_z_m': float(Tco[2, 3]),
          'rgbd_center_z_m': depth_z, 'depth_disagreement_m': abs(float(Tco[2, 3]) - depth_z),
          'target_base_m': Tbo[:3, 3].tolist(), 'T_base_object': Tbo.tolist(),
          'T_base_camera': Tbc.tolist(), 'object_top_dimensions_m': [.08, .065],
          'direct_contact_allowed': False}
(root / 'final_board_localization.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
