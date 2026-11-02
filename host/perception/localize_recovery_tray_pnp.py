"""Read-only four-corner PnP check for the recovered rear tray pose."""
import json
from pathlib import Path
import cv2
import numpy as np

root = Path('/home/aup')
capture = root / 'recovery_rgbd_02'
factory = json.loads((capture / 'summary.json').read_text())
robot = json.loads((root / 'recovery_current_state.json').read_text())
handeye = json.loads((root / 'handeye_used.json').read_text())
intr = factory['native_factory']['color_intrinsic']
ds = factory['native_factory']['color_distortion']
K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
D = np.array([ds.get(k, 0.) for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')])
pixels = np.array([[472., 320.], [621., 307.], [665., 359.], [492., 380.]])
obj = np.array([[-.04, -.0325, 0], [.04, -.0325, 0], [.04, .0325, 0], [-.04, .0325, 0]])
undistorted = cv2.undistortPoints(pixels[:, None, :], K, D, P=K)
ok, rvec, tvec = cv2.solvePnP(obj, undistorted, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
if not ok:
    raise RuntimeError('PnP failed')
R, _ = cv2.Rodrigues(rvec)
Tco = np.eye(4)
Tco[:3, :3] = R
Tco[:3, 3] = tvec.ravel()
Tbc = np.asarray(robot['T_base_gripper']) @ np.asarray(handeye['T_gripper_camera'])
Tbo = Tbc @ Tco
projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
center = cv2.undistortPoints(np.array([[[568., 344.]]]), K, D, P=K)[0, 0]
depth = np.load(capture / 'aligned_depth.npy')
values = depth[330:360, 550:590]
values = values[values > 0] * factory['aligned_depth_scale_m']
depth_z = float(np.median(values))
expected_z = float(Tco[2, 3])
result = {'source': 'fresh wrist four-corner PnP with original hand-eye; no-contact proposal',
          'manual_corners_px': pixels.tolist(), 'reprojection_rms_px': rms,
          'pnp_camera_z_m': expected_z, 'rgbd_center_z_m': depth_z,
          'depth_disagreement_m': abs(expected_z - depth_z), 'target_base_m': Tbo[:3, 3].tolist(),
          'T_base_object': Tbo.tolist(), 'object_top_dimensions_m': [.08, .065],
          'direct_contact_allowed': False}
(root / 'recovery_pnp_localization.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
