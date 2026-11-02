"""Read-only ChArUco diagnostics on the newly captured scene."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--image', required=True)
p.add_argument('--camera-summary')
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
image = cv2.imread(a.image)
if image is None:
    raise RuntimeError('No input image')
board = cv2.aruco.CharucoBoard((10, 6), .015, .011, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
board.setLegacyPattern(True)
K = D = None
if a.camera_summary:
    summary = json.loads(Path(a.camera_summary).read_text())
    factory = summary['native_factory']
    c = factory['color_intrinsic']
    d = factory['color_distortion']
    K = np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1.]], dtype=float)
    D = np.array([d[x] for x in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']], dtype=float)
rows = []
for variant in ['original', 'contrast', 'upscaled']:
    if variant == 'original':
        work = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif variant == 'contrast':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    else:
        work = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 53
    detector_params.adaptiveThreshWinSizeStep = 4
    detector_params.minMarkerPerimeterRate = .012
    detector_params.polygonalApproxAccuracyRate = .05
    detector_params.minCornerDistanceRate = .02
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    charuco_params.minMarkers = 1
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(work)
    row = {'variant': variant, 'markers': 0 if marker_ids is None else len(marker_ids),
           'corners': 0 if ids is None else len(ids)}
    if ids is not None and len(ids) >= 4:
        scale = 2 if variant == 'upscaled' else 1
        pixels = np.asarray(corners).reshape(-1, 2) / scale
        obj = board.getChessboardCorners()[ids.flatten()]
        if K is None:
            h, w = image.shape[:2]
            K0 = np.array([[700., 0, (w - 1) / 2], [0, 700., (h - 1) / 2], [0, 0, 1.]])
            pose, rvec, tvec, inliers = cv2.solvePnPRansac(obj, pixels, K0, None, reprojectionError=3, iterationsCount=200)
        else:
            undistorted = cv2.undistortPoints(pixels[:, None, :], K, D, P=K).reshape(-1, 2)
            pose, rvec, tvec, inliers = cv2.solvePnPRansac(obj, undistorted, K, None, reprojectionError=3, iterationsCount=200)
        if pose:
            projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
            residual = projected.reshape(-1, 2) - pixels
            row.update(pose_ok=True, reprojection_rms_px=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                       T_camera_board=np.block([[cv2.Rodrigues(rvec)[0], tvec.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist())
            overlay = image.copy()
            cv2.aruco.drawDetectedCornersCharuco(overlay, np.asarray(corners) / scale, ids, (0, 255, 255))
            cv2.drawFrameAxes(overlay, K, D, rvec, tvec, .03)
            cv2.imwrite(str(out / (variant + '_overlay.jpg')), overlay)
    rows.append(row)
result = {'image': a.image, 'factory_intrinsics_used': K is not None, 'rows': rows}
(out / 'diagnostic.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
