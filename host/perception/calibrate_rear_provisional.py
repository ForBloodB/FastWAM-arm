from pathlib import Path
import argparse
import json

import cv2
import numpy as np


def solve_at_focal(obj, img, focal, width, height):
    matrix = np.array([[focal, 0.0, (width - 1) / 2],
                       [0.0, focal, (height - 1) / 2],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float("inf"), None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, matrix, None, rvec, tvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, None)
    residual = projected.reshape(-1, 2) - img.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    return rms, (matrix, rvec, tvec, projected.reshape(-1, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--handeye", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError("Cannot read rear camera image")
    height, width = image.shape[:2]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard((10, 6), 0.015, 0.011, dictionary)
    board.setLegacyPattern(True)
    parameters = cv2.aruco.CharucoParameters()
    parameters.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(board, parameters)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 30:
        raise RuntimeError(f"Insufficient ChArUco corners: {0 if ids is None else len(ids)}")
    obj = board.getChessboardCorners()[ids.flatten()].astype(np.float64)
    img = np.asarray(corners, dtype=np.float64).reshape(-1, 2)

    # A single planar view can provide only a provisional focal estimate.
    # Principal point, square pixels and zero distortion are explicit assumptions.
    low, high = 300.0, 1800.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = high - ratio * (high - low), low + ratio * (high - low)
    f1 = solve_at_focal(obj, img, x1, width, height)[0]
    f2 = solve_at_focal(obj, img, x2, width, height)[0]
    for _ in range(70):
        if f1 < f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - ratio * (high - low)
            f1 = solve_at_focal(obj, img, x1, width, height)[0]
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + ratio * (high - low)
            f2 = solve_at_focal(obj, img, x2, width, height)[0]
    focal = (low + high) / 2.0
    rms, result = solve_at_focal(obj, img, focal, width, height)
    if result is None or rms > 1.5:
        raise RuntimeError(f"Provisional rear-camera fit failed: RMS={rms}")
    matrix, rvec, tvec, projected = result
    rotation, _ = cv2.Rodrigues(rvec)
    t_camera_board = np.eye(4)
    t_camera_board[:3, :3] = rotation
    t_camera_board[:3, 3] = tvec.ravel()
    handeye = json.loads(args.handeye.read_text())
    if not handeye.get("quality_gate_passed"):
        raise RuntimeError("Source wrist hand-eye calibration did not pass its quality gate")
    t_base_board = np.asarray(handeye["T_base_board"], dtype=np.float64)
    t_base_camera = t_base_board @ np.linalg.inv(t_camera_board)

    overlay = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids, (0, 255, 255))
    cv2.drawFrameAxes(overlay, matrix, None, rvec, tvec, 0.045)
    cv2.imwrite(str(args.out / "rear_calibration_overlay.jpg"), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    fovy_deg = float(np.degrees(2.0 * np.arctan(height / (2.0 * focal))))
    output = {
        "status": "provisional_single_view_not_for_hardware_motion",
        "camera_role": "rear_fixed_eye_to_hand",
        "image": str(args.image),
        "resolution": [width, height],
        "charuco_corners": int(len(ids)),
        "reprojection_rms_px": rms,
        "assumptions": [
            "Calibration board has not moved since the wrist hand-eye solution",
            "Principal point is at image center; fx=fy; lens distortion is zero",
            "Single-view focal estimate is for simulation view matching only",
        ],
        "camera_matrix": matrix.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "fovy_deg": fovy_deg,
        "T_camera_board": t_camera_board.tolist(),
        "T_base_camera_opencv": t_base_camera.tolist(),
        "camera_position_base_m": t_base_camera[:3, 3].tolist(),
        "source_T_base_board": t_base_board.tolist(),
        "multi_view_intrinsics_completed": False,
        "independent_physical_validation_required": True,
        "hardware_execution_enabled": False,
    }
    (args.out / "rear_camera_provisional.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
