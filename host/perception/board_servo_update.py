"""Closed-loop update from the visible flat ChArUco board; no motion commands."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

prefix = sys.argv[1]
root = Path('/home/aup/marker_closed_loop')
cap = Path(f'/home/aup/{prefix}_rgbd')
state = json.loads(Path(f'/home/aup/{prefix}_state.json').read_text())
handeye = json.loads((root / 'handeye_used.json').read_text())
initial = json.loads((root / 'board_object_localization.json').read_text())
factory = json.loads((cap / 'summary.json').read_text())
image = cv2.imread(str(cap / 'color.jpg'))
intr = factory['native_factory']['color_intrinsic']; ds = factory['native_factory']['color_distortion']
K = np.array([[intr['fx'],0.,intr['cx']],[0.,intr['fy'],intr['cy']],[0.,0.,1.]])
D = np.array([ds.get(k,0.) for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
board = cv2.aruco.CharucoBoard((10,6), .015, .011, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
board.setLegacyPattern(True)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cp = cv2.aruco.CharucoParameters(); cp.tryRefineMarkers=True; cp.minMarkers=1
dp = cv2.aruco.DetectorParameters(); dp.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
corners, ids, _, _ = cv2.aruco.CharucoDetector(board, cp, dp).detectBoard(gray)
if ids is None or len(ids)<20: raise RuntimeError('Insufficient board corners')
obj = board.getChessboardCorners()[ids.flatten()]
pix = corners.reshape(-1,2)
und = cv2.undistortPoints(pix[:,None,:],K,D,P=K).reshape(-1,2)
ok,rvec,tvec,inliers = cv2.solvePnPRansac(obj,und,K,None,reprojectionError=3.,iterationsCount=1000)
if not ok: raise RuntimeError('Board PnP failed')
rvec,tvec = cv2.solvePnPRefineLM(obj,und,K,None,rvec,tvec)
R,_ = cv2.Rodrigues(rvec)
Tcb=np.eye(4);Tcb[:3,:3]=R;Tcb[:3,3]=tvec.ravel()
proj,_=cv2.projectPoints(obj,rvec,tvec,K,D)
rms=float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2)-pix)**2,axis=1))))
Tbg=np.asarray(state['T_base_gripper']);Tgc=np.asarray(handeye['T_gripper_camera'])
Tbb=Tbg@Tgc@Tcb
Tbo=np.asarray(initial['T_board_object'])
Ttarget=Tbb@Tbo
previous=np.asarray(initial['target_base_m']) if 'target_base_m' in initial else np.array([0.4438909534166615,-0.08314708360881555,0.18400671633321541])
current_tcp=Tbg[:3,3]
desired_pre=Ttarget[:3,3]+np.array([0.,0.,.144])
error=desired_pre-current_tcp
placement=Tbb@np.array([.065,.0375,0.,1.])
result={'board_corners':int(len(ids)),'board_inliers':int(len(inliers)),
        'board_reprojection_rms_px':rms,'T_camera_board':Tcb.tolist(),
        'T_base_board':Tbb.tolist(),'updated_target_top_base_m':Ttarget[:3,3].tolist(),
        'target_update_from_initial_mm':((Ttarget[:3,3]-previous)*1000).tolist(),
        'current_tcp_m':current_tcp.tolist(),'desired_pregrasp_tcp_m':desired_pre.tolist(),
        'pregrasp_correction_mm':(error*1000).tolist(),
        'placement_surface_base_m':placement[:3].tolist(),
        'quality_passed':bool(rms<2. and np.linalg.norm(error)<.08),
        'motion_authorized':False}
Path(f'/home/aup/{prefix}_update.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
