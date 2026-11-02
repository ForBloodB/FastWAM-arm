"""Fixed-camera checkerboard fit from a fresh global image; no actuators."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--handeye',type=Path,required=True);p.add_argument('--intrinsics',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
im=cv2.imread(str(a.image));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ok,corners=cv2.findChessboardCorners(gray,(9,5),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
if not ok:raise RuntimeError('9x5 checkerboard not detected')
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,80,.001);corners=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
obj=np.array([[x*.015,y*.015,0.] for y in range(1,6) for x in range(1,10)],float);prior=json.loads(a.intrinsics.read_text());K=np.asarray(prior['camera_matrix'],float);ok,rvec,tvec,inliers=cv2.solvePnPRansac(obj,corners,K,None,reprojectionError=2.,iterationsCount=500)
if not ok:raise RuntimeError('Global PnP failed')
rvec,tvec=cv2.solvePnPRefineLM(obj,corners,K,None,rvec,tvec);projected,_=cv2.projectPoints(obj,rvec,tvec,K,None);rms=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2,axis=1))))
if rms>1.5:raise RuntimeError(f'Global reprojection error too high: {rms}')
Tcb=np.eye(4);Tcb[:3,:3]=cv2.Rodrigues(rvec)[0];Tcb[:3,3]=tvec.ravel();hand=json.loads(a.handeye.read_text());Tbb=np.asarray(hand['T_base_board']);Tbc=Tbb@np.linalg.inv(Tcb)
result={'source':'fresh fixed-global-camera 9x5 square-grid corners; independent of current wrist transform','corners':len(corners),'reprojection_rms_px':rms,'camera_matrix':K.tolist(),'intrinsics':'same-camera prior from 2026-10-29 single-view fit; zero distortion assumed','T_camera_board':Tcb.tolist(),'T_base_camera_opencv':Tbc.tolist(),'camera_position_base_m':Tbc[:3,3].tolist(),'prior_camera_position_base_m':prior['camera_position_base_m'],'camera_position_shift_m':float(np.linalg.norm(Tbc[:3,3]-np.asarray(prior['camera_position_base_m']))),'T_base_board_assumed_unchanged':Tbb.tolist(),'direct_contact_allowed':False}
(a.out/'calibration.json').write_text(json.dumps(result,indent=2));overlay=im.copy();cv2.drawChessboardCorners(overlay,(9,5),corners,True);cv2.drawFrameAxes(overlay,K,None,rvec,tvec,.04);cv2.imwrite(str(a.out/'overlay.jpg'),overlay);np.save(a.out/'corners.npy',corners);print(json.dumps(result,indent=2))
