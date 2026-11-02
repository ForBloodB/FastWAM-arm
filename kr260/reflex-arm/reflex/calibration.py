"""ChArUco geometry; T_A_B maps B into A. This module cannot move motors."""
import cv2
import numpy as np

def board_model():
    return cv2.aruco.CharucoBoard((5,7),.024,.018,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

def camera_model(factory):
    c=factory['native_factory'];i=c['color_intrinsic'];d=c['color_distortion']
    k=np.array([[i['fx'],0,i['cx']],[0,i['fy'],i['cy']],[0,0,1.]],float)
    # Preserve all eight rational-model terms; truncating to five is incorrect.
    dist=np.array([d.get(s,0.) for s in ('k1','k2','p1','p2','k3','k4','k5','k6')],float)
    return k,dist,(i['width'],i['height'])

def rigid(t):
    t=np.asarray(t,float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('Invalid homogeneous transform')
    if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(t[:3,:3])<.99999:
        raise ValueError('Invalid rigid rotation')
    return t

def board_pose(image,factory):
    k,d,size=camera_model(factory)
    if image.shape[1::-1]!=size:raise ValueError('Image resolution differs from factory intrinsics')
    b=board_model();corners,ids,_,_=cv2.aruco.CharucoDetector(b).detectBoard(image)
    if ids is None or len(ids)<12:raise ValueError('At least 12 ChArUco corners required')
    obj=b.getChessboardCorners()[ids.flatten()].astype(float)
    pts=np.asarray(corners,float).reshape(-1,2)
    if cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))<.02*image.shape[0]*image.shape[1]:
        raise ValueError('Calibration board too small in image')
    ok,r,t=cv2.solvePnP(obj,pts,k,d,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('PnP failed')
    projected=cv2.projectPoints(obj,r,t,k,d)[0].reshape(-1,2)
    errors=np.linalg.norm(projected-pts,axis=1)
    transform=np.eye(4);transform[:3,:3]=cv2.Rodrigues(r)[0];transform[:3,3]=t.ravel()
    if np.min((transform[:3,:3]@obj.T+transform[:3,3:4])[2])<=0:raise ValueError('Board behind camera')
    rms=float(np.sqrt(np.mean(errors**2)))
    if rms>.7:raise ValueError(f'Board reprojection RMS {rms:.3f} px exceeds 0.7 px')
    return {'T_camera_board':transform.tolist(),'reprojection_rms_px':rms,'corners':len(ids),
            'ids':ids.flatten().tolist(),'image_points_px':pts.tolist(),'object_points_m':obj.tolist()}

def fixed_camera_candidate(t_base_board,t_camera_board):
    return rigid(t_base_board)@np.linalg.inv(rigid(t_camera_board))

def eye_in_hand_candidate(samples):
    """Fit on 80%, gate on held-out poses; no automatic actuator activation."""
    if len(samples)<15:raise ValueError('At least 15 stationary pose/image pairs required')
    gs=[rigid(s['T_base_gripper']) for s in samples];cs=[rigid(s['T_camera_board']) for s in samples]
    if any(s['reprojection_rms_px']>.7 for s in samples):raise ValueError('Poor image pose quality')
    rot=np.array([cv2.Rodrigues(g[:3,:3]@gs[0][:3,:3].T)[0].ravel() for g in gs[1:]])
    if np.linalg.svd(rot,compute_uv=False)[1]<np.deg2rad(10):
        raise ValueError('Rotations must excite at least two independent axes')
    if np.linalg.norm(np.ptp(np.array([g[:3,3] for g in gs]),axis=0))<.03:
        raise ValueError('Insufficient translation spread')
    train=[i for i in range(len(samples)) if i%5!=0];hold=[i for i in range(len(samples)) if i%5==0]
    r,t=cv2.calibrateHandEye([gs[i][:3,:3] for i in train],[gs[i][:3,3] for i in train],
                            [cs[i][:3,:3] for i in train],[cs[i][:3,3] for i in train],method=cv2.CALIB_HAND_EYE_PARK)
    x=np.eye(4);x[:3,:3]=r;x[:3,3]=t.ravel();rigid(x)
    worlds=[g@x@c for g,c in zip(gs,cs)]
    mean=np.eye(4);u,_,vt=np.linalg.svd(sum(worlds[i][:3,:3] for i in train))
    mean[:3,:3]=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    mean[:3,3]=np.mean([worlds[i][:3,3] for i in train],axis=0)
    errors=[]
    for i in hold:
        delta=np.linalg.inv(mean)@worlds[i]
        errors.append({'sample':i,'translation_m':float(np.linalg.norm(delta[:3,3])),
                       'rotation_deg':float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(delta[:3,:3])[0])))})
    accepted=all(e['translation_m']<.003 and e['rotation_deg']<1. for e in errors)
    return {'T_gripper_camera':x.tolist(),'T_base_board':mean.tolist(),'held_out_errors':errors,
            'quality_gate_passed':accepted,'independent_physical_validation_required':True,
            'hardware_execution_enabled':False,'fit_samples':train,'validation_samples':hold}
