"""Localize the left solid block and right placement platform in board/base frames."""
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
factory = json.loads((cap / 'summary.json').read_text())
image = cv2.imread(str(cap / 'color.jpg')); depth = np.load(cap / 'aligned_depth.npy')
intr=factory['native_factory']['color_intrinsic'];ds=factory['native_factory']['color_distortion']
K=np.array([[intr['fx'],0.,intr['cx']],[0.,intr['fy'],intr['cy']],[0.,0.,1.]])
D=np.array([ds.get(k,0.) for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
board=cv2.aruco.CharucoBoard((10,6),.015,.011,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50));board.setLegacyPattern(True)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray)
cp=cv2.aruco.CharucoParameters();cp.tryRefineMarkers=True;cp.minMarkers=1
dp=cv2.aruco.DetectorParameters();dp.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
corners,ids,_,_=cv2.aruco.CharucoDetector(board,cp,dp).detectBoard(enhanced)
if ids is None or len(ids)<12:raise RuntimeError(f'Insufficient board corners: {0 if ids is None else len(ids)}')
obj=board.getChessboardCorners()[ids.flatten()];pix=corners.reshape(-1,2)
und=cv2.undistortPoints(pix[:,None,:],K,D,P=K).reshape(-1,2)
ok,rb,tb,inliers=cv2.solvePnPRansac(obj,und,K,None,reprojectionError=3.,iterationsCount=1000)
rb,tb=cv2.solvePnPRefineLM(obj,und,K,None,rb,tb)
Rb,_=cv2.Rodrigues(rb);Tcb=np.eye(4);Tcb[:3,:3]=Rb;Tcb[:3,3]=tb.ravel()
proj,_=cv2.projectPoints(obj,rb,tb,K,D);rms=float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2)-pix)**2,axis=1))))

def solve_object(name,pixels,dimensions,depth_roi):
    pixels=np.asarray(pixels,dtype=float)
    dx,dy=dimensions[0]/2,dimensions[1]/2
    points=np.array([[-dx,-dy,0.],[dx,-dy,0.],[dx,dy,0.],[-dx,dy,0.]])
    und=cv2.undistortPoints(pixels[:,None,:],K,D,P=K)
    ok,r,t=cv2.solvePnP(points,und,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise RuntimeError(name+' PnP failed')
    R,_=cv2.Rodrigues(r);Tco=np.eye(4);Tco[:3,:3]=R;Tco[:3,3]=t.ravel()
    Tbo=np.linalg.inv(Tcb)@Tco
    projected,_=cv2.projectPoints(points,r,t,K,D)
    error=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-pixels)**2,axis=1))))
    x0,y0,x1,y1=depth_roi;sample=depth[y0:y1,x0:x1];sample=sample[sample>0]*factory['aligned_depth_scale_m']
    measured=float(np.median(sample))
    return {'corners_px':pixels.tolist(),'dimensions_m':list(dimensions),'reprojection_rms_px':error,
            'pnp_camera_depth_m':float(Tco[2,3]),'rgbd_depth_m':measured,
            'depth_disagreement_m':abs(float(Tco[2,3])-measured),
            'T_camera_object':Tco.tolist(),'T_board_object':Tbo.tolist(),
            'top_board_m':Tbo[:3,3].tolist(),'height_above_board_mm':float(-Tbo[2,3]*1000)}

left=solve_object('left_grip_block',[[428,398],[511,401],[491,500],[389,496]],(.030,.036,.070),(420,420,480,480))
right=solve_object('right_platform',[[645,389],[788,394],[849,485],[654,493]],(.080,.065,.040),(680,410,780,460))
Tbb=np.asarray(state['T_base_gripper'])@np.asarray(handeye['T_gripper_camera'])@Tcb
for item in (left,right):
    item['T_base_object']= (Tbb@np.asarray(item['T_board_object'])).tolist()
    item['top_base_m']=np.asarray(item['T_base_object'])[:3,3].tolist()
result={'source':'fresh wrist RGB-D + ChArUco + correct left/right task geometry',
        'board_corners':int(len(ids)),'board_inliers':int(len(inliers)),'board_reprojection_rms_px':rms,
        'T_camera_board':Tcb.tolist(),'T_base_board':Tbb.tolist(),'left_grip_block':left,'right_platform':right,
        'task':'pick left 36x30x70 mm solid block; place on right 80x65x40 mm platform',
        'motion_authorized':False}
Path(f'/home/aup/{prefix}_localization.json').write_text(json.dumps(result,indent=2))
overlay=image.copy();cv2.aruco.drawDetectedCornersCharuco(overlay,corners,ids)
for item,color in ((left,(0,255,0)),(right,(0,0,255))):
    for p in np.asarray(item['corners_px']).astype(int):cv2.circle(overlay,tuple(p),6,color,2)
cv2.imwrite(f'/home/aup/{prefix}_overlay.png',overlay)
print(json.dumps(result,indent=2))
