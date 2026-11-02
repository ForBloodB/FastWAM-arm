"""Fuse wrist RGB-D and a manually verified planar top with global board pose."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--global-calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
im=cv2.imread(str(a.capture/'color.jpg'));depth=np.load(a.capture/'aligned_depth.npy');factory=json.loads((a.capture/'summary.json').read_text());intr=factory['native_factory']['color_intrinsic'];ds=factory['native_factory']['color_distortion'];K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]]);D=np.array([ds.get(k,0.) for k in ('k1','k2','p1','p2','k3','k4','k5','k6')]);cal=json.loads(a.global_calibration.read_text());Tbc=np.asarray(cal['T_base_camera_opencv']);pixels=np.array([[688.,328.],[833.,349.],[873.,409.],[677.,378.]])
# High rectangular target top: four corners, clockwise in the printed top plane.
rows=[]
for sx,sy in [(.036,.030),(.080,.065)]:
 obj=np.array([[-sx/2,-sy/2,0],[sx/2,-sy/2,0],[sx/2,sy/2,0],[-sx/2,sy/2,0]])
 ok,rvec,tvec=cv2.solvePnP(obj,pixels,K,D,flags=cv2.SOLVEPNP_IPPE)
 if not ok:continue
 center=np.array([[0.,0.,0.,1.]]).T;bc=Tbc@np.eye(4);Tco=np.eye(4);Tco[:3,:3]=cv2.Rodrigues(rvec)[0];Tco[:3,3]=tvec.ravel();base=(Tbc@Tco@center)[:3,0]
 patch=depth[350:388,730:820];z=patch[patch>0]*factory['aligned_depth_scale_m'];measured=float(np.median(z));predicted=float(tvec[2,0]);error=abs(predicted-measured)
 rows.append({'model_xy_m':[sx,sy],'T_camera_object':Tco.tolist(),'target_base_m':base.tolist(),'pnp_camera_z_m':predicted,'rgbd_median_z_m':measured,'depth_disagreement_m':error,'depth_scale_consistent':error<.025})
valid=[r for r in rows if r['depth_scale_consistent']]
if len(valid)!=1:raise RuntimeError(f'Could not uniquely match target geometry: {rows}')
result={'source':'fresh wrist RGB-D scale plus four visually checked top corners and independent global board pose','target_base_m':valid[0]['target_base_m'],'target_model':valid[0],'candidate_models':rows,'T_base_camera':Tbc.tolist(),'manual_corners_px':pixels.tolist(),'direct_contact_allowed':False,'requires_no_contact_reobservation':True}
(a.out/'localization.json').write_text(json.dumps(result,indent=2));annotated=im.copy()
for i,point in enumerate(pixels):cv2.circle(annotated,tuple(point.astype(int)),7,(0,0,255),-1);cv2.putText(annotated,str(i),tuple((point+8).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
cv2.imwrite(str(a.out/'corners.jpg'),annotated);print(json.dumps(result,indent=2))
