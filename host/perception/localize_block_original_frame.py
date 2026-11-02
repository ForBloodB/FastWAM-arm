"""Read-only high-block RGB-D localization in the original hand-eye frame."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--robot-state',type=Path,required=True);p.add_argument('--handeye',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
image=cv2.imread(str(a.capture/'color.jpg'));depth=np.load(a.capture/'aligned_depth.npy');factory=json.loads((a.capture/'summary.json').read_text());robot=json.loads(a.robot_state.read_text());handeye=json.loads(a.handeye.read_text())
intr=factory['native_factory']['color_intrinsic'];ds=factory['native_factory']['color_distortion'];K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]]);D=np.array([ds.get(k,0.) for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
Tbc=np.asarray(robot['T_base_gripper'])@np.asarray(handeye['T_gripper_camera']);hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);raw=cv2.inRange(hsv,np.array([75,45,55]),np.array([115,255,255]));mask=np.zeros_like(raw);mask[270:465,550:960]=raw[270:465,550:960];mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8));count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask);label=max(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA]);x,y,w,h,area=stats[label]
if area<1000:raise RuntimeError('Block ROI missing')
component=labels==label;eroded=cv2.erode(component.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);ys,xs=np.nonzero(eroded&(depth>0));z=depth[ys,xs].astype(float)*factory['aligned_depth_scale_m'];norm=cv2.undistortPoints(np.stack([xs,ys],1).astype(float).reshape(-1,1,2),K,D).reshape(-1,2);pc=np.column_stack([norm*z[:,None],z,np.ones(len(z))]);pb=(Tbc@pc.T).T[:,:3]
bands=[]
for percentile in (75,80,85,90):
 ztop=float(np.percentile(pb[:,2],percentile));sel=np.abs(pb[:,2]-ztop)<.0025
 if sel.sum()<50:continue
 points=pb[sel];spread=np.percentile(points,[10,90],axis=0);widths=(spread[1]-spread[0])*1000;bands.append({'percentile':percentile,'pixels':int(sel.sum()),'z_m':float(np.median(points[:,2])),'std_mm':float(np.std(points[:,2])*1000),'widths_mm':widths.tolist(),'target_base_m':np.median(points,axis=0).tolist()})
valid=[b for b in bands if b['std_mm']<=2 and all(12<=v<=48 for v in b['widths_mm'][:2])]
result={'source':'fresh RGB-D original hand-eye frame; no-contact proposal only','T_base_camera':Tbc.tolist(),'bbox_xywh':[int(x),int(y),int(w),int(h)],'area_px':int(area),'valid_depth_pixels':len(z),'bands':bands,'valid':bool(valid),'target_base_m':valid[0]['target_base_m'] if valid else None,'direct_contact_allowed':False}
(a.out/'localization.json').write_text(json.dumps(result,indent=2));cv2.imwrite(str(a.out/'mask.png'),mask);annotated=image.copy();cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,0,255),3);cv2.imwrite(str(a.out/'annotated.jpg'),annotated);print(json.dumps(result,indent=2));raise SystemExit(0 if valid else 2)
