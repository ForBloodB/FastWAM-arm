"""Localize the front 80x65x40 mm shallow sorting tray from validated RGB-D."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--robot-state',type=Path,required=True)
p.add_argument('--handeye',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
image=cv2.imread(str(a.capture/'color.jpg'));depth=np.load(a.capture/'aligned_depth.npy')
factory=json.loads((a.capture/'summary.json').read_text());robot=json.loads(a.robot_state.read_text());handeye=json.loads(a.handeye.read_text())
if image is None or depth.shape!=image.shape[:2] or not handeye.get('quality_gate_passed'):raise RuntimeError('Invalid capture or handeye')
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,np.array([75,45,55]),np.array([115,255,255]));mask[:image.shape[0]//2]=0
mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8));mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
candidates=[]
for label in range(1,count):
    x,y,w,h,area=stats[label];component=labels==label;values=depth[component&(depth>0)]*float(factory['aligned_depth_scale_m'])
    if area>=500 and y>image.shape[0]*.55 and values.size and np.median(values)<.6:
        candidates.append({'label':label,'area_px':int(area),'bbox_xywh':[int(x),int(y),int(w),int(h)],'centroid_px':centroids[label].tolist(),'depth_median_m':float(np.median(values))})
if not candidates:raise RuntimeError('No platform candidate')
# In the validated capture the tray is the closest/front-most separate cyan component.
candidate=max(candidates,key=lambda c:c['centroid_px'][1]);component=labels==candidate['label'];eroded=cv2.erode(component.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
ys,xs=np.nonzero(eroded&(depth>0));z=depth[ys,xs].astype(float)*float(factory['aligned_depth_scale_m'])
if len(z)<100:raise RuntimeError('Insufficient platform depth')
intr=factory['native_factory']['color_intrinsic'];k=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
d=factory['native_factory']['color_distortion'];dist=np.array([d.get(s,0.) for s in ('k1','k2','p1','p2','k3','k4','k5','k6')])
norm=cv2.undistortPoints(np.stack([xs,ys],1).astype(float).reshape(-1,1,2),k,dist).reshape(-1,2);pc=np.column_stack([norm*z[:,None],z,np.ones(len(z))])
tb=np.asarray(robot['T_base_gripper'])@np.asarray(handeye['T_gripper_camera']);pb=(tb@pc.T).T[:,:3]
ztop=.00627817315665888;coarse=np.abs(pb[:,2]-ztop)<.008
if np.count_nonzero(coarse)<40:raise RuntimeError(f'Sparse platform rim near model height: {np.count_nonzero(coarse)}')
zmed=float(np.median(pb[coarse,2]));top=coarse&(np.abs(pb[:,2]-zmed)<.003)
if np.count_nonzero(top)<40:raise RuntimeError('Sparse refined platform rim')
points=pb[top];target=np.median(points,axis=0);target[2]=ztop;spread=np.percentile(points,[10,90],axis=0);std=float(np.std(points[:,2])*1000);widths=(spread[1]-spread[0])*1000
if std>2.5 or np.any(widths[:2]<30) or np.any(widths[:2]>110) or not -.02<=target[2]<=.03:raise RuntimeError(f'Platform geometry rejected target={target.tolist()} std={std} widths={widths.tolist()} candidate={candidate} all={candidates}')
result={'source':'validated earlier RGB-D pose; objects subsequently unchanged','object':'printed shallow sorting tray 80x65x40 mm','candidate':candidate,'all_candidates':candidates,'target_base_m':target.tolist(),'rim_z_std_mm':std,'rim_p10_p90_m':spread.tolist(),'rim_widths_mm':widths.tolist(),'hardware_execution_enabled':False}
(a.out/'localization.json').write_text(json.dumps(result,indent=2));annotated=image.copy();x,y,w,h=candidate['bbox_xywh'];cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,0,0),3);cv2.putText(annotated,'sorting tray rim',(x,max(30,y-12)),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,0,0),2);cv2.imwrite(str(a.out/'annotated.jpg'),annotated)
print(json.dumps(result,indent=2))
