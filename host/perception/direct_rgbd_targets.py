"""Direct RGB-D top-center localization for the corrected left-to-right task."""
import json
from pathlib import Path
import cv2
import numpy as np

prefix='corrected_home'
root=Path('/home/aup/marker_closed_loop');cap=Path(f'/home/aup/{prefix}_rgbd')
state=json.loads(Path(f'/home/aup/{prefix}_state.json').read_text())
handeye=json.loads((root/'handeye_used.json').read_text())
factory=json.loads((cap/'summary.json').read_text());depth=np.load(cap/'aligned_depth.npy')*factory['aligned_depth_scale_m']
intr=factory['native_factory']['color_intrinsic'];ds=factory['native_factory']['color_distortion']
K=np.array([[intr['fx'],0.,intr['cx']],[0.,intr['fy'],intr['cy']],[0.,0.,1.]])
D=np.array([ds.get(k,0.) for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
Tbc=np.asarray(state['T_base_gripper'])@np.asarray(handeye['T_gripper_camera'])

def measure(name,pixel):
    u,v=pixel
    ray=cv2.undistortPoints(np.array([[[float(u),float(v)]]]),K,D).reshape(2)
    sample=depth[v-10:v+11,u-10:u+11];valid=sample[(sample>.05)&(sample<2.)]
    if len(valid)<100:raise RuntimeError(name+' depth unavailable')
    z=float(np.median(valid));point=np.array([ray[0]*z,ray[1]*z,z,1.])
    base=Tbc@point
    return {'center_px':[u,v],'valid_depth_pixels':int(len(valid)),
            'depth_median_m':z,'depth_std_mm':float(np.std(valid)*1000),
            'camera_point_m':point[:3].tolist(),'top_base_m':base[:3].tolist()}
left=measure('left',(450,450));right=measure('right',(735,440))
result={'source':'direct RGB-D top-center rays + current TF + eye-in-hand',
        'left_grip_block':left,'right_platform':right,
        'task':'pick left solid block and place on right platform','motion_authorized':False}
Path(f'/home/aup/{prefix}_direct_targets.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
