"""Offline RGB-D localization with explicit calibration, segmentation and units.

Does not claim to recognize unknown materials. A supplied instance mask comes
from a detector/operator; task selection must supply its semantic label.
"""
import numpy as np


def locate_mask(depth, mask, intrinsics, camera_to_base, depth_scale=.001):
    depth=np.asarray(depth)
    mask=np.asarray(mask,bool)
    k=np.asarray(intrinsics,float)
    transform=np.asarray(camera_to_base,float)
    if depth.shape!=mask.shape or depth.ndim!=2 or k.shape!=(3,3) or transform.shape!=(4,4):
        raise ValueError('Invalid RGB-D/calibration dimensions')
    if not np.isfinite(k).all() or not np.isfinite(transform).all() or not np.isfinite(depth_scale) or depth_scale<=0 or k[0,0]<=0 or k[1,1]<=0:
        raise ValueError('Invalid calibration or depth scale')
    if not np.allclose(transform[3],[0,0,0,1]) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-4):
        raise ValueError('Camera-to-base must be a rigid homogeneous transform')
    v,u=np.nonzero(mask & np.isfinite(depth) & (depth>0))
    if len(v)<20: raise ValueError('Too few valid depth pixels')
    z=depth[v,u].astype(float)*depth_scale
    median=np.median(z)
    good=np.abs(z-median)<max(.015,.03*median)
    u,v,z=u[good],v[good],z[good]
    if len(z)<20 or not .05<float(np.median(z))<3.:
        raise ValueError('Insufficient inliers or invalid operating depth')
    xyz=np.c_[(u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z]
    base=xyz@transform[:3,:3].T+transform[:3,3]
    center=np.median(base,axis=0)
    _,vectors=np.linalg.eigh(np.cov(base[:,:2],rowvar=False))
    axis=vectors[:,-1]
    yaw=float(np.arctan2(axis[1],axis[0]))
    return {'position_m':center.tolist(),'yaw_rad':yaw,'inliers':len(z),
            'depth_mad_m':float(np.median(abs(z-np.median(z)))),
            'yaw_ambiguity_rad':float(np.pi),'source':'calibrated_RGBD_mask'}
