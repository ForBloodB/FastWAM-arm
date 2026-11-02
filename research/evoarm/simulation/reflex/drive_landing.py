"""Measured object-to-TCP landing correction after bounded preload.

Landing uses a fresh RGBD object observation; no privileged simulated pose.
"""
import numpy as np
from .drive_preload import DrivePreloadEnv
from .official_grasp import DESTINATION
VARIANTS=['zero','mapped_preload','mapped_smooth','landing_preload','landing_smooth']

class DriveLandingEnv(DrivePreloadEnv):
    def __init__(self,variant='zero',**kw):
        if variant not in VARIANTS:raise ValueError(variant)
        self.landing_variant=variant
        super().__init__(dict(landing_preload='mapped_preload',landing_smooth='mapped_smooth').get(variant,variant),**kw)

    def move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        if phase=='lower' and self.landing_variant.startswith('landing_'):
            seen=self.observe('landing_localization',prior_xy=DESTINATION)
            relative=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp] if seen.get('pose_valid') else self.follow_relative
            if relative is not None and np.isfinite(relative).all():
                requested=np.r_[DESTINATION,.1+self.estimated_size[2]/2]+np.array([0.,0.,.003])-relative
                correction=np.clip(requested-np.asarray(xyz),[-.015,-.015,-.015],[.015,.015,.035])
                xyz=np.asarray(xyz)+correction
                self.events.append(dict(event='measured_relative_landing',fresh_pose=bool(seen.get('pose_valid')),relative_m=np.asarray(relative).tolist(),correction_m=correction.tolist(),target_m=xyz.tolist()))
            else:self.events.append(dict(event='landing_pose_unavailable_original_target'))
        return super().move(xyz,yaw,duration,phase,gap,force,apply_residual,track)

    def run(self):
        r=super().run();r['variant']=self.landing_variant
        r['controller_scope']='bounded preload and observed relative-pose landing control; not RL'
        return r
