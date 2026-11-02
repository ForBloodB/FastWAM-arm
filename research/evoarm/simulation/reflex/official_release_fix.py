"""Explicit follow-up controller variant; frozen main experiment stays intact."""
import numpy as np
from .official_grasp import OfficialGraspEnv

class MeasuredReleaseEnv(OfficialGraspEnv):
    def move(self,xyz,yaw,duration,phase,gap,force):
        if phase=='release':
            # During grasp, the commanded gap is zero but contact holds the real
            # fingers apart. Opening from the old zero reference first squeezes
            # the object. Start the opening reference at measured coupler travel.
            # This changes only the command reference, never physical state.
            self.gap=float(np.clip(self.data.qpos[6],0,.05))
        return super().move(xyz,yaw,duration,phase,gap,force)
