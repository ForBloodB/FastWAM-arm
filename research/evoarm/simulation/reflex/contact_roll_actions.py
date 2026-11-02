"""Closing-axis tilt as a bounded pose repair; same official model and caps.

Pitch about the finger closing axis cannot rotate that axis toward gravity.
Here a yaw-aligned WORLD-x tilt changes the closing direction, with a small
upward pickup correction to keep the open lower finger clear of the table.
The virtual trajectory, all collisions and native force peaks are evaluated.
The side-pinch traction formula is a diagnostic motivation, not actor input.
"""
import numpy as np
import mujoco
from .contact_actions import ContactActionsEnv,spec
from .official_grasp import orientation,rotation_error


def tilt(angle,z):
    return dict(**spec((0.,0.,z)),closing_tilt_rad=angle,reclose_duration_s=8.,
                post_pose_drive_cap_N=27.6 if abs(angle)<=.5 else 27.2 if abs(angle)<=.55 else 26.8)


ACTIONS={'zero':spec(),'recenter':spec(),
    'tilt_pos50_up6':tilt(.50,.006),'tilt_neg50_up6':tilt(-.50,.006),
    'tilt_pos60_up10':tilt(.60,.010),'tilt_neg60_up10':tilt(-.60,.010),
    'tilt_neg60_up6':tilt(-.60,.006),'tilt_neg60_up4':tilt(-.60,.004),
    'tilt_neg55_up6':tilt(-.55,.006)}


def tilt_orientation(yaw,pitch,tilt):
    """Rz(yaw) Rx(tilt) Ry(1+pitch); tilt=0 retains original pose."""
    c,s=np.cos(yaw),np.sin(yaw)
    rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    c,s=np.cos(tilt),np.sin(tilt)
    rx=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    c,s=np.cos(pitch),np.sin(pitch)
    ry=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    return rz@rx@rz.T@orientation(yaw)@ry


class ContactRollActionsEnv(ContactActionsEnv):
    action_specs=ACTIONS
    max_correction_z_m=.012
    # At least 60% of the requested 15 mm probe must be realized before this
    # larger pose correction. FK is measured feedback, not a payload label.
    minimum_probe_tracking_m=.009
    def __init__(self,**kwargs):
        self.closing_tilt_rad=0.
        self.pose_extent_estimate=None
        self.force_cap_log=[]
        super().__init__(**kwargs)

    def reset(self,**kwargs):
        self.closing_tilt_rad=0.
        self.pose_extent_estimate=None
        self.force_cap_log=[]
        return super().reset(**kwargs)

    def _action_event(self,event,**extra):
        if event=='open_pose_correction':
            # This event is reached only AFTER stationary opening, open encoder
            # verification, clearance and a second open check in the macro.
            self.closing_tilt_rad=float(self.action_specs[self.residual_decision['executed_action']].get('closing_tilt_rad',0.))
            if abs(self.closing_tilt_rad)>.60+1e-12:raise RuntimeError('closing_tilt_bound')
            extra['closing_tilt_rad']=self.closing_tilt_rad
            # Existing RGBD localization is top surface minus a fixed height
            # prior. A tilted box needs a larger height prior during transport
            # and landing. This conservative box projection uses the INITIAL
            # visual dimensions and commanded pose only, never simulated pose.
            # h + w*|sin(theta)| >= h*cos(theta)+w*|sin(theta)|.
            if self.closing_tilt_rad:
                original=self.estimated_size.copy()
                height=float(original[2]+original[1]*abs(np.sin(self.closing_tilt_rad)))
                if not original[2]<=height<.07:raise RuntimeError('pose_extent_estimate_guard')
                self.estimated_size=self.estimated_size.copy();self.estimated_size[2]=height
                self.pose_extent_estimate=dict(initial_visual_size_m=original.tolist(),
                    closing_tilt_rad=self.closing_tilt_rad,height_prior_m=height,
                    formula='h_initial + width_initial * abs(sin(commanded_closing_tilt))',
                    scope='Conservative rigid-box grasp prior, not measured 6D object orientation')
                extra['pose_extent_estimate']=self.pose_extent_estimate
        return super()._action_event(event,**extra)

    def _clocked_move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        if self.closing_tilt_rad and phase=='retry_close':
            # A tilted approach meets a box corner before the original width
            # threshold. Start the inherited near-contact closing speed early;
            # do not alter the native velocity gain or contact parameters.
            old=float(self.estimated_size[1])
            initial=self.pose_extent_estimate['initial_visual_size_m']
            self.estimated_size[1]=float(min(.068,initial[1]+initial[2]*abs(np.sin(self.closing_tilt_rad))))
            try:return super()._clocked_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
            finally:self.estimated_size[1]=old
        return super()._clocked_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)

    def _timed_roll(self,implementation,*args):
        supplied=list(args)
        cap=float(supplied[6]);requested=cap
        if self.closing_tilt_rad:
            action=self.action_specs[self.residual_decision['executed_action']]
            cap=min(cap,action['post_pose_drive_cap_N'])
            supplied[6]=cap;supplied[11]=min(float(supplied[11]),cap);supplied[12]=min(float(supplied[12]),cap)
        self.force_cap_log.append(dict(time_s=float(self.data.time),phase=self.phase,
            requested_cap_N=requested,applied_cap_N=cap))
        return super()._timed_roll(implementation,*supplied)

    def ik(self,xyz,yaw,seed):
        if not self.closing_tilt_rad or self._joint_hold_q is not None:
            return super().ik(xyz,yaw,seed)
        q=np.clip(np.asarray(seed,float).copy(),self.ranges[:,0]+1e-5,self.ranges[:,1]-1e-5)
        target=tilt_orientation(yaw,self.command_pitch,self.closing_tilt_rad)
        jp=np.zeros((3,self.model.nv));jr=jp.copy()
        for _ in range(100):
            self.ikdata.qpos[:6]=q;mujoco.mj_forward(self.model,self.ikdata)
            ep=np.asarray(xyz)-self.ikdata.site_xpos[self.tcp]
            er=rotation_error(target,self.ikdata.site_xmat[self.tcp].reshape(3,3))
            if np.linalg.norm(ep)<.00008 and np.linalg.norm(er)<.001:break
            mujoco.mj_jacSite(self.model,self.ikdata,jp,jr,self.tcp)
            j=np.vstack([jp[:,:6],.15*jr[:,:6]])
            dq=j.T@np.linalg.solve(j@j.T+.00002*np.eye(6),np.r_[ep,.15*er])
            q=np.clip(q+np.clip(dq,-.12,.12),self.ranges[:,0]+1e-5,self.ranges[:,1]-1e-5)
        self.ikdata.qpos[:6]=q;mujoco.mj_forward(self.model,self.ikdata)
        ep=np.linalg.norm(np.asarray(xyz)-self.ikdata.site_xpos[self.tcp])
        er=np.linalg.norm(rotation_error(target,self.ikdata.site_xmat[self.tcp].reshape(3,3)))
        if ep>.002 or er>.02:raise RuntimeError(f'ik_unreachable:{ep:.5f}:{er:.5f}')
        return q

    def run(self):
        row=super().run();row['closing_tilt_rad']=self.closing_tilt_rad
        row['pose_extent_estimate']=self.pose_extent_estimate
        row['pose_actuation_caps']=dict(windows=len(self.force_cap_log),
            original_limit_N=28.,maximum_applied_N=max((r['applied_cap_N'] for r in self.force_cap_log),default=0.),
            capped_windows=sum(r['applied_cap_N']<r['requested_cap_N'] for r in self.force_cap_log))
        row['pose_repair_family']='yaw-aligned-world-x closing-axis tilt, activation only after verified opening'
        return row
