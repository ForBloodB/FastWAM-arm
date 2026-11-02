"""Simulation-only bounded drive mapping, acquisition and lift ablations.

The frozen scene/contact model, score, camera and delayed force sensor are kept.
No true mass, friction, object position or native peak force enters control.
"""
from copy import deepcopy
import numpy as np
from .action_rescue import ActionRescueEnv
from .drive_preload_native import roll

VARIANTS=['zero','previous_z','ramped_gap','mapped_preload','mapped_smooth','mapped_guarded']

class DrivePreloadEnv(ActionRescueEnv):
    def __init__(self,variant='zero',**kwargs):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.control_log=[]
        super().__init__('z_minus_2mm' if variant=='previous_z' else 'zero',**kwargs)

    def reset(self,**kwargs):
        self.control_log=[];self.force_mode=False;self.preload_drive=0.;self.touch_s=0.;self.lost_s=0.;self.relief=False
        return super().reset(**kwargs)

    def _guarded_move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        if self.variant in ['zero','previous_z']:
            return super()._guarded_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        xyz=np.asarray(xyz,float).copy()
        if apply_residual:xyz+=self.offset;yaw+=self.residual_yaw
        if not (.22<xyz[0]<.42 and -.18<xyz[1]<.16 and .095<xyz[2]<.40):raise RuntimeError('command_workspace_guard')
        self.command_pitch=self.residual_pitch if apply_residual else 0.
        mapped=self.variant in ['mapped_preload','mapped_smooth','mapped_guarded'];smooth=self.variant in ['mapped_smooth','mapped_guarded']
        active=gap==0 and phase!='release'
        if phase in ['close','retry_close']:duration=max(duration,4.0);self.touch_s=0.
        if mapped and phase=='preload':duration=max(duration,.8)
        if smooth and phase in ['probe_lift','lift','transport_a','transport','lower']:duration*=1.65
        qstart=self.data.qpos[:6].copy()
        target=qstart.copy() if self.in_tracking_tick and phase=='preload' else self.ik(xyz,yaw,self.command)
        self.phase=phase;self.active=active;n=round(duration/self.model.opt.timestep);k=0
        next_track=round(.15/self.model.opt.timestep);touchdown=False
        if not active:self.force_mode=False;self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
        phase_start=self.grip_ref;cap_limit=min(28.,1.8*self.force_budget) if self.force_budget else 28.
        stable=0.;goal=min(28.,1.4*self.force_budget) if self.predicted==3 else min(cap_limit,self.start_drive)
        while k<n:
            count=min(self.sample_steps,n-k);dt=count*self.model.opt.timestep
            previous=self.grip_ref;old_drive=self.preload_drive
            F=self.sensor_force.copy();measured=float(max(F));bilateral=bool(min(F)>.15)
            self.touch_s=self.touch_s+dt if bilateral else 0.
            if active:
                near=self.data.qpos[6]<self.estimated_size[1]/2+.003 or measured>.10*self.force_budget
                close_speed=.003 if near else .018
                if mapped and self.touch_s>=.03 and not self.force_mode:
                    self.force_mode=True;old_drive=self.preload_drive=float(np.clip(-self.data.ctrl[6],0,cap_limit))
                self.lost_s=self.lost_s+dt if min(F)<.08 else 0.
                if self.force_mode and self.lost_s>=.05:
                    self.force_mode=False;self.grip_ref=float(np.clip(self.data.qpos[6],0,.035));previous=self.grip_ref
                if self.force_mode:
                    # Shared motor: target total drive, not per-finger force.
                    requested=goal
                    if measured>.82*self.force_budget:requested=max(1.,goal-3*(measured-.78*self.force_budget))
                    self.preload_drive=float(np.clip(requested,old_drive-80*dt,old_drive+20*dt))
                    self.preload_drive=float(np.clip(self.preload_drive,0,cap_limit));cap=cap_limit
                    self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
                else:
                    nominal=previous-close_speed*dt
                    if measured>.80*self.force_budget:nominal=max(nominal,float(self.data.qpos[6])+.001)
                    self.grip_ref=float(np.clip(nominal,0,.035));cap=min(cap_limit,max(force,self.reference_drive))
                self.drive=cap;self.actual_cap=cap
                if self.variant=='mapped_guarded':
                    if measured>.90*self.force_budget:self.relief=True
                    if measured<.72*self.force_budget:self.relief=False
                    if self.relief:
                        previous=float(np.clip(self.data.qpos[6],0,.035))
                        self.grip_ref=float(np.clip(previous+min(.0015,max(0.,measured-.70*self.force_budget)/1800.),0,.035))
                        self.preload_drive=max(0.,old_drive-80*dt)
            else:
                u=(k+count)/n;s=u*u*(3-2*u);self.grip_ref=float(np.clip(phase_start+s*(gap-phase_start),0,.035));cap=min(cap_limit,force)
            self.control_bounds['min_cap_N']=min(self.control_bounds['min_cap_N'],cap);self.control_bounds['max_cap_N']=max(self.control_bounds['max_cap_N'],cap)
            stats=roll(self.model,self.data,qstart,target,previous,self.grip_ref,cap,k,n,count,active and self.force_mode and not self.relief,old_drive,self.preload_drive,smooth and phase in ['probe_lift','lift','transport_a','transport','lower'],self.og,self.left,self.right,self.table,float(self.params()['force_budget_N_per_finger']))
            self._record(stats,cap)
            self.control_log.append(dict(time_s=float(self.data.time),phase=phase,mode='relief' if active and self.relief else 'force' if active and self.force_mode else 'gap',left_delayed_N=float(F[0]),right_delayed_N=float(F[1]),drive_target_N=self.preload_drive,drive_actual_mean_N=float(-stats[5]),drive_actual_abs_max_N=float(stats[6]),encoder_m=float(self.data.qpos[6]),gap_ref_m=self.grip_ref,goal_total_N=goal))
            self.sample(dt);k+=count
            if phase in ['close','retry_close']:
                ready=self.touch_s>.15 and (min(self.sensor_force)>.40*goal) and (not mapped or abs(self.preload_drive-goal)<1.)
                stable=stable+dt if ready else 0.
                if stable>=.15:break
            if phase in ['lower','retry_lower'] and measured>.85*self.force_budget and np.linalg.norm(self.data.site_xpos[self.tcp]-xyz)<.020:
                self.events.append(dict(event='near_target_touchdown_stop',phase=phase,tcp=self.data.site_xpos[self.tcp].tolist()));touchdown=True;break
            if track and k>=next_track:
                held=self._tracking_tick(phase,yaw);next_track=k+round(.15/self.model.opt.timestep);self.phase=phase;self.active=True
                if held and k<n:qstart=self.data.qpos[:6].copy();n-=k;k=0;next_track=round(.15/self.model.opt.timestep)
        self.command=self.data.qpos[:6].copy() if touchdown else target;self.gap=self.grip_ref;self.active=False
        self.command_updates.append(dict(phase=phase,drive_N=self.drive,cap_N=cap,sensor=self.sensor_observation(),tcp_target=xyz.tolist(),duration_s=duration))

    def _record(self,stats,cap):
        if max(stats[:2])>max(self.fullrate[:2]):self.fullrate_peak_details=dict(phase=self.phase,time_s=float(self.data.time),cap_N=cap,opening_encoder_m=float(self.data.qpos[7]+self.data.qpos[8]))
        self.fullrate[:2]=np.maximum(self.fullrate[:2],stats[:2]);self.fullrate[2]+=stats[2];self.fullrate[3:]=np.maximum(self.fullrate[3:],stats[3:5])
        self.force_chunks.append([float(self.data.time),self.phase,float(stats[0]),float(stats[1]),float(stats[2]),float(stats[3]),float(stats[4]),float(cap),float(self.sensor_force[0]),float(self.sensor_force[1]),float(self.data.qpos[7]+self.data.qpos[8])])
        p=self.phase_forces.setdefault(self.phase,dict(native_peak_N=0.,native_overforce_s=0.,finger_depth_m=0.,table_depth_m=0.,command_chunks=0))
        for key,value in [('native_peak_N',max(stats[:2])),('finger_depth_m',stats[3]),('table_depth_m',stats[4])]:p[key]=max(p[key],float(value))
        p['native_overforce_s']+=float(stats[2]);p['command_chunks']+=1

    def run(self):
        r=super().run();r.update(variant=self.variant,controller_scope='bounded feedforward and preload/lift control; not RL',control_log_rows=len(self.control_log))
        return r
