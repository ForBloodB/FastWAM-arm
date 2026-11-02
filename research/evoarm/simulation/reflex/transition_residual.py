"""Simulation-only smooth transitions and observed touchdown control.

The contact model and terminal scorer remain inherited. Native force truth is
an output accumulator only; control uses encoders/FK, rendered RGBD, and the
existing delayed 100 Hz force feedback. The legacy implementation is retained
as a development-only ablation, never mixed into a learned comparison.
"""
from copy import deepcopy
import numpy as np
from .boundary_residual import BoundaryResidualEnv, ACTIONS, ACTION_NAMES, observed_release_context, available_actions as original_mask
from .contact_state import ContactStateEnv
from .drive_preload_native import roll

VARIANTS = ('legacy', 'phase_guard', 'touchdown', 'all')

def available_actions(x,raw):
    allowed=original_mask(x,raw)
    supported=bool(raw['vision_valid'] and x[6]>=.65 and min(x[10:12])>=.20 and abs(x[12])<=.25 and x[14]<=.6)
    if not supported:allowed=[a for a in allowed if a not in ('hold05_slow2','hold10_slow2')]
    return allowed


class TransitionResidualEnv(BoundaryResidualEnv):
    def __init__(self, action='zero', action_selector=None, controller_variant='all', **kwargs):
        if controller_variant not in VARIANTS:raise ValueError(controller_variant)
        self.controller_variant=controller_variant
        self.transition_log=[];self.landing_feedback=[]
        super().__init__(action,action_selector,**kwargs)

    def reset(self, **kwargs):
        self.transition_log=[];self.landing_feedback=[]
        return super().reset(**kwargs)

    def probe_features(self,closed,probe,close_tcp,close_opening):
        if self.controller_variant in ('legacy','touchdown'):
            return super().probe_features(closed,probe,close_tcp,close_opening)
        x,raw=ContactStateEnv.probe_features(self,closed,probe,close_tcp,close_opening)
        if not self._decision_ready:
            self._probe_safety=observed_release_context(probe,self.estimated_size,self.estimate,raw)
            self._probe_time=float(self.data.time)
            actor_raw=deepcopy(raw);actor_raw['regrasp_safety']=deepcopy(self._probe_safety)
            allowed=available_actions(x,actor_raw);actor_raw['available_actions']=list(allowed)
            requested=self.requested_action if self.action_selector is None else self.action_selector(x.copy(),actor_raw)
            if requested not in ACTIONS:raise ValueError('Unknown transition action: '+str(requested))
            executed=requested if requested in allowed else 'zero'
            self.residual_decision=dict(time_s=float(self.data.time),features=x.tolist(),observation=deepcopy(raw),
                requested_action=requested,selected_action=requested,executed_action=executed,available_actions=list(allowed),allowed_actions=list(allowed),
                masked=requested!=executed,specification=deepcopy(ACTIONS[executed]),regrasp_safety=deepcopy(self._probe_safety),
                execution_status='not_started',mask_version='transition_supported_hold_v1')
            self._decision_ready=True
        return x,raw

    def _guarded_move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        if self.controller_variant=='legacy':
            return super()._guarded_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        xyz=np.asarray(xyz,float).copy()
        if apply_residual:xyz+=self.offset;yaw+=self.residual_yaw
        if not (.22<xyz[0]<.42 and -.18<xyz[1]<.16 and .095<xyz[2]<.40):raise RuntimeError('command_workspace_guard')
        self.command_pitch=self.residual_pitch if apply_residual else 0.
        active=gap==0 and phase!='release'
        if phase in ['close','retry_close']:duration=max(duration,4.);self.touch_s=0.
        if phase=='preload':duration=max(duration,.8)
        smooth_phase=phase in ['probe_lift','lift','transport_a','transport','lower']
        if smooth_phase:duration*=1.65
        slew=self.controller_variant in ('phase_guard','all')
        protected=slew and phase in ('preload','residual_hold','hold','lower')
        touchdown_enabled=self.controller_variant in ('touchdown','all')
        control_scale=.85 if self.predicted==2 and slew else 1.
        if self._soft_relief_active:control_scale*=ACTIONS['soft_relief']['soft_scale']
        budget=self.force_budget*control_scale
        cap_limit=min(28.,1.8*budget)
        qstart=self.data.qpos[:6].copy()
        stationary=active and phase in ('preload','residual_hold','hold') and slew
        target=qstart.copy() if stationary or self.in_tracking_tick and phase=='preload' else self.ik(xyz,yaw,self.command)
        self.phase=phase;self.active=active;n=round(duration/self.model.opt.timestep);k=0
        next_track=round(.15/self.model.opt.timestep);touchdown=False
        if not active:self.force_mode=False;self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
        phase_start=self.grip_ref;stable=0.;last_force=float(max(self.sensor_force))
        base_goal=min(28.,1.4*budget) if self.predicted==3 else min(cap_limit,self.start_drive*control_scale)
        hold_goal=min(base_goal,max(.3,self.preload_drive)) if stationary and self.force_mode else base_goal
        if stationary and phase in ('residual_hold','hold'):
            # A stop preserves the observed contact support, rather than
            # continuing to drive toward the acquisition force target.
            hold_goal=min(base_goal,max(.3,1.03*float(sum(self.sensor_force))))
        self.preload_drive=float(np.clip(self.preload_drive,0,cap_limit))
        relative=None;next_vision=0
        if phase=='lower' and touchdown_enabled:
            seen=self.observe('transition_landing_entry',prior_xy=self.data.site_xpos[self.tcp,:2])
            if seen.get('pose_valid'):
                relative=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp]
                self.landing_feedback.append(dict(event='observed_landing_entry',time_s=float(self.data.time),relative_m=relative.tolist(),estimated_size_m=self.estimated_size.tolist()))
        if self._soft_relief_active:
            self.soft_control_log.append(dict(phase=phase,time_s=float(self.data.time),original_predicted_budget_N=float(self.force_budget),control_budget_N=float(budget),control_cap_N=cap_limit,scale=control_scale))
        while k<n:
            count=min(self.sample_steps,n-k);dt=count*self.model.opt.timestep
            previous=self.grip_ref;old_drive=self.preload_drive
            F=self.sensor_force.copy();measured=float(max(F));bilateral=bool(min(F)>.15)
            predicted=measured+.020*max(0.,(measured-last_force)/max(dt,1e-6)) if protected else measured
            predicted=min(predicted,2*budget);last_force=measured
            self.touch_s=self.touch_s+dt if bilateral else 0.
            if phase=='lower' and touchdown_enabled and relative is not None:
                if k>=next_vision:
                    seen=self.observe('transition_landing_follow',prior_xy=self.data.site_xpos[self.tcp,:2]+relative[:2])
                    if seen.get('pose_valid'):
                        new=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp]
                        if np.linalg.norm(new-relative)<.015:relative=new
                    next_vision=k+round(.10/self.model.opt.timestep)
                bottom=float(self.data.site_xpos[self.tcp,2]+relative[2]-self.estimated_size[2]/2-.100)
                self.landing_feedback.append(dict(event='encoder_projected_bottom',time_s=float(self.data.time),bottom_m=bottom,relative_m=relative.tolist()))
                if k*self.model.opt.timestep>=.05 and bottom<=.006 and np.linalg.norm(self.data.site_xpos[self.tcp]-xyz)<.025:
                    self.events.append(dict(event='observed_table_touchdown_stop',time_s=float(self.data.time),observed_bottom_m=bottom,phase=phase))
                    touchdown=True;break
            if active:
                near=self.data.qpos[6]<self.estimated_size[1]/2+.003 or measured>.10*budget
                close_speed=(.0015 if near else .018) if protected else (.003 if near else .018)
                if self.touch_s>=.03 and not self.force_mode:
                    self.force_mode=True;old_drive=self.preload_drive=float(np.clip(-self.data.ctrl[6],0,cap_limit))
                self.lost_s=self.lost_s+dt if min(F)<.08 else 0.
                if self.force_mode and self.lost_s>=(.12 if protected else .05):
                    self.force_mode=False;self.grip_ref=float(np.clip(self.data.qpos[6],0,.035));previous=self.grip_ref
                if self.force_mode:
                    requested=hold_goal
                    threshold=.75 if protected else .82
                    if predicted>threshold*budget:
                        requested=max(.3 if protected else 1.,hold_goal-(6. if protected else 3.)*(predicted-(.70 if protected else .78)*budget))
                    self.preload_drive=float(np.clip(requested,old_drive-80*dt,old_drive+(12 if protected else 20)*dt))
                    self.preload_drive=float(np.clip(self.preload_drive,0,cap_limit));cap=cap_limit
                    self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
                else:
                    nominal=previous-close_speed*dt
                    if predicted>(.75 if protected else .80)*budget:nominal=max(nominal,float(self.data.qpos[6])+(.00015 if protected else .001))
                    self.grip_ref=float(np.clip(nominal,0,.035));cap=min(cap_limit,max(force,self.reference_drive*control_scale))
                self.drive=cap;self.actual_cap=cap
            else:
                u=(k+count)/n;s=u*u*(3-2*u);self.grip_ref=float(np.clip(phase_start+s*(gap-phase_start),0,.035));cap=min(cap_limit,force)
            self.control_bounds['min_cap_N']=min(self.control_bounds['min_cap_N'],cap)
            self.control_bounds['max_cap_N']=max(self.control_bounds['max_cap_N'],cap)
            # Preserve the calibrated fast PD/encoder damping loop. Only the
            # reference trajectory, contact acquisition logic and force target
            # are shaped; limiting the complete damping output was unstable.
            stats=roll(self.model,self.data,qstart,target,previous,self.grip_ref,cap,k,n,count,active and self.force_mode,old_drive,self.preload_drive,smooth_phase,self.og,self.left,self.right,self.table,float(self.params()['force_budget_N_per_finger']))
            self._record(stats,cap)
            self.control_log.append(dict(time_s=float(self.data.time),phase=phase,mode='force' if active and self.force_mode else 'gap',left_delayed_N=float(F[0]),right_delayed_N=float(F[1]),drive_target_N=self.preload_drive,drive_actual_mean_N=float(-stats[5]),drive_actual_abs_max_N=float(stats[6]),encoder_m=float(self.data.qpos[6]),gap_ref_m=self.grip_ref,goal_total_N=hold_goal))
            self.transition_log.append(dict(time_s=float(self.data.time),phase=phase,stationary=stationary,control_budget_N=float(budget),predicted_delayed_force_N=predicted,native_step_s=float(self.model.opt.timestep),reference_guard_enabled=protected,native_feedback_unchanged=True,goal_total_N=hold_goal))
            self.sample(dt);k+=count
            if phase in ('close','retry_close'):
                ready=self.touch_s>.15 and min(self.sensor_force)>.40*hold_goal and abs(self.preload_drive-hold_goal)<1.
                stable=stable+dt if ready else 0.
                if stable>=.15:break
            if phase=='lower' and measured>.85*budget and np.linalg.norm(self.data.site_xpos[self.tcp]-xyz)<.020:
                self.events.append(dict(event='near_target_touchdown_stop',phase=phase,tcp=self.data.site_xpos[self.tcp].tolist()));touchdown=True;break
            if track and k>=next_track:
                held=self._tracking_tick(phase,yaw);next_track=k+round(.15/self.model.opt.timestep);self.phase=phase;self.active=True
                if held and k<n:
                    qstart=self.data.qpos[:6].copy();n-=k;k=0;next_track=round(.15/self.model.opt.timestep)
        self.command=self.data.qpos[:6].copy() if touchdown else target
        self.gap=self.grip_ref;self.active=False
        self.command_updates.append(dict(phase=phase,drive_N=self.drive,cap_N=cap_limit,sensor=self.sensor_observation(),tcp_target=xyz.tolist(),duration_s=duration))

    def run(self):
        row=super().run()
        row.update(fixed_controller=f'TransitionResidualEnv({self.controller_variant})',transition_controller=self.controller_variant,transition_log=deepcopy(self.transition_log),landing_feedback=deepcopy(self.landing_feedback),zero_reference='TransitionResidualEnv(action=zero): shared repaired control',controller_inputs='encoders/FK, rendered RGBD and existing delayed force only')
        return row
