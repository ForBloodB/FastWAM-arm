"""Bounded, observed post-probe contact repairs over verified MULTICCD.

Simulation only. Physics, native controller, force caps and scorer are inherited
unchanged. A zero request is the strong B (slow2), not the faster legacy zero.
All actual release, opening, clearance and retry operations remain in the same
episode; no object state reset, teleportation or privileged force feedback.
"""
from copy import deepcopy
import numpy as np
from .converged_contact import ConvergenceEnv
from .contact_state import ContactStateEnv
from .contact_repair import local_offset_to_world
from .boundary_residual import observed_release_context


def spec(offset=(0.,0.,0.), pitch_delta=0., yaw_delta=0.):
    return dict(local_offset_m=list(offset),pitch_delta_rad=float(pitch_delta),
                yaw_delta_rad=float(yaw_delta),lift_scale=2.,hold_s=0.,pitch_rad=0.)


ACTIONS={
    'zero':spec(),
    'recenter':spec(),
    'x_minus4':spec((-.004,0.,0.)),
    'x_plus4':spec((.004,0.,0.)),
    'y_minus3':spec((0.,-.003,0.)),
    'y_plus3':spec((0.,.003,0.)),
    'z_minus2':spec((0.,0.,-.002)),
    'z_plus2':spec((0.,0.,.002)),
    'pitch_minus08':spec(pitch_delta=-.08),
    'pitch_plus08':spec(pitch_delta=.08),
    'pitch_minus16':spec(pitch_delta=-.16),
    'pitch_plus16':spec(pitch_delta=.16),
    'yaw_minus12':spec(yaw_delta=-.12),
    'yaw_plus12':spec(yaw_delta=.12),
    'lower_pitch_plus':spec((0.,0.,-.002),pitch_delta=.08),
    'lower_pitch_minus':spec((0.,0.,-.002),pitch_delta=-.16),
}


def release_mask(x,raw,context,minimum_tracking_m=.005):
    """No object mass/friction/native contact values are actor inputs."""
    x=np.asarray(x,float)
    if x.shape!=(15,) or not np.isfinite(x).all():raise ValueError('Invalid observed features')
    reasons=list(context.get('reasons',[]))
    if not context.get('permitted'):reasons.append('release_context_not_permitted')
    if context.get('observed_bottom_height_m',1.)>.006:reasons.append('above_6mm_release_limit')
    if not raw.get('vision_valid'):reasons.append('vision_invalid')
    if raw.get('vision_label')!=3:reasons.append('not_visual_metal_proxy')
    if max(raw.get('left_force_N',float('inf')),raw.get('right_force_N',float('inf')))>=.8*raw.get('force_budget_N',0.):reasons.append('delayed_force_high')
    if raw.get('tcp_probe_displacement_m',[0,0,0])[2]<minimum_tracking_m:reasons.append('insufficient_actual_probe')
    if not context.get('weak_probe'):reasons.append('already_supported')
    return dict(permitted=not reasons,reasons=reasons,minimum_fk_probe_m=minimum_tracking_m,
                source='rendered RGBD, known table plane, FK/finger encoders and 20ms delayed force')


class ContactActionsEnv(ConvergenceEnv):
    action_specs=ACTIONS
    max_correction_z_m=.005
    minimum_probe_tracking_m=.005
    def __init__(self,rescue_action='zero',rescue_selector=None,**kwargs):
        if rescue_action not in self.action_specs:raise ValueError(rescue_action)
        self.rescue_action=rescue_action;self.rescue_selector=rescue_selector
        self.action_events=[];self._action_attempts=0;self._probe_seen=None
        super().__init__(numerical_mode='multiccd',variant='combo_observed',**kwargs)

    def reset(self,**kwargs):
        self.action_events=[];self._action_attempts=0;self._probe_seen=None
        return super().reset(**kwargs)

    def _action_event(self,event,**extra):
        self.action_events.append(dict(event=event,time_s=float(self.data.time),**extra))

    def probe_features(self,closed,probe,close_tcp,close_opening):
        x,raw=ContactStateEnv.probe_features(self,closed,probe,close_tcp,close_opening)
        if self.repair_first_probe is None:
            self.repair_first_probe=dict(time_s=float(self.data.time),features=x.tolist(),observation=deepcopy(raw))
        if not self._decision_ready:
            self._probe_safety=observed_release_context(probe,self.estimated_size,self.estimate,raw)
            mask=release_mask(x,raw,self._probe_safety,self.minimum_probe_tracking_m)
            if self._saved_descent is None:mask['permitted']=False;mask['reasons'].append('missing_descent')
            allowed=list(self.action_specs) if mask['permitted'] else ['zero']
            actor=deepcopy(raw);actor.update(available_actions=allowed,regrasp_safety=deepcopy(self._probe_safety),repair_mask=deepcopy(mask))
            request=self.rescue_action if self.rescue_selector is None else self.rescue_selector(x.copy(),actor)
            if request not in self.action_specs:raise ValueError('Unknown contact action '+str(request))
            executed=request if request in allowed else 'zero'
            self.residual_decision=dict(time_s=float(self.data.time),features=x.tolist(),observation=deepcopy(raw),
                requested_action=request,selected_action=request,executed_action=executed,
                allowed_actions=allowed,available_actions=allowed,masked=executed!=request,
                specification=deepcopy(self.action_specs[executed]),regrasp_safety=deepcopy(self._probe_safety),
                repair_mask=mask,execution_status='not_started',mask_version='contact_actions_v1')
            self._probe_time=float(self.data.time);self._probe_seen=probe;self._decision_ready=True
        return x,raw

    def _absolute_move(self,target,yaw,duration,phase,gap,force,track=False):
        """Convert measured/world target to inherited pickup-offset coordinates.

        ContactRepairEnv adds its initial geometry offset before FixedResidualEnv
        sees a target. A saved descent ALREADY contains that offset. Subtract the
        two inherited offsets here to avoid applying the initial shift twice.
        """
        from .contact_repair import PICKUP_TARGET_PHASES
        nominal=np.asarray(target,float).copy()-self.offset
        if (self.repair_spec['family'].startswith('initial_') and self._initial_started
                and not self._pickup_cleared and phase in PICKUP_TARGET_PHASES):
            nominal-=self._initial_offset
        return self.move(nominal,yaw,duration,phase,gap,force,track=track)

    def _consume_residual(self,yaw):
        decision=self.residual_decision;name=decision['executed_action']
        if name=='zero':
            decision['execution_status']='fixed_B';return super()._consume_residual(yaw)
        self._residual_applied=True
        if self._action_attempts or self._probe_time is None or self.data.time-self._probe_time>.10:
            raise RuntimeError('contact_action_retry_or_stale_guard')
        if max(self.sensor_force)>=.8*self.force_budget:raise RuntimeError('contact_action_force_guard')
        self._action_attempts+=1;self._inside_residual_macro=True
        try:
            action=self.action_specs[name]
            old_pitch=float(self.residual_pitch)
            base_low,base_yaw=self._saved_descent
            base_physical=np.asarray(base_low,float).copy()+self.offset
            self._action_event('repair_begin',action=name,base_low_m=base_physical.tolist(),
                original_pitch_rad=old_pitch,attempt=self._action_attempts,mask=decision['repair_mask'])
            # Inherited measured-joint hold opens before ANY arm translation.
            self._open_at_measured_joints(yaw)
            self._action_event('open_verified',**self._check_open())
            seen,xyz,delta=self._settled_observation('contact_action_released')
            clear=self.data.site_xpos[self.tcp].copy()
            clear[2]=max(clear[2]+.020,base_physical[2]+.035,xyz[2]+self.estimated_size[2]/2+.015)
            if clear[2]-self.data.site_xpos[self.tcp,2]>.055:raise RuntimeError('contact_action_clearance_guard')
            self._absolute_move(clear,yaw,1.20,'regrasp_clear',.035,12.)
            seen,xyz,delta=self._settled_observation('contact_action_clear')
            observed_delta=np.clip(delta,[-.006,-.006,-.003],[.006,.006,.003])
            correction=observed_delta+local_offset_to_world(action['local_offset_m'],base_yaw)
            if np.linalg.norm(correction[:2])>.012 or abs(correction[2])>self.max_correction_z_m:raise RuntimeError('contact_action_offset_guard')
            self.offset=np.clip(self.offset+correction,[-.012,-.012,-self.max_correction_z_m],[.012,.012,self.max_correction_z_m])
            target_low=np.asarray(base_low,float).copy()+self.offset
            self._pickup_offset=self.offset.copy()
            self.residual_yaw=float(np.clip(seen['yaw']-base_yaw+action['yaw_delta_rad'],-.25,.25))
            self._check_open()
            self.residual_pitch=float(np.clip(old_pitch+action['pitch_delta_rad'],-.16,.24))
            self.intervention['pitch_rad']=self.residual_pitch
            self._action_event('open_pose_correction',observed_delta_m=observed_delta.tolist(),
                commanded_offset_m=self.offset.tolist(),target_low_m=target_low.tolist(),
                pitch_rad=self.residual_pitch,yaw_delta_rad=self.residual_yaw)
            self._absolute_move(clear,base_yaw,1.20,'regrasp_align',.035,12.)
            self._absolute_move(target_low,base_yaw,1.50,'regrasp_descend',.035,12.)
            self._check_open()
            self._absolute_move(target_low,base_yaw,.25,'hold',.035,12.)
            self._absolute_move(target_low,base_yaw,action.get('reclose_duration_s',2.20),'retry_close',0.,self.drive)
            self._absolute_move(target_low,base_yaw,.30,'preload',0.,self.drive)
            closed=self.observe('contact_action_after_close',prior_xy=xyz[:2])
            close_tcp=self.data.site_xpos[self.tcp].copy();opening=float(self.data.qpos[7]+self.data.qpos[8])
            probe=target_low.copy();probe[2]+=.015
            self._absolute_move(probe,base_yaw,.65,'probe_lift',0.,self.drive)
            seen=self.observe('contact_action_probe',prior_xy=xyz[:2])
            new_x,new_raw=ContactStateEnv.probe_features(self,closed,seen,close_tcp,opening)
            self.repair_post_probe=dict(time_s=float(self.data.time),features=new_x.tolist(),observation=deepcopy(new_raw))
            if self._probe_seen is not None:
                for index,record in enumerate(self.vision_calls):
                    if record is self._probe_seen:self.vision_calls[index]=deepcopy(record)
                self._probe_seen.clear();self._probe_seen.update(deepcopy(seen))
            self.follow_relative=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp] if seen.get('valid') else None
            self.slip_anchor=None;self.regrasp_done=True
            decision['execution_status']='macro_complete'
            self._action_event('repair_complete',post_probe=new_raw)
        finally:self._inside_residual_macro=False

    def run(self):
        row=super().run()
        row.update(contact_action=self.rescue_action,action_events=deepcopy(self.action_events),
            contact_action_attempts=self._action_attempts,contact_action_limit=1,
            action_observation_privileged=False,
            action_scope='post-probe measured-open-first real trajectory; strong B zero; unchanged scoring/forces')
        return row
