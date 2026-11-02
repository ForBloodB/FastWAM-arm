"""Timestamped simulation feedback and evaluation-only lift diagnosis.

Legacy mode preserves TransitionResidualEnv physical operations exactly. Clocked
mode uses the same dynamics, native motor controller, caps and terminal scorer;
its interventions are a 10 ms phase/control clock and capture-time force delay.
No hardware interface is imported. Ground-truth geometry/contact fields below
are recorded only by _diagnose(), never used by the control method.
"""
from collections import deque
from copy import deepcopy
import math
import numpy as np
from . import transition_residual as parent
from .drive_preload_native import roll as native_roll
from .official_grasp import OfficialGraspEnv
from .boundary_residual import ACTIONS


class TimedContactEnv(parent.TransitionResidualEnv):
    SAMPLE_PERIOD_S = .010
    FORCE_DELAY_S = .020

    def __init__(self, timing='legacy', **kwargs):
        if timing not in ('legacy', 'clocked'):
            raise ValueError(timing)
        if timing=='clocked' and kwargs.get('controller_variant','all')=='legacy':
            raise ValueError('Clocked variant requires the repaired transition controller')
        self.timing = timing
        self._clear_timing()
        super().__init__(**kwargs)
        ratio=self.SAMPLE_PERIOD_S/self.model.opt.timestep
        if timing=='clocked' and abs(ratio-round(ratio))>1e-8:
            raise ValueError('Native timestep must divide the 10 ms sensor period')

    def _clear_timing(self):
        self.timing_log=[];self.diagnostic_log=[]
        self._force_pending=deque();self._legacy_captures=deque(maxlen=3)
        self._force_packet=None;self._capture_seq=0;self._timing_depth=0
        self._clock_origin=None;self._last_capture_time=None

    def reset(self, **kwargs):
        self._clear_timing()
        result=super().reset(**kwargs)
        self._clock_origin=float(self.data.time)
        return result

    @classmethod
    def quantize_duration(cls, duration):
        return max(cls.SAMPLE_PERIOD_S, math.ceil(float(duration)/cls.SAMPLE_PERIOD_S-1e-9)*cls.SAMPLE_PERIOD_S)

    def _capture_clocked(self, now):
        """Called at a physical sampling event; never exposes future data."""
        expected=self._clock_origin+(self._capture_seq+1)*self.SAMPLE_PERIOD_S
        if abs(now-expected)>2e-7:
            raise RuntimeError('force_capture_clock_violation')
        self._capture_seq+=1
        measured=np.maximum(0,self.lastforces+self.sensor_rng.normal(0,.02,2))
        packet=dict(sequence=self._capture_seq,capture_time_s=float(now),
                    available_time_s=float(now+self.FORCE_DELAY_S),force_N=measured.tolist())
        self.sensor_queue.append(measured.copy())
        self._force_pending.append(packet)
        while self._force_pending and self._force_pending[0]['available_time_s']<=now+2e-8:
            self._force_packet=self._force_pending.popleft()
        if self._force_packet is not None:
            self.sensor_force=np.asarray(self._force_packet['force_N'],float).copy()
        self._last_capture_time=now
        self.timing_log.append(dict(event='capture',phase=self.phase,**packet))

    def _capture_legacy_metadata(self, now):
        self._capture_seq+=1
        packet=dict(sequence=self._capture_seq,capture_time_s=float(now),
                    nominal_available_time_s=float(now+self.FORCE_DELAY_S),
                    force_N=self.sensor_queue[-1].tolist())
        self._legacy_captures.append(packet)
        if len(self._legacy_captures)==3:
            self._force_packet={**self._legacy_captures[0], 'available_time_s':float(now)}
        self._last_capture_time=now
        self.timing_log.append(dict(event='capture',phase=self.phase,**packet))

    def sample(self, dt):
        if self.timing=='legacy':
            try:
                super().sample(dt)
                self._capture_legacy_metadata(float(self.data.time))
            finally:
                self._diagnose(dt)
            return
        # Same evaluator and sensor noise law as FeedbackGraspEnv/ContactStateEnv.
        # The timestamp queue replaces their sample-count delay before guards run.
        try:
            OfficialGraspEnv.sample(self,dt)
            if self.phase=='transport_a':
                self.metrics['slip'] |= bool(self.metrics['lifted'] and self.data.xpos[self.ob,2]-self.initial[2]<.025)
            self._capture_clocked(float(self.data.time))
        finally:
            for contact in self.data.contact:
                if self.og not in (contact.geom1,contact.geom2):
                    continue
                other=contact.geom2 if contact.geom1==self.og else contact.geom1
                body=int(self.model.geom_bodyid[other])
                if body in (self.left,self.right):
                    self.max_penetration['finger_m']=max(self.max_penetration['finger_m'],-float(contact.dist))
                elif self.model.body(body).name=='task_table':
                    self.max_penetration['table_m']=max(self.max_penetration['table_m'],-float(contact.dist))
                kind='finger' if body in (self.left,self.right) else 'table' if self.model.body(body).name=='task_table' else 'other'
                self.contact_parameters_seen[kind]={'condim':int(contact.dim),'solref':contact.solref.tolist(),'friction':contact.friction.tolist()}
            self._diagnose(dt)
        if self.active and self.force_budget:
            self.over_sensor_s=self.over_sensor_s+dt if max(self.sensor_force)>1.5*self.force_budget else 0.
            if self.over_sensor_s>=.02:
                raise RuntimeError('delayed_force_peak_guard')
        else:
            self.over_sensor_s=0.

    def _diagnose(self, dt):
        """Privileged fields are evaluation outputs, never control observations."""
        contacts=[]
        for contact in self.data.contact:
            if self.og not in (contact.geom1,contact.geom2):
                continue
            other=contact.geom2 if contact.geom1==self.og else contact.geom1
            contacts.append(dict(other_geom=self.model.geom(other).name,
                other_body=self.model.body(int(self.model.geom_bodyid[other])).name,
                position_m=contact.pos.tolist(),distance_m=float(contact.dist)))
        tcp=self.data.site_xpos[self.tcp].copy();obj=self.data.xpos[self.ob].copy()
        self.diagnostic_log.append(dict(time_s=float(self.data.time),phase=self.phase,window_s=float(dt),
            object_xyz_m=obj.tolist(),object_velocity_m_s=self.data.qvel[self.od:self.od+3].tolist(),
            tcp_xyz_m=tcp.tolist(),object_relative_tcp_m=(obj-tcp).tolist(),
            object_height_gain_m=float(obj[2]-self.initial[2]),
            encoder_half_gap_m=float(self.data.qpos[6]),finger_opening_m=float(self.data.qpos[7]+self.data.qpos[8]),
            native_endpoint_force_N=self.lastforces.tolist(),delayed_force_N=self.sensor_force.tolist(),
            drive_ctrl_N=float(self.data.ctrl[6]),force_mode=bool(self.force_mode),
            contacts=contacts,evaluation_only=True))

    def _timed_roll(self, implementation, *args):
        now=float(self.data.time);packet=self._force_packet
        self.timing_log.append(dict(event='control_read',phase=self.phase,read_time_s=now,
            capture_time_s=None if packet is None else packet['capture_time_s'],
            available_time_s=None if packet is None else packet['available_time_s'],
            age_s=None if packet is None else now-packet['capture_time_s'],
            sequence=None if packet is None else packet['sequence'],
            force_N=self.sensor_force.tolist(),native_steps=int(args[9]),
            planned_window_s=float(args[9]*self.model.opt.timestep),valid=packet is not None))
        if self.timing=='clocked' and packet is not None and packet['available_time_s']>now+2e-8:
            raise RuntimeError('future_force_sample_exposed')
        return implementation(*args)

    def _guarded_move(self, xyz, yaw, duration, phase, gap, force, apply_residual=True, track=False):
        if self.timing=='clocked':
            return self._clocked_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        if self._timing_depth:
            return super()._guarded_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        original=parent.roll
        parent.roll=lambda *args:self._timed_roll(original,*args)
        self._timing_depth+=1
        try:
            return super()._guarded_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        finally:
            parent.roll=original
            self._timing_depth-=1

    def _clocked_move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
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
        requested_duration = float(duration)
        duration = self.quantize_duration(duration)
        self.phase=phase;self.active=active;n=round(duration/self.model.opt.timestep);k=0
        self.timing_log.append(dict(event='phase_clock', phase=phase, read_time_s=float(self.data.time), requested_duration_s=requested_duration, scheduled_duration_s=duration, period_s=self.SAMPLE_PERIOD_S))
        next_track=round(.15/self.model.opt.timestep);touchdown=False
        if not active:self.force_mode=False;self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
        phase_start=self.grip_ref;stable=0.;last_force=float(max(self.sensor_force))
        last_capture = None if self._force_packet is None else self._force_packet['capture_time_s']
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
            capture = None if self._force_packet is None else self._force_packet['capture_time_s']
            observation_dt = 0. if capture is None or last_capture is None else capture-last_capture
            slope = (measured-last_force)/observation_dt if observation_dt>1e-9 else 0.
            predicted=measured+self.FORCE_DELAY_S*max(0.,slope) if protected else measured
            predicted=min(predicted,2*budget);last_force=measured;last_capture=capture
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
            stats=self._timed_roll(native_roll,self.model,self.data,qstart,target,previous,self.grip_ref,cap,k,n,count,active and self.force_mode,old_drive,self.preload_drive,smooth_phase,self.og,self.left,self.right,self.table,float(self.params()['force_budget_N_per_finger']))
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
        row.update(timing_variant=self.timing,timing_log=deepcopy(self.timing_log),
            diagnostic_log=deepcopy(self.diagnostic_log),
            timing_scope='10 ms phase quantization and timestamped 20 ms simulated force delivery; unchanged native motor loop and scorer',
            diagnosis_scope='Native geometry, contact and force are evaluation-only outputs')
        return row
