"""Depth-hardening contact and delayed-force admittance, simulation only.

Vendor geometry, density/friction distributions and the 28 N motor cap are
unchanged. Contact regularization is not a real foam deformation model.
"""
from copy import deepcopy
import time
import mujoco,numpy as np
from .contact_state import ContactStateEnv,StatePolicy,FEATURE_NAMES
from .official_grasp import MATERIALS,FORCE_BUDGET,DESTINATION,orientation
from .safe_contact_native import roll

OPTIONS=[
 {'name':'zero','slow':1.,'preload_s':0.,'guard':False,'regrasp':False,'gentle':1.,'balance':False,'pitch':0.,'traction':1.},
 {'name':'traction_hold','slow':1.6,'preload_s':.3,'guard':True,'regrasp':False,'gentle':1.,'balance':False,'pitch':0.,'traction':1.08},
 {'name':'visual_recenter','slow':1.5,'preload_s':.25,'guard':True,'regrasp':True,'gentle':1.,'balance':False,'pitch':0.,'traction':1.},
 {'name':'gentle_hold','slow':1.3,'preload_s':.2,'guard':True,'regrasp':False,'gentle':.8,'balance':False,'pitch':0.,'traction':1.},
 {'name':'positive_pitch_reseat','slow':1.15,'preload_s':.15,'guard':True,'regrasp':True,'gentle':1.,'balance':False,'pitch':.10,'traction':1.},
 {'name':'negative_pitch_reseat','slow':1.15,'preload_s':.15,'guard':True,'regrasp':True,'gentle':1.,'balance':False,'pitch':-.10,'traction':1.},
]

def contact_parameters(requested_tc,material):
    # A bounded numerical relaxation interval replaces the old 20--35 ms
    # softness proxy. Impedance stiffens at 1 mm depth; no pose clipping.
    tc=.003+.003*np.clip((requested_tc-.004)/.031,0,1)
    if material=='sponge':tc=max(.004,tc)
    return [float(tc),1.],[.98,.9999,.001,.5,2.]

def reward(row):
    peak=max(row['contact_metrics']['max_force_N'],row['fullrate']['peak_force_N'])
    excess=max(0.,peak/row['physics']['force_budget_N_per_finger']-1.)
    return float(2*row['success']-6*row['overclamp']-4*row['safety_abort']-.4*row['contact_metrics']['slip']-.1*min(excess,3.)-.015*max(0.,row['duration_s']-10.))

def allowed_options(features,configured):
    x=np.asarray(features);risk=bool(x[13] or x[7]>.3 or abs(x[9])>.2 or abs(x[12])>.25 or max(x[10:12])>.8)
    if not risk:return [0],False
    permitted=[0,3] if max(x[10:12])>.8 else [0,2] if x[13] else [0,2,3,4,5]
    return [a for a in configured if a in permitted],True

class SafeContactEnv(ContactStateEnv):
    def __init__(self,policy=None,learning=False,fixed_action=None,record=False,render_mode=None,dt=.001,controller='guarded_gap',contact='hardening'):
        self.controller_kind=controller;self.contact_kind=contact
        super().__init__(policy=policy or StatePolicy(actions=[0,2,3,4,5]),learning=learning,fixed_action=fixed_action,profile='audited',record=record,render_mode=render_mode,dt=dt)
        self.table=self.model.body('task_table').id

    def reset(self,*,seed=None,options=None):
        self.fullrate=np.zeros(5);self.fullrate_peak_details={};self.slip_anchor=None;self.in_tracking_tick=False;self.admittance_log=[];self.grip_ref=.035;self.last_feedback_force=0.;self.filtered_force=0.;self.force_slope=0.;self.close_stable_s=0.
        obs,info=super().reset(seed=seed,options=options)
        if self.contact_kind in ['hardening','perturbed']:
            solref,solimp=contact_parameters(self.soft,MATERIALS[self.mat])
            if self.contact_kind=='perturbed':solref[0]*=1.2;solimp=[.97,.99985,.0012,.5,2.]
            self.model.pair_solref[self.finger_pair_ids]=solref;self.model.pair_solimp[self.finger_pair_ids]=solimp
        elif self.contact_kind!='previous':raise ValueError(self.contact_kind)
        mujoco.mj_forward(self.model,self.data)
        self.grip_ref=float(np.clip(self.data.qpos[6],0,.035));self.option=OPTIONS[0]
        self.force_goal=float(min(.65*self.force_budget,max(.3*self.force_budget,.5*self.start_drive))) if self.force_budget else 0.
        return obs,info

    def params(self):
        p=super().params();p.update(contact_model=self.contact_kind,controller=self.controller_kind,
          requested_old_solref_s=self.soft,effective_finger_solref=self.model.pair_solref[self.finger_pair_ids[0]].tolist(),
          effective_finger_solimp=self.model.pair_solimp[self.finger_pair_ids[0]].tolist(),
          numerical_contact_scope='bounded rigid-contact regularization, not measured material compression')
        return p

    def _raw_roll(self,a,b,ga,gb,cap,start,total,n):
        stats=roll(self.model,self.data,a,b,ga,gb,cap,start,total,n,self.og,self.left,self.right,self.table,float(FORCE_BUDGET[self.mat]))
        if max(stats[:2])>max(self.fullrate[:2]):self.fullrate_peak_details={'phase':self.phase,'time_s':float(self.data.time),'cap_N':cap,'opening_encoder_m':float(self.data.qpos[7]+self.data.qpos[8])}
        self.fullrate[:2]=np.maximum(self.fullrate[:2],stats[:2]);self.fullrate[2]+=stats[2];self.fullrate[3:]=np.maximum(self.fullrate[3:],stats[3:])

    def _tracking_tick(self,phase,yaw):
        self.in_tracking_tick=True
        try:
            self.track_index+=1;reference=self.follow_relative
            seen,now=self._observe_follow(f'{phase}_track_{self.track_index}',reference)
            if now is None or reference is None:return False
            if not hasattr(self,'slip_anchor') or self.slip_anchor is None:self.slip_anchor=reference.copy();self.slip_opening=float(self.data.qpos[7]+self.data.qpos[8])
            delta=now-self.slip_anchor;opening=float(self.data.qpos[7]+self.data.qpos[8]);F=self.sensor_force
            visual_drop=bool(delta[2]<-.003 or np.linalg.norm(delta[:2])>.006)
            corroborated=bool(min(F)<.15*self.force_budget or abs((F[0]-F[1])/(sum(F)+.2))>.35 or opening<self.slip_opening-.001 or delta[2]<-.008)
            self.slip_signals=self.slip_signals+1 if visual_drop and corroborated else 0
            if visual_drop:self.events.append({'event':'relative_visual_drop','phase':phase,'delta_m':delta.tolist(),'corroborated':corroborated})
            held=False
            if self.option['guard'] and self.slip_signals>=2 and self.guard_holds<2:
                tcp=self.data.site_xpos[self.tcp].copy()-self.offset
                self.move(tcp,yaw-self.residual_yaw,.25,'preload',0,self.drive);self.guard_holds+=1;held=True
                self.events.append({'event':'corroborated_stop_joint_hold','phase':phase})
                seen,now=self._observe_follow(f'{phase}_track_{self.track_index}_held',reference);self.slip_signals=0
                if now is not None:self.slip_anchor=now.copy();self.slip_opening=float(self.data.qpos[7]+self.data.qpos[8])
            if now is not None:self.follow_relative=now
            return held
        finally:self.in_tracking_tick=False

    def move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        if self.controller_kind=='legacy_gap':return super().move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        if self.controller_kind=='guarded_gap':return self._guarded_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)
        if self.controller_kind!='admittance':raise ValueError(self.controller_kind)
        xyz=np.asarray(xyz,float).copy()
        if apply_residual:
            xyz+=self.offset;yaw+=self.residual_yaw
            self.control_bounds['max_offset_m']=max(self.control_bounds['max_offset_m'],float(np.linalg.norm(self.offset)))
        if not (.22<xyz[0]<.42 and -.18<xyz[1]<.16 and .095<xyz[2]<.40):raise RuntimeError('command_workspace_guard')
        self.command_pitch=self.residual_pitch if apply_residual else 0.
        if phase=='release':self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
        if phase in ['close','retry_close']:duration=max(duration,2.2);self.close_stable_s=0.
        if phase in ['lift','transport_a','transport','lower']:duration*=1.25
        target=self.data.qpos[:6].copy() if self.in_tracking_tick and phase=='preload' else self.ik(xyz,yaw,self.command)
        self.phase=phase;n=round(duration/self.model.opt.timestep)
        self.active=gap==0 and phase not in ['release','regrasp_clear','regrasp_align','regrasp_descend']
        qstart=self.data.qpos[:6].copy();k=0;next_track=round(.15/self.model.opt.timestep);open_start=self.grip_ref
        cap_limit=min(28.,1.8*self.force_budget) if self.force_budget else 28.;cap=min(cap_limit,force)
        while k<n:
            count=min(self.sample_steps,n-k);dt=count*self.model.opt.timestep;previous=self.grip_ref
            if self.active:
                measured=float(max(self.sensor_force));slope=(measured-self.last_feedback_force)/dt
                self.force_slope=.5*self.force_slope+.5*slope;self.last_feedback_force=measured
                predicted=measured+min(max(0.,self.force_slope)*.025,.3*self.force_budget)
                ratio=self.reference_drive/max(self.start_drive,.1)
                goal=float(np.clip(self.force_goal*ratio*self.option.get('traction',1.),.2*self.force_budget,.75*self.force_budget))
                near=float(self.data.qpos[6])<float(self.estimated_size[1]/2+.004)
                closing_limit=.003 if near or measured>.1*self.force_budget else .020
                velocity=float(np.clip(.002*(predicted-goal),-closing_limit,.010))
                if measured>.8*self.force_budget:velocity=max(velocity,.006)
                self.grip_ref=float(np.clip(previous+velocity*dt,0,.035))
                cap=float(min(cap_limit,max(1.5,2*goal+.15*self.force_budget)))
                self.drive=cap;self.actual_cap=cap;self.force_changes+=int(velocity>0);self.force_recoveries+=int(velocity<0)
                self.admittance_log.append({'phase':phase,'time_s':float(self.data.time),'force_N':measured,'predicted_force_N':predicted,
                  'goal_N':goal,'gap_ref_m':self.grip_ref,'velocity_m_s':velocity,'cap_N':cap})
            else:
                u=(k+count)/n;s=u*u*(3-2*u);self.grip_ref=float(open_start+s*(gap-open_start));cap=float(min(cap_limit,force))
            self.control_bounds['min_cap_N']=min(self.control_bounds['min_cap_N'],cap);self.control_bounds['max_cap_N']=max(self.control_bounds['max_cap_N'],cap)
            self._raw_roll(qstart,target,previous,self.grip_ref,cap,k,n,count);self.sample(dt);k+=count
            if phase in ['close','retry_close']:
                stable=min(self.sensor_force)>.5*goal and max(self.sensor_force)>.8*goal and abs(self.data.qvel[6])<.001
                self.close_stable_s=self.close_stable_s+dt if stable else 0.
                if self.close_stable_s>=.12:break
            if track and k>=next_track:
                held=self._tracking_tick(phase,yaw);next_track=k+round(.15/self.model.opt.timestep);self.phase=phase;self.active=True
                if held and k<n:qstart=self.data.qpos[:6].copy();n-=k;k=0;next_track=round(.15/self.model.opt.timestep)
        self.command=target;self.gap=self.grip_ref;self.active=False
        self.command_updates.append({'phase':phase,'drive_N':self.drive,'cap_N':cap,'sensor':self.sensor_observation(),'tcp_target':xyz.tolist(),'duration_s':duration})

    def _guarded_move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        xyz=np.asarray(xyz,float).copy()
        if apply_residual:
            xyz+=self.offset;yaw+=self.residual_yaw;self.control_bounds['max_offset_m']=max(self.control_bounds['max_offset_m'],float(np.linalg.norm(self.offset)))
        if not (.22<xyz[0]<.42 and -.18<xyz[1]<.16 and .095<xyz[2]<.40):raise RuntimeError('command_workspace_guard')
        self.command_pitch=self.residual_pitch if apply_residual else 0.
        if phase in ['close','retry_close']:duration=max(duration,1.1)
        target=self.data.qpos[:6].copy() if self.in_tracking_tick and phase=='preload' else self.ik(xyz,yaw,self.command)
        self.phase=phase;n=round(duration/self.model.opt.timestep)
        # A tracking hold can enter mid-path. The previous phase command is not
        # the current state and would cause a backwards target jump here.
        self.active=gap==0 and phase!='release';qstart=self.data.qpos[:6].copy();k=0;next_track=round(.15/self.model.opt.timestep)
        if phase=='release':self.grip_ref=float(np.clip(self.data.qpos[6],0,.035))
        phase_start=self.grip_ref;cap_limit=min(28.,1.8*self.force_budget) if self.force_budget else 28.;cap=min(cap_limit,force);touchdown=False
        while k<n:
            count=min(self.sample_steps,n-k);dt=count*self.model.opt.timestep;previous=self.grip_ref
            u=(k+count)/n;s=u*u*(3-2*u);nominal=phase_start+s*(gap-phase_start)
            if self.active:
                measured=float(max(self.sensor_force));old_filtered=self.filtered_force
                self.filtered_force+=dt/(.025+dt)*(measured-self.filtered_force)
                self.force_slope=(self.filtered_force-old_filtered)/dt;self.last_feedback_force=measured
                predicted=self.filtered_force+min(max(0.,self.force_slope)*.020,.15*self.force_budget)
                limit=self.reference_drive*self.option.get('traction',1.)
                if predicted>.80*self.force_budget:self.drive=min(self.drive,max(1.5,limit*.80*self.force_budget/max(predicted,.1)))
                else:self.drive=min(limit,self.drive+20*dt)
                self.drive=float(np.clip(self.drive,1.5,cap_limit));cap=self.drive
                if predicted>.90*self.force_budget or measured>1.05*self.force_budget:
                    nominal=max(nominal,float(self.data.qpos[6])+max(0.,measured-.85*self.force_budget)/1800.)
                    self.force_changes+=1
                self.admittance_log.append({'phase':phase,'time_s':float(self.data.time),'force_N':measured,'predicted_force_N':predicted,'goal_N':.72*self.force_budget,
                  'gap_ref_m':float(np.clip(nominal,0,.035)),'velocity_m_s':float((np.clip(nominal,0,.035)-previous)/dt),'cap_N':cap})
            else:cap=float(min(cap_limit,force))
            self.grip_ref=float(np.clip(nominal,0,.035))
            self.control_bounds['min_cap_N']=min(self.control_bounds['min_cap_N'],cap);self.control_bounds['max_cap_N']=max(self.control_bounds['max_cap_N'],cap)
            self._raw_roll(qstart,target,previous,self.grip_ref,cap,k,n,count);self.sample(dt);k+=count
            if phase in ['lower','retry_lower'] and max(self.sensor_force)>.90*self.force_budget and np.linalg.norm(self.data.site_xpos[self.tcp]-xyz)<.020:
                self.events.append({'event':'near_target_touchdown_stop','phase':phase,'tcp':self.data.site_xpos[self.tcp].tolist()});touchdown=True;break
            if track and k>=next_track:
                held=self._tracking_tick(phase,yaw);next_track=k+round(.15/self.model.opt.timestep);self.phase=phase;self.active=True
                if held and k<n:qstart=self.data.qpos[:6].copy();n-=k;k=0;phase_start=self.grip_ref;next_track=round(.15/self.model.opt.timestep)
        self.command=self.data.qpos[:6].copy() if touchdown else target;self.gap=self.grip_ref;self.active=False
        self.command_updates.append({'phase':phase,'drive_N':self.drive,'cap_N':cap,'sensor':self.sensor_observation(),'tcp_target':xyz.tolist(),'duration_s':duration})

    def _regrasp(self,low,yaw):
        self.retries+=1;landing=low.copy();landing[2]+=.004;self.move(landing,yaw,.55,'retry_lower',0,self.drive)
        release_at=self.data.site_xpos[self.tcp].copy()-self.offset;self.move(release_at,yaw,.5,'release',.035,12)
        clear=low.copy();clear[2]+=.025;self.move(clear,yaw,.5,'regrasp_clear',.035,12)
        seen=self.observe('regrasp_open',prior_xy=self.estimate[:2])
        if seen.get('pose_valid'):
            delta=np.asarray(seen['xyz'])-self.estimate
            if np.linalg.norm(delta[:2])<.02 and abs(delta[2])<.008:
                self.offset+=np.clip(delta,[-.004,-.004,-.002],[.004,.004,.002])
                self.offset=np.clip(self.offset,[-.006,-.006,-.003],[.006,.006,.003]);self.residual_yaw=float(np.clip(seen['yaw']-yaw,-.15,.15))
                self.events.append({'event':'measured_visual_recenter','correction_m':self.offset.tolist()})
        self.residual_pitch=self.option['pitch']
        self.move(clear,yaw,.4,'regrasp_align',.035,12);self.move(low,yaw,.6,'regrasp_descend',.035,12)
        self.move(low,yaw,.25,'hold',.035,12);self.move(low,yaw,2.2,'retry_close',0,self.drive);self.regrasp_done=True

    def run(self):
        start=time.perf_counter();failure=self.failure
        try:
            if failure:raise RuntimeError(failure)
            yaw=float(np.clip(self.estimated_yaw,-.6,.6));low=self.estimate+.012*orientation(yaw)[:,0];high=low.copy();high[2]=.215
            self.move(high,yaw,.7,'approach',.035,12);seen=self.observe('before_descend')
            if seen.get('pose_valid'):
                delta=np.asarray(seen['xyz'])-self.estimate
                if np.linalg.norm(delta[:2])<.015 and abs(delta[2])<.008:
                    self.estimate=np.asarray(seen['xyz']);yaw=float(np.clip(seen['yaw'],-.6,.6));low=self.estimate+.012*orientation(yaw)[:,0];high=low.copy();high[2]=.215
            self.move(low,yaw,.8,'descend',.035,12);self.move(low,yaw,.25,'descend_settle',.035,12)
            self.move(low,yaw,.70,'close',0,self.drive);self.move(low,yaw,.10,'preload',0,self.drive)
            closed=self.observe('after_close');close_tcp=self.data.site_xpos[self.tcp].copy();opening=float(self.data.qpos[7]+self.data.qpos[8])
            probe=low.copy();probe[2]+=.015;self.move(probe,yaw,.65,'probe_lift',0,self.drive)
            seen=self.observe('probe_lift');x,raw=self.probe_features(closed,seen,close_tcp,opening)
            allowed,risk=allowed_options(x,self.policy.actions)
            if self.fixed_action is None:self.option_id,values=self.policy.act(x,self.learning,allowed)
            else:self.option_id=self.fixed_action;values=None
            self.option=OPTIONS[self.option_id];self.decision={'features':x.tolist(),'feature_names':list(FEATURE_NAMES),'observation':raw,'action':self.option_id,'values':values,'observable_risk':risk,'allowed_actions':allowed}
            self.reference_drive*=self.option['gentle'];self.reference_drive=float(np.clip(self.reference_drive,1.5,min(28.,1.8*self.force_budget)))
            weak=not raw['vision_valid'] or raw['object_probe_displacement_m'][2]<max(.003,.65*raw['tcp_probe_displacement_m'][2])
            if weak and self.option['regrasp']:
                self._regrasp(low,yaw);self.move(probe,yaw,.65,'probe_lift',0,self.drive);seen=self.observe('probe_retry')
            elif weak:
                self.retries+=1;self.reference_drive=min(28.,1.8*self.force_budget,self.reference_drive*1.15)
                # Maintain the trial-lift height while force settles. Repeatedly
                # lowering a held object onto the table created clamp impulses.
                self.move(probe,yaw,.50,'preload',0,self.drive);seen=self.observe('probe_retry')
            if self.option['preload_s']:self.move(probe,yaw,self.option['preload_s'],'preload',0,self.drive)
            self.follow_relative=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp] if seen['valid'] else None
            self.move(high,yaw,.8*self.option['slow'],'lift',0,self.drive,track=True)
            dest=np.r_[DESTINATION,.1+self.estimated_size[2]/2]+.012*orientation(yaw)[:,0];dh=dest.copy();dh[2]=.215
            self.move((high+dh)/2,yaw,.6*self.option['slow'],'transport_a',0,self.drive,track=True);self.move(dh,yaw,.6*self.option['slow'],'transport',0,self.drive,track=True)
            landing=dest.copy();landing[2]+=.003
            self.move(dh,yaw,.25,'hold',0,self.drive);self.move(landing,yaw,.9*self.option['slow'],'lower',0,self.drive)
            release_at=self.data.site_xpos[self.tcp].copy()-self.offset
            self.move(release_at,yaw,.5,'release',.035,12);self.move(dh,yaw,.8,'retreat',.035,12);self.move(dh,yaw,.35,'settle',.035,12)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:failure=str(exc)
        self.finished=True;m=self.metrics;final=self.data.xpos[self.ob].copy();placed=bool(np.max(abs(final[:2]-DESTINATION))<.025 and abs(final[2]-(.1+self.size[2]/2))<.010 and np.linalg.norm(self.data.qvel[self.od:self.od+3])<.08)
        full={'peak_force_N':float(max(self.fullrate[:2])),'overforce_s':float(self.fullrate[2]),'finger_penetration_m':float(self.fullrate[3]),'table_penetration_m':float(self.fullrate[4]),'peak_details':self.fullrate_peak_details}
        over=bool(m['overforce_s']>=.06 or full['overforce_s']>=.06-1e-10)
        success=bool(not failure and m['bilateral'] and m['lifted'] and not m['slip'] and not over and placed)
        reason=failure or ('ok' if success else 'force_budget_exceeded' if over else 'no_bilateral_contact' if not m['bilateral'] else 'not_lifted' if not m['lifted'] else 'slip' if m['slip'] else 'placement_failed')
        row={'success':success,'failure_type':reason,'seed':self.seed_value,'material':MATERIALS[self.mat],'condition':self.case,'sensor_condition':self.options.get('sensor_condition','nominal'),
         'profile':'safe_contact','physics':self.params(),'initial_object_xyz_m':self.initial.tolist(),'final_object_xyz_m':final.tolist(),'placement_error_m':float(np.linalg.norm(final[:2]-DESTINATION)),
         'contact_metrics':deepcopy(m),'fullrate':full,'placed':placed,'overclamp':over,'safety_abort':bool(failure),'policy_updated':False,'decision':self.decision,'option_id':self.option_id,'option':self.option,
         'events':deepcopy(self.events),'vision_calls':deepcopy(self.vision_calls),'decisions':self.command_updates,'retries':self.retries,'regrasp_performed':self.regrasp_done,'guard_holds':self.guard_holds,
         'penetration':self.max_penetration,'contact_parameters_seen':self.contact_parameters_seen,'duration_s':float(self.data.time-self.start_time),'wall_s':time.perf_counter()-start,
         'force_adjustments':self.force_changes,'force_recoveries':self.force_recoveries,'control_bounds':self.control_bounds,'object_pose_writes_during_run':0,'hardware_motor_commands':0,'policy_observation_privileged':False,
         'camera':'virtual RGBD, known simulated extrinsics','force_sensor':'100Hz, 20ms delay, 0.02N noise; full-rate force is evaluation only',
         'opening_measurement':'two finger encoders, not hardware calibrated jaw width','admittance_summary':{'updates':len(self.admittance_log),
          'max_goal_budget_ratio':float(max([d['goal_N']/self.force_budget for d in self.admittance_log],default=0.))}}
        row['reward']=reward(row)
        if self.learning and self.fixed_action is None and self.decision is not None:self.policy.update(row['reward']);row['policy_updated']=True
        else:self.policy.last=None
        return row
