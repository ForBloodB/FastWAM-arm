"""Sensor-conditioned residual options on separately audited contact pairs.

No hardware interfaces. Object truth is used only by reset, scoring and
diagnostics. Action selection uses RGB-D, FK/encoders and delayed force only.
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
import hashlib,json,time
import gymnasium as gym
import mujoco
import numpy as np
from .official_grasp import (OfficialGraspEnv,MATERIALS,CONDITIONS,FRICTION,DENSITY,
                            DRIVE_FORCE,FORCE_BUDGET,COLORS,DESTINATION,orientation,task_model)
from .feedback_grasp import FeedbackGraspEnv
from .feedback_vision import pixel_labels
from .slip_residual import SlipResidualEnv
from .official_native import roll

ROOT=Path(__file__).resolve().parents[1]
OPTIONS=[
    {'name':'zero','slow':1.,'preload_s':0.,'guard':False,'regrasp':False,'gentle':1.,'balance':False,'pitch':0.},
    {'name':'preload_slow','slow':1.5,'preload_s':.25,'guard':True,'regrasp':False,'gentle':1.,'balance':False,'pitch':0.},
    {'name':'recenter_regrasp','slow':1.5,'preload_s':.20,'guard':True,'regrasp':True,'gentle':1.,'balance':False,'pitch':.10},
    {'name':'gentle_slow','slow':1.3,'preload_s':.20,'guard':True,'regrasp':False,'gentle':.75,'balance':False,'pitch':0.},
    {'name':'balance_settle','slow':1.5,'preload_s':.20,'guard':True,'regrasp':False,'gentle':.9,'balance':True,'pitch':0.},
    {'name':'upright_regrasp','slow':1.6,'preload_s':.25,'guard':True,'regrasp':True,'gentle':1.,'balance':False,'pitch':.45},
    {'name':'seat_regrasp','slow':1.6,'preload_s':.25,'guard':True,'regrasp':True,'gentle':1.,'balance':False,'pitch':.30,'seat_m':-.003},
]
FEATURE_NAMES=('bias','visual_wood','visual_plastic','visual_sponge','visual_metal','visual_volume',
               'probe_rise_ratio','relative_probe_drop','opening','opening_change',
               'left_force_ratio','right_force_ratio','force_imbalance','vision_missing','lateral_probe_slip')

class StatePolicy:
    """Discounted linear action-value regression with explicit epsilon exploration.

    One post-probe decision and one terminal utility update per attempt.
    All features are passed in, never read from environment internals.
    """
    def __init__(self,seed=0,epsilon=.2,decay=.997,actions=None,ridge=2.):
        self.rng=np.random.default_rng(seed);self.epsilon=epsilon;self.decay=decay;self.ridge=ridge
        self.actions=list(range(len(OPTIONS))) if actions is None else list(actions)
        n=len(FEATURE_NAMES);self.A=np.tile(np.eye(n)*ridge,(len(self.actions),1,1));self.b=np.zeros((len(self.actions),n))
        self.count=np.zeros(len(self.actions),int);self.updates=0;self.last=None
    def act(self,x,learn=False,allowed=None):
        x=np.asarray(x,float)
        if x.shape!=(len(FEATURE_NAMES),) or not np.isfinite(x).all():raise ValueError('Invalid sensor feature vector')
        values=np.array([np.linalg.solve(a,b)@x for a,b in zip(self.A,self.b)])
        available=[i for i,a in enumerate(self.actions) if allowed is None or a in allowed]
        if not available:raise ValueError('Empty observable action mask')
        if learn and self.rng.random()<self.epsilon:i=int(self.rng.choice(available))
        else:
            best=max(values[i] for i in available);i=next(i for i in available if values[i]==best)
        self.last=(x.copy(),i);return self.actions[i],values.tolist()
    def update(self,reward):
        x,i=self.last;I=np.eye(len(x))*self.ridge
        self.A=I+self.decay*(self.A-I);self.b*=self.decay
        self.A[i]+=np.outer(x,x);self.b[i]+=float(reward)*x;self.count[i]+=1;self.updates+=1;self.last=None
    def state(self):
        return {'A':self.A.tolist(),'b':self.b.tolist(),'count':self.count.tolist(),'updates':self.updates,
                'actions':self.actions,'decay':self.decay,'ridge':self.ridge,'features':list(FEATURE_NAMES)}
    def load(self,s):
        assert s['features']==list(FEATURE_NAMES)
        self.A=np.array(s['A'],float);self.b=np.array(s['b'],float);self.count=np.array(s['count'],int)
        self.updates=s['updates'];self.actions=s['actions'];self.decay=s['decay'];self.ridge=s['ridge'];return self
    def digest(self):return hashlib.sha256(json.dumps(self.state(),sort_keys=True).encode()).hexdigest()

class ContactStateEnv(SlipResidualEnv):
    def __init__(self,policy=None,learning=False,fixed_action=None,profile='pairs6',record=False,render_mode=None,dt=.001):
        # Construct explicitly to keep all old loaders and their snapshots intact.
        self.command_pitch=0.;self.profile=profile;self.domain='online';self.condition=None;self.record=record;self.render_mode=render_mode
        if profile=='legacy':self.model=task_model()
        else:
            if profile not in ['pairs3','pairs6','audited']:raise ValueError(profile)
            cache_profile='pairs6' if profile=='audited' else profile
            self.model=mujoco.MjModel.from_binary_path(str(ROOT/f'build/contact_state/official_{cache_profile}.mjb'))
        if profile=='audited':
            self.model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
            self.model.opt.impratio=10.;self.model.opt.tolerance=1e-10
        self.model.opt.timestep=dt
        self.table_pair_id=None if profile=='legacy' else self.model.pair('task_table_pair').id
        self.finger_pair_ids=[] if profile=='legacy' else [self.model.pair(f'task_{side}_{part}_pair').id
            for side in ['left','right'] for part in ['front','mid','rear','carriage','stop']]
        self.data=mujoco.MjData(self.model);self.ikdata=mujoco.MjData(self.model)
        self.ranges=self.model.jnt_range[:6].copy();self.tcp=self.model.site('tcp').id
        self.og=self.model.geom('object_geom').id;self.ob=self.model.body('task_object').id
        self.oq=int(self.model.joint('object_free').qposadr[0]);self.od=int(self.model.joint('object_free').dofadr[0])
        self.left=self.model.body('gripper_left').id;self.right=self.model.body('gripper_right').id
        self.robotbodies=set(range(1,self.model.body('task_table').id))
        self.renderer=None;self.sensor_renderer=None;self.finished=True
        self.home=self.ik([.30,0,.22],0,np.array([0,1.2,1.4,-1.2,0,0.]))
        self.policy=policy or StatePolicy();self.learning=learning;self.fixed_action=fixed_action
        self.group='vision_contact';self.backend=pixel_labels;self.sample_steps=round(.01/dt)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed);rng=self.np_random;options=options or {};self.options=options;self.seed_value=seed
        self.command_pitch=0.;self.mat=MATERIALS.index(options['material']) if 'material' in options else int(rng.integers(4))
        # Preserve the original reset's eager default draw, including when an
        # explicit condition is provided, for paired legacy physics checks.
        self.case=options.get('condition',CONDITIONS[int(rng.integers(4))])
        width=rng.uniform(.029,.037);length=rng.uniform(.040,.048);height=rng.uniform(.038,.044)
        self.size=np.array([length,width,height]);self.yaw=float(rng.uniform(-.25,.25))
        self.xyz=np.array([rng.uniform(.29,.34),rng.uniform(-.025,.035),.1+height/2])
        if self.case=='position':self.xyz[:2]+=rng.uniform(-.012,.012,2);self.yaw=float(rng.uniform(-.45,.45))
        self.mu=float(FRICTION[self.mat]*rng.uniform(.85,1.15))
        if self.case=='friction':self.mu*=float(rng.uniform(.7,1.) if self.domain=='pretrain' else rng.uniform(.45,.75))
        self.mass=float(np.prod(self.size)*DENSITY[self.mat]*rng.uniform(.9,1.1));self.soft=.012 if self.mat==2 else .004
        if self.case=='compliance':self.soft=float(rng.uniform(.010,.020) if self.domain=='pretrain' else rng.uniform(.020,.035))
        self.model.geom_size[self.og]=self.size/2;self.model.geom_rbound[self.og]=np.linalg.norm(self.size/2)
        self.model.geom_friction[self.og]=[self.mu,.005,.0002];self.model.geom_solref[self.og]=[self.soft,1]
        self.model.geom_rgba[self.og]=COLORS[self.mat];self.model.geom_matid[self.og]=-1
        self.model.body_mass[self.ob]=self.mass;hx,hy,hz=self.size/2
        self.model.body_inertia[self.ob]=self.mass/3*np.array([hy*hy+hz*hz,hx*hx+hz*hz,hx*hx+hy*hy])
        if self.profile!='legacy':
            self.model.pair_friction[:]=[self.mu,self.mu,.005,.0002,.0002]
            self.model.pair_solref[self.table_pair_id]=[.003,1]
            self.model.pair_solref[self.finger_pair_ids]=[self.soft,1]
        mujoco.mj_setConst(self.model,self.data);mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:6]=self.home;self.data.qpos[6:9]=.035
        self.data.qpos[self.oq:self.oq+7]=[*self.xyz,np.cos(self.yaw/2),0,0,np.sin(self.yaw/2)]
        mujoco.mj_forward(self.model,self.data);warm=round(.08/self.model.opt.timestep)
        roll(self.model,self.data,self.home,self.home,.035,.035,12,0,warm,warm)
        self.initial=self.data.xpos[self.ob].copy();self.command=self.home.copy();self.gap=.035
        self.observed=self.initial.copy();self.observed_yaw=self.yaw
        self.frames=[];self.trace=[];self.lastforces=np.zeros(2);self.finished=False;self.phase='reset';self.start_time=float(self.data.time)
        self.metrics={'bilateral':False,'lifted':False,'max_lift_m':0.,'slip':False,'max_force_N':0.,'overforce_s':0.,'collision':False,'peak_torque_Nm':[0.]*6}
        self.events=[];self.vision_calls=[];self.frame_records=[];self.command_updates=[];self.sensor_queue=deque(maxlen=3)
        self.sensor_rng=np.random.default_rng(seed+929292);self.sensor_force=np.zeros(2)
        self.predicted=None;self.estimate=None;self.estimated_size=None;self.drive=0.;self.force_changes=0;self.retries=0
        self.failure=None;self.vision_loss=0;self.force_budget=0.;self.capture_index=0
        self.actual_cap=12.;self.reference_drive=0.;self.force_recoveries=0;self.over_sensor_s=0.;self.active=False
        self.max_penetration={'finger_m':0.,'table_m':0.};self.offset=np.zeros(3);self.residual_pitch=0.;self.option_id=0
        self.residual_yaw=0.;self.follow_relative=None;self.track_index=0
        self.contact_parameters_seen={}
        self.option=OPTIONS[0];self.decision=None;self.guard_holds=0;self.regrasp_done=False;self.slip_signals=0
        self.control_bounds={'min_cap_N':30.,'max_cap_N':0.,'max_offset_m':0.};self.policy.last=None
        clear=np.array([.65,.4,.8,-.4,0,0.]);start=self.command.copy();self.phase='camera_clear';n=round(1./self.model.opt.timestep)
        for k in range(0,n,self.sample_steps):
            count=min(self.sample_steps,n-k);roll(self.model,self.data,start,clear,.035,.035,12,k,n,count)
            OfficialGraspEnv.sample(self,count*self.model.opt.timestep)
        self.command=clear.copy();initial=self.observe('initial')
        if not initial['valid']:self.failure='initial_detection_failed';return self.sensor_observation(),{}
        self.predicted=initial['label']-1;self.estimate=np.array(initial['xyz']);self.estimated_yaw=initial['yaw'];self.estimated_size=np.array(initial['size'])
        self.initial_estimate=self.estimate.copy();self.force_budget=float(FORCE_BUDGET[self.predicted])
        nominal_mass=np.prod(self.estimated_size)*DENSITY[self.predicted]
        self.drive=float(np.clip(max(DRIVE_FORCE[self.predicted],2*nominal_mass*9.81/FRICTION[self.predicted]),1.5,min(28.,1.8*self.force_budget)))
        self.reference_drive=self.drive;self.start_drive=self.drive;return self.sensor_observation(),{}

    def params(self):
        p=OfficialGraspEnv.params(self)
        p.update(profile=self.profile,dt=float(self.model.opt.timestep),table_solref_s=self.soft if self.profile=='legacy' else float(self.model.pair_solref[self.table_pair_id,0]),
                 finger_condim=3 if self.profile=='legacy' else int(self.model.pair_dim[self.finger_pair_ids[0]]),
                 cone=int(self.model.opt.cone),impratio=float(self.model.opt.impratio),solver_tolerance=float(self.model.opt.tolerance))
        return p

    def sample(self,dt):
        try:FeedbackGraspEnv.sample(self,dt)
        finally:
            for c in self.data.contact:
                if self.og in (c.geom1,c.geom2):
                    other=c.geom2 if c.geom1==self.og else c.geom1;body=int(self.model.geom_bodyid[other])
                    if body in (self.left,self.right):self.max_penetration['finger_m']=max(self.max_penetration['finger_m'],-float(c.dist))
                    elif self.model.body(body).name=='task_table':self.max_penetration['table_m']=max(self.max_penetration['table_m'],-float(c.dist))
                    kind='finger' if body in (self.left,self.right) else 'table' if self.model.body(body).name=='task_table' else 'other'
                    self.contact_parameters_seen[kind]={'condim':int(c.dim),'solref':c.solref.tolist(),'friction':c.friction.tolist()}
        if self.active and self.force_budget:
            self.over_sensor_s=self.over_sensor_s+dt if max(self.sensor_force)>1.5*self.force_budget else 0.
            if self.over_sensor_s>=.02:raise RuntimeError('delayed_force_peak_guard')
        else:self.over_sensor_s=0.

    def move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        xyz=np.asarray(xyz,float).copy()
        if apply_residual:
            xyz+=self.offset
            yaw+=self.residual_yaw
            self.control_bounds['max_offset_m']=max(self.control_bounds['max_offset_m'],float(np.linalg.norm(self.offset)))
        if not (.22<xyz[0]<.42 and -.18<xyz[1]<.16 and .095<xyz[2]<.40):raise RuntimeError('command_workspace_guard')
        self.command_pitch=self.residual_pitch if apply_residual else 0.
        if phase=='release':self.gap=float(np.clip(self.data.qpos[6],0,.05))
        target=self.ik(xyz,yaw,self.command);self.phase=phase;n=round(duration/self.model.opt.timestep)
        self.active=phase in ['close','probe_lift','retry_lower','retry_close','lift','transport_a','transport','hold','lower','preload','balance']
        cap_limit=min(28.,1.8*self.force_budget) if self.force_budget else 28.
        qstart=self.command.copy();k=0;next_track=round(.15/self.model.opt.timestep)
        while k<n:
            dt=min(self.sample_steps,n-k)*self.model.opt.timestep
            if self.active:
                measured=float(max(self.sensor_force));old=self.drive
                if measured>.85*self.force_budget:
                    self.drive=max(1.5,self.drive-60*dt*min(1.,(measured/.85/self.force_budget-1.)+.1))
                elif measured<.65*self.force_budget:
                    self.drive=min(self.reference_drive,self.drive+6*dt)
                self.drive=float(np.clip(self.drive,1.5,cap_limit));self.force_changes+=int(self.drive<old-1e-8)
                self.force_recoveries+=int(self.drive>old+1e-8);requested=self.drive
                self.actual_cap=float(np.clip(requested,self.actual_cap-80*dt,self.actual_cap+20*dt))
                self.actual_cap=float(np.clip(self.actual_cap,1.5,cap_limit));cap=self.actual_cap
            else:cap=force
            self.control_bounds['min_cap_N']=min(self.control_bounds['min_cap_N'],cap)
            self.control_bounds['max_cap_N']=max(self.control_bounds['max_cap_N'],cap)
            count=min(self.sample_steps,n-k);roll(self.model,self.data,qstart,target,self.gap,gap,cap,k,n,count);self.sample(dt);k+=count
            if track and k>=next_track:
                held=self._tracking_tick(phase,yaw)
                next_track=k+round(.15/self.model.opt.timestep)
                self.phase=phase;self.active=True
                if held and k<n:
                    # Resume from measured joint encoders after the hold, with
                    # a fresh smooth trajectory over the remaining duration.
                    qstart=self.data.qpos[:6].copy();n-=k;k=0;next_track=round(.15/self.model.opt.timestep)
        self.command=target;self.gap=gap;self.active=False
        self.command_updates.append({'phase':phase,'drive_N':self.drive,'cap_N':cap,'sensor':self.sensor_observation(),'tcp_target':xyz.tolist(),'duration_s':duration})

    def probe_features(self,closed,probe,close_tcp,close_opening):
        valid=bool(closed.get('valid') and probe.get('valid'))
        tcp=np.asarray(self.data.site_xpos[self.tcp]);tcp_delta=tcp-np.asarray(close_tcp)
        visual_delta=np.array(probe['xyz'])-closed['xyz'] if valid else np.zeros(3)
        relative=visual_delta-tcp_delta;opening=float(self.data.qpos[7]+self.data.qpos[8]);F=self.sensor_force.copy()
        raw={'vision_label':int(self.predicted),'estimated_volume_m3':float(np.prod(self.estimated_size)),
             'vision_valid':valid,'object_probe_displacement_m':visual_delta.tolist(),'tcp_probe_displacement_m':tcp_delta.tolist(),
             'relative_probe_displacement_m':relative.tolist(),'opening_encoder_m':opening,'opening_change_m':opening-close_opening,
             'left_force_N':float(F[0]),'right_force_N':float(F[1]),'force_difference_N':float(F[0]-F[1]),
             'force_budget_N':self.force_budget}
        x=np.r_[1.,np.eye(4)[self.predicted],np.clip(raw['estimated_volume_m3']/.000060-1,-1,1),
                np.clip(visual_delta[2]/max(tcp_delta[2],.003),-1,2),np.clip(-relative[2]/.01,-1,2),
                np.clip(opening/.07,0,1.5),np.clip((opening-close_opening)/.01,-2,2),
                np.clip(F/self.force_budget,0,2),np.clip((F[0]-F[1])/(sum(F)+.2),-1,1),float(not valid),
                np.clip(np.linalg.norm(relative[:2])/.01,0,2)]
        assert len(x)==len(FEATURE_NAMES);return x,raw

    def _observe_follow(self,name,relative):
        prior=self.data.site_xpos[self.tcp,:2]+(relative[:2] if relative is not None else np.zeros(2))
        seen=self.observe(name,prior_xy=prior)
        if not seen['valid']:self.vision_loss+=1;return seen,None
        now=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp]
        return seen,now

    def _regrasp(self,low,yaw):
        self.retries+=1;self.move(low,yaw,.45,'retry_lower',0,self.drive)
        self.move(low,yaw,.45,'release',.035,12)
        clear=low.copy();clear[2]+=.025;self.move(clear,yaw,.5,'regrasp_clear',.035,12)
        seen=self.observe('regrasp_open',prior_xy=self.estimate[:2])
        if seen.get('pose_valid'):
            delta=np.asarray(seen['xyz'])-self.estimate
            if np.linalg.norm(delta[:2])<.02 and abs(delta[2])<.008:
                self.offset+=np.clip(delta,[-.01,-.01,-.004],[.01,.01,.004])
                self.residual_yaw=float(np.clip(seen['yaw']-yaw,-.25,.25))
                self.events.append({'event':'regrasp_visual_recenter','correction_m':self.offset.tolist()})
        self.offset[:2]+=.004*np.array([np.cos(yaw),np.sin(yaw)]);self.offset=np.clip(self.offset,[-.012,-.012,-.004],[.012,.012,.004])
        self.offset[2]+=self.option.get('seat_m',0.);self.offset[2]=float(np.clip(self.offset[2],-.006,.004))
        self.residual_pitch=self.option['pitch'];self.move(clear,yaw,.4,'regrasp_align',.035,12)
        self.move(low,yaw,.6,'regrasp_descend',.035,12);self.move(low,yaw,.25,'hold',.035,12)
        self.move(low,yaw,.75,'retry_close',0,self.drive);self.move(low,yaw,.2,'preload',0,self.drive)
        self.regrasp_done=True

    def _tracking_tick(self,phase,yaw):
        self.track_index+=1;relative=self.follow_relative
        seen,now=self._observe_follow(f'{phase}_track_{self.track_index}',relative);held=False
        if now is not None and relative is not None:
            falling=bool(now[2]<relative[2]-.003 or np.linalg.norm(now[:2]-relative[:2])>.006)
            self.slip_signals=self.slip_signals+1 if falling else 0
            if falling:
                if max(self.sensor_force)<.8*self.force_budget:
                    self.reference_drive=min(28.,1.8*self.force_budget,self.reference_drive*1.1)
                self.events.append({'event':'relative_visual_slip','phase':phase,'delta_m':(now-relative).tolist()})
            if self.option['guard'] and self.slip_signals>=2 and self.guard_holds<2:
                # Hold the measured FK position to stop advancing the path while
                # bounded force feedback settles. No object pose correction.
                tcp=self.data.site_xpos[self.tcp].copy()-self.offset
                # yaw passed here already includes the residual; remove it
                # before move applies the same residual again.
                self.move(tcp,yaw-self.residual_yaw,.25,'preload',0,self.drive);self.guard_holds+=1;held=True
                self.events.append({'event':'stop_preload_hold','phase':phase})
                seen,now=self._observe_follow(f'{phase}_track_{self.track_index}_held',relative);self.slip_signals=0
        if now is not None:self.follow_relative=now
        return held

    def run(self):
        start=time.perf_counter();failure=self.failure
        try:
            if failure:raise RuntimeError(failure)
            yaw=float(np.clip(self.estimated_yaw,-.6,.6));low=self.estimate+.012*orientation(yaw)[:,0];high=low.copy();high[2]=.215
            self.move(high,yaw,.7,'approach',.035,12)
            seen=self.observe('before_descend')
            if seen.get('pose_valid'):
                delta=np.asarray(seen['xyz'])-self.estimate
                if np.linalg.norm(delta[:2])<.015 and abs(delta[2])<.008:
                    self.estimate=np.asarray(seen['xyz']);yaw=float(np.clip(seen['yaw'],-.6,.6))
                    low=self.estimate+.012*orientation(yaw)[:,0];high=low.copy();high[2]=.215
            self.move(low,yaw,.8,'descend',.035,12)
            self.move(low,yaw,.25,'descend_settle',.035,12)
            self.move(low,yaw,.70,'close',0,self.drive);self.move(low,yaw,.10,'preload',0,self.drive)
            closed=self.observe('after_close');close_tcp=self.data.site_xpos[self.tcp].copy();close_opening=float(self.data.qpos[7]+self.data.qpos[8])
            probe=low.copy();probe[2]+=.015;self.move(probe,yaw,.45,'probe_lift',0,self.drive)
            seen=self.observe('probe_lift');x,raw=self.probe_features(closed,seen,close_tcp,close_opening)
            risk=bool(not raw['vision_valid'] or raw['relative_probe_displacement_m'][2]<-.003 or
                      abs(raw['opening_change_m'])>.002 or abs(x[12])>.25 or max(x[10:12])>.8)
            allowed=self.policy.actions if risk else [0]
            if self.fixed_action is None:self.option_id,values=self.policy.act(x,self.learning,allowed)
            else:self.option_id=self.fixed_action;values=None
            self.option=OPTIONS[self.option_id];self.decision={'features':x.tolist(),'feature_names':list(FEATURE_NAMES),'observation':raw,
                'action':self.option_id,'values':values,'observable_risk':risk,'allowed_actions':allowed}
            self.reference_drive*=self.option['gentle'];self.reference_drive=float(np.clip(self.reference_drive,1.5,min(28.,1.8*self.force_budget)))
            self.drive=min(self.drive,self.reference_drive)
            weak=not raw['vision_valid'] or raw['object_probe_displacement_m'][2]<max(.003,.65*raw['tcp_probe_displacement_m'][2])
            if self.option['regrasp'] and weak:
                self._regrasp(low,yaw);self.move(probe,yaw,.5,'probe_lift',0,self.drive);seen=self.observe('probe_retry')
            elif weak:
                self.retries+=1;self.reference_drive=min(28.,1.8*self.force_budget,self.reference_drive*1.2)
                self.move(low,yaw,.4,'retry_lower',0,self.drive);self.move(low,yaw,.4,'retry_close',0,self.drive)
                self.move(probe,yaw,.45,'probe_lift',0,self.drive);seen=self.observe('probe_retry')
            if self.option['balance']:
                F=self.sensor_force;delta=-.002*np.clip((F[0]-F[1])/(sum(F)+.2),-1,1)
                self.offset[:2]+=delta*np.array([-np.sin(yaw),np.cos(yaw)])
                self.move(probe,yaw,.3,'balance',0,self.drive)
                self.events.append({'event':'force_balance_shift','local_y_m':float(delta)})
            if self.option['preload_s']:self.move(probe,yaw,self.option['preload_s'],'preload',0,self.drive)
            self.follow_relative=np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp] if seen['valid'] else None
            self.move(high,yaw,.8*self.option['slow'],'lift',0,self.drive,track=True)
            dest=np.r_[DESTINATION,.1+self.estimated_size[2]/2]+.012*orientation(yaw)[:,0];dh=dest.copy();dh[2]=.215
            mid=(high+dh)/2
            self.move(mid,yaw,.6*self.option['slow'],'transport_a',0,self.drive,track=True)
            self.move(dh,yaw,.6*self.option['slow'],'transport',0,self.drive,track=True)
            self.move(dh,yaw,.25,'hold',0,self.drive);self.move(dest,yaw,.9*self.option['slow'],'lower',0,self.drive)
            self.move(dest,yaw,.5,'release',.035,12);self.move(dh,yaw,.8,'retreat',.035,12);self.move(dh,yaw,.35,'settle',.035,12)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:failure=str(exc)
        self.finished=True;m=self.metrics;final=self.data.xpos[self.ob].copy();distance=float(np.linalg.norm(final[:2]-DESTINATION))
        placed=bool(np.max(abs(final[:2]-DESTINATION))<.025 and abs(final[2]-(.1+self.size[2]/2))<.010 and np.linalg.norm(self.data.qvel[self.od:self.od+3])<.08)
        over=bool(m['overforce_s']>=.06);success=bool(not failure and m['bilateral'] and m['lifted'] and not m['slip'] and not over and placed)
        reason=failure or ('ok' if success else 'force_budget_exceeded' if over else 'no_bilateral_contact' if not m['bilateral'] else 'not_lifted' if not m['lifted'] else 'slip' if m['slip'] else 'placement_failed')
        peak_excess=max(0.,m['max_force_N']/float(FORCE_BUDGET[self.mat])-1.)
        reward=float(2*success-1.*over-.4*m['slip']-1.5*bool(failure)-.15*min(peak_excess,3.))
        updated=bool(self.learning and self.fixed_action is None and self.decision is not None)
        if updated:self.policy.update(reward)
        else:self.policy.last=None
        return {'success':success,'failure_type':reason,'seed':self.seed_value,'material':MATERIALS[self.mat],'condition':self.case,
          'sensor_condition':self.options.get('sensor_condition','nominal'),'profile':self.profile,'physics':self.params(),
          'initial_object_xyz_m':self.initial.tolist(),'final_object_xyz_m':final.tolist(),'placement_error_m':distance,
          'contact_metrics':deepcopy(m),'placed':placed,'overclamp':over,'safety_abort':bool(failure),'reward':reward,'policy_updated':updated,
          'decision':self.decision,'option_id':self.option_id,'option':self.option,'events':deepcopy(self.events),'vision_calls':deepcopy(self.vision_calls),
          'decisions':self.command_updates,'retries':self.retries,'regrasp_performed':self.regrasp_done,'guard_holds':self.guard_holds,
          'vision_losses':self.vision_loss,'penetration':self.max_penetration,'contact_parameters_seen':self.contact_parameters_seen,
          'duration_s':float(self.data.time-self.start_time),'wall_s':time.perf_counter()-start,
          'force_adjustments':self.force_changes,'force_recoveries':self.force_recoveries,'control_bounds':self.control_bounds,
          'object_pose_writes_during_run':0,'hardware_motor_commands':0,'policy_observation_privileged':False,
          'camera':'virtual RGBD, exact simulated extrinsics','force_sensor':'simulated 100 Hz, 20 ms delay',
          'opening_measurement':'sum of two finger slide encoders; not hardware calibrated jaw width',
          'reward_source':'simulator terminal scoring; never used as action input'}
