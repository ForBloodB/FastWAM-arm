"""Paired sensor-feedback ablations on the repaired official mesh simulator.

Feedback is an explicit bounded rule controller, not a newly trained RL policy.
Vision is stage-wise; force sensor emulation is 100 Hz, delayed by 20 ms.
Oracle object state is read only by reset, evaluation, and the named oracle arm.
"""
from collections import deque
from copy import deepcopy
import time
import mujoco
import numpy as np
from .official_grasp import MATERIALS,DRIVE_FORCE,FORCE_BUDGET,DENSITY,FRICTION,DESTINATION,orientation
from .official_release_fix import MeasuredReleaseEnv
from .official_native import roll
from .feedback_vision import detect,pixel_labels

GROUPS=('oracle_open','vision_open','vision_reobserve','vision_contact','vision_analytic')

class FeedbackGraspEnv(MeasuredReleaseEnv):
    def __init__(self,group='vision_contact',backend=pixel_labels,record=False,render_mode=None):
        super().__init__(record=record,render_mode=render_mode)
        if group not in GROUPS:raise ValueError(group)
        self.group=group;self.backend=backend;self.sensor_renderer=None;self.events=[];self.stage='reset'

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed,options=options)
        self.options=options or {};self.events=[];self.vision_calls=[];self.frame_records=[];self.sensor_queue=deque(maxlen=3)
        self.sensor_rng=np.random.default_rng(seed+929292);self.sensor_force=np.zeros(2)
        self.predicted=None;self.estimate=None;self.estimated_size=None;self.drive=0.;self.force_changes=0;self.retries=0
        self.failure=None;self.vision_loss=0;self.command_updates=[];self.force_budget=0.;self.capture_index=0
        # Equal camera-clear motion for ALL groups; physics state is not teleported.
        clear=np.array([.65,.4,.8,-.4,0,0.]);start=self.command.copy();self.phase='camera_clear'
        for k in range(0,1000,10):
            roll(self.model,self.data,start,clear,.035,.035,12,k,1000,10);super().sample(.01)
        self.command=clear.copy();self.model.geom_matid[self.og]=-1
        initial=self.observe('initial')
        if not initial['valid']:
            self.failure='initial_detection_failed';return self.sensor_observation(),{}
        self.predicted=initial['label']-1;self.estimate=np.array(initial['xyz']);self.estimated_yaw=initial['yaw'];self.estimated_size=np.array(initial['size'])
        if self.group=='oracle_open':
            self.predicted=self.mat;self.estimate=self.initial.copy();self.estimated_yaw=self.yaw;self.estimated_size=self.size.copy()
        self.initial_estimate=self.estimate.copy();self.force_budget=float(FORCE_BUDGET[self.predicted])
        self.drive=float(DRIVE_FORCE[self.predicted])
        if self.group=='vision_analytic':
            mass=np.prod(self.estimated_size)*DENSITY[self.predicted]
            self.drive=float(np.clip(max(DRIVE_FORCE[self.predicted],2.*mass*9.81/FRICTION[self.predicted]),1.5,min(28,1.8*self.force_budget)))
        self.start_drive=self.drive;return self.sensor_observation(),{}

    def observe(self,name,prior_xy=None):
        if self.sensor_renderer is None:self.sensor_renderer=mujoco.Renderer(self.model,width=320,height=240)
        camera='overhead_rgb' if name=='initial' else 'wrist_rgb'
        cid=self.model.camera(camera).id;r=self.sensor_renderer
        r.update_scene(self.data,camera=camera);rgb=r.render().copy()
        r.enable_depth_rendering();r.update_scene(self.data,camera=camera);depth=r.render().copy();r.disable_depth_rendering()
        # Deterministic phase-keyed sensor noise, independent of controller calls.
        key=sum((i+1)*ord(c) for i,c in enumerate(name));rng=np.random.default_rng(self.seed_value+key*100003)
        depth_sigma=.0015 if self.options.get('sensor_condition')=='degraded' else .0003
        depth=depth+rng.normal(0,depth_sigma,depth.shape).astype(np.float32)
        if self.options.get('sensor_condition')=='degraded':depth[rng.random(depth.shape)<.08]=np.nan
        rgb=np.clip(rgb.astype(float)+rng.normal(0,2,rgb.shape),0,255).astype(np.uint8)
        T=np.eye(4);T[:3,:3]=self.data.cam_xmat[cid].reshape(3,3);T[:3,3]=self.data.cam_xpos[cid]
        result,labels=detect(rgb,depth,T,float(self.model.cam_fovy[cid]),backend=self.backend,
                             height=None if self.estimated_size is None else float(self.estimated_size[2]),
                             wanted_label=None if self.predicted is None else self.predicted+1,
                             prior_xy=prior_xy if prior_xy is not None else None if self.estimate is None else self.estimate[:2],prior_size=self.estimated_size)
        result.update(phase=name,camera=camera,sim_time_s=float(self.data.time),tcp=self.data.site_xpos[self.tcp].tolist())
        self.vision_calls.append(result)
        if self.record:self.frame_records.append({'phase':name,'rgb':rgb,'depth':depth,'labels':labels,'camera_to_world':T.copy(),'fovy':float(self.model.cam_fovy[cid])})
        return result

    def sensor_observation(self):
        # Explicit policy input whitelist: no self.mat, mass, mu, true pose, reward.
        return {'object_xyz':None if self.estimate is None else self.estimate.tolist(),
                'object_size':None if self.estimated_size is None else self.estimated_size.tolist(),
                'predicted_label':self.predicted,'joint_q':self.data.qpos[:6].tolist(),
                'gripper_encoder':float(self.data.qpos[6]),'force_sensor_N':self.sensor_force.tolist(),'phase':self.phase}

    def sample(self,dt):
        super().sample(dt)
        if self.phase=='transport_a':
            # Evaluation only: retain the original slip criterion on both halves.
            self.metrics['slip'] |= bool(self.metrics['lifted'] and self.data.xpos[self.ob,2]-self.initial[2]<.025)
        # Simulated fingertip force transducers, NOT real motor-current estimates.
        measured=np.maximum(0,self.lastforces+self.sensor_rng.normal(0,.02,2))
        self.sensor_queue.append(measured)
        if len(self.sensor_queue)==3:self.sensor_force=self.sensor_queue[0].copy()

    def move(self,xyz,yaw,duration,phase,gap,force):
        if phase=='release':self.gap=float(np.clip(self.data.qpos[6],0,.05))
        target=self.ik(xyz,yaw,self.command);self.phase=phase;n=round(duration/self.model.opt.timestep)
        active=phase in ('close','probe_lift','retry_lower','retry_close','lift','transport','transport_a','hold','lower')
        for start in range(0,n,10):
            cap=self.drive if active else force
            if self.group=='vision_contact' and active and len(self.sensor_queue)==3:
                # Only decrease cap in response to measured force excess. Visual
                # lift verification requests increases, within the SAME cap bound.
                excess=float(max(self.sensor_force)-.88*self.force_budget)
                if excess>0:
                    old=self.drive;self.drive=max(1.5,self.drive*.80*self.force_budget/max(self.sensor_force))
                    self.force_changes+=int(abs(old-self.drive)>1e-6);cap=self.drive
            count=min(10,n-start);roll(self.model,self.data,self.command,target,self.gap,gap,cap,start,n,count)
            self.sample(count*self.model.opt.timestep)
        self.command=target;self.gap=gap
        self.command_updates.append({'phase':phase,'drive_N':self.drive,'sensor':self.sensor_observation(),'tcp_target':np.asarray(xyz).tolist()})

    def run(self):
        t0=time.perf_counter();failure=self.failure;low=np.zeros(3);yaw=0.;repeated=self.group in ('vision_reobserve','vision_contact')
        try:
            if failure:raise RuntimeError(failure)
            yaw=float(np.clip(self.estimated_yaw,-.6,.6));low=self.estimate+.012*orientation(yaw)[:,0]
            high=low.copy();high[2]=.215
            self.move(high,yaw,.7,'approach',.035,12)
            if repeated:
                seen=self.observe('before_descend')
                if seen.get('pose_valid',False):
                    delta=np.array(seen['xyz'])-self.estimate
                    if np.linalg.norm(delta[:2])<.015 and abs(delta[2])<.008:
                        self.estimate=np.array(seen['xyz']);yaw=float(np.clip(seen['yaw'],-.6,.6))
                        low=self.estimate+.012*orientation(yaw)[:,0];high=low.copy();high[2]=.215
                else:self.vision_loss+=1
            if not (.24<low[0]<.39 and -.09<low[1]<.10 and .098<low[2]<.14):raise RuntimeError('workspace_abort')
            self.move(low,yaw,.8,'descend',.035,12)
            self.move(low,yaw,.55,'close',0,self.drive)
            closed=self.observe('after_close') if repeated else None
            close_tcp=self.data.site_xpos[self.tcp].copy()
            probe=low.copy();probe[2]+=.015
            self.move(probe,yaw,.35,'probe_lift',0,self.drive)
            relative=None
            if repeated:
                seen=self.observe('probe_lift')
                if seen['valid']:
                    rise=seen['xyz'][2]-(closed['xyz'][2] if closed and closed['valid'] else self.initial_estimate[2])
                    actual_tcp_rise=float(self.data.site_xpos[self.tcp,2]-close_tcp[2])
                    if rise<max(.003,.65*actual_tcp_rise):
                        self.retries+=1;self.drive=min(28.,1.8*self.force_budget,self.drive*2.)
                        self.events.append({'event':'visual_lift_retry','observed_rise_m':rise,'drive_N':self.drive})
                        self.move(low,yaw,.35,'retry_lower',0,self.drive)
                        self.move(low,yaw,.25,'retry_close',0,self.drive)
                        self.move(probe,yaw,.35,'probe_lift',0,self.drive)
                        seen=self.observe('probe_retry')
                    if seen['valid']:relative=np.array(seen['xyz'])-self.data.site_xpos[self.tcp]
                else:self.vision_loss+=1
            self.move(high,yaw,.8,'lift',0,self.drive)
            if repeated:
                seen=self.observe('after_lift')
                if seen['valid'] and seen['xyz'][2]-self.initial_estimate[2]<.04:
                    self.events.append({'event':'visual_lift_unconfirmed'})
                elif not seen['valid']:self.vision_loss+=1
                if seen['valid'] and relative is not None:
                    now=np.array(seen['xyz'])-self.data.site_xpos[self.tcp]
                    if now[2]<relative[2]-.003 or np.linalg.norm(now[:2]-relative[:2])>.005:
                        self.drive=min(28.,1.8*self.force_budget,self.drive*1.35)
                        self.events.append({'event':'visual_slip_correction','phase':'lift','drive_N':self.drive})
                    relative=now
            # Placement height comes from INITIAL RGB-D size, not sim truth.
            dest=np.r_[DESTINATION,.1+self.estimated_size[2]/2]+.012*orientation(yaw)[:,0]
            dh=dest.copy();dh[2]=.215
            mid=(high+dh)/2
            self.move(mid,yaw,.6,'transport_a',0,self.drive)
            if repeated:
                prior=self.data.site_xpos[self.tcp,:2]+(relative[:2] if relative is not None else np.zeros(2))
                seen=self.observe('mid_transport',prior_xy=prior)
                if seen['valid'] and relative is not None:
                    now=np.array(seen['xyz'])-self.data.site_xpos[self.tcp]
                    if now[2]<relative[2]-.003 or np.linalg.norm(now[:2]-relative[:2])>.005:
                        self.drive=min(28.,1.8*self.force_budget,self.drive*1.35)
                        self.events.append({'event':'visual_slip_correction','phase':'transport','drive_N':self.drive})
                elif not seen['valid']:self.vision_loss+=1
            self.move(dh,yaw,.6,'transport',0,self.drive)
            self.move(dh,yaw,.25,'hold',0,self.drive)
            self.move(dest,yaw,.8,'lower',0,self.drive)
            self.move(dest,yaw,.4,'release',.035,12)
            self.move(dh,yaw,.8,'retreat',.035,12)
            self.move(dh,yaw,.35,'settle',.035,12)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:failure=str(exc)
        self.finished=True;m=self.metrics;final=self.data.xpos[self.ob].copy()
        distance=float(np.linalg.norm(final[:2]-DESTINATION))
        placed=bool(np.max(abs(final[:2]-DESTINATION))<.025 and abs(final[2]-(.1+self.size[2]/2))<.010 and np.linalg.norm(self.data.qvel[self.od:self.od+3])<.08)
        over=bool(m['overforce_s']>=.06)
        success=bool(not failure and m['bilateral'] and m['lifted'] and not m['slip'] and not over and placed)
        reason=failure or ('ok' if success else 'force_budget_exceeded' if over else 'no_bilateral_contact' if not m['bilateral'] else 'not_lifted' if not m['lifted'] else 'slip' if m['slip'] else 'placement_failed')
        return {'success':success,'failure_type':reason,'group':self.group,'seed':self.seed_value,'material':MATERIALS[self.mat],
                'condition':self.case,'sensor_condition':self.options.get('sensor_condition','nominal'),
                'placement_error_m':distance,'contact_metrics':deepcopy(m),'initial_object_xyz_m':self.initial.tolist(),'final_object_xyz_m':final.tolist(),
                'physics':self.params(),'placed':placed,'overclamp':over,'safety_abort':bool(failure),'duration_s':float(self.data.time-self.start_time),
                'wall_s':time.perf_counter()-t0,'initial_drive_N':getattr(self,'start_drive',0.),'final_drive_N':self.drive,
                'force_adjustments':self.force_changes,'retries':self.retries,'vision_losses':self.vision_loss,
                'vision_calls':deepcopy(self.vision_calls),'events':self.events,'decisions':self.command_updates,
                'object_pose_writes_during_run':0,'hardware_motor_commands':0,'controller':'bounded_rules_not_RL',
                'camera':'virtual_overhead_initial_and_official_wrist_feedback_not_Gemini_handeye','contact_sensor':'simulated_force_transducer_100Hz_20ms_delay'}

    def close(self):
        if self.sensor_renderer:self.sensor_renderer.close();self.sensor_renderer=None
        super().close()
