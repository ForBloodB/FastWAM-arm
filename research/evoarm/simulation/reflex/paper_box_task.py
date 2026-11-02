"""Known empty paper box proxy, simulation only. No hardware interfaces.

Initialization specializes the audited reset before settling/rendering. Closed
thin-shell mass/inertia and external convex collision envelope are explicit.
No local panel bending, buckling, tearing or measured crush-strength claim.
"""
from collections import deque
from copy import deepcopy
import gymnasium as gym
import mujoco,numpy as np
from .contact_state import ContactStateEnv,OPTIONS
from .official_grasp import OfficialGraspEnv,MATERIALS,CONDITIONS,FORCE_BUDGET,DENSITY,orientation
from .official_native import roll
from .foam_task import FoamTaskEnv

MODES=('baseline','gentle','gentle_recolored')

class PaperBoxInitialization(ContactStateEnv):
    """Inserted before ContactStateEnv by C3 MRO; other reset layers are retained."""
    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed);rng=self.np_random;options=options or {};self.options=options;self.seed_value=seed
        self.command_pitch=0.;self.mat=MATERIALS.index(options['material']) if 'material' in options else int(rng.integers(4))
        # Preserve the original reset's eager default draw, including when an
        # explicit condition is provided, for paired legacy physics checks.
        self.case=options.get('condition',CONDITIONS[int(rng.integers(4))])
        # Scenario assumptions, not measured paper properties. Closed uniform shell.
        width=rng.uniform(.029,.037);length=rng.uniform(.040,.048);height=rng.uniform(.038,.044)
        self.size=np.array([length,width,height]);self.yaw=float(rng.uniform(-.25,.25))
        self.xyz=np.array([rng.uniform(.29,.34),rng.uniform(-.025,.035),.1+height/2])
        if self.case=='position':self.xyz[:2]+=rng.uniform(-.012,.012,2);self.yaw=float(rng.uniform(-.45,.45))
        self.mu=float(rng.uniform(.12,.22) if self.case=='friction' else rng.uniform(.25,.6))
        thickness=float(rng.uniform(.00025,.00045));density=float(rng.uniform(500.,700.))
        inner=self.size-2*thickness;outer_v=float(np.prod(self.size));inner_v=float(np.prod(inner))
        self.mass=density*(outer_v-inner_v)
        shell_I=np.array([density/12*(outer_v*(self.size[j]**2+self.size[k]**2)-inner_v*(inner[j]**2+inner[k]**2)) for j,k in ((1,2),(0,2),(0,1))])
        self.soft=.012
        tc=float(rng.uniform(.003,.006)) if self.case=='compliance' else .004
        self.box_setup=dict(thickness_m=thickness,shell_density_kg_m3=density,mass_kg=self.mass,
            shell_inertia_kg_m2=shell_I.tolist(),effective_contact_tc_s=tc,measured=False)
        self.model.geom_size[self.og]=self.size/2;self.model.geom_rbound[self.og]=np.linalg.norm(self.size/2)
        self.model.geom_friction[self.og]=[self.mu,.005,.0002];self.model.geom_solref[self.og]=[self.soft,1]
        self.model.geom_rgba[self.og]=[.62,.38,.17,1.];self.model.geom_matid[self.og]=-1
        self.model.body_mass[self.ob]=self.mass;hx,hy,hz=self.size/2
        self.model.body_inertia[self.ob]=shell_I
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
        self.drive=2.5 if self.control_variant=='baseline' else 1.5
        self.reference_drive=self.drive;self.start_drive=self.drive;return self.sensor_observation(),{}


class PaperBoxEnv(FoamTaskEnv,PaperBoxInitialization):
    def __init__(self,mode='gentle',**kwargs):
        if mode not in MODES:raise ValueError(mode)
        self.paper_mode=mode;self.control_variant='baseline' if mode=='baseline' else 'gentle'
        self.box_setup=None
        super().__init__(perception_mode='depth_recolored' if mode=='gentle_recolored' else 'depth_geometry',**kwargs)

    def reset(self,*,seed=None,options=None):
        if not options or options.get('material')!='paper_box':raise ValueError('Known paper-box scenarios only')
        internal=deepcopy(options);internal['material']='sponge'
        # Index 2 is an inherited 3 N controller/evaluator category, not a material prediction.
        result=super().reset(seed=seed,options=internal)
        self.model.pair_solref[self.finger_pair_ids]=[self.box_setup['effective_contact_tc_s'],1.]
        self.model.pair_solimp[self.finger_pair_ids]=[.98,.9999,.001,.5,2.]
        mujoco.mj_forward(self.model,self.data)
        assert self.model.body_mass[self.ob]==self.mass
        assert np.all(self.model.body_inertia[self.ob]>0)
        return result

    def observe(self,*args,**kwargs):
        r=super().observe(*args,**kwargs)
        r['material_source']='known paper-box task prior; label 3 only for inherited 3 N budget'
        return r

    def params(self):
        r=super().params();r.update(material='paper_box',paper_box=deepcopy(self.box_setup),
            contact_model='rigid hollow-shell inertia with convex outer collision envelope',
            numerical_contact_scope='declared numerical contact variation; not measured cardboard crushing')
        return r

    def run(self):
        r=super().run()
        r.update(material='paper_box',known_material_prior='paper_box',paper_mode=self.paper_mode,
                 control_variant=self.control_variant,known_mass_input_to_controller=False,
                 initial_drive_target_N=self.start_drive,scope='Known closed empty paper-box rigid proxy; actual dimensions and material calibration pending',
                 material_recognition_claim=False,paper_damage_prediction_claim=False,
                 virtual_color_only=self.paper_mode=='gentle_recolored')
        return r
