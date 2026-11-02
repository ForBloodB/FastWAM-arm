"""Official mesh B601-RS contact task. One residual action per grasp attempt.

All object state writes are in reset. No weld, attraction or scripted object
motion. Native substeps use the previously validated torque PD controller.
"""
from copy import deepcopy
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .official_b601 import MODEL, DERIVED
from .official_native import roll

MATERIALS=('wood','plastic','sponge','metal')
CONDITIONS=('nominal','position','friction','compliance')
FRICTION=np.array([.65,.45,.8,.25])
DENSITY=np.array([600,950,80,5000.])
DRIVE_FORCE=np.array([6.,6.,2.5,10.])
FORCE_BUDGET=np.array([12.,12.,3.,20.])
COLORS=np.array([[.55,.015,.015,1],[.015,.025,.55,1],[.55,.45,.01,1],[.5,.01,.5,1]])
ACTION_SCALE=np.array([.008,.008,.004,.15,8.])
DESTINATION=np.array([.29,-.10])
EVIDENCE='official_b601_rs_mesh_contact_simulation'

def rotation_error(target,current):
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,(target@current.T).reshape(9))
    if quat[0]<0:quat=-quat
    s=np.linalg.norm(quat[1:])
    return 2*quat[1:] if s<1e-12 else quat[1:]/s*2*np.arctan2(s,max(0.,quat[0]))

def orientation(yaw):
    c,s=np.cos(yaw),np.sin(yaw);p=1.0;cp,sp=np.cos(p),np.sin(p)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])

CACHE=Path(__file__).resolve().parents[1]/'build/official_control/official_grasp.mjb'

def compile_task_model():
    tree=ET.parse(MODEL);root=tree.getroot()
    root.find('include').set('file',str(DERIVED/'rs_urdf_aligned_arm.xml'))
    root.find('compiler').set('meshdir',str((DERIVED/root.find('compiler').get('meshdir')).resolve()))
    world=root.find('worldbody')
    for name in ['blue_block','yellow_cylinder','ik_target']:
        world.remove(world.find(f"body[@name='{name}']"))
    body=world.find("body[@name='red_cube']");body.set('name','task_object')
    body.find('freejoint').set('name','object_free')
    geom=body.find('geom');geom.set('name','object_geom');geom.set('priority','2')
    # Higher object priority makes the declared material friction/compliance
    # actually govern contact instead of being masked by finger friction.
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))

def task_model():
    # Compile once in a separate process. Loading the identical serialized model
    # avoids MuJoCo's retained XML mesh compiler allocations in every actor.
    if not CACHE.is_file():raise RuntimeError('Run scripts/build_official_task_cache.py before starting actors')
    return mujoco.MjModel.from_binary_path(str(CACHE))

class OfficialGraspEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':25}
    def __init__(self,domain='online',condition=None,render_mode=None,record=False):
        super().__init__();self.domain=domain;self.condition=condition;self.render_mode=render_mode;self.record=record
        self.model=task_model();self.data=mujoco.MjData(self.model);self.ikdata=mujoco.MjData(self.model)
        self.ranges=self.model.jnt_range[:6].copy();self.tcp=self.model.site('tcp').id
        assert [self.model.joint(f'joint{i}').qposadr[0] for i in range(1,8)]==list(range(7))
        assert [self.model.actuator(f'joint{i}_motor').id for i in range(1,8)]==list(range(7))
        self.og=self.model.geom('object_geom').id;self.ob=self.model.body('task_object').id
        self.oq=int(self.model.joint('object_free').qposadr[0]);self.od=int(self.model.joint('object_free').dofadr[0])
        self.left=self.model.body('gripper_left').id;self.right=self.model.body('gripper_right').id
        self.robotbodies=set(range(1,self.model.body('task_table').id))
        self.action_space=spaces.Box(-1.,1.,shape=(5,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(20,),dtype=np.float32)
        self.renderer=None;self.finished=True
        self.home=self.ik([.30,0,.22],0,np.array([0,1.2,1.4,-1.2,0,0.]))

    def ik(self,xyz,yaw,seed):
        q=np.clip(np.asarray(seed,float).copy(),self.ranges[:,0]+1e-5,self.ranges[:,1]-1e-5)
        R=orientation(yaw);jp=np.zeros((3,self.model.nv));jr=jp.copy()
        for _ in range(100):
            self.ikdata.qpos[:6]=q;mujoco.mj_forward(self.model,self.ikdata)
            ep=np.asarray(xyz)-self.ikdata.site_xpos[self.tcp]
            er=rotation_error(R,self.ikdata.site_xmat[self.tcp].reshape(3,3))
            if np.linalg.norm(ep)<.00008 and np.linalg.norm(er)<.001:break
            mujoco.mj_jacSite(self.model,self.ikdata,jp,jr,self.tcp)
            J=np.vstack([jp[:,:6],.15*jr[:,:6]])
            delta=J.T@np.linalg.solve(J@J.T+.00002*np.eye(6),np.r_[ep,.15*er])
            q=np.clip(q+np.clip(delta,-.12,.12),self.ranges[:,0]+1e-5,self.ranges[:,1]-1e-5)
        self.ikdata.qpos[:6]=q;mujoco.mj_forward(self.model,self.ikdata)
        ep=np.linalg.norm(np.asarray(xyz)-self.ikdata.site_xpos[self.tcp])
        er=np.linalg.norm(rotation_error(R,self.ikdata.site_xmat[self.tcp].reshape(3,3)))
        if ep>.002 or er>.02:raise RuntimeError(f'ik_unreachable:{ep:.5f}:{er:.5f}')
        return q

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);rng=self.np_random;options=options or {};self.seed_value=seed
        self.mat=MATERIALS.index(options['material']) if 'material' in options else int(rng.integers(4))
        self.case=options.get('condition',self.condition or CONDITIONS[int(rng.integers(4))])
        width=rng.uniform(.029,.037);length=rng.uniform(.040,.048);height=rng.uniform(.038,.044)
        self.size=np.array([length,width,height]);self.yaw=float(rng.uniform(-.25,.25))
        self.xyz=np.array([rng.uniform(.29,.34),rng.uniform(-.025,.035),.1+height/2])
        if self.case=='position':self.xyz[:2]+=rng.uniform(-.012,.012,2);self.yaw=float(rng.uniform(-.45,.45))
        self.mu=float(FRICTION[self.mat]*rng.uniform(.85,1.15))
        if self.case=='friction':self.mu*=float(rng.uniform(.7,1.) if self.domain=='pretrain' else rng.uniform(.45,.75))
        self.mass=float(np.prod(self.size)*DENSITY[self.mat]*rng.uniform(.9,1.1))
        self.soft=.012 if self.mat==2 else .004
        if self.case=='compliance':self.soft=float(rng.uniform(.010,.020) if self.domain=='pretrain' else rng.uniform(.020,.035))
        self.model.geom_size[self.og]=self.size/2;self.model.geom_rbound[self.og]=np.linalg.norm(self.size/2)
        self.model.geom_friction[self.og]=[self.mu,.005,.0002];self.model.geom_solref[self.og]=[self.soft,1]
        self.model.geom_rgba[self.og]=COLORS[self.mat];self.model.body_mass[self.ob]=self.mass
        hx,hy,hz=self.size/2;self.model.body_inertia[self.ob]=self.mass/3*np.array([hy*hy+hz*hz,hx*hx+hz*hz,hx*hx+hy*hy])
        mujoco.mj_setConst(self.model,self.data);mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:6]=self.home;self.data.qpos[6:9]=.035
        self.data.qpos[self.oq:self.oq+7]=[*self.xyz,np.cos(self.yaw/2),0,0,np.sin(self.yaw/2)]
        mujoco.mj_forward(self.model,self.data)
        roll(self.model,self.data,self.home,self.home,.035,.035,12,0,80,80)
        self.initial=self.data.xpos[self.ob].copy();self.command=self.home.copy();self.gap=.035
        sigma=.0005 if self.domain=='pretrain' else .001
        self.observed=self.initial+rng.normal(0,sigma,3)
        if self.case=='position':self.observed+=np.array([.002,-.001,0]) if self.domain=='pretrain' else np.array([.005,-.004,0])
        self.observed_yaw=self.yaw+float(rng.normal(0,.015))
        if 'observed_xyz' in options:self.observed=np.asarray(options['observed_xyz'],float)
        if 'observed_yaw' in options:self.observed_yaw=float(options['observed_yaw'])
        self.frames=[];self.trace=[];self.lastforces=np.zeros(2);self.finished=False;self.phase='reset';self.start_time=float(self.data.time)
        self.metrics={'bilateral':False,'lifted':False,'max_lift_m':0.,'slip':False,'max_force_N':0.,'overforce_s':0.,'collision':False,'peak_torque_Nm':[0.]*6}
        return self.obs(),{'evidence':EVIDENCE,'parameters':self.params()}

    def params(self):
        return {'size_m':self.size.tolist(),'mass_kg':self.mass,'friction':self.mu,'solref_s':self.soft,'material':MATERIALS[self.mat],
                'condition':self.case,'force_budget_N_per_finger':float(FORCE_BUDGET[self.mat]),'contact_priority':2}

    def obs(self):
        return np.r_[self.observed,self.observed_yaw,self.data.site_xpos[self.tcp],np.eye(4)[self.mat],self.size[1],
                     self.data.qpos[:6]/np.pi,self.lastforces/20].astype(np.float32)

    def sample(self,dt):
        left=right=0.;force=np.zeros(6);collisions=[]
        for index,c in enumerate(self.data.contact):
            a,b=int(self.model.geom_bodyid[c.geom1]),int(self.model.geom_bodyid[c.geom2])
            if self.ob in (a,b):
                other=b if a==self.ob else a
                mujoco.mj_contactForce(self.model,self.data,index,force)
                if other==self.left:left+=max(0.,force[0])
                elif other==self.right:right+=max(0.,force[0])
                elif other in self.robotbodies and c.dist<-.0005:collisions.append('palm_or_arm_object')
            elif (a in self.robotbodies or b in self.robotbodies) and c.dist<-.0005:
                collisions.append(self.model.body(a).name+'/'+self.model.body(b).name)
        self.lastforces[:]=[left,right]
        m=self.metrics;m['bilateral']|=bool(left>.03 and right>.03)
        lift=float(self.data.xpos[self.ob,2]-self.initial[2]);m['max_lift_m']=max(m['max_lift_m'],lift)
        m['lifted']|=bool(lift>=.06)
        if self.phase in ['lift','transport','hold']:m['slip']|=bool(m['lifted'] and lift<.025)
        m['max_force_N']=max(m['max_force_N'],float(left),float(right))
        if max(left,right)>FORCE_BUDGET[self.mat]:m['overforce_s']+=dt
        m['peak_torque_Nm']=np.maximum(m['peak_torque_Nm'],abs(self.data.ctrl[:6])).tolist()
        m['collision']|=bool(collisions)
        if self.record:
            self.trace.append({'t':float(self.data.time),'phase':self.phase,'q':self.data.qpos[:6].tolist(),'tcp':self.data.site_xpos[self.tcp].tolist(),
                               'object':self.data.xpos[self.ob].tolist(),'force':[float(left),float(right)],'gap':float(self.data.qpos[6]),'collisions':collisions})
            if self.render_mode=='rgb_array' and len(self.trace)%4==0:self.frames.append(self.render())
        if collisions:raise RuntimeError('collision:'+','.join(collisions))

    def move(self,xyz,yaw,duration,phase,gap,force):
        target=self.ik(xyz,yaw,self.command);self.phase=phase;n=round(duration/self.model.opt.timestep)
        for start in range(0,n,10):
            count=min(10,n-start);roll(self.model,self.data,self.command,target,self.gap,gap,force,start,n,count)
            self.sample(count*self.model.opt.timestep)
        self.command=target;self.gap=gap

    def step(self,action):
        if self.finished:raise RuntimeError('Reset required')
        started=time.perf_counter();action=np.asarray(action,float);failure=None;force=0.;low=np.zeros(3);yaw=0.
        try:
            if action.shape!=(5,) or not np.isfinite(action).all() or np.max(abs(action))>1.000001:raise RuntimeError('invalid_action')
            yaw=float(np.clip(self.observed_yaw+ACTION_SCALE[3]*action[3],-.6,.6))
            low=self.observed+ACTION_SCALE[:3]*action[:3]+.012*orientation(yaw)[:,0]
            if not (.24<low[0]<.39 and -.09<low[1]<.10 and .098<low[2]<.135):raise RuntimeError('workspace_abort')
            force=float(np.clip(DRIVE_FORCE[self.mat]+ACTION_SCALE[4]*action[4],1.5,28.))
            high=low.copy();high[2]=.215
            dest=np.r_[DESTINATION,self.initial[2]]+.012*orientation(yaw)[:,0]
            dh=dest.copy();dh[2]=.215
            self.move(high,yaw,.7,'approach',.035,12)
            self.move(low,yaw,.8,'descend',.035,12)
            self.move(low,yaw,.55,'close',0,force)
            self.move(high,yaw,1.0,'lift',0,force)
            self.move(dh,yaw,1.0,'transport',0,force)
            self.move(dh,yaw,.25,'hold',0,force)
            self.move(dest,yaw,.8,'lower',0,force)
            self.move(dest,yaw,.4,'release',.035,12)
            self.move(dh,yaw,.8,'retreat',.035,12)
            self.move(dh,yaw,.35,'settle',.035,12)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:failure=str(exc)
        self.finished=True;m=self.metrics;final=self.data.xpos[self.ob].copy()
        distance=float(np.linalg.norm(final[:2]-DESTINATION));placed=bool(np.max(abs(final[:2]-DESTINATION))<.025 and
            abs(final[2]-(.1+self.size[2]/2))<.010 and np.linalg.norm(self.data.qvel[self.od:self.od+3])<.08)
        over=bool(m['overforce_s']>=.06)
        success=bool(not failure and m['bilateral'] and m['lifted'] and not m['slip'] and not over and placed)
        reason=failure or ('ok' if success else 'force_budget_exceeded' if over else 'no_bilateral_contact' if not m['bilateral'] else
            'not_lifted' if not m['lifted'] else 'slip' if m['slip'] else 'placement_failed')
        reward=float(-2 if failure else 2*success+.5*m['lifted']+.2*m['bilateral']-.2*min(distance/.15,2)-.5*m['slip']-.5*over-.01*np.mean(action**2))
        info={'is_success':success,'reward':reward,'failure_type':reason,'safety_abort':bool(failure),'overclamp':over,'slip':m['slip'],
            'contact_metrics':deepcopy(m),'initial_object_xyz_m':self.initial.tolist(),'final_object_xyz_m':final.tolist(),
            'placement_error_m':distance,'placed':placed,'duration_s':float(self.data.time-self.start_time),'wall_seconds':time.perf_counter()-started,
            'material':MATERIALS[self.mat],'condition':self.case,'physical_parameters':self.params(),'commanded_tcp':low.tolist(),'commanded_yaw':yaw,
            'drive_force_N':force,'interventions':0,'object_pose_writes_during_step':0,'object_welds':0,'evidence':EVIDENCE,'eligible_for_real_robot_summary':False}
        obs=self.obs()
        if not np.isfinite(obs).all():
            info['terminal_observation_sanitized']=True
            obs=np.nan_to_num(obs,nan=0.,posinf=0.,neginf=0.)
        return obs,reward,True,False,info

    def render(self,camera=None):
        if self.renderer is None:
            self.model.vis.global_.offwidth=640;self.model.vis.global_.offheight=480
            self.renderer=mujoco.Renderer(self.model,width=640,height=480)
        if camera is None:
            camera=mujoco.MjvCamera();camera.lookat[:]=[.25,-.015,.23];camera.distance=1.05;camera.azimuth=135;camera.elevation=-23
        self.renderer.update_scene(self.data,camera=camera);return self.renderer.render().copy()

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
