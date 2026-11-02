"""Simulation-only known-foam task, geometry localization, unchanged dynamics.

The task prior is one foam proxy in the declared workspace, not recognition of
unknown material. Object color is a rendering-only nuisance. Inherited physics,
3 N per-finger budget, full-rate scoring and 28 N hard drive cap are untouched.
"""
import time
import cv2,mujoco,numpy as np
from .contact_roll_actions import ContactRollActionsEnv
from .feedback_vision import pixel_labels

MODES=('color_reference','depth_geometry','depth_recolored')
PALETTE=((.75,.75,.75,1),(.85,.10,.12,1),(.12,.20,.85,1),(.85,.75,.12,1))

def known_foam_color(rgb,foreground):
    labels=pixel_labels(rgb,foreground)
    return np.where(labels==3,3,0).astype(np.uint8)

def known_foam_geometry(rgb,foreground):
    return (np.asarray(foreground,dtype=bool).astype(np.uint8)*3)

def depth_target(depth,transform,fovy,prior_xyz=None,prior_size=None):
    """Segment the visible horizontal target surface in a predicted depth slab.

    Priors come from previous accepted depth detections and robot FK. No object
    simulator IDs, pose, mass, friction or rendering color enter this function.
    """
    start=time.perf_counter_ns();h,w=depth.shape;f=h/2/np.tan(np.deg2rad(fovy)/2)
    v,u=np.indices(depth.shape);cam=np.stack([(u-w/2)*depth/f,-(v-h/2)*depth/f,-depth],axis=-1)
    world=cam@transform[:3,:3].T+transform[:3,3]
    fg=np.isfinite(depth)&(depth>0)&(world[:,:,0]>.23)&(world[:,:,0]<.40)&(world[:,:,1]>-.18)&(world[:,:,1]<.16)
    if prior_xyz is None:
        fg&=(world[:,:,2]>.12)&(world[:,:,2]<.17)
    else:
        top=prior_xyz[2]+prior_size[2]/2
        fg&=(np.linalg.norm(world[:,:,:2]-prior_xyz[:2],axis=2)<.038)&(abs(world[:,:,2]-top)<.015)
    pts=world[fg]
    if len(pts)<12:return dict(valid=False,pixels=len(pts),reason='depth_track_empty'),np.zeros(depth.shape,np.uint8)
    bins=np.arange(float(np.min(pts[:,2]))-.001,float(np.max(pts[:,2]))+.002,.001)
    hist,edges=np.histogram(pts[:,2],bins);index=int(np.argmax(np.convolve(hist,[1,1,1],mode='same')))
    plane=(edges[index]+edges[index+1])/2
    surface=fg&(abs(world[:,:,2]-plane)<.002)
    count,cc,stats,_=cv2.connectedComponentsWithStats(surface.astype(np.uint8))
    choices=[(int(stats[k,cv2.CC_STAT_AREA]),k) for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>=12]
    if not choices:return dict(valid=False,pixels=0,reason='no_surface_component'),np.zeros(depth.shape,np.uint8)
    n,k=max(choices);mask=cc==k;pts=world[mask];xy=pts[:,:2]
    values,basis=np.linalg.eigh(np.cov(xy.T));axis=basis[:,-1]
    yaw=float((np.arctan2(axis[1],axis[0])+np.pi/2)%np.pi-np.pi/2)
    projected=xy@basis;extent=np.ptp(projected,axis=0);center=((projected.min(axis=0)+projected.max(axis=0))/2)@basis.T
    top=float(np.median(pts[:,2]));height=top-.1 if prior_size is None else float(prior_size[2])
    size=[float(max(extent)),float(min(extent)),height]
    valid=bool(.012<size[1]<.065 and .02<size[0]<.085 and .02<height<.07)
    if prior_size is not None:valid &= bool(size[0]>.6*prior_size[0] and size[1]>.6*prior_size[1])
    return dict(valid=valid,pose_valid=valid,label=3,xyz=[float(center[0]),float(center[1]),top-height/2],
        yaw=yaw,size=size,pixels=n,wall_ms=(time.perf_counter_ns()-start)/1e6),mask.astype(np.uint8)*3

class FoamTaskEnv(ContactRollActionsEnv):
    def __init__(self,perception_mode='depth_geometry',**kwargs):
        if perception_mode not in MODES:raise ValueError(perception_mode)
        self.perception_mode=perception_mode;self.color_applied=False;self.depth_track=None
        super().__init__(rescue_action='zero',**kwargs)
        self.backend=known_foam_color if perception_mode=='color_reference' else known_foam_geometry

    def reset(self,*,seed=None,options=None):
        if not options or options.get('material')!='sponge':raise ValueError('Known-foam task only')
        self.color_applied=False;self.depth_track=None
        return super().reset(seed=seed,options=options)

    def observe(self,name,prior_xy=None):
        if name=='initial' and self.perception_mode=='depth_recolored' and not self.color_applied:
            self.model.geom_rgba[self.og]=PALETTE[int(self.seed_value)%len(PALETTE)]
            self.color_applied=True
        if self.perception_mode=='color_reference':result=super().observe(name,prior_xy)
        else:
            if self.sensor_renderer is None:self.sensor_renderer=mujoco.Renderer(self.model,width=320,height=240)
            camera='overhead_rgb' if name=='initial' else 'wrist_rgb';cid=self.model.camera(camera).id;r=self.sensor_renderer
            r.update_scene(self.data,camera=camera);rgb=r.render().copy()
            r.enable_depth_rendering();r.update_scene(self.data,camera=camera);depth=r.render().copy();r.disable_depth_rendering()
            key=sum((i+1)*ord(c) for i,c in enumerate(name));rng=np.random.default_rng(self.seed_value+key*100003)
            depth_sigma=.0015 if self.options.get('sensor_condition')=='degraded' else .0003
            depth=depth+rng.normal(0,depth_sigma,depth.shape).astype(np.float32)
            if self.options.get('sensor_condition')=='degraded':depth[rng.random(depth.shape)<.08]=np.nan
            rgb=np.clip(rgb.astype(float)+rng.normal(0,2,rgb.shape),0,255).astype(np.uint8)
            transform=np.eye(4);transform[:3,:3]=self.data.cam_xmat[cid].reshape(3,3);transform[:3,3]=self.data.cam_xpos[cid]
            tcp=self.data.site_xpos[self.tcp].copy();prior=None;size=None
            if self.depth_track is not None:
                prior=np.array(self.depth_track['xyz']);size=self.depth_track['size']
                if self.phase in ('probe_lift','lift','transport','transport_a','hold','lower','release','settle'):
                    prior+=tcp-np.array(self.depth_track['tcp'])
                    prior[2]=max(.1+size[2]/2,prior[2])
                if prior_xy is not None:prior[:2]=prior_xy
            result,labels=depth_target(depth,transform,float(self.model.cam_fovy[cid]),prior,size)
            result.update(phase=name,camera=camera,sim_time_s=float(self.data.time),tcp=tcp.tolist())
            if result['valid']:self.depth_track=dict(xyz=result['xyz'],size=result['size'] if size is None else size,tcp=tcp.tolist())
            self.vision_calls.append(result)
            if self.record:self.frame_records.append(dict(phase=name,rgb=rgb,depth=depth,labels=labels,camera_to_world=transform.copy(),fovy=float(self.model.cam_fovy[cid])))
        result['material_source']='known single-foam task declaration; not predicted from image'
        result['localization_source']='RGB threshold + depth' if self.perception_mode=='color_reference' else 'depth foreground geometry; RGB ignored'
        return result

    def run(self):
        result=super().run()
        result.update(perception_mode=self.perception_mode,known_material_prior='sponge',
            material_recognition_claim=False,physical_robot_commands=0,
            scope='Official-model simulation; rigid foam proxy; no real deformable foam or real camera validation')
        return result
