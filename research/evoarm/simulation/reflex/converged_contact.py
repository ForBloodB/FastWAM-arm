"""Isolated numerical candidates; no motor interfaces or relaxed force limits."""
import ctypes,inspect,textwrap,hashlib
from copy import deepcopy
from pathlib import Path
import mujoco,numpy as np
from .impulse_audit import ImpulseContactEnv
from . import impulse_audit as audit

MODES={'legacy':(0,False),'current_bias':(1,False),'aligned':(2,False),'synchronized':(2,True),'clocked_sync':(3,True),
       'implicit_gripper':(4,True),'implicit_all':(5,True),'implicit_integrator':(6,True),
       'damping2':(7,True),'damping4':(8,True),'damping6':(9,True),'guarded_implicit':(10,True),'force_separated':(11,True),
       'force_averaged':(12,True),'averaged_damping4':(13,True),'averaged_smooth':(14,True),
       'velocity_brake':(15,True),'velocity_brake4':(16,True),'averaged_brake':(17,True),
       'multiccd':(2,True),'multiccd_averaged':(12,True),'multiccd_smooth':(14,True),
       'multiccd_default':(2,True),'tight_single':(2,True)}
AVERAGED_KINDS={12,13,14,17}
BASE_DT=.0000625
class NumericalBridge:
 def __init__(self,kind,dt,sync):
  relative='build/convergence_brake/control.so' if kind>=15 else 'build/convergence_observation/control.so' if kind>=12 else 'build/convergence_forcecontrol/control.so' if kind==11 else 'build/convergence_guarded/control.so' if kind==10 else 'build/convergence_damping/control.so' if kind>=7 else 'build/convergence_implicit/control.so' if kind>=4 else 'build/convergence_native/control.so'
  path=Path(__file__).resolve().parents[1]/relative
  self.library_sha256=hashlib.sha256(path.read_bytes()).hexdigest();self.lib=ctypes.CDLL(str(path));P=ctypes.POINTER(ctypes.c_double);self.P=P;self.sync=sync;self.kind=kind
  self.lib.convergence_roll.argtypes=[ctypes.c_void_p,ctypes.c_void_p,P,P,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,P]
  self.lib.convergence_roll.restype=ctypes.c_int;self.lib.convergence_configure(kind,round(BASE_DT/dt))
 def __call__(self,*args):
  m,d,a,b,*tail=args;a=np.ascontiguousarray(a);b=np.ascontiguousarray(b);stats=np.zeros(43 if self.kind>=12 else 41)
  if int(m.actuator_forcelimited[6]):m.actuator_forcerange[6]=[-tail[2],tail[2]]
  rc=self.lib.convergence_roll(m._address,d._address,a.ctypes.data_as(self.P),b.ctypes.data_as(self.P),*tail,stats.ctypes.data_as(self.P))
  if rc:raise RuntimeError('convergence_native_failure_'+str(rc))
  if self.kind>=12:self.window_mean=stats[41:43]/stats[34];self.window_duration=stats[34]
  if self.sync:mujoco.mj_forward(m,d)
  return stats

class ConvergenceEnv(ImpulseContactEnv):
 def __init__(self,numerical_mode='synchronized',dt=BASE_DT,**kwargs):
  if numerical_mode not in MODES:raise ValueError(numerical_mode)
  self.numerical_mode=numerical_mode;self.integration_step=float(dt)
  # All reset/warm-up/camera-clear operations use the SAME physical clock.
  super().__init__(dt=BASE_DT,**kwargs)
  self.bridge=None
  self.original_actuation={name:getattr(self.model,name).copy() for name in ['actuator_biastype','actuator_biasprm','actuator_ctrllimited','actuator_forcelimited','actuator_forcerange']}
  self.original_collision_options=(self.model.opt.enableflags,self.model.opt.ccd_tolerance,self.model.opt.ccd_iterations)
 def reset(self,**kwargs):
  self.model.opt.timestep=BASE_DT;self.sample_steps=round(.01/BASE_DT);self.model.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
  for name,value in self.original_actuation.items():getattr(self.model,name)[:]=value
  self.model.opt.enableflags,self.model.opt.ccd_tolerance,self.model.opt.ccd_iterations=self.original_collision_options
  result=super().reset(**kwargs)
  self.model.opt.timestep=self.integration_step;self.sample_steps=round(.01/self.integration_step)
  kind,sync=MODES[self.numerical_mode]
  if self.numerical_mode.startswith('multiccd'):
   self.model.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_MULTICCD);self.model.opt.ccd_tolerance=1e-9;self.model.opt.ccd_iterations=200
   if self.numerical_mode=='multiccd_default':self.model.opt.ccd_tolerance=self.original_collision_options[1]
  elif self.numerical_mode=='tight_single':self.model.opt.ccd_tolerance=1e-9;self.model.opt.ccd_iterations=200
  if kind>=4:self.model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  if kind in (4,5,10):
   for i in (range(7) if kind==5 else [6]):
    assert self.model.actuator_gear[i,0]==1 and self.model.actuator_gainprm[i,0]==1
    self.model.actuator_biastype[i]=mujoco.mjtBias.mjBIAS_AFFINE
    self.model.actuator_biasprm[i,:]=0
    self.model.actuator_ctrllimited[i]=False;self.model.actuator_forcelimited[i]=True
    self.model.actuator_forcerange[i]=self.model.actuator_ctrlrange[i]
  self.bridge=NumericalBridge(kind,self.integration_step,sync)
  if sync:mujoco.mj_forward(self.model,self.data)
  self.common_start_digest=__import__('hashlib').sha256(np.r_[self.data.qpos,self.data.qvel,self.data.ctrl,self.data.time].tobytes()).hexdigest()
  return result
 def run(self):
  # Rebind the process-local native context at execution, so sequential use of
  # independently reset environments cannot silently inherit another mode.
  self.bridge.lib.convergence_configure(MODES[self.numerical_mode][0],round(BASE_DT/self.integration_step))
  old=audit.audited_roll;audit.audited_roll=self.bridge
  try:
   r=super().run();r['numerical_mode']=self.numerical_mode;r['common_start_digest']=self.common_start_digest;r['native_library_sha256']=self.bridge.library_sha256
   r['gripper_damping_Nsm']={'damping2':2.,'damping4':4.,'damping6':6.}.get(self.numerical_mode,18.)
   if self.numerical_mode=='force_separated':r.update(gripper_force_mode_damping_Nsm=0.,gripper_gap_mode_damping_Nsm=4.,lost_contact_brake_Nsm=18.)
   if MODES[self.numerical_mode][0] in AVERAGED_KINDS:
    r.update(force_observation='Causal native-step force integral over previous 10ms; 20ms delivery after window end; centroid age includes additional 5ms',force_sensor_group_delay_s=.005)
    r['gripper_damping_Nsm']=4. if MODES[self.numerical_mode][0] in (13,14) else 18.
   r['collision_manifold']=dict(multiccd=bool(self.model.opt.enableflags&int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),ccd_tolerance_m=float(self.model.opt.ccd_tolerance),ccd_iterations=int(self.model.opt.ccd_iterations))
   if MODES[self.numerical_mode][0]>=15:r.update(closure_brake_trigger_m_s=-.006,closure_brake_displacement_deadzone_m=0.,gripper_damping_Nsm=4. if self.numerical_mode=='velocity_brake4' else 18.)
   if self.numerical_mode=='guarded_implicit':
    self.bridge.lib.convergence_retries.restype=ctypes.c_longlong;self.bridge.lib.convergence_max_effective.restype=ctypes.c_double
    r['implicit_effective_force_max_N']=self.bridge.lib.convergence_max_effective();r['implicit_saturation_redone_steps']=self.bridge.lib.convergence_retries()
   r['numerical_fix_scope']='Current-stage feedback callback, reference timing, optional 100Hz observation synchronization; explicit damping candidates change the stated gain, not limits or contact physics'
   return r
  finally:audit.audited_roll=old
 def _capture_clocked(self,now):
  if MODES[self.numerical_mode][0] not in AVERAGED_KINDS:return super()._capture_clocked(now)
  if abs(self.bridge.window_duration-.01)>1e-9:raise RuntimeError('force_window_not_10ms')
  endpoint=self.lastforces.copy();self.lastforces=self.bridge.window_mean.copy()
  try:super()._capture_clocked(now)
  finally:self.lastforces=endpoint
  extra=dict(measurement_window_start_s=float(now-.01),measurement_window_end_s=float(now),measurement_centroid_s=float(now-.005),measurement_kind='10ms native integral / duration')
  self._force_pending[-1].update(extra);self.timing_log[-1].update(extra)

# Preserve the frozen 100Hz controller except its actuator feedback read: with
# affine damping, ctrl is a feedforward input, actuator_force is the applied force.
from .timed_contact import TimedContactEnv
from . import timed_contact as timed
_method=textwrap.dedent(inspect.getsource(TimedContactEnv._clocked_move))
assert _method.count('-self.data.ctrl[6]')==1
_method=_method.replace('-self.data.ctrl[6]','-self.data.actuator_force[6]')
_method=_method.replace('if predicted>threshold*budget:',"if predicted>threshold*budget or self.numerical_mode in ('averaged_smooth','multiccd_smooth'):")
_method=_method.replace('hold_goal-(6. if protected else 3.)*(predicted-(.70 if protected else .78)*budget)','hold_goal-(6. if protected else 3.)*max(0.,predicted-(.70 if protected else .78)*budget)')
_scope=dict(vars(timed));exec(compile(_method,'<applied-force-read-clocked-move>','exec'),_scope)
ConvergenceEnv._clocked_move=_scope['_clocked_move']
