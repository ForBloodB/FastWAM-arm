"""Native-step force integration plus bounded Cartesian-reference experiments.

All extra forces, masses and true object states are evaluation-only. Candidate
references read robot FK, visual category and the existing delayed force.
"""
import ctypes
from copy import deepcopy
from pathlib import Path
import mujoco
import numpy as np
from .contact_repair import ContactRepairEnv,REPAIRS

_LIB=None
VARIANTS=('baseline','combo_all','combo_observed','straight','straight_combo','recovery_verified')

def audited_roll(*args):
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build/impulse_audit/libimpulse_audit.so'))
        ptr=ctypes.POINTER(ctypes.c_double);D=ctypes.c_double;I=ctypes.c_int
        _LIB.impulse_audit_roll.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ptr,ptr,D,D,D,I,I,I,I,D,D,I,I,I,I,I,D,ptr]
        _LIB.impulse_audit_roll.restype=I
    m,d,a,b,ga,gb,cap,start,total,n,mode,fa,fb,shape,og,left,right,table,budget=args
    a=np.ascontiguousarray(a,dtype=float);b=np.ascontiguousarray(b,dtype=float);stats=np.zeros(41)
    assert a.shape==b.shape==(6,) and np.isfinite(np.r_[a,b,ga,gb,cap,fa,fb,budget]).all()
    assert 0<=ga<=.05 and 0<=gb<=.05 and 0<cap<=28 and 0<=fa<=28 and 0<=fb<=28
    assert start>=0 and n>0 and start+n<=total
    assert np.all(a>=m.jnt_range[:6,0]) and np.all(a<=m.jnt_range[:6,1])
    assert np.all(b>=m.jnt_range[:6,0]) and np.all(b<=m.jnt_range[:6,1])
    ptr=ctypes.POINTER(ctypes.c_double)
    rc=_LIB.impulse_audit_roll(m._address,d._address,a.ctypes.data_as(ptr),b.ctypes.data_as(ptr),ga,gb,cap,start,total,n,int(mode),fa,fb,int(shape),og,left,right,table,budget,stats.ctypes.data_as(ptr))
    if rc:raise RuntimeError('Native simulation diverged:'+str(rc))
    return stats

class ImpulseContactEnv(ContactRepairEnv):
    def __init__(self,variant='baseline',**kwargs):
        if variant not in VARIANTS:raise ValueError(variant)
        self.experiment_variant=variant;self.impulse_log=[];self.reference_log=[];self._category_selected=False
        self._segment=None;self._phase_yaw=0.
        super().__init__(repair='recover_center' if variant=='recovery_verified' else 'z_minus2_pitch_pos' if variant=='combo_all' else 'none',timing='clocked',**kwargs)

    def reset(self,**kwargs):
        self.impulse_log=[];self.reference_log=[];self._category_selected=False;self._segment=None
        if self.experiment_variant in ('combo_observed','straight_combo'):self.repair_spec=deepcopy(REPAIRS['none'])
        return super().reset(**kwargs)

    def move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        if phase=='descend' and not self._category_selected:
            self._category_selected=True
            if self.experiment_variant in ('combo_observed','straight_combo'):
                selected=bool(self.predicted==3)
                self.repair_spec=deepcopy(REPAIRS['z_minus2_pitch_pos' if selected else 'none'])
                self.reference_log.append(dict(event='observed_category_selection',visual_label=self.predicted+1,
                    selected=selected,source='RGBD color-coded proxy category, not true material or friction'))
        return super().move(xyz,yaw,duration,phase,gap,force,apply_residual,track)

    def _clocked_move(self,xyz,yaw,duration,phase,gap,force,apply_residual=True,track=False):
        self._phase_yaw=float(yaw);self._segment=None
        return super()._clocked_move(xyz,yaw,duration,phase,gap,force,apply_residual,track)

    def _fk(self,q):
        self.ikdata.qpos[:]=self.data.qpos;self.ikdata.qpos[:6]=q
        mujoco.mj_forward(self.model,self.ikdata)
        return self.ikdata.site_xpos[self.tcp].copy()

    def _timed_roll(self,implementation,*args):
        supplied=list(args)
        if self.experiment_variant in ('straight','straight_combo') and self.phase in ('probe_lift','lift'):
            m,d,a,b,ga,gb,cap,start,total,n,*_=supplied
            if start==0 or self._segment is None:
                self._segment=dict(start=self._fk(a),end=self._fk(b),previous=np.asarray(a).copy())
            segment=self._segment;u=(start+n)/total;s=u*u*u*(10+u*(-15+6*u))
            point=segment['start']+s*(segment['end']-segment['start'])
            target=self.ik(point,self._phase_yaw,segment['previous'])
            supplied[2]=segment['previous'].copy();supplied[3]=target.copy()
            supplied[7]=0;supplied[8]=n;supplied[13]=2
            segment['previous']=target.copy()
            self.reference_log.append(dict(event='straight_tcp_reference',phase=self.phase,time_s=float(self.data.time),
                fraction=float(u),desired_xyz_m=point.tolist(),q_target=target.tolist(),
                original_duration_s=float(total*m.opt.timestep),shared_cap_N=cap))
        stats=super()._timed_roll(audited_roll,*supplied)
        impulse=stats[22:25]+stats[16:19];delta=stats[25:28]
        self.impulse_log.append(dict(phase=self.phase,end_time_s=float(self.data.time),native_steps=int(stats[33]),duration_s=float(stats[34]),
            finger_impulse_Ns=stats[7:10].tolist(),table_impulse_Ns=stats[10:13].tolist(),other_impulse_Ns=stats[13:16].tolist(),
            smooth_impulse_Ns=stats[16:19].tolist(),constraint_impulse_Ns=stats[19:22].tolist(),contact_impulse_Ns=stats[22:25].tolist(),
            momentum_change_Ns=delta.tolist(),qacc_impulse_Ns=stats[28:31].tolist(),
            momentum_residual_Ns=(delta-impulse).tolist(),contact_constraint_max_error_N=float(stats[31]),
            native_step_max_momentum_residual_Ns=float(stats[32]),velocity_before_m_s=stats[35:38].tolist(),velocity_after_m_s=stats[38:41].tolist(),evaluation_only=True))
        return stats[:7]

    def _recovery_context(self,probe,raw):
        context=super()._recovery_context(probe,raw)
        if self.experiment_variant=='recovery_verified':
            reasons=list(context['reasons'])
            if max(self.sensor_force)>.1*self.force_budget:reasons.append('contact_still_supporting')
            if raw.get('tcp_probe_displacement_m',[0,0,0])[2]<.008:reasons.append('insufficient_probe_motion')
            if context.get('observed_bottom_height_m',1)>.002:reasons.append('not_visually_on_table')
            context.update(permitted=not reasons,reasons=reasons)
        return context

    def run(self):
        row=super().run()
        delta=np.sum([x['momentum_change_Ns'] for x in self.impulse_log],axis=0)
        contact=np.sum([x['contact_impulse_Ns'] for x in self.impulse_log],axis=0)
        smooth=np.sum([x['smooth_impulse_Ns'] for x in self.impulse_log],axis=0)
        row.update(experiment_variant=self.experiment_variant,impulse_audit=dict(native_steps=sum(x['native_steps'] for x in self.impulse_log),
            momentum_change_Ns=delta.tolist(),contact_impulse_Ns=contact.tolist(),smooth_impulse_Ns=smooth.tolist(),
            total_momentum_residual_norm_Ns=float(np.linalg.norm(delta-contact-smooth)),
            maximum_window_momentum_residual_Ns=max(np.linalg.norm(x['momentum_residual_Ns']) for x in self.impulse_log),
            maximum_contact_constraint_error_N=max(x['contact_constraint_max_error_N'] for x in self.impulse_log),
            scope='Native-step integration; privileged values evaluation only'),
            reference_scope='Same fast PD and caps; straight references replace lift trajectory only; recovery trigger reads delayed force and RGBD')
        return row
