"""Implicit damping with an effective-force bound and rollback on saturation.

No passive damping is added. The original 18 Ns/m gain is kept. If the implicit
linearization would exceed the actuator cap, redo that physical step with the
bounded constant force and zero actuator derivative. Native contact peaks from
discarded attempts never enter statistics; accepted physical steps do.
"""
from pathlib import Path
import subprocess,mujoco
s=Path('build/convergence_native/control.c').read_text()
s=s.replace('static int kind=0,stride=1,iteration=0;','static int kind=0,stride=1,iteration=0,force_override=0; static double override_value=0; static long long retries=0; static double worst_effective=0;')
s=s.replace('kind=k;stride=st;cursor=0;', 'kind=k;stride=st;cursor=0;retries=0;worst_effective=0;')
s=s.replace('static void current_control(', 'long long convergence_retries(){return retries;}\ndouble convergence_max_effective(){return worst_effective;}\nstatic void current_control(')
s=s.replace('  d->ctrl[6]=fmax(-cap,fmin(cap,force));','''
  if(kind==10){
   mjModel *w=(mjModel*)m;int interior=force>-cap && force<cap;
   w->actuator_biasprm[mjNBIAS*6+2]=force_override?0:(interior?-18:0);
   d->ctrl[6]=force_override?override_value:(interior?force+18*d->qvel[6]:fmax(-cap,fmin(cap,force)));
  }else d->ctrl[6]=fmax(-cap,fmin(cap,force));''')
needle='''  mjfGeneric previous=mjcb_control;
  if(kind)mjcb_control=current_control;
  mj_step(m,d);
  mjcb_control=previous;cursor++;'''
replacement='''
  mjfGeneric previous=mjcb_control;
  double saved_q[m->nq],saved_v[m->nv],saved_warm[m->nv],saved_t=d->time;
  if(kind==10){mju_copy(saved_q,d->qpos,m->nq);mju_copy(saved_v,d->qvel,m->nv);mju_copy(saved_warm,d->qacc_warmstart,m->nv);}
  if(kind)mjcb_control=current_control;
  mj_step(m,d);
  if(kind==10){
   double effective=d->actuator_force[6]+m->actuator_biasprm[mjNBIAS*6+2]*(d->qvel[6]-saved_v[6]);
   if(fabs(effective)>cap+1e-10){
    retries++;force_override=1;override_value=fmax(-cap,fmin(cap,effective));
    mju_copy(d->qpos,saved_q,m->nq);mju_copy(d->qvel,saved_v,m->nv);mju_copy(d->qacc_warmstart,saved_warm,m->nv);d->time=saved_t;
    mj_step(m,d);effective=d->actuator_force[6];force_override=0;
   }
   if(fabs(effective)>cap+1e-9 || fabs(d->actuator_force[6])>cap+1e-9){mjcb_control=previous;return 12;}
   worst_effective=fmax(worst_effective,fabs(effective));
   ((mjModel*)m)->actuator_biasprm[mjNBIAS*6+2]=0;d->ctrl[6]=effective;
  }
  mjcb_control=previous;cursor++;'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
out=Path('build/convergence_guarded');out.mkdir(parents=True,exist_ok=True);(out/'control.c').write_text(s)
r=Path(mujoco.__file__).parent
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'control.c'),'-I'+str(r/'include'),str(next(r.glob('libmujoco.so*'))),'-Wl,-rpath,'+str(r),'-lm','-o',str(out/'control.so')],check=True)
