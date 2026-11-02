"""Piecewise-consistent affine damping; all applied forces remain clipped."""
from pathlib import Path
import subprocess,mujoco
s=Path('build/convergence_native/control.c').read_text()
s=s.replace('   d->ctrl[i]=fmax(m->actuator_ctrlrange[2*i],fmin(m->actuator_ctrlrange[2*i+1],torque));', '''
   double lo=m->actuator_ctrlrange[2*i],hi=m->actuator_ctrlrange[2*i+1];
   double clipped=fmax(lo,fmin(hi,torque));
   if(kind==5){
    mjModel *w=(mjModel*)m;
    int interior=torque>lo && torque<hi;
    w->actuator_biasprm[mjNBIAS*i+2]=interior?-kd[i]:0;
    d->ctrl[i]=interior?torque+kd[i]*d->qvel[i]:clipped;
   }else d->ctrl[i]=clipped;''')
s=s.replace('  d->ctrl[6]=fmax(-cap,fmin(cap,force));', '''
  if(kind==4 || kind==5){
   mjModel *w=(mjModel*)m;int interior=force>-cap && force<cap;
   w->actuator_biasprm[mjNBIAS*6+2]=interior?-18:0;
   d->ctrl[6]=interior?force+18*d->qvel[6]:fmax(-cap,fmin(cap,force));
  }else d->ctrl[6]=fmax(-cap,fmin(cap,force));''')
s=s.replace('stats[5]+=d->ctrl[6]/n;stats[6]=fmax(stats[6],fabs(d->ctrl[6]));', '''
  if(fabs(d->actuator_force[6])>cap+1e-9)return 11;
  stats[5]+=d->actuator_force[6]/n;stats[6]=fmax(stats[6],fabs(d->actuator_force[6]));''')
out=Path('build/convergence_implicit');out.mkdir(parents=True,exist_ok=True);(out/'control.c').write_text(s)
r=Path(mujoco.__file__).parent
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'control.c'),'-I'+str(r/'include'),str(next(r.glob('libmujoco.so*'))),'-Wl,-rpath,'+str(r),'-lm','-o',str(out/'control.so')],check=True)
