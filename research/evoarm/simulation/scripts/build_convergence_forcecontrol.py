"""Separate bounded force feedforward from free-space gap damping."""
from pathlib import Path
import subprocess,mujoco
s=Path('build/convergence_native/control.c').read_text()
old='double force=mode?-(fa+z*(fb-fa))-18*d->qvel[6]:1800*(ga+z*(gb-ga)-d->qpos[6])-18*d->qvel[6];'
new='double force=mode?-(fa+z*(fb-fa))-(kind==11?0.0:18.0)*d->qvel[6]:1800*(ga+z*(gb-ga)-d->qpos[6])-(kind==11?4.0:18.0)*d->qvel[6];'
assert s.count(old)==2;s=s.replace(old,new)
s=s.replace('  mjcb_control=previous;cursor++;','''  mjcb_control=previous;cursor++;
  if(fabs(d->actuator_force[6])>cap+1e-9)return 11;''')
out=Path('build/convergence_forcecontrol');out.mkdir(parents=True,exist_ok=True);(out/'control.c').write_text(s)
r=Path(mujoco.__file__).parent
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'control.c'),'-I'+str(r/'include'),str(next(r.glob('libmujoco.so*'))),'-Wl,-rpath,'+str(r),'-lm','-o',str(out/'control.so')],check=True)
