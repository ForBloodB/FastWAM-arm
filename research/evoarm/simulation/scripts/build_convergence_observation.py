"""Native 10ms force observation integration, without filtering risk peaks."""
from pathlib import Path
import subprocess,mujoco
s=Path('build/convergence_native/control.c').read_text()
s=s.replace('-18*d->qvel[6]','-((kind==13 || kind==14)?4.0:18.0)*d->qvel[6]')
s=s.replace('  stats[0]=fmax(stats[0],left);stats[1]=fmax(stats[1],right);','''
  stats[41]+=left*m->opt.timestep;stats[42]+=right*m->opt.timestep;
  stats[0]=fmax(stats[0],left);stats[1]=fmax(stats[1],right);''')
s=s.replace('  mjcb_control=previous;cursor++;','''  mjcb_control=previous;cursor++;
  if(fabs(d->actuator_force[6])>cap+1e-9)return 11;''')
out=Path('build/convergence_observation');out.mkdir(parents=True,exist_ok=True);(out/'control.c').write_text(s)
r=Path(mujoco.__file__).parent
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'control.c'),'-I'+str(r/'include'),str(next(r.glob('libmujoco.so*'))),'-Wl,-rpath,'+str(r),'-lm','-o',str(out/'control.so')],check=True)
