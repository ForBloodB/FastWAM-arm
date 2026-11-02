"""Build evaluation instrumentation around the frozen native controller."""
from pathlib import Path
import subprocess
import mujoco
p=Path('reflex/drive_preload_control.c');src=p.read_text()
src=src.replace('int drive_preload_roll(', 'int impulse_audit_roll(')
src=src.replace(' const double kp[6]', ''' int ob=m->geom_bodyid[object_geom],od=m->body_dofadr[ob];
 double mass=m->body_mass[ob],v0[3];
 for(int i=0;i<3;i++){v0[i]=d->qvel[od+i];stats[35+i]=v0[i];}
 const double kp[6]''')
src=src.replace('double s=quintic?u*u*u*(10+u*(-15+6*u)):u*u*(3-2*u);',
                'double s=quintic==2?u:(quintic?u*u*u*(10+u*(-15+6*u)):u*u*(3-2*u));')
src=src.replace('  mj_step(m,d);','''  double before[3];for(int i=0;i<3;i++)before[i]=d->qvel[od+i];
  mj_step(m,d);
  double net[3]={0,0,0},finger[3]={0,0,0},tablef[3]={0,0,0},otherf[3]={0,0,0};
  for(int j=0;j<d->ncon;j++){
   const mjContact* c=d->contact+j;
   if(c->geom[0]!=object_geom && c->geom[1]!=object_geom)continue;
   double local[6],sign=c->geom[0]==object_geom?-1:1;
   int other=c->geom[0]==object_geom?c->geom[1]:c->geom[0],body=m->geom_bodyid[other];
   mj_contactForce(m,d,j,local);
   for(int i=0;i<3;i++){
    double f=sign*(c->frame[i]*local[0]+c->frame[3+i]*local[1]+c->frame[6+i]*local[2]);
    net[i]+=f;
    if(body==left_body||body==right_body)finger[i]+=f;
    else if(body==table_body)tablef[i]+=f;else otherf[i]+=f;
   }
  }
  double error=0,step_error=0;
  for(int i=0;i<3;i++){
   double dt=m->opt.timestep;
   stats[7+i]+=finger[i]*dt;stats[10+i]+=tablef[i]*dt;stats[13+i]+=otherf[i]*dt;
   stats[16+i]+=d->qfrc_smooth[od+i]*dt;
   stats[19+i]+=d->qfrc_constraint[od+i]*dt;stats[22+i]+=net[i]*dt;
   stats[25+i]+=mass*(d->qvel[od+i]-before[i]);stats[28+i]+=mass*d->qacc[od+i]*dt;
   error+=(net[i]-d->qfrc_constraint[od+i])*(net[i]-d->qfrc_constraint[od+i]);
   double r=mass*(d->qvel[od+i]-before[i])-(net[i]+d->qfrc_smooth[od+i])*dt;
   step_error+=r*r;stats[38+i]=d->qvel[od+i];
  }
  stats[31]=fmax(stats[31],sqrt(error));stats[32]=fmax(stats[32],sqrt(step_error));
  stats[33]+=1;stats[34]+=m->opt.timestep;''')
out=Path('build/impulse_audit');out.mkdir(parents=True,exist_ok=True)
(out/'impulse_control.c').write_text(src)
root=Path(mujoco.__file__).parent;lib=next(root.glob('libmujoco.so*'))
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'impulse_control.c'),'-I'+str(root/'include'),str(lib),'-Wl,-rpath,'+str(root),'-lm','-o',str(out/'libimpulse_audit.so')],check=True)
