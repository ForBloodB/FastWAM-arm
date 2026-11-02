#include <math.h>
#include <mujoco/mujoco.h>
/* The frozen PD, gravity compensation, damping, force caps and contact solver
 * are copied exactly. Only the arm REFERENCE polynomial can be changed.
 * Extra native statistics are evaluation outputs and never control inputs. */
int lift_validation_roll(const mjModel* m,mjData* d,const double* a,const double* b,
 double ga,double gb,double cap,int start,int total,int n,int mode,
 double fa,double fb,int shape,int object_geom,int left_body,int right_body,
 int table_body,double budget,double* stats) {
 const double kp[6]={180,220,180,90,55,30},kd[6]={18,22,18,7,4,2};
 for(int k=0;k<n;k++) {
  double u=(double)(start+k+1)/total;
  double s=shape?u*u*u*(10+u*(-15+6*u)):u*u*(3-2*u);
  if(shape==2)s=u*u*u*u*(35+u*(-84+u*(70-20*u)));
  for(int i=0;i<6;i++) {
   double torque=kp[i]*(a[i]+s*(b[i]-a[i])-d->qpos[i])-kd[i]*d->qvel[i]+d->qfrc_bias[i];
   d->ctrl[i]=fmax(m->actuator_ctrlrange[2*i],fmin(m->actuator_ctrlrange[2*i+1],torque));
  }
  double v=(double)(k+1)/n,z=v*v*(3-2*v);
  double force=mode?-(fa+z*(fb-fa))-18*d->qvel[6]:1800*(ga+z*(gb-ga)-d->qpos[6])-18*d->qvel[6];
  if(mode && d->qpos[6]<ga-.001 && d->qvel[6]<-.030)force=fmax(force,-18*(d->qvel[6]+.030));
  d->ctrl[6]=fmax(-cap,fmin(cap,force));
  mj_step(m,d);
  for(int i=0;i<m->nq;i++)if(!isfinite(d->qpos[i]))return 1;
  for(int i=0;i<m->nv;i++)if(!isfinite(d->qvel[i])||fabs(d->qvel[i])>100)return 2;
  double left=0,right=0,f[6];
  for(int j=0;j<d->ncon;j++) {
   const mjContact* c=d->contact+j;
   if(c->geom[0]!=object_geom&&c->geom[1]!=object_geom)continue;
   int other=c->geom[0]==object_geom?c->geom[1]:c->geom[0],body=m->geom_bodyid[other];
   mj_contactForce(m,d,j,f);
   if(body==left_body)left+=f[0];
   if(body==right_body)right+=f[0];
   if(body==left_body||body==right_body)stats[3]=fmax(stats[3],-c->dist);
   if(body==table_body)stats[4]=fmax(stats[4],-c->dist);
  }
  stats[0]=fmax(stats[0],left);stats[1]=fmax(stats[1],right);
  if(fmax(left,right)>budget)stats[2]+=m->opt.timestep;
  stats[5]+=d->ctrl[6]/n;stats[6]=fmax(stats[6],fabs(d->ctrl[6]));
  stats[7]+=fmax(left,right)*m->opt.timestep;
  stats[8]+=fmax(0.,fmax(left,right)-budget)*m->opt.timestep;
  for(int i=0;i<6;i++){
   stats[9]=fmax(stats[9],fabs(d->qvel[i]));
   stats[10]=fmax(stats[10],fabs(d->qacc[i]));
  }
  stats[11]+=1;
 }
 return 0;
}
