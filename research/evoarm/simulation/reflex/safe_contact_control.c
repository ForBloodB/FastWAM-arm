#include <math.h>
#include <mujoco/mujoco.h>
/* Joint controller identical to official_control.c. Gripper references have a
 * separate local clock so a 100 Hz admittance loop cannot restart arm motion.
 * Object/budget arguments below are diagnostic only, after mj_step. */
int safe_contact_roll(const mjModel* m,mjData* d,const double* a,const double* b,
 double ga,double gb,double cap,int start,int total,int n,
 int object_geom,int left_body,int right_body,int table_body,double scoring_budget,double* stats) {
 const double kp[6]={180,220,180,90,55,30},kd[6]={18,22,18,7,4,2};
 for(int k=0;k<n;k++) {
  double u=(double)(start+k+1)/total,s=u*u*(3-2*u);
  for(int i=0;i<6;i++) {
   double torque=kp[i]*(a[i]+s*(b[i]-a[i])-d->qpos[i])-kd[i]*d->qvel[i]+d->qfrc_bias[i];
   d->ctrl[i]=fmax(m->actuator_ctrlrange[2*i],fmin(m->actuator_ctrlrange[2*i+1],torque));
  }
  double v=(double)(k+1)/n,z=v*v*(3-2*v);
  double force=1800*(ga+z*(gb-ga)-d->qpos[6])-18*d->qvel[6];
  d->ctrl[6]=fmax(-cap,fmin(cap,force));
  mj_step(m,d);
  for(int i=0;i<m->nq;i++)if(!isfinite(d->qpos[i]))return 1;
  for(int i=0;i<m->nv;i++)if(!isfinite(d->qvel[i])||fabs(d->qvel[i])>100)return 2;
  /* These observations never feed the controller above. */
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
  if(fmax(left,right)>scoring_budget)stats[2]+=m->opt.timestep;
 }
 return 0;
}
