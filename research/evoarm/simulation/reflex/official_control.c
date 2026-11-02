#include <math.h>
#include <mujoco/mujoco.h>
/* Native version of the already validated PD + model-bias controller.
   No state writes, object forces, networking or model mutation. */
int reflex_roll(const mjModel* m, mjData* d, const double* a, const double* b,
                double ga, double gb, double cap, int start, int total, int n) {
  const double kp[6]={180,220,180,90,55,30},kd[6]={18,22,18,7,4,2};
  for(int k=0;k<n;k++) {
    double u=(double)(start+k+1)/total, s=u*u*(3-2*u);
    for(int i=0;i<6;i++) {
      double target=a[i]+s*(b[i]-a[i]);
      double torque=kp[i]*(target-d->qpos[i])-kd[i]*d->qvel[i]+d->qfrc_bias[i];
      double lo=m->actuator_ctrlrange[2*i],hi=m->actuator_ctrlrange[2*i+1];
      d->ctrl[i]=fmax(lo,fmin(hi,torque));
    }
    double force=1800*(ga+s*(gb-ga)-d->qpos[6])-18*d->qvel[6];
    d->ctrl[6]=fmax(-cap,fmin(cap,force));
    mj_step(m,d);
    for(int i=0;i<m->nq;i++) if(!isfinite(d->qpos[i]))return 1;
    for(int i=0;i<m->nv;i++) if(!isfinite(d->qvel[i]) || fabs(d->qvel[i])>100)return 2;
  }
  return 0;
}
