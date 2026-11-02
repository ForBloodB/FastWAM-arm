#include <math.h>
#include <mujoco/mujoco.h>
/* Same arm controller as official_control.c. During grasp retention only,
   replace the gripper position spring with a bounded closing-force command.
   Object state and material parameters are never read or written. */
int reflex_force_roll(const mjModel* m, mjData* d, const double* a, const double* b,
                      double cap, int start, int total, int n) {
  const double kp[6]={180,220,180,90,55,30}, kd[6]={18,22,18,7,4,2};
  for (int k=0; k<n; k++) {
    double u=(double)(start+k+1)/total, s=u*u*(3-2*u);
    for (int i=0;i<6;i++) {
      double target=a[i]+s*(b[i]-a[i]);
      double torque=kp[i]*(target-d->qpos[i])-kd[i]*d->qvel[i]+d->qfrc_bias[i];
      d->ctrl[i]=fmax(m->actuator_ctrlrange[2*i],fmin(m->actuator_ctrlrange[2*i+1],torque));
    }
    d->ctrl[6]=fmax(-cap,fmin(cap,-cap-18*d->qvel[6]));
    mj_step(m,d);
    for(int i=0;i<m->nq;i++) if(!isfinite(d->qpos[i]))return 1;
    for(int i=0;i<m->nv;i++) if(!isfinite(d->qvel[i]) || fabs(d->qvel[i])>100)return 2;
  }
  return 0;
}
