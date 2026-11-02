"""Current-stage control callback; original limits and native force audit."""
from pathlib import Path
import subprocess,mujoco
s=Path('build/impulse_audit/impulse_control.c').read_text()
s=s.replace('int impulse_audit_roll(', 'int convergence_roll(')
begin=s.index('  double u=');end=s.index('  double before[3];',begin)
body=s[begin:end]
context='''
static int kind=0,stride=1,iteration=0;
static long long cursor=0;
static const double *ca,*cb;
static double cga,cgb,ccap,cfa,cfb;
static int cstart,ctotal,cn,cmode,cquintic;
void convergence_configure(int k,int st){kind=k;stride=st;cursor=0;}
static void current_control(const mjModel *m,mjData *d){
 const double kp[6]={180,220,180,90,55,30},kd[6]={18,22,18,7,4,2};
 if(kind==3 && cursor%stride!=0)return;
 const double *a=ca,*b=cb;
 double ga=cga,gb=cgb,cap=ccap,fa=cfa,fb=cfb;
 int start=cstart,total=ctotal,n=cn,mode=cmode,quintic=cquintic,k=iteration;
 CONTROL_BODY
}
'''
context=context.replace('CONTROL_BODY',body.replace('(start+k+1)','(start+k+(kind==1?1:0))').replace('(k+1)','(k+(kind==1?1:0))'))
s=s[:begin]+'''  if(kind==0){
'''+body+'''  }else{
   ca=a;cb=b;cga=ga;cgb=gb;ccap=cap;cfa=fa;cfb=fb;
   cstart=start;ctotal=total;cn=n;cmode=mode;cquintic=quintic;iteration=k;
  }
'''+s[end:]
s=s.replace('  mj_step(m,d);', '''  mjfGeneric previous=mjcb_control;
  if(kind)mjcb_control=current_control;
  mj_step(m,d);
  mjcb_control=previous;cursor++;
''')
s=s.replace('#include <mujoco/mujoco.h>','#include <mujoco/mujoco.h>\n'+context)
out=Path('build/convergence_native');out.mkdir(parents=True,exist_ok=True)
(out/'control.c').write_text(s)
r=Path(mujoco.__file__).parent
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'control.c'),'-I'+str(r/'include'),str(next(r.glob('libmujoco.so*'))),'-Wl,-rpath,'+str(r),'-lm','-o',str(out/'control.so')],check=True)
