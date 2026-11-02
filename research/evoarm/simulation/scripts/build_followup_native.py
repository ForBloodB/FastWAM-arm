"""Isolated native actuator recording/replay and fixed physical PD clock."""
from pathlib import Path
import subprocess,mujoco
s=Path('build/impulse_audit/impulse_control.c').read_text()
s=s.replace('int impulse_audit_roll(', 'int followup_roll(')
header='''
#include <stdint.h>
static int experiment_mode=0,stride=1,used=0,capacity=0;
static long long cursor=0;
static double *record_buffer=0,*replay_buffer=0,base_dt=0.0000625;
static int replay_count=0;
void configure(int mode,int control_stride,double *record,int cap,double *replay,int count){
 experiment_mode=mode;stride=control_stride;record_buffer=record;capacity=cap;
 replay_buffer=replay;replay_count=count;cursor=0;used=0;
}
int recorded_count(){return used;}
'''
s=s.replace('#include <math.h>','#include <math.h>\n'+header)
s=s.replace('  double u=(double)(start+k+1)/total;', '''  if(experiment_mode==2){
   long long index=cursor/stride;
   if(index>=replay_count)return 8;
   for(int i=0;i<7;i++)d->ctrl[i]=replay_buffer[7*index+i];
  }else if(experiment_mode==0 || cursor%stride==0){
  int advance=experiment_mode==1?stride:1;
  double u=(double)(start+k+advance)/total;''')
s=s.replace('double v=(double)(k+1)/n', 'double v=(double)(k+advance)/n')
s=s.replace('  double before[3];', '''  }
  if(record_buffer && cursor%stride==0){
   if(used>=capacity)return 9;
   for(int i=0;i<7;i++)record_buffer[7*used+i]=d->ctrl[i];
   used++;
  }
  cursor++;
  double before[3];''')
out=Path('build/followup_native');out.mkdir(parents=True,exist_ok=True)
(out/'control.c').write_text(s)
r=Path(mujoco.__file__).parent
subprocess.run(['gcc','-O3','-shared','-fPIC',str(out/'control.c'),'-I'+str(r/'include'),str(next(r.glob('libmujoco.so*'))),'-Wl,-rpath,'+str(r),'-lm','-o',str(out/'control.so')],check=True)
