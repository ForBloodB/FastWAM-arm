"""Portable, simulation-only entry point. Linux x86-64, MuJoCo 3.3.7."""
import os,sys,json,argparse,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')

def build():
 import mujoco
 assert mujoco.__version__=='3.3.7','Use the pinned solver version'
 pkg=Path(mujoco.__file__).parent;lib=next(pkg.glob('libmujoco.so*'))
 pairs={'reflex/official_control.c':'build/official_control/libreflex_control.so','reflex/safe_contact_control.c':'build/safe_contact/libsafe_contact.so','reflex/slip_control.c':'build/slip_control/libslip_control.so','reflex/drive_preload_control.c':'build/drive_preload/libdrive_preload.so','build/impulse_audit/impulse_control.c':'build/impulse_audit/libimpulse_audit.so'}
 pairs.update({str(p):str(p.with_suffix('.so')) for p in Path('build').glob('*/control.c')})
 for src,dst in pairs.items():
  Path(dst).parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['gcc','-O2' if 'official_control' in src else '-O3','-shared','-fPIC',src,'-I'+str(pkg/'include'),str(lib),'-Wl,-rpath,'+str(pkg),'-lm','-o',dst],check=True)
 if not Path('build/contact_state/official_pairs6.mjb').exists():subprocess.run([sys.executable,'-m','scripts.build_contact_pair_cache','--dim','6'],check=True)
 print('Native source and official model compiled')

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','smoke','train','adapt','test']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.phase=='build':build();return
 from scripts.run_submission_confirmation import isolate
 sys.addaudithook(isolate)
 from scripts import run_submission_confirmation as c
 from scripts import run_transition_gate as io
 c.ROOT=Path('runs/reproduced_confirmation');c.configure()
 original=Path('runs/submission_confirmation_20261102')
 if a.phase=='smoke':
  from scripts import run_paper_box as paper
  from reflex.submission_paper_task import PaperBoxEnv
  paper.PaperBoxEnv=PaperBoxEnv
  s=io.read('runs/submission_paper_20261102_v2/plan.json')['test'][0]
  out=Path('runs/reproduced_paper');out.mkdir(exist_ok=False)
  row=paper.job(str(out),s,'gentle',.0000625)
  old=io.read('runs/submission_paper_20261102_v2/test_recovered/complete.json')
  expected=next(r for r in old['rows'] if r['scene_id']==s['id'] and r['mode']=='gentle')
  checks={k:row[k]==expected[k] for k in ['success','overclamp','abort','common_start_digest']}
  checks['peak_reproduced']=abs(row['peak_N']-expected['peak_N'])<1e-6
  io.write(out/'verification.json',dict(checks=checks,actual=row,expected=expected,hardware_motor_commands=0))
  assert all(checks.values()),checks
  print(json.dumps(checks));return
 if a.phase=='train':
  c.ROOT.mkdir(exist_ok=False)
  for f in ['shield.json','training_data.json']:shutil.copy2(original/f,c.ROOT/f)
  plan=io.read(original/'plan.json');plan['source_sha256']={str(p):io.sha(p) for folder in ['reflex','scripts'] for p in Path(folder).rglob('*.py')}
  plan['portable_source_changes']='Endpoint defaults anonymized; JSON numpy-bool metadata fix. Dynamics, source replay, shield and scene definitions unchanged.'
  io.write(c.ROOT/'plan.json',plan)
  data=io.read(c.ROOT/'training_data.json');training=[]
  class Zero:
   def digest(self):return 'B'
   def act(self,*a,**k):return 'zero'
  base=c.guard_metrics(Zero(),data['guard'])
  for seed in c.SEEDS:
   model=c.ResidualBandit(c.ACTIONS,seed=seed,epsilon=.03);curve=[];previous=0
   for step in [100,400,1200,2400]:
    fit=model.train_offline(data['records'],steps=step-previous);previous=step
    metrics=c.guard_metrics(model,data['guard']);curve.append(dict(step=step,fit=fit,adaptive_monitor_metrics=metrics))
   checks=c.acceptance(metrics,base,base,min_utility_gain=0.)
   io.checkpoint(c.ROOT/f'models/frozen_{seed}.json',model,source_only=True)
   old=c.runner.load(original/f'models/frozen_{seed}.json')
   assert old.model_digest()==model.model_digest(),'Training weights did not reproduce'
   training.append(dict(seed=seed,curve=curve,monitor_checks=checks,monitor_passed=all(checks.values()),model_digest_matches_published=True))
  io.write(c.ROOT/'training.json',dict(results=training,learning_curves_are_development_not_independent=True))
  print('All three frozen model digests match published evidence');return
 sys.argv=[sys.argv[0],a.phase,'--workers',str(a.workers)];c.main()

if __name__=='__main__':main()
