"""New 40-scene paired confirmation at the user's measured box dimensions."""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')
from scripts.run_submission_confirmation import isolate
sys.addaudithook(isolate)
from pathlib import Path
import argparse,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from scripts import run_paper_box as prior
from scripts import run_transition_gate as io
from reflex.submission_paper_task import PaperBoxEnv

ROOT=Path('runs/submission_paper_20261102_v2')
DT=.0000625
def bank(split,n):
    base=99110250000 if split=='test' else 99110240000
    return [dict(id=f'measured_paper_{split}_{i:03d}',seed=base+i,options=dict(material='paper_box',
        condition=['nominal','position','friction','compliance'][i%4],sensor_condition='degraded' if (i//4)%2 else 'nominal')) for i in range(n)]
def job(scene,mode,dt,stage):
    prior.PaperBoxEnv=PaperBoxEnv
    return prior.job(str(ROOT/stage),scene,mode,dt)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','numerics','test']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.phase=='prepare':
        ROOT.mkdir(exist_ok=False)
        sources={str(p):io.sha(p) for folder in ['reflex','scripts','build/convergence_native','assets/sim_official_b601_rs']
            for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.c','.so','.xml','.json')}
        io.write(ROOT/'plan.json',dict(test=bank('test',40),numerical=bank('numerical',8),dimensions_mm=[41,31,37],
          modes=['baseline','gentle','gentle_recolored'],dt_s=DT,ladder_s=[DT,DT/2,DT/4],source_sha256=sources,
          mass_and_contact_assumptions_not_measured=True,old_test_reused=False,hardware_motor_commands=0,
          task_gate={'min_success_rate':.90,'overclamp':0,'abort':0,'max_native_per_finger_force_N':3.,'max_native_overbudget_s':0.},
          comparative_gate='Success not lower; per-scene force peaks and overall max/P95 not increased',
          numerical_gate={'max_delta_N':1.,'p95_delta_N':.5,'labels_unchanged':True}))
        print('Prepared measured paper confirmation',flush=True);return
    plan=io.read(ROOT/'plan.json')
    for p,h in plan['source_sha256'].items():assert io.sha(p)==h,p
    stage=ROOT/a.phase;stage.mkdir(exist_ok=False)
    if a.phase=='test':
        numeric=io.read(ROOT/'numerics/complete.json');rows=numeric['rows'];checks=[]
        import numpy as np
        for h,k in zip(plan['ladder_s'][:-1],plan['ladder_s'][1:]):
            aa={r['scene_id']:r for r in rows if r['dt_s']==h};bb={r['scene_id']:r for r in rows if r['dt_s']==k}
            diffs=[abs(aa[i]['peak_N']-bb[i]['peak_N']) for i in aa]
            changed=[i for i in aa if any(aa[i][x]!=bb[i][x] for x in ['success','overclamp','abort'])]
            checks.append(dict(coarse_s=h,fine_s=k,max_delta_N=max(diffs),p95_delta_N=float(np.quantile(diffs,.95)),label_changes=changed,
                passed=bool(not changed and max(diffs)<=1 and np.quantile(diffs,.95)<=.5)))
        io.write(ROOT/'numerical_gate.json',checks)
        if not all(x['passed'] for x in checks):raise RuntimeError('Original numerical gate failed; no test promotion')
    scenes=plan['test'] if a.phase=='test' else plan['numerical'];modes=plan['modes'] if a.phase=='test' else ['gentle']
    steps=[DT] if a.phase=='test' else plan['ladder_s'];jobs=[(s,m,h,a.phase) for s in scenes for m in modes for h in steps]
    io.write(stage/'declaration.json',dict(scenes=scenes,modes=modes,steps=steps,expected=len(jobs),before_unix=time.time()))
    rows=[];errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(job,*x):x for x in jobs}
        for f in as_completed(fs):
            try:rows.append(f.result())
            except Exception as e:errors.append(dict(job=fs[f],error=str(e),traceback=traceback.format_exc()))
            print(io.canonical(dict(phase=a.phase,done=len(rows),errors=len(errors),expected=len(jobs),elapsed_s=time.time()-start)),flush=True)
    io.write(stage/('incomplete.json' if errors else 'complete.json'),dict(rows=rows,errors=errors,elapsed_s=time.time()-start,physical_motor_commands=0))
    if errors:raise RuntimeError('Simulation failures retained')
if __name__=='__main__':main()
