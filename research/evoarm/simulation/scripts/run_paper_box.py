"""Prospective paper-box-only closed-loop validation; never accesses robot buses."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,gzip,hashlib,json,time,traceback
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from reflex.paper_box_task import PaperBoxEnv,MODES
from scripts import run_transition_gate as io

ROOT=Path('runs/paper_box_20261101')
DT=.0000625
LADDER=[DT,DT/2,DT/4]

def scenes(split,n):
    base=2611012000 if split=='development' else 2611013000
    return [dict(id=f'paper_box_{split}_{i:03d}',seed=base+i,
        options=dict(material='paper_box',condition=['nominal','position','friction','compliance'][i%4],
            sensor_condition='degraded' if (i//4)%2 else 'nominal')) for i in range(n)]

def prepare():
    ROOT.mkdir(exist_ok=False)
    names=sorted({str(p) for folder in ['reflex','scripts','build/convergence_native','assets/sim_official_b601_rs']
        for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.c','.so','.xml','.json')})
    hashes={name:io.sha(name) for name in names}
    for name in names:
        dst=ROOT/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(name).read_bytes())
    plan=dict(schema='paper_box_v1',created_unix=time.time(),development=scenes('development',12),test=scenes('test',40),
        numerical_indices=list(range(8)),modes=list(MODES),dt_s=DT,ladder_s=LADDER,source_sha256=hashes,
        task_success='Unchanged inherited bilateral contact + lift + no slip + placement and settling; no abort or overclamp',
        task_gate=dict(min_success_rate=.90,overclamp=0,abort=0,max_native_per_finger_force_N=3.,max_native_overbudget_s=0.),
        numerical_gate=dict(max_peak_difference_N=1.,p95_peak_difference_N=.5,unchanged_labels=True),
        comparative_gate='Gentle >= baseline success, no increase in native max/P95 or risk; separate from absolute task acceptance',
        old_60_scene_test_reused=False,old_foam_test_reused=False,
        paper_box_assumptions=dict(size_mm=[[40,48],[29,37],[38,44]],wall_thickness_mm=[.25,.45],shell_density_kg_m3=[500,700],normal_friction=[.25,.6],low_friction=[.12,.22],contact_tc_s=[.003,.006],baseline_initial_drive_N=2.5,gentle_initial_drive_N=1.5),training_or_online_update=False,hardware_motor_commands=0,
        limits='Known empty closed paper box, shell inertia and rigid outer collisions. Assumed dimensions, thickness, density and friction; not calibrated cardboard or damage prediction.')
    io.write(ROOT/'plan.json',plan);print(json.dumps({'prepared':str(ROOT),'source_files':len(names)}))

def job(folder,scene,mode,dt):
    start=time.time();env=PaperBoxEnv(mode=mode,dt=dt)
    try:
        env.reset(seed=scene['seed'],options=deepcopy(scene['options']));r=env.run()
        assert r['hardware_motor_commands']==r['object_pose_writes_during_run']==0
        assert r['control_bounds']['max_cap_N']<=28.+1e-12
        assert r['physics']['force_budget_N_per_finger']==3.
        r.update(scene=scene,perception_mode=mode,dt_s=dt,wall_job_s=time.time()-start)
        token=scene['id']+'__'+mode+'__'+str(round(dt*1e9));out=Path(folder)/'trials';out.mkdir(exist_ok=True)
        trace={k:deepcopy(getattr(env,k,[])) for k in ('diagnostic_log','impulse_log','force_chunks','control_log','timing_log',
            'action_events','repair_events','residual_move_commands','vision_calls','transition_log','force_cap_log')}
        trace.update(final_qpos=env.data.qpos.tolist(),final_qvel=env.data.qvel.tolist())
        path=out/(token+'.trace.json.gz')
        with path.open('xb') as f:f.write(gzip.compress(io.canonical(trace).encode(),compresslevel=2,mtime=0))
        for k in ('diagnostic_log','events','timing_log','vision_calls','decisions','transition_log','residual_move_commands'):r.pop(k,None)
        r.update(trace_file=path.relative_to(folder).as_posix(),trace_sha256=io.sha(path));io.write(out/(token+'.json'),r)
        return dict(scene_id=scene['id'],mode=mode,dt_s=dt,success=r['success'],failure=r['failure_type'],
            overclamp=r['overclamp'],abort=r['safety_abort'],peak_N=r['fullrate']['peak_force_N'],
            overbudget_s=r['fullrate']['overforce_s'],duration_s=r['duration_s'],
            initial_detection_valid=bool(trace['vision_calls'] and trace['vision_calls'][0]['valid']),
            common_start_digest=r['common_start_digest'],wall_job_s=time.time()-start,trial_file=str(out/(token+'.json')))
    finally:env.close()

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','development','test','numerics']);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--root',default=str(ROOT));a=p.parse_args();ROOT=Path(a.root)
    if a.stage=='prepare':return prepare()
    plan=io.read(ROOT/'plan.json')
    for name,digest in plan['source_sha256'].items():
        if io.sha(name)!=digest:raise RuntimeError('Source changed after declaration: '+name)
    folder=ROOT/a.stage;folder.mkdir(exist_ok=False)
    bank=plan['development'][:4] if a.stage=='pilot' else plan['development'] if a.stage=='development' else plan['test'] if a.stage=='test' else [plan['development'][i] for i in plan['numerical_indices']]
    modes=list(MODES) if a.stage!='numerics' else ['gentle']
    steps=[DT] if a.stage!='numerics' else LADDER
    jobs=[(s,m,h) for s in bank for m in modes for h in steps]
    io.write(folder/'declaration.json',dict(stage=a.stage,scenes=bank,modes=modes,steps=steps,expected=len(jobs)))
    rows=[];errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(job,str(folder),s,m,h):(s['id'],m,h) for s,m,h in jobs}
        for f in as_completed(fs):
            try:rows.append(f.result())
            except Exception as exc:errors.append(dict(job=fs[f],error=str(exc),traceback=traceback.format_exc()))
            print(json.dumps(dict(done=len(rows),errors=len(errors),expected=len(jobs),elapsed_s=time.time()-start,latest=rows[-1] if rows else None)),flush=True)
    io.write(folder/('incomplete.json' if errors else 'complete.json'),dict(rows=rows,errors=errors,motor_commands=0,elapsed_s=time.time()-start))
    if errors:raise RuntimeError('Incomplete experiment; all errors retained')

if __name__=='__main__':main()
