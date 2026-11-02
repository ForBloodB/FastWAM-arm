"""Continue after a metadata-only numpy.bool_ JSON error; original physics untouched."""
from scripts.run_submission_paper import ROOT,DT,job
from scripts import run_transition_gate as io
import numpy as np,time,traceback,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed

def main():
    plan=io.read(ROOT/'plan.json')
    for p,h in plan['source_sha256'].items():assert io.sha(p)==h,p
    rows=io.read(ROOT/'numerics/complete.json')['rows'];checks=[]
    for h,k in zip(plan['ladder_s'][:-1],plan['ladder_s'][1:]):
        aa={r['scene_id']:r for r in rows if r['dt_s']==h};bb={r['scene_id']:r for r in rows if r['dt_s']==k}
        assert len(aa)==len(bb)==8
        diffs=[abs(aa[i]['peak_N']-bb[i]['peak_N']) for i in aa]
        changed=[i for i in aa if any(aa[i][x]!=bb[i][x] for x in ['success','overclamp','abort'])]
        checks.append(dict(coarse_s=h,fine_s=k,max_delta_N=float(max(diffs)),p95_delta_N=float(np.quantile(diffs,.95)),label_changes=changed,
            passed=bool(not changed and max(diffs)<=1 and np.quantile(diffs,.95)<=.5)))
    io.write(ROOT/'numerical_gate_recovered.json',dict(checks=checks,reason='Convert numpy.bool_ to JSON bool only; original trajectories preserved',source_sha256=io.sha(__file__)))
    assert all(x['passed'] for x in checks),checks
    stage=ROOT/'test_recovered';stage.mkdir(exist_ok=False)
    jobs=[(s,m,DT,'test_recovered') for s in plan['test'] for m in plan['modes']]
    io.write(stage/'declaration.json',dict(expected=len(jobs),before_unix=time.time(),plan_sha256=io.sha(ROOT/'plan.json')))
    results=[];errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(job,*x):x for x in jobs}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as e:errors.append(dict(job=fs[f],error=str(e),traceback=traceback.format_exc()))
            print(io.canonical(dict(done=len(results),errors=len(errors),expected=len(jobs))),flush=True)
    io.write(stage/('incomplete.json' if errors else 'complete.json'),dict(rows=results,errors=errors,elapsed_s=time.time()-start,physical_motor_commands=0))
    assert not errors,errors

if __name__=='__main__':main()
