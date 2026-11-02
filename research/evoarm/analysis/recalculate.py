import sys,json,math,csv,collections
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
def summary(rr):
 k=sum(r['success'] for r in rr);f=[r['peak_N'] for r in rr]
 return dict(n=len(rr),success=k,rate=k/len(rr),wilson95=wilson(k,len(rr)),overclamp=sum(r['overclamp'] for r in rr),abort=sum(r['abort'] for r in rr),peak_max_N=max(f),peak_p95_N=float(np.quantile(f,.95)),overbudget_s=sum(r['overbudget_s'] for r in rr))
def paired(a,b):
 a={r['scene_id']:r for r in a};b={r['scene_id']:r for r in b};assert a.keys()==b.keys()
 d=np.array([int(b[i]['success'])-int(a[i]['success']) for i in sorted(a)]);f=np.array([b[i]['peak_N']-a[i]['peak_N'] for i in sorted(a)])
 rng=np.random.default_rng(991102);boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
 return dict(n=len(a),rescued=int(sum(d==1)),harmed=int(sum(d==-1)),success_delta=float(d.mean()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),peak_delta_max_N=float(f.max()),peak_delta_mean_N=float(f.mean()),peak_increase_cases=int(sum(f>1e-10)),all_paired_peaks_not_increased=bool(np.all(f<=1e-10)))
rl=read(ROOT/'simulation/runs/submission_confirmation_20261102/test/complete.json');rows=[]
for e in rl['results']:
 for r in e['rows']:
  rows.append(dict(scene_id=r['scene_id'],mode=r['mode'],learner_seed=r['learner_seed'],material=r['physics']['material'],action=r['executed'],success=r['success'],overclamp=r['overclamp'],abort=r['safety_abort'],peak_N=r['fullrate']['peak_force_N'],overbudget_s=r['fullrate']['overforce_s'],budget_N=r['physics']['force_budget_N_per_finger'],duration_s=r['duration_s'],failure=r['failure_type']))
groups=collections.defaultdict(list)
for r in rows:groups[r['mode']+(f"_{r['learner_seed']}" if r['learner_seed'] else '')].append(r)
stats={g:dict(overall=summary(rr),low_friction_metal=summary([r for r in rr if r['material']=='metal']),action_counts=dict(collections.Counter(r['action'] for r in rr)),failure_counts=dict(collections.Counter(r['failure'] for r in rr))) for g,rr in groups.items()}
pairs={g:paired(groups['B_zero'],rr) for g,rr in groups.items() if g!='B_zero'}
adapt=read(ROOT/'simulation/runs/submission_confirmation_20261102/adaptation/complete.json');ad=[]
for r in adapt['results']:
 ad.append(dict(seed=r['seed'],completed=len(r['rows']),accepted=r['accepted'],rejected=r['rejected'],safety_rollbacks=r['safety_rollbacks'],reasons=dict(collections.Counter(k for ev in r['events'] for k in ev.get('failed_criteria',[])))))
res=dict(rl=dict(groups=stats,paired_vs_zero=pairs,adaptation=ad,unique_simulated_test_episodes=sum(len(e['physical_rows']) for e in rl['results']),independent_scenes=60,displayed_rows=len(rows),online_vs_frozen={str(s):paired(groups[f'C_frozen_{s}'],groups[f'D_online_{s}']) for s in [11,23,37]}))
pp=ROOT/'simulation/runs/submission_paper_20261102_v2/test_recovered/complete.json'
if pp.exists():
 pr=read(pp)['rows'];pg={m:[r for r in pr if r['mode']==m] for m in ['baseline','gentle','gentle_recolored']}
 res['paper']=dict(groups={m:summary(rr) for m,rr in pg.items()},paired_gentle_vs_baseline=paired(pg['baseline'],pg['gentle']),paired_recolored_vs_gentle=paired(pg['gentle'],pg['gentle_recolored']),numeric=read(ROOT/'simulation/runs/submission_paper_20261102_v2/numerical_gate_recovered.json'))
 res['paper']['peak_reduction_fraction']=1-res['paper']['groups']['gentle']['peak_max_N']/res['paper']['groups']['baseline']['peak_max_N']
 with (OUT/'paper.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(pr[0]));w.writeheader();w.writerows(pr)
with (OUT/'rl.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(OUT/'metrics.json',res)
print(json.dumps(res,ensure_ascii=False,indent=2))
