"""Prospective simulation-only confirmation for competition evidence.

Re-trains three source-only bandits, adapts on new completed episodes, then
locks all policies before a new paired test bank. No material/physics truth is
passed to the selector. Identical selected actions share real simulated traces.
"""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')
def isolate(event,args):
    if event=='socket.__new__' and args[1] in (2,10,29):raise RuntimeError('Simulation forbids IP/CAN sockets')
    if event=='open' and isinstance(args[0],str) and args[0].startswith(('/dev/tty','/dev/pcan','/dev/mem')):
        raise RuntimeError('Simulation forbids actuator devices')
sys.addaudithook(isolate)
import argparse,json,time,hashlib,shutil,traceback
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from scripts import run_shielded_actions as runner
from scripts import run_transition_gate as io
from reflex.residual_bandit import ResidualBandit
from reflex.guarded_bandit import guard_metrics,acceptance

ROOT=Path('runs/submission_confirmation_20261102')
OLD=Path('runs/shielded_action_learning_20261030')
SEEDS=[11,23,37]
ACTIONS=['zero','tilt_neg55_up6','tilt_neg60_up6']

def configure():runner.ROOT=ROOT
def scenes(split):
    spec=dict(adaptation=([.254,.274,.434],[.89,.93,1.05],[-.0014],99110210000),
              test=([.253,.273,.433],[.88,.92,.96,1.06],[-.0016,.0016],99110230000))[split]
    masses,margins,biases,seedbase=spec;rows=[]
    for mass in masses:
        for margin in margins:
            for bias in biases:
                for sensor in ['nominal','degraded']:
                    i=len(rows);mu=margin*mass*9.81/28.
                    rows.append(dict(id=f'submission_{split}_metal_{i:03d}',seed=seedbase+i,
                        options=dict(material='metal',condition='friction',sensor_condition=sensor,
                            scene_overrides=dict(mass_kg=mass,friction=mu,vision_bias_m=[bias,0.,0.]))))
    for material in ['wood','plastic','sponge']:
        for condition in (['friction'] if split=='adaptation' else ['friction','compliance']):
            for sensor in ['nominal','degraded']:
                i=len(rows);rows.append(dict(id=f'submission_{split}_control_{i:03d}',seed=seedbase+i,
                    options=dict(material=material,condition=condition,sensor_condition=sensor)))
    if split=='adaptation':rows=[rows[i] for i in np.random.default_rng(seedbase+90000).permutation(len(rows))]
    return rows

def prepare():
    ROOT.mkdir(exist_ok=False);configure()
    oldplan=io.read(OLD/'plan.json')
    for name,h in oldplan['source_sha256'].items():assert io.sha(name)==h,name
    assert io.read(OLD/'policy_numerics_C/complete.json')['passed']
    for f in ['shield.json','training_data.json']:shutil.copy2(OLD/f,ROOT/f)
    data=io.read(ROOT/'training_data.json')
    banks={s:scenes(s) for s in ['adaptation','test']}
    oldseeds={s['seed'] for bank in oldplan['protocol']['banks'].values() for s in bank}
    assert not oldseeds.intersection(s['seed'] for bank in banks.values() for s in bank)
    names=oldplan['source_sha256'].copy();names['scripts/run_submission_confirmation.py']=io.sha(__file__)
    plan=dict(created_unix=time.time(),banks=banks,seeds=SEEDS,actions=ACTIONS,dt_s=.0000625,
        training_source_records=len(data['records']),training_data_sha256=io.sha(ROOT/'training_data.json'),
        source_sha256=names,prior_test_not_used_for_training=True,
        source_and_adaptive_monitor_reused=True,online_config=dict(block_size=4,epsilon=.03,min_utility_gain=.01),
        offline_steps=2400,algorithm='one-decision 15D neural contextual bandit, not SAC',
        test_policy_updates=False,risk_gate='No increase in per-material max/P95 force, overclamp, abort or overbudget time; unchanged',
        physical_motor_commands=0)
    io.write(ROOT/'plan.json',plan)
    class Zero:
        def digest(self):return 'B'
        def act(self,*a,**k):return 'zero'
    baseline=guard_metrics(Zero(),data['guard']);training=[]
    for seed in SEEDS:
        model=ResidualBandit(ACTIONS,seed=seed,epsilon=.03);curve=[];previous=0
        for step in [100,400,1200,2400]:
            fit=model.train_offline(data['records'],steps=step-previous);previous=step
            metrics=guard_metrics(model,data['guard']);curve.append(dict(step=step,fit=fit,adaptive_monitor_metrics=metrics))
        checks=acceptance(metrics,baseline,baseline,min_utility_gain=0.)
        io.checkpoint(ROOT/f'models/frozen_{seed}.json',model,source_only=True)
        training.append(dict(seed=seed,curve=curve,monitor_checks=checks,monitor_passed=all(checks.values())))
    io.write(ROOT/'training.json',dict(results=training,learning_curves_are_development_not_independent=True))
    print(json.dumps(dict(phase='prepare',seed_monitor=[(r['seed'],r['monitor_passed']) for r in training],test_scenes=len(banks['test']))),flush=True)

def checked():
    configure();p=io.read(ROOT/'plan.json')
    for name,h in p['source_sha256'].items():assert io.sha(name)==h,name
    return p

def adapt_one(seed):
    p=checked();data=io.read(ROOT/'training_data.json');trained=io.read(ROOT/'training.json')
    passed=next(r for r in trained['results'] if r['seed']==seed)['monitor_passed']
    frozen=runner.load(ROOT/f'models/frozen_{seed}.json')
    if not passed:
        io.checkpoint(ROOT/f'models/online_{seed}.json',frozen,monitor_rejected=True)
        return dict(seed=seed,monitor_rejected=True,rows=[],accepted=0,rejected=0)
    online=runner.online_class()(frozen,data['guard'],**p['online_config']);rows=[]
    for s in p['banks']['adaptation']:
        row=runner.trial(s,ROOT/f'adaptation/seed_{seed}',policy=online,online=True)
        rows.append(row);print(json.dumps(dict(phase='adapt',seed=seed,done=len(rows),accepted=online.accepted_blocks)),flush=True)
    event=online.finalize();io.preserve_proposal(ROOT,f'adaptation/seed_{seed}',online,event)
    io.checkpoint(ROOT/f'models/online_{seed}.json',online,independent_test_used=False)
    return dict(seed=seed,rows=rows,accepted=online.accepted_blocks,rejected=online.rejected_blocks,
                safety_rollbacks=online.safety_rollbacks,events=online.events)

def test_one(scene):
    checked();folder=ROOT/'test/physical';base=runner.trial(scene,folder)
    fixed=next(a for a in ['tilt_neg60_up6','tilt_neg55_up6','zero'] if a in base['allowed'])
    selections=[dict(mode='B_zero',seed=None,action='zero'),dict(mode='F_fixed',seed=None,action=fixed)]
    trained=io.read(ROOT/'training.json')
    for seed in SEEDS:
        valid=next(r for r in trained['results'] if r['seed']==seed)['monitor_passed']
        for mode,file in [('C_frozen','frozen'),('D_online','online')]:
            pol=runner.load(ROOT/f'models/{file}_{seed}.json');before=pol.digest()
            action='zero' if not valid or base['features'] is None else pol.act(base['features'],base['allowed'],explore=False)
            assert before==pol.digest()
            selections.append(dict(mode=mode,seed=seed,action=action,model_digest=pol.model_digest(),monitor_rejected=not valid))
    physical={'zero':base}
    for action in sorted({r['action'] for r in selections}-{'zero'}):
        trial=runner.trial(scene,folder,name=action)
        assert trial['common_start_digest']==base['common_start_digest'] and trial['features']==base['features']
        physical[action]=trial
    return dict(scene=scene,selections=selections,physical_rows=list(physical.values()),
        rows=[dict(**physical[r['action']],mode=r['mode'],learner_seed=r['seed'],selection=r) for r in selections])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','adapt','test']);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    if a.phase=='prepare':prepare();return
    p=checked();start=time.time();results=[];errors=[]
    if a.phase=='test':
        assert (ROOT/'adaptation/complete.json').exists()
        hashes={str(f):io.sha(f) for f in (ROOT/'models').glob('*.json')}
        io.write(ROOT/'test_declaration.json',dict(before_test_unix=time.time(),checkpoints=hashes,scenes=p['banks']['test'],updates=False))
    fn=adapt_one if a.phase=='adapt' else test_one;items=SEEDS if a.phase=='adapt' else p['banks']['test']
    with ProcessPoolExecutor(max_workers=min(a.workers,len(items)),mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(fn,s):s for s in items}
        for f in as_completed(futures):
            try:results.append(f.result())
            except Exception as ex:errors.append(dict(item=futures[f],error=str(ex),traceback=traceback.format_exc()))
            print(json.dumps(dict(phase=a.phase,done=len(results),errors=len(errors),expected=len(items),elapsed_s=time.time()-start)),flush=True)
    if a.phase=='test':
        for f,h in hashes.items():assert io.sha(f)==h
    folder='adaptation' if a.phase=='adapt' else 'test'
    io.write(ROOT/folder/('incomplete.json' if errors else 'complete.json'),
        dict(results=results,errors=errors,elapsed_s=time.time()-start,
             independent_test_scenes=len(items) if a.phase=='test' else 0,
             physical_motor_commands=0))
    if errors:raise RuntimeError('Incomplete simulation batch; errors retained')

if __name__=='__main__':main()
