"""Strictly gated contact-action contextual bandit and honest paired evaluation.

New source/monitor/adaptation/test banks are registered separately. No optimizer
is constructed before action numerical/risk gates and source rescue coverage.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import argparse,gzip,json,time,traceback
import numpy as np
from scripts import run_transition_gate as io
from scripts import contact_action_protocol as protocol
from scripts.train_convergence_residual import numerical_gate,utility

ROOT=Path('runs/contact_action_learning_20261030')
MATRIX=Path('runs/contact_actions_20261030')
SOURCES=['scripts/train_contact_actions.py','scripts/contact_action_protocol.py','scripts/run_contact_actions.py',
    'configs/contact_action_protocol_20261030.json','reflex/contact_actions.py','reflex/contact_roll_actions.py',
    'reflex/converged_contact.py','reflex/residual_bandit.py','reflex/guarded_bandit.py',
    'build/convergence_native/control.c','build/convergence_native/control.so']


def checked_matrix_sources(root,declaration):
    for name,h in declaration['source_sha256'].items():
        if io.sha(name)==h:continue
        # The historical collector failed only while serializing np.bool_.
        # Accept exactly that explicit one-expression scalar cast, using the
        # immutable pre-run collector snapshot. Physics/action/library files
        # are never exempted from the original source digest checks.
        assert name=='scripts/run_contact_actions.py',name
        snapshot=Path(root)/'source_snapshot'/name
        assert io.sha(snapshot)==h
        old='passed=not changes and max(delta)<=1. and np.quantile(delta,.95)<=.5))'
        new='passed=bool(not changes and max(delta)<=1. and np.quantile(delta,.95)<=.5)))'
        text=snapshot.read_text();assert text.count(old)==1
        assert text.replace(old,new)==Path(name).read_text()


def action_gates():
    numerical_gate('runs/convergence_repair_20261029/multiccd_full12/complete.json','multiccd',protocol.DT)
    old=io.read('runs/contact_repair_20261028/plan.json')
    for category in ['old_source_sha256','old_assets_sha256']:
        for name,h in old[category].items():assert io.sha(name)==h,name
    checks=[]
    nroot=MATRIX/'guarded_numerics_v6';n=io.read(nroot/'complete.json');decl=io.read(nroot/'declaration.json')
    checked_matrix_sources(nroot,decl)
    original={s['id'] for s in old['numerical']}
    ids={r['scene_id'] for r in n['rows']};assert len(ids)==20 and len(original)==12 and original<ids
    assert decl['steps']==protocol.LADDER
    for action in protocol.ACTIONS[1:]:
        for bank,wanted in [('original12',original),('failure8',ids-original)]:
            for h,k in zip(protocol.LADDER[:-1],protocol.LADDER[1:]):
                aa={r['scene_id']:r for r in n['rows'] if r['action']==action and r['dt_s']==h and r['scene_id'] in wanted}
                bb={r['scene_id']:r for r in n['rows'] if r['action']==action and r['dt_s']==k and r['scene_id'] in wanted}
                assert set(aa)==set(bb)==wanted
                delta=[abs(aa[i]['peak_N']-bb[i]['peak_N']) for i in sorted(wanted)]
                changes=[i for i in wanted if any(aa[i][key]!=bb[i][key] for key in ['success','overclamp','safety_abort'])]
                assert all(aa[i]['common_start_digest']==bb[i]['common_start_digest'] for i in wanted)
                checks.append(dict(action=action,bank=bank,coarse_s=h,fine_s=k,max_delta_N=max(delta),
                    p95_delta_N=float(np.quantile(delta,.95)),label_changes=changes,
                    passed=bool(not changes and max(delta)<=1 and np.quantile(delta,.95)<=.5)))
    # Re-read raw trace clocks, not only the summary's declared timestep.
    for r in n['rows']:
        token=r['scene_id']+'__'+r['action']+'__'+str(round(r['dt_s']*1e9))
        full=io.read(nroot/'trials'/(token+'.json'))
        trace=nroot/full['trace_file'];assert io.sha(trace)==full['trace_sha256']
        windows=json.loads(gzip.decompress(trace.read_bytes()))['impulse_log']
        assert full['physics']['dt']==r['dt_s'] and full['control_bounds']['max_cap_N']<=28.
        assert all(np.isclose(w['duration_s']/w['native_steps'],r['dt_s'],rtol=1e-10,atol=1e-14) for w in windows)
    droot=MATRIX/'guarded_development_v6';development=io.read(droot/'complete.json');dd=io.read(droot/'declaration.json')
    checked_matrix_sources(droot,dd)
    risks=[]
    for action in protocol.ACTIONS[1:]:
        for material in ['overall','metal','wood','plastic','sponge']:
            b=[r for r in development['rows'] if r['action']=='zero' and (material=='overall' or r['material']==material)]
            c=[r for r in development['rows'] if r['action']==action and (material=='overall' or r['material']==material)]
            assert len(b)==len(c) and b
            metrics=lambda rr:dict(success=sum(r['success'] for r in rr),overclamp=sum(r['overclamp'] for r in rr),
                abort=sum(r['safety_abort'] for r in rr),peak_max=max(r['peak_N'] for r in rr),
                peak_p95=float(np.quantile([r['peak_N'] for r in rr],.95)),overforce_s=sum(r['overforce_s'] for r in rr))
            bm,cm=metrics(b),metrics(c)
            risk_ok=all(cm[k]<=bm[k]+1e-10 for k in ['overclamp','abort','peak_max','peak_p95','overforce_s'])
            success_ok=cm['success']>bm['success'] if material in ['overall','metal'] else cm['success']>=bm['success']
            risks.append(dict(action=action,material=material,before=bm,after=cm,passed=risk_ok and success_ok))
    receipt=dict(numerics=checks,risk_checks=risks,numerical_file_sha256=io.sha(nroot/'complete.json'),
        development_file_sha256=io.sha(droot/'complete.json'),physical_motor_commands=0)
    if not all(c['passed'] for c in checks+risks):raise RuntimeError('Unchanged action numerical or baseline risk gate failed')
    return receipt


def prepare():
    registered=io.read('configs/contact_action_protocol_20261030.json')
    assert registered==protocol.declaration()
    gates=action_gates();ROOT.mkdir(exist_ok=False)
    config=dict(protocol=registered,gate_receipt=gates,source_sha256={p:io.sha(p) for p in SOURCES},
        algorithm='15D neural contextual bandit; not SAC',physical_motor_commands=0)
    io.write(ROOT/'plan.json',config)
    for name in SOURCES:
        p=ROOT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes())
    return config


def config():
    c=io.read(ROOT/'plan.json')
    for p,h in c['source_sha256'].items():assert io.sha(p)==h,p
    assert action_gates()==c['gate_receipt']
    return c


def trial(scene,folder,name='zero',policy=None,online=False,dt=None):
    from reflex.contact_roll_actions import ContactRollActionsEnv
    dt=protocol.DT if dt is None else dt
    selected=[];before=None if policy is None else policy.digest()
    def selector(x,raw):
        allowed=[a for a in protocol.ACTIONS if a in raw['available_actions']]
        act=policy.act(x,allowed_names=allowed,explore=online) if policy else (name if name in allowed else 'zero')
        selected.append(dict(features=x.tolist(),allowed=allowed,selected=act))
        return act
    env=ContactRollActionsEnv(rescue_selector=selector,dt=dt)
    try:
        env.reset(seed=scene['seed'],options=deepcopy(scene['options']));r=env.run()
        assert env.finished and r['hardware_motor_commands']==r['object_pose_writes_during_run']==0
        assert r['control_bounds']['max_cap_N']<=28. and r['policy_digest_before']==r['policy_digest_after']
        decision=r['residual_decision'];assert len(selected)==int(decision is not None)
        executed='zero' if decision is None else decision['executed_action']
        assert executed in protocol.ACTIONS
        if policy is not None and not online:assert policy.digest()==before
        trace={k:deepcopy(getattr(env,k,[])) for k in ['diagnostic_log','impulse_log','force_chunks',
            'control_log','timing_log','action_events','repair_events','residual_move_commands','vision_calls','force_cap_log']}
        path=Path(folder);path.mkdir(parents=True,exist_ok=True)
        token=scene['id']+'__'+name+'__'+str(round(dt*1e9))
        tp=path/(token+'.trace.json.gz')
        with tp.open('xb') as f:f.write(gzip.compress(io.canonical(trace).encode(),compresslevel=2,mtime=0))
        update=None
        if online and policy is not None and decision is not None:
            update=policy.update(decision['features'],executed,utility(r),completed_outcome=r)
            io.preserve_proposal(ROOT,path.relative_to(ROOT),policy,update['acceptance_event'] or update['safety_reset'])
        row=dict(scene_id=scene['id'],scene_seed=scene['seed'],request=name,executed=executed,
            features=None if not selected else selected[0]['features'],allowed=['zero'] if not selected else selected[0]['allowed'],
            success=r['success'],failure_type=r['failure_type'],overclamp=r['overclamp'],safety_abort=r['safety_abort'],
            fullrate=r['fullrate'],physics=r['physics'],duration_s=r['duration_s'],utility=utility(r),
            common_start_digest=r['common_start_digest'],actual_dt_s=dt,attempts=r['contact_action_attempts'],
            policy_digest_before=before,policy_digest_after=None if policy is None else policy.digest(),
            online_update_after_completed_trial=update,trace_file=tp.relative_to(ROOT).as_posix(),trace_sha256=io.sha(tp),
            numerical_model=r['collision_manifold'],physical_motor_commands=0)
        io.write(path/(token+'.json'),row);return row
    finally:env.close()


def table_job(split,scene):
    folder=ROOT/'tables'/split
    base=trial(scene,folder)
    rows=[base]
    for action in base['allowed']:
        if action=='zero':continue
        r=trial(scene,folder,name=action)
        assert r['features']==base['features'] and r['common_start_digest']==base['common_start_digest']
        rows.append(r)
    return dict(scene=scene,rows=rows)


def tables(c,workers):
    cases=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(table_job,split,s):split for split in ['source','monitor'] for s in c['protocol']['banks'][split]}
        for f in as_completed(futures):
            item=f.result();item['split']=futures[f];cases.append(item)
            print(json.dumps(dict(table_contexts=len(cases),expected=len(futures))),flush=True)
    io.write(ROOT/'tables/complete.json',dict(cases=cases,actual_rollouts=sum(len(c['rows']) for c in cases),
        masked_requests_not_resimulated=True,all_failures_retained=True,plan_sha256=io.sha(ROOT/'plan.json')))


def checked_tables(c):
    value=io.read(ROOT/'tables/complete.json');assert value['plan_sha256']==io.sha(ROOT/'plan.json')
    records=[];guard=[];rescues=[]
    for case in value['cases']:
        rr=case['rows'];base=next(r for r in rr if r['request']=='zero')
        for r in rr:assert io.sha(ROOT/r['trace_file'])==r['trace_sha256']
        results={r['executed']:dict(success=r['success'],overclamp=r['overclamp'],abort=r['safety_abort'],
            peak_N=r['fullrate']['peak_force_N'],budget_N=r['physics']['force_budget_N_per_finger'],
            native_overforce_s=r['fullrate']['overforce_s'],material=r['physics']['material'],utility=r['utility']) for r in rr}
        assert set(results)==set(base['allowed'])
        if case['split']=='monitor':guard.append(dict(scene_id=base['scene_id'],x=base['features'],allowed=base['allowed'],results=results))
        if case['split']=='source':
            for r in rr:
                if r['features'] is not None:records.append(dict(x=r['features'],action=r['executed'],reward=r['utility']))
            if not base['success'] and any(r['success'] and r['executed']!='zero' and not r['overclamp'] and not r['safety_abort']
                and r['fullrate']['peak_force_N']<=1.35*r['physics']['force_budget_N_per_finger']
                and r['fullrate']['peak_force_N']<=1.10*base['fullrate']['peak_force_N'] for r in rr):rescues.append(case['scene'])
    return records,guard,rescues


def train(c):
    records,guard,rescues=checked_tables(c)
    from collections import Counter
    sensors=Counter(s['options']['sensor_condition'] for s in rescues)
    masses=Counter(s['mass_kg'] for s in rescues)
    coverage=dict(safe_rescue_contexts=len(rescues),sensor_counts=dict(sensors),mass_counts=dict(masses),
        passed=len(rescues)>=12 and all(sensors[s]>=4 for s in ['nominal','degraded']) and sum(n>=2 for n in masses.values())>=2)
    io.write(ROOT/'source_coverage.json',coverage)
    if not coverage['passed']:raise RuntimeError('Source rescue diversity gate failed; optimizer not constructed')
    from reflex.residual_bandit import ResidualBandit
    from reflex.guarded_bandit import guard_metrics,acceptance
    class Zero:
        def digest(self):return 'strong_B'
        def act(self,*args,**kwargs):return 'zero'
    b=guard_metrics(Zero(),guard);outputs=[]
    for seed in c['protocol']['seeds']:
        policy=ResidualBandit(protocol.ACTIONS,seed=seed,epsilon=.03)
        fit=policy.train_offline(records,steps=c['protocol']['offline_steps'])
        metrics=guard_metrics(policy,guard);checks=acceptance(metrics,b,b,min_utility_gain=0.)
        io.checkpoint(ROOT/f'models/frozen_{seed}.json',policy,source_records=len(records))
        outputs.append(dict(seed=seed,fit=fit,metrics=metrics,fixed_B=b,checks=checks,passed=all(checks.values())))
    io.write(ROOT/'models/complete.json',dict(results=outputs,source_records=len(records),coverage=coverage,
        table_sha256=io.sha(ROOT/'tables/complete.json'),algorithm='contextual bandit, not SAC'))


def online_class():
    from reflex.guarded_bandit import GuardedBandit
    class ValidatedActionBandit(GuardedBandit):
        algorithm='validated_contact_action_guarded_bandit_v1'
        def act(self,x,allowed_names=None,explore=False):
            allowed=list(self.action_names if allowed_names is None else allowed_names)
            if explore and self.rng.random()<self.config['epsilon']:
                return allowed[int(self.rng.integers(len(allowed)))]
            return self.deployed.act(x,allowed,explore=False)
    return ValidatedActionBandit


def load(path):
    saved=io.read(path)
    if saved['state']['algorithm']=='validated_contact_action_guarded_bandit_v1':p=online_class().from_state(saved['state'])
    else:return io.load_policy(path)
    assert p.digest()==saved['digest'] and p.model_digest()==saved['model_digest'];return p


def adapt(c):
    assert io.read(ROOT/'policy_numerics_C/complete.json')['passed']
    trained=io.read(ROOT/'models/complete.json')
    assert trained['table_sha256']==io.sha(ROOT/'tables/complete.json')
    if not all(r['passed'] for r in trained['results']):raise RuntimeError('Frozen monitor rejected; online adaptation blocked')
    _,guard,_=checked_tables(c);results=[]
    for seed in c['protocol']['seeds']:
        frozen=load(ROOT/f'models/frozen_{seed}.json');before=frozen.digest()
        online=online_class()(frozen,guard,**c['protocol']['online']);rows=[]
        for s in c['protocol']['banks']['adaptation']:
            for mode,p,up in [('B_zero',None,False),('C_frozen',frozen,False),('D_online',online,True)]:
                rows.append(dict(mode=mode,**trial(s,ROOT/f'adaptation/seed_{seed}/{mode}',policy=p,online=up)))
            print(json.dumps(dict(seed=seed,adaptation_trials=len(rows),accepted=online.accepted_blocks,rejected=online.rejected_blocks)),flush=True)
        event=online.finalize();io.preserve_proposal(ROOT,f'adaptation/seed_{seed}',online,event)
        assert frozen.digest()==before
        io.checkpoint(ROOT/f'models/online_{seed}.json',online,independent_test_used=False)
        results.append(dict(seed=seed,rows=rows,accepted=online.accepted_blocks,rejected=online.rejected_blocks,
            rollbacks=online.safety_rollbacks,events=online.events))
    io.write(ROOT/'adaptation/complete.json',dict(results=results,physical_motor_commands=0))


def evaluation_job(scene):
    # One-decision policies have an identical physical prefix. Select every
    # frozen policy from that prefix BEFORE simulating its selected outcome.
    # Identical choices then share one actual counterfactual trajectory, with
    # explicit hashes; repeated model seeds are not extra independent objects.
    folder=ROOT/'test/physical';base=trial(scene,folder)
    selections=[dict(mode='B_zero',learner_seed=None,action='zero'),
        dict(mode='F_fixed',learner_seed=None,action='tilt_neg60_up6' if 'tilt_neg60_up6' in base['allowed'] else 'zero')]
    for seed in protocol.SEEDS:
        for mode in ['C_frozen','D_online']:
            p=load(ROOT/f'models/{"frozen" if mode=="C_frozen" else "online"}_{seed}.json');before=p.digest()
            act='zero' if base['features'] is None else p.act(base['features'],allowed_names=base['allowed'],explore=False)
            assert p.digest()==before
            selections.append(dict(mode=mode,learner_seed=seed,action=act,checkpoint_digest=before,model_digest=p.model_digest()))
    physical={'zero':base}
    for act in sorted({r['action'] for r in selections}-{'zero'}):
        row=trial(scene,folder,name=act)
        assert row['features']==base['features'] and row['common_start_digest']==base['common_start_digest']
        physical[act]=row
    rows=[dict(**physical[item['action']],mode=item['mode'],learner_seed=item['learner_seed'],
        evaluation_selection=item,counterfactual_shared=True) for item in selections]
    return dict(scene_id=scene['id'],selections=selections,physical_rows=list(physical.values()),rows=rows)


def test(c,workers):
    assert (ROOT/'adaptation/complete.json').is_file()
    assert io.read(ROOT/'policy_numerics_D/complete.json')['passed']
    checkpoints={str(p):io.sha(p) for p in (ROOT/'models').glob('*.json')}
    io.write(ROOT/'test_declaration.json',dict(checkpoint_sha256=checkpoints,scenes=c['protocol']['banks']['test'],
        policies_read_only=True,updates_permitted=False,source_sha256=c['source_sha256']))
    rows=[];cases=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(evaluation_job,s) for s in c['protocol']['banks']['test']]
        for f in as_completed(fs):
            case=f.result();cases.append(case);rows.extend(case['rows'])
            print(json.dumps(dict(test_contexts=len(cases),expected=len(fs),physical_trials=sum(len(x['physical_rows']) for x in cases))),flush=True)
    for p,h in checkpoints.items():assert io.sha(p)==h,p
    io.write(ROOT/'test/complete.json',dict(rows=rows,cases=cases,comparison_rows=len(rows),
        actual_trials=sum(len(x['physical_rows']) for x in cases),independent_scenes=60,
        scope='Read-only frozen-policy routing over full actual counterfactual simulations; identical choices share hashed trajectories',
        algorithm_seed_repeats_not_independent=True,read_only=True,physical_motor_commands=0))


def policy_check_job(scene,tag,seed,dt):
    policy=load(ROOT/f'models/{"frozen" if tag=="C" else "online"}_{seed}.json')
    return dict(tag=tag,learner_seed=seed,**trial(scene,ROOT/f'policy_numerics_{tag}/seed_{seed}',policy=policy,dt=dt))


def policy_numerics(c,tag,workers):
    source=io.read(MATRIX/'guarded_numerics_v6/declaration.json')['scenes']
    original={s['id'] for s in io.read('runs/contact_repair_20261028/plan.json')['numerical']}
    rows=[];aliases=[];jobs=[]
    hashes={}
    for seed in protocol.SEEDS:
        p=ROOT/f'models/{"frozen" if tag=="C" else "online"}_{seed}.json';policy=load(p);hashes[str(p)]=io.sha(p)
        frozen=load(ROOT/f'models/frozen_{seed}.json')
        if tag=='D' and policy.model_digest()==frozen.model_digest():
            aliases.append(dict(seed=seed,source='policy_numerics_C',model_digest=policy.model_digest(),
                reason='Identical deployed weights and greedy allowed-action function; no duplicate trajectories claimed'))
        else:jobs.extend((s,tag,seed,h) for s in source for h in protocol.LADDER)
    io.write(ROOT/f'policy_numerics_{tag}/declaration.json',dict(checkpoint_sha256=hashes,jobs=len(jobs),aliases=aliases,
        source_sha256=c['source_sha256'],steps=protocol.LADDER,original_scene_ids=sorted(original)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(policy_check_job,*j) for j in jobs]
        for f in as_completed(fs):rows.append(f.result());print(json.dumps(dict(policy_numerics=tag,completed=len(rows),expected=len(fs))),flush=True)
    comparisons=[]
    for seed in protocol.SEEDS:
        if any(r['seed']==seed for r in aliases):
            before=io.read(ROOT/'policy_numerics_C/complete.json')
            comparisons.extend(dict(r,aliased_identical_weights=True) for r in before['comparisons'] if r['seed']==seed)
            continue
        for label,ids in [('original12',original),('failure8',{s['id'] for s in source}-original)]:
            for h,k in zip(protocol.LADDER[:-1],protocol.LADDER[1:]):
                aa={r['scene_id']:r for r in rows if r['learner_seed']==seed and r['actual_dt_s']==h and r['scene_id'] in ids}
                bb={r['scene_id']:r for r in rows if r['learner_seed']==seed and r['actual_dt_s']==k and r['scene_id'] in ids}
                assert set(aa)==set(bb)==ids
                delta=[abs(aa[i]['fullrate']['peak_force_N']-bb[i]['fullrate']['peak_force_N']) for i in ids]
                changes=[i for i in ids if any(aa[i][key]!=bb[i][key] for key in ['success','overclamp','safety_abort'])]
                assert all(aa[i]['common_start_digest']==bb[i]['common_start_digest'] for i in ids)
                comparisons.append(dict(seed=seed,bank=label,coarse_s=h,fine_s=k,max_delta_N=max(delta),p95_delta_N=float(np.quantile(delta,.95)),
                    label_changes=changes,passed=bool(not changes and max(delta)<=1. and np.quantile(delta,.95)<=.5)))
    for p,h in hashes.items():assert io.sha(p)==h
    result=dict(rows=rows,actual_rollouts=len(rows),aliases=aliases,comparisons=comparisons,passed=all(r['passed'] for r in comparisons))
    io.write(ROOT/f'policy_numerics_{tag}/complete.json',result)
    if not result['passed']:raise RuntimeError('Frozen learned policy numerical regression failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','tables','train','policy_check_C','adapt','policy_check_D','test']);p.add_argument('--workers',type=int,default=10);a=p.parse_args()
    c=prepare() if a.phase=='prepare' else config()
    if a.phase=='tables':tables(c,a.workers)
    elif a.phase=='train':train(c)
    elif a.phase.startswith('policy_check_'):policy_numerics(c,a.phase[-1],a.workers)
    elif a.phase=='adapt':adapt(c)
    elif a.phase=='test':test(c,a.workers)
    print(json.dumps(dict(phase=a.phase,completed=True)),flush=True)


if __name__=='__main__':main()
