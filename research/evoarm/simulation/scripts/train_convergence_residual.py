"""Numerically gated DEVELOPMENT residual contextual-bandit pilot.

This is not SAC and not independent policy validation. It uses only the original
56 development scenes; sealed confirmation scenes are never loaded. A passed
full-12, three-step numerical gate and freshly collected action tables bound to
the same source hashes are required before constructing an optimizer.

Policy action ``zero`` means no additional residual over the strong fixed rule:
it maps to the environment's ``slow2``, not its faster legacy ``zero`` action.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np

from scripts import run_transition_gate as io

PROJECT = Path(__file__).resolve().parents[1]
ACTION_MAP = {'zero': 'slow2', 'hold05_slow2': 'hold05_slow2',
              'hold10_slow2': 'hold10_slow2', 'soft_relief': 'soft_relief'}
ENV_TO_POLICY = {v: k for k, v in ACTION_MAP.items()}
ORIGINAL_DT_LADDER = [.0000625, .00003125, .000015625]
SOURCE_NAMES = ['scripts/train_convergence_residual.py', 'reflex/converged_contact.py',
                'reflex/impulse_audit.py', 'reflex/contact_repair.py',
                'reflex/timed_contact.py', 'reflex/transition_residual.py',
                'reflex/boundary_residual.py', 'reflex/fixed_residual.py',
                'reflex/residual_bandit.py', 'reflex/guarded_bandit.py',
                'scripts/build_convergence_native.py',
                'build/convergence_native/control.c', 'build/convergence_native/control.so',
                'scripts/build_convergence_implicit.py',
                'build/convergence_implicit/control.c', 'build/convergence_implicit/control.so',
                'scripts/build_convergence_damping.py',
                'build/convergence_damping/control.c', 'build/convergence_damping/control.so',
                'scripts/build_convergence_guarded.py',
                'build/convergence_guarded/control.c', 'build/convergence_guarded/control.so',
                'scripts/build_convergence_forcecontrol.py',
                'build/convergence_forcecontrol/control.c', 'build/convergence_forcecontrol/control.so',
                'scripts/build_convergence_observation.py',
                'build/convergence_observation/control.c', 'build/convergence_observation/control.so',
                'scripts/build_convergence_brake.py',
                'build/convergence_brake/control.c', 'build/convergence_brake/control.so']
LIBRARY_FAMILY = {0:'native',1:'native',2:'native',3:'native',
                  4:'implicit',5:'implicit',6:'implicit',
                  7:'damping',8:'damping',9:'damping',10:'guarded',
                  11:'forcecontrol',12:'observation',13:'observation',14:'observation',
                  15:'brake',16:'brake',17:'brake'}
_ENV = None


def numerical_gate(path, mode, dt):
    """Recompute the original gate rather than trusting a hand-edited bool."""
    path = Path(path).resolve()
    declaration_path = path.parent / 'declaration.json'
    declaration, complete = io.read(declaration_path), io.read(path)
    ladder = declaration.get('steps')
    valid_ladder = (isinstance(ladder, list) and len(ladder) == 3
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and np.isfinite(v) and v > 0 for v in ladder))
    if valid_ladder:
        valid_ladder = (ladder[0] <= ORIGINAL_DT_LADDER[0] + 1e-15
                        and all(abs(ladder[i] / ladder[i+1] - 2.) < 1e-10
                                for i in range(2)))
    if (len(declaration.get('scenes', [])) != 12 or mode not in declaration.get('modes', [])
            or not valid_ladder or dt not in ladder):
        raise RuntimeError('Requires the full twelve-scene, three-step gate for this mode')
    if declaration.get('original_ladder', ORIGINAL_DT_LADDER) != ORIGINAL_DT_LADDER:
        raise RuntimeError('Original step ladder changed in the numerical declaration')
    if ladder != ORIGINAL_DT_LADDER and declaration.get('original_ladder') != ORIGINAL_DT_LADDER:
        raise RuntimeError('A finer step domain must explicitly retain the original ladder')
    expected_gate = dict(max_peak_difference_N=1., p95_peak_difference_N=.5,
                         unchanged_labels=True)
    if declaration.get('gate') != expected_gate or declaration.get('shared_drive_cap_N') != 28.:
        raise RuntimeError('Numerical/drive limits differ from the original gate')
    if not declaration.get('contact_parameters_unchanged'):
        raise RuntimeError('Contact model changes require a separately audited declaration')
    scenes = io.read(PROJECT / 'runs/contact_repair_20261028/plan.json')['numerical']
    expected_ids = {s['id'] for s in scenes}
    if {s['id'] for s in declaration['scenes']} != expected_ids:
        raise RuntimeError('The full original numerical scene bank is required')
    for name, expected in declaration['source_sha256'].items():
        if io.sha(PROJECT / name) != expected:
            raise RuntimeError('Numerical-gate source changed: ' + name)
    rows = [r for r in complete['rows'] if r['mode'] == mode]
    lookup = {(r['scene_id'], r['dt_s']): r for r in rows}
    if len(rows) != 36 or len(lookup) != 36:
        raise RuntimeError('Numerical rows are incomplete or duplicated')
    # Bind the receipt to the library actually loaded by NumericalBridge, not
    # merely the historical native library that a damping mode does not use.
    from reflex.converged_contact import MODES
    if mode not in MODES:
        raise RuntimeError('Unknown frozen numerical mode')
    kind = MODES[mode][0]
    if kind not in LIBRARY_FAMILY:
        raise RuntimeError('Numerical mode has no audited native-library binding')
    family = LIBRARY_FAMILY[kind]
    library = f'build/convergence_{family}/control.so'
    required = {'scripts/build_convergence_native.py',
                'build/convergence_native/control.c','build/convergence_native/control.so',
                'reflex/converged_contact.py'}
    required.update({f'scripts/build_convergence_{family}.py',
                     f'build/convergence_{family}/control.c',library})
    if not required.issubset(declaration['source_sha256']):
        raise RuntimeError('Numerical declaration omits an actual candidate generator/library')
    evidence = []
    for scene_id in sorted(expected_ids):
        for integration_dt in ladder:
            compact = lookup[(scene_id, integration_dt)]
            token = scene_id + '__' + mode + '__' + str(round(integration_dt * 1e9))
            row_path = path.parent / 'trials' / (token + '.json')
            trace_path = path.parent / 'trials' / (token + '.trace.json.gz')
            full = io.read(row_path)
            if (full['native_library_sha256'] != declaration['source_sha256'][library]
                    or full['physics']['dt'] != integration_dt
                    or full['dt_s'] != integration_dt
                    or full['control_bounds']['max_cap_N'] > 28. + 1e-12):
                raise RuntimeError('Actual integration/library/force limit differs from declaration')
            if kind == 10:
                effective = full['implicit_effective_force_max_N']
                if (not np.isfinite(effective) or effective < 0
                        or effective > full['control_bounds']['max_cap_N'] + 1e-9
                        or full['gripper_damping_Nsm'] != 18.):
                    raise RuntimeError('Guarded implicit effective-force/gain audit failed')
            if mode.startswith('multiccd'):
                manifold = full['collision_manifold']
                if (manifold.get('multiccd') is not True
                        or manifold.get('ccd_iterations') != 200
                        or not np.isclose(manifold.get('ccd_tolerance_m', float('nan')),
                                          1e-9, rtol=0., atol=1e-18)):
                    raise RuntimeError('Declared multiccd collision manifold did not take effect')
            if (full['fullrate']['peak_force_N'] != compact['peak_N']
                    or any(full[key] != compact[key] for key in
                           ('success','overclamp','safety_abort','common_start_digest'))):
                raise RuntimeError('Numerical summary differs from the full trial record')
            trace = json.loads(gzip.decompress(trace_path.read_bytes()))
            impulses = trace.get('impulse_log', [])
            if not impulses or any(w['native_steps'] <= 0 or not np.isclose(
                    w['duration_s'] / w['native_steps'], integration_dt,
                    rtol=1e-10, atol=1e-14) for w in impulses):
                raise RuntimeError('Actual native steps do not match the declared ladder')
            evidence.append(dict(row_file=str(row_path),row_sha256=io.sha(row_path),
                                 trace_file=str(trace_path),trace_sha256=io.sha(trace_path)))
    comparisons = []
    for a, b in zip(ladder[:-1], ladder[1:]):
        differences, changed = [], []
        for scene_id in sorted(expected_ids):
            left, right = lookup[(scene_id, a)], lookup[(scene_id, b)]
            if left['common_start_digest'] != right['common_start_digest']:
                raise RuntimeError('Numerical initial states differ')
            if any(left[k] != right[k] for k in ('success', 'overclamp', 'safety_abort')):
                changed.append(scene_id)
            differences.append(abs(left['peak_N'] - right['peak_N']))
        result = dict(coarse_s=a, fine_s=b, n=12, label_changes=changed,
                      max_delta_N=max(differences),
                      p95_delta_N=float(np.quantile(differences, .95)))
        result['passed'] = bool(not changed and result['max_delta_N'] <= 1.
                                and result['p95_delta_N'] <= .5)
        comparisons.append(result)
    if not all(r['passed'] for r in comparisons):
        raise RuntimeError('Full twelve-scene numerical gate failed; optimizer is blocked')
    return dict(full12_gate_passed=True, numerical_mode=mode, dt_s=dt,
                original_ladder_s=ORIGINAL_DT_LADDER, actual_ladder_s=ladder,
                numerical_domain=('original ladder' if ladder == ORIGINAL_DT_LADDER else
                                  'finer declared integration domain; original coarse-step failures remain'),
                active_native_library=library,
                complete_file=str(path), complete_sha256=io.sha(path),
                declaration_file=str(declaration_path),
                declaration_sha256=io.sha(declaration_path), comparisons=comparisons,
                native_step_evidence=evidence,
                numerical_source_sha256=declaration['source_sha256'])


def prepare(args):
    root = args.root.resolve()
    receipt = numerical_gate(args.gate, args.mode, args.dt)
    scene_plan = io.read(args.scene_plan)
    bank = scene_plan['development']
    if len(bank) != 56 or len({s['id'] for s in bank}) != 56:
        raise ValueError('This pilot requires the original 56 development scene bank')
    subsets = dict(source=[s for i, s in enumerate(bank) if i % 4 in (0, 1)],
                   monitor=[s for i, s in enumerate(bank) if i % 4 == 2],
                   adaptation=[s for i, s in enumerate(bank) if i % 4 == 3])
    for name, limit in [('source', args.source_count), ('monitor', args.monitor_count),
                        ('adaptation', args.adapt_count)]:
        if limit is not None:
            if limit < 1 or limit > len(subsets[name]):
                raise ValueError('Invalid development subset limit: ' + name)
            subsets[name] = subsets[name][:limit]
    manifest = {name: io.sha(PROJECT / name) for name in SOURCE_NAMES}
    frozen = {}
    for category in ('old_source_sha256', 'old_assets_sha256'):
        expected = scene_plan.get(category)
        if not expected:
            raise RuntimeError('Missing original frozen source/asset manifest')
        for name, h in expected.items():
            if io.sha(PROJECT / name) != h:
                raise RuntimeError('Original frozen file changed: ' + name)
        frozen[category] = expected
    config = dict(schema=1, numerical_mode=args.mode, dt_s=args.dt,
                  numerical_gate=receipt, source_sha256=manifest,
                  frozen_baseline_manifest=frozen,
                  action_map=ACTION_MAP, fixed_rule='combo_observed + slow2',
                  shared_drive_cap_N=28., scenarios=subsets,
                  original_development_bank=bank, scene_plan_sha256=io.sha(args.scene_plan),
                  seeds=args.seeds, offline_steps=args.steps,
                  algorithm='development_guarded_contextual_bandit_v1; NOT SAC',
                  online=dict(block_size=4, epsilon=.03, min_utility_gain=.01),
                  guard_scope='Adaptive development monitor only, not independent safety evidence',
                  claim_scope='Development pilot; no independent success/generalization claim',
                  sealed_confirmation_used=0, hardware_motor_commands=0)
    plan = root / 'plan.json'
    if plan.exists():
        if io.read(plan) != config:
            raise RuntimeError('Frozen pilot plan differs; use a new output directory')
    else:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError('New pilot root must be empty')
        io.write(plan, config)
        io.write(root / 'full12_gate_passed.json', receipt)
        for name in SOURCE_NAMES:
            dest = root / 'source_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, dest)
    return config


def utility(row):
    """Native terminal labels are targets only, never actor observations."""
    ratio = row['fullrate']['peak_force_N'] / row['physics']['force_budget_N_per_finger']
    return float(2 * row['success'] - (not row['success']) - 12 * row['overclamp']
                 - 6 * row['safety_abort'] - 1.2 * np.clip(ratio - 1., 0., 3.)
                 - .01 * max(0., row['duration_s'] - 15.))


def environment(config):
    global _ENV
    if _ENV is None:
        from reflex.converged_contact import ConvergenceEnv
        # The callback is explicitly installed by this external pilot after
        # construction; the frozen parent StatePolicy remains read-only.
        _ENV = ConvergenceEnv(variant='combo_observed',
                              numerical_mode=config['numerical_mode'], dt=config['dt_s'])
    return _ENV


def trial(root, config, split, scene, index, *, request='zero', policy=None,
          online=False, seed=None):
    env = environment(config)
    selected = []
    before = None if policy is None else policy.digest()

    def selector(x, raw):
        allowed = [name for name, physical in ACTION_MAP.items()
                   if physical in raw['available_actions']]
        if 'zero' not in allowed:
            raise AssertionError('Strong fixed baseline is unavailable')
        try:
            chosen = policy.act(x, allowed_names=allowed, explore=online) if policy else (
                request if request in allowed else 'zero')
        except Exception as exc:
            raise AssertionError('Residual pilot selector error') from exc
        selected.append(dict(features=x.tolist(), allowed=allowed, requested_policy_action=request,
                             selected_policy_action=chosen, environment_action=ACTION_MAP[chosen]))
        return ACTION_MAP[chosen]

    env.requested_action = 'slow2'
    env.action_selector = selector
    env.reset(seed=scene['seed'], options=deepcopy(scene['options']))
    row = env.run()
    if not env.finished or row['hardware_motor_commands'] or row['object_pose_writes_during_run']:
        raise AssertionError('Trial completion or simulation-only contract failed')
    if row['control_bounds']['max_cap_N'] > 28. + 1e-12:
        raise AssertionError('Shared drive cap increased')
    parent_before, parent_after = row.get('policy_digest_before'), row.get('policy_digest_after')
    if parent_before != parent_after:
        raise AssertionError('Frozen parent policy changed')
    decision = row.get('residual_decision')
    if len(selected) != (1 if decision else 0):
        raise AssertionError('Unexpected number of post-probe decisions')
    executed = 'zero' if decision is None else ENV_TO_POLICY.get(decision['executed_action'])
    if executed is None:
        raise AssertionError('Unexpected physical residual action outside whitelist')
    reward = utility(row)
    recognized = (not row['safety_abort'] or row['failure_type'] in {
        'initial_detection_failed', 'command_workspace_guard', 'delayed_force_peak_guard',
        'acquisition_not_verified'} or row['failure_type'].startswith(('collision:', 'ik_unreachable:')))
    update = None
    directory = Path(root) / split
    directory.mkdir(parents=True, exist_ok=True)
    token = f'{index:03d}_{scene["id"]}_{request}'
    trace = {k:deepcopy(getattr(env,k,[])) for k in ('force_chunks','control_log',
        'diagnostic_log','impulse_log','timing_log','transition_log','repair_events',
        'residual_move_commands','vision_calls')}
    trace.update(final_qpos=env.data.qpos.tolist(),final_qvel=env.data.qvel.tolist())
    trace_path=directory/(token+'.trace.json.gz')
    with trace_path.open('xb') as f:
        f.write(gzip.compress(io.canonical(trace).encode(),compresslevel=2,mtime=0))
    if online and policy is not None and decision is not None and recognized:
        update = policy.update(decision['features'], executed, reward, completed_outcome=row)
        io.preserve_proposal(root, split, policy,
                            update['acceptance_event'] or update['safety_reset'])
    if policy is not None and not online and before != policy.digest():
        raise AssertionError('Frozen/evaluation policy changed')
    result = dict(scene_id=scene['id'], scene_seed=scene['seed'], request=request,
        learner_seed=seed, numerical_mode=config['numerical_mode'],dt_s=config['dt_s'],
        action_map=ACTION_MAP,executed_policy_action=executed,
        environment_executed_action=None if decision is None else decision['executed_action'],
        features=None if not selected else selected[0]['features'],
        allowed=['zero'] if not selected else selected[0]['allowed'], selection=selected,
        success=row['success'],failure_type=row['failure_type'],overclamp=row['overclamp'],
        safety_abort=row['safety_abort'],physics=row['physics'],fullrate=row['fullrate'],
        duration_s=row['duration_s'],utility=reward,
        common_start_digest=row['common_start_digest'],
        initial_state_sha256=row.get('initial_state_sha256'),physics_sha256=row.get('physics_sha256'),
        parent_policy_digest_before=parent_before,parent_policy_digest_after=parent_after,
        policy_digest_before=before,policy_digest_after=None if policy is None else policy.digest(),
        update_after_completed_attempt=update,trace_file=trace_path.relative_to(root).as_posix(),
        trace_sha256=io.sha(trace_path),plan_sha256=io.sha(Path(root)/'plan.json'),
        development_only=True,sealed_test_used=0,motor_commands=0)
    io.write(directory/(token+'.json'),result)
    if not recognized:
        raise AssertionError('Unexpected simulator failure retained: '+row['failure_type'])
    return result


def table_job(root,config,split,scene,index):
    return [trial(root,config,'tables/'+split,scene,index,request=name) for name in ACTION_MAP]


def collect_tables(root,config,workers):
    folder=Path(root)/'tables';folder.mkdir(exist_ok=False)
    tasks=[(split,scene,i) for split in ('source','monitor')
           for i,scene in enumerate(config['scenarios'][split])]
    results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(table_job,str(root),config,*task) for task in tasks]
        for f in as_completed(futures):
            results.extend(f.result());print(json.dumps(dict(table_rows=len(results))),flush=True)
    manifest={p.relative_to(root).as_posix():io.sha(p)
              for p in folder.glob('*/*.json')}
    io.write(folder/'complete.json',dict(plan_sha256=io.sha(Path(root)/'plan.json'),
        numerical_gate_sha256=io.sha(Path(root)/'full12_gate_passed.json'),
        rows=results,row_sha256=manifest,action_map=ACTION_MAP,
        same_controller_recapture=True,motor_commands=0))


def checked_tables(root,config):
    numerical_gate(config['numerical_gate']['complete_file'],config['numerical_mode'],config['dt_s'])
    path=Path(root)/'tables/complete.json';table=io.read(path)
    if (not table.get('same_controller_recapture') or table['action_map']!=ACTION_MAP
            or table['plan_sha256']!=io.sha(Path(root)/'plan.json')
            or table['numerical_gate_sha256']!=io.sha(Path(root)/'full12_gate_passed.json')):
        raise RuntimeError('Action-table provenance differs; optimizer is blocked')
    for name,h in config['source_sha256'].items():
        if io.sha(PROJECT/name)!=h:raise RuntimeError('Frozen training source changed: '+name)
    for manifest in config['frozen_baseline_manifest'].values():
        for name,h in manifest.items():
            if io.sha(PROJECT/name)!=h:raise RuntimeError('Frozen source/asset changed: '+name)
    verified_rows=[]
    for name,h in table['row_sha256'].items():
        if io.sha(Path(root)/name)!=h:raise RuntimeError('Action table changed: '+name)
        row=io.read(Path(root)/name)
        if io.sha(Path(root)/row['trace_file'])!=row['trace_sha256']:
            raise RuntimeError('Action trace changed: '+row['trace_file'])
        verified_rows.append(row)
    if sorted(io.canonical(r) for r in verified_rows)!=sorted(io.canonical(r) for r in table['rows']):
        raise RuntimeError('Action-table summary differs from signed per-trial files')
    expected={(s['id'],a) for split in ('source','monitor')
              for s in config['scenarios'][split] for a in ACTION_MAP}
    lookup={(r['scene_id'],r['request']):r for r in verified_rows}
    if len(lookup)!=len(verified_rows) or set(lookup)!=expected:
        raise RuntimeError('Same-controller action table is incomplete')
    for scene_id in {i[0] for i in expected}:
        base=lookup[(scene_id,'zero')]
        for action in ACTION_MAP:
            r=lookup[(scene_id,action)]
            for key in ('common_start_digest','initial_state_sha256','physics_sha256','features','allowed'):
                if r[key]!=base[key]:raise RuntimeError('Unpaired predecision context: '+key)
            if r['numerical_mode']!=config['numerical_mode'] or r['dt_s']!=config['dt_s']:
                raise RuntimeError('Action table used a different numerical controller')
    return lookup


def model_data(lookup,scenes):
    replay=[];guard=[];ceiling=0;base_success=0
    for scene in scenes:
        baseline=lookup[(scene['id'],'zero')];results={}
        seen=set()
        for action in ACTION_MAP:
            r=lookup[(scene['id'],action)];executed=r['executed_policy_action']
            results[action]=dict(success=r['success'],overclamp=r['overclamp'],
                abort=r['safety_abort'],peak_N=r['fullrate']['peak_force_N'],
                budget_N=r['physics']['force_budget_N_per_finger'],
                native_overforce_s=r['fullrate']['overforce_s'],
                material=r['physics']['material'],utility=r['utility'])
            if r['features'] is not None and executed not in seen:
                replay.append(dict(x=r['features'],action=executed,reward=r['utility']))
                seen.add(executed)
        guard.append(dict(scene_id=scene['id'],x=baseline['features'],
                          allowed=baseline['allowed'],results=results))
        base_success+=int(baseline['success'])
        ceiling+=int(any(lookup[(scene['id'],a)]['success'] for a in ACTION_MAP))
    return replay,guard,dict(scenes=len(scenes),fixed_success=base_success,
                            finite_action_hindsight_success=ceiling,
                            remaining_observed_headroom=ceiling-base_success)


def train(root,config):
    lookup=checked_tables(root,config)  # Must precede importing/constructing learners.
    records,_,coverage=model_data(lookup,config['scenarios']['source'])
    _,monitor,monitor_coverage=model_data(lookup,config['scenarios']['monitor'])
    if not records or not monitor:raise RuntimeError('Empty recaptured source/monitor data')
    folder=Path(root)/'models';folder.mkdir(exist_ok=False)
    from reflex.residual_bandit import ResidualBandit
    from reflex.guarded_bandit import guard_metrics,acceptance
    class FixedRule:
        def digest(self):return 'fixed_combo_observed_slow2'
        def act(self,x,allowed,explore=False):
            if 'zero' not in allowed:raise AssertionError('B fixed rule unavailable')
            return 'zero'
    fixed_metrics=guard_metrics(FixedRule(),monitor)
    outputs=[]
    for seed in config['seeds']:
        policy=ResidualBandit(ACTION_MAP,seed=seed,epsilon=.03)
        fit=policy.train_offline(records,steps=config['offline_steps'])
        io.checkpoint(folder/f'frozen_{seed}.json',policy,development_only=True,
                      training_records=len(records),source_coverage=coverage)
        observed=guard_metrics(policy,monitor)
        checks=acceptance(observed,fixed_metrics,fixed_metrics,min_utility_gain=0.)
        outputs.append(dict(seed=seed,fit=fit,monitor=observed,
                            fixed_rule_monitor=fixed_metrics,monitor_checks_against_fixed_B=checks,
                            monitor_safe_against_fixed_B=all(checks.values()),
                            anchor_digest=policy.anchor_digest,model_digest=policy.model_digest()))
    io.write(folder/'complete.json',dict(results=outputs,source_coverage=coverage,
        monitor_coverage=monitor_coverage,table_sha256=io.sha(Path(root)/'tables/complete.json'),
        independent_test_used=False,algorithm='one_decision_neural_contextual_bandit; NOT SAC'))
    print(json.dumps(dict(phase='train',seeds=config['seeds'],coverage=coverage)),flush=True)


def adapt(root,config):
    lookup=checked_tables(root,config)
    trained=io.read(Path(root)/'models/complete.json')
    if trained['table_sha256']!=io.sha(Path(root)/'tables/complete.json'):
        raise RuntimeError('Training action table changed')
    if not all(r['monitor_safe_against_fixed_B'] for r in trained['results']):
        raise RuntimeError('Frozen development models failed the B fixed-rule monitor; adaptation is blocked')
    _,monitor,_=model_data(lookup,config['scenarios']['monitor'])
    from reflex.guarded_bandit import GuardedBandit
    folder=Path(root)/'adaptation';folder.mkdir(exist_ok=False);summaries=[]
    for seed in config['seeds']:
        frozen=io.load_policy(Path(root)/'models'/f'frozen_{seed}.json')
        online=GuardedBandit(frozen,monitor,**config['online']);frozen_digest=frozen.digest();rows=[]
        for index,scene in enumerate(config['scenarios']['adaptation']):
            for mode,policy,is_online in [('B_fixed_rule',None,False),('C_frozen',frozen,False),('D_online',online,True)]:
                rows.append(trial(root,config,f'adaptation/seed_{seed}/{mode}',scene,index,
                                  policy=policy,online=is_online,seed=seed))
        event=online.finalize();io.preserve_proposal(root,f'adaptation/seed_{seed}',online,event)
        if frozen.digest()!=frozen_digest:raise AssertionError('Frozen comparator mutated')
        io.checkpoint(folder/f'online_{seed}.json',online,development_only=True)
        summary=dict(seed=seed,rows=rows,accepted_blocks=online.accepted_blocks,
                     rejected_blocks=online.rejected_blocks,safety_rollbacks=online.safety_rollbacks,
                     accepted_events=online.events,interpretation='Prequential development stream, not independent frozen test')
        io.write(folder/f'seed_{seed}.json',summary);summaries.append(summary)
    io.write(folder/'complete.json',dict(results=summaries,independent_test_used=False,
        sealed_confirmation_used=0,motor_commands=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','tables','train','adapt'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True,help='Full-12 stage complete.json, with adjacent declaration.json')
    p.add_argument('--mode',required=True)
    p.add_argument('--dt',type=float,required=True)
    p.add_argument('--scene-plan',type=Path,default=PROJECT/'runs/contact_repair_20261028/plan.json')
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23])
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--source-count',type=int)
    p.add_argument('--monitor-count',type=int)
    p.add_argument('--adapt-count',type=int)
    a=p.parse_args()
    if min(a.steps,a.workers)<1 or len(a.seeds)<2 or len(set(a.seeds))!=len(a.seeds):
        p.error('Positive steps/workers and at least two distinct learner seeds required')
    config=prepare(a)
    if a.phase=='tables':collect_tables(a.root.resolve(),config,a.workers)
    elif a.phase=='train':train(a.root.resolve(),config)
    elif a.phase=='adapt':adapt(a.root.resolve(),config)
    else:print(json.dumps(dict(prepared=str(a.root),full12_gate_passed=True,optimizer_created=False)))
if __name__=='__main__':main()
