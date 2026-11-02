"""Prospective simulation-only safe-boundary residual learning experiment.

Phases run explicitly: prepare, tables, train, adapt, test. Source/development action
tables are deterministic counterfactual simulation data. Adaptation updates
only after completed attempts; held-out test policies are always read-only.
All source/development cases are retained. An inadequate source rescue gate
blocks training while preserving evidence for further authorized development.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from scripts.transition_protocol import TRAIN_SEEDS, DT, declaration, utility, source_coverage

PROJECT = Path(__file__).resolve().parents[1]
_ENV = None
NEW_SOURCES = ('reflex/fixed_residual.py', 'reflex/boundary_residual.py',
               'reflex/residual_bandit.py', 'reflex/transition_residual.py',
               'reflex/guarded_bandit.py', 'scripts/transition_protocol.py',
               'scripts/run_transition_gate.py')
LARGE_ROW_FIELDS = ('vision_calls', 'decisions', 'events', 'residual_move_commands',
                    'boundary_events', 'soft_control_log', 'transition_log', 'landing_feedback')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Exclusive files prevent accidentally overwriting experiment evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def _checked_manifest(manifest, label):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Missing baseline ' + label)
    checks = []
    for relative, expected in sorted(manifest.items()):
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError('Missing frozen baseline file: ' + str(path))
        actual = sha(path)
        if actual != expected:
            raise ValueError('Frozen baseline checksum differs: ' + relative)
        checks.append(dict(path=relative, expected=expected, actual=actual, equal=True))
    return checks


def prepare(root, baseline_plan):
    """Write the declaration and snapshot before producing any trial result."""
    from reflex.transition_residual import ACTION_NAMES, ACTIONS
    if len(ACTION_NAMES) != 8 or 'soft_relief' not in ACTION_NAMES:
        raise ValueError('The boundary protocol requires the declared eight-action bank')
    root = Path(root)
    if (root / 'plan.json').exists():
        plan = read(root / 'plan.json')
        if plan['declaration'] != declaration():
            raise ValueError('Protocol changed after declaration')
        if plan['actions'] != list(ACTION_NAMES) or plan['action_specifications'] != ACTIONS:
            raise ValueError('Action bank changed after declaration')
        _checked_manifest(plan['source_sha256'], 'current source snapshot')
        _checked_manifest(plan['assets_sha256'], 'current binary assets')
        return plan
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('New experiment root must be empty: ' + str(root))
    baseline_plan = Path(baseline_plan)
    baseline = read(baseline_plan)
    baseline_checks = {
        'source': _checked_manifest(baseline.get('source_sha256'), 'source_sha256'),
        'assets': _checked_manifest(baseline.get('assets_sha256'), 'assets_sha256'),
    }
    sources = dict(baseline['source_sha256'])
    sources.update({name: sha(PROJECT / name) for name in NEW_SOURCES})
    assets = dict(baseline['assets_sha256'])
    for name in list(sources) + list(assets):
        destination = root / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, destination)
    shutil.copyfile(baseline_plan, root / 'baseline_plan.json')
    versions = {}
    for package in ('numpy', 'torch', 'mujoco', 'gymnasium'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    protocol = declaration()
    expected_counts = dict(source=len(protocol['source']) * len(ACTION_NAMES),
                           development=len(protocol['development']) * len(ACTION_NAMES),
                           adaptation=sum(len(items) for items in protocol['adaptation'].values()) * 3,
                           test=len(protocol['test']) * (1 + 2 * len(protocol['training_seeds'])))
    plan = dict(schema_version=1, created_unix=time.time(), declaration=protocol,
                actions=list(ACTION_NAMES), action_specifications=deepcopy(ACTIONS),
                timestep_s=DT, expected_rollouts=expected_counts,
                source_sha256=sources, assets_sha256=assets,
                baseline_plan_sha256=sha(baseline_plan), baseline_assertions=baseline_checks,
                baseline_reference=dict(path=str(baseline_plan), plan_sha256=sha(baseline_plan),
                                        source_sha256=baseline['source_sha256'],
                                        assets_sha256=baseline['assets_sha256'], verified_unchanged=True,
                                        snapshot='baseline_plan.json'),
                provenance=dict(python=platform.python_version(), platform=platform.platform(),
                                packages=versions, thread_limits={name: os.environ[name] for name in
                                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}),
                raw_evidence='Per-trial gzip JSON preserves native force chunks, feedback control log, and command/vision/event logs.',
                mask_deduplication='Source replay contains each scene/executed action once; masked-to-zero repeats must have identical physical results.',
                training_gate='tables/source_coverage.json must pass and match the complete source table before any optimization; failure preserves data and keeps future test unused',
                replay_sampling='Uniform executed-action-deduplicated source records; every failed/risky completed decision retained',
                offline_gradient_checkpoints=[0, 200, 600, 1200],
                online_checkpoint_attempts=[24, 48, 72, 96],
                hardware_commands=0)
    write(root / 'plan.json', plan)
    return plan


def worker_init():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    atexit.register(close_env)


def close_env():
    global _ENV
    if _ENV is not None:
        _ENV.close()
        _ENV = None


def environment():
    global _ENV
    if _ENV is None:
        from reflex.transition_residual import TransitionResidualEnv
        _ENV = TransitionResidualEnv(dt=DT)
    return _ENV


def checkpoint(path, policy, **metadata):
    state = policy.state()
    result = dict(state=state, digest=policy.digest(), model_digest=policy.model_digest(),
                  created_unix=time.time(), **metadata)
    assert result['digest'] == digest(state)
    write(path, result)
    return dict(path=str(path), digest=result['digest'], model_digest=result['model_digest'])


def load_policy(path):
    from reflex.residual_bandit import ResidualBandit
    saved = read(path)
    from reflex.guarded_bandit import GuardedBandit
    policy = (GuardedBandit.from_state(saved['state']) if saved['state'].get('algorithm') == GuardedBandit.algorithm else ResidualBandit(saved['state']['config']['action_names']).load(saved['state']))
    if policy.digest() != saved['digest'] or policy.model_digest() != saved['model_digest']:
        raise ValueError('Checkpoint checksum failed: ' + str(path))
    return policy


def preserve_proposal(root,directory,policy,event):
    if event is None:return
    if policy.proposal_snapshot is None:raise AssertionError('Gate event has no candidate/rollback evidence')
    path=Path(root)/directory/'proposals'/f'feedback_{policy.feedback_count:04d}.json'
    proof=deepcopy(policy.proposal_snapshot)
    proof.update(completed_feedbacks=policy.feedback_count,guard_table_digest=policy.guard_digest,
                 candidate_digest=digest(proof['candidate']),incumbent_digest=digest(proof['deployed_before']))
    write(path,proof)
    event['proposal_file']=path.relative_to(root).as_posix();event['proposal_sha256']=sha(path)
    policy.proposal_snapshot=None


def preserve_trace(root, directory, trial_id, row, env):
    trace = dict(schema_version=1, scene_id=row['scene_id'], mode=row['mode'],
                 learner_seed=row['learner_seed'], force_chunks=env.force_chunks,
                 control_log=env.control_log, action_calls=env.action_calls,
                 regrasp_safety=deepcopy((row.get('residual_decision') or {}).get('regrasp_safety')),
                 final_qpos=env.data.qpos.tolist(), final_qvel=env.data.qvel.tolist(),
                 final_ctrl=env.data.ctrl.tolist(), final_simulation_time_s=float(env.data.time))
    row['trace_fields'] = []
    for key in LARGE_ROW_FIELDS:
        if key in row:
            trace[key] = row.pop(key)
            row['trace_fields'].append(key)
    path = Path(root) / directory / 'traces' / (trial_id + '.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trace, ensure_ascii=False, allow_nan=False).encode('utf-8')
    with path.open('xb') as stream:
        stream.write(gzip.compress(payload, compresslevel=3, mtime=0))
    row.update(trace_file=path.relative_to(root).as_posix(), trace_sha256=sha(path),
               native_force_chunk_count=len(env.force_chunks),
               control_log_count=len(env.control_log))


def trial(root, directory, scene, index, mode, learner_seed=None,
          action='zero', policy=None, online=False):
    """The selector receives only sensor features and their observable mask."""
    env = environment()
    selections = []
    before = policy.digest() if policy is not None else None
    model_before = policy.model_digest() if policy is not None else None
    def select(x, raw):
        try:
            allowed = raw['available_actions']
            name = policy.act(x, allowed_names=allowed, explore=online)
            selections.append(dict(action=name, allowed_actions=list(allowed),
                                   predicted_utility=policy.predict(x)))
            return name
        except Exception as exc:
            # SafeContact.run absorbs RuntimeError/ValueError as physical
            # aborts. Programming errors in an actor must stop the runner.
            raise AssertionError('Residual selector failed') from exc
    env.requested_action = action
    env.action_selector = select if policy is not None else None
    env.reset(seed=scene['seed'], options=deepcopy(scene['options']))
    row = env.run()
    recognized_abort = (row['failure_type'] in {
        'initial_detection_failed', 'command_workspace_guard', 'delayed_force_peak_guard',
        'residual_regrasp_not_open', 'boundary_regrasp_not_open',
        'boundary_regrasp_pose_unavailable', 'boundary_regrasp_settle_bounds',
        'boundary_regrasp_clearance_bounds'} or row['failure_type'].startswith(('ik_unreachable:', 'collision:')))
    unexpected_failure = bool(row['safety_abort'] and not recognized_abort)
    row['parent_policy_digest_before'] = row.pop('policy_digest_before')
    row['parent_policy_digest_after'] = row.pop('policy_digest_after')
    if row['parent_policy_digest_before'] != row['parent_policy_digest_after']:
        raise AssertionError('Frozen baseline policy changed')
    decision = row.get('residual_decision')
    if len(selections) != (1 if policy is not None and decision is not None else 0):
        raise AssertionError('Unexpected residual decision count')
    completed_digest = policy.digest() if policy is not None else None
    feedback = utility(row)
    update = None
    # This is the only learning call after an actual environment trial.
    if online and decision is not None and not unexpected_failure:
        if not env.finished:
            raise AssertionError('Cannot update from an incomplete attempt')
        update = policy.update(decision['features'], decision['executed_action'], feedback, completed_outcome=row)
        preserve_proposal(root,directory,policy,update['acceptance_event'] or update['safety_reset'])
    after = policy.digest() if policy is not None else None
    if not online and before != after:
        raise AssertionError('Read-only policy changed during evaluation')
    row.update(split=directory.split('/')[0] if not directory.startswith('tables/') else directory.split('/')[1],
               mode=mode, learner_seed=learner_seed, scene_id=scene['id'],
               scene_seed=scene['seed'], trial_index=index, scenario=deepcopy(scene),
               requested_action=decision['requested_action'] if decision else action,
               executed_action=decision['executed_action'] if decision else 'no_decision',
               utility=feedback, policy_selections=selections,
               policy_digest_before=before, policy_digest_before_update=completed_digest,
               policy_digest_after=after, policy_model_digest_before=model_before,
               policy_model_digest_after=policy.model_digest() if policy is not None else None,
               policy_updated=update is not None, update_after_trial=update,
               unexpected_failure=unexpected_failure,
               update_boundary='completed_attempt_only',
               timestep_s=DT,
               scenario_metadata_scope='evaluation/reset only; never passed to action selector')
    token = f'{index:04d}_{scene["id"]}_{mode}'
    if mode == 'table':
        token += '_' + action
    preserve_trace(root, directory, token, row, env)
    if unexpected_failure:
        write(Path(root) / directory / ('unexpected_' + token + '.json'), row)
        raise AssertionError('Unexpected simulator failure: ' + row['failure_type'])
    return row


def summary(items):
    def group(rr):
        return dict(n=len(rr), success=sum(r['success'] for r in rr),
                    overclamp=sum(r['overclamp'] for r in rr),
                    safety_abort=sum(r['safety_abort'] for r in rr),
                    mean_utility=float(np.mean([r['utility'] for r in rr])) if rr else None,
                    peak_force_N=max((r['fullrate']['peak_force_N'] for r in rr), default=None),
                    p95_force_N=float(np.quantile([r['fullrate']['peak_force_N'] for r in rr], .95)) if rr else None,
                    failure_counts=dict(Counter(r['failure_type'] for r in rr)),
                    selected_actions=dict(Counter(r['executed_action'] for r in rr)))
    return dict(all=group(items), low_friction=group([r for r in items if r['scenario']['low_friction']]))


def table_job(root, split, action, shard, scenes):
    folder = Path(root) / 'tables' / split
    filename = folder / f'rows_{action}_{shard:02d}.jsonl'
    result = []
    with filename.open('x', encoding='utf-8') as stream:
        for index, scene in scenes:
            row = trial(root, f'tables/{split}', scene, index, 'table', action=action)
            append(stream, row)
            result.append(row)
    return dict(split=split, action=action, shard=shard, row_file=str(filename),
                row_sha256=sha(filename), summary=summary(result))


def pool_run(function, jobs, workers):
    result = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=worker_init) as pool:
        futures = [pool.submit(function, *job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            result.append(item)
            print(json.dumps(item, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def load_tables(root, split, plan):
    paths = sorted((Path(root) / 'tables' / split).glob('rows_*.jsonl'))
    found = {}
    for path in paths:
        for number, row in enumerate(rows(path), 1):
            key = (row['scene_id'], row['requested_action'])
            if key in found:
                raise AssertionError('Duplicate full-action table row: ' + str(key))
            row['_source_ref'] = dict(file=path.relative_to(root).as_posix(), line=number,
                                      sha256=sha(path))
            found[key] = row
    expected = {(scene['id'], action) for scene in plan['declaration'][split]
                for action in plan['actions']}
    if set(found) != expected:
        raise AssertionError(f'Incomplete {split} table: {len(found)} / {len(expected)} rows')
    return found


def physical_signature(row):
    # Excludes wall-clock diagnostics and action request labels. Native force,
    # outcomes, initial state and the whole sensor decision remain exact.
    fields = ('initial_state_sha256', 'physics_sha256', 'success', 'failure_type',
              'overclamp', 'safety_abort', 'contact_metrics', 'fullrate',
              'final_object_xyz_m', 'duration_s', 'control_bounds', 'utility')
    result = {name: row[name] for name in fields}
    if row['residual_decision'] is not None:
        result['features'] = row['residual_decision']['features']
        result['observation'] = row['residual_decision']['observation']
    return digest(result)


def table_manifest(root, split):
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha(path)
            for path in sorted((root / 'tables' / split).glob('rows_*.jsonl'))}


def checked_source_gate(root, plan):
    """Recompute the gate and validate data provenance before any learning."""
    root = Path(root)
    receipt = read(root / 'tables' / 'source_coverage.json')
    if receipt['plan_sha256'] != sha(root / 'plan.json'):
        raise AssertionError('Source gate was calculated under a different plan')
    current_manifest = table_manifest(root, 'source')
    if receipt['source_table_sha256'] != current_manifest:
        raise AssertionError('Source action table changed after coverage evaluation')
    table = load_tables(root, 'source', plan)
    actual = source_coverage(table.values(), plan['actions'])
    recorded = {key: receipt[key] for key in actual}
    if recorded != actual:
        raise AssertionError('Source coverage receipt differs from complete table')
    if not actual['passed']:
        raise RuntimeError('Source safe-rescue coverage gate failed. Training is blocked; retained data remain available for source/development or action repair. Do not run the future test.')
    replay = read(root / 'tables' / 'source_replay.json')
    if replay['source_table_sha256'] != current_manifest or replay['record_sha256'] != digest(replay['records']):
        raise AssertionError('Source replay provenance differs')
    summary_receipt = read(root / 'tables' / 'summary.json')
    if summary_receipt['source_replay_sha256'] != sha(root / 'tables' / 'source_replay.json'):
        raise AssertionError('Source replay file changed after collection')
    if summary_receipt['source_coverage_sha256'] != sha(root / 'tables' / 'source_coverage.json'):
        raise AssertionError('Source coverage receipt changed after collection')
    for job in summary_receipt['jobs']:
        path = root / 'tables' / job['split'] / Path(job['row_file']).name
        if sha(path) != job['row_sha256']:
            raise AssertionError('Collected source/development row file changed: ' + str(path))
    guard=read(root/'tables'/'guard_table.json')
    if (guard['digest']!=digest(guard['cases']) or
            guard['source_development_sha256']!=table_manifest(root,'development') or
            summary_receipt['guard_table_sha256']!=sha(root/'tables'/'guard_table.json')):
        raise AssertionError('Development validation guard table changed')
    return receipt


def source_replay(root, plan):
    table = load_tables(root, 'source', plan)
    records, provenance, excluded = [], [], []
    for scene in plan['declaration']['source']:
        by_action = {}
        for requested in plan['actions']:
            row = table[(scene['id'], requested)]
            decision = row['residual_decision']
            if decision is None:
                excluded.append(dict(scene_id=scene['id'], requested_action=requested,
                                     source_row=row['_source_ref']))
                continue
            executed = decision['executed_action']
            if executed in by_action:
                previous = by_action[executed]['row']
                if physical_signature(previous) != physical_signature(row):
                    raise AssertionError('Masked duplicate changed physical result: ' + scene['id'])
                by_action[executed]['requests'].append(requested)
                by_action[executed]['refs'].append(row['_source_ref'])
            else:
                by_action[executed] = dict(row=row, requests=[requested], refs=[row['_source_ref']])
        for executed in plan['actions']:
            if executed not in by_action:
                continue
            item = by_action[executed]
            row = item['row']
            # Match policy serialization exactly; normalization remains fixed.
            features = np.asarray(row['residual_decision']['features'], dtype=np.float32)
            records.append(dict(x=[float(v) for v in features], action=executed, reward=row['utility']))
            provenance.append(dict(scene_id=scene['id'], executed_action=executed,
                                   requested_actions=item['requests'], source_rows=item['refs']))
    if not records:
        raise AssertionError('No completed source decisions available')
    result = dict(records=records, provenance=provenance, excluded_no_decision=excluded,
                  record_sha256=digest(records), source_table_rows=len(table),
                  deduplicated_record_count=len(records),
                  source_table_sha256=table_manifest(root, 'source'),
                  retention='All executed-action records retained, including failures and unsafe outcomes; only physically identical masked-to-zero duplicates are merged',
                  sampling='uniform', negative_utility_records=sum(record['reward'] < 0 for record in records))
    write(Path(root) / 'tables' / 'source_replay.json', result)
    return result


def table_diagnostics(root, split, plan):
    table = load_tables(root, split, plan)
    scene_list = plan['declaration'][split]
    for scene in scene_list:
        baseline = table[(scene['id'], 'zero')]
        for action in plan['actions']:
            other = table[(scene['id'], action)]
            for field in ('initial_state_sha256', 'physics_sha256'):
                if baseline[field] != other[field]:
                    raise AssertionError('Unpaired action table ' + field)
            first, second = baseline['residual_decision'], other['residual_decision']
            if (first is None) != (second is None):
                raise AssertionError('Action changed whether the common probe completed')
            if first is not None and (first['features'] != second['features'] or
                                      first['observation'] != second['observation']):
                raise AssertionError('Action changed its own predecision observation')
            if first is not None and (first['available_actions'] != second['available_actions'] or
                                      first.get('regrasp_safety') != second.get('regrasp_safety')):
                raise AssertionError('Action changed its own observable mask/regrasp context')
    per_action = {}
    for action in plan['actions']:
        chosen = [table[(scene['id'], action)] for scene in scene_list]
        per_action[action] = dict(summary=summary(chosen),
            rescued_zero_failures=sum(not table[(s['id'], 'zero')]['success'] and table[(s['id'], action)]['success'] for s in scene_list),
            harmed_zero_successes=sum(table[(s['id'], 'zero')]['success'] and not table[(s['id'], action)]['success'] for s in scene_list))
    oracle = {}
    for label, ss in (('all', scene_list), ('low_friction', [s for s in scene_list if s['low_friction']])):
        oracle[label] = dict(n=len(ss), success=sum(any(table[(s['id'], a)]['success'] for a in plan['actions']) for s in ss),
                             zero_success=sum(table[(s['id'], 'zero')]['success'] for s in ss))
    return dict(per_action=per_action, oracle_success_upper_bound=oracle,
                oracle_semantics='Uses completed outcomes from every action; unattainable privileged selection benchmark, not a policy result',
                common_probe_observations_identical=True)


def run_tables(root, plan, workers):
    folder = Path(root) / 'tables'
    folder.mkdir(exist_ok=False)
    jobs = []
    shards = max(1, min(workers, 8))
    for split in ('source', 'development'):
        (folder / split).mkdir()
        indexed = list(enumerate(plan['declaration'][split]))
        for action in plan['actions']:
            for shard in range(shards):
                jobs.append((str(root), split, action, shard, indexed[shard::shards]))
    result = pool_run(table_job, jobs, workers)
    replay = source_replay(root, plan)
    diagnostics = {split: table_diagnostics(root, split, plan) for split in ('source', 'development')}
    complete_source = load_tables(root, 'source', plan)
    gate = source_coverage(complete_source.values(), plan['actions'])
    gate.update(plan_sha256=sha(Path(root) / 'plan.json'),
                source_table_sha256=table_manifest(root, 'source'),
                calculated_after_complete_source=True)
    write(folder / 'source_coverage.json', gate)
    guard_cases=[]
    dev_table=load_tables(root,'development',plan)
    for scene in plan['declaration']['development']:
        zero=dev_table[(scene['id'],'zero')];decision=zero['residual_decision']
        results={}
        for action in plan['actions']:
            r=dev_table[(scene['id'],action)]
            results[action]=dict(success=bool(r['success']),overclamp=bool(r['overclamp']),abort=bool(r['safety_abort']),
                peak_N=r['fullrate']['peak_force_N'],budget_N=r['physics']['force_budget_N_per_finger'],
                native_overforce_s=r['fullrate']['overforce_s'],material=r['material'],utility=r['utility'])
        guard_cases.append(dict(scene_id=scene['id'],x=decision['features'] if decision else None,
                                allowed=decision['available_actions'] if decision else ['zero'],results=results))
    write(folder/'guard_table.json',dict(cases=guard_cases,digest=digest(guard_cases),
          scope='Repeated adaptive development validation only; never independent final test.',
          source_development_sha256=table_manifest(root,'development')))
    write(folder / 'summary.json', dict(jobs=result, source_replay_count=len(replay['records']),
                                       guard_table_sha256=sha(folder/'guard_table.json'),
                                       source_replay_sha256=sha(folder / 'source_replay.json'),
                                       diagnostics=diagnostics, source_coverage=gate,
                                       source_coverage_sha256=sha(folder / 'source_coverage.json'),
                                       all_tables_complete=True, training_allowed=gate['passed'],
                                       completed_unix=time.time()))
    # Collection is complete even when training is not justified. The train
    # entry point enforces the gate before creating a model output directory.
    print(json.dumps(dict(phase='source_coverage', passed=gate['passed'],
                          criteria=gate['criteria'], qualifying_contexts=gate['qualifying_boundary_contexts'],
                          training_allowed=gate['passed'], future_test_unused=True)), flush=True)


def development_curve(root, policy, table, plan, seed, steps):
    before = policy.digest()
    selected = []
    folder = Path(root) / 'models' / f'seed_{seed}'
    with (folder / f'development_{steps}.jsonl').open('x', encoding='utf-8') as stream:
        for scene in plan['declaration']['development']:
            zero = table[(scene['id'], 'zero')]
            decision = zero['residual_decision']
            name = 'zero' if decision is None else policy.act(
                decision['features'], allowed_names=decision['available_actions'], explore=False)
            chosen = deepcopy(table[(scene['id'], name)])
            chosen.update(mode='development_table_readonly', learner_seed=seed,
                          gradient_steps=steps, policy_digest_before=before,
                          policy_digest_after=policy.digest())
            append(stream, chosen)
            selected.append(chosen)
    if before != policy.digest():
        raise AssertionError('Development table evaluation changed policy state')
    return dict(gradient_steps=steps, policy_digest=before, summary=summary(selected),
                semantics='Read-only counterfactual lookup in development full-action table; no updates or checkpoint selection')


def train_job(root, seed):
    from reflex.residual_bandit import ResidualBandit
    root = Path(root)
    plan = read(root / 'plan.json')
    checked_source_gate(root, plan)
    records = read(root / 'tables' / 'source_replay.json')['records']
    table = load_tables(root, 'development', plan)
    folder = root / 'models' / f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    learner = plan['declaration']['learner']
    policy = ResidualBandit(plan['actions'], seed=seed, hidden=tuple(learner['hidden']),
                            epsilon=learner['epsilon'], online_steps=learner['online_gradient_steps_per_completed_decision'],
                            anchor_fraction=learner['anchor_fraction'], lr=learner['learning_rate'],
                            batch_size=learner['batch_size'], replay_capacity=learner['replay_capacity'])
    checkpoints, curves, fits = [], [], []
    checkpoints.append(checkpoint(folder / 'offline_0.json', policy, learner_seed=seed,
                                  split='source', offline_gradient_steps=0))
    curves.append(development_curve(root, policy, table, plan, seed, 0))
    for steps, increment in ((200, 200), (600, 400), (1200, 600)):
        fits.append(dict(ending_step=steps, **policy.train_offline(records if steps == 200 else None,
                                                                steps=increment)))
        if policy.offline_gradient_steps != steps:
            raise AssertionError('Offline gradient step count differs')
        checkpoints.append(checkpoint(folder / f'offline_{steps}.json', policy,
                                      learner_seed=seed, split='source', offline_gradient_steps=steps))
        curves.append(development_curve(root, policy, table, plan, seed, steps))
    write(folder / 'train_curve.json', dict(fits=fits, development=curves, checkpoints=checkpoints,
                                           frozen_checkpoint='offline_1200.json',
                                           source_record_sha256=digest(records),
                                           anchor_digest=policy.anchor_digest))
    return dict(phase='train', learner_seed=seed, offline_gradient_steps=policy.offline_gradient_steps,
                anchor_digest=policy.anchor_digest, frozen_digest=policy.digest(),
                development=curves[-1]['summary'])


def run_train(root, plan, workers):
    if not (Path(root) / 'tables' / 'summary.json').is_file():
        raise ValueError('Complete source/development tables before training')
    checked_source_gate(root, plan)
    (Path(root) / 'models').mkdir(exist_ok=False)
    seeds = plan['declaration']['training_seeds']
    result = pool_run(train_job, [(str(root), seed) for seed in seeds], min(workers, len(seeds)))
    if len({r['anchor_digest'] for r in result}) != 1:
        raise AssertionError('Learners did not share the exact same source replay')
    write(Path(root) / 'models' / 'summary.json', result)


def adapt_job(root, seed):
    root = Path(root)
    plan = read(root / 'plan.json')
    scenes = plan['declaration']['adaptation'][str(seed)]
    if not scenes:
        raise AssertionError('Adaptation scene declaration is empty')
    frozen = load_policy(root / 'models' / f'seed_{seed}' / 'offline_1200.json')
    from reflex.guarded_bandit import GuardedBandit
    online = GuardedBandit(frozen, read(root / 'tables' / 'guard_table.json')['cases'], **{k:plan['declaration']['online_gate'][k] for k in ['block_size','epsilon','min_utility_gain']})
    frozen_digest = frozen.digest()
    folder = root / 'adaptation' / f'seed_{seed}'
    folder.mkdir(exist_ok=False)
    checkpoint(folder / 'pre.json', online, learner_seed=seed, completed_adaptation_attempts=0)
    results = {'zero': [], 'frozen': [], 'online': []}
    with (folder / 'rows.jsonl').open('x', encoding='utf-8') as stream:
        for index, scene in enumerate(scenes):
            for mode, policy in (('zero', None), ('frozen', frozen), ('online', online)):
                row = trial(root, f'adaptation/seed_{seed}', scene, index, mode,
                            learner_seed=seed, policy=policy, online=mode == 'online')
                append(stream, row)
                results[mode].append(row)
            if (index + 1) % 24 == 0:
                checkpoint(folder / f'checkpoint_{index + 1}.json', online,
                           learner_seed=seed, completed_adaptation_attempts=index + 1)
                write(folder / f'progress_{index + 1}.json',
                      dict(completed_attempts=index + 1, online_updates=online.updates,
                           risks={mode: summary(rr) for mode, rr in results.items()}))
        if frozen.digest() != frozen_digest:
            raise AssertionError('Frozen adaptation comparator changed')
    final_gate = online.finalize()
    preserve_proposal(root,f'adaptation/seed_{seed}',online,final_gate)
    write(folder / 'final_gate.json', dict(event=final_gate, final_digest=online.digest()))
    write(folder / 'gate_events.json', online.events)
    checkpoint(folder / 'post.json', online, learner_seed=seed,
               completed_adaptation_attempts=len(scenes))
    result = dict(phase='adaptation', learner_seed=seed, completed_attempts=len(scenes),
                  online_updates=online.updates, online_gradient_steps=online.online_gradient_steps,
                  accepted_blocks=online.accepted_blocks, rejected_blocks=online.rejected_blocks, safety_rollbacks=online.safety_rollbacks,
                  frozen_digest=frozen_digest, post_digest=online.digest(),
                  summaries={mode: summary(rr) for mode, rr in results.items()},
                  rows_sha256=sha(folder / 'rows.jsonl'))
    write(folder / 'summary.json', result)
    return result


def run_adapt(root, plan, workers):
    if not (Path(root) / 'models' / 'summary.json').is_file():
        raise ValueError('Complete all six offline training runs before adaptation')
    (Path(root) / 'adaptation').mkdir(exist_ok=False)
    seeds = plan['declaration']['training_seeds']
    result = pool_run(adapt_job, [(str(root), seed) for seed in seeds], min(workers, len(seeds)))
    write(Path(root) / 'adaptation' / 'summary.json', result)


def test_job(root, mode, seed, shard, scenes):
    root = Path(root)
    if mode == 'zero':
        policy, dirname = None, 'zero'
    else:
        location = (root / 'models' / f'seed_{seed}' / 'offline_1200.json' if mode == 'frozen'
                    else root / 'adaptation' / f'seed_{seed}' / 'post.json')
        policy = load_policy(location)
        dirname = f'{mode}_seed_{seed}'
    initial_digest = policy.digest() if policy else None
    folder = root / 'test' / dirname
    result = []
    with (folder / f'rows_{shard:02d}.jsonl').open('x', encoding='utf-8') as stream:
        for index, scene in scenes:
            row = trial(root, 'test/' + dirname, scene, index, mode,
                        learner_seed=seed, policy=policy, online=False)
            append(stream, row)
            result.append(row)
    if policy and initial_digest != policy.digest():
        raise AssertionError('Held-out test mutated policy state')
    return dict(phase='test', mode=mode, learner_seed=seed, shard=shard,
                policy_digest_before=initial_digest,
                policy_digest_after=policy.digest() if policy else None,
                summary=summary(result), rows_sha256=sha(folder / f'rows_{shard:02d}.jsonl'))


def run_test(root, plan, workers):
    if not (Path(root) / 'adaptation' / 'summary.json').is_file():
        raise ValueError('Complete all six adaptation streams before held-out test')
    folder = Path(root) / 'test'
    folder.mkdir(exist_ok=False)
    scenes = list(enumerate(plan['declaration']['test']))
    if not scenes:
        raise AssertionError('Held-out test scene declaration is empty')
    jobs = []
    shards = max(1, min(workers, 8))
    modes = [('zero', None)] + [(mode, seed) for mode in ('frozen', 'online') for seed in plan['declaration']['training_seeds']]
    for mode, seed in modes:
        dirname = 'zero' if mode == 'zero' else f'{mode}_seed_{seed}'
        (folder / dirname).mkdir()
        for shard in range(shards):
            jobs.append((str(root), mode, seed, shard, scenes[shard::shards]))
    result = pool_run(test_job, jobs, workers)
    write(folder / 'summary.json', dict(jobs=result, expected_rows=len(scenes) * len(modes),
                                       completed_unix=time.time(), policies_readonly=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'tables', 'train', 'adapt', 'test'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--baseline-plan', type=Path,
                        default=PROJECT / 'runs/boundary_residual_20261025/main/plan.json')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root.resolve()
    plan = prepare(root, args.baseline_plan)
    if args.phase == 'prepare':
        print(json.dumps(dict(root=str(root), plan_sha256=sha(root / 'plan.json'))))
        return
    phase_file = root / ('phase_' + args.phase + '.json')
    write(phase_file, dict(phase=args.phase, started_unix=time.time(), workers=args.workers,
                           plan_sha256=sha(root / 'plan.json'), source_sha256=plan['source_sha256']))
    functions = dict(tables=run_tables, train=run_train, adapt=run_adapt, test=run_test)
    functions[args.phase](root, plan, args.workers)
    # Guard against source or binary changes during a long phase as well.
    _checked_manifest(plan['source_sha256'], 'post-phase source')
    _checked_manifest(plan['assets_sha256'], 'post-phase binary assets')
    write(root / ('phase_' + args.phase + '_complete.json'),
          dict(completed_unix=time.time(), plan_sha256=sha(root / 'plan.json')))


if __name__ == '__main__':
    main()
