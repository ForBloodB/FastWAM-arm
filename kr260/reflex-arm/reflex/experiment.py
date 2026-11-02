"""Reproducible A/B/C surrogate experiments with isolated actor and learner.

Training seeds, promotion seeds and held-out evaluation seeds are disjoint.
The learner receives only actor replay, never promotion/evaluation transitions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from .env import GraspSurrogate, CONDITIONS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def evaluate(model, n, seed_base, condition=None, output=None, label=''):
    env = GraspSurrogate('online', condition)
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=seed_base+i)
        a = np.zeros(5, dtype='float32') if model is None else model.predict(obs, deterministic=True)[0]
        if not np.isfinite(a).all() or np.max(np.abs(a)) > 1.000001:
            raise ValueError('Policy produced unsafe/nonfinite residual')
        _, reward, _, _, info = env.step(a)
        rows.append({'label': label, 'scenario_seed': seed_base+i, 'reward': reward,
                     'action': a.tolist(), **info})
    if output:
        Path(output).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    return {'n': n, 'successes': sum(r['is_success'] for r in rows),
            'success_rate': float(np.mean([r['is_success'] for r in rows])),
            'mean_reward': float(np.mean([r['reward'] for r in rows])),
            'slips': sum(r['slip'] for r in rows), 'overclamps': sum(r['overclamp'] for r in rows),
            'evidence': 'task_space_surrogate'}


def candidate_job(active_path, initial_buffer, replay_path, outpath, updates, seed):
    """Executed in another process. The active actor's weights cannot be mutated."""
    torch.set_num_threads(1)
    model = SAC.load(active_path, env=GraspSurrogate('online'), device='cpu', seed=seed)
    model.set_logger(configure(None, []))
    model.load_replay_buffer(initial_buffer)
    count = 0
    with open(replay_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            obs = np.array(row['obs'], dtype='float32')[None, :]
            a = np.array(row['action'], dtype='float32')[None, :]
            if not np.isfinite(obs).all() or not np.isfinite(a).all() or np.max(abs(a)) > 1.000001:
                raise ValueError('Invalid replay transition')
            model.replay_buffer.add(obs, np.array(row['next_obs'], dtype='float32')[None,:],
                                   a, np.array([row['reward']]), np.array([True]), [{}])
            count += 1
    model.train(gradient_steps=updates, batch_size=128)
    model.save(outpath)
    return {'candidate': outpath+'.zip', 'actor_transitions': count,
            'learner_gradient_updates': updates, 'sha256': sha(outpath+'.zip')}


def run_seed(args, seed, out):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = GraspSurrogate('pretrain')
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=100_000,
                learning_starts=256, batch_size=128, train_freq=4, gradient_steps=1,
                gamma=0., ent_coef='auto_0.05', target_entropy=-5,
                policy_kwargs={'net_arch': [64,64]}, device='cpu', seed=seed, verbose=0)
    model.learn(args.pretrain)
    initial = str(out/'initial.zip')
    model.save(initial)
    buffer = str(out/'pretrain_replay.pkl')
    model.save_replay_buffer(buffer)
    frozen = SAC.load(initial, device='cpu')
    initial_hash = sha(initial)
    # Gate/evaluation scenario seeds are fixed before training; actor seeds < 1e8.
    gate_base = 1_000_000_000+seed*100_000
    test_base = 2_000_000_000+seed*100_000
    a_result = evaluate(None, args.eval_episodes, test_base, output=out/'A_000000.jsonl', label='A')
    b_result = evaluate(frozen, args.eval_episodes, test_base, output=out/'B_000000.jsonl', label='B')
    curves = [{'seed': seed, 'interactions': 0, 'A': a_result, 'B': b_result, 'C': b_result}]
    active_path = initial
    active_version = 'initial'
    active = SAC.load(active_path, device='cpu')
    actor_env = GraspSurrogate('online')
    replay_path = out/'online_replay.jsonl'
    rng = np.random.default_rng(seed+300)
    switches = []
    count = 0
    with ProcessPoolExecutor(1, mp_context=mp.get_context('spawn')) as pool:
        while count < args.online:
            # First half batch is persisted before the learner snapshots the log.
            size = min(args.batch, args.online-count)
            snapshot = out/f'replay_snapshot_{count:06}.jsonl'
            with replay_path.open('a', encoding='utf-8') as log:
                for j in range(size):
                    obs, _ = actor_env.reset(seed=seed*1_000_000+count+j)
                    action = active.predict(obs, deterministic=True)[0]
                    action = np.clip(action+rng.normal(0, .15, 5), -1, 1).astype('float32')
                    nxt, reward, _, _, info = actor_env.step(action)
                    log.write(json.dumps({'obs': obs.tolist(), 'action': action.tolist(),
                        'next_obs': nxt.tolist(), 'reward': reward, 'done': True,
                        'scenario_seed': seed*1_000_000+count+j,
                        'policy_version': active_version, 'monotonic_ns': time.monotonic_ns(), **info})+'\n')
            count += size
            snapshot.write_bytes(replay_path.read_bytes())
            candidate_path = str(out/f'candidate_{count:06}')
            job = pool.submit(candidate_job, active_path, buffer, str(snapshot), candidate_path, args.updates, seed+count)
            # Actor remains available while learning: these are real extra surrogate
            # interactions, count toward the budget and enter the next snapshot.
            concurrent_size = min(args.batch, args.online-count)
            with replay_path.open('a', encoding='utf-8') as log:
                for j in range(concurrent_size):
                    obs, _ = actor_env.reset(seed=seed*1_000_000+count+j)
                    action = np.clip(active.predict(obs, deterministic=True)[0]+rng.normal(0,.15,5),-1,1).astype('float32')
                    nxt, reward, _, _, info = actor_env.step(action)
                    log.write(json.dumps({'obs':obs.tolist(),'action':action.tolist(),'next_obs':nxt.tolist(),
                        'reward':reward,'done':True,'scenario_seed':seed*1_000_000+count+j,
                        'policy_version':active_version,'monotonic_ns':time.monotonic_ns(),**info})+'\n')
            count += concurrent_size
            result = job.result(timeout=300)
            candidate = SAC.load(result['candidate'], device='cpu')
            old_gate = evaluate(active, args.gate_episodes, gate_base)
            new_gate = evaluate(candidate, args.gate_episodes, gate_base)
            accepted = (new_gate['success_rate'] >= old_gate['success_rate'] and
                        new_gate['mean_reward'] > old_gate['mean_reward']+.005)
            switches.append({**result, 'at_interactions':count, 'accepted':accepted,
                             'old_gate':old_gate,'new_gate':new_gate,'previous_version':active_version})
            if accepted:
                active, active_path = candidate, result['candidate']
                active_version = Path(active_path).stem
            # This test set is NEVER used by the promotion decision or learner.
            c = evaluate(active, args.eval_episodes, test_base, output=out/f'C_{count:06}.jsonl', label='C')
            curves.append({'seed':seed,'interactions':count,'A':a_result,'B':b_result,'C':c})
            write_json(out/'learning_curve.json', curves)
            write_json(out/'policy_switches.json', switches)
            print(json.dumps({'seed':seed,'steps':count,'accepted':accepted,
                              'test_A':a_result['success_rate'],'test_B':b_result['success_rate'],
                              'test_C':c['success_rate']}), flush=True)
    active.save(str(out/'champion.zip'))
    # Explicit rollback version is kept immutable for a later task-boundary switch.
    assert sha(initial) == initial_hash, 'Frozen B changed during C training'
    write_json(out/'policy_registry.json', {'active':'champion.zip','rollback':'initial.zip',
                'evidence':'task_space_surrogate','initial_sha256':initial_hash,
                'champion_sha256':sha(out/'champion.zip'),'active_origin':active_version})
    adaptation = {}
    for k, condition in enumerate(CONDITIONS):
        adaptation[condition] = {
            'before':evaluate(frozen,args.eval_episodes,test_base+10_000*(k+1),condition,
                             out/f'adaptation_{condition}_before.jsonl','B'),
            'after':evaluate(active,args.eval_episodes,test_base+10_000*(k+1),condition,
                            out/f'adaptation_{condition}_after.jsonl','C')}
    summary = {'seed':seed,'A':a_result,'B':b_result,'C':curves[-1]['C'],
        'online_interactions':count,'promotions':sum(s['accepted'] for s in switches),
        'adaptation':adaptation,'wall_seconds':time.monotonic()-started,
        'evidence':'task_space_surrogate','frozen_initial_sha256':initial_hash}
    write_json(out/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',default='runs/abc')
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--pretrain',type=int,default=8000)
    p.add_argument('--online',type=int,default=8192)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--updates',type=int,default=600)
    p.add_argument('--eval-episodes',type=int,default=512)
    p.add_argument('--gate-episodes',type=int,default=256)
    args = p.parse_args()
    if min(args.pretrain,args.online,args.batch,args.updates,args.eval_episodes,args.gate_episodes) <= 0:
        p.error('All budgets must be positive')
    out = Path(args.out)
    if out.exists():
        p.error('Output already exists; choose a new directory to preserve previous evidence')
    out.mkdir(parents=True)
    write_json(out/'config.json',vars(args))
    results = [run_seed(args,seed,out/f'seed_{seed}') for seed in args.seeds]
    write_json(out/'summary.json',{'evidence':'task_space_surrogate','seeds':results,
        'mean_success_rate':{g:float(np.mean([r[g]['success_rate'] for r in results])) for g in 'ABC'},
        'std_between_seeds':{g:float(np.std([r[g]['success_rate'] for r in results],ddof=1)) if len(results)>1 else None for g in 'ABC'},
        'limitations':['Synthetic one-step contextual grasp model; not contact dynamics',
                      'No real robot grasp or board-level FPGA control measured',
                      'Gate is a simulation screening step, not hardware safety certification']})
    print(json.dumps(json.loads((out/'summary.json').read_text())),flush=True)


if __name__ == '__main__':
    main()
