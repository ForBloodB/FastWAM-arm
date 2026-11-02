"""Prospective simulation-only protocol for safely rescuable boundary contexts.

All preceding experiment results are development evidence. These partitions use
new scene seeds. Source-only empirical rescue labels gate training coverage;
they are not deployable safety guarantees or actor inputs. No failed or risky
source records are removed or downweighted. The test split stays untouched if
the source coverage gate requires further action/development work.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np


TRAIN_SEEDS = [11, 23, 37, 53, 71, 89]
DT = .00025  # Locked after preregistration contact/macro numerical pilots.
CONVERGENCE_TIMESTEPS = [.001, .0005, .00025, .000125]
BASES = dict(source=97100000000, development=97200000000,
             adaptation=97300000000, test=97400000000)
ADAPTATION_SHUFFLE_BASE = 97500000000
SPECS = {
    # 252 boundary/infeasible metal contexts + 48 balanced material controls.
    'source': dict(margins=[.86, 1.02, 1.08, 1.12, 1.16, 1.20, 1.50],
                   masses=[.251, .341, .431], biases=[-.003, 0., .003], reps=2, controls=4),
    # 72 metal contexts + 48 controls; no checkpoint cherry-picking.
    'development': dict(margins=[.85, 1.04, 1.10, 1.14, 1.18, 1.48],
                        masses=[.261, .351, .441], biases=[-.0026, .0026], reps=1, controls=4),
    # 72 metal contexts + 24 controls per optimization seed.
    'adaptation': dict(margins=[.845, 1.055, 1.155, 1.445],
                       masses=[.249, .339, .429], biases=[-.0022, 0., .0022], reps=1, controls=2),
    # 324 metal contexts + 72 controls, shared by all final learned models.
    'test': dict(margins=[.837, 1.037, 1.097, 1.137, 1.177, 1.477],
                 masses=[.259, .349, .439], biases=[-.0027, 0., .0027], reps=3, controls=6),
}
SAFE_RESCUE = dict(absolute_peak_budget_ratio=1.35, matched_zero_peak_ratio=1.10,
                   source_contexts_min=12, each_sensor_contexts_min=4,
                   each_mass_contexts_min=2, represented_boundary_ratios_min=2,
                   boundary_ratio_min=1.08, boundary_ratio_max=1.20,
                   label_scope='Empirical source-only paired label, not a deployment safety guarantee')
LEARNER = dict(algorithm='one_decision_neural_contextual_bandit', feature_count=15,
               hidden=[64, 64], offline_gradient_steps=1200,
               offline_checkpoints=[0, 200, 600, 1200], epsilon=.03,
               online_gradient_steps_per_completed_decision=8, anchor_fraction=.50,
               batch_size=64, replay_capacity=512, learning_rate=.001,
               source_sampling='uniform retained records; no oversampling or case removal')


def scenes(split, replicate=0):
    if split not in SPECS:
        raise ValueError('Unknown split: ' + str(split))
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError('replicate must be a nonnegative integer')
    if split != 'adaptation' and replicate != 0:
        raise ValueError('Only adaptation may use learner-specific scene replicates')
    if split == 'adaptation' and replicate > len(TRAIN_SEEDS):
        raise ValueError('Adaptation replicate exceeds the declared learner count')
    spec = SPECS[split]
    base = BASES[split] + replicate * 100000
    result = []
    for margin in spec['margins']:
        for mass in spec['masses']:
            friction = margin * mass * 9.81 / 28.
            for bias in spec['biases']:
                for sensor in ('nominal', 'degraded'):
                    for rep in range(spec['reps']):
                        index = len(result)
                        result.append(dict(
                            id=f'transition_{split}_metal_{index:04d}', seed=base + index,
                            stratum='metal_capacity_boundary', capacity_margin=margin,
                            mass_kg=mass, friction=friction, bias_m=bias, rep=rep,
                            low_friction=friction <= .18,
                            options=dict(material='metal', condition='friction', sensor_condition=sensor,
                                         scene_overrides=dict(friction=friction, mass_kg=mass,
                                                              vision_bias_m=[bias, 0., 0.]))))
    for material in ('wood', 'plastic', 'sponge'):
        for condition in ('friction', 'compliance'):
            for sensor in ('nominal', 'degraded'):
                for rep in range(spec['controls']):
                    index = len(result)
                    result.append(dict(
                        id=f'transition_{split}_control_{index:04d}', seed=base + index,
                        stratum='material_control', rep=rep, low_friction=False,
                        options=dict(material=material, condition=condition, sensor_condition=sensor)))
    if split == 'adaptation':
        rng = np.random.default_rng(ADAPTATION_SHUFFLE_BASE + replicate)
        result = [result[index] for index in rng.permutation(len(result))]
    return result


def _force(row):
    peak = float(row['fullrate']['peak_force_N'])
    budget = float(row['physics']['force_budget_N_per_finger'])
    if not math.isfinite(peak) or not math.isfinite(budget) or peak < 0 or budget <= 0:
        raise ValueError('Invalid native peak or material budget')
    return peak, budget


def utility(row):
    """Terminal simulation labels only; never concatenate these values to state.

    Every failed trial and every peak above 1.35 times its material budget has
    negative utility. Full trajectory peak, overclamp and abort labels remain
    visible even when an action would otherwise achieve a successful placement.
    """
    peak, budget = _force(row)
    duration = float(row['duration_s'])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Invalid completed-trial duration')
    success = bool(row['success'])
    excessive_peak = peak > SAFE_RESCUE['absolute_peak_budget_ratio'] * budget
    return float(2 * success - (not success) - 12 * bool(row['overclamp'])
                 - 6 * bool(row['safety_abort']) - 3 * excessive_peak
                 - 1.2 * np.clip(peak / budget - 1, 0, 3)
                 - .01 * max(0, duration - 15))


def safe_rescue_label(zero, candidate):
    """Evaluate the prespecified empirical paired source rescue criterion.

    The caller must establish paired scene/state identity. Coverage validation
    below does so and forbids using development/adaptation/test outcomes here.
    """
    zero_peak, zero_budget = _force(zero)
    peak, budget = _force(candidate)
    if not math.isclose(budget, zero_budget, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Paired material budgets differ')
    return bool(not zero['success'] and candidate['success']
                and not candidate['overclamp'] and not candidate['safety_abort']
                and peak <= SAFE_RESCUE['absolute_peak_budget_ratio'] * budget + 1e-12
                and peak <= SAFE_RESCUE['matched_zero_peak_ratio'] * zero_peak + 1e-12)


def source_coverage(rows, action_names):
    """Strict complete-source gate; each scene counts once, not once per action.

    All requests, including masked actions and every failure, must be present.
    Only a genuinely executed nonzero action can establish a rescue context.
    A failed gate prevents model training, not further authorized development:
    preserve these data, improve source/development coverage or the action bank,
    re-register the changed training stage, and leave the future test unused.
    """
    rows = list(rows)
    actions = tuple(action_names)
    if not actions or 'zero' not in actions or len(set(actions)) != len(actions):
        raise ValueError('A unique action bank including zero is required')
    manifest = {scene['id']: scene for scene in scenes('source')}
    lookup = {}
    for row in rows:
        key = row['scene_id'], row['requested_action']
        if row.get('split') != 'source' or key[0] not in manifest or key[1] not in actions:
            raise ValueError('Coverage labels may use declared source action-table rows only')
        if key in lookup:
            raise ValueError('Duplicate source scene/requested-action row')
        scene = manifest[key[0]]
        if row['scene_seed'] != scene['seed']:
            raise ValueError('Source scene seed differs from prospective declaration')
        for label in ('material', 'condition', 'sensor_condition'):
            if row[label] != scene['options'][label]:
                raise ValueError('Source scene metadata differs: ' + label)
        lookup[key] = row
    expected = {(scene_id, action) for scene_id in manifest for action in actions}
    if set(lookup) != expected:
        raise ValueError(f'Incomplete source action table: {len(lookup)} / {len(expected)}')
    rescued = []
    per_action = Counter()
    for scene_id, scene in manifest.items():
        zero = lookup[(scene_id, 'zero')]
        qualifying = []
        for action in actions:
            row = lookup[(scene_id, action)]
            for key in ('initial_state_sha256', 'physics_sha256'):
                if row[key] != zero[key]:
                    raise ValueError('Unpaired source state or physics: ' + scene_id)
            decision, zero_decision = row['residual_decision'], zero['residual_decision']
            if (decision is None) != (zero_decision is None):
                raise ValueError('Source predecision reachability differs')
            if decision is not None:
                if not np.allclose(decision['features'], zero_decision['features'], atol=1e-12, rtol=0):
                    raise ValueError('Source predecision features differ')
                if row['residual_action'] != decision['executed_action']:
                    raise ValueError('Source executed-action label differs')
            if (action != 'zero' and decision is not None and not decision['masked']
                    and decision['executed_action'] == action and safe_rescue_label(zero, row)):
                qualifying.append(action)
                per_action[action] += 1
        if qualifying:
            rescued.append(dict(scene_id=scene_id, scene_seed=scene['seed'],
                                material=scene['options']['material'],
                                sensor=scene['options']['sensor_condition'],
                                mass_kg=scene.get('mass_kg'), capacity_margin=scene.get('capacity_margin'),
                                low_friction=scene['low_friction'], safe_rescue_actions=qualifying))
    # The training claim targets low-friction metal; soft-control rescues do not
    # satisfy its boundary coverage requirement, although they remain recorded.
    boundary = [row for row in rescued if row['material'] == 'metal' and row['low_friction']
                and SAFE_RESCUE['boundary_ratio_min'] <= row['capacity_margin'] <= SAFE_RESCUE['boundary_ratio_max']]
    sensors = Counter(row['sensor'] for row in boundary)
    masses = Counter(row['mass_kg'] for row in boundary)
    ratios = Counter(row['capacity_margin'] for row in boundary)
    criteria = dict(
        distinct_safely_rescued_boundary_source_contexts=len(boundary) >= SAFE_RESCUE['source_contexts_min'],
        each_sensor_stratum=all(sensors[sensor] >= SAFE_RESCUE['each_sensor_contexts_min'] for sensor in ('nominal', 'degraded')),
        each_mass_stratum=all(masses[mass] >= SAFE_RESCUE['each_mass_contexts_min'] for mass in SPECS['source']['masses']),
        multiple_boundary_ratios=len(ratios) >= SAFE_RESCUE['represented_boundary_ratios_min'])
    return dict(passed=all(criteria.values()), criteria=criteria, source_rows=len(rows),
                source_scenes=len(manifest), qualifying_boundary_contexts=len(boundary),
                all_safe_rescue_contexts=len(rescued), safe_rescue_actions=dict(per_action),
                boundary_sensor_counts=dict(sensors), boundary_mass_counts={str(k): v for k, v in sorted(masses.items())},
                boundary_ratio_counts={str(k): v for k, v in sorted(ratios.items())},
                rescued_contexts=rescued, threshold=deepcopy(SAFE_RESCUE),
                sampling='All source records retained uniformly; no weight or label filtering',
                training_allowed=all(criteria.values()),
                if_failed='Gate training; retain all data and continue only source/development or action repair under a new declaration. Keep future test unexecuted.',
                hardware_commands=0)


def declaration():
    partitions = dict(source=scenes('source'), development=scenes('development'),
                      adaptation={str(seed): scenes('adaptation', index + 1) for index, seed in enumerate(TRAIN_SEEDS)},
                      test=scenes('test'))
    return dict(
        protocol_name='smooth_transition_guarded_residual_v1', training_seeds=list(TRAIN_SEEDS),
        **partitions, timestep_s=DT,
        timestep_selection=dict(pilot_candidates_s=list(CONVERGENCE_TIMESTEPS), selected_s=DT,
                                rule='Main0.25ms and fine0.125ms fixed prospectively. Development regression compares identical requested actions; final frozen-policy half-step checks remain separate. No complete convergence claim and no timestep selection from final test.'),
        learner=deepcopy(LEARNER), source_coverage_gate=deepcopy(SAFE_RESCUE),
        prior_data_scope='Every previously inspected test set is development evidence only; no prior test scene seed is reused as new independent evidence.',
        baseline='TransitionResidualEnv(all) shared across zero/frozen/online: reference/force-target shaping, supported-hold mask and observed tabletop stop. Fast native PD/encoder damping, contact physics, terminal scorer and 28N drive ceiling unchanged.',
        online_gate=dict(block_size=8,epsilon=.03,min_utility_gain=.01,development_reused=True,independent_confirmation='test only',criteria='Success not lower; overclamp/abort, P95/max native force, P95/max force/budget and total native exceed duration not higher than incumbent and original frozen, both overall and by material; mean utility improves >=0.01 versus incumbent and is not below original frozen.',terminal_risk_rollback='Completed overclamp, abort or native_peak>1.35*budget resets deployed weights/optimizer to original frozen; actual failed/risky trial and all experience retained. Simulation labels are validation/terminal feedback only, never actor inputs.'),
        low_friction_definition='metal class with latent mu <= 0.18, evaluation/label stratification only',
        scene_design='Boundary-enriched source and new parameter combinations; retain infeasible cases and balanced wood/plastic/sponge controls. Same simulator, not evidence of unknown-material or cross-simulator transfer.',
        source_sharing='Six optimizer/RNG seeds share one full-action source dataset. Uniform executed-action-deduplicated replay retains every failure and risky action.',
        terminal_utility='2success -1failure -12overclamp -6abort -3I(native_peak/budget>1.35) -1.2clip(native_peak/budget-1,0,3) -.01max(duration_s-15,0)',
        reward_privilege='Simulation success, native peak and material budget are terminal labels only; actor remains the unchanged 15 sensor features.',
        freeze='After source coverage gate passes, exactly1200 offline gradient steps; fixed checkpoints0/200/600/1200, no development/test selection',
        online='96 completed new attempts per optimizer seed, epsilon0.03 from a development-selected exploratory subset. Shadow receives eight updates per completed decision, 50% immutable source anchor. Every eight feedbacks, validate against both incumbent and original frozen policy on the reusable development action table; rejected weights/optimizer/RNG are rolled back while all completed experience is retained. Terminal overclamp/abort or peak>1.35budget resets deployment to original frozen. Final partial block is explicitly validated.',
        test_semantics='396 common independent scenes; shared zero run once, six frozen and six post-update policies evaluated read-only; test never changes model/config/replay/optimizer/RNG',
        primary='post-update minus frozen low-friction success; paired by learner seed and common test scene, crossed paired10000 bootstrap resamples stratified by material/sensor',
        acceptance='Test primary paired95% interval lower>0, at least5/6 learner seeds improve, observed overall test overclamp/abort and full-trajectory native-force P95/max do not increase versus frozen. Whole-learning claim also requires adaptation-stream overclamp/abort and full-trajectory P95/max not increased versus paired frozen. All unmet conditions are reported.',
        acceptance_details=dict(bootstrap_resamples=10000, bootstrap_seed=97600000001,
                                positive_seed_count_min=5, primary_ci95_lower_strictly_positive=True,
                                test_risk_metrics_no_increase=['n_overclamp', 'n_abort', 'native_peak_p95_N', 'native_peak_max_N'],
                                adaptation_risk_metrics_no_increase=['n_overclamp', 'n_abort', 'native_peak_p95_N', 'native_peak_max_N'],
                                force_scope='Entire approach, probe, residual/regrasp, lift, transport, landing and release; never only the post-action window',
                                statistical_limit='Observed no-increase gates are not formal safety equivalence proofs; zero observed events does not imply zero risk.'),
        no_test_use_after_failed_source_gate=True, hardware_commands=0)
