"""Prospective new-source/adaptation/test banks for validated contact actions.

No previously sealed test is opened. All actor observations remain 15D RGBD,
FK/finger encoders and delayed force. Evaluation uses true parameters only to
set domains and score completed trials, never as policy inputs.
"""
import numpy as np

SEEDS=[11,23]
ACTIONS=['zero','tilt_neg55_up6','tilt_neg60_up6']
DT=.0000625
LADDER=[.0000625,.00003125,.000015625]
ONLINE=dict(block_size=4,epsilon=.03,min_utility_gain=.01)
SPECS={
    'source':dict(base=99030000000,masses=[.247,.267,.427],margins=[.88,.91,.97],biases=[-.002,.002],controls=1),
    'monitor':dict(base=99040000000,masses=[.257,.277,.437],margins=[.89,.95,1.07],biases=[.0013],controls=1),
    'adaptation':dict(base=99050000000,masses=[.249,.269,.429],margins=[.885,.925,1.055],biases=[-.0015],controls=0),
    'test':dict(base=99060000000,masses=[.251,.271,.431],margins=[.875,.905,.945,1.065],biases=[-.0017,.0017],controls=1),
}


def scenes(split):
    spec=SPECS[split];rows=[]
    for mass in spec['masses']:
        for margin in spec['margins']:
            for bias in spec['biases']:
                for sensor in ['nominal','degraded']:
                    index=len(rows);mu=margin*mass*9.81/28.
                    rows.append(dict(id=f'action_{split}_metal_{index:04d}',seed=spec['base']+index,
                        low_friction=mu<=.18,capacity_margin=margin,mass_kg=mass,friction=mu,
                        options=dict(material='metal',condition='friction',sensor_condition=sensor,
                            scene_overrides=dict(mass_kg=mass,friction=mu,vision_bias_m=[bias,0.,0.]))))
    for material in ['wood','plastic','sponge']:
        for condition in ['friction','compliance']:
            for sensor in ['nominal','degraded']:
                for rep in range(spec['controls']):
                    index=len(rows)
                    rows.append(dict(id=f'action_{split}_control_{index:04d}',seed=spec['base']+index,low_friction=False,
                        options=dict(material=material,condition=condition,sensor_condition=sensor)))
    if split=='adaptation':
        # Six common nonmetal controls; no separate policy-dependent object RNG.
        for material in ['wood','plastic','sponge']:
            for sensor in ['nominal','degraded']:
                index=len(rows);rows.append(dict(id=f'action_{split}_control_{index:04d}',seed=spec['base']+index,
                    low_friction=False,options=dict(material=material,condition='friction',sensor_condition=sensor)))
        rows=[rows[i] for i in np.random.default_rng(spec['base']+80000).permutation(len(rows))]
    return rows


def declaration():
    banks={s:scenes(s) for s in SPECS}
    assert [len(banks[s]) for s in SPECS]==[48,30,24,60]
    ids=[s['id'] for b in banks.values() for s in b];seeds=[s['seed'] for b in banks.values() for s in b]
    assert len(ids)==len(set(ids)) and len(seeds)==len(set(seeds))
    return dict(banks=banks,seeds=SEEDS,actions=ACTIONS,dt_s=DT,original_ladder_s=LADDER,
        offline_steps=1200,online=ONLINE,source_rescue_min=12,
        source_each_sensor_rescue_min=4,source_each_mass_rescue_min=2,
        source_paired_peak_ratio_max=1.10,source_absolute_peak_budget_ratio_max=1.35,
        numerical_gate=dict(max_peak_difference_N=1.,p95_peak_difference_N=.5,unchanged_labels=True),
        candidate_gate='success increase; no increase overall or per-material overclamp, abort, peak P95/max, normalized peak P95/max, native overbudget time; unchanged utility rule',
        test_scope='New 60 paired independent scenes. Policies and action source frozen before opening outcomes. Two learner seeds share these scenes, not 120 independent objects.',
        original_sealed_confirmation_used=0,physical_motor_commands=0)
