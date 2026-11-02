"""Paired, fixed-action capability diagnostics. Simulation only; no learning.

The frozen SafeContactEnv remains the common dynamics, controller and scorer.
Scene interventions are applied before run; action interventions use only
declared constants and the existing delayed sensor controller.
"""
from copy import deepcopy
import hashlib
import json

import mujoco
import numpy as np

from .safe_contact import SafeContactEnv
from .safe_contact_native import roll


def action(name, family, **kw):
    return dict(name=name, family=family, offset_m=[0., 0., 0.], pitch_rad=0.,
                yaw_rad=0., probe_time_scale=1., lift_time_scale=1.,
                stop_guard=False, parent_option=0, **kw)


ACTIONS = [
    action('zero', 'baseline'),
    {**action('x_minus_4mm', 'position'), 'offset_m': [-.004, 0., 0.]},
    {**action('x_plus_4mm', 'position'), 'offset_m': [.004, 0., 0.]},
    {**action('y_minus_3mm', 'position'), 'offset_m': [0., -.003, 0.]},
    {**action('y_plus_3mm', 'position'), 'offset_m': [0., .003, 0.]},
    {**action('z_minus_2mm', 'position'), 'offset_m': [0., 0., -.002]},
    {**action('z_plus_2mm', 'position'), 'offset_m': [0., 0., .002]},
    {**action('pitch_minus_0p10', 'orientation'), 'pitch_rad': -.10},
    {**action('pitch_plus_0p10', 'orientation'), 'pitch_rad': .10},
    {**action('yaw_minus_0p10', 'orientation'), 'yaw_rad': -.10},
    {**action('yaw_plus_0p10', 'orientation'), 'yaw_rad': .10},
    {**action('slow_probe', 'speed'), 'probe_time_scale': 1.8},
    {**action('slow_lift', 'speed'), 'lift_time_scale': 1.8},
    {**action('stop_hold', 'slip'), 'stop_guard': True},
    {**action('slow_stop', 'combination'), 'probe_time_scale': 1.8,
     'lift_time_scale': 1.8, 'stop_guard': True},
    {**action('legacy_recenter', 'existing_composite'), 'parent_option': 2},
]
ACTION_BY_NAME = {a['name']: a for a in ACTIONS}
PICKUP_PHASES = {'descend', 'descend_settle', 'close', 'preload', 'probe_lift',
                 'lift', 'retry_lower', 'retry_close', 'regrasp_clear',
                 'regrasp_align', 'regrasp_descend'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


class ActionRescueEnv(SafeContactEnv):
    def __init__(self, intervention='zero', **kwargs):
        self.intervention = deepcopy(ACTION_BY_NAME[intervention])
        self.scene_overrides = {}
        self.force_chunks = []
        self.phase_forces = {}
        self.action_calls = []
        self.intervention_started = False
        super().__init__(fixed_action=self.intervention['parent_option'], **kwargs)

    def set_intervention(self, name):
        self.intervention = deepcopy(ACTION_BY_NAME[name])
        self.fixed_action = self.intervention['parent_option']

    def observe(self, name, prior_xy=None):
        result = super().observe(name, prior_xy)
        bias = np.asarray(self.scene_overrides.get('vision_bias_m', [0., 0., 0.]))
        if result.get('valid') and np.any(bias):
            result['xyz'] = (np.asarray(result['xyz']) + bias).tolist()
            result['declared_camera_bias_m'] = bias.tolist()
        return result

    def reset(self, *, seed=None, options=None):
        options = deepcopy(options or {})
        self.scene_overrides = deepcopy(options.pop('scene_overrides', {}))
        self.force_chunks = []
        self.phase_forces = {}
        self.action_calls = []
        self.intervention_started = False
        obs, info = super().reset(seed=seed, options=options)
        # The common reset first settles the same object on the same table.
        # These randomized properties are installed before the trial begins.
        # No object pose/velocity is written and no hidden parameter reaches
        # the action controller. The vision-based initial drive is unchanged.
        changed = False
        if 'friction' in self.scene_overrides:
            self.mu = float(self.scene_overrides['friction'])
            self.model.geom_friction[self.og] = [self.mu, .005, .0002]
            self.model.pair_friction[:] = [self.mu, self.mu, .005, .0002, .0002]
            changed = True
        if 'mass_kg' in self.scene_overrides:
            self.mass = float(self.scene_overrides['mass_kg'])
            self.model.body_mass[self.ob] = self.mass
            hx, hy, hz = self.size / 2
            self.model.body_inertia[self.ob] = self.mass / 3 * np.array(
                [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])
            changed = True
        if changed:
            q, v, tm = self.data.qpos.copy(), self.data.qvel.copy(), self.data.time
            # mj_setConst resets its working data to qpos0. Use a disposable
            # MjData so the paired trial's settled object is never teleported.
            scratch = mujoco.MjData(self.model)
            mujoco.mj_setConst(self.model, scratch)
            assert np.array_equal(q, self.data.qpos)
            assert np.array_equal(v, self.data.qvel) and tm == self.data.time
            mujoco.mj_forward(self.model, self.data)
        self.initial_snapshot = {
            'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
            'ctrl': self.data.ctrl.tolist(), 'time_s': float(self.data.time),
            'warmstart': self.data.qacc_warmstart.tolist(),
            'physics': self.params(), 'observation': self.sensor_observation(),
            'sensor_rng': deepcopy(self.sensor_rng.bit_generator.state),
            'sensor_queue': [x.tolist() for x in self.sensor_queue],
        }
        self.initial_state_sha256 = digest(self.initial_snapshot)
        self.physics_sha256 = digest(self.params())
        self.before_policy_digest = self.policy.digest()
        return obs, info

    def move(self, xyz, yaw, duration, phase, gap, force,
             apply_residual=True, track=False):
        spec = self.intervention
        if phase == 'descend':
            self.intervention_started = True
        target = np.asarray(xyz, float).copy()
        self.option = dict(self.option)  # never mutate the frozen global bank
        if spec['stop_guard']:
            self.option['guard'] = True
        if self.intervention_started:
            self.residual_pitch = float(spec['pitch_rad'])
            if phase in PICKUP_PHASES:
                target += np.asarray(spec['offset_m'])
            yaw += spec['yaw_rad']
            if phase == 'probe_lift':
                duration *= spec['probe_time_scale']
            if phase == 'lift':
                duration *= spec['lift_time_scale']
        self.action_calls.append(dict(phase=phase, target_m=target.tolist(),
                                      yaw_rad=float(yaw), duration_s=float(duration),
                                      tracking_hold=bool(self.in_tracking_tick)))
        return super().move(target, yaw, duration, phase, gap, force,
                            apply_residual, track)

    def _raw_roll(self, a, b, ga, gb, cap, start, total, n):
        # Same numerical call and five accumulators as frozen SafeContactEnv.
        # Additional values are diagnostics only, never read by control.
        stats = roll(self.model, self.data, a, b, ga, gb, cap, start, total, n,
                     self.og, self.left, self.right, self.table,
                     float(self.params()['force_budget_N_per_finger']))
        if max(stats[:2]) > max(self.fullrate[:2]):
            self.fullrate_peak_details = {
                'phase': self.phase, 'time_s': float(self.data.time), 'cap_N': cap,
                'opening_encoder_m': float(self.data.qpos[7] + self.data.qpos[8])}
        self.fullrate[:2] = np.maximum(self.fullrate[:2], stats[:2])
        self.fullrate[2] += stats[2]
        self.fullrate[3:] = np.maximum(self.fullrate[3:], stats[3:])
        self.force_chunks.append([
            float(self.data.time), self.phase, float(stats[0]), float(stats[1]),
            float(stats[2]), float(stats[3]), float(stats[4]), float(cap),
            float(self.sensor_force[0]), float(self.sensor_force[1]),
            float(self.data.qpos[7] + self.data.qpos[8])])
        p = self.phase_forces.setdefault(self.phase, dict(
            native_peak_N=0., native_overforce_s=0., finger_depth_m=0.,
            table_depth_m=0., command_chunks=0))
        p['native_peak_N'] = max(p['native_peak_N'], float(max(stats[:2])))
        p['native_overforce_s'] += float(stats[2])
        p['finger_depth_m'] = max(p['finger_depth_m'], float(stats[3]))
        p['table_depth_m'] = max(p['table_depth_m'], float(stats[4]))
        p['command_chunks'] += 1

    def run(self):
        row = super().run()
        row.update(
            action_name=self.intervention['name'], action_spec=deepcopy(self.intervention),
            scene_overrides=deepcopy(self.scene_overrides),
            initial_state_sha256=self.initial_state_sha256,
            physics_sha256=self.physics_sha256,
            initial_controller_observation=deepcopy(self.initial_snapshot['observation']),
            phase_force_summary=deepcopy(self.phase_forces),
            action_command_count=len(self.action_calls),
            diagnostic_force_chunks=len(self.force_chunks),
            policy_digest_before=self.before_policy_digest,
            policy_digest_after=self.policy.digest(),
            experiment_scope='fixed-action paired capability diagnostic, not RL',
            scene_parameter_timing='mass/friction installed after common reset, before run',
        )
        return row
