"""Bounded post-probe residuals over the frozen landing_smooth controller.

This module never changes the base controller, physics, drive limit, scoring,
or initial state. Learning is supplied by an external action_selector callback.
All residual decisions use the existing 15 sensor features and observation.
"""
from copy import deepcopy
import numpy as np
from .drive_landing import DriveLandingEnv

ACTION_NAMES = (
    'zero', 'slow2', 'slow3', 'hold05_slow2', 'hold10_slow2',
    'pitch_pos_regrasp', 'pitch_neg_regrasp',
)
ACTIONS = {
    'zero': dict(lift_scale=1., hold_s=0., pitch_rad=0.),
    'slow2': dict(lift_scale=2., hold_s=0., pitch_rad=0.),
    'slow3': dict(lift_scale=3., hold_s=0., pitch_rad=0.),
    'hold05_slow2': dict(lift_scale=2., hold_s=.5, pitch_rad=0.),
    'hold10_slow2': dict(lift_scale=2., hold_s=1., pitch_rad=0.),
    'pitch_pos_regrasp': dict(lift_scale=2., hold_s=0., pitch_rad=.08),
    'pitch_neg_regrasp': dict(lift_scale=2., hold_s=0., pitch_rad=-.08),
}


def available_actions(x, raw=None):
    """Observable mask only; no material truth, mass or friction input."""
    x = np.asarray(x, dtype=float)
    if x.shape != (15,) or not np.isfinite(x).all():
        raise ValueError('Invalid residual feature vector')
    allowed = list(ACTION_NAMES[:5])
    valid_vision = not bool(x[13]) and bool((raw or {}).get('vision_valid', True))
    if valid_vision and max(x[10:12]) < .80:
        allowed.extend(ACTION_NAMES[5:])
    return allowed


class FixedResidualEnv(DriveLandingEnv):
    """One sensor-conditioned decision after the unmodified baseline probe.

    action_selector(x, raw) -> action name is called once per attempt. A masked
    request executes zero and is logged, so callers can train on the executed
    action rather than accidentally crediting an unavailable action.
    """
    available_actions = staticmethod(available_actions)

    def __init__(self, action='zero', action_selector=None, **kwargs):
        if action not in ACTIONS:
            raise ValueError(action)
        if kwargs.get('learning', False):
            raise ValueError('Use external residual learning, not the frozen parent policy')
        self.requested_action = action
        self.action_selector = action_selector
        self._joint_hold_q = None
        self.residual_move_commands = []
        self._inside_residual_macro = False
        self._decision_ready = False
        self._residual_applied = False
        self._saved_descent = None
        self._pitch_requested = 0.
        self._pickup_offset = None
        self.residual_decision = None
        self.pitch_activation_log = []
        super().__init__('landing_smooth', **kwargs)

    def reset(self, **kwargs):
        self._joint_hold_q = None
        self.residual_move_commands = []
        self._inside_residual_macro = False
        self._decision_ready = False
        self._residual_applied = False
        self._saved_descent = None
        self._pitch_requested = 0.
        self._pickup_offset = None
        self.residual_decision = None
        self.pitch_activation_log = []
        # The per-environment intervention dict is a deep copy of the frozen
        # bank. Clear the previous attempt's pitch before the common reset.
        self.intervention['pitch_rad'] = 0.
        return super().reset(**kwargs)

    def probe_features(self, closed, probe, close_tcp, close_opening):
        x, raw = super().probe_features(closed, probe, close_tcp, close_opening)
        if not self._decision_ready:
            allowed = available_actions(x, raw)
            requested = self.requested_action
            if self.action_selector is not None:
                selector_observation = deepcopy(raw)
                selector_observation['available_actions'] = list(allowed)
                requested = self.action_selector(x.copy(), selector_observation)
            if requested not in ACTIONS:
                raise ValueError('Unknown residual action: ' + str(requested))
            executed = requested if requested in allowed else 'zero'
            self.residual_decision = dict(
                time_s=float(self.data.time), features=x.tolist(),
                observation=deepcopy(raw), requested_action=requested,
                selected_action=requested, executed_action=executed,
                available_actions=list(allowed), allowed_actions=allowed,
                masked=requested != executed, specification=deepcopy(ACTIONS[executed]))
            self._decision_ready = True
        return x, raw

    def ik(self, xyz, yaw, seed):
        if self._joint_hold_q is not None:
            return self._joint_hold_q.copy()
        return super().ik(xyz, yaw, seed)

    def _residual_joint_hold(self, duration, yaw):
        # 'preload' would silently enforce a minimum 0.8 s in the base class.
        # A named phase plus measured-joint IK hook implements the exact hold
        # duration while retaining the same arm and grip control laws.
        self._joint_hold_q = self.data.qpos[:6].copy()
        tcp = self.data.site_xpos[self.tcp].copy() - self.offset
        try:
            self.move(tcp, yaw, duration, 'residual_hold', 0., self.drive)
        finally:
            self._joint_hold_q = None

    def _consume_residual(self, yaw):
        self._residual_applied = True  # prevent recursion through macro moves
        spec = self.residual_decision['specification']
        self._inside_residual_macro = True
        try:
            if spec['hold_s']:
                self._residual_joint_hold(spec['hold_s'], yaw)
            if spec['pitch_rad']:
                if self._saved_descent is None:
                    raise RuntimeError('residual_missing_descent_pose')
                low, grasp_yaw = self._saved_descent
                self._pitch_requested = float(spec['pitch_rad'])
                # Reuse the frozen measured-release/recentering trajectory.
                # Pitch activates only at open-gripper alignment below.
                self._regrasp(low.copy(), grasp_yaw)
                self._pickup_offset = self.offset.copy()
                probe = low.copy()
                probe[2] += .015
                self.move(probe, grasp_yaw, .65, 'probe_lift', 0., self.drive)
                seen = self.observe('residual_probe_retry', prior_xy=self.estimate[:2])
                self.follow_relative = (
                    np.asarray(seen['xyz']) - self.data.site_xpos[self.tcp]
                    if seen.get('valid') else None)
                self.slip_anchor = None
        finally:
            self._inside_residual_macro = False

    def move(self, xyz, yaw, duration, phase, gap, force,
             apply_residual=True, track=False):
        if phase == 'descend' and not self._decision_ready:
            self._saved_descent = (np.asarray(xyz, float).copy(), float(yaw))
        if (self._decision_ready and not self._residual_applied
                and phase in ['preload', 'lift']):
            self._consume_residual(yaw)
        if phase == 'regrasp_align' and self._inside_residual_macro and self._pitch_requested:
            opening = float(self.data.qpos[7] + self.data.qpos[8])
            required = min(.065, max(.045, float(self.estimated_size[1]) + .006))
            measured_force = float(max(self.sensor_force))
            if opening < required or measured_force > .20 * self.force_budget:
                raise RuntimeError('residual_regrasp_not_open')
            self.intervention['pitch_rad'] = self._pitch_requested
            self.pitch_activation_log.append(dict(
                phase=phase, time_s=float(self.data.time),
                opening_encoder_m=opening, required_opening_m=required,
                delayed_max_force_N=measured_force,
                predicted_budget_N=float(self.force_budget),
                pitch_rad=self._pitch_requested))
        if phase == 'transport_a' and self._pickup_offset is not None:
            # The old regrasp helper's visual correction is a pickup offset.
            # Clear it for the fixed destination so the new measured-relative
            # landing correction is not followed by a second global offset.
            self.events.append(dict(event='residual_pickup_offset_cleared',
                                    offset_m=self.offset.tolist()))
            self.offset = np.zeros(3)
            self._pickup_offset = None
        effective_duration = float(duration)
        if self._decision_ready and self._residual_applied and phase in ['lift', 'probe_lift']:
            effective_duration *= self.residual_decision['specification']['lift_scale']
        start_time = float(self.data.time)
        joint_start = self.data.qpos[:6].copy()
        record = dict(phase=phase, target_m=np.asarray(xyz, float).tolist(),
                      yaw_rad=float(yaw), requested_duration_s=float(duration),
                      residual_duration_s=effective_duration,
                      pitch_rad=float(self.intervention['pitch_rad']),
                      joint_hold=self._joint_hold_q is not None,
                      macro=self._inside_residual_macro,
                      drive_cap_argument_N=float(force), start_time_s=start_time,
                      joint_start=joint_start.tolist())
        try:
            return super().move(xyz, yaw, effective_duration, phase, gap, force,
                                apply_residual, track)
        finally:
            record.update(actual_duration_s=float(self.data.time) - start_time,
                          joint_command=self.command.tolist())
            self.residual_move_commands.append(record)

    def run(self):
        row = super().run()
        row.update(
            fixed_controller='DriveLandingEnv(landing_smooth)',
            residual_decision=deepcopy(self.residual_decision),
            residual_move_commands=deepcopy(self.residual_move_commands),
            residual_pitch_activations=deepcopy(self.pitch_activation_log),
            residual_action='zero' if self.residual_decision is None else self.residual_decision['executed_action'],
            residual_policy_observation_privileged=False,
            residual_learning_scope='External policy callback; no update inside simulator')
        return row
