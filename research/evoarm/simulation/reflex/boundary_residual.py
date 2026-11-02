"""Observable, bounded residual repair over the frozen landing_smooth baseline.

Regrasp opens at the measured joint pose before translating the arm. A visual
metal label and a low observed object height gate release. The support-plane
height is the declared task geometry, not a read of simulated object state.
The optional soft relief changes control references only after the common
probe; the native evaluator and physical material parameters remain unchanged.
"""
from copy import deepcopy
import numpy as np
from .contact_state import ContactStateEnv
from .fixed_residual import FixedResidualEnv, ACTIONS as OLD_ACTIONS

SUPPORT_PLANE_Z_M = .100
ACTIONS = deepcopy(OLD_ACTIONS)
ACTIONS['soft_relief'] = dict(lift_scale=2., hold_s=0., pitch_rad=0., soft_scale=.70)
ACTION_NAMES = tuple(ACTIONS)


def available_actions(x, raw=None):
    """Safety mask from RGBD, encoders/FK and delayed force only."""
    x = np.asarray(x, dtype=float)
    if x.shape != (15,) or not np.isfinite(x).all():
        raise ValueError('Invalid boundary residual feature vector')
    raw = raw or {}
    allowed = list(ACTION_NAMES[:5])
    label = raw.get('vision_label')
    if label == 3 and x[4] > .5 and not x[13] and raw.get('vision_valid'):
        safety = raw.get('regrasp_safety', {})
        if safety.get('permitted', False) and max(x[10:12]) < .80:
            allowed.extend(['pitch_pos_regrasp', 'pitch_neg_regrasp'])
    if label == 2 and x[3] > .5:
        allowed.append('soft_relief')
    return allowed


def observed_release_context(probe, estimated_size, estimate, raw):
    """Check visual object bounds against the task's declared support plane."""
    reasons = []
    result = dict(support_plane_z_m=SUPPORT_PLANE_Z_M,
                  support_plane_source='declared fixed task geometry',
                  permitted=False, reasons=reasons)
    if not (probe.get('valid') and probe.get('pose_valid') and raw.get('vision_valid')):
        reasons.append('pose_unavailable')
        return result
    xyz = np.asarray(probe.get('xyz'), float)
    extent = np.asarray(estimated_size, float)
    reference = np.asarray(estimate, float)
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (xyz, extent, reference)):
        reasons.append('nonfinite_geometry')
        return result
    bottom = float(xyz[2] - extent[2] / 2 - SUPPORT_PLANE_Z_M)
    lateral = float(np.linalg.norm(np.asarray(raw['object_probe_displacement_m'])[:2]))
    displacement = float(np.linalg.norm(xyz[:2] - reference[:2]))
    weak = bool(raw['object_probe_displacement_m'][2] <
                max(.003, .65 * raw['tcp_probe_displacement_m'][2]))
    result.update(observed_object_xyz_m=xyz.tolist(), estimated_size_m=extent.tolist(),
                  observed_bottom_height_m=bottom, lateral_probe_displacement_m=lateral,
                  lateral_from_initial_estimate_m=displacement, weak_probe=weak)
    if raw.get('vision_label') != 3:
        reasons.append('not_visually_metal')
    if not (-.005 <= bottom <= .020):
        reasons.append('release_height_outside_bounds')
    if not (.24 < xyz[0] < .40 and -.14 < xyz[1] < .12):
        reasons.append('outside_pickup_workspace')
    if not (.015 <= extent[1] <= .055 and .015 <= extent[2] <= .065):
        reasons.append('observed_size_outside_bounds')
    if lateral > .012 or displacement > .025:
        reasons.append('excessive_observed_displacement')
    if not weak:
        reasons.append('probe_already_supported')
    result['permitted'] = not reasons
    return result


class BoundaryResidualEnv(FixedResidualEnv):
    available_actions = staticmethod(available_actions)

    def __init__(self, action='zero', action_selector=None, **kwargs):
        if action not in ACTIONS:
            raise ValueError(action)
        self.boundary_events = []
        self.soft_control_log = []
        self._soft_relief_active = False
        self._soft_control_depth = 0
        self._probe_safety = None
        self._probe_time = None
        # The inherited constructor knows the original seven action names.
        super().__init__('zero' if action == 'soft_relief' else action,
                         action_selector=action_selector, **kwargs)
        self.requested_action = action

    def reset(self, **kwargs):
        self.boundary_events = []
        self.soft_control_log = []
        self._soft_relief_active = False
        self._soft_control_depth = 0
        self._probe_safety = None
        self._probe_time = None
        return super().reset(**kwargs)

    def probe_features(self, closed, probe, close_tcp, close_opening):
        # Call the frozen feature calculation directly; the old wrapper's
        # global action mask is intentionally not used for this new version.
        x, raw = ContactStateEnv.probe_features(self, closed, probe, close_tcp, close_opening)
        if not self._decision_ready:
            self._probe_safety = observed_release_context(probe, self.estimated_size, self.estimate, raw)
            self._probe_time = float(self.data.time)
            actor_raw = deepcopy(raw)
            actor_raw['regrasp_safety'] = deepcopy(self._probe_safety)
            allowed = available_actions(x, actor_raw)
            actor_raw['available_actions'] = list(allowed)
            requested = self.requested_action if self.action_selector is None else self.action_selector(x.copy(), actor_raw)
            if requested not in ACTIONS:
                raise ValueError('Unknown boundary residual action: ' + str(requested))
            executed = requested if requested in allowed else 'zero'
            self.residual_decision = dict(
                time_s=float(self.data.time), features=x.tolist(), observation=deepcopy(raw),
                requested_action=requested, selected_action=requested, executed_action=executed,
                available_actions=list(allowed), allowed_actions=list(allowed),
                masked=requested != executed, specification=deepcopy(ACTIONS[executed]),
                regrasp_safety=deepcopy(self._probe_safety),
                execution_status='not_started', mask_version='boundary_open_first_v1')
            self._decision_ready = True
        # Keep the baseline's own decision/weak-probe branch exactly unchanged.
        return x, raw

    def _consume_residual(self, yaw):
        decision = self.residual_decision
        spec = decision['specification']
        if spec['pitch_rad']:
            reasons = []
            if not self._probe_safety or not self._probe_safety['permitted']:
                reasons.append('release_context_invalid')
            if self._saved_descent is None:
                reasons.append('descent_pose_missing')
            if self._probe_time is None or self.data.time - self._probe_time > .10:
                reasons.append('probe_pose_stale')
            if max(self.sensor_force) >= .8 * self.force_budget:
                reasons.append('delayed_force_high')
            if reasons:
                decision.update(executed_action='zero', masked=True,
                                specification=deepcopy(ACTIONS['zero']),
                                execution_status='fallback_before_motion', fallback_reasons=reasons)
            else:
                decision['execution_status'] = 'macro_attempted'
        elif decision['executed_action'] == 'soft_relief':
            self._soft_relief_active = True
            decision['execution_status'] = 'soft_control_active'
        return super()._consume_residual(yaw)

    def _open_at_measured_joints(self, yaw):
        self._joint_hold_q = self.data.qpos[:6].copy()
        release_at = self.data.site_xpos[self.tcp].copy() - self.offset
        self.boundary_events.append(dict(event='stationary_open_before_arm_translation',
                                         time_s=float(self.data.time), tcp_m=release_at.tolist(),
                                         measured_joint_q=self._joint_hold_q.tolist()))
        try:
            self.move(release_at, yaw, .80, 'regrasp_release', .035, 12.)
            self.move(release_at, yaw, .35, 'regrasp_settle', .035, 12.)
        finally:
            self._joint_hold_q = None

    def _check_open(self):
        opening = float(self.data.qpos[7] + self.data.qpos[8])
        required = min(.065, max(.045, float(self.estimated_size[1]) + .006))
        delayed_force = float(max(self.sensor_force))
        if opening < required or delayed_force > .10 * self.force_budget:
            raise RuntimeError('boundary_regrasp_not_open')
        return dict(opening_encoder_m=opening, required_opening_m=required,
                    delayed_max_force_N=delayed_force)

    def _settled_observation(self, name):
        seen = self.observe(name, prior_xy=self.estimate[:2])
        if not (seen.get('valid') and seen.get('pose_valid')):
            raise RuntimeError('boundary_regrasp_pose_unavailable')
        xyz = np.asarray(seen['xyz'], float)
        delta = xyz - self.estimate
        bottom = xyz[2] - self.estimated_size[2] / 2 - SUPPORT_PLANE_Z_M
        if (not np.isfinite(xyz).all() or not -.005 <= bottom <= .006 or
                np.linalg.norm(delta[:2]) > .020 or abs(delta[2]) > .010):
            raise RuntimeError('boundary_regrasp_settle_bounds')
        return seen, xyz, delta

    def _regrasp(self, low, yaw):
        # Deliberately omit the old closed-gripper retry_lower. Once opening
        # begins, a later guard failure remains an attempted pitch action.
        self.retries += 1
        self._open_at_measured_joints(yaw)
        self.boundary_events.append(dict(event='open_verified', **self._check_open(),
                                         time_s=float(self.data.time)))
        seen, xyz, delta = self._settled_observation('boundary_release_settled')
        clear = self.data.site_xpos[self.tcp].copy() - self.offset
        clear[2] = max(clear[2] + .020, low[2] + .035,
                       xyz[2] + self.estimated_size[2] / 2 + .015)
        if clear[2] - (self.data.site_xpos[self.tcp, 2] - self.offset[2]) > .055:
            raise RuntimeError('boundary_regrasp_clearance_bounds')
        self.move(clear, yaw, .90, 'regrasp_clear', .035, 12.)
        seen, xyz, delta = self._settled_observation('boundary_regrasp_clear')
        # Carry the bounded fresh pickup correction through the parent's local
        # low/probe/high poses. FixedResidualEnv clears it before transport.
        self.offset = np.clip(self.offset + np.clip(delta, [-.006, -.006, -.003], [.006, .006, .003]),
                              [-.006, -.006, -.003], [.006, .006, .003])
        self.residual_yaw = float(np.clip(seen['yaw'] - yaw, -.15, .15))
        self.boundary_events.append(dict(event='open_visual_recenter', correction_m=self.offset.tolist(),
                                         observed_xyz_m=xyz.tolist(), time_s=float(self.data.time)))
        self._check_open()
        # The inherited move hook activates pitch here, after both open checks.
        self.move(clear, yaw, .80, 'regrasp_align', .035, 12.)
        self.move(low, yaw, 1.10, 'regrasp_descend', .035, 12.)
        self.move(low, yaw, .25, 'hold', .035, 12.)
        self._check_open()
        self.move(low, yaw, 2.2, 'retry_close', 0., self.drive)
        self.regrasp_done = True
        self.residual_decision['execution_status'] = 'macro_complete'

    def _guarded_move(self, xyz, yaw, duration, phase, gap, force,
                      apply_residual=True, track=False):
        if not self._soft_relief_active or self._soft_control_depth:
            return super()._guarded_move(xyz, yaw, duration, phase, gap, force, apply_residual, track)
        original = (self.force_budget, self.start_drive, self.reference_drive)
        scale = ACTIONS['soft_relief']['soft_scale']
        self.force_budget *= scale
        self.start_drive *= scale
        self.reference_drive *= scale
        control_cap = min(28., 1.8 * self.force_budget)
        self.preload_drive = float(np.clip(self.preload_drive, 0., control_cap))
        self._soft_control_depth += 1
        self.soft_control_log.append(dict(phase=phase, time_s=float(self.data.time),
                                         original_predicted_budget_N=float(original[0]),
                                         control_budget_N=float(self.force_budget),
                                         control_start_drive_N=float(self.start_drive),
                                         control_reference_drive_N=float(self.reference_drive),
                                         control_cap_N=float(control_cap), scale=scale))
        try:
            return super()._guarded_move(xyz, yaw, duration, phase, gap, force, apply_residual, track)
        finally:
            self.force_budget, self.start_drive, self.reference_drive = original
            self._soft_control_depth -= 1

    def run(self):
        row = super().run()
        row.update(residual_controller='BoundaryResidualEnv(open_first_v1)',
                   zero_reference='DriveLandingEnv(landing_smooth)',
                   boundary_events=deepcopy(self.boundary_events),
                   soft_control_log=deepcopy(self.soft_control_log),
                   residual_feature_count=15,
                   scoring_and_physics='Inherited unchanged; soft relief scales temporary control references only')
        return row
