"""Bounded contact-location and open-first repair experiments, simulation only.

Every condition retains the repaired controller, fixed slow2 action, contact
physics, scorer, and shared drive cap. Initial geometry interventions and true
post-probe recoveries are distinct experimental families. Control reads RGBD,
arm/finger encoders and delayed force only; no object state is written.
"""
from copy import deepcopy
import numpy as np

from .timed_contact import TimedContactEnv
from .contact_state import ContactStateEnv
from .boundary_residual import observed_release_context


def _spec(family, local_xyz=(0., 0., 0.), pitch=0., hold=0., align=False):
    return dict(family=family, local_offset_m=list(local_xyz),
                pitch_rad=float(pitch), extra_hold_s=float(hold),
                open_alignment=bool(align or pitch))


REPAIRS = {
    'none': _spec('baseline'),
    'x_minus3': _spec('initial_geometry', (-.003, 0., 0.)),
    'x_plus3': _spec('initial_geometry', (.003, 0., 0.)),
    'y_minus2': _spec('initial_geometry', (0., -.002, 0.)),
    'y_plus2': _spec('initial_geometry', (0., .002, 0.)),
    'z_minus2': _spec('initial_geometry', (0., 0., -.002)),
    'z_plus2': _spec('initial_geometry', (0., 0., .002)),
    'pitch_pos': _spec('initial_geometry', pitch=.08),
    'pitch_neg': _spec('initial_geometry', pitch=-.08),
    'align_only': _spec('initial_alignment', align=True),
    'hold030': _spec('initial_preload', hold=.30),
    'z_minus2_pitch_pos': _spec('initial_combination', (0., 0., -.002), pitch=.08),
    'z_minus2_pitch_neg': _spec('initial_combination', (0., 0., -.002), pitch=-.08),
    'recover_center': _spec('recovery'),
    'recover_z_minus2': _spec('recovery', (0., 0., -.002)),
    'recover_pitch_pos': _spec('recovery', pitch=.08),
    'recover_pitch_neg': _spec('recovery', pitch=-.08),
}

# Targets in these phases are expressed relative to the original pickup pose.
# Measured-pose holds and the transport/destination phases are intentionally
# absent. No force-controller or material parameter changes accompany offsets.
PICKUP_TARGET_PHASES = frozenset({
    'descend', 'descend_settle', 'close', 'preload', 'probe_lift', 'lift',
    'retry_lower', 'retry_close', 'regrasp_clear', 'regrasp_align',
    'regrasp_descend',
})


def local_offset_to_world(local_xyz, yaw):
    """Horizontal yaw-aligned x/y and world z; not the tilted tool frame."""
    x, y, z = np.asarray(local_xyz, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c*x-s*y, s*x+c*y, z])


class ContactRepairEnv(TimedContactEnv):
    def __init__(self, repair='none', timing='legacy', **kwargs):
        if repair not in REPAIRS:
            raise ValueError('Unknown contact repair: '+str(repair))
        if kwargs.pop('action', 'slow2') != 'slow2':
            raise ValueError('Contact repair experiments use the fixed slow2 action')
        if kwargs.get('action_selector') is not None or kwargs.get('learning', False):
            raise ValueError('Contact repair is a fixed-action diagnostic, not training')
        self.repair = repair
        self.repair_spec = deepcopy(REPAIRS[repair])
        self._clear_repair()
        super().__init__(timing=timing, action='slow2', **kwargs)

    def _clear_repair(self):
        self.repair_events = []
        self.repair_first_probe = None
        self.repair_post_probe = None
        self._initial_started = False
        self._initial_offset = np.zeros(3)
        self._initial_hold_done = False
        self._pickup_cleared = False
        self._recovery_considered = False
        self._recovery_attempted = False
        self._repair_macro_running = False

    def reset(self, **kwargs):
        self._clear_repair()
        return super().reset(**kwargs)

    def _event(self, event, **kwargs):
        self.repair_events.append(dict(event=event, time_s=float(self.data.time), **kwargs))

    def move(self, xyz, yaw, duration, phase, gap, force,
             apply_residual=True, track=False):
        spec = self.repair_spec
        initial = spec['family'].startswith('initial_')
        target = np.asarray(xyz, dtype=float).copy()
        if phase == 'descend' and initial and not self._initial_started:
            self._initial_started = True
            self._initial_offset = local_offset_to_world(spec['local_offset_m'], yaw)
            self._event('initial_contact_intervention',
                        local_offset_m=spec['local_offset_m'],
                        world_offset_m=self._initial_offset.tolist(),
                        pitch_rad=spec['pitch_rad'],
                        reference_yaw_rad=float(yaw))
            if spec['open_alignment']:
                # Rotation starts in free space with the gripper verified open.
                check = self._check_open()
                self.intervention['pitch_rad'] = spec['pitch_rad']
                self.intervention_started = True
                self.residual_pitch = spec['pitch_rad']
                current = self.data.site_xpos[self.tcp].copy() - self.offset
                self._event('initial_open_alignment', **check,
                            requested_pitch_rad=spec['pitch_rad'])
                super().move(current, yaw, .80, 'repair_initial_align', .035,
                             12., apply_residual, False)
        if (phase == 'probe_lift' and not self._decision_ready and initial
                and spec['extra_hold_s'] and not self._initial_hold_done):
            self._initial_hold_done = True
            start = float(self.data.time)
            # The existing protected residual_hold preserves observed support;
            # unlike preload it does not silently impose a minimum 0.8 s.
            self._residual_joint_hold(spec['extra_hold_s'], yaw)
            self._event('extra_measured_joint_hold', requested_s=spec['extra_hold_s'],
                        actual_s=float(self.data.time)-start,
                        reference_limits_unchanged=True,
                        control_mode='measured_support_hold')
        if (initial and self._initial_started and not self._pickup_cleared
                and phase in PICKUP_TARGET_PHASES and not self.in_tracking_tick):
            target += self._initial_offset
            if np.any(self._initial_offset):
                self._event('pickup_target_offset', phase=phase,
                            original_target_m=np.asarray(xyz, float).tolist(),
                            requested_target_m=target.tolist(),
                            offset_m=self._initial_offset.tolist())
        if phase == 'transport_a' and initial and self._initial_started and not self._pickup_cleared:
            self._pickup_cleared = True
            self._event('initial_pickup_offset_finished',
                        offset_m=self._initial_offset.tolist(),
                        pitch_retained_for_closed_gripper=spec['pitch_rad'])
        return super().move(target, yaw, duration, phase, gap, force,
                            apply_residual, track)

    def _recovery_context(self, probe, raw):
        context = observed_release_context(probe, self.estimated_size, self.estimate, raw)
        reasons = list(context['reasons'])
        # Tighten, never relax, the inherited release mask. A weakly lifted
        # object above this limit is left to the baseline rather than dropped.
        if context.get('observed_bottom_height_m', float('inf')) > .006:
            reasons.append('release_bottom_above_6mm')
        if max(self.sensor_force) >= .8*self.force_budget:
            reasons.append('delayed_force_high')
        if self._saved_descent is None:
            reasons.append('descent_pose_missing')
        context.update(permitted=not reasons, reasons=reasons,
                       permitted_release_bottom_max_m=.006)
        return context

    def probe_features(self, closed, probe, close_tcp, close_opening):
        x, raw = super().probe_features(closed, probe, close_tcp, close_opening)
        if self.repair_first_probe is None:
            self.repair_first_probe = dict(time_s=float(self.data.time),
                                          features=x.tolist(), observation=deepcopy(raw))
        if (self.repair_spec['family'] != 'recovery'
                or self._recovery_considered or self._repair_macro_running):
            return x, raw
        self._recovery_considered = True
        weak = bool(raw['vision_valid'] and raw['object_probe_displacement_m'][2]
                    < max(.003, .65*raw['tcp_probe_displacement_m'][2]))
        context = self._recovery_context(probe, raw)
        self._event('recovery_trigger', weak_probe=weak,
                    permitted=bool(weak and context['permitted']),
                    release_context=context, observation=deepcopy(raw))
        if not weak or not context['permitted']:
            return x, raw
        self._recovery_attempted = True
        self._repair_macro_running = True
        old_macro = self._inside_residual_macro
        try:
            # Consume slow2 exactly once before invoking the inherited macro.
            # The original first-probe decision remains unchanged for pairing.
            low, yaw = self._saved_descent
            low = low.copy() + local_offset_to_world(self.repair_spec['local_offset_m'], yaw)
            self._consume_residual(yaw)
            self._inside_residual_macro = True
            self._pitch_requested = self.repair_spec['pitch_rad']
            self._event('recovery_begin', requested_low_m=low.tolist(),
                        requested_pitch_rad=self._pitch_requested,
                        retry_limit=1, release_context=context)
            # BoundaryResidualEnv._regrasp opens at measured joints, verifies
            # opening, observes settlement, recentres and rotates only open.
            self._regrasp(low, yaw)
            self._pickup_offset = self.offset.copy()
            new_closed = self.observe('contact_repair_after_close', prior_xy=self.estimate[:2])
            new_close_tcp = self.data.site_xpos[self.tcp].copy()
            new_close_opening = float(self.data.qpos[7]+self.data.qpos[8])
            new_probe = low.copy()
            new_probe[2] += .015
            self.move(new_probe, yaw, .65, 'probe_lift', 0., self.drive)
            seen = self.observe('contact_repair_probe', prior_xy=self.estimate[:2])
            new_x, new_raw = ContactStateEnv.probe_features(
                self, new_closed, seen, new_close_tcp, new_close_opening)
            self.repair_post_probe = dict(time_s=float(self.data.time),
                                         features=new_x.tolist(), observation=deepcopy(new_raw))
            # SafeContactEnv.run retains this dictionary as `seen`. Updating it
            # prevents a stale first-probe pose from anchoring transport after
            # the genuine physical recovery. No object/physics state is reset.
            # observe() also stores the same dictionary in its audit list;
            # detach that original record before updating the caller's value.
            for index, observation in enumerate(self.vision_calls):
                if observation is probe:
                    self.vision_calls[index] = deepcopy(probe)
            probe.clear()
            probe.update(deepcopy(seen))
            self.follow_relative = (np.asarray(seen['xyz'])-self.data.site_xpos[self.tcp]
                                    if seen.get('valid') else None)
            self.slip_anchor = None
            self._event('recovery_complete', post_probe=deepcopy(new_raw),
                        pickup_offset_m=self.offset.tolist())
            return new_x, new_raw
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self._event('recovery_failed', reason=str(exc))
            raise
        finally:
            self._inside_residual_macro = old_macro
            self._repair_macro_running = False

    def run(self):
        row = super().run()
        row.update(contact_repair=self.repair,
                   contact_repair_spec=deepcopy(self.repair_spec),
                   repair_events=deepcopy(self.repair_events),
                   repair_first_probe=deepcopy(self.repair_first_probe),
                   repair_post_probe=deepcopy(self.repair_post_probe),
                   recovery_attempted=self._recovery_attempted,
                   recovery_attempt_limit=1,
                   repair_scope='initial geometry/preload or post-probe open-first recovery; fixed slow2',
                   repair_feedback='RGBD, encoder/FK and delayed force; no native truth')
        return row
