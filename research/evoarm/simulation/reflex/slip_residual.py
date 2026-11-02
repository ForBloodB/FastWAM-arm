"""Bounded geometric/temporal grasp residuals using only RGB-D/encoder/force.

Development and formal runs live separately. Physics, native joint controller,
force limits, and terminal success evaluator are inherited unchanged.
"""
import hashlib
import json
import numpy as np
import mujoco
from .residual_feedback import ResidualFeedbackEnv
from .official_native import roll
from .official_grasp import orientation, rotation_error
from .slip_native import force_roll

# Candidate library for development; index zero is the zero residual.
OPTIONS = [
    {'dx': dx, 'dz': dz, 'slow': slow, 'force': 1.0}
    for dx, dz, slow in [
        (0., 0., 1.), (-.006, 0., 1.), (.006, 0., 1.),
        (0., .006, 1.), (0., -.004, 1.),
        (-.006, .006, 1.), (.006, .006, 1.),
        (0., 0., 1.6), (-.006, 0., 1.6), (.006, 0., 1.6),
        (0., .006, 1.6), (.006, .006, 1.6),
    ]
]
OPTIONS += [{'dx': 0., 'dz': 0., 'slow': slow, 'force': 1., 'pitch': pitch}
            for pitch in [-.3, -.15, .15, .3] for slow in [1., 1.6]]
OPTIONS += [{'dx': dx, 'dz': dz, 'slow': 1.6, 'force': 1., 'pitch': pitch, 'yaw': yaw}
            for dx, dz, pitch, yaw in [
                (0., 0., 0., -.25), (0., 0., 0., -.12),
                (0., 0., 0., .12), (0., 0., 0., .25),
                (0., -.008, 0., 0.), (.012, -.004, 0., 0.),
                (-.012, -.004, 0., 0.), (0., -.008, .15, 0.),
                (.006, -.004, .15, 0.), (-.006, -.004, .15, 0.),
                (0., -.004, .15, -.12), (0., -.004, .15, .12),
            ]]

class SlipPolicy:
    """Discounted contextual Thompson bandit over bounded motion options.

    One chosen option receives one terminal safe-success outcome. Context is
    RGB-D category and estimated volume only. No phase/case/truth IDs.
    """
    def __init__(self, seed=0, actions=None, decay=.99):
        self.rng = np.random.default_rng(seed)
        self.actions = list(range(len(OPTIONS))) if actions is None else list(actions)
        self.decay = decay
        self.stats = {}
        self.updates = 0
        self.last = None

    def act(self, key, learn=False):
        counts = self.stats.get(key, np.ones((2, len(self.actions))))
        score = self.rng.beta(counts[0], counts[1]) if learn else counts[0] / counts.sum(axis=0)
        best = np.flatnonzero(score == score.max())
        a = int(best[0])
        self.last = (key, a)
        return self.actions[a]

    def update(self, success):
        key, a = self.last
        counts = self.stats.setdefault(key, np.ones((2, len(self.actions))))
        counts[:] = 1 + self.decay * (counts - 1)
        counts[0 if success else 1, a] += 1
        self.updates += 1
        self.last = None

    def state(self):
        return {'stats': {k: v.tolist() for k, v in sorted(self.stats.items())},
                'updates': self.updates, 'actions': self.actions, 'decay': self.decay}

    def load(self, state):
        self.stats = {k: np.array(v, float) for k, v in state['stats'].items()}
        self.updates = state['updates']; self.actions = state['actions']; self.decay = state['decay']
        return self

    def digest(self):
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()


class SlipResidualEnv(ResidualFeedbackEnv):
    def __init__(self, slip_policy=None, learn=False, fixed_action=None, recovery=True, force_hold=False, recovery_step=.06, **kwargs):
        self.command_pitch = 0.
        super().__init__(neutral=True, learning=False, **kwargs)
        self.slip_policy = slip_policy or SlipPolicy()
        self.learn = learn
        self.fixed_action = fixed_action
        self.recovery = recovery
        self.force_hold = force_hold
        self.recovery_step = recovery_step

    def ik(self, xyz, yaw, seed):
        if not self.command_pitch:
            return super().ik(xyz, yaw, seed)
        q = np.clip(np.asarray(seed, float).copy(), self.ranges[:, 0] + 1e-5, self.ranges[:, 1] - 1e-5)
        c, s = np.cos(self.command_pitch), np.sin(self.command_pitch)
        R = orientation(yaw) @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        jp = np.zeros((3, self.model.nv)); jr = jp.copy()
        for _ in range(100):
            self.ikdata.qpos[:6] = q; mujoco.mj_forward(self.model, self.ikdata)
            ep = np.asarray(xyz) - self.ikdata.site_xpos[self.tcp]
            er = rotation_error(R, self.ikdata.site_xmat[self.tcp].reshape(3, 3))
            if np.linalg.norm(ep) < .00008 and np.linalg.norm(er) < .001: break
            mujoco.mj_jacSite(self.model, self.ikdata, jp, jr, self.tcp)
            J = np.vstack([jp[:, :6], .15 * jr[:, :6]])
            delta = J.T @ np.linalg.solve(J @ J.T + .00002 * np.eye(6), np.r_[ep, .15 * er])
            q = np.clip(q + np.clip(delta, -.12, .12), self.ranges[:, 0] + 1e-5, self.ranges[:, 1] - 1e-5)
        self.ikdata.qpos[:6] = q; mujoco.mj_forward(self.model, self.ikdata)
        ep = np.linalg.norm(np.asarray(xyz) - self.ikdata.site_xpos[self.tcp])
        er = np.linalg.norm(rotation_error(R, self.ikdata.site_xmat[self.tcp].reshape(3, 3)))
        if ep > .002 or er > .02: raise RuntimeError(f'ik_unreachable:{ep:.5f}:{er:.5f}')
        return q

    def context(self):
        return f'{int(self.predicted)},{int(np.prod(self.estimated_size) > .000060)}'

    def reset(self, **kwargs):
        self.command_pitch = 0.
        self.option_id = 0
        self.option = OPTIONS[0]
        self.force_recoveries = 0
        self.control_bounds = {'min_cap_N': 30., 'max_cap_N': 0., 'max_translation_m': 0.}
        obs, info = super().reset(**kwargs)
        if not self.failure:
            self.option_id = self.fixed_action if self.fixed_action is not None else self.slip_policy.act(self.context(), self.learn)
            self.option = OPTIONS[self.option_id]
            self.desired_drive = self.drive
        return obs, info

    def move(self, xyz, yaw, duration, phase, gap, force):
        # Geometric residual is applied consistently from descent to placement.
        # It changes TCP targets, never object qpos or model contact properties.
        xyz = np.asarray(xyz, float).copy()
        active = phase in ('close', 'probe_lift', 'retry_lower', 'retry_close', 'lift',
                           'transport', 'transport_a', 'hold', 'lower')
        shifted = phase not in ('approach', 'retreat', 'settle')
        if shifted:
            yaw += self.option.get('yaw', 0.)
            offset = np.array([self.option['dx'] * np.cos(yaw), self.option['dx'] * np.sin(yaw), self.option['dz']])
            xyz += offset
            self.control_bounds['max_translation_m'] = max(self.control_bounds['max_translation_m'], float(np.linalg.norm(offset)))
        if active and phase not in ('close', 'retry_close'):
            duration *= self.option['slow']
        if phase == 'release':
            self.gap = float(np.clip(self.data.qpos[6], 0, .05))
        self.command_pitch = self.option.get('pitch', 0.) if shifted else 0.
        target = self.ik(xyz, yaw, self.command)
        self.phase = phase
        n = round(duration / self.model.opt.timestep)
        if active:
            self.desired_drive = max(self.desired_drive, self.drive)
            self.desired_drive = min(self.desired_drive, 28., 1.8 * self.force_budget)
        for start in range(0, n, 10):
            cap = self.drive if active else force
            if active and len(self.sensor_queue) == 3:
                measured = float(max(self.sensor_force))
                if measured > .88 * self.force_budget:
                    old = self.drive
                    self.drive = max(1.5, self.drive * .80 * self.force_budget / measured)
                    self.force_changes += int(abs(old - self.drive) > 1e-6)
                elif self.recovery and measured < .72 * self.force_budget:
                    # Restore slowly after a transient peak (100 Hz feedback).
                    old = self.drive
                    self.drive = min(self.desired_drive, self.drive + self.recovery_step)
                    self.force_recoveries += int(self.drive > old + 1e-6)
                cap = self.drive
            self.control_bounds['min_cap_N'] = min(self.control_bounds['min_cap_N'], cap)
            self.control_bounds['max_cap_N'] = max(self.control_bounds['max_cap_N'], cap)
            count = min(10, n - start)
            if self.force_hold and active and phase != 'close':
                force_roll(self.model, self.data, self.command, target, cap, start, n, count)
            else:
                roll(self.model, self.data, self.command, target, self.gap, gap, cap, start, n, count)
            self.sample(count * self.model.opt.timestep)
        self.command = target; self.gap = gap
        self.command_updates.append({'phase': phase, 'drive_N': self.drive, 'sensor': self.sensor_observation(),
                                     'tcp_target': xyz.tolist(), 'duration_s': duration})

    def run(self):
        row = super().run()
        # Terminal simulator outcome is training reward, never an action input.
        if self.learn and self.fixed_action is None and not self.failure:
            self.slip_policy.update(bool(row['success']))
        row.update(controller='sensor_feedback_bounded_geometric_temporal_residual',
                   residual_option=self.option_id, residual_parameters=self.option,
                   policy_context=None if self.failure else self.context(),
                   force_recoveries=self.force_recoveries, control_bounds=self.control_bounds,
                   force_hold=self.force_hold, recovery_step=self.recovery_step,
                   reward=float(row['success']), learned_joint_trajectory=False,
                   policy_observation_privileged=False)
        return row
