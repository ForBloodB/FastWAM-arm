"""Fast task-space grasp SURROGATE, not a contact physics or real robot experiment.

One step represents one entire scripted pick/place attempt. Contact alignment,
friction and deformation use an explicit synthetic score. No success from this
environment is evidence of a physical grasp or FPGA behavior.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .safety import Envelope

MATERIALS = ('wood', 'plastic', 'sponge', 'metal')
CONDITIONS = ('nominal', 'position', 'friction', 'compliance')


class GraspSurrogate(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, domain='online', condition=None):
        self.domain, self.condition = domain, condition
        self.action_space = spaces.Box(-1., 1., (5,), dtype=np.float32)
        # observed target xyz/yaw, ee xyz, material one-hot, measured width.
        # Unknown friction, bias, compliance and reward-optimal actions are hidden.
        self.observation_space = spaces.Box(-5., 5., (12,), dtype=np.float32)
        self.envelope = Envelope()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.material = int(rng.integers(4))
        self.case = CONDITIONS.index(self.condition) if self.condition else int(rng.integers(4))
        self.xyz = rng.uniform([.20, -.12, .05], [.38, .12, .13])
        self.yaw = rng.uniform(-.6, .6)
        self.width = rng.uniform(.025, .055)
        self.base = np.r_[self.xyz, self.yaw, 6.]
        self.obs = np.r_[(self.xyz-[.29,0,.09])*5, self.yaw,
                         [.29, 0, .25], np.eye(4)[self.material], self.width*10].astype('float32')
        # A reproducible domain shift emulates changed calibration/contact conditions.
        shift = 1.0 if self.domain == 'online' else .35
        force = np.array([-.10, .35, -.65, .60])[self.material]
        self.optimum = np.array([
            shift*(.35 + .12*np.sin(self.yaw*2)),
            shift*(-.28 + .18*self.xyz[1]/.12),
            shift*(.08 + .08*(self.material == 2)),
            shift*(.22*np.sin(self.yaw)), force*.5 + shift*.15])
        if self.case == 1:
            self.optimum[:2] += shift*np.array([.16, -.10])
        elif self.case == 2:
            self.optimum[4] += shift*.25
        elif self.case == 3:
            self.optimum[2] += shift*.12
            self.optimum[4] -= shift*.20*(self.material == 2)
        self.optimum += rng.normal(0, .025, 5)
        self.finished = False
        return self.obs.copy(), {'evidence': 'task_space_surrogate', 'condition': CONDITIONS[self.case]}

    def step(self, action):
        if self.finished:
            raise RuntimeError('reset required before another grasp')
        a = np.asarray(action, float)
        target = self.envelope.project(self.base, a)
        executed = (target-self.base)/np.asarray(self.envelope.residual_scale)
        error = executed-self.optimum
        alignment = np.linalg.norm(error[:2])
        slip = abs(error[4]) > .44
        damage = self.material == 2 and error[4] > .34
        success = bool(alignment < .30 and abs(error[2]) < .30 and abs(error[3]) < .35 and not slip and not damage)
        reward = float(1.5*success - 3*np.mean(error**2) - .015*np.mean(executed**2))
        self.finished = True
        info = {'is_success': success, 'slip': bool(slip), 'overclamp': bool(damage),
                'failure': 'ok' if success else 'overclamp' if damage else 'slip' if slip else 'alignment',
                'material': MATERIALS[self.material], 'condition': CONDITIONS[self.case],
                'target': target.tolist(), 'evidence': 'task_space_surrogate',
                'interventions': 0, 'simulated_task_duration_s': float(4.+2*alignment)}
        return self.obs.copy(), reward, True, False, info
