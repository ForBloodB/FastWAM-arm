"""One-decision neural contextual bandit for bounded residual action choices.

This is NOT SAC or temporal-difference RL. An entire completed grasp supplies one
utility target for its selected action. The caller owns sensor features, allowed
actions, reward calculation and train/adaptation/test splits. This module has no
environment, robot, hardware or filesystem interfaces.

Offline replay is an immutable anchor. Online updates mix completed adaptation
trials with that anchor to limit forgetting. ``act(..., explore=False)`` is fully
read-only; evaluation must never call ``update``. Exploration defaults to 10% and
is used only when the caller explicitly requests it during adaptation.
"""

from collections import deque
import hashlib
import json
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pack(value):
    """JSON encoding preserving tensor dtypes and optimizer integer map keys."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {"tensor": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape), "data": tensor.reshape(-1).tolist()}
    if isinstance(value, dict):
        return {"mapping": [[_pack(k), _pack(v)] for k, v in
                            sorted(value.items(), key=lambda item: repr(item[0]))]}
    if isinstance(value, tuple):
        return {"tuple": [_pack(v) for v in value]}
    if isinstance(value, list):
        return [_pack(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    if isinstance(value, dict):
        if "tensor" in value:
            dtypes = {name: getattr(torch, name) for name in
                      ("float32", "float64", "int64", "int32", "uint8", "bool")}
            if value["tensor"] not in dtypes:
                raise ValueError("Unsupported checkpoint tensor dtype")
            return torch.tensor(value["data"], dtype=dtypes[value["tensor"]]).reshape(value["shape"])
        if "mapping" in value:
            return {_unpack(k): _unpack(v) for k, v in value["mapping"]}
        if "tuple" in value:
            return tuple(_unpack(v) for v in value["tuple"])
        raise ValueError("Unknown checkpoint encoding")
    return value


class ResidualBandit:
    """CPU MLP predicting completed-trial utility for each residual action.

    Records are ``(x, action_name, reward)`` or dictionaries containing ``x``,
    ``action`` and ``reward``. Features must be 15 finite sensor-derived numbers;
    normalization is fixed by the caller, never fitted using test observations.
    No undiscounted future value, Bellman target, actor or critic is introduced.

    ``train_offline`` fits chosen-action Huber regression using Adam. ``update``
    takes exactly one *completed* adaptation trial and performs a fixed number
    of replay updates. The module cannot determine whether a supplied reward
    belongs to a completed trial, so the runner must enforce this boundary.
    """

    algorithm = "one_decision_neural_contextual_bandit"
    schema_version = 1
    feature_count = 15

    def __init__(self, action_names, seed=0, hidden=(64, 64), epsilon=0.10,
                 lr=1e-3, batch_size=64, online_steps=8, replay_capacity=512,
                 anchor_fraction=0.50, grad_clip=5.0):
        names = tuple(action_names)
        if not names or any(not isinstance(a, str) or not a for a in names) or len(set(names)) != len(names):
            raise ValueError("action_names must be nonempty unique strings")
        hidden = tuple(hidden)
        if not hidden or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in hidden):
            raise ValueError("hidden must contain positive integer layer widths")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if not math.isfinite(anchor_fraction) or not 0 < anchor_fraction <= 1:
            raise ValueError("anchor_fraction must be in (0, 1]")
        for name, number in (("batch_size", batch_size), ("online_steps", online_steps), ("replay_capacity", replay_capacity)):
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 to mix anchor and online trials")
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(grad_clip) or grad_clip <= 0:
            raise ValueError("lr and grad_clip must be finite positive numbers")
        self.action_names = names
        self.action_index = {name: i for i, name in enumerate(names)}
        self.seed, self.hidden, self.epsilon = int(seed), hidden, float(epsilon)
        self.lr, self.batch_size, self.online_steps = float(lr), batch_size, online_steps
        self.replay_capacity, self.anchor_fraction = replay_capacity, float(anchor_fraction)
        self.grad_clip = float(grad_clip)
        self.rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        # Build deterministically while leaving the caller's Torch RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            layers = []
            widths = (self.feature_count,) + hidden + (len(names),)
            for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(left, right, device="cpu", dtype=torch.float32))
                if i < len(widths) - 2:
                    layers.append(nn.ReLU())
            self.model = nn.Sequential(*layers).cpu().eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._anchor = ()
        self._anchor_digest = _hash([])
        self._online = deque(maxlen=self.replay_capacity)
        self.offline_fit_calls = 0
        self.offline_gradient_steps = 0
        self.online_updates = 0
        self.online_gradient_steps = 0

    @property
    def updates(self):
        """Number of completed adaptation trials supplied to update()."""
        return self.online_updates

    @property
    def anchor_digest(self):
        return self._anchor_digest

    @property
    def replay_size(self):
        return len(self._online)

    def _features(self, x):
        array = np.asarray(x, dtype=np.float32)
        if array.shape != (self.feature_count,) or not np.isfinite(array).all():
            raise ValueError(f"x must contain {self.feature_count} finite features")
        return tuple(float(v) for v in array)

    def _record(self, record):
        if isinstance(record, Mapping):
            x, action, reward = record["x"], record["action"], record["reward"]
        else:
            x, action, reward = record
        if action not in self.action_index:
            raise ValueError(f"Unknown action: {action!r}")
        reward = float(reward)
        if not math.isfinite(reward) or not np.isfinite(np.float32(reward)):
            raise ValueError("reward must be finite in float32")
        return self._features(x), action, reward

    @staticmethod
    def _rows(records):
        return [{"x": list(x), "action": action, "reward": reward}
                for x, action, reward in records]

    def predict(self, x):
        """Return action-name -> predicted utility without modifying any state."""
        with torch.no_grad():
            values = self.model(torch.tensor(self._features(x), dtype=torch.float32)).tolist()
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError("Nonfinite policy prediction")
        return dict(zip(self.action_names, values))

    def act(self, x, allowed_names=None, explore=False):
        """Return an allowed action name; greedy evaluation never consumes RNG."""
        allowed = tuple(self.action_names if allowed_names is None else allowed_names)
        if not allowed or len(set(allowed)) != len(allowed) or any(name not in self.action_index for name in allowed):
            raise ValueError("allowed_names must be a nonempty unique subset of action_names")
        predictions = self.predict(x)
        # Configured order provides stable tie breaking independent of mask order.
        ordered = [name for name in self.action_names if name in allowed]
        if explore and self.rng.random() < self.epsilon:
            return ordered[int(self.rng.integers(len(ordered)))]
        return max(ordered, key=predictions.__getitem__)

    def _fit_batch(self, records):
        features = torch.tensor([row[0] for row in records], dtype=torch.float32)
        actions = torch.tensor([self.action_index[row[1]] for row in records], dtype=torch.long)
        rewards = torch.tensor([row[2] for row in records], dtype=torch.float32)
        self.optimizer.zero_grad(set_to_none=True)
        predicted = self.model(features).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predicted, rewards)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    def _sample(self, records, count):
        indices = torch.randint(len(records), (count,), generator=self.torch_rng).tolist()
        return [records[index] for index in indices]

    def train_offline(self, records=None, epochs=20, steps=None, batch_size=None):
        """Fit immutable source replay; use epochs or override with fixed steps.

        The first call records the anchor. Later calls may reuse it by omitting
        records or supplying identical records in identical order. This permits
        development-only checkpoint selection without replacing training data.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        for label, number in (("batch_size", batch_size), ("epochs", epochs if steps is None else steps)):
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if records is not None:
            anchor = tuple(self._record(row) for row in records)
            if not anchor:
                raise ValueError("Offline records cannot be empty")
            anchor_digest = _hash(self._rows(anchor))
            if self._anchor and anchor_digest != self._anchor_digest:
                raise ValueError("Offline anchor is immutable; create a new learner to replace it")
            self._anchor, self._anchor_digest = anchor, anchor_digest
        if not self._anchor:
            raise ValueError("Initial train_offline call requires records")
        losses = []
        if steps is not None:
            for _ in range(steps):
                losses.append(self._fit_batch(self._sample(self._anchor, batch_size)))
        else:
            for _ in range(epochs):
                indices = torch.randperm(len(self._anchor), generator=self.torch_rng).tolist()
                for start in range(0, len(indices), batch_size):
                    losses.append(self._fit_batch([self._anchor[i] for i in indices[start:start + batch_size]]))
        self.offline_fit_calls += 1
        self.offline_gradient_steps += len(losses)
        return {"loss_mean": float(np.mean(losses)), "loss_last": losses[-1],
                "gradient_steps": len(losses), "anchor_size": len(self._anchor)}

    def update(self, x, name, reward):
        """Learn from one completed adaptation trial, never a test outcome.

        Each minibatch contains at least one anchor and one online record. The
        requested anchor fraction is rounded and capped to keep both sources.
        Replay uses sampling with replacement, including when only one online
        trial is available. No intermediate feedback calls belong here.
        """
        if not self._anchor:
            raise RuntimeError("Train the offline anchor before online adaptation")
        record = self._record((x, name, reward))
        self._online.append(record)
        anchor_count = min(self.batch_size - 1, max(1, round(self.batch_size * self.anchor_fraction)))
        losses = []
        for _ in range(self.online_steps):
            batch = self._sample(self._anchor, anchor_count)
            batch.extend(self._sample(self._online, self.batch_size - anchor_count))
            losses.append(self._fit_batch(batch))
        self.online_updates += 1
        self.online_gradient_steps += len(losses)
        return {"loss_mean": float(np.mean(losses)), "loss_last": losses[-1],
                "gradient_steps": len(losses), "online_updates": self.online_updates,
                "replay_size": len(self._online), "anchor_count_per_batch": anchor_count}

    def state(self):
        """Complete, JSON-safe checkpoint including optimizer, replay and RNG."""
        return {
            "schema_version": self.schema_version, "algorithm": self.algorithm,
            "feature_count": self.feature_count,
            "config": {"action_names": list(self.action_names), "seed": self.seed,
                       "hidden": list(self.hidden), "epsilon": self.epsilon,
                       "lr": self.lr, "batch_size": self.batch_size,
                       "online_steps": self.online_steps, "replay_capacity": self.replay_capacity,
                       "anchor_fraction": self.anchor_fraction, "grad_clip": self.grad_clip},
            "model": _pack(self.model.state_dict()), "optimizer": _pack(self.optimizer.state_dict()),
            "numpy_rng": json.loads(_canonical(self.rng.bit_generator.state)),
            "torch_rng": self.torch_rng.get_state().tolist(),
            "anchor": self._rows(self._anchor), "anchor_digest": self._anchor_digest,
            "online_replay": self._rows(self._online),
            "counters": {"offline_fit_calls": self.offline_fit_calls,
                         "offline_gradient_steps": self.offline_gradient_steps,
                         "online_updates": self.online_updates,
                         "online_gradient_steps": self.online_gradient_steps},
        }

    def load(self, state):
        """Replace this learner from a validated checkpoint; return self."""
        if state.get("schema_version") != self.schema_version or state.get("algorithm") != self.algorithm:
            raise ValueError("Unsupported bandit checkpoint")
        if state.get("feature_count") != self.feature_count:
            raise ValueError("Checkpoint feature count differs")
        restored = type(self)(**state["config"])
        restored.model.load_state_dict(_unpack(state["model"]), strict=True)
        restored.optimizer.load_state_dict(_unpack(state["optimizer"]))
        restored.rng.bit_generator.state = state["numpy_rng"]
        restored.torch_rng.set_state(torch.tensor(state["torch_rng"], dtype=torch.uint8))
        restored._anchor = tuple(restored._record(row) for row in state["anchor"])
        restored._anchor_digest = _hash(restored._rows(restored._anchor))
        if restored._anchor_digest != state["anchor_digest"]:
            raise ValueError("Checkpoint anchor checksum differs")
        replay = tuple(restored._record(row) for row in state["online_replay"])
        if len(replay) > restored.replay_capacity:
            raise ValueError("Checkpoint online replay exceeds configured capacity")
        restored._online.extend(replay)
        for name in ("offline_fit_calls", "offline_gradient_steps", "online_updates", "online_gradient_steps"):
            value = state["counters"][name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid checkpoint counter")
            setattr(restored, name, value)
        for tensor in restored.model.state_dict().values():
            if not torch.isfinite(tensor).all():
                raise ValueError("Checkpoint has nonfinite weights")
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)
        return self

    def clone(self):
        return type(self)(self.action_names).load(self.state())

    def digest(self):
        """Deterministic SHA-256 of every persisted field, including both RNGs."""
        return _hash(self.state())

    def model_digest(self):
        """Weights-only SHA-256 for separately auditing learned model changes."""
        return _hash(_pack(self.model.state_dict()))
