"""ProCuRL-Target for the binary PointMass task pool."""
import itertools
import random

import numpy as np
import torch
from scipy.spatial.distance import cdist
from stable_baselines3.common.callbacks import BaseCallback

from envs.pointmass.PointMassWrapper import PointmassWrapper


def task_weights(contexts, values, targets, target_values, beta):
    """Compute each task's largest target-weighted potential, in blocks."""
    p = np.clip(np.asarray(values, dtype=np.float64), 0, 1)
    q = np.clip(np.asarray(target_values, dtype=np.float64), 0, 1)
    best = np.zeros(len(contexts))
    for i in range(0, len(targets), 32):
        distances = cdist(targets[i:i + 32], contexts, metric="euclidean")
        block_q = q[i:i + 32, None]
        weighted = np.exp(-distances) * block_q * (1 - block_q)
        best = np.maximum(best, weighted.max(axis=0))
    return np.exp(beta * p * (1 - p) * best)


class ICPointMass(PointmassWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SB3 resets the environment once before the teacher sees its first policy.
        self.cumulative_weights = list(range(1, self.num_of_tasks + 1))
        self.c2ctest = None  # The IC implementation stores only one weight per task.

    def pick_curr_id_proxcorl(self):
        return random.choices(range(self.num_of_tasks), cum_weights=self.cumulative_weights, k=1)[0]


class ICTeacher(BaseCallback):
    def __init__(self, env, period):
        super().__init__()
        self.env = env
        self.period = period
        self.contexts = env.contexts_arr
        self.start_obs = np.asarray([env.get_context_state_obs(c) for c in self.contexts])

    @torch.no_grad()
    def values(self, observations):
        policy = self.model.policy
        obs, _ = policy.obs_to_tensor(observations)
        return policy.predict_values(obs).flatten().cpu().numpy()

    def refresh(self):
        targets = np.array([self.env.generate_one_target_context(self.env.target_type)
                            for _ in range(self.env.num_of_target_samples)])
        # Identical target contexts contribute identical terms to the maximum.
        targets = np.unique(targets, axis=0)
        target_obs = np.array([self.env.get_context_state_obs(c) for c in targets])
        weights = task_weights(self.contexts, self.values(self.start_obs),
                               targets, self.values(target_obs), self.env.beta)
        self.env.cumulative_weights = list(itertools.accumulate(weights))

    def _on_training_start(self):
        self.refresh()

    def _on_step(self):
        # Upstream refreshes before the PPO update at the end of this rollout.
        if self.n_calls % self.period == 0:
            self.refresh()
        return True
