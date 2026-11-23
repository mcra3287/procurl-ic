"""Check the IC probabilities against the authors' actual sampling method."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper
from envs.pointmass.PointMassWrapper import PointmassWrapper
from pointmass_ic import task_weights


class TeacherParity(unittest.TestCase):
    def test_same_distribution_as_authors(self):
        rng = np.random.default_rng(87)
        train = rng.uniform(-2, 2, (41, 3))
        raw = rng.uniform(-0.5, 1.5, len(train)).astype(np.float32)
        for size in (1, 77):
            with self.subTest(n_targets=size):
                targets = rng.normal(size=(size, 3))
                # Include repeated targets with identical critic outputs.
                targets[-1] = targets[0]
                q_raw = targets[:, 0] + 0.5
                q = np.clip(q_raw, 0, 1)
                env = SimpleNamespace(contexts_arr=train, posl=np.clip(raw, 0, 1),
                                      post=np.ones(len(train)), num_of_tasks=len(train), beta=130)
                env.c2ctest = np.array([p * (1-p) * PointmassWrapper.sim_kernel_with_context(env, c)
                                        for p, c in zip(q, targets)])
                with patch('random.choices', return_value=[0]) as sample:
                    AbstractCurriculumGymWrapper.pick_curr_id_proxcorl(env)
                expected = sample.call_args.kwargs['weights']
                observed = task_weights(train, raw, targets, q_raw, env.beta)
                np.testing.assert_allclose(observed / observed.sum(), expected / expected.sum(),
                                           rtol=1e-6, atol=1e-12)
