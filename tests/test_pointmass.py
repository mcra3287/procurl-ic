"""Contract/regression checks for the Gymnasium compatibility changes."""
import unittest
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from envs.pointmass.pm_sparse.binary_pointmass import BinaryContextualPointMass


class PointMassContractTests(unittest.TestCase):
    def setUp(self):
        self.env = BinaryContextualPointMass(context=np.array([0.0, 0.5, 0.0]))

    def tearDown(self):
        self.env.close()

    def test_gymnasium_contract(self):
        # Upstream keeps 4D bounds for physics and exposes 7D observations via
        # its curriculum wrappers. Do not replace the bounds used by dynamics.
        context_wrapper = gym.Wrapper(self.env)
        context_wrapper.observation_space = self.env.observation_space_context
        check_env(context_wrapper, warn=True, skip_render_check=True)

    def test_seed_repeats_noisy_transition(self):
        self.env.reset(seed=12)
        first = self.env.step(np.array([1.0, -2.0]))
        self.env.reset(seed=12)
        second = self.env.step(np.array([1.0, -2.0]))
        np.testing.assert_array_equal(first[0], second[0])

    def test_gate_collision_remains_unsuccessful_terminal(self):
        self.env.reset(seed=12)
        self.env.state = np.array([3.0, 0.0, 0.001, -1.0])
        _, reward, terminated, truncated, info = self.env.step(np.zeros(2))
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])

    def test_goal_remains_successful_terminal(self):
        self.env.reset(seed=12)
        self.env.state = np.array([0.0, 0.0, -2.9, 0.0])
        _, reward, terminated, truncated, info = self.env.step(np.zeros(2))
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])

    def test_horizon_keeps_upstream_terminal_semantics(self):
        self.env.reset(seed=12)
        self.env.num_step = 99
        _, reward, terminated, truncated, _ = self.env.step(np.zeros(2))
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
