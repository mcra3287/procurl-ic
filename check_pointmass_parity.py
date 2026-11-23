"""Freeze one checkpoint and compare the task distributions of both teachers."""
import argparse
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.pointmass.PointMassWrapper import PointmassWrapper
from pointmass_ic import ICPointMass, ICTeacher
from run_pointmass import make_teacher


def check(model, target, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = ICPointMass(cur="proxcorl", env_type="binary", beta=130, target_type=target,
                      metrics=False, beta_plr=0.1, rho_plr=0.5)
    try:
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        ref = make_teacher(env, {"n_steps": 5120}, "proxcorl")
        ref.model = model
        env.posl = ref.value_critic()
        env.c2ctest = ref.get_expected_target_performance()
        with patch("random.choices", return_value=[0]) as sample:
            PointmassWrapper.pick_curr_id_proxcorl(env)
        ref_prob = np.array(sample.call_args.kwargs["weights"])
        ref_prob /= ref_prob.sum()

        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        ours = ICTeacher(env, 5120)
        ours.model = model
        ours.refresh()
        with patch("random.choices", return_value=[0]) as sample:
            env.pick_curr_id_proxcorl()
        cumulative = sample.call_args.kwargs["cum_weights"]
        prob = np.diff(cumulative, prepend=0) / cumulative[-1]
        delta = np.abs(prob - ref_prob)
        return {"target": target, "seed": seed, "tasks": env.num_of_tasks,
                "target_samples": env.num_of_target_samples,
                "max_abs_probability_error": float(delta.max()),
                "total_variation_distance": float(delta.sum() / 2)}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = PPO.load(args.checkpoint, device="cpu")
    rows = [check(model, target, args.seed) for target in ("single-task", "double-mode-gaussian")]
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if any(r["total_variation_distance"] > 1e-5 for r in rows):
        raise SystemExit("Distributions differ by more than 1e-5 in total variation.")
