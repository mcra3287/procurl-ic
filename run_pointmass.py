"""Reproduction entry point using the upstream environment, teacher and policy.

No proposed research modification is implemented here. The complete training
pool and all 400 target samples are retained even in the short smoke profile.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.pointmass.PointMassWrapper import PointmassWrapper, PointmassWrapperEval
from envs.pointmass.PointMassTeacher import PointMassTeacher
from utils.custom_mlp.custom_actor_critic import CustomActorCriticPolicy

ROOT = Path(__file__).resolve().parent
UPSTREAM = "9c7df278fb2520188d4dafb10f5888acec67bd4d"


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(args):
    config = json.loads((ROOT / "configs/pointmass-paper.json").read_text())
    if args.target_type != "single-task":
        config["eval_episodes_per_task"] = 10
    if args.profile == "smoke":
        config.update(total_timesteps=10240, eval_freq=5120,
                      eval_episodes_per_task=10)
    for name in ("total_timesteps", "eval_freq", "eval_episodes_per_task"):
        value = getattr(args, name, None)
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            config[name] = value
    return config


def make_teacher(env, config, curriculum):
    return PointMassTeacher(
        env=env, type_env="binary", target_type=env.target_type, cur=curriculum,
        eps=0.0, n_steps=config["n_steps"], spdl_pthresh=0.5,
        Na=200000, alpha=0.01, epsilon=0.5, currot_perf_lb=0.4,
        currot_metric_eps=1.5, metrics=False, verbose=0,
    )


def evaluate(model, env, n_episodes, seed):
    # Final evaluation has its own repeatable seed and does not train the model.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=False,
        return_episode_rewards=True,
    )
    rewards = np.asarray(rewards)
    return {
        "mean_return": float(rewards.mean()),
        "std_return": float(rewards.std()),
        "success_rate": float(np.mean(rewards > 0)),
        "n_eval_episodes": n_episodes,
        "mean_episode_length": float(np.mean(lengths)),
        "returns": rewards.tolist(),
        "episode_lengths": list(map(int, lengths)),
        "evaluation_seed": seed,
        "deterministic_actions": False,
    }


def train(args):
    config = load_config(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The upstream CSV paths are relative to the repository root.
    import os
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    started = time.perf_counter()
    train_env = PointmassWrapper(
        cur=args.curriculum, env_type="binary", beta=config["beta"],
        target_type=args.target_type, metrics=False, beta_plr=0.1, rho_plr=0.5,
    )
    eval_env = PointmassWrapperEval(env_type="binary", target_type=args.target_type)
    teacher = make_teacher(train_env, config, args.curriculum)
    wrapped_train = Monitor(train_env, str(output / "train.monitor.csv"))
    wrapped_eval = Monitor(eval_env, str(output / "eval.monitor.csv"))
    ppo_keys = ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
                "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm")
    model = PPO(
        CustomActorCriticPolicy, wrapped_train, seed=args.seed, device=args.device,
        policy_kwargs={"activation_fn": torch.nn.ReLU,
                       "net_arch": [64, {"pi": [64], "vf": [64]}]},
        tensorboard_log=str(output / "tensorboard") if args.tensorboard else None,
        verbose=1, **{key: config[key] for key in ppo_keys},
    )
    csv_paths = [ROOT / "envs/pointmass/task_datasets/pm_train.csv"]
    eval_csv = "pm_test_single_task.csv" if args.target_type == "single-task" else "pm_test_heldout.csv"
    csv_paths.append(ROOT / "envs/pointmass/task_datasets" / eval_csv)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_repository": "https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl",
        "upstream_commit": UPSTREAM, "working_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_output("status", "--porcelain", "--untracked-files=no")),
        "profile": args.profile,
        "purpose": "execution check, not paper reproduction" if args.profile == "smoke" else "reproduction attempt",
        "curriculum": args.curriculum, "target_type": args.target_type,
        "seed": args.seed, "device": str(model.device), "threads": args.threads,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {pkg: version(pkg) for pkg in ["torch", "numpy", "scipy", "gymnasium", "stable-baselines3"]},
        "config": config, "training_tasks": train_env.num_of_tasks,
        "target_samples": train_env.num_of_target_samples, "test_tasks": eval_env.num_of_tasks,
        "dataset_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in csv_paths},
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    episodes = config["eval_episodes_per_task"] * eval_env.num_of_tasks
    callback = EvalCallback(
        wrapped_eval, eval_freq=config["eval_freq"], n_eval_episodes=episodes,
        deterministic=config["deterministic_eval"],
        best_model_save_path=str(output / "best"), log_path=str(output),
    )
    try:
        model.learn(total_timesteps=config["total_timesteps"], callback=[callback, teacher])
        model.save(str(output / "final_model"))
        results = evaluate(model, wrapped_eval, episodes, args.seed + 100000)
        results.update(actual_training_steps=model.num_timesteps,
                       elapsed_seconds=time.perf_counter() - started,
                       profile=args.profile, paper_reproduction_completed=False)
        (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({key: value for key, value in results.items()
                          if key not in ("returns", "episode_lengths")}, indent=2))
        print(f"Artifacts: {output}")
    finally:
        wrapped_train.close()
        wrapped_eval.close()


def test_checkpoint(args):
    import os
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    env = Monitor(PointmassWrapperEval(env_type="binary", target_type=args.target_type))
    model = PPO.load(args.checkpoint, device=args.device)
    try:
        results = evaluate(model, env, args.episodes_per_task * env.unwrapped.num_of_tasks, args.seed)
        print(json.dumps(results, indent=2))
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--target-type", choices=["single-task", "double-mode-gaussian"], default="single-task")
        p.add_argument("--seed", type=int, default=420)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=1)
        if command == "train":
            p.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
            p.add_argument("--curriculum", choices=["proxcorl", "procurl-val", "iid", "target", "spdl", "plr", "currot"], default="proxcorl")
            p.add_argument("--output", required=True)
            p.add_argument("--total-timesteps", type=int)
            p.add_argument("--eval-freq", type=int)
            p.add_argument("--eval-episodes-per-task", type=int)
            p.add_argument("--tensorboard", action="store_true")
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--episodes-per-task", type=int, default=100)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        test_checkpoint(args)
