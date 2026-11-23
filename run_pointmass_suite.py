"""Run full-budget ProCuRL experiments; aggregate independent training seeds.

The default protocol is ProCuRL-Target, two PointMass settings, 25 seeds each.
This defaults to our PointMass teacher, with the authors' teacher as a control. It does
not implement a new curriculum or assert agreement with the published curves.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
TARGETS = ("single-task", "double-mode-gaussian")
METHODS = ("proxcorl-ic", "proxcorl", "procurl-val", "iid", "target")


def jobs_for(args):
    root = Path(args.output).resolve()
    return [
        {"target_type": target, "curriculum": method, "seed": seed,
         "output": str(root / target / method / f"seed-{seed}")}
        for target in args.targets for method in args.curricula for seed in args.seeds
    ]


def completed(job):
    """Never treat a shortened run or a partial folder as a completed run."""
    path = Path(job["output"])
    try:
        meta = json.loads((path / "config.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        expected = json.loads((ROOT / "configs/pointmass-paper.json").read_text())
        if job["target_type"] != "single-task":
            expected["eval_episodes_per_task"] = 10
        with np.load(path / "evaluations.npz") as curve:
            steps, returns = curve["timesteps"], curve["results"]
            correct_grid = np.array_equal(steps, np.arange(25000, 2000001, 25000))
            correct_shape = returns.shape == (80, expected["eval_episodes_per_task"] * meta["test_tasks"])
        return bool(
            meta["profile"] == "paper" and meta["config"] == expected
            and all(meta[k] == job[k] for k in ("target_type", "curriculum", "seed"))
            and meta["training_tasks"] == 20000 and meta["target_samples"] == 400
            and 2000000 <= metrics["actual_training_steps"] < 2005120
            and (path / "final_model.zip").is_file()
            and correct_grid and correct_shape
        )
    except (OSError, ValueError, KeyError):
        return False


def command(job, args):
    return [sys.executable, "-u", str(ROOT / "run_pointmass.py"), "train",
            "--profile", "paper", "--target-type", job["target_type"],
            "--curriculum", job["curriculum"], "--seed", str(job["seed"]),
            "--output", job["output"], "--device", args.device,
            "--threads", str(args.threads)]


def execute(job, args):
    if completed(job):
        return {**job, "status": "already_completed"}
    path = Path(job["output"])
    if path.exists():
        return {**job, "status": "incomplete_existing_directory",
                "reason": "Preserved; move it aside or choose a new output root before retrying."}
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.parent / (path.name + ".log")
    with log.open("w") as stream:
        result = subprocess.run(command(job, args), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return {**job, "status": "completed" if result.returncode == 0 and completed(job) else "failed",
            "returncode": result.returncode, "log": str(log)}


def aggregate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.output).resolve()
    summary = {"required_seeds_per_setting": 25, "settings": [],
               "meaning": "Execution status, not a claim of matching the paper or a new contribution."}
    for target in args.targets:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for method in args.curricula:
            candidates = [j for j in jobs_for(args) if j["target_type"] == target and j["curriculum"] == method]
            valid = [j for j in candidates if completed(j)]
            item = {"target_type": target, "curriculum": method,
                    "requested_seeds": args.seeds, "completed_seeds": [j["seed"] for j in valid],
                    "n_completed": len(valid), "full_25_seed_execution": len(valid) >= 25}
            summary["settings"].append(item)
            if not valid:
                continue
            arrays = []
            fingerprints = set()
            final_returns = []
            for job in valid:
                folder = Path(job["output"])
                meta = json.loads((folder / "config.json").read_text())
                # Refuse to silently combine distinct software or datasets.
                fingerprints.add(json.dumps({k: meta[k] for k in ("working_commit", "working_tree_dirty", "versions", "dataset_sha256", "config")}, sort_keys=True))
                with np.load(folder / "evaluations.npz") as data:
                    steps = data["timesteps"]
                    arrays.append(data["results"].mean(axis=1))
                final_returns.append(json.loads((folder / "metrics.json").read_text())["mean_return"])
            if len(fingerprints) != 1:
                raise ValueError(f"Incompatible code/configuration/datasets in {target}/{method}; aggregate separately.")
            raw = np.stack(arrays)
            # The paper smooths two consecutive evaluation snapshots.
            smooth = (raw[:, :-1] + raw[:, 1:]) / 2
            for label, matrix, x in (("raw", raw, steps), ("smooth2", smooth, steps[1:])):
                mean = matrix.mean(axis=0)
                sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else np.full_like(mean, np.nan)
                destination = root / f"{target}-{method}-{label}.csv"
                with destination.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["training_steps", "mean_return", "sem_across_training_seeds", "n_seeds"])
                    writer.writerows(zip(x, mean, sem, np.repeat(len(valid), len(x))))
                if label == "smooth2":
                    ax.plot(x, mean, label=f"{method} (n={len(valid)})")
                    if len(valid) > 1:
                        ax.fill_between(x, mean - sem, mean + sem, alpha=0.2)
            item["mean_final_checkpoint_return"] = float(np.mean(final_returns))
            item["sem_final_checkpoint_return"] = float(np.std(final_returns, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else None
            plotted = True
        if plotted:
            ax.set(xlabel="Interações de treino", ylabel="Retorno médio no conjunto-alvo",
                   title=f"PointMass {target} — média ± 1 erro padrão", ylim=(-0.02, 1.02))
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(root / f"{target}.png", dpi=180)
            fig.savefig(root / f"{target}.pdf")
        plt.close(fig)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "plan", "aggregate"])
    parser.add_argument("--output", default="results/pointmass-full")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(25)))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--curricula", nargs="+", choices=METHODS, default=["proxcorl-ic"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("workers/threads must be positive, and seeds must be unique")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args)
    if args.action != "aggregate":
        plan = {"paper_url": "https://arxiv.org/html/2405.02481v1", "training_steps_per_run": 2000000,
                "n_runs": len(jobs), "requested_training_steps": 2000000 * len(jobs),
                "seed_note": "0..24 is our chosen reproducible seed list; the authors' exact seed list was not published in the inspected repository.",
                "jobs": [{**job, "command": command(job, args)} for job in jobs]}
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"{len(jobs)} full-budget runs; {2000000 * len(jobs):,} requested training interactions.", flush=True)
        if args.action == "plan":
            return
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, job, args) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
                (root / "execution.json").write_text(json.dumps(results, indent=2) + "\n")
        aggregate(args)
        if any(r["status"] not in ("completed", "already_completed") for r in results):
            raise SystemExit(1)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
