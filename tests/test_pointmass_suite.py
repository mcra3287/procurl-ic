"""Prevent partial experiments and episode variance from becoming paper claims."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_pointmass_suite import ROOT, aggregate, completed, jobs_for


class FullProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.args = argparse.Namespace(output=self.temp.name, seeds=[0, 1],
                                       targets=["single-task"], curricula=["proxcorl"])

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, seed, value):
        job = next(j for j in jobs_for(self.args) if j["seed"] == seed)
        folder = Path(job["output"])
        folder.mkdir(parents=True)
        config = json.loads((ROOT / "configs/pointmass-paper.json").read_text())
        metadata = {**{k: job[k] for k in ("seed", "curriculum", "target_type")},
                    "profile": "paper", "config": config, "training_tasks": 20000,
                    "target_samples": 400, "test_tasks": 1,
                    "working_commit": "same-commit", "working_tree_dirty": False,
                    "versions": {}, "dataset_sha256": {}}
        (folder / "config.json").write_text(json.dumps(metadata))
        (folder / "metrics.json").write_text(json.dumps({"actual_training_steps": 2001920, "mean_return": value}))
        (folder / "final_model.zip").write_bytes(b"placeholder for completion detector")
        np.savez(folder / "evaluations.npz", timesteps=np.arange(25000, 2000001, 25000),
                 results=np.full((80, 100), value))
        return job

    def test_default_scientific_scope_is_fifty_runs(self):
        args = argparse.Namespace(output=self.temp.name, seeds=list(range(25)),
                                  targets=["single-task", "double-mode-gaussian"], curricula=["proxcorl"])
        self.assertEqual(len(jobs_for(args)), 50)

    def test_shortened_run_does_not_count_as_full(self):
        job = self.make_run(0, 0.0)
        self.assertTrue(completed(job))
        path = Path(job["output"]) / "metrics.json"
        path.write_text(json.dumps({"actual_training_steps": 10240, "mean_return": 0.0}))
        self.assertFalse(completed(job))

    def test_incomplete_evaluation_grid_is_rejected(self):
        job = self.make_run(0, 0.0)
        np.savez(Path(job["output"]) / "evaluations.npz", timesteps=[25000], results=np.zeros((1, 100)))
        self.assertFalse(completed(job))

    def test_sem_uses_training_seeds_and_partial_status_is_explicit(self):
        self.make_run(0, 0.0)
        self.make_run(1, 1.0)
        summary = aggregate(self.args)
        item = summary["settings"][0]
        self.assertFalse(item["full_25_seed_execution"])
        self.assertAlmostEqual(item["mean_final_checkpoint_return"], 0.5)
        self.assertAlmostEqual(item["sem_final_checkpoint_return"], 0.5)
        table = np.genfromtxt(Path(self.temp.name) / "single-task-proxcorl-smooth2.csv", delimiter=",", names=True)
        np.testing.assert_allclose(table["mean_return"], 0.5)
        np.testing.assert_allclose(table["sem_across_training_seeds"], 0.5)
        self.assertTrue((Path(self.temp.name) / "single-task.png").is_file())

    def test_different_implementations_are_not_silently_pooled(self):
        self.make_run(0, 0.0)
        second = self.make_run(1, 1.0)
        path = Path(second["output"]) / "config.json"
        meta = json.loads(path.read_text())
        meta["working_commit"] = "different-commit"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            aggregate(self.args)


if __name__ == "__main__":
    unittest.main()
