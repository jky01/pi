import unittest
import os
import numpy as np

from run import run_one, summarize
from trainers import TRAINER_REGISTRY

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

class TestSmoke(unittest.TestCase):
    def setUp(self):
        # Create a tiny digits sequence for quick testing
        self.digits_file = "test_digits.txt"
        self.digits_path = os.path.join(OUT_DIR, self.digits_file)
        with open(self.digits_path, "w") as f:
            f.write("31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679")

    def tearDown(self):
        if os.path.exists(self.digits_path):
            os.remove(self.digits_path)

    def test_smoke_pipelines(self):
        K = 4
        n_tasks = 2
        steps_per_task = 10
        test_per_task = 5
        lr = 0.1
        
        methods = list(TRAINER_REGISTRY.keys())
        modes = ["label_permuted", "input_permuted", "conflicting"]
        
        for mode in modes:
            for method in methods:
                print(f"Running smoke test for method={method}, mode={mode}...")
                stream_kwargs = dict(
                    digits_file=self.digits_file,
                    K=K, n_tasks=n_tasks, steps_per_task=steps_per_task, test_per_task=test_per_task,
                    mode=mode
                )
                model_kwargs = dict(in_dim=10 * K, h1=8, h2=8, out_dim=10)
                
                res = run_one(method, seed=42, stream_kwargs=stream_kwargs, model_kwargs=model_kwargs, lr=lr, batch_size=2)
                summary = summarize(res)
                
                # Check that result fields are populated
                self.assertEqual(res["method"], method)
                self.assertEqual(res["mode"], mode)
                self.assertEqual(res["n_tasks"], n_tasks)
                self.assertEqual(len(res["acc_matrix"]), n_tasks)
                self.assertEqual(len(res["diagnostics"]), n_tasks)
                self.assertEqual(len(res["plasticity_first_batch_acc"]), n_tasks)
                self.assertTrue(0.0 <= summary["final_avg_acc"] <= 1.0)
                self.assertTrue(summary["learned_avg_acc"] >= 0.0)
                self.assertTrue(summary["retention_ratio"] >= 0.0)
                self.assertFalse(np.isnan(summary["mean_forgetting"]))
                
                # Check for no NaNs in final row of accuracy matrix
                acc_last_row = res["acc_matrix"][-1]
                for acc in acc_last_row:
                    self.assertFalse(np.isnan(acc), f"NaN found in accuracy matrix for {method} {mode}")
                    self.assertTrue(0.0 <= acc <= 1.0, f"Invalid accuracy {acc} for {method} {mode}")

                # Check diagnostics
                for d in res["diagnostics"]:
                    self.assertTrue(d["eff_rank_h1"] > 0)
                    self.assertTrue(0.0 <= d["dead_frac_h1"] <= 1.0)
                    self.assertTrue(d["weight_norm"] > 0)
                
                print(f"method={method}, mode={mode} smoke test PASSED.")

if __name__ == "__main__":
    unittest.main()
