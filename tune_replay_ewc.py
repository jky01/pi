"""
Small hyperparameter sweep for ReplayEWC-family methods.

This script runs compact validation experiments and stores concise summaries
instead of full accuracy matrices. Use the best configuration here as a candidate
for a full 80-task run through run.py.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

from run import OUT_DIR, run_one, summarize


def _stats(values):
    values = np.array(values, dtype=np.float64)
    return dict(mean=float(values.mean()), std=float(values.std()))


def main():
    parser = argparse.ArgumentParser(description="Tune replay/EWC methods on Permuted-Pi-Digits.")
    parser.add_argument("mode", nargs="?", default="label_permuted",
                        choices=["label_permuted", "input_permuted"])
    parser.add_argument("--method", default="ReplayEWC",
                        choices=["ReplayEWC", "SurpriseReplayEWC", "MarginSurpriseReplayEWC", "DarkReplayEWC"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--n-tasks", type=int, default=40)
    parser.add_argument("--steps-per-task", type=int, default=2000)
    parser.add_argument("--test-per-task", type=int, default=200)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lams", nargs="+", type=float, default=[2.5, 5.0, 10.0])
    parser.add_argument("--capacities", nargs="+", type=int, default=[500])
    parser.add_argument("--replay-batches", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--candidate-mults", nargs="+", type=int, default=[8])
    parser.add_argument("--margin-weights", nargs="+", type=float, default=[0.5])
    parser.add_argument("--dark-alphas", nargs="+", type=float, default=[0.1])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stream_kwargs = dict(
        digits_file="pi_digits_600000.txt",
        K=args.K,
        n_tasks=args.n_tasks,
        steps_per_task=args.steps_per_task,
        test_per_task=args.test_per_task,
        mode=args.mode,
    )
    model_kwargs = dict(in_dim=10 * args.K, h1=64, h2=64, out_dim=10)

    candidate_mults = args.candidate_mults if args.method in ("SurpriseReplayEWC", "MarginSurpriseReplayEWC") else [args.candidate_mults[0]]
    margin_weights = args.margin_weights if args.method == "MarginSurpriseReplayEWC" else [args.margin_weights[0]]
    dark_alphas = args.dark_alphas if args.method == "DarkReplayEWC" else [args.dark_alphas[0]]

    runs = []
    t_start = time.time()
    for lam, capacity, replay_batch, candidate_mult, margin_weight, dark_alpha in itertools.product(
            args.lams, args.capacities, args.replay_batches, candidate_mults,
            margin_weights, dark_alphas):
        per_seed = []
        cfg = dict(lam=lam, capacity=capacity, replay_batch=replay_batch)
        if args.method == "SurpriseReplayEWC":
            cfg["candidate_mult"] = candidate_mult
        if args.method == "MarginSurpriseReplayEWC":
            cfg["candidate_mult"] = candidate_mult
            cfg["margin_weight"] = margin_weight
        if args.method == "DarkReplayEWC":
            cfg["dark_alpha"] = dark_alpha
        combo_start = time.time()
        for seed in args.seeds:
            res = run_one(
                args.method,
                seed=seed,
                stream_kwargs=stream_kwargs,
                model_kwargs=model_kwargs,
                lr=args.lr,
                batch_size=args.batch_size,
                trainer_kwargs=cfg,
            )
            s = summarize(res)
            per_seed.append(dict(seed=seed, **{k: v for k, v in s.items() if k != "diag"}))

        rec = dict(
            method=args.method,
            mode=args.mode,
            n_tasks=args.n_tasks,
            steps_per_task=args.steps_per_task,
            test_per_task=args.test_per_task,
            seeds=args.seeds,
            trainer_kwargs=cfg,
            final_avg_acc=_stats([s["final_avg_acc"] for s in per_seed]),
            bwt=_stats([s["bwt"] for s in per_seed]),
            mean_forgetting=_stats([s["mean_forgetting"] for s in per_seed]),
            retention_ratio=_stats([s["retention_ratio"] for s in per_seed]),
            plasticity_slope=_stats([s["plasticity_slope"] for s in per_seed]),
            per_seed=per_seed,
            seconds=float(time.time() - combo_start),
        )
        runs.append(rec)
        print(
            f"method={args.method} lam={lam:g} capacity={capacity} replay_batch={replay_batch} "
            f"candidate_mult={candidate_mult} margin_weight={margin_weight:g} dark_alpha={dark_alpha:g} "
            f"final={rec['final_avg_acc']['mean']:.3f}±{rec['final_avg_acc']['std']:.3f} "
            f"bwt={rec['bwt']['mean']:.3f} retention={rec['retention_ratio']['mean']:.3f} "
            f"forget={rec['mean_forgetting']['mean']:.3f} ({rec['seconds']:.1f}s)"
        )

    runs.sort(key=lambda r: (r["final_avg_acc"]["mean"], r["retention_ratio"]["mean"]), reverse=True)
    out = dict(
        mode=args.mode,
        objective="Maximize final_avg_acc, then retention_ratio.",
        total_seconds=float(time.time() - t_start),
        runs=runs,
        best=runs[0] if runs else None,
    )

    out_path = args.output or os.path.join(OUT_DIR, f"tune_{args.method}_{args.mode}.json")
    if not os.path.isabs(out_path):
        out_path = os.path.join(OUT_DIR, out_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("best:", out["best"]["trainer_kwargs"] if out["best"] else None)
    print("saved to", out_path)


if __name__ == "__main__":
    main()
