"""
在標準 MNIST CL benchmark 上跑既有 trainers（外部效度驗證）。

完全重用 run.py 的訓練迴圈邏輯、_build_trainer / _eval_loss_acc / summarize、
所有 trainers 與 MLP——只把資料來源換成 MNISTStream。輸出 results_mnist_<mode>.json，
格式與 results_<mode>.json 一致，可直接套 summarize 比較。

用法：
  PY=/Users/jackyyeh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3

  # Permuted-MNIST（多頭 Task-IL）：驗證 DER++ 函數錨定
  $PY run_mnist.py permuted --methods Naive ReplayEWC DarkReplayEWC --seeds 0 1 2 \
      --n-tasks 20 --steps-per-task 6000 --dark-alpha 0.5

  # Split-MNIST（Class-IL，無 task id）：驗證 NCM 修 recency bias
  $PY run_mnist.py split --methods Naive ReplayEWC DarkReplayEWC NCMReplayEWC --seeds 0 1 2 \
      --n-tasks 5 --steps-per-task 8000
"""
import argparse
import json
import os

import numpy as np

from mnist_benchmark import MNISTStream
from model import MLP
from run import _build_trainer, _eval_loss_acc, summarize

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one_mnist(method_name, seed, mode, n_tasks, steps_per_task, test_per_task,
                  lr, h1, h2, batch_size=10, trainer_kwargs=None):
    stream = MNISTStream(n_tasks=n_tasks, steps_per_task=steps_per_task,
                         test_per_task=test_per_task, seed=seed, mode=mode)

    multi_head = (mode == "permuted")
    model = MLP(in_dim=784, h1=h1, h2=h2, out_dim=10, seed=seed,
                multi_head=multi_head, n_tasks=n_tasks, input_adapter=False)
    trainer, used_trainer_kwargs = _build_trainer(model, method_name, lr, seed, trainer_kwargs)

    acc_matrix = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    diagnostics_over_time = []
    plasticity_first_batch_acc = []

    test_sets = [stream.get_test_set(t) for t in range(n_tasks)]

    for t in range(n_tasks):
        X, Y = stream.get_train_batch_stream(t)
        n = X.shape[0]
        first_batch_accs = []
        for i in range(0, n, batch_size):
            xb, yb = X[i:i + batch_size], Y[i:i + batch_size]
            if len(xb) == 0:
                continue
            loss, acc = trainer.train_step(xb, yb, t)
            if i < batch_size * 3:
                first_batch_accs.append(acc)
        plasticity_first_batch_acc.append(float(np.mean(first_batch_accs)))

        trainer.on_task_end(X, Y, t)

        for j in range(t + 1):
            Xt, Yt = test_sets[j]
            _, acc_j = _eval_loss_acc(trainer, model, Xt, Yt, j)
            acc_matrix[t, j] = acc_j

        diag = model.diagnostics(X[:200], t)
        diagnostics_over_time.append(dict(task=t, **diag))

    return dict(
        method=method_name,
        seed=seed,
        acc_matrix=acc_matrix.tolist(),
        diagnostics=diagnostics_over_time,
        plasticity_first_batch_acc=plasticity_first_batch_acc,
        n_tasks=n_tasks,
        mode=f"mnist_{mode}",
        trainer_kwargs=used_trainer_kwargs,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["permuted", "split"])
    p.add_argument("--methods", nargs="+", default=["Naive", "ReplayEWC", "DarkReplayEWC"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--steps-per-task", type=int, default=6000)
    p.add_argument("--test-per-task", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--h1", type=int, default=256)
    p.add_argument("--h2", type=int, default=256)
    p.add_argument("--dark-alpha", type=float, default=0.5)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.mode == "split" and args.n_tasks > 5:
        raise SystemExit("split-MNIST 最多 5 個 task")

    results = []
    for method in args.methods:
        tkw = {}
        if method in ("DarkReplayEWC", "AdaptiveDarkReplayEWC", "PressureDarkReplayEWC"):
            tkw["dark_alpha"] = args.dark_alpha
        seed_summaries = []
        for seed in args.seeds:
            res = run_one_mnist(method, seed, args.mode, args.n_tasks,
                                args.steps_per_task, args.test_per_task, args.lr,
                                args.h1, args.h2, trainer_kwargs=dict(tkw))
            s = summarize(res)
            res["summary"] = s
            results.append(res)
            seed_summaries.append(s)
            print(f"[{method} seed={seed} mnist_{args.mode}] "
                  f"final_avg_acc={s['final_avg_acc']:.3f} BWT={s['bwt']:.3f} "
                  f"forget={s['mean_forgetting']:.3f} retention={s['retention_ratio']:.3f}")
        fa = np.array([s["final_avg_acc"] for s in seed_summaries])
        bw = np.array([s["bwt"] for s in seed_summaries])
        fg = np.array([s["mean_forgetting"] for s in seed_summaries])
        print(f"  >>> {method}: final_avg_acc {fa.mean():.3f} ± {fa.std():.3f} | "
              f"BWT {bw.mean():.3f} | forget {fg.mean():.3f}\n")

    out = args.output or os.path.join(OUT_DIR, f"results_mnist_{args.mode}.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print("saved", out)


if __name__ == "__main__":
    main()
