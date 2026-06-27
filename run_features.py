"""
P8：在 frozen pretrained 特徵上跑既有 trainers（Split-CIFAR-100 Class-IL）。

重用 run.py 的訓練迴圈邏輯、_build_trainer/_eval_loss_acc/summarize、全部 trainers
與 MLP，只把資料來源換成 FeatureSplitStream。單頭、全域類別、推論無 task id。

用法：
  PY=/Users/jackyyeh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
  $PY run_features.py --feat cifar100_resnet18.npz \
      --methods Naive ReplayEWC DarkReplayEWC NCMReplayEWC --seeds 0 1 2 \
      --n-tasks 20 --classes-per-task 5 --steps-per-task 200 --test-per-task 100
"""
import argparse
import json
import os

import numpy as np

from feature_benchmark import FeatureSplitStream
from model import MLP
from run import _build_trainer, _eval_loss_acc, summarize

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one_feature(method_name, seed, feat_npz, n_tasks, classes_per_task,
                    steps_per_task, test_per_task, lr, h1, h2, batch_size=10,
                    epochs=1, trainer_kwargs=None):
    stream = FeatureSplitStream(feat_npz, n_tasks=n_tasks, classes_per_task=classes_per_task,
                                steps_per_task=steps_per_task, test_per_task=test_per_task,
                                seed=seed)
    out_dim = stream.n_classes
    model = MLP(in_dim=stream.in_dim, h1=h1, h2=h2, out_dim=out_dim, seed=seed,
                multi_head=False, n_tasks=n_tasks, input_adapter=False)
    trainer, used_trainer_kwargs = _build_trainer(model, method_name, lr, seed, trainer_kwargs)

    acc_matrix = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    diagnostics_over_time = []
    plasticity_first_batch_acc = []
    test_sets = [stream.get_test_set(t) for t in range(n_tasks)]

    rng = np.random.RandomState(1000 + seed)
    for t in range(n_tasks):
        X, Y = stream.get_train_batch_stream(t)
        n = X.shape[0]
        first_batch_accs = []
        for ep in range(epochs):
            order = rng.permutation(n) if ep > 0 else np.arange(n)
            Xe, Ye = X[order], Y[order]
            for i in range(0, n, batch_size):
                xb, yb = Xe[i:i + batch_size], Ye[i:i + batch_size]
                if len(xb) == 0:
                    continue
                loss, acc = trainer.train_step(xb, yb, t)
                if ep == 0 and i < batch_size * 3:
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
        method=method_name, seed=seed, acc_matrix=acc_matrix.tolist(),
        diagnostics=diagnostics_over_time,
        plasticity_first_batch_acc=plasticity_first_batch_acc,
        n_tasks=n_tasks, mode="feature_split", trainer_kwargs=used_trainer_kwargs,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", default="cifar100_resnet18.npz")
    p.add_argument("--methods", nargs="+",
                   default=["Naive", "ReplayEWC", "DarkReplayEWC", "NCMReplayEWC"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--classes-per-task", type=int, default=5)
    p.add_argument("--steps-per-task", type=int, default=200)
    p.add_argument("--test-per-task", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--h1", type=int, default=256)
    p.add_argument("--h2", type=int, default=256)
    p.add_argument("--dark-alpha", type=float, default=0.5)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    results = []
    for method in args.methods:
        tkw = {}
        if method in ("DarkReplayEWC", "AdaptiveDarkReplayEWC", "PressureDarkReplayEWC"):
            tkw["dark_alpha"] = args.dark_alpha
        seed_summaries = []
        for seed in args.seeds:
            res = run_one_feature(method, seed, args.feat, args.n_tasks,
                                  args.classes_per_task, args.steps_per_task,
                                  args.test_per_task, args.lr, args.h1, args.h2,
                                  epochs=args.epochs, trainer_kwargs=dict(tkw))
            s = summarize(res)
            res["summary"] = s
            results.append(res)
            seed_summaries.append(s)
            print(f"[{method} seed={seed} feature_split] "
                  f"final_avg_acc={s['final_avg_acc']:.3f} BWT={s['bwt']:.3f} "
                  f"forget={s['mean_forgetting']:.3f} retention={s['retention_ratio']:.3f}",
                  flush=True)
        fa = np.array([s["final_avg_acc"] for s in seed_summaries])
        fg = np.array([s["mean_forgetting"] for s in seed_summaries])
        print(f"  >>> {method}: final_avg_acc {fa.mean():.3f} ± {fa.std():.3f} | "
              f"forget {fg.mean():.3f}\n", flush=True)

    out = args.output or os.path.join(OUT_DIR, "results_feature_split_cifar100.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print("saved", out)


if __name__ == "__main__":
    main()
