"""
P10：量測「正向遷移 / 累積」——真正的持續學習，而不只是抗遺忘。

問題：持續學習過去 k 個 task 之後，模型學「第 k+1 個新 task」是否變得**更快**？
（這是 CL 與「只是別忘記」的根本差別。§10.3 觀察到「表徵有遷移、學習速度沒有」，
P10 把它在 frozen-feature regime 下量化。）

設計（在 Split-CIFAR-100 frozen ResNet18 特徵流上）：
- 對每個 task k，量「task-k 受限 5-way 準確率」隨訓練步數的曲線（只在 task k 自己的
  5 個類別間做 argmax，隔離「新任務本身學多快」，避開 class-IL 干擾）。
- **持續模型**：已歷經 task 0..k-1 的 ReplayEWC，其隱藏層累積了過去知識。
- **fresh 對照**：同架構、隨機初始化的新 head，只在 task k 上從零訓練（plain SGD）。
- 兩者吃**同一個 frozen backbone 特徵**，唯一差別是隱藏層是否被持續訓練過。
  → 持續 − fresh 的早期學習優勢 = 表徵的正向遷移。看它**是否隨 k 增長**（真累積）。
"""
import argparse
import json
import os

import numpy as np

from feature_benchmark import FeatureSplitStream
from model import MLP
from run import _build_trainer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = [0, 2, 5, 10, 20, 40, 80, 160, 320]  # 量測的累積梯度步數


def acc_5way(model, Xt, Yt, task_classes):
    """task-k 受限 5-way 準確率：只在該 task 的類別間 argmax。"""
    logits = model.forward(Xt)["logits"]
    cols = np.array(task_classes)
    pred = cols[np.argmax(logits[:, cols], axis=1)]
    return float(np.mean(pred == Yt))


def train_curve(trainer, model, X, Y, Xt, Yt, task_classes, task_idx, epochs, batch_size, rng):
    """訓練 task_idx，並在 CHECKPOINTS 指定的步數記錄 5-way 準確率。回傳 {step: acc}。"""
    n = X.shape[0]
    curve = {}
    s = 0
    if 0 in CHECKPOINTS:
        curve[0] = acc_5way(model, Xt, Yt, task_classes)
    todo = [c for c in CHECKPOINTS if c > 0]
    for ep in range(epochs):
        order = rng.permutation(n) if ep > 0 else np.arange(n)
        Xe, Ye = X[order], Y[order]
        for i in range(0, n, batch_size):
            xb, yb = Xe[i:i + batch_size], Ye[i:i + batch_size]
            if len(xb) == 0:
                continue
            trainer.train_step(xb, yb, task_idx)
            s += 1
            if todo and s == todo[0]:
                curve[s] = acc_5way(model, Xt, Yt, task_classes)
                todo.pop(0)
        if not todo:
            break
    # 若總步數 < 最大 checkpoint，補記最終狀態
    if todo:
        curve[s] = acc_5way(model, Xt, Yt, task_classes)
    return curve


def run_one(seed, feat_npz, n_tasks, classes_per_task, steps_per_task, test_per_task,
            lr, h1, h2, epochs, batch_size=10, continual_method="ReplayEWC"):
    stream = FeatureSplitStream(feat_npz, n_tasks=n_tasks, classes_per_task=classes_per_task,
                               steps_per_task=steps_per_task, test_per_task=test_per_task, seed=seed)
    out_dim = stream.n_classes

    cont_model = MLP(in_dim=stream.in_dim, h1=h1, h2=h2, out_dim=out_dim, seed=seed,
                     multi_head=False, n_tasks=n_tasks)
    cont_trainer, _ = _build_trainer(cont_model, continual_method, lr, seed, None)

    per_task = []
    for k in range(n_tasks):
        X, Y = stream.get_train_batch_stream(k)
        Xt, Yt = stream.get_test_set(k)
        tcls = stream.task_classes[k]

        # 持續模型：邊訓練 task k 邊量曲線（這同時也推進了它的持續學習）
        rng_c = np.random.RandomState(1000 + seed)
        cont_curve = train_curve(cont_trainer, cont_model, X, Y, Xt, Yt, tcls, k,
                                 epochs, batch_size, rng_c)
        cont_trainer.on_task_end(X, Y, k)

        # fresh 對照：全新隨機 head，只在 task k 上從零學（同特徵、同架構、同資料順序）
        fresh_model = MLP(in_dim=stream.in_dim, h1=h1, h2=h2, out_dim=out_dim,
                          seed=9000 + seed * 100 + k, multi_head=False, n_tasks=n_tasks)
        fresh_trainer, _ = _build_trainer(fresh_model, "Naive", lr, seed, None)
        rng_f = np.random.RandomState(1000 + seed)
        fresh_curve = train_curve(fresh_trainer, fresh_model, X, Y, Xt, Yt, tcls, k,
                                  epochs, batch_size, rng_f)

        per_task.append(dict(task=k, continual=cont_curve, fresh=fresh_curve))
    return per_task


def summarize(all_seed_per_task, early_step=20):
    """彙整：每個 task 的 continual / fresh 早期 5-way acc，與其差（正向遷移）。"""
    n_tasks = len(all_seed_per_task[0])

    def at(curve, step):
        # 取 <= step 的最大 checkpoint（曲線 key 是字串化的 step）
        ks = sorted(int(x) for x in curve.keys())
        best = ks[0]
        for x in ks:
            if x <= step:
                best = x
        return curve[str(best)] if str(best) in curve else curve[best]

    rows = []
    for k in range(n_tasks):
        c = np.mean([at(s[k]["continual"], early_step) for s in all_seed_per_task])
        f = np.mean([at(s[k]["fresh"], early_step) for s in all_seed_per_task])
        rows.append((k, c, f, c - f))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat", default="cifar100_resnet18.npz")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--classes-per-task", type=int, default=5)
    p.add_argument("--steps-per-task", type=int, default=1000)
    p.add_argument("--test-per-task", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--h1", type=int, default=256)
    p.add_argument("--h2", type=int, default=256)
    p.add_argument("--early-step", type=int, default=20)
    p.add_argument("--continual-method", default="ReplayEWC")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    all_seed = []
    for seed in args.seeds:
        per_task = run_one(seed, args.feat, args.n_tasks, args.classes_per_task,
                           args.steps_per_task, args.test_per_task, args.lr,
                           args.h1, args.h2, args.epochs,
                           continual_method=args.continual_method)
        all_seed.append(per_task)
        print(f"seed {seed} done", flush=True)

    # 多個早期步數預算，看是否有「早期領先、很快被追平」的正向遷移
    print(f"\n=== Forward transfer (continual={args.continual_method} vs fresh-from-scratch) ===")
    print("task-k 5-way acc | continual vs fresh, at several step budgets")
    budgets = [2, 5, 10, 20]
    rows_main = summarize(all_seed, early_step=args.early_step)
    for step in budgets:
        rows = summarize(all_seed, early_step=step)
        c = np.mean([r[1] for r in rows]); f = np.mean([r[2] for r in rows])
        early = np.mean([d for k, _, _, d in rows if k < 5])
        late = np.mean([d for k, _, _, d in rows if k >= 15])
        overall = np.mean([d for _, _, _, d in rows])
        grow = late - early
        tag = "GROWS w/ accumulation" if grow > 0.02 else ("FLAT" if abs(grow) <= 0.02 else "SHRINKS")
        print(f"  @{step:>3} steps: continual {c:.3f} | fresh {f:.3f} | "
              f"Δ overall {overall:+.3f} | Δ early {early:+.3f} -> late {late:+.3f} "
              f"(cumulative {grow:+.3f} {tag})")

    print(f"\nPer-task Δ (continual - fresh) @ {args.early_step} steps:")
    print("task |  continual |  fresh  |  Δ (transfer)")
    for k, c, f, d in rows_main:
        print(f" {k:>3} |   {c:.3f}   |  {f:.3f} |   {d:+.3f}")
    rows = rows_main

    out = args.output or os.path.join(OUT_DIR, "results_forward_transfer_cifar100.json")
    with open(out, "w") as fp:
        json.dump(dict(early_step=args.early_step, rows=rows,
                       per_seed=[[{"task": t["task"],
                                   "continual": {str(k): v for k, v in t["continual"].items()},
                                   "fresh": {str(k): v for k, v in t["fresh"].items()}}
                                  for t in ps] for ps in all_seed]), fp)
    print("saved", out)


if __name__ == "__main__":
    main()
