"""
主實驗腳本：在 Permuted-Pi-Digits 串流上跑多種 continual-learning trainer，
建立 accuracy matrix，計算可塑性對角線、BWT（遺忘）、有效秩/死神經元診斷曲線。

對應對話 Stage 0-D 的設計：
- 基準線：Naive（下界）、Replay/EWC（強參考），再加入 replay/regularization/plasticity 的複合方法。
- accuracy matrix A[i][j]：訓練完 task i 後，在 task j 測試集上的準確率。
  對角線 A[i][i] = 可塑性；最後一列 vs 對角線 = 遺忘（BWT）。
- 多 seed、固定任務順序、同硬體同計算預算（所有方法看過的資料量一致）。
"""
import json
import os
import time
import inspect

import numpy as np

from benchmark import PermutedPiDigitsStream
from model import MLP
from trainers import TRAINER_REGISTRY

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METHODS = [
    "Naive",
    "EWC",
    "Replay",
    "ReplayEWC",
    "SurpriseReplayEWC",
    "HippocampalReplayEWC",
    "TaskBalancedReplay",
    "ContinualBP",
    "ReplayContinualBP",
]


def _build_trainer(model, method_name, lr, seed, trainer_kwargs=None):
    trainer_kwargs = dict(trainer_kwargs or {})
    Cls = TRAINER_REGISTRY[method_name]
    sig = inspect.signature(Cls.__init__)
    if "lr" in sig.parameters:
        trainer_kwargs.setdefault("lr", lr)
    if "seed" in sig.parameters:
        trainer_kwargs.setdefault("seed", seed)
    filtered = {k: v for k, v in trainer_kwargs.items() if k in sig.parameters}
    return Cls(model, **filtered), filtered


def _eval_loss_acc(trainer, model, X, Y, task_idx):
    if hasattr(trainer, "loss_acc"):
        return trainer.loss_acc(X, Y, task_idx)
    return model.loss_acc(X, Y, task_idx)


def run_one(method_name, seed, stream_kwargs, model_kwargs, lr, batch_size=10,
            diag_every_n_tasks=1, trainer_kwargs=None):
    stream_kwargs = dict(stream_kwargs)
    digits_file = stream_kwargs.pop("digits_file")
    with open(os.path.join(OUT_DIR, digits_file)) as f:
        digits = f.read().strip()
    
    # Extract mode to configure model_kwargs
    mode = stream_kwargs.get("mode", "label_permuted")
    model_kwargs = dict(model_kwargs)
    model_kwargs["multi_head"] = (mode in ("label_permuted", "conflicting"))
    model_kwargs["n_tasks"] = stream_kwargs.get("n_tasks", 80)
    # Class-IL: single head over a global, disjoint label space (n_classes * n_tasks),
    # no task id at inference. The 10-bucket base classes become 10*n_tasks global classes.
    if mode == "class_il":
        model_kwargs["out_dim"] = 10 * stream_kwargs.get("n_tasks", 80)
    # input_adapter is opt-in (only meaningful for the single-head input_permuted mode):
    # gives each task a learnable input transform so a shared net can serve all permutations.
    model_kwargs["input_adapter"] = model_kwargs.get("input_adapter", False)
    
    stream = PermutedPiDigitsStream(digits, seed=seed, **stream_kwargs)
    n_tasks = stream.n_tasks

    model = MLP(**model_kwargs, seed=seed)
    trainer, used_trainer_kwargs = _build_trainer(model, method_name, lr, seed, trainer_kwargs)

    n_tasks_arr = n_tasks
    acc_matrix = np.full((n_tasks_arr, n_tasks_arr), np.nan, dtype=np.float64)
    diagnostics_over_time = []  # (task_idx, eff_rank_h1, eff_rank_h2, dead_h1, dead_h2, weight_norm)
    plasticity_first_batch_acc = []  # 每個 task 剛開始訓練時的 batch 準確率

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

        # 在所有看過的 task 測試集上評估，填 accuracy matrix 第 t 列
        for j in range(t + 1):
            Xt, Yt = test_sets[j]
            _, acc_j = _eval_loss_acc(trainer, model, Xt, Yt, j)
            acc_matrix[t, j] = acc_j

        diag = model.diagnostics(X[:200], t)
        diagnostics_over_time.append(dict(task=t, **diag))

    trainer_diagnostics = {}
    if hasattr(trainer, "alpha_trace") and len(trainer.alpha_trace) > 0:
        alpha_trace = np.array(trainer.alpha_trace, dtype=np.float64)
        conflict_trace = np.array(getattr(trainer, "conflict_trace", []), dtype=np.float64)
        tail_n = min(200, len(alpha_trace))
        trainer_diagnostics["adaptive_dark_alpha_mean"] = float(np.mean(alpha_trace))
        trainer_diagnostics["adaptive_dark_alpha_tail_mean"] = float(np.mean(alpha_trace[-tail_n:]))
        trainer_diagnostics["adaptive_dark_alpha_min"] = float(np.min(alpha_trace))
        trainer_diagnostics["adaptive_dark_alpha_max"] = float(np.max(alpha_trace))
        if len(conflict_trace) > 0:
            trainer_diagnostics["adaptive_conflict_mean"] = float(np.mean(conflict_trace))
            trainer_diagnostics["adaptive_conflict_tail_mean"] = float(np.mean(conflict_trace[-tail_n:]))
            trainer_diagnostics["adaptive_conflict_min"] = float(np.min(conflict_trace))
            trainer_diagnostics["adaptive_conflict_max"] = float(np.max(conflict_trace))
        conflict_ema_trace = np.array(getattr(trainer, "conflict_ema_trace", []), dtype=np.float64)
        if len(conflict_ema_trace) > 0:
            trainer_diagnostics["adaptive_conflict_ema_mean"] = float(np.mean(conflict_ema_trace))
            trainer_diagnostics["adaptive_conflict_ema_tail_mean"] = float(np.mean(conflict_ema_trace[-tail_n:]))
        for attr, out_key in [
            ("current_replay_ce_cos_trace", "adaptive_current_replay_ce_cos"),
            ("current_dark_cos_trace", "adaptive_current_dark_cos"),
            ("ce_dark_cos_trace", "adaptive_ce_dark_cos"),
        ]:
            vals = np.array(getattr(trainer, attr, []), dtype=np.float64)
            if len(vals) > 0:
                trainer_diagnostics[f"{out_key}_mean"] = float(np.mean(vals))
                trainer_diagnostics[f"{out_key}_tail_mean"] = float(np.mean(vals[-tail_n:]))

    dark_alpha_trace = np.array(getattr(trainer, "dark_alpha_trace", []), dtype=np.float64)
    if len(dark_alpha_trace) > 0:
        tail_n = min(200, len(dark_alpha_trace))
        trainer_diagnostics["dark_effective_alpha_mean"] = float(np.mean(dark_alpha_trace))
        trainer_diagnostics["dark_effective_alpha_tail_mean"] = float(np.mean(dark_alpha_trace[-tail_n:]))

    for attr, out_key in [
        ("pressure_weight_trace", "pressure_dark_weight"),
        ("reliability_trace", "pressure_dark_reliability"),
        ("loss_pressure_trace", "pressure_dark_loss_pressure"),
        ("drift_pressure_trace", "pressure_dark_drift_pressure"),
    ]:
        vals = np.array(getattr(trainer, attr, []), dtype=np.float64)
        if len(vals) > 0:
            tail_n = min(200, len(vals))
            trainer_diagnostics[f"{out_key}_mean"] = float(np.mean(vals))
            trainer_diagnostics[f"{out_key}_tail_mean"] = float(np.mean(vals[-tail_n:]))

    return dict(
        method=method_name,
        seed=seed,
        acc_matrix=acc_matrix.tolist(),
        diagnostics=diagnostics_over_time,
        trainer_diagnostics=trainer_diagnostics,
        plasticity_first_batch_acc=plasticity_first_batch_acc,
        n_tasks=n_tasks,
        mode=mode,
        trainer_kwargs=used_trainer_kwargs,
    )


def summarize(result):
    A = np.array(result["acc_matrix"])
    n = A.shape[0]
    diag = np.array([A[i, i] for i in range(n)])
    last_row = A[-1, :]
    bwt_terms = [last_row[j] - A[j, j] for j in range(n - 1)]
    bwt = float(np.mean(bwt_terms)) if len(bwt_terms) > 0 else float("nan")
    final_avg_acc = float(np.nanmean(last_row))
    forgetting_terms = []
    for j in range(n - 1):
        history = A[j:, j]
        best_seen = np.nanmax(history)
        forgetting_terms.append(best_seen - last_row[j])
    mean_forgetting = float(np.mean(forgetting_terms)) if forgetting_terms else float("nan")
    learned_avg_acc = float(np.nanmean(diag))
    retention_ratio = float(final_avg_acc / max(learned_avg_acc, 1e-12))
    if n >= 2:
        plasticity_slope = float(np.polyfit(np.arange(n), diag, 1)[0])
    else:
        plasticity_slope = float("nan")
    return dict(
        diag=diag.tolist(),
        bwt=bwt,
        final_avg_acc=final_avg_acc,
        learned_avg_acc=learned_avg_acc,
        mean_forgetting=mean_forgetting,
        retention_ratio=retention_ratio,
        plasticity_slope=plasticity_slope,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Permuted-Pi-Digits continual-learning benchmark.")
    parser.add_argument("mode", nargs="?", default="label_permuted",
                        choices=["label_permuted", "input_permuted", "class_il", "conflicting"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=sorted(TRAINER_REGISTRY),
                        help="Methods to run. Defaults to all core and improved methods.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-tasks", type=int, default=80)
    parser.add_argument("--steps-per-task", type=int, default=4000)
    parser.add_argument("--test-per-task", type=int, default=300)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--replay-capacity", type=int, default=None,
                        help="Override replay buffer capacity for replay-based methods.")
    parser.add_argument("--replay-batch", type=int, default=None,
                        help="Override replay mini-batch size for replay-based methods.")
    parser.add_argument("--ewc-lam", type=float, default=None,
                        help="Override EWC lambda for EWC and ReplayEWC.")
    parser.add_argument("--ewc-fisher-batches", type=int, default=None,
                        help="Override the number of batches used to estimate Fisher.")
    parser.add_argument("--ewc-fisher-decay", type=float, default=None,
                        help="Override online EWC Fisher decay.")
    parser.add_argument("--ewc-grad-clip", type=float, default=None,
                        help="Override EWC regularizer gradient clipping norm.")
    parser.add_argument("--dark-alpha", type=float, default=None,
                        help="Override logit-consistency strength for DarkReplayEWC.")
    parser.add_argument("--dark-confidence-threshold", type=float, default=None,
                        help="Only distill stored logits above this target confidence; lower-confidence samples get downweighted.")
    parser.add_argument("--dark-require-correct", action="store_true",
                        help="Only distill stored logits whose argmax matched the sample label when saved.")
    parser.add_argument("--distill-start-task", type=int, default=None,
                        help="Do not apply dark-logit distillation before this task index.")
    parser.add_argument("--distill-ramp-tasks", type=int, default=None,
                        help="Linearly ramp dark-logit distillation over this many tasks after distill-start-task.")
    parser.add_argument("--pressure-loss-low", type=float, default=None,
                        help="Replay CE loss where PressureDarkReplayEWC begins turning on distillation.")
    parser.add_argument("--pressure-loss-high", type=float, default=None,
                        help="Replay CE loss where PressureDarkReplayEWC reaches full loss pressure.")
    parser.add_argument("--pressure-drift-low", type=float, default=None,
                        help="Logit RMSE where PressureDarkReplayEWC begins turning on distillation.")
    parser.add_argument("--pressure-drift-high", type=float, default=None,
                        help="Logit RMSE where PressureDarkReplayEWC reaches full drift pressure.")
    parser.add_argument("--dark-alpha-min", type=float, default=None,
                        help="Minimum logit-consistency strength for AdaptiveDarkReplayEWC.")
    parser.add_argument("--conflict-margin", type=float, default=None,
                        help="Positive shared-gradient cosine needed for AdaptiveDarkReplayEWC to reach maximum distillation.")
    parser.add_argument("--alpha-smoothing", type=float, default=None,
                        help="EMA smoothing factor for AdaptiveDarkReplayEWC's distillation strength.")
    parser.add_argument("--conflict-ema-decay", type=float, default=None,
                        help="EMA decay for AdaptiveDarkReplayEWC's stream-level conflict estimate.")
    parser.add_argument("--replay-weight", type=float, default=None,
                        help="Override replay/current gradient mixing weight for replay methods that support it.")
    parser.add_argument("--candidate-mult", type=int, default=None,
                        help="Override candidate pool multiplier for SurpriseReplayEWC.")
    parser.add_argument("--margin-weight", type=float, default=None,
                        help="Override low-margin boundary priority for MarginSurpriseReplayEWC.")
    parser.add_argument("--memory-alpha", type=float, default=None,
                        help="Override episodic-memory interpolation weight for HippocampalReplayEWC.")
    parser.add_argument("--memory-temperature", type=float, default=None,
                        help="Override episodic prototype softmax temperature for HippocampalReplayEWC.")
    parser.add_argument("--memory-min-examples", type=int, default=None,
                        help="Override minimum memory examples before episodic readout is used.")
    parser.add_argument("--memory-task-filter", action="store_true",
                        help="Filter episodic memory by task id even for single-head mode.")
    parser.add_argument("--memory-fixed-alpha", action="store_true",
                        help="Use a fixed episodic-memory weight instead of uncertainty-gated memory.")
    parser.add_argument("--sleep-steps", type=int, default=None,
                        help="Run this many replay-only consolidation steps at the end of each task.")
    parser.add_argument("--sleep-batch", type=int, default=None,
                        help="Replay batch size used during sleep consolidation.")
    parser.add_argument("--sleep-lr-scale", type=float, default=None,
                        help="Learning-rate multiplier used during sleep consolidation.")
    parser.add_argument("--cbp-replacement-rate", type=float, default=None,
                        help="Override ContinualBP replacement rate.")
    parser.add_argument("--cbp-maturity-threshold", type=int, default=None,
                        help="Override ContinualBP maturity threshold.")
    parser.add_argument("--cbp-util-decay", type=float, default=None,
                        help="Override ContinualBP utility EMA decay.")
    parser.add_argument("--input-adapter", action="store_true",
                        help="Give each task a learnable input adapter (single-head input_permuted only). "
                             "Lets a shared net serve all input permutations; breaks the ~11%% floor.")
    parser.add_argument("--joint-batch", type=int, default=None,
                        help="Joint upper-bound mini-batch size sampled from all seen tasks.")
    parser.add_argument("--joint-steps", type=int, default=None,
                        help="Joint upper-bound i.i.d. update steps per incoming batch.")
    parser.add_argument("--bf-levels", type=int, default=None,
                        help="Benna-Fusi synapse chain length (number of timescales).")
    parser.add_argument("--bf-g0", type=float, default=None,
                        help="Benna-Fusi base conductance (coupling strength of the fastest variable).")
    parser.add_argument("--bf-dt", type=float, default=None,
                        help="Benna-Fusi relaxation step size: stability<->plasticity knob (higher = more stable).")
    parser.add_argument("--gpm-off", action="store_true",
                        help="Disable GPM gradient projection in FunctionSpaceReplay (leaves replay + DER++).")
    parser.add_argument("--gpm-threshold", type=float, default=None,
                        help="GPM explained-variance threshold for old-task subspace basis (higher = more protection, less plasticity).")
    parser.add_argument("--lookahead-beta", type=float, default=None,
                        help="Beta scaling parameter for LookaheadDarkReplayEWC.")
    parser.add_argument("--lookahead-ema-decay", type=float, default=None,
                        help="EMA decay parameter for LookaheadDarkReplayEWC's conflict tracker.")
    parser.add_argument("--rtp-cos-threshold", type=float, default=None,
                        help="Cosine threshold for RtpDarkReplayEWC regime detector.")
    parser.add_argument("--rtp-probe-steps", type=int, default=None,
                        help="Number of steps to probe gradient cosine in RtpDarkReplayEWC.")
    parser.add_argument("--consolidate-every", type=int, default=None,
                        help="Step interval for task-free online EWC consolidation (Online*EWC* trainers).")
    parser.add_argument("--fisher-sample", type=int, default=None,
                        help="Buffer sample size for task-free online Fisher estimation (Online*EWC* trainers).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to results_<mode>.json in the project folder.")
    args = parser.parse_args()

    mode = args.mode
    n_tasks = args.n_tasks
    steps_per_task = args.steps_per_task
    test_per_task = args.test_per_task
    K = args.K
    seeds = args.seeds
    methods = args.methods
    lr = args.lr
    trainer_kwargs = {}
    if args.replay_capacity is not None:
        trainer_kwargs["capacity"] = args.replay_capacity
    if args.replay_batch is not None:
        trainer_kwargs["replay_batch"] = args.replay_batch
    if args.ewc_lam is not None:
        trainer_kwargs["lam"] = args.ewc_lam
    if args.ewc_fisher_batches is not None:
        trainer_kwargs["fisher_batches"] = args.ewc_fisher_batches
    if args.ewc_fisher_decay is not None:
        trainer_kwargs["fisher_decay"] = args.ewc_fisher_decay
    if args.ewc_grad_clip is not None:
        trainer_kwargs["grad_clip_norm"] = args.ewc_grad_clip
    if args.dark_alpha is not None:
        trainer_kwargs["dark_alpha"] = args.dark_alpha
    if args.dark_confidence_threshold is not None:
        trainer_kwargs["dark_confidence_threshold"] = args.dark_confidence_threshold
    if args.dark_require_correct:
        trainer_kwargs["dark_require_correct"] = True
    if args.distill_start_task is not None:
        trainer_kwargs["distill_start_task"] = args.distill_start_task
    if args.distill_ramp_tasks is not None:
        trainer_kwargs["distill_ramp_tasks"] = args.distill_ramp_tasks
    if args.pressure_loss_low is not None:
        trainer_kwargs["pressure_loss_low"] = args.pressure_loss_low
    if args.pressure_loss_high is not None:
        trainer_kwargs["pressure_loss_high"] = args.pressure_loss_high
    if args.pressure_drift_low is not None:
        trainer_kwargs["pressure_drift_low"] = args.pressure_drift_low
    if args.pressure_drift_high is not None:
        trainer_kwargs["pressure_drift_high"] = args.pressure_drift_high
    if args.dark_alpha_min is not None:
        trainer_kwargs["dark_alpha_min"] = args.dark_alpha_min
    if args.conflict_margin is not None:
        trainer_kwargs["conflict_margin"] = args.conflict_margin
    if args.alpha_smoothing is not None:
        trainer_kwargs["alpha_smoothing"] = args.alpha_smoothing
    if args.conflict_ema_decay is not None:
        trainer_kwargs["conflict_ema_decay"] = args.conflict_ema_decay
    if args.replay_weight is not None:
        trainer_kwargs["replay_weight"] = args.replay_weight
    if args.candidate_mult is not None:
        trainer_kwargs["candidate_mult"] = args.candidate_mult
    if args.margin_weight is not None:
        trainer_kwargs["margin_weight"] = args.margin_weight
    if args.memory_alpha is not None:
        trainer_kwargs["memory_alpha"] = args.memory_alpha
    if args.memory_temperature is not None:
        trainer_kwargs["memory_temperature"] = args.memory_temperature
    if args.memory_min_examples is not None:
        trainer_kwargs["memory_min_examples"] = args.memory_min_examples
    if args.memory_task_filter:
        trainer_kwargs["memory_task_filter"] = True
    if args.memory_fixed_alpha:
        trainer_kwargs["memory_gate"] = False
    if args.sleep_steps is not None:
        trainer_kwargs["sleep_steps"] = args.sleep_steps
    if args.sleep_batch is not None:
        trainer_kwargs["sleep_batch"] = args.sleep_batch
    if args.sleep_lr_scale is not None:
        trainer_kwargs["sleep_lr_scale"] = args.sleep_lr_scale
    if args.cbp_replacement_rate is not None:
        trainer_kwargs["replacement_rate"] = args.cbp_replacement_rate
    if args.cbp_maturity_threshold is not None:
        trainer_kwargs["maturity_threshold"] = args.cbp_maturity_threshold
    if args.cbp_util_decay is not None:
        trainer_kwargs["util_decay"] = args.cbp_util_decay
    if args.joint_batch is not None:
        trainer_kwargs["joint_batch"] = args.joint_batch
    if args.joint_steps is not None:
        trainer_kwargs["joint_steps"] = args.joint_steps
    if args.bf_levels is not None:
        trainer_kwargs["bf_levels"] = args.bf_levels
    if args.bf_g0 is not None:
        trainer_kwargs["bf_g0"] = args.bf_g0
    if args.bf_dt is not None:
        trainer_kwargs["bf_dt"] = args.bf_dt
    if args.gpm_off:
        trainer_kwargs["use_gpm"] = False
    if args.gpm_threshold is not None:
        trainer_kwargs["gpm_threshold"] = args.gpm_threshold
    if args.lookahead_beta is not None:
        trainer_kwargs["lookahead_beta"] = args.lookahead_beta
    if args.lookahead_ema_decay is not None:
        trainer_kwargs["lookahead_ema_decay"] = args.lookahead_ema_decay
    if args.rtp_cos_threshold is not None:
        trainer_kwargs["rtp_cos_threshold"] = args.rtp_cos_threshold
    if args.rtp_probe_steps is not None:
        trainer_kwargs["rtp_probe_steps"] = args.rtp_probe_steps
    if args.consolidate_every is not None:
        trainer_kwargs["consolidate_every"] = args.consolidate_every
    if args.fisher_sample is not None:
        trainer_kwargs["fisher_sample"] = args.fisher_sample

    stream_kwargs_template = dict(
        digits_file="pi_digits_600000.txt",
        K=K, n_tasks=n_tasks, steps_per_task=steps_per_task, test_per_task=test_per_task,
        mode=mode,
    )
    model_kwargs = dict(in_dim=10 * K, h1=64, h2=64, out_dim=10, input_adapter=args.input_adapter)

    all_results = {}
    t_start = time.time()
    for method in methods:
        all_results[method] = []
        for seed in seeds:
            t0 = time.time()
            res = run_one(method, seed, dict(stream_kwargs_template), model_kwargs,
                          lr=lr, batch_size=args.batch_size, trainer_kwargs=trainer_kwargs)
            summary = summarize(res)
            res["summary"] = summary
            all_results[method].append(res)
            print(f"[{method} seed={seed} mode={mode}] final_avg_acc={summary['final_avg_acc']:.3f} "
                  f"bwt={summary['bwt']:.3f} diag_first={summary['diag'][0]:.3f} "
                  f"diag_last={summary['diag'][-1]:.3f} retention={summary['retention_ratio']:.3f} "
                  f"forget={summary['mean_forgetting']:.3f}  ({time.time()-t0:.1f}s)")
    print(f"total time {time.time()-t_start:.1f}s")

    out_path = args.output or os.path.join(OUT_DIR, f"results_{mode}.json")
    if not os.path.isabs(out_path):
        out_path = os.path.join(OUT_DIR, out_path)
    with open(out_path, "w") as f:
        json.dump(all_results, f)
    print("saved to", out_path)


if __name__ == "__main__":
    main()
