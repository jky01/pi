# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Research log & roadmap:** `report.md` is the running findings log (§0 has a TL;DR). **`NEXT_STEPS.md` lists the unfinished work** (rationale + planned approach + the correct numpy Python interpreter path) — read it first to resume the ongoing research. Established results not to re-derive are summarized there.

## Commands

```bash
# Required local interpreter (system python/python3 lacks numpy here)
PY=/Users/jackyyeh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3

# Run the full benchmark (saves results_<mode>.json)
$PY run.py label_permuted
$PY run.py input_permuted

# Run specific methods / seeds / hyperparameters
$PY run.py label_permuted --methods ReplayEWC HippocampalReplayEWC --seeds 0 1 --n-tasks 40

# Joint offline upper bound (ceiling for final_avg_acc; ignore its BWT/diagonal)
$PY run.py label_permuted --methods Joint --output results_joint_label_permuted.json

# input_permuted with a per-task input adapter (breaks the single-head ~11% floor)
$PY run.py input_permuted --methods Naive Joint --input-adapter --output results_adapter.json

# Generate summary stats and PNG figures from results
$PY analyze.py label_permuted
$PY analyze.py input_permuted

# Smoke tests (all methods × label/input/conflicting modes, tiny scale)
$PY -m unittest test_smoke

# Run a single (method, seed, hyperparams) combo — used for tuning
$PY run_one_combo.py
$PY tune_replay_ewc.py

# Standard-benchmark external-validity check on real MNIST (report §17 / P7).
# Pure-numpy MNIST loader + a stream with the same interface as the pi stream, so it
# reuses the MLP and every trainer. Validates which pi-digit findings transfer.
$PY run_mnist.py permuted --methods Naive ReplayEWC DarkReplayEWC --seeds 0 1 2  # Permuted-MNIST Task-IL
$PY run_mnist.py split --methods Naive ReplayEWC DarkReplayEWC NCMReplayEWC --seeds 0 1 2  # Split-MNIST class-IL

# Frozen pretrained features + Class-IL on Split-CIFAR-100 (report §18 / P8).
# extract_features.py is the ONLY step that uses torch (frozen ImageNet ResNet18 ->
# cifar100_resnet18.npz, gitignored); run_features.py then trains a small head in pure numpy.
$PY extract_features.py                       # one-time: cache CIFAR-100 ResNet18 features
$PY run_features.py --methods Naive ReplayEWC DarkReplayEWC NCMReplayEWC --seeds 0 1 2
```

## Architecture

The project is a continual learning research benchmark with pure-numpy training (no PyTorch/JAX).

### Data pipeline (`benchmark.py`)
`PermutedPiDigitsStream` reads `pi_digits_600000.txt` and constructs a stream of tasks. Each task is a sliding-window classification problem: predict which of 10 equal-frequency buckets the sum of K consecutive π digits falls into. Bucket thresholds are calibrated on a held-out tail segment (not train/test data). Two modes:
- **`label_permuted`** — same input encoding per task, but label→class mapping is randomly permuted per task. This is Task-IL; the model is multi-head.
- **`input_permuted`** — label mapping is identity, but input features are permuted per task. This is Domain-IL; the model is single-head.
- **`class_il`** — single head over a global label space of `10×n_tasks` fine-grained sum-buckets, **no task id at inference**. Each task owns a disjoint contiguous slice of sum-buckets (a distinct sum-range), built by scanning all windows and grouping by global bucket (`_build_class_il`). The honest hard test (report §11): the linear head suffers catastrophic recency bias (DER++ *hurts* here), but `NCMReplayEWC` (nearest-class-mean prototype readout) fixes it — **0.31 → 0.858, forgetting → 0.06**. Benchmark caps at ~200 classes (K=8 sum has only ~73 distinct values).
- **`conflicting`** — multi-head Task-IL, but the underlying task rule rotates across sum / weighted sum / half-window sums / adjacent-product features. This is the main true-conflict benchmark (report §12): ReplayEWC is the safe backbone, while fixed DER++ can over-constrain learning.

### Model (`model.py`)
`MLP`: pure numpy 2-hidden-layer network with manual `forward`/`backward`. Supports single-head or multi-head (one output head per task, shared hidden layers). `diagnostics()` computes `eff_rank_h1/h2` (effective rank via SVD entropy, measures representation collapse) and `dead_frac_h1/h2` (fraction of always-zero ReLU units). `label_permuted` → `multi_head=True`; `input_permuted` → `multi_head=False`.

### Trainer hierarchy (`trainers.py`)
All trainers implement `train_step(X, Y, task_idx) -> (loss, acc)` and `on_task_end(task_X, task_Y, task_idx)`. Registered in `TRAINER_REGISTRY` (30 methods):

```
Naive, Joint, EWC, Replay, ReplayEWC, DarkReplayEWC,
AdaptiveDarkReplayEWC, PressureDarkReplayEWC,
LookaheadDarkReplayEWC, RtpDarkReplayEWC, HorizonDarkReplayEWC,
BenefitDarkReplayEWC, SlowBenefitDarkReplayEWC,
OnlineEWCReplay, OnlineDarkReplayEWC,
GenerativeReplayEWC, NBGenerativeReplayEWC,
ScholarGenerativeReplayEWC, ScholarGlobalGenerativeReplayEWC,
SurpriseReplayEWC, MarginSurpriseReplayEWC, HippocampalReplayEWC,
NCMReplayEWC, TaskBalancedReplay,
ContinualBP, ReplayContinualBP, SustainableReplayEWC,
BennaFusi, BennaFusiReplay, FunctionSpaceReplay
```

The sustainable-learning investigation (report §9) established that **forgetting, not plasticity, is the bottleneck wherever replay is present** — the diagonal keeps rising to 250 tasks. So `SustainableReplayEWC` (Fisher-protected recycling) and `BennaFusi` (power-law forgetting) are both validated mechanisms that only pay off in the **replay-free** regime; with replay, Fisher-selective `ReplayEWC` dominates. `bf_dt` is the stability↔plasticity knob for Benna-Fusi.

`FunctionSpaceReplayTrainer` (report §10) attacks forgetting in function space: Replay + DER++ logit distillation + GPM gradient projection, all toggleable (`use_gpm`, `dark_alpha`, `lam`). The ablation found **GPM is counterproductive here** — it protects against input-subspace shift, but these tasks have stationary inputs (label_permuted permutes labels; adapters restore canonical inputs), so GPM freezes the shared layers. **DER++ logit distillation is the win**: `DarkReplayEWC` (= Replay+EWC+distillation, `--dark-alpha 0.5`) drives 130-task retention to ~1.0 (BWT≈0), the project's best anti-forgetting result. Lesson: anchor the *function* (outputs), not weights or input subspaces.

Adaptive/regime-gated distillation status (report §12):
- `AdaptiveDarkReplayEWC` and `PressureDarkReplayEWC` are useful safety gates, but too reactive to recover full long-stream DER++ gains.
- `RtpDarkReplayEWC` is a negative result: task-onset shared-gradient cosine almost always shuts DER++ off, so it fails on 130-task label_permuted.
- `HorizonDarkReplayEWC` (P2.7) is an oracle validation: known horizon can choose ReplayEWC-like behavior for short/conflicting streams and DER++ for standard long streams, but horizon alone is too crude (80-task × 2000 steps is a counterexample).
- `BenefitDarkReplayEWC` (P2.8) is a partial negative result: one-step label-loss probes are safe but collapse to ReplayEWC on long streams; logit-MSE probes open DER++ but false-positive on short/conflicting/immature streams.
- `SlowBenefitDarkReplayEWC` (P2.9) is a negative result: recent-window 5-step shadow rollout preserves P2.8 safety but still keeps alpha=0 on long streams; tiny logit weight (0.02) false-positives on 80-task × 2000. Next open direction is persistent shadow-model bandit or explicit horizon/budget/regime prior, not another local rollout.

`HippocampalReplayEWCTrainer` overrides `loss_acc()` to blend MLP softmax with prototype-based episodic predictions at evaluation time (the buffer doubles as a hippocampal episodic memory). The blending weight is optionally uncertainty-gated by the MLP's top-2 margin.

`JointTrainer` stores all seen samples and trains on i.i.d. mini-batches sampled from the union of all tasks. It deliberately breaks compute/memory parity, so it is only a ceiling — report its `final_avg_acc`, not its BWT/diagonal (it is not a sequential learner, so those metrics are artifacts). Run it explicitly via `--methods Joint`; it is not in `DEFAULT_METHODS`.

### Per-task input adapter (`model.py`, `--input-adapter`)
For `input_permuted`, a single shared `W1` cannot invert 80 different input permutations. `MLP(input_adapter=True)` adds one identity-initialized `in_dim × in_dim` matrix per task (`self.adapters[task_idx]`), applied as `X @ adapter` before `W1`. Each task's adapter learns to map its permuted input back into the shared network's canonical space, so the shared layers only need to learn one `sum→bucket` function. Replay and memory paths are adapter-correct: replayed samples are grouped/routed by their stored `task_idx`, so old samples use the right adapter.

`on_task_end` is where EWC computes Fisher diagonals and updates anchors; `HippocampalReplayEWC` also runs optional sleep-consolidation replay steps here.

### Orchestration (`run.py`)
`run_one(method, seed, stream_kwargs, model_kwargs, lr, ...)` runs one (method × seed) trial and returns a dict with `acc_matrix` (n_tasks × n_tasks), per-task `diagnostics`, and `plasticity_first_batch_acc`. `main()` iterates over methods and seeds, prints per-run metrics, and saves to `results_<mode>.json`.

### Analysis (`analyze.py`)
Loads `results_<mode>.json`, aggregates across seeds, and outputs:
- `summary_stats_<mode>.json` — mean/std of all metrics per method
- `fig1_diagonal_accuracy_<mode>.png` — plasticity curve (diagonal A[t,t])
- `fig2_bwt_finalacc_<mode>.png` — BWT and final average accuracy bar charts
- `fig3_plasticity_diagnostics_<mode>.png` — dead unit fraction and effective rank over tasks

### Key metrics
- `acc_matrix[i][j]` — accuracy on task j after training through task i
- **Diagonal** `A[i,i]` — plasticity (how well the model learned task i just after seeing it)
- **BWT** — `mean(A[n-1,j] - A[j,j])`, negative = forgetting
- **mean_forgetting** — `mean(best_seen[j] - final[j])`, always ≥ 0
- **retention_ratio** — `final_avg_acc / learned_avg_acc`
- **plasticity_slope** — linear trend of diagonal over task index (negative = plasticity decay)
