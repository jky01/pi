# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full benchmark (saves results_<mode>.json)
python run.py label_permuted
python run.py input_permuted

# Run specific methods / seeds / hyperparameters
python run.py label_permuted --methods ReplayEWC HippocampalReplayEWC --seeds 0 1 --n-tasks 40

# Joint offline upper bound (ceiling for final_avg_acc; ignore its BWT/diagonal)
python run.py label_permuted --methods Joint --output results_joint_label_permuted.json

# input_permuted with a per-task input adapter (breaks the single-head ~11% floor)
python run.py input_permuted --methods Naive Joint --input-adapter --output results_adapter.json

# Generate summary stats and PNG figures from results
python analyze.py label_permuted
python analyze.py input_permuted

# Smoke tests (all methods × both modes, tiny scale)
python -m unittest test_smoke.py

# Run a single (method, seed, hyperparams) combo — used for tuning
python run_one_combo.py
python tune_replay_ewc.py
```

## Architecture

The project is a continual learning research benchmark with pure-numpy training (no PyTorch/JAX).

### Data pipeline (`benchmark.py`)
`PermutedPiDigitsStream` reads `pi_digits_600000.txt` and constructs a stream of tasks. Each task is a sliding-window classification problem: predict which of 10 equal-frequency buckets the sum of K consecutive π digits falls into. Bucket thresholds are calibrated on a held-out tail segment (not train/test data). Two modes:
- **`label_permuted`** — same input encoding per task, but label→class mapping is randomly permuted per task. This is Task-IL; the model is multi-head.
- **`input_permuted`** — label mapping is identity, but input features are permuted per task. This is Domain-IL; the model is single-head.

### Model (`model.py`)
`MLP`: pure numpy 2-hidden-layer network with manual `forward`/`backward`. Supports single-head or multi-head (one output head per task, shared hidden layers). `diagnostics()` computes `eff_rank_h1/h2` (effective rank via SVD entropy, measures representation collapse) and `dead_frac_h1/h2` (fraction of always-zero ReLU units). `label_permuted` → `multi_head=True`; `input_permuted` → `multi_head=False`.

### Trainer hierarchy (`trainers.py`)
All trainers implement `train_step(X, Y, task_idx) -> (loss, acc)` and `on_task_end(task_X, task_Y, task_idx)`. Registered in `TRAINER_REGISTRY`:

```
NaiveTrainer                          # SGD lower bound
JointTrainer                          # offline i.i.d. upper bound (unbounded buffer, breaks compute parity)
ReplayTrainer                         # reservoir buffer, uniform sampling
  TaskBalancedReplayTrainer           # per-task reservoir, balanced slots
    ReplayContinualBackpropTrainer    # +continual backprop (neuron recycling)
  BennaFusiReplayTrainer              # +Benna-Fusi complex synapses (instead of EWC)
  ReplayEWCTrainer                    # +online EWC regularization
    SustainableReplayEWCTrainer       # +Fisher-protected neuron recycling (plasticity w/o memory damage)
    DarkReplayEWCTrainer              # +logit-consistency loss on replayed items
      FunctionSpaceReplayTrainer      # +DER++ distillation +GPM gradient projection (toggleable ablation)
    SurpriseReplayEWCTrainer          # +surprise-prioritized replay (top-K CE loss)
      MarginSurpriseReplayEWCTrainer  # +low-margin boundary priority
      HippocampalReplayEWCTrainer     # +episodic prototype memory at inference
EWCTrainer                            # EWC only (no replay)
ContinualBackpropTrainer              # neuron recycling only (no replay)
BennaFusiTrainer                      # multi-timescale complex synapses, online (no replay/Fisher); --bf-dt knob
```

The sustainable-learning investigation (report §9) established that **forgetting, not plasticity, is the bottleneck wherever replay is present** — the diagonal keeps rising to 250 tasks. So `SustainableReplayEWC` (Fisher-protected recycling) and `BennaFusi` (power-law forgetting) are both validated mechanisms that only pay off in the **replay-free** regime; with replay, Fisher-selective `ReplayEWC` dominates. `bf_dt` is the stability↔plasticity knob for Benna-Fusi.

`FunctionSpaceReplayTrainer` (report §10) attacks forgetting in function space: Replay + DER++ logit distillation + GPM gradient projection, all toggleable (`use_gpm`, `dark_alpha`, `lam`). The ablation found **GPM is counterproductive here** — it protects against input-subspace shift, but these tasks have stationary inputs (label_permuted permutes labels; adapters restore canonical inputs), so GPM freezes the shared layers. **DER++ logit distillation is the win**: `DarkReplayEWC` (= Replay+EWC+distillation, `--dark-alpha 0.5`) drives 130-task retention to ~1.0 (BWT≈0), the project's best anti-forgetting result. Lesson: anchor the *function* (outputs), not weights or input subspaces.

`HippocampalReplayEWCTrainer` overrides `loss_acc()` to blend MLP softmax with prototype-based episodic predictions at evaluation time (the buffer doubles as a hippocampal episodic memory). The blending weight is optionally uncertainty-gated by the MLP's top-2 margin.

`JointTrainer` stores all seen samples and trains on i.i.d. mini-batches sampled from the union of all tasks. It deliberately breaks compute/memory parity, so it is only a ceiling — report its `final_avg_acc`, not its BWT/diagonal (it is not a sequential learner, so those metrics are artifacts). Run it explicitly via `--methods Joint`; it is not in `DEFAULT_METHODS`.

### Per-task input adapter (`model.py`, `--input-adapter`)
For `input_permuted`, a single shared `W1` cannot invert 80 different input permutations. `MLP(input_adapter=True)` adds one identity-initialized `in_dim × in_dim` matrix per task (`self.adapters[task_idx]`), applied as `X @ adapter` before `W1`. Each task's adapter learns to map its permuted input back into the shared network's canonical space, so the shared layers only need to learn one `sum→bucket` function. Adapter-correct trainers: `Naive` and `Joint` (they forward each sample with its own `task_idx`). Replay-based trainers do **not** yet route replayed samples through the right adapter — adding that requires per-task grouping in the single-head replay path.

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
