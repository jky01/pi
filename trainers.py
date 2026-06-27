"""
多種持續學習策略，皆共用同一個 MLP (model.py)，差別只在「怎麼更新權重」：

1. NaiveTrainer            — 下界。單純線上 SGD，什麼保護機制都沒有。
2. EWCTrainer               — Elastic Weight Consolidation（正則化派代表）。
3. ReplayTrainer            — Experience Replay，小型 reservoir buffer（重播派代表）。
4. ReplayEWCTrainer         — Replay + online EWC，結合樣本重播與參數保護。
5. DarkReplayEWCTrainer     — ReplayEWC + logits consistency（DER/SER 系列方向）。
6. AdaptiveDarkReplayEWCTrainer
                            — DarkReplayEWC + gradient-conflict gated distillation。
7. PressureDarkReplayEWCTrainer
                            — DarkReplayEWC + reliability × forgetting-pressure distillation。
8. HorizonDarkReplayEWCTrainer
                            — oracle horizon-gated DER++：長流開 logits distillation，
                              短流/衝突關閉，驗證 P2.7 regime 訊號需求。
9. BenefitDarkReplayEWCTrainer
                            — P2.8：用 function-space 反事實收益偵測是否開 DER++。
10. SlowBenefitDarkReplayEWCTrainer
                            — P2.9：多步 shadow rollout，量測慢時間尺度 DER++ 收益。
11. SurpriseReplayEWCTrainer — ReplayEWC + loss/surprise-prioritized replay sampling。
12. MarginSurpriseReplayEWCTrainer
                            — ReplayEWC + loss/surprise + low-margin boundary replay。
13. HippocampalReplayEWCTrainer
                            — SurpriseReplayEWC + episodic prototype memory at inference。
14. ContinualBackpropTrainer — Sutton/Dohare 的 continual backprop：只選擇性地
                              重置「低效用、夠老」的死/低貢獻單元，其餘權重完全
                              不動——這是對話第一輪明確回答「不重置權重」的機制，
                              主打可塑性流失（失效 B），跟前兩者主打遺忘（失效 A）形成對照。

所有 trainer 都實作同一介面：
    train_step(X, Y) -> (loss, acc)   在這個 batch 上更新一次參數，回傳更新前的 loss/acc
    on_task_end(task_X, task_Y)       task 訓練段結束時呼叫的 hook（EWC 用來算 Fisher）
"""
import copy

import numpy as np

from model import MLP


class NaiveTrainer:
    name = "Naive"

    def __init__(self, model: MLP, lr: float = 0.05):
        self.model = model
        self.lr = lr

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)
        self.model.sgd_step(grads, self.lr, task_idx=task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        pass


class EWCTrainer:
    """Online EWC（Schwarz et al. 2018 的做法，而非原版逐 task 無上限累加 Fisher）：
    fisher <- gamma * fisher + new_fisher，而不是 fisher += new_fisher。
    """
    name = "EWC"

    def __init__(self, model: MLP, lr: float = 0.05, lam: float = 20.0, fisher_batches: int = 30,
                 fisher_decay: float = 0.9, grad_clip_norm: float = 50.0):
        self.model = model
        self.lr = lr
        self.lam = lam
        self.fisher_batches = fisher_batches
        self.fisher_decay = fisher_decay
        self.grad_clip_norm = grad_clip_norm
        
        self.reg_keys = ["W1", "b1", "W2", "b2"]
        if not model.multi_head:
            self.reg_keys.append("W3")
            self.reg_keys.append("b3")
            
        init_params = {}
        if model.multi_head:
            p_list = model.params(task_idx=0)
            init_params["W1"], init_params["b1"], init_params["W2"], init_params["b2"] = p_list[0], p_list[1], p_list[2], p_list[3]
        else:
            p_list = model.params()
            init_params["W1"], init_params["b1"], init_params["W2"], init_params["b2"], init_params["W3"], init_params["b3"] = p_list
            
        self.fisher = {k: np.zeros_like(init_params[k]) for k in self.reg_keys}
        self.anchor = {k: init_params[k].copy() for k in self.reg_keys}

    def _ewc_grad(self, task_idx=None):
        m = self.model
        if m.multi_head:
            p_list = m.params(task_idx)
            cur = dict(W1=p_list[0], b1=p_list[1], W2=p_list[2], b2=p_list[3])
        else:
            p_list = m.params()
            cur = dict(W1=p_list[0], b1=p_list[1], W2=p_list[2], b2=p_list[3], W3=p_list[4], b3=p_list[5])
            
        grads = {k: self.lam * self.fisher[k] * (cur[k] - self.anchor[k]) for k in self.reg_keys}
        total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if total_norm > self.grad_clip_norm and total_norm > 0:
            scale = self.grad_clip_norm / total_norm
            grads = {k: g * scale for k, g in grads.items()}
        return grads

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)
        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        """用這個 task 的訓練資料估計 Fisher 對角線，以 gamma 衰減後累加（online EWC），
        並把 anchor 移到目前參數。"""
        m = self.model
        n = task_X.shape[0]
        bs = max(1, n // self.fisher_batches)
        accum = {k: np.zeros_like(self.fisher[k]) for k in self.reg_keys}
        n_batches = 0
        for i in range(0, n, bs):
            xb, yb = task_X[i:i + bs], task_Y[i:i + bs]
            if len(xb) == 0:
                continue
            cache = m.forward(xb, task_idx)
            grads = m.backward(cache, yb, task_idx)
            for k in self.reg_keys:
                accum[k] += grads[k] ** 2
            n_batches += 1
        for k in self.reg_keys:
            self.fisher[k] = self.fisher_decay * self.fisher[k] + accum[k] / max(1, n_batches)
            
        if m.multi_head:
            p_list = m.params(task_idx)
            self.anchor = dict(W1=p_list[0].copy(), b1=p_list[1].copy(), W2=p_list[2].copy(), b2=p_list[3].copy())
        else:
            p_list = m.params()
            self.anchor = dict(W1=p_list[0].copy(), b1=p_list[1].copy(), W2=p_list[2].copy(),
                               b2=p_list[3].copy(), W3=p_list[4].copy(), b3=p_list[5].copy())


class ReplayTrainer:
    name = "Replay"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 500,
                 replay_batch: int = 16, seed: int = 0):
        self.model = model
        self.lr = lr
        self.capacity = capacity
        self.replay_batch = replay_batch
        self.buf_X = []
        self.buf_Y = []
        self.buf_task = []
        self.seen = 0
        self.rng = np.random.RandomState(seed)

    def _reservoir_insert(self, X, Y, task_idx=None):
        for i in range(X.shape[0]):
            self.seen += 1
            if len(self.buf_X) < self.capacity:
                self.buf_X.append(X[i].copy())
                self.buf_Y.append(int(Y[i]))
                self.buf_task.append(task_idx)
            else:
                j = self.rng.randint(0, self.seen)
                if j < self.capacity:
                    self.buf_X[j] = X[i].copy()
                    self.buf_Y[j] = int(Y[i])
                    self.buf_task[j] = task_idx

    def _sample_replay(self):
        if len(self.buf_X) == 0:
            return None
        idx = self.rng.randint(0, len(self.buf_X), size=min(self.replay_batch, len(self.buf_X)))
        rX = np.stack([self.buf_X[i] for i in idx])
        rY = np.array([self.buf_Y[i] for i in idx], dtype=np.int64)
        rtasks = [self.buf_task[i] for i in idx]
        return rX, rY, rtasks

    def _mix_replay_grads(self, grads, rX, rY, rtasks, task_idx):
        if self.model.multi_head:
            rgrads_accum = {k: np.zeros_like(v) for k, v in grads.items()}
            head_grads = {}

            for t in sorted(set(rtasks)):
                sub_idx = [i for i, val in enumerate(rtasks) if val == t]
                sub_X = rX[sub_idx]
                sub_Y = rY[sub_idx]

                sub_cache = self.model.forward(sub_X, t)
                sub_grads = self.model.backward(sub_cache, sub_Y, t)

                weight = len(sub_idx) / len(rtasks)
                for k in ["W1", "b1", "W2", "b2"]:
                    rgrads_accum[k] += weight * sub_grads[k]

                head_grads[t] = {
                    "W3": weight * sub_grads["W3"],
                    "b3": weight * sub_grads["b3"],
                }

            for k in ["W1", "b1", "W2", "b2"]:
                grads[k] = 0.5 * grads[k] + 0.5 * rgrads_accum[k]

            grads["W3"] = 0.5 * grads["W3"]
            grads["b3"] = 0.5 * grads["b3"]
            if task_idx in head_grads:
                grads["W3"] += 0.5 * head_grads[task_idx]["W3"]
                grads["b3"] += 0.5 * head_grads[task_idx]["b3"]

            for t, h_g in head_grads.items():
                if t != task_idx:
                    t_grads = {
                        "W1": np.zeros_like(self.model.W1),
                        "b1": np.zeros_like(self.model.b1),
                        "W2": np.zeros_like(self.model.W2),
                        "b2": np.zeros_like(self.model.b2),
                        "W3": h_g["W3"],
                        "b3": h_g["b3"],
                    }
                    self.model.sgd_step(t_grads, self.lr * 0.5, task_idx=t)
            return grads

        if self.model.input_adapter:
            return self._mix_replay_grads_adapter(grads, rX, rY, rtasks, task_idx)

        rcache = self.model.forward(rX)
        rgrads = self.model.backward(rcache, rY)
        return {k: 0.5 * grads[k] + 0.5 * rgrads[k] for k in grads}

    def _mix_replay_grads_adapter(self, grads, rX, rY, rtasks, task_idx):
        """Single-head + per-task input adapter: shared head (W3/b3) is shared across
        tasks, but each task's samples must go through its own adapter. Group replayed
        samples by task, accumulate shared grads, and update each replayed task's adapter."""
        shared_keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
        rgrads_accum = {k: np.zeros_like(grads[k]) for k in shared_keys}
        adapter_grads = {}
        for t in sorted(set(rtasks)):
            sub_idx = [i for i, val in enumerate(rtasks) if val == t]
            sub_cache = self.model.forward(rX[sub_idx], t)
            sub_grads = self.model.backward(sub_cache, rY[sub_idx], t)
            weight = len(sub_idx) / len(rtasks)
            for k in shared_keys:
                rgrads_accum[k] += weight * sub_grads[k]
            if "A" in sub_grads:
                adapter_grads[t] = weight * sub_grads["A"]

        mixed = {k: 0.5 * grads[k] + 0.5 * rgrads_accum[k] for k in shared_keys}
        # current task's adapter: carried in returned grads, applied by the outer sgd_step
        if "A" in grads:
            mixed["A"] = 0.5 * grads["A"]
            if task_idx in adapter_grads:
                mixed["A"] += 0.5 * adapter_grads[task_idx]
        # replayed (non-current) tasks' adapters: applied here
        for t, gA in adapter_grads.items():
            if t != task_idx:
                self.model.adapters[t] -= self.lr * 0.5 * gA
        return mixed

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        sample = self._sample_replay()
        if sample is not None:
            grads = self._mix_replay_grads(grads, *sample, task_idx)

        self.model.sgd_step(grads, self.lr, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        pass


class TaskBalancedReplayTrainer(ReplayTrainer):
    """Replay with a per-task reservoir and balanced sampling.

    A single global reservoir is cheap, but in long task streams it can leave early
    tasks with very few examples. This variant keeps the same total memory budget
    while spreading slots across observed tasks, then samples replay mini-batches
    round-robin across tasks.
    """
    name = "TaskBalancedReplay"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 500,
                 replay_batch: int = 16, seed: int = 0):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch, seed=seed)
        self.task_buffers = {}
        self.task_seen = {}

    @staticmethod
    def _task_key(task_idx):
        return 0 if task_idx is None else int(task_idx)

    def _quotas(self):
        tasks = sorted(self.task_buffers)
        if not tasks:
            return {}
        base = self.capacity // len(tasks)
        rem = self.capacity % len(tasks)
        return {task: base + (1 if i < rem else 0) for i, task in enumerate(tasks)}

    def _sync_flat_buffers(self):
        self.buf_X = []
        self.buf_Y = []
        self.buf_task = []
        for task in sorted(self.task_buffers):
            for x, y in self.task_buffers[task]:
                self.buf_X.append(x)
                self.buf_Y.append(y)
                self.buf_task.append(task)

    def _rebalance(self):
        quotas = self._quotas()
        for task, buf in self.task_buffers.items():
            quota = quotas.get(task, 0)
            while len(buf) > quota:
                del buf[self.rng.randint(0, len(buf))]
        self._sync_flat_buffers()

    def _reservoir_insert(self, X, Y, task_idx=None):
        task = self._task_key(task_idx)
        if task not in self.task_buffers:
            self.task_buffers[task] = []
            self.task_seen[task] = 0
            self._rebalance()

        quotas = self._quotas()
        quota = quotas.get(task, 0)
        buf = self.task_buffers[task]
        for i in range(X.shape[0]):
            self.seen += 1
            self.task_seen[task] += 1
            if quota <= 0:
                continue
            item = (X[i].copy(), int(Y[i]))
            if len(buf) < quota:
                buf.append(item)
            else:
                j = self.rng.randint(0, self.task_seen[task])
                if j < quota:
                    buf[j] = item
        self._sync_flat_buffers()

    def _sample_replay(self):
        tasks = [task for task, buf in self.task_buffers.items() if len(buf) > 0]
        if not tasks:
            return None
        total = sum(len(self.task_buffers[task]) for task in tasks)
        n = min(self.replay_batch, total)
        picks = []
        task_order = list(tasks)

        while len(picks) < n:
            self.rng.shuffle(task_order)
            for task in task_order:
                if len(picks) >= n:
                    break
                buf = self.task_buffers[task]
                if len(buf) == 0:
                    continue
                x, y = buf[self.rng.randint(0, len(buf))]
                picks.append((x, y, task))

        rX = np.stack([p[0] for p in picks])
        rY = np.array([p[1] for p in picks], dtype=np.int64)
        rtasks = [p[2] for p in picks]
        return rX, rY, rtasks


class ReplayEWCTrainer(ReplayTrainer):
    """Experience replay with online EWC regularization.

    Replay gives the optimizer old examples; EWC adds a soft penalty against
    moving shared weights away from high-Fisher anchors. In multi-head Task-IL
    mode the EWC term protects only shared hidden layers, leaving task heads free.
    """
    name = "ReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch, seed=seed)
        self.lam = lam
        self.fisher_batches = fisher_batches
        self.fisher_decay = fisher_decay
        self.grad_clip_norm = grad_clip_norm

        self.reg_keys = ["W1", "b1", "W2", "b2"]
        if not model.multi_head:
            self.reg_keys.extend(["W3", "b3"])

        init_params = self._current_reg_params(task_idx=0 if model.multi_head else None)
        self.fisher = {k: np.zeros_like(init_params[k]) for k in self.reg_keys}
        self.anchor = {k: init_params[k].copy() for k in self.reg_keys}

    def _current_reg_params(self, task_idx=None):
        m = self.model
        if m.multi_head:
            p_list = m.params(task_idx)
            return dict(W1=p_list[0], b1=p_list[1], W2=p_list[2], b2=p_list[3])
        p_list = m.params()
        return dict(W1=p_list[0], b1=p_list[1], W2=p_list[2],
                    b2=p_list[3], W3=p_list[4], b3=p_list[5])

    def _ewc_grad(self, task_idx=None):
        cur = self._current_reg_params(task_idx)
        grads = {k: self.lam * self.fisher[k] * (cur[k] - self.anchor[k]) for k in self.reg_keys}
        total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if total_norm > self.grad_clip_norm and total_norm > 0:
            scale = self.grad_clip_norm / total_norm
            grads = {k: g * scale for k, g in grads.items()}
        return grads

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        sample = self._sample_replay()
        if sample is not None:
            grads = self._mix_replay_grads(grads, *sample, task_idx)

        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        n = task_X.shape[0]
        bs = max(1, n // self.fisher_batches)
        accum = {k: np.zeros_like(self.fisher[k]) for k in self.reg_keys}
        n_batches = 0
        for i in range(0, n, bs):
            xb, yb = task_X[i:i + bs], task_Y[i:i + bs]
            if len(xb) == 0:
                continue
            cache = self.model.forward(xb, task_idx)
            grads = self.model.backward(cache, yb, task_idx)
            for k in self.reg_keys:
                accum[k] += grads[k] ** 2
            n_batches += 1

        for k in self.reg_keys:
            self.fisher[k] = self.fisher_decay * self.fisher[k] + accum[k] / max(1, n_batches)
        self.anchor = {k: v.copy() for k, v in self._current_reg_params(task_idx).items()}


class DarkReplayEWCTrainer(ReplayEWCTrainer):
    """ReplayEWC plus logit consistency replay.

    Inspired by dark/strong experience replay: each buffer item stores the logits
    produced when it entered memory. During replay we train on both labels and a
    small MSE consistency loss against those stored logits.
    """
    name = "DarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.1,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
        )
        self.dark_alpha = dark_alpha
        self.replay_weight = replay_weight
        self.dark_confidence_threshold = float(np.clip(dark_confidence_threshold, 0.0, 1.0))
        self.dark_require_correct = bool(dark_require_correct)
        self.distill_start_task = max(0, int(distill_start_task))
        self.distill_ramp_tasks = max(0, int(distill_ramp_tasks))
        self._active_task_idx = None
        self.dark_alpha_trace = []
        self.buf_logits = []

    @staticmethod
    def _add_grads(a, b, scale=1.0):
        return {k: a[k] + scale * b[k] for k in a}

    def _reservoir_insert(self, X, Y, task_idx=None):
        logits = self.model.forward(X, task_idx)["logits"]
        for i in range(X.shape[0]):
            self.seen += 1
            item_x = X[i].copy()
            item_y = int(Y[i])
            item_logits = logits[i].copy()
            if len(self.buf_X) < self.capacity:
                self.buf_X.append(item_x)
                self.buf_Y.append(item_y)
                self.buf_task.append(task_idx)
                self.buf_logits.append(item_logits)
            else:
                j = self.rng.randint(0, self.seen)
                if j < self.capacity:
                    self.buf_X[j] = item_x
                    self.buf_Y[j] = item_y
                    self.buf_task[j] = task_idx
                    self.buf_logits[j] = item_logits

    def _sample_replay(self):
        if len(self.buf_X) == 0:
            return None
        idx = self.rng.randint(0, len(self.buf_X), size=min(self.replay_batch, len(self.buf_X)))
        rX = np.stack([self.buf_X[i] for i in idx])
        rY = np.array([self.buf_Y[i] for i in idx], dtype=np.int64)
        rtasks = [self.buf_task[i] for i in idx]
        rlogits = np.stack([self.buf_logits[i] for i in idx])
        return rX, rY, rtasks, rlogits

    @staticmethod
    def _target_probs(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _dark_sample_weights(self, cache, target_logits, labels=None):
        n = target_logits.shape[0]
        weights = np.ones(n, dtype=np.float64)
        if self.dark_confidence_threshold > 0.0 or self.dark_require_correct:
            target_probs = self._target_probs(target_logits)
            confidence = target_probs.max(axis=1)
            if self.dark_confidence_threshold > 0.0:
                denom = max(1e-12, 1.0 - self.dark_confidence_threshold)
                weights *= np.clip((confidence - self.dark_confidence_threshold) / denom, 0.0, 1.0)
            if self.dark_require_correct and labels is not None:
                weights *= (target_probs.argmax(axis=1) == labels).astype(np.float64)
        return weights

    def _dark_grads(self, cache, target_logits, task_idx, labels=None):
        n, out_dim = target_logits.shape
        dlogits = 2.0 * (cache["logits"] - target_logits) / max(1, n * out_dim)
        weights = self._dark_sample_weights(cache, target_logits, labels)
        if not np.allclose(weights, 1.0):
            dlogits = dlogits * weights[:, None]
        return self.model.backward_from_logits_grad(cache, dlogits, task_idx)

    def _effective_dark_alpha(self, current_grads, ce_grads, dark_grads):
        alpha = self.dark_alpha * self._distill_age_scale()
        self.dark_alpha_trace.append(float(alpha))
        if len(self.dark_alpha_trace) > 2000:
            self.dark_alpha_trace = self.dark_alpha_trace[-1000:]
        return alpha

    def _distill_age_scale(self):
        if self._active_task_idx is None:
            return 1.0
        task_idx = int(self._active_task_idx)
        if task_idx < self.distill_start_task:
            return 0.0
        if self.distill_ramp_tasks <= 0:
            return 1.0
        return float(np.clip((task_idx - self.distill_start_task + 1) / self.distill_ramp_tasks, 0.0, 1.0))

    def _mix_dark_replay_grads(self, grads, rX, rY, rtasks, rlogits, task_idx):
        replay_weight = self.replay_weight
        current_weight = 1.0 - replay_weight

        if self.model.multi_head:
            rgrads_accum = {k: np.zeros_like(v) for k, v in grads.items()}
            head_grads = {}

            for t in sorted(set(rtasks)):
                sub_idx = [i for i, val in enumerate(rtasks) if val == t]
                sub_X = rX[sub_idx]
                sub_Y = rY[sub_idx]
                sub_logits = rlogits[sub_idx]

                sub_cache = self.model.forward(sub_X, t)
                ce_grads = self.model.backward(sub_cache, sub_Y, t)
                dark_grads = self._dark_grads(sub_cache, sub_logits, t, sub_Y)
                alpha = self._effective_dark_alpha(grads, ce_grads, dark_grads)
                sub_grads = self._add_grads(ce_grads, dark_grads, alpha)

                weight = len(sub_idx) / len(rtasks)
                for k in ["W1", "b1", "W2", "b2"]:
                    rgrads_accum[k] += weight * sub_grads[k]

                head_grads[t] = {
                    "W3": weight * sub_grads["W3"],
                    "b3": weight * sub_grads["b3"],
                }

            for k in ["W1", "b1", "W2", "b2"]:
                grads[k] = current_weight * grads[k] + replay_weight * rgrads_accum[k]

            grads["W3"] = current_weight * grads["W3"]
            grads["b3"] = current_weight * grads["b3"]
            if task_idx in head_grads:
                grads["W3"] += replay_weight * head_grads[task_idx]["W3"]
                grads["b3"] += replay_weight * head_grads[task_idx]["b3"]

            for t, h_g in head_grads.items():
                if t != task_idx:
                    t_grads = {
                        "W1": np.zeros_like(self.model.W1),
                        "b1": np.zeros_like(self.model.b1),
                        "W2": np.zeros_like(self.model.W2),
                        "b2": np.zeros_like(self.model.b2),
                        "W3": h_g["W3"],
                        "b3": h_g["b3"],
                    }
                    self.model.sgd_step(t_grads, self.lr * replay_weight, task_idx=t)
            return grads

        if self.model.input_adapter:
            shared_keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
            rgrads_accum = {k: np.zeros_like(grads[k]) for k in shared_keys}
            adapter_grads = {}
            for t in sorted(set(rtasks)):
                sub_idx = [i for i, val in enumerate(rtasks) if val == t]
                sub_cache = self.model.forward(rX[sub_idx], t)
                ce_grads = self.model.backward(sub_cache, rY[sub_idx], t)
                dark_grads = self._dark_grads(sub_cache, rlogits[sub_idx], t, rY[sub_idx])
                alpha = self._effective_dark_alpha(grads, ce_grads, dark_grads)
                sub_grads = self._add_grads(ce_grads, dark_grads, alpha)
                weight = len(sub_idx) / len(rtasks)
                for k in shared_keys:
                    rgrads_accum[k] += weight * sub_grads[k]
                if "A" in sub_grads:
                    adapter_grads[t] = weight * sub_grads["A"]

            mixed = {k: current_weight * grads[k] + replay_weight * rgrads_accum[k] for k in shared_keys}
            if "A" in grads:
                mixed["A"] = current_weight * grads["A"]
                if task_idx in adapter_grads:
                    mixed["A"] += replay_weight * adapter_grads[task_idx]
            for t, gA in adapter_grads.items():
                if t != task_idx:
                    self.model.adapters[t] -= self.lr * replay_weight * gA
            return mixed

        rcache = self.model.forward(rX)
        ce_grads = self.model.backward(rcache, rY)
        dark_grads = self._dark_grads(rcache, rlogits, None, rY)
        alpha = self._effective_dark_alpha(grads, ce_grads, dark_grads)
        rgrads = self._add_grads(ce_grads, dark_grads, alpha)
        return {k: current_weight * grads[k] + replay_weight * rgrads[k] for k in grads}

    def train_step(self, X, Y, task_idx: int = None):
        self._active_task_idx = task_idx
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        sample = self._sample_replay()
        if sample is not None:
            grads = self._mix_dark_replay_grads(grads, *sample, task_idx)

        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc


def _online_consolidate_fisher(self, task_idx):
    """Task-free (boundary-agnostic) EWC consolidation.

    Instead of waiting for ``on_task_end`` to estimate the Fisher diagonal on the
    just-finished task's data, estimate it from a random sample of the reservoir
    buffer (a boundary-free mixture of everything seen so far) and snapshot the
    anchor. Called on a fixed step interval, never aligned to task boundaries, so
    the consolidation never uses any task-boundary knowledge. The squared-gradient
    accumulation and ``fisher_decay`` EMA mirror the boundary-aware ``on_task_end``
    so the same ``lam`` transfers; only the *timing* and *data source* differ.
    """
    if len(self.buf_X) == 0:
        return
    n = min(self.fisher_sample, len(self.buf_X))
    idx = self.rng.randint(0, len(self.buf_X), size=n)
    sX = np.stack([self.buf_X[i] for i in idx])
    sY = np.array([self.buf_Y[i] for i in idx], dtype=np.int64)
    stasks = [self.buf_task[i] for i in idx]
    accum = {k: np.zeros_like(self.fisher[k]) for k in self.reg_keys}
    if self.model.multi_head or self.model.input_adapter:
        # Route each buffered sample through its own head/adapter (task id is an
        # architectural routing key here, not boundary-timing knowledge).
        groups = sorted(set(stasks), key=lambda v: (v is None, v))
        for t in groups:
            sub = [i for i, v in enumerate(stasks) if v == t]
            cache = self.model.forward(sX[sub], t)
            grads = self.model.backward(cache, sY[sub], t)
            for k in self.reg_keys:
                accum[k] += grads[k] ** 2
        denom = max(1, len(groups))
        for k in self.reg_keys:
            accum[k] /= denom
    else:
        cache = self.model.forward(sX)
        grads = self.model.backward(cache, sY)
        for k in self.reg_keys:
            accum[k] = grads[k] ** 2
    for k in self.reg_keys:
        self.fisher[k] = self.fisher_decay * self.fisher[k] + accum[k]
    self.anchor = {k: v.copy() for k, v in self._current_reg_params(task_idx).items()}


class OnlineEWCReplayTrainer(ReplayEWCTrainer):
    """Task-free ReplayEWC: online Fisher/anchor, no ``on_task_end`` boundary use.

    Reservoir replay is already boundary-agnostic; the only piece of ReplayEWC
    that needs task boundaries is the Fisher/anchor consolidation in
    ``on_task_end``. Here that is replaced by ``_online_consolidate_fisher`` fired
    every ``consolidate_every`` steps, so the trainer never relies on knowing when
    a task switches. Used to measure the cost of losing boundary knowledge.
    """
    name = "OnlineEWCReplay"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, consolidate_every: int = 400,
                 fisher_sample: int = 256):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm)
        self.consolidate_every = max(1, int(consolidate_every))
        self.fisher_sample = int(fisher_sample)
        self._online_step = 0

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = super().train_step(X, Y, task_idx)
        self._online_step += 1
        if self._online_step % self.consolidate_every == 0:
            _online_consolidate_fisher(self, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        pass  # task-free: consolidation is driven by step interval, not boundaries


class OnlineDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """Task-free DarkReplayEWC (DER++): online Fisher/anchor, no boundary use.

    Same boundary-free consolidation as ``OnlineEWCReplay`` but keeping DER++ logit
    distillation. Note the DER++ logit targets are captured at insertion time
    (already boundary-free), so this is a fully task-free anti-forgetting trainer.
    """
    name = "OnlineDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0, consolidate_every: int = 400,
                 fisher_sample: int = 256):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm,
                         dark_alpha=dark_alpha, replay_weight=replay_weight,
                         dark_confidence_threshold=dark_confidence_threshold,
                         dark_require_correct=dark_require_correct,
                         distill_start_task=distill_start_task,
                         distill_ramp_tasks=distill_ramp_tasks)
        self.consolidate_every = max(1, int(consolidate_every))
        self.fisher_sample = int(fisher_sample)
        self._online_step = 0

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = super().train_step(X, Y, task_idx)
        self._online_step += 1
        if self._online_step % self.consolidate_every == 0:
            _online_consolidate_fisher(self, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        pass  # task-free: consolidation is driven by step interval, not boundaries


class GenerativeReplayEWCTrainer(ReplayEWCTrainer):
    """Buffer-free replay: no raw samples are ever stored.

    The inputs are one-hot digit windows (``K`` positions x 10 digit values).
    Instead of a reservoir of raw samples, this keeps, per (task, class), a
    factorized categorical generative model — the per-position digit frequencies
    accumulated as sufficient statistics from the data stream as it passes. Replay
    draws *synthetic* one-hot windows from that model and routes them through the
    matching head, so the shared layers and old heads keep being rehearsed without
    retaining any raw data. Storage is bounded by ``n_tasks * n_classes * in_dim``
    and, crucially, does not grow with stream length.

    The EWC Fisher/anchor consolidation still runs at ``on_task_end`` on the
    *current* task's transient data (no old data retained), so this is the
    generative analogue of ``ReplayEWC`` and is compared head-to-head against it.
    """
    name = "GenerativeReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, gen_smoothing: float = 0.1,
                 gen_sum_match: bool = True, gen_sum_tol: float = 1.0):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm)
        self.in_dim = int(self.model.dims[0])
        self.alphabet = 10
        self.K = self.in_dim // self.alphabet
        self.gen_smoothing = float(gen_smoothing)
        # Conditional generation: keep synthetic windows whose digit-sum matches the
        # class's observed sum band (the sum is what defines the label here), within
        # gen_sum_tol * std. Without this, independent per-position sampling produces
        # windows whose sum falls in the wrong bucket -> synthetic label noise.
        self.gen_sum_match = bool(gen_sum_match)
        self.gen_sum_tol = float(gen_sum_tol)
        # digit value of each one-hot dim: [0..9, 0..9, ...] so window_sum = X @ value_vec
        self.value_vec = np.tile(np.arange(self.alphabet), self.K).astype(np.float64)
        self.gen_sum = {}      # (task, class) -> (in_dim,) running one-hot counts
        self.gen_count = {}    # (task, class) -> int
        self.gen_sval = {}     # (task, class) -> running sum of window-sums
        self.gen_sval_sq = {}  # (task, class) -> running sum of window-sums^2
        self.gen_keys = []     # ordered list of seen (task, class) keys

    def _reservoir_insert(self, X, Y, task_idx=None):
        # Update per-class sufficient statistics instead of storing raw samples.
        Y = np.asarray(Y).astype(np.int64)
        svals = X @ self.value_vec
        for c in np.unique(Y):
            key = (task_idx, int(c))
            mask = (Y == c)
            if key not in self.gen_sum:
                self.gen_sum[key] = np.zeros(self.in_dim, dtype=np.float64)
                self.gen_count[key] = 0
                self.gen_sval[key] = 0.0
                self.gen_sval_sq[key] = 0.0
                self.gen_keys.append(key)
            self.gen_sum[key] += X[mask].sum(axis=0)
            self.gen_count[key] += int(mask.sum())
            self.gen_sval[key] += float(svals[mask].sum())
            self.gen_sval_sq[key] += float((svals[mask] ** 2).sum())

    def _sample_digits(self, probs, m):
        out = np.zeros((m, self.in_dim), dtype=np.float64)
        rows = np.arange(m)
        digit_sum = np.zeros(m, dtype=np.float64)
        for p in range(self.K):
            digits = self.rng.choice(self.alphabet, size=m, p=probs[p])
            out[rows, p * self.alphabet + digits] = 1.0
            digit_sum += digits
        return out, digit_sum

    def _generate(self, key, m):
        # Sample m synthetic one-hot windows from the (task, class) categorical model,
        # optionally keeping only those whose digit-sum matches the class sum band.
        counts = self.gen_sum[key].reshape(self.K, self.alphabet) + self.gen_smoothing
        probs = counts / counts.sum(axis=1, keepdims=True)
        if not self.gen_sum_match or self.gen_count[key] < 2:
            return self._sample_digits(probs, m)[0]

        n = self.gen_count[key]
        mean = self.gen_sval[key] / n
        var = max(0.0, self.gen_sval_sq[key] / n - mean * mean)
        band = self.gen_sum_tol * np.sqrt(var)
        kept = []
        for _ in range(8):  # bounded rejection sampling
            cand, csum = self._sample_digits(probs, max(m, 4 * m))
            ok = np.abs(csum - mean) <= band
            if ok.any():
                kept.append(cand[ok])
                if sum(len(a) for a in kept) >= m:
                    break
        if not kept:
            return self._sample_digits(probs, m)[0]  # fall back if band too tight
        pool = np.concatenate(kept, axis=0)
        if pool.shape[0] < m:  # top up with unconditional samples
            pool = np.concatenate([pool, self._sample_digits(probs, m - pool.shape[0])[0]], axis=0)
        return pool[:m]

    def _sample_replay(self):
        if not self.gen_keys:
            return None
        counts = np.array([self.gen_count[k] for k in self.gen_keys], dtype=np.float64)
        probs = counts / counts.sum()
        idx = self.rng.choice(len(self.gen_keys), size=self.replay_batch, p=probs)
        rX_list, rY, rtasks = [], [], []
        for ki in np.unique(idx):
            key = self.gen_keys[int(ki)]
            m = int(np.sum(idx == ki))
            rX_list.append(self._generate(key, m))
            rY.extend([key[1]] * m)
            rtasks.extend([key[0]] * m)
        rX = np.concatenate(rX_list, axis=0)
        rY = np.array(rY, dtype=np.int64)
        return rX, rY, rtasks


class NBGenerativeReplayEWCTrainer(GenerativeReplayEWCTrainer):
    """Rule-agnostic buffer-free generative replay (P5).

    P4's ``GenerativeReplayEWC`` conditions synthetic windows on the total
    digit-sum — the right label-defining statistic only for the default ``sum``
    rule. Under ``conflicting`` mode each task uses a different rule
    (weighted/half-window/adjacent-product), so matching the total sum is the
    wrong statistic and fidelity drops.

    Here we instead accept a synthetic window only if a naive-Bayes classifier
    built from the *stored per-class categoricals of the same task* assigns it to
    the target class. Because NB weighs each position by how class-discriminative
    its stored marginal is, rejection automatically concentrates on whichever
    positions actually define this task's rule — with no model in the loop (so no
    staleness/confirmation bias). One generator works rule-agnostically for both
    ``label_permuted`` and ``conflicting``.
    """
    name = "NBGenerativeReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, gen_smoothing: float = 0.1,
                 gen_sum_tol: float = 1.0, nb_margin: float = 0.0):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm,
                         gen_smoothing=gen_smoothing, gen_sum_match=False,
                         gen_sum_tol=gen_sum_tol)
        self.nb_margin = float(nb_margin)

    def _task_logtables(self, task):
        # log P(digit | task, class) per position, for every class seen in this task.
        tables = {}
        for (tt, c) in self.gen_keys:
            if tt == task:
                counts = self.gen_sum[(tt, c)].reshape(self.K, self.alphabet) + self.gen_smoothing
                tables[c] = np.log(counts / counts.sum(axis=1, keepdims=True))
        return tables

    def _generate(self, key, m):
        task, target = key
        counts = self.gen_sum[key].reshape(self.K, self.alphabet) + self.gen_smoothing
        probs = counts / counts.sum(axis=1, keepdims=True)
        tables = self._task_logtables(task)
        if len(tables) < 2:
            return self._sample_digits(probs, m)[0]
        classes = sorted(tables.keys())
        logt = np.stack([tables[c] for c in classes])  # (C, K, alphabet)
        tgt_i = classes.index(target)
        pos = np.arange(self.K)
        kept = []
        for _ in range(8):  # bounded rejection sampling
            cand, _ = self._sample_digits(probs, max(m, 4 * m))
            digits = cand.reshape(-1, self.K, self.alphabet).argmax(axis=2)  # (n, K)
            # NB log-scores per class: sum_p logt[c, p, digit_p]
            scores = np.stack([logt[ci][pos, digits].sum(axis=1)
                               for ci in range(len(classes))], axis=1)  # (n, C)
            tgt_score = scores[:, tgt_i].copy()
            scores[:, tgt_i] = -np.inf
            ok = (tgt_score - scores.max(axis=1)) >= self.nb_margin
            if ok.any():
                kept.append(cand[ok])
                if sum(len(a) for a in kept) >= m:
                    break
        if not kept:
            return self._sample_digits(probs, m)[0]  # fall back if accept region empty
        pool = np.concatenate(kept, axis=0)
        if pool.shape[0] < m:
            pool = np.concatenate([pool, self._sample_digits(probs, m - pool.shape[0])[0]], axis=0)
        return pool[:m]


class ScholarGenerativeReplayEWCTrainer(GenerativeReplayEWCTrainer):
    """Rule-agnostic buffer-free generative replay via a scholar/teacher (P5).

    Deep-generative-replay style. At each task boundary we snapshot the model as a
    frozen *scholar*. During later tasks we draw synthetic inputs from the
    per-class categorical generator (unconditional — the realized hard label may be
    wrong) and train the current model to match the *scholar's soft logits* on
    those inputs (generative DER++/distillation). The scholar encodes every past
    task's actual input->class rule, including nonlinear interaction rules, so it
    labels synthetic inputs correctly where a hand-picked statistic (P4 sum-match)
    or a factorized naive-Bayes classifier (NBGenerativeReplayEWC) cannot. Cost:
    one constant-size model snapshot; still no raw samples retained.
    """
    name = "ScholarGenerativeReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, gen_smoothing: float = 0.1,
                 dark_alpha: float = 0.5, replay_weight: float = 0.5):
        # The scholar provides labels, so the generator need not condition on a
        # hand-picked statistic — generate unconditionally (gen_sum_match=False).
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm,
                         gen_smoothing=gen_smoothing, gen_sum_match=False)
        self.dark_alpha = float(dark_alpha)
        self.replay_weight = float(replay_weight)
        self.scholar = None
        self.scholar_max_task = -1  # highest task index the scholar has learned

    def _distill_grads(self, cache, teacher_logits, task_idx):
        n, out_dim = teacher_logits.shape
        dlogits = 2.0 * (cache["logits"] - teacher_logits) / max(1, n * out_dim)
        return self.model.backward_from_logits_grad(cache, dlogits, task_idx)

    def _mix_scholar_grads(self, grads, rX, rtasks, task_idx):
        w = self.replay_weight
        cw = 1.0 - w
        if self.model.multi_head:
            shared = ["W1", "b1", "W2", "b2"]
            accum = {k: np.zeros_like(grads[k]) for k in shared}
            for t in sorted(set(rtasks)):
                idx = [i for i, v in enumerate(rtasks) if v == t]
                sx = rX[idx]
                tlog = self.scholar.forward(sx, t)["logits"]
                cache = self.model.forward(sx, t)
                g = self._distill_grads(cache, tlog, t)
                weight = (len(idx) / len(rtasks)) * self.dark_alpha
                for k in shared:
                    accum[k] += weight * g[k]
                # apply the replayed task's own head update directly
                t_grads = {
                    "W1": np.zeros_like(self.model.W1), "b1": np.zeros_like(self.model.b1),
                    "W2": np.zeros_like(self.model.W2), "b2": np.zeros_like(self.model.b2),
                    "W3": weight * g["W3"], "b3": weight * g["b3"],
                }
                self.model.sgd_step(t_grads, self.lr * w, task_idx=t)
            for k in shared:
                grads[k] = cw * grads[k] + w * accum[k]
            return grads
        # single head (no adapter routing for synthetic inputs): distill globally
        tlog = self.scholar.forward(rX)["logits"]
        cache = self.model.forward(rX)
        g = self._distill_grads(cache, tlog, None)
        return {k: cw * grads[k] + w * self.dark_alpha * g[k] for k in grads}

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        if self.scholar is not None:
            sample = self._sample_replay()
            if sample is not None:
                rX, _, rtasks = sample
                keep = [i for i, t in enumerate(rtasks)
                        if t is not None and t <= self.scholar_max_task and t != task_idx]
                if keep:
                    grads = self._mix_scholar_grads(grads, rX[keep],
                                                    [rtasks[i] for i in keep], task_idx)

        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        super().on_task_end(task_X, task_Y, task_idx)  # EWC Fisher/anchor on current data
        self.scholar = copy.deepcopy(self.model)
        if task_idx is not None:
            self.scholar_max_task = task_idx


class ScholarGlobalGenerativeReplayEWCTrainer(ScholarGenerativeReplayEWCTrainer):
    """P6: scholar generative replay with an on-manifold input generator.

    P5's scholar sampled synthetic inputs from each class's per-position marginals
    — a skewed, atypical distribution that is *off* the data manifold, so the
    teacher's logits there are unreliable (capping plasticity). But pi's digits are
    ~iid uniform, so the true input distribution is genuinely factorized: sampling
    each position from the *global* (class-agnostic) per-task marginal reproduces
    the real digit-window distribution (on-manifold), and the scholar supplies the
    label. This decouples input generation (now correct) from labeling (the
    teacher), so it is fully rule-agnostic and stays close to the true manifold.
    """
    name = "ScholarGlobalGenerativeReplayEWC"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_global = {}    # task -> (in_dim,) class-agnostic one-hot counts
        self.gen_global_n = {}  # task -> total count

    def _reservoir_insert(self, X, Y, task_idx=None):
        super()._reservoir_insert(X, Y, task_idx)  # keep per-class stats too (unused here)
        if task_idx not in self.gen_global:
            self.gen_global[task_idx] = np.zeros(self.in_dim, dtype=np.float64)
            self.gen_global_n[task_idx] = 0
        self.gen_global[task_idx] += X.sum(axis=0)
        self.gen_global_n[task_idx] += X.shape[0]

    def _generate_global(self, task, m):
        counts = self.gen_global[task].reshape(self.K, self.alphabet) + self.gen_smoothing
        probs = counts / counts.sum(axis=1, keepdims=True)
        return self._sample_digits(probs, m)[0]

    def _sample_replay(self):
        tasks = [t for t in self.gen_global if self.gen_global_n[t] > 0]
        if not tasks:
            return None
        counts = np.array([self.gen_global_n[t] for t in tasks], dtype=np.float64)
        probs = counts / counts.sum()
        idx = self.rng.choice(len(tasks), size=self.replay_batch, p=probs)
        rX_list, rtasks = [], []
        for ti in np.unique(idx):
            t = tasks[int(ti)]
            m = int(np.sum(idx == ti))
            rX_list.append(self._generate_global(t, m))
            rtasks.extend([t] * m)
        rX = np.concatenate(rX_list, axis=0)
        rY = np.zeros(len(rtasks), dtype=np.int64)  # unused; the scholar provides labels
        return rX, rY, rtasks


class AdaptiveDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """DarkReplayEWC with gradient-conflict gated logit distillation.

    Fixed DER++ can over-anchor old functions when the incoming task truly needs
    different shared features. This variant keeps ReplayEWC as the backbone, but
    lowers the dark-logit weight when the replay distillation gradient opposes
    either the current-task gradient or the replay label gradient on shared layers.
    """
    name = "AdaptiveDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_alpha_min: float = 0.0,
                 conflict_margin: float = 0.2, alpha_smoothing: float = 0.2,
                 conflict_ema_decay: float = 0.95, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
        )
        self.dark_alpha_max = float(dark_alpha)
        self.dark_alpha_min = float(dark_alpha_min)
        self.conflict_margin = max(1e-6, float(conflict_margin))
        self.alpha_smoothing = float(np.clip(alpha_smoothing, 0.0, 1.0))
        self.conflict_ema_decay = float(np.clip(conflict_ema_decay, 0.0, 0.999))
        self.conflict_ema = None
        self.adaptive_dark_alpha = None
        self.alpha_trace = []
        self.conflict_trace = []
        self.conflict_ema_trace = []
        self.current_replay_ce_cos_trace = []
        self.current_dark_cos_trace = []
        self.ce_dark_cos_trace = []

    @staticmethod
    def _shared_cosine(a, b):
        keys = [k for k in ("W1", "b1", "W2", "b2") if k in a and k in b]
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for k in keys:
            av = a[k]
            bv = b[k]
            dot += float(np.sum(av * bv))
            norm_a += float(np.sum(av * av))
            norm_b += float(np.sum(bv * bv))
        denom = np.sqrt(norm_a) * np.sqrt(norm_b)
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(dot / denom, -1.0, 1.0))

    def _alpha_from_conflict(self, conflict_score):
        if conflict_score <= 0.0:
            scale = 0.0
        elif conflict_score >= self.conflict_margin:
            scale = 1.0
        else:
            scale = conflict_score / self.conflict_margin
        return self.dark_alpha_min + (self.dark_alpha_max - self.dark_alpha_min) * scale

    def _effective_dark_alpha(self, current_grads, ce_grads, dark_grads):
        current_replay_ce_cos = self._shared_cosine(current_grads, ce_grads)
        current_dark_cos = self._shared_cosine(current_grads, dark_grads)
        ce_dark_cos = self._shared_cosine(ce_grads, dark_grads)
        conflict_score = min(current_dark_cos, ce_dark_cos)
        if self.conflict_ema is None:
            self.conflict_ema = conflict_score
        else:
            d = self.conflict_ema_decay
            self.conflict_ema = d * self.conflict_ema + (1.0 - d) * conflict_score
        target_alpha = self._alpha_from_conflict(self.conflict_ema)

        if self.adaptive_dark_alpha is None:
            alpha = target_alpha
        else:
            s = self.alpha_smoothing
            alpha = (1.0 - s) * self.adaptive_dark_alpha + s * target_alpha
        alpha *= self._distill_age_scale()
        self.adaptive_dark_alpha = float(alpha)

        self.alpha_trace.append(float(alpha))
        self.conflict_trace.append(float(conflict_score))
        self.conflict_ema_trace.append(float(self.conflict_ema))
        self.current_replay_ce_cos_trace.append(float(current_replay_ce_cos))
        self.current_dark_cos_trace.append(float(current_dark_cos))
        self.ce_dark_cos_trace.append(float(ce_dark_cos))
        if len(self.alpha_trace) > 2000:
            self.alpha_trace = self.alpha_trace[-1000:]
            self.conflict_trace = self.conflict_trace[-1000:]
            self.conflict_ema_trace = self.conflict_ema_trace[-1000:]
            self.current_replay_ce_cos_trace = self.current_replay_ce_cos_trace[-1000:]
            self.current_dark_cos_trace = self.current_dark_cos_trace[-1000:]
            self.ce_dark_cos_trace = self.ce_dark_cos_trace[-1000:]
        return float(alpha)


class PressureDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """DarkReplayEWC with reliability × forgetting-pressure gating.

    Reliability asks whether the stored logits were worth consolidating when they
    entered memory. Pressure asks whether the replayed old sample is currently
    being forgotten. By default this is label-loss dominant; logit drift is kept
    as an opt-in signal because geometric drift can be harmless plasticity.
    """
    name = "PressureDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.4,
                 dark_require_correct: bool = True, pressure_loss_low: float = 0.8,
                 pressure_loss_high: float = 2.3, pressure_drift_low: float = 999.0,
                 pressure_drift_high: float = 1000.0, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
        )
        self.pressure_loss_low = float(pressure_loss_low)
        self.pressure_loss_high = max(self.pressure_loss_low + 1e-6, float(pressure_loss_high))
        self.pressure_drift_low = float(pressure_drift_low)
        self.pressure_drift_high = max(self.pressure_drift_low + 1e-6, float(pressure_drift_high))
        self.pressure_weight_trace = []
        self.reliability_trace = []
        self.loss_pressure_trace = []
        self.drift_pressure_trace = []

    @staticmethod
    def _ramp(values, low, high):
        return np.clip((values - low) / max(1e-12, high - low), 0.0, 1.0)

    def _forgetting_pressure(self, cache, target_logits, labels=None):
        n = target_logits.shape[0]
        loss_pressure = np.zeros(n, dtype=np.float64)
        if labels is not None:
            probs = cache["probs"]
            losses = -np.log(probs[np.arange(n), labels] + 1e-12)
            loss_pressure = self._ramp(losses, self.pressure_loss_low, self.pressure_loss_high)

        drift = np.sqrt(np.mean((cache["logits"] - target_logits) ** 2, axis=1))
        drift_pressure = self._ramp(drift, self.pressure_drift_low, self.pressure_drift_high)
        return np.maximum(loss_pressure, drift_pressure), loss_pressure, drift_pressure

    def _dark_sample_weights(self, cache, target_logits, labels=None):
        reliability = super()._dark_sample_weights(cache, target_logits, labels)
        pressure, loss_pressure, drift_pressure = self._forgetting_pressure(cache, target_logits, labels)
        weights = reliability * pressure

        self.pressure_weight_trace.append(float(np.mean(weights)))
        self.reliability_trace.append(float(np.mean(reliability)))
        self.loss_pressure_trace.append(float(np.mean(loss_pressure)))
        self.drift_pressure_trace.append(float(np.mean(drift_pressure)))
        if len(self.pressure_weight_trace) > 2000:
            self.pressure_weight_trace = self.pressure_weight_trace[-1000:]
            self.reliability_trace = self.reliability_trace[-1000:]
            self.loss_pressure_trace = self.loss_pressure_trace[-1000:]
            self.drift_pressure_trace = self.drift_pressure_trace[-1000:]
        return weights


class LookaheadDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """DarkReplayEWC with Counterfactual Lookahead Probing for regime detection.

    Before applying the update, we simulate the current task's step. If it
    hurts replay loss (conflicting tasks), we scale down dark alpha to preserve plasticity.
    If it is neutral or helps (shared rule / label_permuted), we keep dark alpha high.
    """
    name = "LookaheadDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0, lookahead_beta: float = 500.0,
                 lookahead_ema_decay: float = 0.9):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
        )
        self.lookahead_beta = float(lookahead_beta)
        self.lookahead_alpha = float(dark_alpha)
        self.lookahead_ema_decay = float(lookahead_ema_decay)
        self.delta_L_ema = None
        self.alpha_trace = []
        self.delta_L_trace = []

    def _compute_replay_loss(self, rX, rY, rtasks):
        if self.model.multi_head or self.model.input_adapter:
            total_loss = 0.0
            for t in sorted(set(rtasks)):
                sub_idx = [i for i, val in enumerate(rtasks) if val == t]
                sub_X = rX[sub_idx]
                sub_Y = rY[sub_idx]
                loss, _ = self.model.loss_acc(sub_X, sub_Y, t)
                total_loss += loss * len(sub_idx)
            return total_loss / len(rtasks)
        else:
            loss, _ = self.model.loss_acc(rX, rY)
            return loss

    def _lookahead_conflict_check(self, current_grads, rX, rY, rtasks, current_task_idx):
        if current_task_idx is None or current_task_idx <= 0:
            return 0.0

        # Only measure conflict on PAST tasks' samples
        past_idx = [i for i, t in enumerate(rtasks) if t is not None and t < current_task_idx]
        if len(past_idx) == 0:
            # If no past samples, assume current EMA holds or is neutral
            return self.delta_L_ema if self.delta_L_ema is not None else 0.0

        past_X = rX[past_idx]
        past_Y = rY[past_idx]
        past_tasks = [rtasks[i] for i in past_idx]

        L_old = self._compute_replay_loss(past_X, past_Y, past_tasks)

        # Save current parameters
        W1_bak = self.model.W1.copy()
        b1_bak = self.model.b1.copy()
        W2_bak = self.model.W2.copy()
        b2_bak = self.model.b2.copy()

        if self.model.multi_head:
            W3_bak = self.model.heads_W[current_task_idx].copy()
            b3_bak = self.model.heads_b[current_task_idx].copy()
        else:
            W3_bak = self.model.W3.copy()
            b3_bak = self.model.b3.copy()

        if self.model.input_adapter and current_task_idx is not None:
            A_bak = self.model.adapters[current_task_idx].copy()

        # Apply virtual step
        self.model.W1 -= self.lr * current_grads["W1"]
        self.model.b1 -= self.lr * current_grads["b1"]
        self.model.W2 -= self.lr * current_grads["W2"]
        self.model.b2 -= self.lr * current_grads["b2"]

        if self.model.multi_head:
            self.model.heads_W[current_task_idx] -= self.lr * current_grads["W3"]
            self.model.heads_b[current_task_idx] -= self.lr * current_grads["b3"]
        else:
            self.model.W3 -= self.lr * current_grads["W3"]
            self.model.b3 -= self.lr * current_grads["b3"]

        if self.model.input_adapter and current_task_idx is not None and "A" in current_grads:
            self.model.adapters[current_task_idx] -= self.lr * current_grads["A"]

        # Compute loss after update
        L_new = self._compute_replay_loss(past_X, past_Y, past_tasks)

        # Restore parameters
        self.model.W1 = W1_bak
        self.model.b1 = b1_bak
        self.model.W2 = W2_bak
        self.model.b2 = b2_bak

        if self.model.multi_head:
            self.model.heads_W[current_task_idx] = W3_bak
            self.model.heads_b[current_task_idx] = b3_bak
        else:
            self.model.W3 = W3_bak
            self.model.b3 = b3_bak

        if self.model.input_adapter and current_task_idx is not None:
            self.model.adapters[current_task_idx] = A_bak

        delta_L = L_new - L_old

        # Maintain Exponential Moving Average
        if self.delta_L_ema is None:
            self.delta_L_ema = delta_L
        else:
            d = self.lookahead_ema_decay
            self.delta_L_ema = d * self.delta_L_ema + (1.0 - d) * delta_L

        return self.delta_L_ema

    def _effective_dark_alpha(self, current_grads, ce_grads, dark_grads):
        return self.lookahead_alpha

    def train_step(self, X, Y, task_idx: int = None):
        self._active_task_idx = task_idx
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        sample = self._sample_replay()
        if sample is not None:
            rX, rY, rtasks, rlogits = sample
            # Counterfactual lookahead probing
            delta_L_ema = self._lookahead_conflict_check(grads, rX, rY, rtasks, task_idx)
            scale = np.exp(-self.lookahead_beta * max(0.0, delta_L_ema))
            self.lookahead_alpha = float(self.dark_alpha * scale * self._distill_age_scale())
            
            self.alpha_trace.append(self.lookahead_alpha)
            self.delta_L_trace.append(float(delta_L_ema))
            if len(self.alpha_trace) > 2000:
                self.alpha_trace = self.alpha_trace[-1000:]
                self.delta_L_trace = self.delta_L_trace[-1000:]

            # Mix gradients using self.lookahead_alpha
            grads = self._mix_dark_replay_grads(grads, *sample, task_idx)

        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc


class RtpDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """DarkReplayEWC with Representational Transfer Probe (RTP) using gradient-cosine regime detection.

    We accumulate shared-layer gradients over the first N steps of a new task,
    and compare it to the average gradient of past tasks sampled from the buffer.
    If the cosine similarity is >= 0, we classify as synergistic and activate DER++ distillation.
    If negative, we classify as conflicting and disable distillation (alpha = 0.0).
    """
    name = "RtpDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0, rtp_probe_steps: int = 5,
                 rtp_cos_threshold: float = 0.0):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
        )
        self.rtp_probe_steps = int(rtp_probe_steps)
        self.rtp_cos_threshold = float(rtp_cos_threshold)
        self._last_task_idx = None
        self.rtp_step_counter = 0
        self.rtp_accum_grads = None
        self.rtp_active_alpha = 0.0
        self.rtp_regime = "unknown"
        self.alpha_trace = []
        self.regime_trace = []

    def _compute_past_tasks_avg_gradient(self, current_task_idx):
        if len(self.buf_X) == 0 or current_task_idx is None or current_task_idx <= 0:
            return {
                "W1": np.zeros_like(self.model.W1),
                "b1": np.zeros_like(self.model.b1),
                "W2": np.zeros_like(self.model.W2),
                "b2": np.zeros_like(self.model.b2),
            }
        
        past_idx = [i for i, t in enumerate(self.buf_task) if t is not None and t < current_task_idx]
        if len(past_idx) == 0:
            return {
                "W1": np.zeros_like(self.model.W1),
                "b1": np.zeros_like(self.model.b1),
                "W2": np.zeros_like(self.model.W2),
                "b2": np.zeros_like(self.model.b2),
            }
        
        sample_idx = self.rng.choice(past_idx, size=min(100, len(past_idx)), replace=False)
        sX = np.stack([self.buf_X[i] for i in sample_idx])
        sY = np.array([self.buf_Y[i] for i in sample_idx], dtype=np.int64)
        stasks = [self.buf_task[i] for i in sample_idx]
        
        accum_grads = {
            "W1": np.zeros_like(self.model.W1),
            "b1": np.zeros_like(self.model.b1),
            "W2": np.zeros_like(self.model.W2),
            "b2": np.zeros_like(self.model.b2),
        }
        
        if self.model.multi_head or self.model.input_adapter:
            for t in sorted(set(stasks)):
                sub_idx = [i for i, val in enumerate(stasks) if val == t]
                sub_X = sX[sub_idx]
                sub_Y = sY[sub_idx]
                cache = self.model.forward(sub_X, t)
                grads = self.model.backward(cache, sub_Y, t)
                weight = len(sub_idx) / len(stasks)
                for k in ["W1", "b1", "W2", "b2"]:
                    accum_grads[k] += weight * grads[k]
        else:
            cache = self.model.forward(sX)
            grads = self.model.backward(cache, sY)
            for k in ["W1", "b1", "W2", "b2"]:
                accum_grads[k] = grads[k]
                
        return accum_grads

    def _effective_dark_alpha(self, current_grads, ce_grads, dark_grads):
        return self.rtp_active_alpha

    def train_step(self, X, Y, task_idx: int = None):
        self._active_task_idx = task_idx
        
        # Detect new task transition
        if task_idx != self._last_task_idx:
            self._last_task_idx = task_idx
            self.rtp_step_counter = 0
            self.rtp_accum_grads = {
                "W1": np.zeros_like(self.model.W1),
                "b1": np.zeros_like(self.model.b1),
                "W2": np.zeros_like(self.model.W2),
                "b2": np.zeros_like(self.model.b2),
            }
            self.rtp_active_alpha = 0.0
            self.rtp_regime = "probing" if (task_idx is not None and task_idx > 0) else "task_0"
            if task_idx == 0:
                self.rtp_active_alpha = float(self.dark_alpha)
            
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        # Probing stage logic
        if task_idx is not None and task_idx > 0 and self.rtp_regime == "probing":
            for k in ["W1", "b1", "W2", "b2"]:
                self.rtp_accum_grads[k] += grads[k]
            self.rtp_step_counter += 1
            if self.rtp_step_counter == self.rtp_probe_steps:
                # Compute past task average gradient
                g_past = self._compute_past_tasks_avg_gradient(task_idx)
                v_curr = np.concatenate([self.rtp_accum_grads[k].ravel() for k in ["W1", "b1", "W2", "b2"]])
                v_past = np.concatenate([g_past[k].ravel() for k in ["W1", "b1", "W2", "b2"]])
                cos = np.dot(v_curr, v_past) / (np.linalg.norm(v_curr) * np.linalg.norm(v_past) + 1e-8)
                
                if cos >= self.rtp_cos_threshold:
                    self.rtp_active_alpha = float(self.dark_alpha)
                    self.rtp_regime = "synergistic"
                else:
                    self.rtp_active_alpha = 0.0
                    self.rtp_regime = "conflicting"
                self.rtp_accum_grads = None

        self.alpha_trace.append(self.rtp_active_alpha)
        self.regime_trace.append(self.rtp_regime)

        sample = self._sample_replay()
        if sample is not None:
            # Mix gradients using self.rtp_active_alpha (via _effective_dark_alpha)
            grads = self._mix_dark_replay_grads(grads, *sample, task_idx)

        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc


class HorizonDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """Oracle horizon-gated DER++ baseline for P2.7.

    If the known/estimated stream horizon is long enough, proactive DER++
    consolidation is enabled from the start; otherwise it is disabled and the
    trainer collapses to ReplayEWC. This is not a complete online detector. It is
    an oracle validation of whether a high-level horizon signal can choose
    between the two regimes better than local gradient/drift signals.
    """
    name = "HorizonDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0, horizon_threshold: int = 80,
                 horizon_override: int = None):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
        )
        self.horizon_threshold = max(1, int(horizon_threshold))
        if horizon_override is None:
            self.horizon = int(getattr(model, "n_tasks", 0))
        else:
            self.horizon = int(horizon_override)
        self.horizon_regime = "long" if self.horizon >= self.horizon_threshold else "short"

    def _distill_age_scale(self):
        if self.horizon_regime != "long":
            return 0.0
        return super()._distill_age_scale()


class BenefitDarkReplayEWCTrainer(DarkReplayEWCTrainer):
    """Online function-space benefit detector for DER++ distillation (P2.8).

    The detector periodically compares two reversible virtual updates on the
    same current batch and replay sample: DER++ off (alpha=0) vs DER++ on
    (alpha=dark_alpha). It turns distillation on only when the virtual DER++ step
    improves old replay behavior more than it harms the current batch. This
    avoids peeking at the stream horizon and tests whether function-space
    consequences can replace the failed weight-space RTP signal.
    """
    name = "BenefitDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0, benefit_probe_interval: int = 100,
                 benefit_ema_decay: float = 0.9, benefit_threshold: float = 0.0,
                 benefit_alpha_lr: float = 1.0, benefit_harm_weight: float = 1.0,
                 benefit_logit_weight: float = 0.0, benefit_min_old: int = 4):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
        )
        self.benefit_probe_interval = max(1, int(benefit_probe_interval))
        self.benefit_ema_decay = float(np.clip(benefit_ema_decay, 0.0, 0.999))
        self.benefit_threshold = float(benefit_threshold)
        self.benefit_alpha_lr = float(np.clip(benefit_alpha_lr, 0.0, 1.0))
        self.benefit_harm_weight = float(benefit_harm_weight)
        self.benefit_logit_weight = float(benefit_logit_weight)
        self.benefit_min_old = max(1, int(benefit_min_old))
        self.benefit_alpha = 0.0
        self.benefit_score_ema = None
        self.benefit_step = 0
        self.benefit_probe_count = 0
        self._probe_alpha_override = None
        self._suppress_alpha_trace = False
        self.benefit_alpha_trace = []
        self.benefit_score_trace = []
        self.benefit_score_ema_trace = []
        self.benefit_old_label_gain_trace = []
        self.benefit_old_logit_gain_trace = []
        self.benefit_current_harm_trace = []

    @staticmethod
    def _copy_grads(grads):
        return {k: v.copy() for k, v in grads.items()}

    def _snapshot_model(self):
        snap = {
            "W1": self.model.W1.copy(),
            "b1": self.model.b1.copy(),
            "W2": self.model.W2.copy(),
            "b2": self.model.b2.copy(),
        }
        if self.model.multi_head:
            snap["heads_W"] = [w.copy() for w in self.model.heads_W]
            snap["heads_b"] = [b.copy() for b in self.model.heads_b]
        else:
            snap["W3"] = self.model.W3.copy()
            snap["b3"] = self.model.b3.copy()
        if self.model.input_adapter:
            snap["adapters"] = [a.copy() for a in self.model.adapters]
        return snap

    def _restore_model(self, snap):
        self.model.W1 = snap["W1"]
        self.model.b1 = snap["b1"]
        self.model.W2 = snap["W2"]
        self.model.b2 = snap["b2"]
        if self.model.multi_head:
            self.model.heads_W = snap["heads_W"]
            self.model.heads_b = snap["heads_b"]
        else:
            self.model.W3 = snap["W3"]
            self.model.b3 = snap["b3"]
        if self.model.input_adapter:
            self.model.adapters = snap["adapters"]

    def _routed_loss_acc(self, X, Y, tasks):
        if len(X) == 0:
            return float("nan"), float("nan")
        if self.model.multi_head or self.model.input_adapter:
            total_loss = 0.0
            total_acc = 0.0
            for t in sorted(set(tasks)):
                idx = [i for i, val in enumerate(tasks) if val == t]
                loss, acc = self.model.loss_acc(X[idx], Y[idx], t)
                total_loss += loss * len(idx)
                total_acc += acc * len(idx)
            return total_loss / len(tasks), total_acc / len(tasks)
        return self.model.loss_acc(X, Y)

    def _routed_logit_mse(self, X, tasks, target_logits):
        if len(X) == 0:
            return float("nan")
        if self.model.multi_head or self.model.input_adapter:
            total = 0.0
            for t in sorted(set(tasks)):
                idx = [i for i, val in enumerate(tasks) if val == t]
                logits = self.model.forward(X[idx], t)["logits"]
                total += float(np.mean((logits - target_logits[idx]) ** 2)) * len(idx)
            return total / len(tasks)
        logits = self.model.forward(X)["logits"]
        return float(np.mean((logits - target_logits) ** 2))

    def _old_replay_subset(self, rX, rY, rtasks, rlogits, task_idx):
        if task_idx is None:
            idx = list(range(len(rtasks)))
        else:
            idx = [i for i, t in enumerate(rtasks) if t is not None and t < task_idx]
        if len(idx) == 0:
            return None
        return rX[idx], rY[idx], [rtasks[i] for i in idx], rlogits[idx]

    def _virtual_metrics(self, alpha, current_grads, X, Y, sample, task_idx, old_subset):
        snap = self._snapshot_model()
        prev_override = self._probe_alpha_override
        prev_suppress = self._suppress_alpha_trace
        try:
            self._probe_alpha_override = float(alpha)
            self._suppress_alpha_trace = True
            grads = self._copy_grads(current_grads)
            grads = self._mix_dark_replay_grads(grads, *sample, task_idx)
            ewc_grads = self._ewc_grad(task_idx)
            self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)

            old_X, old_Y, old_tasks, old_logits = old_subset
            old_loss, old_acc = self._routed_loss_acc(old_X, old_Y, old_tasks)
            old_mse = self._routed_logit_mse(old_X, old_tasks, old_logits)
            cur_loss, cur_acc = self.model.loss_acc(X, Y, task_idx)
            return dict(
                old_loss=float(old_loss),
                old_acc=float(old_acc),
                old_logit_mse=float(old_mse),
                current_loss=float(cur_loss),
                current_acc=float(cur_acc),
            )
        finally:
            self._probe_alpha_override = prev_override
            self._suppress_alpha_trace = prev_suppress
            self._restore_model(snap)

    def _update_benefit_alpha(self, current_grads, X, Y, sample, task_idx):
        max_alpha = float(self.dark_alpha * self._distill_age_scale())
        if max_alpha <= 0.0:
            self.benefit_alpha = 0.0
            return
        old_subset = self._old_replay_subset(*sample, task_idx)
        if old_subset is None or len(old_subset[0]) < self.benefit_min_old:
            return
        if self.benefit_step % self.benefit_probe_interval != 0:
            return

        off = self._virtual_metrics(0.0, current_grads, X, Y, sample, task_idx, old_subset)
        on = self._virtual_metrics(max_alpha, current_grads, X, Y, sample, task_idx, old_subset)

        old_label_gain = off["old_loss"] - on["old_loss"]
        old_logit_gain = off["old_logit_mse"] - on["old_logit_mse"]
        current_harm = on["current_loss"] - off["current_loss"]
        score = (
            old_label_gain
            + self.benefit_logit_weight * old_logit_gain
            - self.benefit_harm_weight * max(0.0, current_harm)
        )

        if self.benefit_score_ema is None:
            self.benefit_score_ema = float(score)
        else:
            d = self.benefit_ema_decay
            self.benefit_score_ema = float(d * self.benefit_score_ema + (1.0 - d) * score)

        target_alpha = max_alpha if self.benefit_score_ema > self.benefit_threshold else 0.0
        lr = self.benefit_alpha_lr
        self.benefit_alpha = float(np.clip((1.0 - lr) * self.benefit_alpha + lr * target_alpha, 0.0, max_alpha))
        self.benefit_probe_count += 1

        self.benefit_alpha_trace.append(float(self.benefit_alpha))
        self.benefit_score_trace.append(float(score))
        self.benefit_score_ema_trace.append(float(self.benefit_score_ema))
        self.benefit_old_label_gain_trace.append(float(old_label_gain))
        self.benefit_old_logit_gain_trace.append(float(old_logit_gain))
        self.benefit_current_harm_trace.append(float(current_harm))
        if len(self.benefit_score_trace) > 2000:
            self.benefit_alpha_trace = self.benefit_alpha_trace[-1000:]
            self.benefit_score_trace = self.benefit_score_trace[-1000:]
            self.benefit_score_ema_trace = self.benefit_score_ema_trace[-1000:]
            self.benefit_old_label_gain_trace = self.benefit_old_label_gain_trace[-1000:]
            self.benefit_old_logit_gain_trace = self.benefit_old_logit_gain_trace[-1000:]
            self.benefit_current_harm_trace = self.benefit_current_harm_trace[-1000:]

    def _effective_dark_alpha(self, current_grads, ce_grads, dark_grads):
        if self._probe_alpha_override is not None:
            return float(self._probe_alpha_override)
        alpha = float(np.clip(self.benefit_alpha, 0.0, self.dark_alpha * self._distill_age_scale()))
        if not self._suppress_alpha_trace:
            self.dark_alpha_trace.append(alpha)
            if len(self.dark_alpha_trace) > 2000:
                self.dark_alpha_trace = self.dark_alpha_trace[-1000:]
        return alpha

    def train_step(self, X, Y, task_idx: int = None):
        self._active_task_idx = task_idx
        self.benefit_step += 1
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        sample = self._sample_replay()
        if sample is not None:
            self._update_benefit_alpha(grads, X, Y, sample, task_idx)
            grads = self._mix_dark_replay_grads(grads, *sample, task_idx)

        ewc_grads = self._ewc_grad(task_idx)
        self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc


class SlowBenefitDarkReplayEWCTrainer(BenefitDarkReplayEWCTrainer):
    """Multi-step function-space benefit detector for DER++ distillation (P2.9).

    P2.8 used a single reversible update, which was safe but too short-sighted
    to see DER++'s slow proactive consolidation. This variant replays a short
    recent-batch window in a restored shadow state, compares DER++ on/off after
    that local rollout, then reuses the P2.8 EMA controller.
    """
    name = "SlowBenefitDarkReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, dark_confidence_threshold: float = 0.0,
                 dark_require_correct: bool = False, distill_start_task: int = 0,
                 distill_ramp_tasks: int = 0, benefit_probe_interval: int = 100,
                 benefit_ema_decay: float = 0.9, benefit_threshold: float = 0.0,
                 benefit_alpha_lr: float = 1.0, benefit_harm_weight: float = 1.0,
                 benefit_logit_weight: float = 0.0, benefit_min_old: int = 4,
                 slow_rollout_steps: int = 5):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            dark_alpha=dark_alpha,
            replay_weight=replay_weight,
            dark_confidence_threshold=dark_confidence_threshold,
            dark_require_correct=dark_require_correct,
            distill_start_task=distill_start_task,
            distill_ramp_tasks=distill_ramp_tasks,
            benefit_probe_interval=benefit_probe_interval,
            benefit_ema_decay=benefit_ema_decay,
            benefit_threshold=benefit_threshold,
            benefit_alpha_lr=benefit_alpha_lr,
            benefit_harm_weight=benefit_harm_weight,
            benefit_logit_weight=benefit_logit_weight,
            benefit_min_old=benefit_min_old,
        )
        self.slow_rollout_steps = max(1, int(slow_rollout_steps))
        self.slow_recent_batches = []
        self._slow_window_task = None

    def train_step(self, X, Y, task_idx: int = None):
        if task_idx != self._slow_window_task:
            self.slow_recent_batches = []
            self._slow_window_task = task_idx
        self.slow_recent_batches.append((X.copy(), Y.copy(), task_idx))
        if len(self.slow_recent_batches) > self.slow_rollout_steps:
            self.slow_recent_batches = self.slow_recent_batches[-self.slow_rollout_steps:]
        return super().train_step(X, Y, task_idx)

    def _virtual_metrics(self, alpha, current_grads, X, Y, sample, task_idx, old_subset):
        snap = self._snapshot_model()
        prev_override = self._probe_alpha_override
        prev_suppress = self._suppress_alpha_trace
        try:
            self._probe_alpha_override = float(alpha)
            self._suppress_alpha_trace = True
            rollout_batches = self.slow_recent_batches[-self.slow_rollout_steps:]
            if len(rollout_batches) == 0:
                rollout_batches = [(X, Y, task_idx)]
            for step_X, step_Y, step_task in rollout_batches:
                cache = self.model.forward(step_X, step_task)
                grads = self.model.backward(cache, step_Y, step_task)
                grads = self._mix_dark_replay_grads(grads, *sample, step_task)
                ewc_grads = self._ewc_grad(step_task)
                self.model.sgd_step(grads, self.lr, extra_grads=ewc_grads, task_idx=step_task)

            old_X, old_Y, old_tasks, old_logits = old_subset
            old_loss, old_acc = self._routed_loss_acc(old_X, old_Y, old_tasks)
            old_mse = self._routed_logit_mse(old_X, old_tasks, old_logits)
            cur_loss, cur_acc = self.model.loss_acc(X, Y, task_idx)
            return dict(
                old_loss=float(old_loss),
                old_acc=float(old_acc),
                old_logit_mse=float(old_mse),
                current_loss=float(cur_loss),
                current_acc=float(cur_acc),
            )
        finally:
            self._probe_alpha_override = prev_override
            self._suppress_alpha_trace = prev_suppress
            self._restore_model(snap)


class SurpriseReplayEWCTrainer(ReplayEWCTrainer):
    """ReplayEWC with surprise-prioritized replay sampling.

    Instead of uniformly replaying directly from the buffer, draw a larger
    candidate pool and replay the examples with the highest current CE loss.
    """
    name = "SurpriseReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, candidate_mult: int = 8):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
        )
        self.candidate_mult = max(1, int(candidate_mult))

    @staticmethod
    def _loss_and_margin_from_probs(probs, y):
        losses = -np.log(probs[np.arange(len(y)), y] + 1e-12)
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        top2.sort(axis=1)
        margins = top2[:, 1] - top2[:, 0]
        return losses, margins

    def _score_candidates(self, cX, cY, ctasks):
        n_candidates = len(cY)
        losses = np.zeros(n_candidates, dtype=np.float64)
        margins = np.zeros(n_candidates, dtype=np.float64)

        if self.model.multi_head or self.model.input_adapter:
            for t in sorted(set(ctasks)):
                sub_pos = np.array([i for i, val in enumerate(ctasks) if val == t], dtype=np.int64)
                cache = self.model.forward(cX[sub_pos], t)
                sub_losses, sub_margins = self._loss_and_margin_from_probs(cache["probs"], cY[sub_pos])
                losses[sub_pos] = sub_losses
                margins[sub_pos] = sub_margins
        else:
            cache = self.model.forward(cX)
            losses, margins = self._loss_and_margin_from_probs(cache["probs"], cY)

        return losses, margins

    def _sample_replay(self):
        if len(self.buf_X) == 0:
            return None
        n_replay = min(self.replay_batch, len(self.buf_X))
        n_candidates = min(len(self.buf_X), max(n_replay, n_replay * self.candidate_mult))
        candidate_idx = self.rng.choice(len(self.buf_X), size=n_candidates, replace=False)
        cX = np.stack([self.buf_X[i] for i in candidate_idx])
        cY = np.array([self.buf_Y[i] for i in candidate_idx], dtype=np.int64)
        ctasks = [self.buf_task[i] for i in candidate_idx]
        losses, _ = self._score_candidates(cX, cY, ctasks)

        top_pos = np.argsort(losses)[-n_replay:]
        rX = cX[top_pos]
        rY = cY[top_pos]
        rtasks = [ctasks[i] for i in top_pos]
        return rX, rY, rtasks


class MarginSurpriseReplayEWCTrainer(SurpriseReplayEWCTrainer):
    """ReplayEWC with surprise and decision-boundary replay priority.

    High CE loss catches forgotten or misclassified examples; low top-2 margin
    favors examples near old decision boundaries. The combined priority is a
    small-memory analogue of episodic exemplar selection: replay the old samples
    most likely to protect behavior that is currently fragile.
    """
    name = "MarginSurpriseReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, candidate_mult: int = 8,
                 margin_weight: float = 0.5):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            candidate_mult=candidate_mult,
        )
        self.margin_weight = margin_weight

    @staticmethod
    def _minmax(values):
        values = values.astype(np.float64)
        span = float(values.max() - values.min())
        if span < 1e-12:
            return np.zeros_like(values)
        return (values - values.min()) / span

    def _sample_replay(self):
        if len(self.buf_X) == 0:
            return None
        n_replay = min(self.replay_batch, len(self.buf_X))
        n_candidates = min(len(self.buf_X), max(n_replay, n_replay * self.candidate_mult))
        candidate_idx = self.rng.choice(len(self.buf_X), size=n_candidates, replace=False)
        cX = np.stack([self.buf_X[i] for i in candidate_idx])
        cY = np.array([self.buf_Y[i] for i in candidate_idx], dtype=np.int64)
        ctasks = [self.buf_task[i] for i in candidate_idx]

        losses, margins = self._score_candidates(cX, cY, ctasks)
        boundary_scores = 1.0 - margins
        priority = self._minmax(losses) + self.margin_weight * self._minmax(boundary_scores)

        top_pos = np.argsort(priority)[-n_replay:]
        rX = cX[top_pos]
        rY = cY[top_pos]
        rtasks = [ctasks[i] for i in top_pos]
        return rX, rY, rtasks


class HippocampalReplayEWCTrainer(SurpriseReplayEWCTrainer):
    """ReplayEWC with a small episodic-memory readout.

    The MLP remains the slow parametric learner. The replay buffer also acts as a
    hippocampal memory: at evaluation time we build class prototypes from stored
    episodes in the current hidden representation and blend their prediction with
    the MLP softmax. In multi-head Task-IL mode, memory is context-filtered by
    task id; in single-head mode it defaults to a shared memory unless
    memory_task_filter=True is explicitly requested.
    """
    name = "HippocampalReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, candidate_mult: int = 8,
                 memory_alpha: float = 0.8, memory_temperature: float = 0.1,
                 memory_min_examples: int = 4, memory_task_filter: bool = False,
                 memory_gate: bool = True, sleep_steps: int = 0, sleep_batch: int = 32,
                 sleep_lr_scale: float = 0.25):
        super().__init__(
            model,
            lr=lr,
            capacity=capacity,
            replay_batch=replay_batch,
            seed=seed,
            lam=lam,
            fisher_batches=fisher_batches,
            fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm,
            candidate_mult=candidate_mult,
        )
        self.memory_alpha = float(np.clip(memory_alpha, 0.0, 1.0))
        self.memory_temperature = max(1e-6, float(memory_temperature))
        self.memory_min_examples = max(1, int(memory_min_examples))
        self.memory_task_filter = bool(memory_task_filter)
        self.memory_gate = bool(memory_gate)
        self.sleep_steps = max(0, int(sleep_steps))
        self.sleep_batch = max(1, int(sleep_batch))
        self.sleep_lr_scale = max(0.0, float(sleep_lr_scale))

    @staticmethod
    def _row_normalize(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def _memory_indices(self, task_idx):
        if len(self.buf_X) == 0:
            return []
        # With an input adapter, each task's features live in the shared space only after
        # passing through that task's adapter, so memory must be task-filtered (queried by
        # same-task inputs) for the prototype comparison to be in a consistent space.
        use_context = self.model.multi_head or self.memory_task_filter or self.model.input_adapter
        if not use_context:
            return list(range(len(self.buf_X)))
        if task_idx is None:
            return []
        return [i for i, t in enumerate(self.buf_task) if t == task_idx]

    def _feature_cache(self, X, task_idx):
        if self.model.multi_head or self.model.input_adapter:
            return self.model.forward(X, task_idx)["a2"]
        return self.model.forward(X)["a2"]

    def _episodic_probs(self, X, task_idx, out_dim):
        idx = self._memory_indices(task_idx)
        if len(idx) < self.memory_min_examples:
            return None

        mem_X = np.stack([self.buf_X[i] for i in idx])
        mem_Y = np.array([self.buf_Y[i] for i in idx], dtype=np.int64)
        classes = np.array(sorted(set(int(y) for y in mem_Y)), dtype=np.int64)
        if len(classes) == 0:
            return None

        z = self._row_normalize(self._feature_cache(X, task_idx))
        mem_z = self._row_normalize(self._feature_cache(mem_X, task_idx))

        prototypes = []
        present_classes = []
        for c in classes:
            cls_z = mem_z[mem_Y == c]
            if len(cls_z) == 0:
                continue
            prototypes.append(cls_z.mean(axis=0))
            present_classes.append(c)
        if not prototypes:
            return None

        proto = self._row_normalize(np.stack(prototypes))
        logits = (z @ proto.T) / self.memory_temperature
        present_probs = self.model_softmax(logits)

        probs = np.full((X.shape[0], out_dim), 1e-8 / out_dim, dtype=np.float64)
        for col, c in enumerate(present_classes):
            if 0 <= c < out_dim:
                probs[:, c] = present_probs[:, col]
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    @staticmethod
    def model_softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss_acc(self, X, Y, task_idx: int = None):
        needs_ctx = self.model.multi_head or self.model.input_adapter
        cache = self.model.forward(X, task_idx if needs_ctx else None)
        probs = cache["probs"].astype(np.float64)
        mem_probs = self._episodic_probs(X, task_idx, probs.shape[1])
        if mem_probs is not None and self.memory_alpha > 0:
            if self.memory_gate:
                top2 = np.partition(probs, -2, axis=1)[:, -2:]
                top2.sort(axis=1)
                margins = top2[:, 1] - top2[:, 0]
                alpha = (self.memory_alpha * (1.0 - margins))[:, None]
            else:
                alpha = self.memory_alpha
            probs = (1.0 - alpha) * probs + alpha * mem_probs
            probs /= probs.sum(axis=1, keepdims=True)

        n = X.shape[0]
        loss = float(-np.log(probs[np.arange(n), Y] + 1e-12).mean())
        acc = float((probs.argmax(axis=1) == Y).mean())
        return loss, acc

    def _sleep_replay_step(self, rX, rY, rtasks):
        lr = self.lr * self.sleep_lr_scale
        if lr <= 0:
            return

        if self.model.multi_head:
            shared = {
                "W1": np.zeros_like(self.model.W1),
                "b1": np.zeros_like(self.model.b1),
                "W2": np.zeros_like(self.model.W2),
                "b2": np.zeros_like(self.model.b2),
            }
            head_grads = {}
            for t in sorted(set(rtasks)):
                sub_idx = [i for i, val in enumerate(rtasks) if val == t]
                sub_X = rX[sub_idx]
                sub_Y = rY[sub_idx]
                cache = self.model.forward(sub_X, t)
                grads = self.model.backward(cache, sub_Y, t)
                weight = len(sub_idx) / len(rtasks)
                for k in shared:
                    shared[k] += weight * grads[k]
                head_grads[t] = {
                    "W3": weight * grads["W3"],
                    "b3": weight * grads["b3"],
                }

            ewc = self._ewc_grad(rtasks[0] if rtasks else 0)
            for k in shared:
                shared[k] += ewc.get(k, 0.0)
            self.model.W1 -= lr * shared["W1"]
            self.model.b1 -= lr * shared["b1"]
            self.model.W2 -= lr * shared["W2"]
            self.model.b2 -= lr * shared["b2"]
            for t, grads in head_grads.items():
                self.model.heads_W[t] -= lr * grads["W3"]
                self.model.heads_b[t] -= lr * grads["b3"]
            return

        if self.model.input_adapter:
            shared = {k: np.zeros_like(getattr(self.model, k))
                      for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
            for t in sorted(set(rtasks)):
                sub_idx = [i for i, val in enumerate(rtasks) if val == t]
                cache = self.model.forward(rX[sub_idx], t)
                grads = self.model.backward(cache, rY[sub_idx], t)
                weight = len(sub_idx) / len(rtasks)
                for k in shared:
                    shared[k] += weight * grads[k]
                if "A" in grads:
                    self.model.adapters[t] -= lr * weight * grads["A"]
            ewc = self._ewc_grad(None)
            for k in shared:
                p = getattr(self.model, k)
                p -= lr * (shared[k] + ewc.get(k, 0.0))
            return

        cache = self.model.forward(rX)
        grads = self.model.backward(cache, rY)
        ewc = self._ewc_grad(None)
        self.model.sgd_step(grads, lr, extra_grads=ewc)

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        super().on_task_end(task_X, task_Y, task_idx)
        if self.sleep_steps <= 0:
            return

        original_replay_batch = self.replay_batch
        self.replay_batch = self.sleep_batch
        try:
            for _ in range(self.sleep_steps):
                sample = self._sample_replay()
                if sample is None:
                    break
                self._sleep_replay_step(*sample)
        finally:
            self.replay_batch = original_replay_batch


class NCMReplayEWCTrainer(HippocampalReplayEWCTrainer):
    """Class-IL nearest-class-mean (iCaRL 風格) 訓練器。

    表徵由 replay + EWC 學習；分類**不用線性輸出頭**，改用「特徵空間最近類別原型（cosine）」
    讀出，對所有看過的全域類別、不給 task id。動機：Class-IL 的單一線性頭有嚴重的 recency /
    magnitude bias（新類別 logit 天生偏大→蓋過舊類別），但用 buffer 算出的 class prototypes 是
    類別平衡、無偏的，因此幾乎消除遺忘。這是 HippocampalReplayEWC 在 Class-IL 下的純 NCM 設定
    （memory_alpha=1、不 gating、不 task-filter）。"""
    name = "NCMReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, candidate_mult: int = 8,
                 memory_temperature: float = 0.1, memory_min_examples: int = 3):
        super().__init__(
            model, lr=lr, capacity=capacity, replay_batch=replay_batch, seed=seed,
            lam=lam, fisher_batches=fisher_batches, fisher_decay=fisher_decay,
            grad_clip_norm=grad_clip_norm, candidate_mult=candidate_mult,
            memory_alpha=1.0, memory_temperature=memory_temperature,
            memory_min_examples=memory_min_examples, memory_task_filter=False,
            memory_gate=False, sleep_steps=0)


class ContinualBackpropTrainer:
    """簡化版 continual backprop（Dohare et al.）：
    每個隱藏單元維護一個 utility（效用，貢獻度的指數移動平均）與 age（自上次重置後的步數）。
    每一步，效用低於同層中位數、且 age 已超過成熟門檻的單元，會以一個很小的機率被選去重置：
    - 輸入端權重重新隨機初始化（恢復可塑性）
    - 輸出端權重設為 0（函數保持式的加入，不立刻擾動現有輸出——呼應 Net2Net 的精神）
    - utility、age 歸零
    其餘權重完全不受影響、不重置——這正是「不整體重置、只回收死神經元」的機制。
    """
    name = "ContinualBP"

    def __init__(self, model: MLP, lr: float = 0.05, replacement_rate: float = 1e-4,
                 maturity_threshold: int = 100, util_decay: float = 0.99, seed: int = 0):
        self.model = model
        self.lr = lr
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_decay = util_decay
        self.rng = np.random.RandomState(seed)

        h1, h2 = model.dims[1], model.dims[2]
        self.age = {"h1": np.zeros(h1, dtype=np.int64), "h2": np.zeros(h2, dtype=np.int64)}
        self.util = {"h1": np.zeros(h1, dtype=np.float64), "h2": np.zeros(h2, dtype=np.float64)}
        self.replace_accum = {"h1": 0.0, "h2": 0.0}

    def _update_utility(self, layer_key, activations, outgoing_W):
        contrib = np.abs(activations).mean(axis=0) * np.abs(outgoing_W).sum(axis=1)
        u = self.util[layer_key]
        u *= self.util_decay
        u += (1 - self.util_decay) * contrib
        self.age[layer_key] += 1

    def _maybe_replace(self, layer_key, incoming_W, incoming_b, outgoing_W, task_idx=None):
        n_units = len(self.util[layer_key])
        self.replace_accum[layer_key] += self.replacement_rate * n_units
        n_replace = int(self.replace_accum[layer_key])
        if n_replace < 1:
            return
        self.replace_accum[layer_key] -= n_replace

        eligible = np.where(self.age[layer_key] >= self.maturity_threshold)[0]
        if len(eligible) == 0:
            return
        n_replace = min(n_replace, len(eligible))
        order = eligible[np.argsort(self.util[layer_key][eligible])]
        to_reset = order[:n_replace]

        fan_in = incoming_W.shape[0]
        std = np.sqrt(2.0 / fan_in)
        for idx in to_reset:
            incoming_W[:, idx] = self.rng.randn(fan_in) * std
            incoming_b[idx] = 0.0
            if self.model.multi_head and layer_key == "h2":
                for h_w in self.model.heads_W:
                    h_w[idx, :] = 0.0
            else:
                outgoing_W[idx, :] = 0.0
            self.util[layer_key][idx] = 0.0
            self.age[layer_key][idx] = 0

    def train_step(self, X, Y, task_idx: int = None):
        m = self.model
        loss, acc = m.loss_acc(X, Y, task_idx)
        cache = m.forward(X, task_idx)
        grads = m.backward(cache, Y, task_idx)
        m.sgd_step(grads, self.lr, task_idx=task_idx)

        outgoing_W3 = m.heads_W[task_idx] if m.multi_head else m.W3
        self._update_utility("h1", cache["a1"], m.W2)
        self._update_utility("h2", cache["a2"], outgoing_W3)
        self._maybe_replace("h1", m.W1, m.b1, m.W2, task_idx)
        self._maybe_replace("h2", m.W2, m.b2, outgoing_W3, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        pass


class ReplayContinualBackpropTrainer(TaskBalancedReplayTrainer):
    """Task-balanced replay plus continual backprop.

    Replay protects old task behavior; continual backprop keeps hidden units from
    becoming permanently dormant. The combination is the most direct baseline for
    testing whether stability and plasticity mechanisms are complementary here.
    """
    name = "ReplayContinualBP"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 500,
                 replay_batch: int = 16, replacement_rate: float = 1e-4,
                 maturity_threshold: int = 100, util_decay: float = 0.99,
                 seed: int = 0):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch, seed=seed)
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_decay = util_decay

        h1, h2 = model.dims[1], model.dims[2]
        self.age = {"h1": np.zeros(h1, dtype=np.int64), "h2": np.zeros(h2, dtype=np.int64)}
        self.util = {"h1": np.zeros(h1, dtype=np.float64), "h2": np.zeros(h2, dtype=np.float64)}
        self.replace_accum = {"h1": 0.0, "h2": 0.0}

    def _update_utility(self, layer_key, activations, outgoing_W):
        contrib = np.abs(activations).mean(axis=0) * np.abs(outgoing_W).sum(axis=1)
        self.util[layer_key] *= self.util_decay
        self.util[layer_key] += (1 - self.util_decay) * contrib
        self.age[layer_key] += 1

    def _maybe_replace(self, layer_key, incoming_W, incoming_b, outgoing_W):
        n_units = len(self.util[layer_key])
        self.replace_accum[layer_key] += self.replacement_rate * n_units
        n_replace = int(self.replace_accum[layer_key])
        if n_replace < 1:
            return
        self.replace_accum[layer_key] -= n_replace

        eligible = np.where(self.age[layer_key] >= self.maturity_threshold)[0]
        if len(eligible) == 0:
            return
        n_replace = min(n_replace, len(eligible))
        order = eligible[np.argsort(self.util[layer_key][eligible])]
        to_reset = order[:n_replace]

        fan_in = incoming_W.shape[0]
        std = np.sqrt(2.0 / fan_in)
        for idx in to_reset:
            incoming_W[:, idx] = self.rng.randn(fan_in) * std
            incoming_b[idx] = 0.0
            if self.model.multi_head and layer_key == "h2":
                for h_w in self.model.heads_W:
                    h_w[idx, :] = 0.0
            else:
                outgoing_W[idx, :] = 0.0
            self.util[layer_key][idx] = 0.0
            self.age[layer_key][idx] = 0

    def _refresh_plasticity(self, X, task_idx):
        m = self.model
        cache = m.forward(X, task_idx)
        outgoing_W3 = m.heads_W[task_idx] if m.multi_head else m.W3
        self._update_utility("h1", cache["a1"], m.W2)
        self._update_utility("h2", cache["a2"], outgoing_W3)
        self._maybe_replace("h1", m.W1, m.b1, m.W2)
        self._maybe_replace("h2", m.W2, m.b2, outgoing_W3)

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = super().train_step(X, Y, task_idx)
        self._refresh_plasticity(X, task_idx)
        return loss, acc


class JointTrainer:
    """離線多任務上界（offline / cumulative joint）。

    儲存所有看過的樣本（無上限 buffer），每一步從「目前為止所有 task 的聯集」均勻抽
    mini-batch 做 i.i.d. 更新，等於拿掉持續學習的循序限制。這是判斷其他方法好壞的天花板：
    它的 final average accuracy 就是「同一個網路在沒有遺忘限制下能到多高」。

    注意：這蓄意打破計算/記憶體預算公平性（它能重看所有舊資料），所以只當上界基準，
    不列入與其他方法的同預算比較。多頭與 input-adapter 都支援：抽到的混合 batch 會依
    task 分組，各自走對應的 head / adapter。
    """
    name = "Joint"

    def __init__(self, model: MLP, lr: float = 0.05, joint_batch: int = 64,
                 joint_steps: int = 2, seed: int = 0):
        self.model = model
        self.lr = lr
        self.joint_batch = max(1, int(joint_batch))
        self.joint_steps = max(1, int(joint_steps))
        self.rng = np.random.RandomState(seed)
        self.buf_X = []
        self.buf_Y = []
        self.buf_task = []

    def _insert(self, X, Y, task_idx):
        t = 0 if task_idx is None else int(task_idx)
        for i in range(X.shape[0]):
            self.buf_X.append(X[i].copy())
            self.buf_Y.append(int(Y[i]))
            self.buf_task.append(t)

    def _sample(self):
        n = min(self.joint_batch, len(self.buf_X))
        idx = self.rng.randint(0, len(self.buf_X), size=n)
        bX = np.stack([self.buf_X[i] for i in idx])
        bY = np.array([self.buf_Y[i] for i in idx], dtype=np.int64)
        bt = [self.buf_task[i] for i in idx]
        return bX, bY, bt

    def _joint_update(self, bX, bY, btasks):
        m = self.model
        n = len(bY)
        shared_keys = ["W1", "b1", "W2", "b2"]
        if not m.multi_head:
            shared_keys += ["W3", "b3"]

        shared = None
        per_task = []  # (task, head/adapter grads)
        for t in sorted(set(int(v) for v in btasks)):
            sub_idx = [i for i, val in enumerate(btasks) if int(val) == t]
            cache = m.forward(bX[sub_idx], t)
            g = m.backward(cache, bY[sub_idx], t)
            weight = len(sub_idx) / n
            if shared is None:
                shared = {k: np.zeros_like(g[k]) for k in shared_keys}
            for k in shared_keys:
                shared[k] += weight * g[k]
            extra = {}
            if m.multi_head:
                extra["W3"] = weight * g["W3"]
                extra["b3"] = weight * g["b3"]
            if m.input_adapter and "A" in g:
                extra["A"] = g["A"]
            per_task.append((t, extra))

        for k in shared_keys:
            p = getattr(m, k)
            p -= self.lr * shared[k]
        for t, extra in per_task:
            if "W3" in extra:
                m.heads_W[t] -= self.lr * extra["W3"]
                m.heads_b[t] -= self.lr * extra["b3"]
            if "A" in extra:
                m.adapters[t] -= self.lr * extra["A"]

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        self._insert(X, Y, task_idx)
        for _ in range(self.joint_steps):
            self._joint_update(*self._sample())
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        pass


class SustainableReplayEWCTrainer(ReplayEWCTrainer):
    """可永續學習：ReplayEWC（穩定性）+ Fisher 保護的神經元回收（可塑性）。

    動機：continual backprop 能維持可塑性，但它的「盲目重置」會覆寫舊任務權重——
    這正是 ReplayContinualBP 表現反而比純 Replay 差的原因（48% vs 60%）。本方法的修正是
    讓神經元回收「對舊任務記憶有感」：

    1. 只回收同時「低效用」且「對舊任務不重要（Fisher importance 低）」的成熟單元。
       Fisher 是 EWC 跨任務累積的重要度，高 Fisher 的單元即使現在看似休眠，也可能對舊任務
       關鍵，因此被保護、不重置。
    2. 被回收的單元，連同它在 EWC 的 anchor/Fisher 一併清掉，否則正則項會把新初始化的權重
       又拉回舊的死亡值，回收等於白做。

    目標：在長串流上同時壓住遺忘（失效 A）與可塑性流失（失效 B）。
    """
    name = "SustainableReplayEWC"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 5.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, replacement_rate: float = 1e-4,
                 maturity_threshold: int = 100, util_decay: float = 0.99,
                 fisher_protect_quantile: float = 0.5):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm)
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_decay = util_decay
        self.fisher_protect_quantile = float(np.clip(fisher_protect_quantile, 0.0, 1.0))

        h1, h2 = model.dims[1], model.dims[2]
        self.age = {"h1": np.zeros(h1, dtype=np.int64), "h2": np.zeros(h2, dtype=np.int64)}
        self.util = {"h1": np.zeros(h1, dtype=np.float64), "h2": np.zeros(h2, dtype=np.float64)}
        self.replace_accum = {"h1": 0.0, "h2": 0.0}

    def _update_utility(self, layer_key, activations, outgoing_W):
        contrib = np.abs(activations).mean(axis=0) * np.abs(outgoing_W).sum(axis=1)
        self.util[layer_key] *= self.util_decay
        self.util[layer_key] += (1 - self.util_decay) * contrib
        self.age[layer_key] += 1

    def _fisher_importance(self, layer_key):
        """每個隱藏單元對舊任務的重要度，取自 EWC 累積的 Fisher（越高越該保護）。"""
        if layer_key == "h1":
            return self.fisher["W1"].sum(axis=0) + self.fisher["W2"].sum(axis=1)
        imp = self.fisher["W2"].sum(axis=0)
        if "W3" in self.fisher:  # single-head: outgoing weights are EWC-tracked
            imp = imp + self.fisher["W3"].sum(axis=1)
        return imp

    def _clear_ewc_for_unit(self, layer_key, idx):
        """回收後把該單元的 EWC anchor/Fisher 清掉（重新初始化＝全新單元，不該被舊錨拉回）。"""
        m = self.model
        if layer_key == "h1":
            self.fisher["W1"][:, idx] = 0.0; self.anchor["W1"][:, idx] = m.W1[:, idx]
            self.fisher["b1"][idx] = 0.0;    self.anchor["b1"][idx] = m.b1[idx]
            self.fisher["W2"][idx, :] = 0.0; self.anchor["W2"][idx, :] = m.W2[idx, :]
        else:
            self.fisher["W2"][:, idx] = 0.0; self.anchor["W2"][:, idx] = m.W2[:, idx]
            self.fisher["b2"][idx] = 0.0;    self.anchor["b2"][idx] = m.b2[idx]
            if "W3" in self.fisher:
                self.fisher["W3"][idx, :] = 0.0; self.anchor["W3"][idx, :] = m.W3[idx, :]

    def _maybe_replace(self, layer_key, incoming_W, incoming_b, outgoing_W):
        n_units = len(self.util[layer_key])
        self.replace_accum[layer_key] += self.replacement_rate * n_units
        n_replace = int(self.replace_accum[layer_key])
        if n_replace < 1:
            return
        self.replace_accum[layer_key] -= n_replace

        mature = self.age[layer_key] >= self.maturity_threshold
        if not np.any(mature):
            return
        imp = self._fisher_importance(layer_key)
        # 保護舊任務關鍵單元：只有 Fisher 重要度落在低分位以下的成熟單元才符合回收資格。
        thresh = np.quantile(imp, self.fisher_protect_quantile)
        eligible = np.where(mature & (imp <= thresh))[0]
        if len(eligible) == 0:
            return
        n_replace = min(n_replace, len(eligible))
        order = eligible[np.argsort(self.util[layer_key][eligible])]
        to_reset = order[:n_replace]

        fan_in = incoming_W.shape[0]
        std = np.sqrt(2.0 / fan_in)
        for idx in to_reset:
            incoming_W[:, idx] = self.rng.randn(fan_in) * std
            incoming_b[idx] = 0.0
            if self.model.multi_head and layer_key == "h2":
                for h_w in self.model.heads_W:
                    h_w[idx, :] = 0.0
            else:
                outgoing_W[idx, :] = 0.0
            self.util[layer_key][idx] = 0.0
            self.age[layer_key][idx] = 0
            self._clear_ewc_for_unit(layer_key, idx)

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = super().train_step(X, Y, task_idx)
        m = self.model
        cache = m.forward(X, task_idx)
        outgoing_W3 = m.heads_W[task_idx] if m.multi_head else m.W3
        self._update_utility("h1", cache["a1"], m.W2)
        self._update_utility("h2", cache["a2"], outgoing_W3)
        self._maybe_replace("h1", m.W1, m.b1, m.W2)
        self._maybe_replace("h2", m.W2, m.b2, outgoing_W3)
        return loss, acc


class FunctionSpaceReplayTrainer(DarkReplayEWCTrainer):
    """混合式函數空間抗遺忘訓練器：從三個角度同時逼網路「不要改變舊任務的函數」。

    1. Experience Replay（覆蓋）——讓網路持續看到舊任務輸入（繼承自 ReplayTrainer）。
    2. DER++ logit 蒸餾（函數軟錨）——回放時除了 CE，還用 MSE 把現在對舊樣本的輸出拉回
       它寫入 buffer 當下的 logits，保住整個 softmax 幾何（繼承自 DarkReplayEWC 的 dark 項）。
    3. GPM 梯度投影（參數子空間硬鎖）——維護每個共享層「舊任務輸入子空間」的正交基 M，
       把共享層梯度投影到 M 的正交補：G ← G − M(MᵀG)。因為某層輸出=W·x，更新若與舊輸入
       正交，舊任務的激活與輸出在數學上不被擾動。基在每個 task 結束時用該層輸入的 SVD 增量更新。

    預設關掉 EWC（lam=0）：GPM 是更鋒利的權重保護，與 EWC 在舊子空間方向上會互相抵銷，
    故由 GPM 取代之。三個元件都可切換，方便做 Replay→+DER++→+GPM 的消融。只投影共享層
    （W1/W2，單頭再加 W3）；heads/adapters 是 task-specific，不投影也不互相干擾。
    """
    name = "FunctionSpaceReplay"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, lam: float = 0.0,
                 fisher_batches: int = 30, fisher_decay: float = 0.9,
                 grad_clip_norm: float = 50.0, dark_alpha: float = 0.5,
                 replay_weight: float = 0.5, use_gpm: bool = True,
                 gpm_threshold: float = 0.97, gpm_max_rank: int = None,
                 gpm_samples: int = 1000):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch,
                         seed=seed, lam=lam, fisher_batches=fisher_batches,
                         fisher_decay=fisher_decay, grad_clip_norm=grad_clip_norm,
                         dark_alpha=dark_alpha, replay_weight=replay_weight)
        self.use_gpm = bool(use_gpm)
        self.gpm_threshold = float(gpm_threshold)
        self.gpm_samples = int(gpm_samples)
        # 只保護共享權重矩陣；多頭時 W3 是 per-head（不共享），不投影。
        self.gpm_keys = ["W1", "W2"] + ([] if model.multi_head else ["W3"])
        self.gpm_max_rank = gpm_max_rank if gpm_max_rank is not None else max(model.dims)
        self.gpm_bases = {k: None for k in self.gpm_keys}

    @staticmethod
    def _merge_extra(grads, extra):
        for k, v in extra.items():
            grads[k] = grads[k] + v if k in grads else v
        return grads

    def _project_shared_grads(self, grads):
        """把共享層梯度投影到舊任務輸入子空間的正交補：G ← G − M(MᵀG)。"""
        for k in self.gpm_keys:
            M = self.gpm_bases.get(k)
            if M is None or M.shape[1] == 0 or k not in grads:
                continue
            G = grads[k]
            grads[k] = G - M @ (M.T @ G)
        return grads

    def _update_gpm_basis(self, task_X, task_idx):
        """task 結束後，用該層輸入的 SVD 增量擴充正交基（GPM, Saha et al. 2021）。"""
        m = self.model
        n = len(task_X)
        if n > self.gpm_samples:
            sel = self.rng.choice(n, size=self.gpm_samples, replace=False)
            task_X = task_X[sel]
        cache = m.forward(task_X, task_idx)
        # 每個共享層「看到的輸入」：W1<-X_in（adapter 後）、W2<-a1、W3<-a2
        layer_in = {"W1": cache["X_in"], "W2": cache["a1"]}
        if not m.multi_head:
            layer_in["W3"] = cache["a2"]
        for k in self.gpm_keys:
            A = np.asarray(layer_in[k], dtype=np.float64)        # (n_samples, d)
            total = float((A ** 2).sum())
            if total < 1e-12:
                continue
            M = self.gpm_bases.get(k)
            A_res = A - (A @ M) @ M.T if (M is not None and M.shape[1] > 0) else A
            try:
                _, S, Vt = np.linalg.svd(A_res, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            captured = total - float((A_res ** 2).sum())          # energy already in M
            cum = captured
            new_dirs = []
            for i in range(len(S)):
                if cum / total >= self.gpm_threshold:
                    break
                new_dirs.append(Vt[i])                            # d-dim feature direction
                cum += float(S[i] ** 2)
            if not new_dirs:
                continue
            new = np.stack(new_dirs, axis=1)                      # (d, k)
            M = np.concatenate([M, new], axis=1) if (M is not None and M.shape[1] > 0) else new
            Q, _ = np.linalg.qr(M)                                # re-orthonormalize
            self.gpm_bases[k] = Q[:, :min(Q.shape[1], self.gpm_max_rank)]

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = self.model.loss_acc(X, Y, task_idx)
        cache = self.model.forward(X, task_idx)
        grads = self.model.backward(cache, Y, task_idx)

        sample = self._sample_replay()
        if sample is not None:
            grads = self._mix_dark_replay_grads(grads, *sample, task_idx)

        grads = self._merge_extra(grads, self._ewc_grad(task_idx))
        if self.use_gpm:
            grads = self._project_shared_grads(grads)
        self.model.sgd_step(grads, self.lr, task_idx=task_idx)
        self._reservoir_insert(X, Y, task_idx)
        return loss, acc

    def on_task_end(self, task_X, task_Y, task_idx: int = None):
        super().on_task_end(task_X, task_Y, task_idx)
        if self.use_gpm:
            self._update_gpm_basis(task_X, task_idx)


def _benna_fusi_init(model, keys, levels, g0):
    """每個權重一條 N 級鏈：capacity C_k=2^k（容量幾何遞增）、conductance g_k=g0·2^-k
    （管徑幾何遞減）。隱藏變數 u_2..u_N 以目前可見權重初始化（加入瞬間鏈一致、不擾動函數）。"""
    C = np.array([2.0 ** i for i in range(levels)], dtype=np.float64)
    g = np.array([g0 * 2.0 ** (-i) for i in range(levels)], dtype=np.float64)
    hidden = {k: [getattr(model, k).astype(np.float64) for _ in range(levels - 1)] for k in keys}
    return C, g, hidden


def _benna_fusi_relax(model, keys, hidden, C, g, dt, levels):
    """一步擴散鬆弛（封閉鏈，無洩漏到 ground，保總量守恆）：可見變數 u_1 被往鏈的
    慢速共識拉，抗快速覆寫＝較慢漂移＝較少遺忘；深層慢變數承載鞏固後的歷史。"""
    for k in keys:
        chain = [getattr(model, k)] + hidden[k]          # u_1 (可見) .. u_N
        new = []
        for i in range(levels):
            inflow = g[i - 1] * (chain[i - 1] - chain[i]) if i > 0 else 0.0
            outflow = g[i] * (chain[i] - chain[i + 1]) if i + 1 < levels else 0.0
            new.append(chain[i] + dt * (inflow - outflow) / C[i])
        getattr(model, k)[...] = new[0]
        for j in range(levels - 1):
            hidden[k][j] = new[j + 1]


class BennaFusiTrainer(NaiveTrainer):
    """Benna-Fusi 複雜突觸（Benna & Fusi 2016）：把每個共享權重換成一串耦合、不同時間尺度的
    內部變數，記憶先進快變數再逐步轉移到慢變數，給出冪律（而非指數）遺忘——記憶壽命大幅延長，
    且完全不儲存任何過去樣本。這裡是純機制版（線上 SGD + 複雜突觸），對照 Naive 看遺忘是否下降。"""
    name = "BennaFusi"

    def __init__(self, model: MLP, lr: float = 0.05, bf_levels: int = 5,
                 bf_g0: float = 1.0, bf_dt: float = 0.1):
        super().__init__(model, lr=lr)
        self.bf_levels = max(2, int(bf_levels))
        self.bf_dt = float(bf_dt)
        self.bf_keys = ["W1", "b1", "W2", "b2"] + ([] if model.multi_head else ["W3", "b3"])
        self.bf_C, self.bf_g, self.bf_hidden = _benna_fusi_init(model, self.bf_keys, self.bf_levels, bf_g0)

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = super().train_step(X, Y, task_idx)
        _benna_fusi_relax(self.model, self.bf_keys, self.bf_hidden, self.bf_C, self.bf_g, self.bf_dt, self.bf_levels)
        return loss, acc


class BennaFusiReplayTrainer(ReplayTrainer):
    """Experience Replay + Benna-Fusi 複雜突觸（無 EWC）。複雜突觸取代 EWC 當「權重穩定」機制，
    測試冪律遺忘能否補足、甚至替代 shared-weight 正則：對照 ReplayEWC 看是否打平/超越。"""
    name = "BennaFusiReplay"

    def __init__(self, model: MLP, lr: float = 0.05, capacity: int = 2000,
                 replay_batch: int = 16, seed: int = 0, bf_levels: int = 5,
                 bf_g0: float = 1.0, bf_dt: float = 0.1):
        super().__init__(model, lr=lr, capacity=capacity, replay_batch=replay_batch, seed=seed)
        self.bf_levels = max(2, int(bf_levels))
        self.bf_dt = float(bf_dt)
        self.bf_keys = ["W1", "b1", "W2", "b2"] + ([] if model.multi_head else ["W3", "b3"])
        self.bf_C, self.bf_g, self.bf_hidden = _benna_fusi_init(model, self.bf_keys, self.bf_levels, bf_g0)

    def train_step(self, X, Y, task_idx: int = None):
        loss, acc = super().train_step(X, Y, task_idx)
        _benna_fusi_relax(self.model, self.bf_keys, self.bf_hidden, self.bf_C, self.bf_g, self.bf_dt, self.bf_levels)
        return loss, acc


TRAINER_REGISTRY = {
    "Naive": NaiveTrainer,
    "Joint": JointTrainer,
    "EWC": EWCTrainer,
    "Replay": ReplayTrainer,
    "ReplayEWC": ReplayEWCTrainer,
    "DarkReplayEWC": DarkReplayEWCTrainer,
    "AdaptiveDarkReplayEWC": AdaptiveDarkReplayEWCTrainer,
    "PressureDarkReplayEWC": PressureDarkReplayEWCTrainer,
    "LookaheadDarkReplayEWC": LookaheadDarkReplayEWCTrainer,
    "RtpDarkReplayEWC": RtpDarkReplayEWCTrainer,
    "HorizonDarkReplayEWC": HorizonDarkReplayEWCTrainer,
    "BenefitDarkReplayEWC": BenefitDarkReplayEWCTrainer,
    "SlowBenefitDarkReplayEWC": SlowBenefitDarkReplayEWCTrainer,
    "OnlineEWCReplay": OnlineEWCReplayTrainer,
    "OnlineDarkReplayEWC": OnlineDarkReplayEWCTrainer,
    "GenerativeReplayEWC": GenerativeReplayEWCTrainer,
    "NBGenerativeReplayEWC": NBGenerativeReplayEWCTrainer,
    "ScholarGenerativeReplayEWC": ScholarGenerativeReplayEWCTrainer,
    "ScholarGlobalGenerativeReplayEWC": ScholarGlobalGenerativeReplayEWCTrainer,
    "SurpriseReplayEWC": SurpriseReplayEWCTrainer,
    "MarginSurpriseReplayEWC": MarginSurpriseReplayEWCTrainer,
    "HippocampalReplayEWC": HippocampalReplayEWCTrainer,
    "NCMReplayEWC": NCMReplayEWCTrainer,
    "TaskBalancedReplay": TaskBalancedReplayTrainer,
    "ContinualBP": ContinualBackpropTrainer,
    "ReplayContinualBP": ReplayContinualBackpropTrainer,
    "SustainableReplayEWC": SustainableReplayEWCTrainer,
    "BennaFusi": BennaFusiTrainer,
    "BennaFusiReplay": BennaFusiReplayTrainer,
    "FunctionSpaceReplay": FunctionSpaceReplayTrainer,
}
