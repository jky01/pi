"""
多種持續學習策略，皆共用同一個 MLP (model.py)，差別只在「怎麼更新權重」：

1. NaiveTrainer            — 下界。單純線上 SGD，什麼保護機制都沒有。
2. EWCTrainer               — Elastic Weight Consolidation（正則化派代表）。
3. ReplayTrainer            — Experience Replay，小型 reservoir buffer（重播派代表）。
4. ReplayEWCTrainer         — Replay + online EWC，結合樣本重播與參數保護。
5. DarkReplayEWCTrainer     — ReplayEWC + logits consistency（DER/SER 系列方向）。
6. SurpriseReplayEWCTrainer — ReplayEWC + loss/surprise-prioritized replay sampling。
7. MarginSurpriseReplayEWCTrainer
                            — ReplayEWC + loss/surprise + low-margin boundary replay。
8. HippocampalReplayEWCTrainer
                            — SurpriseReplayEWC + episodic prototype memory at inference。
9. ContinualBackpropTrainer — Sutton/Dohare 的 continual backprop：只選擇性地
                              重置「低效用、夠老」的死/低貢獻單元，其餘權重完全
                              不動——這是對話第一輪明確回答「不重置權重」的機制，
                              主打可塑性流失（失效 B），跟前兩者主打遺忘（失效 A）形成對照。

所有 trainer 都實作同一介面：
    train_step(X, Y) -> (loss, acc)   在這個 batch 上更新一次參數，回傳更新前的 loss/acc
    on_task_end(task_X, task_Y)       task 訓練段結束時呼叫的 hook（EWC 用來算 Fisher）
"""
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

        rcache = self.model.forward(rX)
        rgrads = self.model.backward(rcache, rY)
        return {k: 0.5 * grads[k] + 0.5 * rgrads[k] for k in grads}

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
                 replay_weight: float = 0.5):
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

    def _dark_grads(self, cache, target_logits, task_idx):
        n, out_dim = target_logits.shape
        dlogits = 2.0 * (cache["logits"] - target_logits) / max(1, n * out_dim)
        return self.model.backward_from_logits_grad(cache, dlogits, task_idx)

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
                dark_grads = self._dark_grads(sub_cache, sub_logits, t)
                sub_grads = self._add_grads(ce_grads, dark_grads, self.dark_alpha)

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

        rcache = self.model.forward(rX)
        ce_grads = self.model.backward(rcache, rY)
        dark_grads = self._dark_grads(rcache, rlogits, None)
        rgrads = self._add_grads(ce_grads, dark_grads, self.dark_alpha)
        return {k: current_weight * grads[k] + replay_weight * rgrads[k] for k in grads}

    def train_step(self, X, Y, task_idx: int = None):
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

        if self.model.multi_head:
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
        use_context = self.model.multi_head or self.memory_task_filter
        if not use_context:
            return list(range(len(self.buf_X)))
        if task_idx is None:
            return []
        return [i for i, t in enumerate(self.buf_task) if t == task_idx]

    def _feature_cache(self, X, task_idx):
        if self.model.multi_head:
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
        cache = self.model.forward(X, task_idx if self.model.multi_head else None)
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


TRAINER_REGISTRY = {
    "Naive": NaiveTrainer,
    "EWC": EWCTrainer,
    "Replay": ReplayTrainer,
    "ReplayEWC": ReplayEWCTrainer,
    "DarkReplayEWC": DarkReplayEWCTrainer,
    "SurpriseReplayEWC": SurpriseReplayEWCTrainer,
    "MarginSurpriseReplayEWC": MarginSurpriseReplayEWCTrainer,
    "HippocampalReplayEWC": HippocampalReplayEWCTrainer,
    "TaskBalancedReplay": TaskBalancedReplayTrainer,
    "ContinualBP": ContinualBackpropTrainer,
    "ReplayContinualBP": ReplayContinualBackpropTrainer,
}
