"""
純 numpy 實作的小型 MLP（2 個隱藏層 + softmax 輸出），手動 forward/backward。
不依賴 torch，方便在任意環境下穩定執行。

附帶兩個對話中反覆提到、且本實驗會用到的診斷量：
- effective rank（有效秩）：量測某層表徵的「內在維度」是否塌縮（可塑性流失的徵兆）。
- dead unit fraction（死神經元比例）：ReLU 單元對整批輸入都輸出 0 的比例。
"""
import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def effective_rank(activations: np.ndarray) -> float:
    """activations: (batch, units)。回傳奇異值分布的指數熵 = 有效秩。"""
    if activations.shape[0] < 2:
        return float("nan")
    A = activations - activations.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def dead_unit_fraction(activations: np.ndarray) -> float:
    """activations: (batch, units)，ReLU 後的輸出。回傳整批皆為 0 的單元比例。"""
    return float(np.mean(np.all(activations <= 1e-12, axis=0)))


class MLP:
    """2 個隱藏層的 MLP：in_dim -> h1 -> h2 -> out_dim，ReLU + softmax + cross entropy。"""

    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int, seed: int = 0, multi_head: bool = False, n_tasks: int = 80):
        rng = np.random.RandomState(seed)
        self.dims = (in_dim, h1, h2, out_dim)
        self.W1 = self._he_init(rng, in_dim, h1)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = self._he_init(rng, h1, h2)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.multi_head = multi_head
        if multi_head:
            self.heads_W = [self._he_init(rng, h2, out_dim) for _ in range(n_tasks)]
            self.heads_b = [np.zeros(out_dim, dtype=np.float32) for _ in range(n_tasks)]
            self.W3 = None
            self.b3 = None
        else:
            self.W3 = self._he_init(rng, h2, out_dim)
            self.b3 = np.zeros(out_dim, dtype=np.float32)
            self.heads_W = None
            self.heads_b = None
        self.rng = rng

    @staticmethod
    def _he_init(rng, fan_in, fan_out):
        std = np.sqrt(2.0 / fan_in)
        return (rng.randn(fan_in, fan_out) * std).astype(np.float32)

    def params(self, task_idx=None):
        if self.multi_head:
            if task_idx is None:
                pts = [self.W1, self.b1, self.W2, self.b2]
                for w, b in zip(self.heads_W, self.heads_b):
                    pts.extend([w, b])
                return pts
            else:
                return [self.W1, self.b1, self.W2, self.b2, self.heads_W[task_idx], self.heads_b[task_idx]]
        else:
            return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray, task_idx: int = None) -> dict:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        if self.multi_head:
            if task_idx is None:
                raise ValueError("multi_head model requires task_idx for forward pass")
            W3 = self.heads_W[task_idx]
            b3 = self.heads_b[task_idx]
        else:
            W3 = self.W3
            b3 = self.b3
        logits = a2 @ W3 + b3
        probs = softmax(logits)
        return dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2, logits=logits, probs=probs)

    def loss_acc(self, X: np.ndarray, Y: np.ndarray, task_idx: int = None) -> tuple:
        cache = self.forward(X, task_idx)
        probs = cache["probs"]
        n = X.shape[0]
        ll = -np.log(probs[np.arange(n), Y] + 1e-12)
        loss = float(ll.mean())
        preds = probs.argmax(axis=1)
        acc = float((preds == Y).mean())
        return loss, acc

    def backward_from_logits_grad(self, cache: dict, dlogits: np.ndarray, task_idx: int = None) -> dict:
        """從已給定的 logits 梯度往回傳，回傳每個參數的梯度。"""
        if self.multi_head:
            if task_idx is None:
                raise ValueError("multi_head model requires task_idx for backward pass")
            W3 = self.heads_W[task_idx]
            W2 = self.W2
        else:
            W3 = self.W3
            W2 = self.W2

        a2 = cache["a2"]
        gW3 = a2.T @ dlogits
        gb3 = dlogits.sum(axis=0)

        da2 = dlogits @ W3.T
        dz2 = da2 * (cache["z2"] > 0)
        a1 = cache["a1"]
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * (cache["z1"] > 0)
        X = cache["X"]
        gW1 = X.T @ dz1
        gb1 = dz1.sum(axis=0)

        return dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=gb3)

    def backward(self, cache: dict, Y: np.ndarray, task_idx: int = None) -> dict:
        """回傳每個參數的 cross-entropy 梯度（mean over batch）。"""
        n = Y.shape[0]
        probs = cache["probs"]
        dlogits = probs.copy()
        dlogits[np.arange(n), Y] -= 1.0
        dlogits /= n  # (n, out_dim)
        return self.backward_from_logits_grad(cache, dlogits, task_idx)

    def sgd_step(self, grads: dict, lr: float, extra_grads: dict = None, task_idx: int = None):
        if extra_grads is not None:
            grads = {k: grads[k] + extra_grads.get(k, 0.0) for k in grads}
        self.W1 -= lr * grads["W1"]; self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]; self.b2 -= lr * grads["b2"]
        if self.multi_head:
            if task_idx is None:
                raise ValueError("multi_head model requires task_idx for sgd_step")
            self.heads_W[task_idx] -= lr * grads["W3"]
            self.heads_b[task_idx] -= lr * grads["b3"]
        else:
            self.W3 -= lr * grads["W3"]
            self.b3 -= lr * grads["b3"]

    def weight_norm(self) -> float:
        return float(sum(np.linalg.norm(p) for p in self.params()))

    def diagnostics(self, X: np.ndarray, task_idx: int = None) -> dict:
        cache = self.forward(X, task_idx)
        return dict(
            eff_rank_h1=effective_rank(cache["a1"]),
            eff_rank_h2=effective_rank(cache["a2"]),
            dead_frac_h1=dead_unit_fraction(cache["a1"]),
            dead_frac_h2=dead_unit_fraction(cache["a2"]),
            weight_norm=self.weight_norm(),
        )


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    m = MLP(80, 64, 64, 10, seed=0)
    X = rng.randn(32, 80).astype(np.float32)
    Y = rng.randint(0, 10, size=32)
    loss, acc = m.loss_acc(X, Y)
    print("init loss", loss, "acc", acc)
    cache = m.forward(X)
    grads = m.backward(cache, Y)
    m.sgd_step(grads, lr=0.1)
    loss2, acc2 = m.loss_acc(X, Y)
    print("after one step loss", loss2, "acc", acc2)
    print(m.diagnostics(X))
