"""
MNIST 上的標準持續學習 benchmark，介面與 PermutedPiDigitsStream 完全一致
（n_tasks / n_classes / get_train_batch_stream / get_test_set），讓 run.py 的
訓練迴圈、所有 trainers、MLP 都能原封不動重用。

目的：把專案在 Permuted-Pi-Digits 上的兩個核心結論搬到 CL 社群的標準 benchmark
做外部效度驗證——
- mode="permuted"（多頭 Task-IL，Permuted-MNIST）：每個 task 用固定的像素 permutation，
  共享隱藏層必須跨大量 task 保留可塑性/記憶。對應驗證「DER++ logit 蒸餾的函數錨定」。
- mode="split"（Class-IL，Split-MNIST）：5 個 task，task t 擁有 digit {2t, 2t+1}，
  單頭、全域 10 類、推論不給 task id。對應驗證「NCM 原型讀出修好線性頭 recency bias」。
"""
import numpy as np

from mnist_data import load_mnist


class MNISTStream:
    def __init__(self, n_tasks: int, steps_per_task: int, test_per_task: int,
                 seed: int = 0, mode: str = "permuted", n_classes: int = 10):
        self.n_tasks = n_tasks
        self.steps_per_task = steps_per_task   # = 每個 task 提供的 train 樣本數（run 迴圈以 batch_size 切塊）
        self.test_per_task = test_per_task
        self.mode = mode
        self.n_classes = n_classes
        self.in_dim = 784

        Xtr, ytr, Xte, yte = load_mnist()
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        rng = np.random.RandomState(seed)

        if mode == "permuted":
            if n_tasks > 1:
                # task 0 用 identity permutation（= 原始 MNIST），其餘為隨機像素重排。
                self.permutations = [np.arange(784)] + [rng.permutation(784) for _ in range(n_tasks - 1)]
            else:
                self.permutations = [np.arange(784)]
            # 每個 task 抽一份固定的 train / test 子集（index 預先決定，確保可重現）。
            self.train_idx = [rng.choice(len(Xtr), size=steps_per_task, replace=steps_per_task > len(Xtr))
                              for _ in range(n_tasks)]
            self.test_idx = [rng.choice(len(Xte), size=min(test_per_task, len(Xte)), replace=False)
                             for _ in range(n_tasks)]
        elif mode == "split":
            if n_tasks * 2 > 10:
                raise ValueError(f"split-MNIST 最多 5 個 task（每 task 2 類），收到 n_tasks={n_tasks}")
            self.task_classes = [(2 * t, 2 * t + 1) for t in range(n_tasks)]
            self.train_idx = []
            self.test_idx = []
            for (c0, c1) in self.task_classes:
                tr = np.where((ytr == c0) | (ytr == c1))[0]
                te = np.where((yte == c0) | (yte == c1))[0]
                tr = rng.permutation(tr)
                if steps_per_task < len(tr):
                    tr = tr[:steps_per_task]
                if test_per_task < len(te):
                    te = rng.permutation(te)[:test_per_task]
                self.train_idx.append(tr)
                self.test_idx.append(te)
        else:
            raise ValueError(f"unknown MNIST mode: {mode}")

    def _apply(self, X, task_idx):
        if self.mode == "permuted":
            return X[:, self.permutations[task_idx]]
        return X

    def get_train_batch_stream(self, task_idx: int):
        idx = self.train_idx[task_idx]
        if self.mode == "permuted":
            X = self._apply(self.Xtr[idx], task_idx)
            Y = self.ytr[idx]
        else:  # split：標籤即全域 digit class（class_il，無 task id）
            X = self.Xtr[idx]
            Y = self.ytr[idx]
        return X.astype(np.float32), Y.astype(np.int64)

    def get_test_set(self, task_idx: int):
        if self.mode == "permuted":
            idx = self.test_idx[task_idx]
            X = self._apply(self.Xte[idx], task_idx)
            Y = self.yte[idx]
        else:
            idx = self.test_idx[task_idx]
            X = self.Xte[idx]
            Y = self.yte[idx]
        return X.astype(np.float32), Y.astype(np.int64)


if __name__ == "__main__":
    for mode, nt in [("permuted", 5), ("split", 5)]:
        s = MNISTStream(n_tasks=nt, steps_per_task=4000, test_per_task=1000, seed=0, mode=mode)
        X, Y = s.get_train_batch_stream(0)
        Xt, Yt = s.get_test_set(nt - 1)
        print(f"[{mode}] train task0:", X.shape, "labels:", np.unique(Y),
              "| test last task:", Xt.shape, "labels:", np.unique(Yt))
