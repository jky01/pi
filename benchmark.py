"""
Permuted-Pi-Digits：用 pi 數位序列建構的持續學習 (continual learning) benchmark。

設計動機（對應對話中 Stage C「長串流可塑性探針」的精神）：
- 輸入 x：連續 K 位 pi 小數位，one-hot 編碼（維度 10*K）。pi 的小數位在統計上近似
  均勻、獨立分布、沒有已知短週期重複，拿來當輸入特徵的來源，等同一個決定性、
  可重現、但「近乎不重複」的擬隨機資料產生器（概念上對應 Dohare et al. 的
  bit-flipping / slowly-changing regression 用隨機位元當輸入的做法）。

- 重要的修正（第一版設計的錯誤）：一開始曾經嘗試「用前 K 位去預測下一位 pi 數字」，
  但 pi 的小數位之間沒有可被學習的統計依賴（這正是 pi 數位近似均勻獨立分布的意涵），
  所以那個目標本質上不可學習，準確率會卡在 1/10 的純亂猜水準，沒辦法用來驗證任何
  持續學習演算法。

- 修正後的設計：目標改成輸入「窗口內 K 位數字總和」分桶後的類別（10 個桶、桶界用
  大樣本分位數決定，所以是 pi 小數位裡客觀存在、可被 MLP 學到的非線性函數），
  再對每個 task 套用一個固定的隨機 permutation 重新標號。這跟 Permuted-MNIST
  是同一種 domain-incremental 設計：底層「該怎麼從輸入算出基礎類別」這件事每個
  task 都一樣（共享特徵），但「基礎類別要對應到哪個輸出標籤」每個 task 不同——
  網路必須持續學會新的輸出映射，而舊映射會被新映射蓋過，同時拷問可塑性流失
  （新 task 學不學得動）與遺忘（舊 task 還記不記得）。

- conflicting 模式：仍是多頭 Task-IL，但每個 task 輪換不同的底層 input->class 函數
  （sum、weighted sum、半窗 sum、相鄰乘積等），讓共享層真的面對相互衝突的特徵需求。

- 每個 task 區塊額外保留一段「測試窗口」，訓練時不會看到，用來算 accuracy matrix。
"""
import numpy as np


CONFLICTING_RULES = (
    "sum",
    "weighted_sum",
    "first_half_sum",
    "second_half_sum",
    "adjacent_product_sum",
)


def _windows_sum(digits: np.ndarray, K: int, start_idx: int, n: int) -> np.ndarray:
    sums = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sums[i] = digits[start_idx + i: start_idx + i + K].sum()
    return sums


class PermutedPiDigitsStream:
    def __init__(self, digits: str, K: int, n_tasks: int, steps_per_task: int,
                 test_per_task: int, seed: int = 0, n_classes: int = 10, mode: str = "label_permuted"):
        self.digits = np.array([int(c) for c in digits], dtype=np.int64)
        self.K = K
        self.n_tasks = n_tasks
        self.steps_per_task = steps_per_task
        self.test_per_task = test_per_task
        self.n_classes = n_classes
        self.mode = mode
        self.block_len = steps_per_task + test_per_task

        needed = K + n_tasks * self.block_len
        if len(self.digits) < needed:
            raise ValueError(f"digits 不夠長：需要 {needed}，只有 {len(self.digits)}")

        rng = np.random.RandomState(seed)
        self.permutations = [rng.permutation(n_classes) for _ in range(n_tasks)]
        if mode == "input_permuted":
            # For input_permuted, input dimensions are permuted, and labels are NOT permuted (use identity)
            self.input_permutations = [rng.permutation(10 * K) for _ in range(n_tasks)]
            self.permutations = [np.arange(n_classes) for _ in range(n_tasks)]
        elif mode == "conflicting":
            # Multi-head Task-IL, but the underlying input->class rule changes by task.
            # Keep label ids unpermuted so the measured conflict comes from shared features,
            # not from output remapping.
            self.input_permutations = None
            self.permutations = [np.arange(n_classes) for _ in range(n_tasks)]
            self.conflict_rule_ids = [t % len(CONFLICTING_RULES) for t in range(n_tasks)]
        else:
            self.input_permutations = None

        self.train_starts = []
        self.test_starts = []
        pos = 0
        for t in range(n_tasks):
            self.train_starts.append(pos)
            self.test_starts.append(pos + steps_per_task)
            pos += self.block_len

        # 用一段獨立於所有 train/test 區段之外的尾段資料，估計分桶用的分位數門檻，
        # 確保基礎類別（分桶結果）在這個資料集上近似均勻分布，且門檻不是從訓練/測試資料本身洩漏出來的。
        calib_start = pos
        calib_n = min(20000, len(self.digits) - calib_start - K)
        if calib_n < 1000:
            calib_start = 0
            calib_n = min(20000, len(self.digits) - K)
        calib_sums = _windows_sum(self.digits, K, calib_start, calib_n)
        qs = np.linspace(0, 1, n_classes + 1)[1:-1]
        self.thresholds = np.quantile(calib_sums, qs)

        if mode == "conflicting":
            calib_windows = self._window_matrix(calib_start, calib_n)
            self.conflict_thresholds = []
            for rule_id in range(len(CONFLICTING_RULES)):
                scores = self._conflicting_scores(calib_windows, rule_id)
                self.conflict_thresholds.append(np.quantile(scores, qs))

        if mode == "class_il":
            self._build_class_il(K, n_classes, n_tasks, steps_per_task, test_per_task, calib_sums)

    def _window_matrix(self, start_idx: int, n: int) -> np.ndarray:
        return np.stack([self.digits[start_idx + i: start_idx + i + self.K] for i in range(n)])

    def _conflicting_scores(self, window_mat: np.ndarray, rule_id: int) -> np.ndarray:
        rule = CONFLICTING_RULES[rule_id % len(CONFLICTING_RULES)]
        K = window_mat.shape[1]
        if rule == "sum":
            return window_mat.sum(axis=1)
        if rule == "weighted_sum":
            weights = np.arange(1, K + 1, dtype=np.float64)
            return window_mat @ weights
        if rule == "first_half_sum":
            split = max(1, K // 2)
            return window_mat[:, :split].sum(axis=1)
        if rule == "second_half_sum":
            split = max(1, K // 2)
            return window_mat[:, split:].sum(axis=1)
        if rule == "adjacent_product_sum":
            if K < 2:
                return window_mat[:, 0]
            return (window_mat[:, :-1] * window_mat[:, 1:]).sum(axis=1)
        raise ValueError(f"unknown conflicting rule: {rule}")

    def _build_class_il(self, K, n_classes, n_tasks, steps_per_task, test_per_task, calib_sums):
        """Well-posed Class-IL: a global label is the fine-grained (equal-frequency) bucket
        of the window sum, so it is an unambiguous function of the input (no task id needed).
        Task t owns a disjoint, contiguous slice of those global buckets — i.e. a distinct
        sum-range — so tasks are naturally input-distinguishable. We scan the whole stream,
        assign each window to its global bucket, and hand each task the windows in its slice."""
        n_global = n_classes * n_tasks
        qs_g = np.linspace(0, 1, n_global + 1)[1:-1]
        self.global_thresholds = np.quantile(calib_sums, qs_g)
        scan_n = len(self.digits) - K
        sums = _windows_sum(self.digits, K, 0, scan_n)
        gbin = np.searchsorted(self.global_thresholds, sums, side="right")  # 0..n_global-1
        self.class_il_train_idx = []
        self.class_il_test_idx = []
        need = steps_per_task + test_per_task
        for t in range(n_tasks):
            lo, hi = n_classes * t, n_classes * (t + 1)
            starts = np.where((gbin >= lo) & (gbin < hi))[0]
            if len(starts) < need:
                raise ValueError(
                    f"class_il task {t}: 只有 {len(starts)} 個窗口，需要 {need}。"
                    f" K={K} 的位數和只有 ~{9*K+1} 個相異值，無法切成 {n_global} 個非空細桶"
                    f"（尾端桶被掏空）。請降低 n_tasks 或調大 K（distinct sums ≫ n_classes*n_tasks）。")
            # train from the front, test from the tail (far apart → minimal window overlap)
            self.class_il_train_idx.append(starts[:steps_per_task])
            self.class_il_test_idx.append(starts[-test_per_task:])

    def _bucket(self, sums: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.thresholds, sums, side="right").astype(np.int64)

    def _conflicting_bucket(self, scores: np.ndarray, task_idx: int) -> np.ndarray:
        rule_id = self.conflict_rule_ids[task_idx]
        return np.searchsorted(self.conflict_thresholds[rule_id], scores, side="right").astype(np.int64)

    def _make_xy(self, start_idx: int, n: int, task_idx: int):
        """從數位序列位置 start_idx 開始，產生 n 筆 (x_onehot, y) 樣本（滑動窗口，step=1）。"""
        K = self.K
        window_mat = self._window_matrix(start_idx, n)
        X = np.zeros((n, 10 * K), dtype=np.float32)
        rows = np.repeat(np.arange(n), K)
        cols_k = np.tile(np.arange(K), n)
        X3 = X.reshape(n, K, 10)
        X3[rows, cols_k, window_mat.reshape(-1)] = 1.0
        
        if self.mode == "input_permuted":
            X = X[:, self.input_permutations[task_idx]]

        if self.mode == "conflicting":
            rule_id = self.conflict_rule_ids[task_idx]
            base_class = self._conflicting_bucket(self._conflicting_scores(window_mat, rule_id), task_idx)
        else:
            base_class = self._bucket(window_mat.sum(axis=1))
        Y = self.permutations[task_idx][base_class]
        return X, Y

    def _make_xy_idx(self, starts: np.ndarray):
        """Class-IL: build (x_onehot, global_label) from arbitrary window start indices.
        Label is the global fine-grained bucket of the window sum — no permutation, no task id."""
        K = self.K
        n = len(starts)
        window_mat = np.stack([self.digits[s:s + K] for s in starts])
        X = np.zeros((n, 10 * K), dtype=np.float32)
        rows = np.repeat(np.arange(n), K)
        cols_k = np.tile(np.arange(K), n)
        X3 = X.reshape(n, K, 10)
        X3[rows, cols_k, window_mat.reshape(-1)] = 1.0
        Y = np.searchsorted(self.global_thresholds, window_mat.sum(axis=1), side="right").astype(np.int64)
        return X, Y

    def get_train_batch_stream(self, task_idx: int):
        """回傳該 task 訓練段的全部 (x, y)，依序（線上單次通過）。"""
        if self.mode == "class_il":
            return self._make_xy_idx(self.class_il_train_idx[task_idx])
        start = self.train_starts[task_idx]
        return self._make_xy(start, self.steps_per_task, task_idx)

    def get_test_set(self, task_idx: int):
        """回傳該 task 的固定測試集（訓練時沒看過的窗口）。"""
        if self.mode == "class_il":
            return self._make_xy_idx(self.class_il_test_idx[task_idx])
        start = self.test_starts[task_idx]
        return self._make_xy(start, self.test_per_task, task_idx)


if __name__ == "__main__":
    with open("pi_digits_600000.txt") as f:
        digits = f.read().strip()
    stream = PermutedPiDigitsStream(digits, K=8, n_tasks=80, steps_per_task=1500,
                                     test_per_task=300, seed=0)
    print("bucket thresholds:", stream.thresholds)
    X, Y = stream.get_train_batch_stream(0)
    Xt, Yt = stream.get_test_set(0)
    print("train batch:", X.shape, Y.shape, "label dist:", np.bincount(Y, minlength=10))
    print("test batch:", Xt.shape, Yt.shape)
    print("task0 permutation:", stream.permutations[0])
