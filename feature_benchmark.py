"""
P8：在 frozen pretrained 特徵上的 Class-IL 串流（Split-CIFAR-100）。

介面與 PermutedPiDigitsStream / MNISTStream 完全一致，重用 MLP 與全部 trainers。
資料來自 extract_features.py 抽好的 .npz（CIFAR-100 的 ResNet18 特徵）。

設定：Split-CIFAR-100 = n_tasks 個 task，每個 task 擁有 classes_per_task 個**全域**
類別（disjoint、連續切片），單頭、推論不給 task id（真 Class-IL）。標籤即全域
CIFAR class（0..n_tasks*classes_per_task-1）。
"""
import numpy as np


class FeatureSplitStream:
    def __init__(self, feat_npz: str, n_tasks: int, classes_per_task: int,
                 steps_per_task: int, test_per_task: int, seed: int = 0,
                 standardize: bool = True):
        d = np.load(feat_npz)
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]

        if standardize:
            # 用 train 統計量做 z-score（frozen 特徵尺度差異大，標準化讓小 head 好學）。
            mu = Xtr.mean(axis=0, keepdims=True)
            sd = Xtr.std(axis=0, keepdims=True) + 1e-6
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd

        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task
        self.n_classes = n_tasks * classes_per_task  # 全域類別數
        self.mode = "class_il"
        self.in_dim = Xtr.shape[1]
        self.steps_per_task = steps_per_task
        self.test_per_task = test_per_task

        self.Xtr, self.ytr, self.Xte, self.yte = (
            Xtr.astype(np.float32), ytr.astype(np.int64),
            Xte.astype(np.float32), yte.astype(np.int64))

        rng = np.random.RandomState(seed)
        self.task_classes = [tuple(range(classes_per_task * t, classes_per_task * (t + 1)))
                             for t in range(n_tasks)]
        self.train_idx, self.test_idx = [], []
        for classes in self.task_classes:
            mask_tr = np.isin(self.ytr, classes)
            mask_te = np.isin(self.yte, classes)
            tr = rng.permutation(np.where(mask_tr)[0])
            te = np.where(mask_te)[0]
            if steps_per_task < len(tr):
                tr = tr[:steps_per_task]
            if test_per_task < len(te):
                te = rng.permutation(te)[:test_per_task]
            self.train_idx.append(tr)
            self.test_idx.append(te)

    def get_train_batch_stream(self, task_idx: int):
        idx = self.train_idx[task_idx]
        return self.Xtr[idx], self.ytr[idx]

    def get_test_set(self, task_idx: int):
        idx = self.test_idx[task_idx]
        return self.Xte[idx], self.yte[idx]


if __name__ == "__main__":
    s = FeatureSplitStream("cifar100_resnet18.npz", n_tasks=20, classes_per_task=5,
                           steps_per_task=200, test_per_task=100, seed=0)
    X, Y = s.get_train_batch_stream(0)
    Xt, Yt = s.get_test_set(19)
    print("in_dim:", s.in_dim, "n_classes:", s.n_classes)
    print("task0 train:", X.shape, "classes:", np.unique(Y))
    print("task19 test:", Xt.shape, "classes:", np.unique(Yt))
