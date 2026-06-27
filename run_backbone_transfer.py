"""
P8b：unfreeze backbone，檢驗「表徵必須被建構」時是否出現正向遷移 / 累積。

P10 在 frozen ImageNet backbone 上量到正向遷移≈0，但限制是 backbone 已封頂、
表徵不需要被建構。P8b 改用一個**從零開始的小 CNN，backbone 跨 task 持續適應**，
這才是表徵會被逐步建立、累積學習有機會出現的 regime（也最貼近 LLM 持續微調）。

探針（沿用 P10，但 backbone 改成會動的）：在 Split-CIFAR-100（原始影像）流上，
量「task-k 受限 5-way acc」隨步數的曲線，比較：
- 持續模型：一個小 CNN 依序訓練 task 0..k（backbone 累積結構），plain SGD（Naive-
  continual，隔離純表徵累積、不加抗遺忘 confound）；
- fresh：全新隨機 CNN 只學 task k。
若「會動的表徵」真能累積 → 持續模型學新 task 越來越快（Δ 隨 k 增長）。

只用 torch 做這支實驗；其餘專案維持純 numpy。
"""
import argparse
import copy
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_num_threads(torch.get_num_threads())
CHECKPOINTS = [0, 5, 10, 20, 40, 80, 160]
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():      # NVIDIA GPU (e.g. RTX 2070)
            return "cuda"
        if torch.backends.mps.is_available():  # Apple Silicon GPU
            return "mps"
        return "cpu"
    return name


def load_cifar100_raw(root="./cifar_data/cifar-100-python"):
    def unp(fn):
        with open(os.path.join(root, fn), "rb") as f:
            return pickle.load(f, encoding="latin1")
    tr, te = unp("train"), unp("test")
    Xtr = tr["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    Xte = te["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    ytr = np.array(tr["fine_labels"], dtype=np.int64)
    yte = np.array(te["fine_labels"], dtype=np.int64)
    return Xtr, ytr, Xte, yte


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class SmallCNN(nn.Module):
    """從零訓練的小 CNN：3 個 conv block → 128-d 特徵 → 線性頭。"""
    def __init__(self, n_classes=100, width=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 32->16
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, n_classes)

    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.head(z)


def make_resnet18_cifar(n_classes):
    """CIFAR-adapted ResNet18（從零）：把 7x7/stride2 stem 換成 3x3/stride1、移除 early
    maxpool，否則 32x32 輸入會被過度下採樣。這是 CIFAR ResNet 的標準改法。"""
    net = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def make_model(arch, n_classes, width):
    if arch == "smallcnn":
        return SmallCNN(n_classes=n_classes, width=width)
    if arch == "resnet18":
        return make_resnet18_cifar(n_classes)
    raise ValueError(f"unknown arch: {arch}")


def acc_5way(model, Xt, yt, task_classes):
    model.eval()
    with torch.no_grad():
        logits = model(normalize(Xt))
        cols = torch.tensor(task_classes, device=logits.device)
        sub = logits[:, cols]
        pred = cols[sub.argmax(1)]
    model.train()
    return (pred == yt).float().mean().item()


class ReservoirBuffer:
    """跨 task 的 reservoir replay buffer（存原始 [0,1] 影像 + 全域標籤 + 可選 logits，
    後者供 DER++ logit 蒸餾用，於插入時記錄當下模型輸出）。"""
    def __init__(self, capacity, rng):
        self.cap = capacity
        self.rng = rng
        self.X = None
        self.y = None
        self.Z = None  # stored logits (DER++)
        self.n_seen = 0
        self.size = 0

    def add(self, xb, yb, zb=None):
        if self.X is None:
            self.X = torch.zeros((self.cap,) + xb.shape[1:], dtype=xb.dtype, device=xb.device)
            self.y = torch.zeros(self.cap, dtype=yb.dtype, device=xb.device)
            if zb is not None:
                self.Z = torch.zeros((self.cap,) + zb.shape[1:], dtype=zb.dtype, device=zb.device)
        for i in range(xb.shape[0]):
            slot = -1
            if self.size < self.cap:
                slot = self.size; self.size += 1
            else:
                j = self.rng.randint(0, self.n_seen + 1)
                if j < self.cap:
                    slot = j
            if slot >= 0:
                self.X[slot] = xb[i]; self.y[slot] = yb[i]
                if zb is not None and self.Z is not None:
                    self.Z[slot] = zb[i]
            self.n_seen += 1

    def sample(self, m):
        m = min(m, self.size)
        idx = self.rng.randint(0, self.size, size=m)
        z = self.Z[idx] if self.Z is not None else None
        return self.X[idx], self.y[idx], z


class LwF:
    """Learning without Forgetting：buffer-free 函數空間蒸餾（= DER++ 不存樣本）。
    每個 task 後凍結一份模型快照；訓練新 task 時，對**當前 batch 輸入**用舊快照在
    已學過的類別欄位上的 logits 做 MSE 蒸餾，完全不存原始樣本。"""
    def __init__(self, lam, classes_per_task):
        self.lam = lam
        self.cpt = classes_per_task
        self.old_model = None
        self.old_cols = None

    def extra_loss(self, model, cur_x, out, ncur):
        if self.old_model is None:
            return 0.0
        with torch.no_grad():
            old = self.old_model(cur_x)
        c = self.old_cols
        return self.lam * F.mse_loss(out[:ncur][:, c], old[:, c])

    def after_task(self, model, k):
        self.old_model = copy.deepcopy(model).eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
        self.old_cols = list(range(self.cpt * (k + 1)))  # 已學過的類別


class LwFKD:
    """經典 LwF：softmax knowledge-distillation + 溫度（P9 用的是 DER 式 logit-MSE）。
    §23.2 的 caveat 是「softmax-KD 可能較不致凍結」；這支做乾淨對照。蒸餾項在已學過
    的類別欄位上對舊快照的 soften 分布做 KL，完全 buffer-free。"""
    def __init__(self, lam, classes_per_task, T=2.0):
        self.lam = lam
        self.cpt = classes_per_task
        self.T = T
        self.old_model = None
        self.old_cols = None

    def extra_loss(self, model, cur_x, out, ncur):
        if self.old_model is None:
            return 0.0
        with torch.no_grad():
            old = self.old_model(cur_x)
        c = self.old_cols
        T = self.T
        log_p = F.log_softmax(out[:ncur][:, c] / T, dim=1)
        q = F.softmax(old[:, c] / T, dim=1)
        return self.lam * (T * T) * F.kl_div(log_p, q, reduction="batchmean")

    def after_task(self, model, k):
        self.old_model = copy.deepcopy(model).eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)
        self.old_cols = list(range(self.cpt * (k + 1)))


def train_curve(model, opt, X, y, Xt, yt, task_classes, epochs, batch, rng,
                mode="naive", buffer=None, dark_alpha=0.5, reg=None):
    """訓練一個 task 並記錄 task-k 受限 5-way acc 曲線。
    mode: naive（純當前 batch）/ replay（+ reservoir CE）/ derpp（replay CE + logit 蒸餾）。"""
    n = X.shape[0]
    curve, s = {}, 0
    if 0 in CHECKPOINTS:
        curve[0] = acc_5way(model, Xt, yt, task_classes)
    todo = [c for c in CHECKPOINTS if c > 0]
    use_buf = buffer is not None and mode in ("replay", "derpp")
    for ep in range(epochs):
        order = rng.permutation(n) if ep > 0 else np.arange(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb_raw = X[idx]; yb = y[idx]
            cur_x = normalize(xb_raw); ncur = cur_x.shape[0]
            opt.zero_grad()
            if use_buf and buffer.size >= batch:
                rx, ry, rz = buffer.sample(batch)
                out = model(torch.cat([cur_x, normalize(rx)], 0))
                loss = F.cross_entropy(out[:ncur], yb) + F.cross_entropy(out[ncur:], ry)
                if mode == "derpp" and rz is not None:
                    loss = loss + dark_alpha * F.mse_loss(out[ncur:], rz)
                cur_logits = out[:ncur].detach()
            else:
                out = model(cur_x)
                loss = F.cross_entropy(out, yb)
                cur_logits = out.detach()
            if reg is not None:
                loss = loss + reg.extra_loss(model, cur_x, out, ncur)
            loss.backward(); opt.step()
            if use_buf:
                buffer.add(xb_raw, yb, cur_logits if mode == "derpp" else None)
            s += 1
            if todo and s == todo[0]:
                curve[s] = acc_5way(model, Xt, yt, task_classes)
                todo.pop(0)
        if not todo:
            break
    if todo:
        curve[s] = acc_5way(model, Xt, yt, task_classes)
    return curve


def run_one(seed, n_tasks, classes_per_task, train_per_class, test_per_class,
            lr, epochs, batch, width, continual_mode="naive", buffer_cap=2000,
            device="cpu", arch="smallcnn", dark_alpha=0.5, lwf_lambda=1.0, lwf_temp=2.0):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = load_cifar100_raw()
    rng = np.random.RandomState(seed)
    buffer = (ReservoirBuffer(buffer_cap, np.random.RandomState(7000 + seed))
              if continual_mode in ("replay", "derpp") else None)
    if continual_mode == "lwf":
        reg = LwF(lwf_lambda, classes_per_task)
    elif continual_mode == "lwf_kd":
        reg = LwFKD(lwf_lambda, classes_per_task, T=lwf_temp)
    else:
        reg = None

    task_classes = [tuple(range(classes_per_task * t, classes_per_task * (t + 1)))
                    for t in range(n_tasks)]
    # 每個 task 的 train/test index（子採樣以控制 CPU 成本）
    tr_idx, te_idx = [], []
    for classes in task_classes:
        tr = np.where(np.isin(ytr, classes))[0]
        te = np.where(np.isin(yte, classes))[0]
        tr = rng.permutation(tr)[:train_per_class * classes_per_task]
        tr_idx.append(tr); te_idx.append(te[:test_per_class * classes_per_task])

    cont = make_model(arch, n_tasks * classes_per_task, width).to(device)
    cont_opt = torch.optim.SGD(cont.parameters(), lr=lr, momentum=0.9)

    per_task = []
    test_sets = []  # 留到最後做 retention 評估（continual 終態 vs 剛學完）
    for k in range(n_tasks):
        X = torch.from_numpy(Xtr[tr_idx[k]]).to(device); y = torch.from_numpy(ytr[tr_idx[k]]).to(device)
        Xt = torch.from_numpy(Xte[te_idx[k]]).to(device); yt = torch.from_numpy(yte[te_idx[k]]).to(device)
        tcls = task_classes[k]
        test_sets.append((Xt, yt, tcls))

        rng_c = np.random.RandomState(1000 + seed)
        cont_curve = train_curve(cont, cont_opt, X, y, Xt, yt, tcls, epochs, batch, rng_c,
                                 mode=continual_mode, buffer=buffer, dark_alpha=dark_alpha,
                                 reg=reg)
        if reg is not None:
            reg.after_task(cont, k)

        fresh = make_model(arch, n_tasks * classes_per_task, width).to(device)
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=lr, momentum=0.9)
        rng_f = np.random.RandomState(1000 + seed)
        fresh_curve = train_curve(fresh, fresh_opt, X, y, Xt, yt, tcls, epochs, batch, rng_f,
                                  mode="naive")

        per_task.append(dict(task=k, continual=cont_curve, fresh=fresh_curve))
        print(f"  seed{seed} task{k:>2}: cont@40={cont_curve.get(40, float('nan')):.3f} "
              f"fresh@40={fresh_curve.get(40, float('nan')):.3f}", flush=True)

    # retention：用終態 continual 模型回評每個 task 的受限 5-way acc，對比剛學完時。
    maxc = max(CHECKPOINTS)
    diag = [at(per_task[j]["continual"], maxc) for j in range(n_tasks)]
    final = [acc_5way(cont, Xt, yt, tcls) for (Xt, yt, tcls) in test_sets]
    forgetting = [diag[j] - final[j] for j in range(n_tasks)]
    retention = dict(diag=diag, final=final,
                     mean_forgetting=float(np.mean(forgetting)),
                     mean_final=float(np.mean(final)))
    print(f"  seed{seed} retention: mean_final={retention['mean_final']:.3f} "
          f"mean_forgetting={retention['mean_forgetting']:.3f}", flush=True)
    return dict(per_task=per_task, retention=retention)


# ---------------------------------------------------------------------------
# P9b (a): Progressive Neural Network — buffer-free 累積。每個 task 一個 column；
# 訓練 column k 時，columns 0..k-1 全凍結，其 penultimate 特徵經一個 learned lateral
# 投影餵進 column k 的 head（PNN 的橫向連結）。新任務可「讀取」舊任務的凍結表徵 →
# 完全不存原始樣本就有機會出現正向遷移，且舊 column 永不更新 → 遺忘恆為 0（架構性）。
# 代價：容量隨 task 數線性成長（這正是 PNN 的已知 tradeoff）。
# ---------------------------------------------------------------------------
class FeatBackbone(nn.Module):
    """產生 penultimate 特徵向量（去掉分類頭）的 backbone，供 column 自身與 lateral 共用。"""
    def __init__(self, arch, width):
        super().__init__()
        if arch == "smallcnn":
            self.body = SmallCNN(n_classes=1, width=width).features
            self.out_dim = width * 2
        elif arch == "resnet18":
            net = make_resnet18_cifar(1)
            net.fc = nn.Identity()
            self.body = net
            self.out_dim = 512
        else:
            raise ValueError(f"unknown arch: {arch}")

    def forward(self, x):
        return self.body(x).flatten(1)


class PNNColumn(nn.Module):
    """單一 PNN column：自身 backbone + head；若 n_prior>0，head 額外讀取一個對
    前序凍結 column 特徵串接的 learned lateral 投影（ReLU）。n_prior=0 時退化成
    一個獨立 backbone+head（= fresh baseline 完全同構）。"""
    def __init__(self, arch, n_classes, width, n_prior, lateral_dim):
        super().__init__()
        self.backbone = FeatBackbone(arch, width)
        d = self.backbone.out_dim
        if n_prior > 0:
            self.lateral = nn.Sequential(nn.Linear(n_prior * d, lateral_dim), nn.ReLU())
            self.head = nn.Linear(d + lateral_dim, n_classes)
        else:
            self.lateral = None
            self.head = nn.Linear(d, n_classes)

    def forward(self, x, prior_feats):
        f = self.backbone(x)
        if self.lateral is not None and len(prior_feats) > 0:
            h = torch.cat([f, self.lateral(torch.cat(prior_feats, 1))], 1)
        else:
            h = f
        return self.head(h)


def acc_5way_pnn(col, frozen_bbs, Xt, yt, task_classes):
    col.eval()
    with torch.no_grad():
        nx = normalize(Xt)
        pf = [bb(nx) for bb in frozen_bbs]
        logits = col(nx, pf)
        cols = torch.tensor(task_classes, device=logits.device)
        pred = cols[logits[:, cols].argmax(1)]
    col.train()
    return (pred == yt).float().mean().item()


def train_pnn_column(col, frozen_bbs, opt, X, y, Xt, yt, task_classes, epochs, batch, rng):
    """訓練一個 PNN column（plain CE），記錄 task-k 受限 5-way acc 曲線。
    lateral 來源（frozen 前序 column）以 no_grad 前傳，不接收梯度。"""
    n = X.shape[0]
    curve, s = {}, 0
    if 0 in CHECKPOINTS:
        curve[0] = acc_5way_pnn(col, frozen_bbs, Xt, yt, task_classes)
    todo = [c for c in CHECKPOINTS if c > 0]
    for ep in range(epochs):
        order = rng.permutation(n) if ep > 0 else np.arange(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            cur_x = normalize(X[idx]); yb = y[idx]
            with torch.no_grad():
                pf = [bb(cur_x) for bb in frozen_bbs]
            opt.zero_grad()
            loss = F.cross_entropy(col(cur_x, pf), yb)
            loss.backward(); opt.step()
            s += 1
            if todo and s == todo[0]:
                curve[s] = acc_5way_pnn(col, frozen_bbs, Xt, yt, task_classes)
                todo.pop(0)
        if not todo:
            break
    if todo:
        curve[s] = acc_5way_pnn(col, frozen_bbs, Xt, yt, task_classes)
    return curve


def run_one_pnn(seed, n_tasks, classes_per_task, train_per_class, test_per_class,
                lr, epochs, batch, width, device="cpu", arch="smallcnn", lateral_dim=128):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = load_cifar100_raw()
    rng = np.random.RandomState(seed)
    task_classes = [tuple(range(classes_per_task * t, classes_per_task * (t + 1)))
                    for t in range(n_tasks)]
    tr_idx, te_idx = [], []
    for classes in task_classes:
        tr = np.where(np.isin(ytr, classes))[0]
        te = np.where(np.isin(yte, classes))[0]
        tr = rng.permutation(tr)[:train_per_class * classes_per_task]
        tr_idx.append(tr); te_idx.append(te[:test_per_class * classes_per_task])

    n_classes = n_tasks * classes_per_task
    frozen_bbs = []   # 前序 column 的凍結 backbone（lateral 來源）
    columns = []      # 保留每個 column 供 retention 評估
    per_task, test_sets = [], []
    for k in range(n_tasks):
        X = torch.from_numpy(Xtr[tr_idx[k]]).to(device); y = torch.from_numpy(ytr[tr_idx[k]]).to(device)
        Xt = torch.from_numpy(Xte[te_idx[k]]).to(device); yt = torch.from_numpy(yte[te_idx[k]]).to(device)
        tcls = task_classes[k]
        test_sets.append((Xt, yt, tcls))

        col = PNNColumn(arch, n_classes, width, n_prior=k, lateral_dim=lateral_dim).to(device)
        opt = torch.optim.SGD(col.parameters(), lr=lr, momentum=0.9)
        rng_c = np.random.RandomState(1000 + seed)
        cont_curve = train_pnn_column(col, frozen_bbs, opt, X, y, Xt, yt, tcls, epochs, batch, rng_c)
        col.backbone.eval()
        for p in col.backbone.parameters():
            p.requires_grad_(False)
        frozen_bbs.append(col.backbone)
        columns.append(col)

        # fresh baseline：與其他 mode 完全同構（獨立 backbone+head，只學 task k）
        fresh = make_model(arch, n_classes, width).to(device)
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=lr, momentum=0.9)
        rng_f = np.random.RandomState(1000 + seed)
        fresh_curve = train_curve(fresh, fresh_opt, X, y, Xt, yt, tcls, epochs, batch, rng_f, mode="naive")

        per_task.append(dict(task=k, continual=cont_curve, fresh=fresh_curve))
        print(f"  seed{seed} task{k:>2}: cont@40={cont_curve.get(40, float('nan')):.3f} "
              f"fresh@40={fresh_curve.get(40, float('nan')):.3f}", flush=True)

    # retention：column j 在 task j 後即凍結 → 用 column j（laterals 取前序 0..j-1）回評。
    maxc = max(CHECKPOINTS)
    diag = [at(per_task[j]["continual"], maxc) for j in range(n_tasks)]
    final = [acc_5way_pnn(columns[j], frozen_bbs[:j], Xt, yt, tcls)
             for j, (Xt, yt, tcls) in enumerate(test_sets)]
    forgetting = [diag[j] - final[j] for j in range(n_tasks)]
    retention = dict(diag=diag, final=final,
                     mean_forgetting=float(np.mean(forgetting)),
                     mean_final=float(np.mean(final)))
    print(f"  seed{seed} retention: mean_final={retention['mean_final']:.3f} "
          f"mean_forgetting={retention['mean_forgetting']:.3f} (PNN: 0 by construction)", flush=True)
    return dict(per_task=per_task, retention=retention)


def at(curve, step):
    ks = sorted(int(x) for x in curve.keys())
    best = ks[0]
    for x in ks:
        if x <= step:
            best = x
    return curve[best] if best in curve else curve[str(best)]


def summarize(all_seed, step):
    n_tasks = len(all_seed[0])
    rows = []
    for k in range(n_tasks):
        c = np.mean([at(s[k]["continual"], step) for s in all_seed])
        f = np.mean([at(s[k]["fresh"], step) for s in all_seed])
        rows.append((k, c, f, c - f))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--classes-per-task", type=int, default=5)
    p.add_argument("--train-per-class", type=int, default=200)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--continual-mode",
                   choices=["naive", "replay", "derpp", "lwf", "lwf_kd", "pnn"], default="naive")
    p.add_argument("--dark-alpha", type=float, default=0.5, help="DER++ logit distillation weight")
    p.add_argument("--lwf-lambda", type=float, default=1.0, help="LwF (buffer-free) distillation weight")
    p.add_argument("--lwf-temp", type=float, default=2.0, help="LwF softmax-KD temperature (lwf_kd)")
    p.add_argument("--lateral-dim", type=int, default=128, help="PNN lateral projection dim")
    p.add_argument("--arch", choices=["smallcnn", "resnet18"], default="smallcnn")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--output", default="results_backbone_transfer_cifar100.json")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    all_seed, retentions = [], []
    for seed in args.seeds:
        if args.continual_mode == "pnn":
            res = run_one_pnn(seed, args.n_tasks, args.classes_per_task,
                              args.train_per_class, args.test_per_class,
                              args.lr, args.epochs, args.batch, args.width,
                              device=device, arch=args.arch, lateral_dim=args.lateral_dim)
        else:
            res = run_one(seed, args.n_tasks, args.classes_per_task,
                          args.train_per_class, args.test_per_class,
                          args.lr, args.epochs, args.batch, args.width,
                          continual_mode=args.continual_mode, device=device,
                          arch=args.arch, dark_alpha=args.dark_alpha,
                          lwf_lambda=args.lwf_lambda, lwf_temp=args.lwf_temp)
        all_seed.append(res["per_task"])
        retentions.append(res["retention"])
        print(f"seed {seed} done", flush=True)

    print(f"\n=== P8b/P8c forward transfer with ADAPTING backbone (arch={args.arch}, "
          f"continual={args.continual_mode}) ===")
    print("task-k 5-way acc: continual (backbone adapts across tasks) vs fresh-from-scratch")
    for step in [10, 20, 40, 80]:
        rows = summarize(all_seed, step)
        c = np.mean([r[1] for r in rows]); f = np.mean([r[2] for r in rows])
        early = np.mean([d for k, _, _, d in rows if k < 5])
        late = np.mean([d for k, _, _, d in rows if k >= 15])
        overall = np.mean([d for _, _, _, d in rows])
        grow = late - early
        tag = ("GROWS w/ accumulation" if grow > 0.02 else
               ("FLAT" if abs(grow) <= 0.02 else "SHRINKS"))
        print(f"  @{step:>3} steps: continual {c:.3f} | fresh {f:.3f} | Δ {overall:+.3f} | "
              f"early {early:+.3f} -> late {late:+.3f} (cumulative {grow:+.3f} {tag})")

    rows = summarize(all_seed, 40)
    print("\nPer-task Δ (continual - fresh) @40 steps:")
    for k, c, f, d in rows:
        print(f" {k:>3} | cont {c:.3f} | fresh {f:.3f} | Δ {d:+.3f}")

    # retention（穩定–可塑性甜蜜點的另一軸）：continual 終態 vs 剛學完
    mf = np.mean([r["mean_final"] for r in retentions])
    mfg = np.mean([r["mean_forgetting"] for r in retentions])
    print(f"\n=== Retention (continual={args.continual_mode}): "
          f"mean_final {mf:.3f} | mean_forgetting {mfg:.3f} "
          f"(forward-transfer Δ@40 overall {np.mean([d for *_, d in rows]):+.3f}) ===")

    with open(args.output, "w") as fp:
        json.dump(dict(checkpoints=CHECKPOINTS, continual_mode=args.continual_mode,
                       arch=args.arch, dark_alpha=args.dark_alpha,
                       retention=retentions,
                       per_seed=[[{"task": t["task"],
                                   "continual": {str(k): v for k, v in t["continual"].items()},
                                   "fresh": {str(k): v for k, v in t["fresh"].items()}}
                                  for t in ps] for ps in all_seed]), fp)
    print("saved", args.output)


if __name__ == "__main__":
    main()
