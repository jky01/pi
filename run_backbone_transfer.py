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


def make_resnet18_cifar(n_classes, pretrained=False):
    """CIFAR-adapted ResNet18：把 7x7/stride2 stem 換成 3x3/stride1、移除 early maxpool，
    否則 32x32 輸入會被過度下採樣。這是 CIFAR ResNet 的標準改法。
    pretrained=True（P13c）：載入 ImageNet 預訓 layer1-4（廣泛可分特徵），conv1 stem 與 fc
    為新初始化——架構與從零版完全相同,唯一差別是 body 權重 pretrained vs random（乾淨 A/B）。"""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, n_classes)
    return net


def make_model(arch, n_classes, width, pretrained=False):
    if arch == "smallcnn":
        return SmallCNN(n_classes=n_classes, width=width)
    if arch == "resnet18":
        return make_resnet18_cifar(n_classes, pretrained=pretrained)
    raise ValueError(f"unknown arch: {arch}")


def acc_5way(model, Xt, yt, task_classes):
    """Task-IL：給 task id，只在該 task 的 5 個類別欄位裡 argmax（需要推論時知道 task）。"""
    model.eval()
    with torch.no_grad():
        logits = model(normalize(Xt))
        cols = torch.tensor(task_classes, device=logits.device)
        sub = logits[:, cols]
        pred = cols[sub.argmax(1)]
    model.train()
    return (pred == yt).float().mean().item()


def acc_seen(model, Xt, yt, n_seen):
    """Class-IL（無 task id）：在「目前已看過的所有類別」0..n_seen-1 裡 argmax。
    label 是全域 id，argmax 直接給全域類別。這是誠實的 task-free 評估——推論時不
    被告知是哪個 task，必須把樣本指認到正確類別、且要壓過所有已學類別的干擾。"""
    model.eval()
    with torch.no_grad():
        logits = model(normalize(Xt))[:, :n_seen]
        pred = logits.argmax(1)
    model.train()
    return (pred == yt).float().mean().item()


def penult_feat(model, x_norm, arch, chunk=256):
    """抽 penultimate 特徵（分類頭的輸入）。用 forward hook 對 head/fc 取 input，
    smallcnn 與 resnet18 通用。x_norm 須已 normalize。"""
    cap = {}
    layer = model.head if arch == "smallcnn" else model.fc
    h = layer.register_forward_hook(lambda m, inp, out: cap.__setitem__("z", inp[0].detach()))
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x_norm.shape[0], chunk):
            model(x_norm[i:i + chunk])
            outs.append(cap["z"])
    model.train()
    h.remove()
    return torch.cat(outs, 0)


def build_ncm(model, arch, Xtr, ytr, tr_idx, device, per_class=100):
    """在「最終 backbone」的特徵空間，用每類訓練 exemplar 的平均當原型（iCaRL/NCM）。
    完全 task-free 的讀出：§11/§18 證明 class-IL 線性頭有 recency bias，原型讀出修掉它；
    這裡首次在**會動的 backbone** 上測這個修法是否仍成立。"""
    by_class = {}
    for idx in tr_idx:
        Xk = torch.from_numpy(Xtr[idx]).to(device)
        yk = ytr[idx]
        z = penult_feat(model, normalize(Xk), arch)
        for c in np.unique(yk):
            by_class.setdefault(int(c), []).append(z[yk == c][:per_class])
    labels = sorted(by_class)
    protos = torch.stack([torch.cat(by_class[c], 0).mean(0) for c in labels])
    return protos, torch.tensor(labels, device=device)


def ncm_acc(model, arch, protos, labels, Xt, yt):
    model.eval()
    with torch.no_grad():
        z = penult_feat(model, normalize(Xt), arch)
        pred = labels[torch.cdist(z, protos).argmin(1)]
    model.train()
    return (pred == yt).float().mean().item()


# ---------------------------------------------------------------------------
# P12：會動 backbone 上比 NCM 更好的 task-free 讀出（attack class-IL 部署缺口）。
# P11 證明 class-IL 崩壞主因是 100-way 無偏讀出（線性頭 recency/magnitude bias）。
# 兩個 post-hoc 讀出（不重訓 backbone）：
#  (cos) cosine head：把分類頭權重與特徵都 L2-normalize 再內積→移除「近期類別 ‖W_c‖
#        較大」的 magnitude bias（= weight-alignment / cosine classifier）。
#  (bic) Bias-Correction：對每個 task-group 擬合一組 affine (α_t, β_t) 校正 logits，
#        在一個 class-balanced 校準集（每類等量 exemplar）上以 CE 擬合（Wu+2019 BiC 的
#        post-hoc 變體：一次校正所有 task-group 的相互偏置，而非只校正最新 task）。
# ---------------------------------------------------------------------------
def head_of(model, arch):
    return model.head if arch == "smallcnn" else model.fc


def cosine_acc(model, arch, Xt, yt, n_seen):
    model.eval()
    with torch.no_grad():
        W = head_of(model, arch).weight[:n_seen]
        z = penult_feat(model, normalize(Xt), arch)
        logits = F.normalize(z, dim=1) @ F.normalize(W, dim=1).t()
        pred = logits.argmax(1)
    model.train()
    return (pred == yt).float().mean().item()


def balanced_cal_set(Xtr, ytr, tr_idx, per_class, device):
    """class-balanced 校準集：每個類別取固定數量的訓練 exemplar（跨所有 task，含早期
    已被遺忘的類別 → 暴露 recency bias）。供 BiC 擬合校正用，與 test 集無重疊。"""
    xs, ys = [], []
    for idx in tr_idx:
        yk = ytr[idx]
        for c in np.unique(yk):
            take = idx[yk == c][:per_class]
            xs.append(Xtr[take]); ys.append(ytr[take])
    X = torch.from_numpy(np.concatenate(xs)).to(device)
    y = torch.from_numpy(np.concatenate(ys)).to(device)
    return X, y


def bic_fit(cal_logits, cal_y, group, n_groups, steps=400, lr=0.05):
    a = torch.ones(n_groups, device=cal_logits.device, requires_grad=True)
    b = torch.zeros(n_groups, device=cal_logits.device, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        corr = cal_logits * a[group] + b[group]
        loss = F.cross_entropy(corr, cal_y)
        opt.zero_grad(); loss.backward(); opt.step()
    return a.detach(), b.detach()


def logits_in_chunks(model, X, n_seen, chunk=256):
    outs = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            outs.append(model(normalize(X[i:i + chunk]))[:, :n_seen])
    if was_training:
        model.train()
    return torch.cat(outs, 0)


def bic_acc(model, Xt, yt, n_seen, a, b, group):
    model.eval()
    with torch.no_grad():
        logits = model(normalize(Xt))[:, :n_seen] * a[group] + b[group]
        pred = logits.argmax(1)
    model.train()
    return (pred == yt).float().mean().item()


def forward_with_feat(model, x_norm, arch):
    """Forward pass that also returns the differentiable penultimate feature tensor."""
    cap = {}
    layer = head_of(model, arch)
    h = layer.register_forward_hook(lambda m, inp, out: cap.__setitem__("z", inp[0]))
    logits = model(x_norm)
    h.remove()
    if "z" not in cap:
        raise RuntimeError("failed to capture penultimate features")
    return logits, cap["z"]


def supervised_contrastive_loss(feats, y, temp=0.2):
    """Khosla-style SupCon over a batch. Anchors without a positive pair are ignored."""
    n = feats.shape[0]
    if n <= 1:
        return feats.sum() * 0.0
    z = F.normalize(feats, dim=1)
    logits = (z @ z.t()) / temp
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(n, dtype=torch.bool, device=feats.device)
    same = y.view(-1, 1).eq(y.view(1, -1))
    pos = same & ~eye
    valid = pos.sum(1) > 0
    if not bool(valid.any()):
        return feats.sum() * 0.0

    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (pos.float() * log_prob).sum(1) / pos.sum(1).clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


class ProtoBank:
    """P13b：持久的 per-class 原型記憶庫（proto-contrastive）。

    P13 診斷:在稀疏 replay batch 上做 SupCon 失敗,因為舊類別在 batch 裡湊不出同類正對。
    ProtoBank 把「每類的代表」從 batch 解耦到一個 EMA 更新的原型庫——任何看過的類別永遠
    有一個原型,所以對比 loss 能把當前任務特徵推離**所有**舊類原型(全域可分),即使 batch
    裡沒有該舊類樣本。原型用 current+replay 特徵持續更新 → 在會動特徵空間裡保持新鮮(對抗
    representation drift)。proto loss＝把每個特徵拉向自身類原型、推離其他已見類原型(= 直接
    訓練 NCM 可分性,而 NCM 正是 P11/P12 最好的讀出)。原型為 stop-grad,梯度只流經當前特徵。"""

    def __init__(self, n_classes, momentum=0.9):
        self.n_classes = n_classes
        self.m = momentum
        self.protos = None  # (n_classes, d)，lazy；stop-grad（非葉子 grad）
        self.seen = None    # (n_classes,) bool

    def update(self, feats, y):
        feats = feats.detach()
        if self.protos is None:
            d = feats.shape[1]
            self.protos = torch.zeros(self.n_classes, d, device=feats.device)
            self.seen = torch.zeros(self.n_classes, dtype=torch.bool, device=feats.device)
        for c in torch.unique(y):
            ci = int(c)
            cm = feats[y == c].mean(0)
            if self.seen[ci]:
                self.protos[ci] = self.m * self.protos[ci] + (1 - self.m) * cm
            else:
                self.protos[ci] = cm
                self.seen[ci] = True

    def loss(self, feats, y, temp=0.1):
        if self.protos is None or int(self.seen.sum()) < 2:
            return feats.sum() * 0.0
        idx = self.seen.nonzero(as_tuple=True)[0]            # 已見類別欄
        P = F.normalize(self.protos[idx], dim=1)             # (S, d) stop-grad
        z = F.normalize(feats, dim=1)                        # (b, d)
        logits = (z @ P.t()) / temp                          # (b, S) cosine
        pos = torch.full((self.n_classes,), -1, dtype=torch.long, device=feats.device)
        pos[idx] = torch.arange(idx.numel(), device=feats.device)
        target = pos[y]
        valid = target >= 0
        if not bool(valid.any()):
            return feats.sum() * 0.0
        return F.cross_entropy(logits[valid], target[valid])


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
                mode="naive", buffer=None, dark_alpha=0.5, reg=None,
                eval_mode="taskil", n_seen=None, arch="smallcnn",
                supcon_weight=0.0, supcon_temp=0.2,
                proto=None, proto_weight=0.0, proto_temp=0.1):
    """訓練一個 task 並記錄學習曲線。
    mode: naive（純當前 batch）/ replay（+ reservoir CE）/ derpp（replay CE + logit 蒸餾）。
    eval_mode: taskil → task-k 受限 5-way（給 task id）；classil → 已看過類別 argmax（無 task id）。
    proto: P13b ProtoBank（proto-contrastive，用 current+replay 特徵更新原型並對比）。"""
    probe = (lambda: acc_5way(model, Xt, yt, task_classes)) if eval_mode == "taskil" \
        else (lambda: acc_seen(model, Xt, yt, n_seen))
    use_proto = proto is not None and proto_weight > 0
    need_feats = supcon_weight > 0 or use_proto
    n = X.shape[0]
    curve, s = {}, 0
    if 0 in CHECKPOINTS:
        curve[0] = probe()
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
                all_x = torch.cat([cur_x, normalize(rx)], 0)
                all_y = torch.cat([yb, ry], 0)
                if need_feats:
                    out, feats = forward_with_feat(model, all_x, arch)
                else:
                    out = model(all_x); feats = None
                loss = F.cross_entropy(out[:ncur], yb) + F.cross_entropy(out[ncur:], ry)
                if mode == "derpp" and rz is not None:
                    loss = loss + dark_alpha * F.mse_loss(out[ncur:], rz)
                if supcon_weight > 0:
                    loss = loss + supcon_weight * supervised_contrastive_loss(feats, all_y, supcon_temp)
                if use_proto:
                    proto.update(feats, all_y)  # update-first：當前類別永遠在庫
                    loss = loss + proto_weight * proto.loss(feats, all_y, proto_temp)
                cur_logits = out[:ncur].detach()
            else:
                if need_feats:
                    out, feats = forward_with_feat(model, cur_x, arch)
                else:
                    out = model(cur_x); feats = None
                loss = F.cross_entropy(out, yb)
                if supcon_weight > 0:
                    loss = loss + supcon_weight * supervised_contrastive_loss(feats, yb, supcon_temp)
                if use_proto:
                    proto.update(feats, yb)
                    loss = loss + proto_weight * proto.loss(feats, yb, proto_temp)
                cur_logits = out.detach()
            if reg is not None:
                loss = loss + reg.extra_loss(model, cur_x, out, ncur)
            loss.backward(); opt.step()
            if use_buf:
                buffer.add(xb_raw, yb, cur_logits if mode == "derpp" else None)
            s += 1
            if todo and s == todo[0]:
                curve[s] = probe()
                todo.pop(0)
        if not todo:
            break
    if todo:
        curve[s] = probe()
    return curve


def run_one(seed, n_tasks, classes_per_task, train_per_class, test_per_class,
            lr, epochs, batch, width, continual_mode="naive", buffer_cap=2000,
            device="cpu", arch="smallcnn", dark_alpha=0.5, lwf_lambda=1.0, lwf_temp=2.0,
            eval_mode="taskil", supcon_weight=0.0, supcon_temp=0.2,
            proto_weight=0.0, proto_temp=0.1, proto_momentum=0.9, pretrained=False):
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

    cont = make_model(arch, n_tasks * classes_per_task, width, pretrained=pretrained).to(device)
    cont_opt = torch.optim.SGD(cont.parameters(), lr=lr, momentum=0.9)
    # P13b：continual 的原型庫跨 task 持久累積（這正是它補回舊類訊號的關鍵）。
    cont_proto = ProtoBank(n_tasks * classes_per_task, proto_momentum) if proto_weight > 0 else None

    per_task = []
    test_sets = []  # 留到最後做 retention 評估（continual 終態 vs 剛學完）
    for k in range(n_tasks):
        X = torch.from_numpy(Xtr[tr_idx[k]]).to(device); y = torch.from_numpy(ytr[tr_idx[k]]).to(device)
        Xt = torch.from_numpy(Xte[te_idx[k]]).to(device); yt = torch.from_numpy(yte[te_idx[k]]).to(device)
        tcls = task_classes[k]
        test_sets.append((Xt, yt, tcls))

        n_seen = classes_per_task * (k + 1)  # class-IL：到目前為止看過的類別數
        rng_c = np.random.RandomState(1000 + seed)
        cont_curve = train_curve(cont, cont_opt, X, y, Xt, yt, tcls, epochs, batch, rng_c,
                                 mode=continual_mode, buffer=buffer, dark_alpha=dark_alpha,
                                 reg=reg, eval_mode=eval_mode, n_seen=n_seen, arch=arch,
                                 supcon_weight=supcon_weight, supcon_temp=supcon_temp,
                                 proto=cont_proto, proto_weight=proto_weight, proto_temp=proto_temp)
        if reg is not None:
            reg.after_task(cont, k)

        fresh = make_model(arch, n_tasks * classes_per_task, width, pretrained=pretrained).to(device)
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=lr, momentum=0.9)
        # fresh 只學 task k，給它一個本任務內的新原型庫（保持訓練目標一致、可公平比較）。
        fresh_proto = ProtoBank(n_tasks * classes_per_task, proto_momentum) if proto_weight > 0 else None
        rng_f = np.random.RandomState(1000 + seed)
        fresh_curve = train_curve(fresh, fresh_opt, X, y, Xt, yt, tcls, epochs, batch, rng_f,
                                  mode="naive", eval_mode=eval_mode, n_seen=n_seen, arch=arch,
                                  supcon_weight=supcon_weight, supcon_temp=supcon_temp,
                                  proto=fresh_proto, proto_weight=proto_weight, proto_temp=proto_temp)

        per_task.append(dict(task=k, continual=cont_curve, fresh=fresh_curve))
        print(f"  seed{seed} task{k:>2}: cont@40={cont_curve.get(40, float('nan')):.3f} "
              f"fresh@40={fresh_curve.get(40, float('nan')):.3f}", flush=True)

    # retention：終態 continual 模型回評每個 task，對比剛學完時。
    # taskil → 給 task id 的受限 5-way；classil → 無 task id，在全部 100 類裡 argmax。
    maxc = max(CHECKPOINTS)
    n_all = n_tasks * classes_per_task
    diag = [at(per_task[j]["continual"], maxc) for j in range(n_tasks)]
    if eval_mode == "classil":
        final = [acc_seen(cont, Xt, yt, n_all) for (Xt, yt, tcls) in test_sets]
    else:
        final = [acc_5way(cont, Xt, yt, tcls) for (Xt, yt, tcls) in test_sets]
    forgetting = [diag[j] - final[j] for j in range(n_tasks)]
    retention = dict(diag=diag, final=final,
                     mean_forgetting=float(np.mean(forgetting)),
                     mean_final=float(np.mean(final)))
    msg = (f"  seed{seed} retention: mean_final={retention['mean_final']:.3f} "
           f"mean_forgetting={retention['mean_forgetting']:.3f}")
    if eval_mode == "classil":
        # task-free 讀出（皆 post-hoc，不重訓 backbone）：NCM(§11/§18) + P12 的 cosine/BiC。
        protos, labels = build_ncm(cont, arch, Xtr, ytr, tr_idx, device)
        final_ncm = [ncm_acc(cont, arch, protos, labels, Xt, yt) for (Xt, yt, tcls) in test_sets]
        final_cos = [cosine_acc(cont, arch, Xt, yt, n_all) for (Xt, yt, tcls) in test_sets]
        # BiC：在 class-balanced 校準集上擬合 per-task-group affine 校正。
        cal_X, cal_y = balanced_cal_set(Xtr, ytr, tr_idx, 20, device)
        cal_logits = logits_in_chunks(cont, cal_X, n_all)
        group = torch.tensor([c // classes_per_task for c in range(n_all)], device=device)
        a, b = bic_fit(cal_logits, cal_y, group, n_tasks)
        final_bic = [bic_acc(cont, Xt, yt, n_all, a, b, group) for (Xt, yt, tcls) in test_sets]
        retention["final_ncm"] = final_ncm
        retention["mean_final_ncm"] = float(np.mean(final_ncm))
        retention["final_cos"] = final_cos
        retention["mean_final_cos"] = float(np.mean(final_cos))
        retention["final_bic"] = final_bic
        retention["mean_final_bic"] = float(np.mean(final_bic))
        msg += (f" | class-IL linear={retention['mean_final']:.3f} "
                f"NCM={retention['mean_final_ncm']:.3f} cos={retention['mean_final_cos']:.3f} "
                f"BiC={retention['mean_final_bic']:.3f}")
    print(msg, flush=True)
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
    p.add_argument("--supcon-weight", type=float, default=0.0,
                   help="P13: supervised contrastive loss weight on penultimate features")
    p.add_argument("--supcon-temp", type=float, default=0.2,
                   help="P13: supervised contrastive temperature")
    p.add_argument("--proto-weight", type=float, default=0.0,
                   help="P13b: proto-contrastive loss weight (persistent per-class prototype bank)")
    p.add_argument("--proto-temp", type=float, default=0.1, help="P13b: proto-contrastive temperature")
    p.add_argument("--proto-momentum", type=float, default=0.9, help="P13b: prototype EMA momentum")
    p.add_argument("--pretrained", action="store_true",
                   help="P13c: init resnet18 body from ImageNet pretrained weights (vs from-scratch)")
    p.add_argument("--arch", choices=["smallcnn", "resnet18"], default="smallcnn")
    p.add_argument("--eval-mode", choices=["taskil", "classil"], default="taskil",
                   help="taskil: 給 task id 的受限 5-way；classil: 無 task id 的 task-free 評估")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--output", default="results_backbone_transfer_cifar100.json")
    args = p.parse_args()
    if args.continual_mode == "pnn" and args.eval_mode == "classil":
        p.error("pnn 本質依賴 task-id 路由到對的 column，無法做 classil；task-free 實驗用 "
                "naive/replay/derpp/lwf_kd。")

    device = pick_device(args.device)
    print(f"device: {device} | eval_mode: {args.eval_mode}", flush=True)

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
                          lwf_lambda=args.lwf_lambda, lwf_temp=args.lwf_temp,
                          eval_mode=args.eval_mode, supcon_weight=args.supcon_weight,
                          supcon_temp=args.supcon_temp, proto_weight=args.proto_weight,
                          proto_temp=args.proto_temp, proto_momentum=args.proto_momentum,
                          pretrained=args.pretrained)
        all_seed.append(res["per_task"])
        retentions.append(res["retention"])
        print(f"seed {seed} done", flush=True)

    probe_desc = ("task-k 5-way acc (Task-IL, given task id)" if args.eval_mode == "taskil"
                  else "all-seen-class acc (Class-IL, NO task id)")
    print(f"\n=== forward transfer with ADAPTING backbone (arch={args.arch}, "
          f"continual={args.continual_mode}, eval={args.eval_mode}, "
          f"supcon={args.supcon_weight:g}) ===")
    print(f"{probe_desc}: continual (backbone adapts across tasks) vs fresh-from-scratch")
    for step in [10, 20, 40, 80]:
        rows = summarize(all_seed, step)
        c = np.mean([r[1] for r in rows]); f = np.mean([r[2] for r in rows])
        early = np.mean([d for k, _, _, d in rows[:min(5, len(rows))]])
        late = np.mean([d for k, _, _, d in rows[max(0, len(rows) - 5):]])
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
    print(f"\n=== Retention (continual={args.continual_mode}, eval={args.eval_mode}): "
          f"mean_final {mf:.3f} | mean_forgetting {mfg:.3f} "
          f"(forward-transfer Δ@40 overall {np.mean([d for *_, d in rows]):+.3f}) ===")
    if args.eval_mode == "classil":
        ncm = np.mean([r["mean_final_ncm"] for r in retentions])
        cos = np.mean([r["mean_final_cos"] for r in retentions])
        bic = np.mean([r["mean_final_bic"] for r in retentions])
        print(f"    class-IL final (NO task id): linear {mf:.3f} | NCM {ncm:.3f} | "
              f"cosine {cos:.3f} | BiC {bic:.3f}")

    with open(args.output, "w") as fp:
        json.dump(dict(checkpoints=CHECKPOINTS, continual_mode=args.continual_mode,
                       arch=args.arch, dark_alpha=args.dark_alpha, eval_mode=args.eval_mode,
                       supcon_weight=args.supcon_weight, supcon_temp=args.supcon_temp,
                       proto_weight=args.proto_weight, proto_temp=args.proto_temp,
                       pretrained=args.pretrained,
                       retention=retentions,
                       per_seed=[[{"task": t["task"],
                                   "continual": {str(k): v for k, v in t["continual"].items()},
                                   "fresh": {str(k): v for k, v in t["fresh"].items()}}
                                  for t in ps] for ps in all_seed]), fp)
    print("saved", args.output)


if __name__ == "__main__":
    main()
