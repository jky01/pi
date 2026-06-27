"""
P8：用 frozen pretrained backbone（ImageNet ResNet18）對 CIFAR-100 抽特徵，
存成 .npz。之後的持續學習實驗仍維持純 numpy（在 frozen 特徵上訓練小 head），
只有這一步用到 torch / torchvision。

動機：本專案的抗遺忘機制（NCM 原型、DER++ 函數錨定）都是「對表徵/函數做事」。
當代 CL 重心是 pretrained / foundation model。把輸入從「2 層 MLP 硬學的 raw pixel」
換成「強而通用的 frozen 特徵」，可直接檢驗：強表徵會放大還是抹平這些工具箱的價值。

輸出 cifar100_resnet18.npz：
  X_train (N_tr, 512) float32, y_train (N_tr,) int64,
  X_test  (N_te, 512) float32, y_test  (N_te,) int64
特徵為 ResNet18 penultimate（global avgpool）輸出，512 維。
"""
import numpy as np
import torch
import torchvision
from torchvision import transforms

torch.set_num_threads(torch.get_num_threads())  # 用滿 CPU 執行緒

TRAIN_PER_CLASS = 200   # 控制 CPU 抽取成本（每類抽 200 訓練樣本，100 類 = 20k）
RESIZE = 160            # 32->160 上採樣（224 更標準但 CPU 太慢；160 特徵已足夠）
DEVICE = "cpu"
OUT = "cifar100_resnet18.npz"


def build_extractor():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()  # 去掉分類頭，輸出 512-d penultimate 特徵
    net.eval().to(DEVICE)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def subsample_per_class(targets, per_class, n_classes, seed=0):
    rng = np.random.RandomState(seed)
    idx = []
    targets = np.asarray(targets)
    for c in range(n_classes):
        ci = np.where(targets == c)[0]
        if per_class < len(ci):
            ci = rng.choice(ci, size=per_class, replace=False)
        idx.append(ci)
    return np.concatenate(idx)


@torch.no_grad()
def extract(net, dataset, indices, tfm, batch=128):
    feats, labels = [], []
    for i in range(0, len(indices), batch):
        chunk = indices[i:i + batch]
        imgs = torch.stack([tfm(dataset.data[j]) for j in chunk]).to(DEVICE)
        f = net(imgs).cpu().numpy().astype(np.float32)
        feats.append(f)
        labels.append(np.asarray(dataset.targets)[chunk])
        if (i // batch) % 20 == 0:
            print(f"  ...{i + len(chunk)}/{len(indices)}", flush=True)
    return np.concatenate(feats), np.concatenate(labels).astype(np.int64)


def main():
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("loading CIFAR-100 ...", flush=True)
    train_ds = torchvision.datasets.CIFAR100(root="./cifar_data", train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(root="./cifar_data", train=False, download=True)

    net = build_extractor()
    print("extractor ready (ResNet18 ImageNet, fc removed)", flush=True)

    tr_idx = subsample_per_class(train_ds.targets, TRAIN_PER_CLASS, 100)
    te_idx = np.arange(len(test_ds.targets))  # 全部測試集（100 類 × 100）

    print(f"extracting train ({len(tr_idx)}) ...", flush=True)
    Xtr, ytr = extract(net, train_ds, tr_idx, tfm)
    print(f"extracting test ({len(te_idx)}) ...", flush=True)
    Xte, yte = extract(net, test_ds, te_idx, tfm)

    np.savez_compressed(OUT, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    print(f"saved {OUT}: train {Xtr.shape} test {Xte.shape} "
          f"| feat mean {Xtr.mean():.3f} std {Xtr.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
