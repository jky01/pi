"""
純 numpy 的 MNIST 載入器（不依賴 torch / torchvision / sklearn）。

用途：把專案在 Permuted-Pi-Digits 上得到的核心結論，搬到 CL 社群真正的標準
benchmark（Permuted-MNIST / Split-MNIST）上做外部效度驗證。

第一次呼叫時從公開鏡像下載 4 個原始 IDX gzip 檔，解析後快取成 mnist.npz，
之後直接讀快取（離線也能跑）。
"""
import gzip
import os
import struct
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "mnist.npz")

# Google 維護的 MNIST 鏡像（原 yann.lecun.com 常 403）。
_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _parse_idx(raw: bytes) -> np.ndarray:
    """解析 IDX 格式（MNIST 原始格式）。"""
    magic, = struct.unpack(">I", raw[:4])
    ndim = magic & 0xFF  # 低位元組 = 維度數
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    data = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8)
    return data.reshape(dims)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def load_mnist(cache: bool = True):
    """回傳 (X_train, y_train, X_test, y_test)。

    X：float32，shape (N, 784)，像素正規化到 [0, 1]。
    y：int64，shape (N,)，digit 標籤 0..9。
    """
    if cache and os.path.exists(_CACHE):
        d = np.load(_CACHE)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    arrays = {}
    for key, fname in _FILES.items():
        raw = gzip.decompress(_download(_BASE + fname))
        arrays[key] = _parse_idx(raw)

    X_train = arrays["train_images"].reshape(-1, 784).astype(np.float32) / 255.0
    X_test = arrays["test_images"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = arrays["train_labels"].astype(np.int64)
    y_test = arrays["test_labels"].astype(np.int64)

    if cache:
        np.savez_compressed(_CACHE, X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mnist()
    print("train:", Xtr.shape, ytr.shape, "test:", Xte.shape, yte.shape)
    print("pixel range:", Xtr.min(), Xtr.max(), "label dist:", np.bincount(ytr))
