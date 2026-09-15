"""面试手撕代码（18）：手撕经典 ML 与评测指标 —— k-means、逻辑回归、KNN、PCA、AUC、NDCG、conv2d、NMS。

https://arganzheng.life/coding-interview-classical-ml-and-metrics.html

    python classical_ml_metrics.py            # 数值示例
    python classical_ml_metrics.py --check    # 与 torch / sklearn 风格的参考对拍（sklearn 可选）
"""
from __future__ import annotations

import argparse
import heapq
import math

import numpy as np


# ---------- 1. k-means ----------

def kmeans(X: np.ndarray, k: int, iters: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd 迭代：分配到最近中心 → 中心取均值。k-means++ 初始化。返回 (centers, labels)。"""
    n = len(X)
    centers = X[rng.integers(n)][None]
    for _ in range(1, k):                                       # k-means++：按 D² 概率选下一个中心
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1).min(1)
        centers = np.vstack([centers, X[rng.choice(n, p=d2 / d2.sum())]])
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1)     # (n, k)
        labels = d2.argmin(1)
        new = np.array([X[labels == j].mean(0) if (labels == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, labels


# ---------- 2. 逻辑回归 ----------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, steps: int, l2: float = 0.0) -> tuple[np.ndarray, float]:
    """梯度下降。∂L/∂w = Xᵀ(σ(Xw+b) - y)/n + λw。"""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


# ---------- 3. KNN ----------

def knn_predict(X_train: np.ndarray, y_train: np.ndarray, x: np.ndarray, k: int) -> int:
    """大小 k 的最大堆（存负距离）选最近 k 个，多数投票。"""
    heap: list[tuple[float, int]] = []
    for xi, yi in zip(X_train, y_train):
        d = float(((xi - x) ** 2).sum())
        if len(heap) < k:
            heapq.heappush(heap, (-d, int(yi)))
        elif -d > heap[0][0]:
            heapq.heapreplace(heap, (-d, int(yi)))
    votes = np.bincount([yi for _, yi in heap])
    return int(votes.argmax())


# ---------- 4. PCA ----------

def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """中心化 → SVD：X = U S Vᵀ，主成分是 V 的前 k 列，投影 = X_c V_k。返回 (投影, 主成分, 方差解释比)。"""
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]                                    # (k, d)
    var_ratio = S[:n_components] ** 2 / (S ** 2).sum()
    return Xc @ comps.T, comps, var_ratio


# ---------- 5. 指标 ----------

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC = P(正样本得分 > 负样本得分)，相等算 0.5。排序后用秩求和，O(n log n)；并列分数取平均秩。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):                                       # 并列分数：平均秩
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1                  # 1-based 平均秩
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_auc_brute(y_true: np.ndarray, scores: np.ndarray) -> float:
    """O(n²) 定义式，用来对拍。"""
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    cmp = (pos[:, None] > neg[None]).astype(float) + 0.5 * (pos[:, None] == neg[None])
    return float(cmp.mean())


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """relevance 按模型排序给出的相关度序列。DCG = Σ (2^rel - 1) / log2(i + 1)；IDCG 用理想排序。"""
    rel = relevance[:k]
    discounts = 1 / np.log2(np.arange(2, len(rel) + 2))
    dcg = ((2 ** rel - 1) * discounts).sum()
    ideal = np.sort(relevance)[::-1][:k]
    idcg = ((2 ** ideal - 1) * discounts[:len(ideal)]).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------- 6. 卷积、池化、NMS ----------

def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> tuple[np.ndarray, int, int]:
    """x: (N, C, H, W) → (N·H_out·W_out, C·kh·kw)。每一行是一个感受野展平。"""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = (W + 2 * pad - kw) // stride + 1
    cols = np.empty((N, H_out, W_out, C, kh, kw))
    for i in range(H_out):
        for j in range(W_out):
            cols[:, i, j] = xp[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
    return cols.reshape(N * H_out * W_out, C * kh * kw), H_out, W_out


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None, stride: int = 1, pad: int = 0) -> np.ndarray:
    """x: (N, C_in, H, W)，w: (C_out, C_in, kh, kw)。卷积 = im2col 后一次矩阵乘。"""
    C_out, C_in, kh, kw = w.shape
    cols, H_out, W_out = im2col(x, kh, kw, stride, pad)          # (N·Ho·Wo, C_in·kh·kw)
    out = cols @ w.reshape(C_out, -1).T                          # (N·Ho·Wo, C_out)
    if b is not None:
        out += b
    return out.reshape(x.shape[0], H_out, W_out, C_out).transpose(0, 3, 1, 2)


def max_pool2d(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    N, C, H, W = x.shape
    H_out, W_out = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.empty((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k].max((2, 3))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (4,) 一个框 [x1, y1, x2, y2]；b: (M, 4)。返回 (M,)。"""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """按分数降序：取最高的，压掉与它 IoU > 阈值的，重复。返回保留的下标。"""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thresh]
    return keep


# ---------- demo & check ----------

def demo():
    rng = np.random.default_rng(0)
    print("== k-means：三团高斯点，k=3")
    X = np.vstack([rng.standard_normal((50, 2)) + c for c in ([0, 0], [6, 0], [3, 5])])
    centers, labels = kmeans(X, 3, 50, rng)
    print("  centers ≈", np.round(centers[np.argsort(centers[:, 0])], 2).tolist(), "（真值 [0,0] [3,5] [6,0]）")

    print("\n== 逻辑回归：线性可分数据，训练准确率")
    X = rng.standard_normal((200, 2)); y = (X[:, 0] + X[:, 1] > 0).astype(float)
    w, b = logistic_regression(X, y, lr=0.5, steps=500)
    print(f"  w ≈ {np.round(w, 2).tolist()}（方向应 ≈ [1, 1]），acc = {((sigmoid(X @ w + b) > 0.5) == y).mean():.3f}")

    print("\n== KNN（k=5）在同一数据上的留一预测")
    correct = sum(knn_predict(np.delete(X, i, 0), np.delete(y, i), X[i], 5) == y[i] for i in range(50))
    print(f"  前 50 个点的留一准确率 {correct / 50:.2f}")

    print("\n== PCA：二维数据主方向")
    Z = rng.standard_normal((300, 2)) @ np.array([[3, 0], [0, 0.5]]) @ np.array([[math.cos(0.6), -math.sin(0.6)], [math.sin(0.6), math.cos(0.6)]])
    _, comps, ratio = pca(Z, 2)
    print(f"  第一主成分 {np.round(comps[0], 3).tolist()}（≈ ±[cos 0.6, −sin 0.6] = ±[0.825, −0.565]），方差解释 {np.round(ratio, 3).tolist()}")

    print("\n== 指标")
    yt = np.array([1, 1, 0, 0, 1, 0, 1, 0]); sc = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.3, 0.2])
    p, r, f1 = precision_recall_f1(yt, (sc > 0.5).astype(int))
    print(f"  阈值 0.5：P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"  AUC（秩法）={roc_auc(yt, sc):.4f}  AUC（定义式）={roc_auc_brute(yt, sc):.4f}")
    print(f"  NDCG@5 of rel [3,2,3,0,1,2] = {ndcg_at_k(np.array([3, 2, 3, 0, 1, 2.0]), 5):.4f}（理想排序 [3,3,2,2,1] → 1.0）")

    print("\n== conv2d via im2col：(1,1,4,4) 输入、3×3 核、pad 1")
    x = np.arange(16.0).reshape(1, 1, 4, 4); w = np.zeros((1, 1, 3, 3)); w[0, 0, 1, 1] = 1; w[0, 0, 1, 2] = -1
    print("  核 = 中心 1、右 −1（水平差分），输出：")
    print(np.round(conv2d(x, w, pad=1)[0, 0], 1))
    print(f"  im2col 形状：{im2col(x, 3, 3, 1, 1)[0].shape}（16 个位置 × 9 个权重）")

    print("\n== NMS：三个重叠框 + 一个远处的框，阈值 0.5")
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12], [50, 50, 60, 60.0]]); scores = np.array([0.9, 0.8, 0.7, 0.6])
    print(f"  IoU(框0, 框1) = {iou(boxes[0], boxes[1:2])[0]:.3f}，保留下标 {nms(boxes, scores, 0.5)}")


def check():
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(5)

    # conv2d / max_pool 对拍
    x = rng.standard_normal((2, 3, 7, 6)); w = rng.standard_normal((4, 3, 3, 3)); b = rng.standard_normal(4)
    for stride, pad in ((1, 0), (2, 1), (1, 1)):
        ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=pad).numpy()
        assert np.allclose(conv2d(x, w, b, stride, pad), ref, atol=1e-10)
    assert np.allclose(max_pool2d(x, 2, 2), F.max_pool2d(torch.tensor(x), 2, 2).numpy())

    # AUC：秩法 == 定义式（含并列分数）
    for _ in range(20):
        yt = rng.integers(0, 2, 30); yt[0], yt[1] = 0, 1
        sc = rng.integers(0, 6, 30) / 5                          # 大量并列
        assert abs(roc_auc(yt, sc) - roc_auc_brute(yt, sc)) < 1e-12

    # PCA：投影 == sklearn 风格（用 torch.pca_lowrank 对拍方差解释比与子空间）
    Z = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 5))
    proj, comps, ratio = pca(Z, 2)
    Zc = Z - Z.mean(0)
    cov = Zc.T @ Zc / (len(Z) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    top = eigvec[:, ::-1][:, :2].T
    assert np.allclose(np.abs(comps @ top.T), np.eye(2), atol=1e-8)          # 同一子空间（符号可差）
    assert np.allclose(ratio, eigval[::-1][:2] / eigval.sum(), atol=1e-10)

    # NMS vs torchvision（若可用）
    boxes = rng.random((30, 2)) * 50
    boxes = np.hstack([boxes, boxes + rng.random((30, 2)) * 30 + 1]); scores = rng.random(30)
    try:
        from torchvision.ops import nms as tv_nms
        ref = tv_nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
        assert nms(boxes, scores, 0.5) == ref, (nms(boxes, scores, 0.5), ref)
    except ImportError:
        pass

    # 逻辑回归 vs torch 的同一 GD（同一步数、同一 lr）
    X = rng.standard_normal((64, 3)); y = (X @ np.array([1.0, -2.0, 0.5]) > 0).astype(float)
    w, bb = logistic_regression(X, y, lr=0.1, steps=50)
    wt = torch.zeros(3, requires_grad=True); bt = torch.zeros(1, requires_grad=True)
    for _ in range(50):
        loss = F.binary_cross_entropy_with_logits(torch.tensor(X) @ wt + bt, torch.tensor(y))
        loss.backward()
        with torch.no_grad():
            wt -= 0.1 * wt.grad; bt -= 0.1 * bt.grad
        wt.grad = None; bt.grad = None
    assert np.allclose(w, wt.detach().numpy(), atol=1e-10) and abs(bb - bt.item()) < 1e-10

    # F1 / NDCG 小例子
    assert precision_recall_f1(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == (0.5, 0.5, 0.5)
    assert abs(ndcg_at_k(np.array([3, 3, 2, 2, 1.0]), 5) - 1.0) < 1e-12
    assert ndcg_at_k(np.array([0, 0, 3.0]), 3) < ndcg_at_k(np.array([3, 0, 0.0]), 3)

    # KNN：k=1 时预测就是最近邻的标签
    X = rng.standard_normal((20, 2)); yl = rng.integers(0, 3, 20)
    q = rng.standard_normal(2)
    assert knn_predict(X, yl, q, 1) == yl[((X - q) ** 2).sum(1).argmin()]

    # k-means：中心是各簇均值（不动点性质）
    Xk = np.vstack([rng.standard_normal((30, 2)) + c for c in ([0, 0], [8, 8])])
    centers, labels = kmeans(Xk, 2, 50, rng)
    for j in range(2):
        assert np.allclose(centers[j], Xk[labels == j].mean(0))
    print("classical_ml_metrics.py: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check() if args.check else demo()
