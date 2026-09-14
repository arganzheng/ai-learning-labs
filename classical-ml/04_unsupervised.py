"""无监督（经典 ML 04）：K-Means 与 k 的选择、DBSCAN 在非球形簇上、PCA 的解释方差与二维可视化。
https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html

    python 04_unsupervised.py            # 全部：kmeans dbscan pca；图存到 out/
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

OUT = Path(__file__).with_name("out")


# ---------------- 1. K-Means 与 k 的选择 ----------------
def exp_kmeans():
    print("=== 1. K-Means：真实 6 簇的数据，k 从 2 扫到 10 ===")
    X, y = make_blobs(n_samples=3000, centers=6, cluster_std=1.2, random_state=3)
    print(f"  {'k':>3} {'簇内平方和':>12} {'轮廓系数':>9}")
    for k in range(2, 11):
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        print(f"  {k:>3} {km.inertia_:>12.0f} {silhouette_score(X, km.labels_):>9.3f}{'   ← 肘部 / 轮廓最高' if k == 6 else ''}")
    km = KMeans(6, n_init=10, random_state=0).fit(X)
    print(f"  k=6 时与真实标签的 ARI {adjusted_rand_score(y, km.labels_):.3f}（1 = 完全一致）；每簇样本数 {np.bincount(km.labels_).tolist()}")
    print("  数据工程里的用法：语料 embedding 后聚类 → 每簇抽几条看主题 → 按簇配比 / 找垃圾簇")
    print()


# ---------------- 2. DBSCAN：K-Means 假设簇是球形的 ----------------
def exp_dbscan():
    print("=== 2. 两个月牙：K-Means 切错、DBSCAN 按密度切对 ===")
    X, y = make_moons(1500, noise=0.06, random_state=0)
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    db = DBSCAN(eps=0.15, min_samples=8).fit(X)
    n_noise = int((db.labels_ == -1).sum())
    print(f"  K-Means k=2:  ARI {adjusted_rand_score(y, km.labels_):.3f}（球形假设不成立，一刀切在中间）")
    print(f"  DBSCAN:       ARI {adjusted_rand_score(y, db.labels_):.3f}，找到 {db.labels_.max() + 1} 簇 + {n_noise} 个噪声点（不用指定 k，能标出离群点）")
    print()


# ---------------- 3. PCA：解释方差与可视化 ----------------
def exp_pca():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== 3. PCA：手写数字 64 维 → 前几个主成分解释多少方差 ===")
    d = load_digits()
    X = d.data - d.data.mean(0)
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    for k in (1, 2, 5, 10, 20, 30):
        print(f"  前 {k:>2} 个主成分解释 {cum[k-1]*100:5.1f}% 的方差")
    print(f"  解释 95% 需要 {int(np.searchsorted(cum, 0.95)) + 1} 维（原 64 维）——数据'基本上'是低维的：LoRA 对 ΔW 低秩的假设是同一种直觉")
    # 与 SVD 的关系（L0 第三篇）
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"  协方差矩阵的特征值 = 奇异值² / (n−1)：前 3 个 {np.round(pca.explained_variance_[:3], 1)} vs {np.round(S[:3]**2 / (len(X) - 1), 1)}")
    Z = pca.transform(X)[:, :2]
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=d.target, cmap="tab10", s=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("digits: PCA to 2D")
    fig.colorbar(sc, ax=ax, label="digit")
    path = OUT / "digits_pca.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"  二维投影图已保存 {path}（0 与 1 分得开，3/5/8 混在一起——线性投影只能到这个程度，t-SNE / UMAP 更好但非线性、不可逆）")
    print()


EXPS = {"kmeans": exp_kmeans, "dbscan": exp_dbscan, "pca": exp_pca}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
