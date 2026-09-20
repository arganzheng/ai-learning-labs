"""无监督（经典 ML 04）：用 NumPy 手写 K-Means 与 PCA，与 scikit-learn 对数。
https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html
"""
import numpy as np
from sklearn.datasets import make_blobs, load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

def kmeans(X, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]                 # 初始化：随机挑 k 个点当中心
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)       # [n, k]：每个点到每个中心的距离平方
        labels = d2.argmin(1)                                           # 分配：归到最近的中心
        new_centers = np.array([X[labels == j].mean(0) for j in range(k)])   # 更新：每簇的均值
        if np.allclose(new_centers, centers): break                     # 中心不再移动就收敛
        centers = new_centers
    inertia = d2[np.arange(len(X)), labels].sum()                       # 簇内平方和
    return labels, centers, inertia

def pca(X, k):
    Xc = X - X.mean(0)                                                  # 中心化
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)                   # 数据矩阵的 SVD
    explained = S ** 2 / (len(X) - 1)                                   # 每个主成分的方差 = 奇异值² / (n−1)
    return Xc @ Vt[:k].T, explained / explained.sum()                   # 投影到前 k 个方向；解释方差比

X, y = make_blobs(n_samples=3000, centers=6, cluster_std=1.2, random_state=3)
best=None
for seed in range(10):
    lab, cen, inert = kmeans(X, 6, seed=seed)
    if best is None or inert < best[2]: best=(lab,cen,inert)
lab,cen,inert=best
sk = KMeans(6, n_init=10, random_state=0).fit(X)
print(f"手写 K-Means（10 次取最好）簇内平方和 {inert:.0f}, ARI {adjusted_rand_score(y, lab):.3f}; sklearn 簇内平方和 {sk.inertia_:.0f}, ARI {adjusted_rand_score(y, sk.labels_):.3f}")
D = load_digits().data
Z, ratio = pca(D, 2)
skp = PCA().fit(D)
print(f"手写 PCA 前 2 维解释 {ratio[:2].sum():.1%}, 前 29 维 {ratio[:29].sum():.1%}; sklearn 前 2 维 {skp.explained_variance_ratio_[:2].sum():.1%}")
print("投影绝对值最大差", np.abs(np.abs(Z) - np.abs(PCA(2).fit_transform(D))).max())
