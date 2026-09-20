"""聚类（经典 ML 07）：K-Means 的迭代过程与手写实现、k 怎么选、k-means++、DBSCAN、层次聚类，
以及用一个真实的小语料（78 句、Qwen2.5-0.5B 句向量）做"这批数据里有什么"。
https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html

    python 07_clustering.py          # 全部：iterate kmeans choose_k init dbscan hierarchical corpus
    python 07_clustering.py corpus   # 语料实验第一次跑要加载 Qwen2.5-0.5B（本地缓存），之后读 out/sentence_embeddings.npz

图输出到 out/07-*.svg。
"""
import sys

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _plot import C, plt, save

PALETTE = [C["blue"], C["red"], C["green"], C["orange"], C["purple"], "#b58900", C["gray"]]


# ---------------- 手写 K-Means ----------------
def kmeans(X, k, n_iter=100, seed=0, record=False):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]                     # ① 初始化：随机挑 k 个点当中心
    history = [centers.copy()]
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)           # ② [n, k]：每个点到每个中心的距离平方
        labels = d2.argmin(1)                                               # ③ 分配：每个点归到最近的中心
        new_centers = np.array([X[labels == j].mean(0) for j in range(k)])  # ④ 更新：每个中心移到自己簇的均值
        history.append(new_centers.copy())
        if np.allclose(new_centers, centers): break                         # ⑤ 中心不再移动就收敛
        centers = new_centers
    inertia = d2[np.arange(len(X)), labels].sum()                         # 簇内平方和
    return (labels, centers, inertia, history) if record else (labels, centers, inertia)


def blobs6():
    return make_blobs(n_samples=3000, centers=6, cluster_std=1.2, random_state=3)


# ---------------- 1. 迭代过程：四帧 ----------------
def exp_iterate():
    print("=== 1. K-Means 的迭代：分配 → 更新中心，反复 ===")
    X, y = make_blobs(n_samples=180, centers=3, cluster_std=1.0, random_state=1)
    labels, centers, inertia, hist = kmeans(X, 3, seed=9, record=True)
    print(f"  {len(hist) - 1} 步收敛；每步的簇内平方和：", end="")
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.2))
    frames = [0, 1, 2, len(hist) - 1]
    for ax, t in zip(axes, frames):
        c = hist[t]
        d2 = ((X[:, None, :] - c[None]) ** 2).sum(-1); lab = d2.argmin(1)
        inert = d2[np.arange(len(X)), lab].sum()
        print(f" {inert:.0f}", end="")
        for j in range(3):
            ax.scatter(X[lab == j, 0], X[lab == j, 1], s=5, color=PALETTE[j], alpha=0.7)
        ax.scatter(c[:, 0], c[:, 1], marker="X", s=80, color="k", edgecolor="w", lw=1, zorder=5)
        ax.set_title("初始化：随机挑 3 个点" if t == 0 else (f"收敛（第 {t} 步）" if t == frames[-1] else f"第 {t} 步"), fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    print()
    save(fig, "07-kmeans-iterations")
    print("  每一步簇内平方和都不增：分配步让每个点到中心更近，更新步里均值是让平方和最小的点 → 一定收敛（到局部最优）")
    print()


# ---------------- 2. 手写 vs sklearn ----------------
def exp_kmeans():
    print("=== 2. 手写 K-Means（12 行）vs sklearn：6 簇数据 ===")
    X, y = blobs6()
    best = min((kmeans(X, 6, seed=s) for s in range(10)), key=lambda r: r[2])
    km = KMeans(6, n_init=10, random_state=0).fit(X)
    print(f"  手写（10 个 seed 取簇内平方和最小）：inertia {best[2]:.0f}，ARI {adjusted_rand_score(y, best[0]):.3f}")
    print(f"  sklearn KMeans(6, n_init=10)：       inertia {km.inertia_:.0f}，ARI {adjusted_rand_score(y, km.labels_):.3f}")
    print(f"  每簇样本数 {np.bincount(km.labels_).tolist()}")
    print()


# ---------------- 3. k 怎么选：肘部与轮廓系数 ----------------
def exp_choose_k():
    print("=== 3. k 怎么选：簇内平方和（肘部）与轮廓系数 ===")
    X, y = blobs6()
    ks = list(range(2, 11))
    inertias, sils = [], []
    print(f"  {'k':>3} {'簇内平方和':>12} {'轮廓系数':>9}")
    for k in ks:
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_); sils.append(silhouette_score(X, km.labels_))
        print(f"  {k:>3} {km.inertia_:>12.0f} {sils[-1]:>9.3f}{'   ← 肘部' if k == 6 else ''}")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    ax = axes[0]
    ax.plot(ks, inertias, "o-", color=C["blue"]); ax.axvline(6, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("k"); ax.set_ylabel("簇内平方和"); ax.set_title("肘部法：k = 6 之后下降变缓", fontsize=8.5)
    ax = axes[1]
    ax.plot(ks, sils, "o-", color=C["red"]); ax.axvline(6, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("k"); ax.set_ylabel("轮廓系数"); ax.set_title("轮廓系数：k = 2 最高（3 个大组）", fontsize=8.5)
    ax = axes[2]
    km = KMeans(6, n_init=10, random_state=0).fit(X)
    Xp, lp = X[::5], km.labels_[::5]                                    # 画图只取 1/5 的点，SVG 才不至于太大
    for j in range(6):
        ax.scatter(Xp[lp == j, 0], Xp[lp == j, 1], s=4, color=PALETTE[j], alpha=0.7)
    ax.set_title(f"k = 6，ARI {adjusted_rand_score(y, km.labels_):.2f}", fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "07-choose-k")
    print("  两个判据不一致时按业务定：想看 3 个大主题还是 6 个细主题")
    print()


# ---------------- 4. 初始化：随机 vs k-means++ ----------------
def exp_init():
    print("=== 4. 初始化：随机挑点 vs k-means++（挑得彼此远）===")
    X, y = blobs6()
    rand = np.array([kmeans(X, 6, seed=s)[2] for s in range(50)])
    pp = np.array([KMeans(6, init="k-means++", n_init=1, random_state=s).fit(X).inertia_ for s in range(50)])
    print(f"  50 个 seed：随机初始化的簇内平方和 中位数 {np.median(rand):.0f}，最差 {rand.max():.0f}，落到最优（≈{rand.min():.0f}）的比例 {(rand < rand.min()*1.01).mean():.0%}")
    print(f"           k-means++            中位数 {np.median(pp):.0f}，最差 {pp.max():.0f}，落到最优的比例 {(pp < pp.min()*1.01).mean():.0%}")
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    bins = np.linspace(min(rand.min(), pp.min()) * 0.98, max(rand.max(), pp.max()) * 1.02, 30)
    ax.hist(rand, bins=bins, color=C["gray"], alpha=0.7, label="随机初始化")
    ax.hist(pp, bins=bins, color=C["red"], alpha=0.7, label="k-means++")
    ax.set_xlabel("收敛后的簇内平方和（50 个不同 seed）"); ax.set_ylabel("次数"); ax.legend(frameon=False)
    save(fig, "07-init-kmeanspp")
    print("  局部最优是真实存在的：随机初始化最差能停在 2 倍于最优的解（两个中心挤在一个真簇里）；k-means++ 让初始中心彼此远，最差情况好得多；实践再加 n_init 多跑几次取最好")
    print()


# ---------------- 5. DBSCAN ----------------
def exp_dbscan():
    print("=== 5. 两个月牙：K-Means 切错、DBSCAN 按密度切对 ===")
    X, y = make_moons(1500, noise=0.06, random_state=0)
    km = KMeans(2, n_init=10, random_state=0).fit(X)
    db = DBSCAN(eps=0.15, min_samples=8).fit(X)
    n_noise = int((db.labels_ == -1).sum())
    core = np.zeros(len(X), bool); core[db.core_sample_indices_] = True
    print(f"  K-Means k=2:  ARI {adjusted_rand_score(y, km.labels_):.3f}（球形假设不成立，一刀切在中间）")
    print(f"  DBSCAN(eps=0.15, min_samples=8): ARI {adjusted_rand_score(y, db.labels_):.3f}，{db.labels_.max() + 1} 簇 + {n_noise} 个噪声点；核心点 {core.sum()}，边界点 {(~core & (db.labels_ >= 0)).sum()}")
    for eps in (0.05, 0.1, 0.3):
        d = DBSCAN(eps=eps, min_samples=8).fit(X)
        print(f"    eps = {eps}: {d.labels_.max() + 1} 簇，噪声 {(d.labels_ == -1).sum()} 个，ARI {adjusted_rand_score(y, d.labels_):.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))
    Xp, kl, dl = X[::3], km.labels_[::3], db.labels_[::3]                # 画图取 1/3 的点
    ax = axes[0]
    for j in range(2):
        ax.scatter(Xp[kl == j, 0], Xp[kl == j, 1], s=4, color=PALETTE[j])
    ax.scatter(*km.cluster_centers_.T, marker="X", s=80, color="k", edgecolor="w", zorder=5)
    ax.set_title(f"K-Means k=2：ARI {adjusted_rand_score(y, km.labels_):.2f}", fontsize=8.5)
    ax = axes[1]
    for j in range(2):
        ax.scatter(Xp[dl == j, 0], Xp[dl == j, 1], s=4, color=PALETTE[j])
    ax.scatter(Xp[dl == -1, 0], Xp[dl == -1, 1], s=12, color="k", marker="x")
    ax.set_title(f"DBSCAN：ARI {adjusted_rand_score(y, db.labels_):.2f}，× 是噪声点", fontsize=8.5)
    ax = axes[2]
    Xs, ys = make_moons(200, noise=0.08, random_state=1)
    Xs = np.r_[Xs, np.random.default_rng(2).uniform([-1.2, -0.8], [2.2, 1.4], (8, 2))]      # 撒 8 个离群点
    d = DBSCAN(eps=0.2, min_samples=6).fit(Xs)
    core_s = np.zeros(len(Xs), bool); core_s[d.core_sample_indices_] = True
    border = ~core_s & (d.labels_ >= 0); noise = d.labels_ == -1
    ax.scatter(Xs[core_s, 0], Xs[core_s, 1], s=14, color=C["blue"], label=f"核心点 {core_s.sum()}")
    ax.scatter(Xs[border, 0], Xs[border, 1], s=14, facecolors="none", edgecolors=C["orange"], label=f"边界点 {border.sum()}")
    ax.scatter(Xs[noise, 0], Xs[noise, 1], s=24, color="k", marker="x", label=f"噪声 {noise.sum()}")
    ax.set_title("三种点：半径内 ≥ 6 个是核心点", fontsize=8.5); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    save(fig, "07-kmeans-vs-dbscan")
    print()


# ---------------- 6. 层次聚类与树状图 ----------------
def exp_hierarchical():
    print("=== 6. 层次聚类：从每个点自成一簇开始，反复合并最近的两簇 ===")
    X, y = make_blobs(n_samples=30, centers=3, cluster_std=0.9, random_state=7)
    Z = linkage(X, method="ward")                                   # 每行：合并了哪两簇、合并时的距离、新簇大小
    labels3 = fcluster(Z, 3, criterion="maxclust")
    print(f"  30 个点、Ward 链接：最后三次合并的距离 {np.round(Z[-3:, 2], 2).tolist()}——最后一次跳得很大 → 切成 3 簇；ARI {adjusted_rand_score(y, labels3):.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    for j in range(1, 4):
        ax.scatter(X[labels3 == j, 0], X[labels3 == j, 1], s=20, color=PALETTE[j - 1])
    for i, (xx, yy) in enumerate(X):
        ax.annotate(str(i), (xx, yy), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_title("30 个点，切成 3 簇", fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1]
    dendrogram(Z, ax=ax, color_threshold=Z[-2, 2], leaf_font_size=5, above_threshold_color=C["gray"])
    ax.axhline(Z[-2, 2] * 1.05, color=C["red"], ls="--", lw=0.8)
    ax.set_title("树状图：纵轴是合并时的距离；在红线处横切得 3 簇", fontsize=8.5); ax.set_yticks([])
    save(fig, "07-hierarchical-dendrogram")
    print("  不用先定 k：看树状图上哪一层合并距离突然变大，在那里切；代价 O(n²) 内存，几万个点以上就不合适")
    print()


# ---------------- 7. 真实语料：这批数据里有什么 ----------------
def exp_corpus():
    print("=== 7. 一个真实的小语料：78 句 → Qwen2.5-0.5B 句向量 → K-Means → 每簇抽几条看 ===")
    from _sentences import embeddings
    E, texts, labels, names = embeddings()
    En = E / np.linalg.norm(E, axis=1, keepdims=True)                      # 归一化：欧氏距离 ↔ 余弦相似度
    print(f"  句向量形状 {E.shape}（{len(names) - 1} 个主题各 12 句 + 6 句同一模板的近重复）")
    for k in (4, 7, 10):
        km = KMeans(k, n_init=10, random_state=0).fit(En)
        print(f"  k = {k:>2}: ARI（与真实主题）{adjusted_rand_score(labels, km.labels_):.3f}，每簇大小 {sorted(np.bincount(km.labels_).tolist(), reverse=True)}")
    km = KMeans(7, n_init=10, random_state=0).fit(En)
    print("  k = 7 时每簇抽 2 条：")
    for j in range(7):
        idx = np.where(km.labels_ == j)[0]
        maj = names[np.bincount(labels[idx]).argmax()]
        print(f"    簇 {j}（{len(idx)} 条，主要是「{maj}」）：{texts[idx[0]][:22]}… / {texts[idx[1]][:22]}…")
    print("  模板文本自成一簇——这就是「找垃圾簇」；簇的大小就是主题占比——这就是「按簇配比」的起点")
    print()


EXPS = {"iterate": exp_iterate, "kmeans": exp_kmeans, "choose_k": exp_choose_k, "init": exp_init,
        "dbscan": exp_dbscan, "hierarchical": exp_hierarchical, "corpus": exp_corpus}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
