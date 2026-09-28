"""案例（经典 ML 08）：Eigenfaces——用 PCA 认人脸（Turk & Pentland 1991）。
https://arganzheng.life/dimensionality-reduction-pca-svd-tsne-and-umap.html

    python case_08_eigenfaces.py

数据：Olivetti / AT&T 人脸库，40 个人 × 10 张 = 400 张 64×64 灰度图（4,096 维）。
图输出到 out/case-08-*.svg。
"""
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from _data import olivetti_faces
from _plot import C, plt, save


def main():
    X, y = olivetti_faces()
    n, d = X.shape
    print(f"数据：{n} 张 64×64 = {d} 维的人脸，{len(np.unique(y))} 个人每人 10 张（不同表情、眼镜、光照）")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)   # 每人 7 张训、3 张测
    print(f"划分：每人 7 张训练（{len(Xtr)}）、3 张测试（{len(Xte)}）")

    # 1. PCA：平均脸 + 特征脸
    pca = PCA(n_components=150, whiten=False, random_state=0).fit(Xtr)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    print(f"\n=== 1. 4,096 维里有多少是'真的'===")
    for k in [1, 5, 10, 20, 50, 100, 150]:
        print(f"  前 {k:>3} 个主成分解释 {cum[k - 1]:.1%} 的方差")
    print(f"  训练集只有 {len(Xtr)} 张，所以最多 {len(Xtr) - 1} 个主成分就能 100% 重建它——4,096 维里数据只占一个 {len(Xtr) - 1} 维的子空间；"
          f"而前 50 个就解释了 {cum[49]:.0%}：人脸的变化远比像素数少。")

    # 2. 重建：k 个主成分能还原多少
    print(f"\n=== 2. 用 k 个数重建一张 4,096 个像素的脸 ===")
    face = Xte[0]
    recs = {}
    for k in [5, 20, 50, 150]:
        p = PCA(n_components=k, random_state=0).fit(Xtr)
        rec = p.inverse_transform(p.transform(face[None]))[0]
        recs[k] = rec
        print(f"  k = {k:>3}：每张脸存 {k} 个数而不是 4,096 个（压缩 {d / k:.0f}×），重建均方误差 {np.mean((rec - face) ** 2):.4f}")

    # 3. 识别：PCA 降维后接分类器
    print(f"\n=== 3. 认人：先降维再分类（Turk & Pentland 1991 的做法）===")
    print(f"  {'特征':<24}{'分类器':<14}{'准确率':>7}{'耗时':>8}")
    rows = []
    for name, feat_k in [("原始 4,096 像素", None), ("PCA 10", 10), ("PCA 50", 50), ("PCA 150", 150)]:
        if feat_k is None:
            Ftr, Fte = Xtr, Xte
        else:
            p = PCA(n_components=feat_k, whiten=True, random_state=0).fit(Xtr)
            Ftr, Fte = p.transform(Xtr), p.transform(Xte)
        for cname, clf in [("1-NN", KNeighborsClassifier(1)), ("SVM(RBF)", SVC(C=10, gamma="scale")), ("逻辑回归", LogisticRegression(max_iter=3000, C=1))]:
            t = time.time(); acc = clf.fit(Ftr, ytr).score(Fte, yte); dt = time.time() - t
            rows.append((name, cname, acc)); print(f"  {name:<24}{cname:<14}{acc:>7.3f}{dt:>7.2f}s")
    print("  解读：400 张干净、对齐好的图上，原始像素直接喂分类器就最准（逻辑回归 0.96）——PCA 在这里不是为了准确率，"
          "是为了把 4,096 维压成 50 维：SVM 0.94 几乎不掉，特征少了 80 倍，1991 年的机器才跑得动。"
          "PCA 150 + 1-NN 掉到 0.80 是一个真实的坑：whiten 把每个主成分的方差拉平，第 100–150 个主成分本来是噪声，"
          "被放大成和前几个一样重要，欧氏距离就被噪声主导了——保留多少主成分不是越多越好。"
          "Turk & Pentland 1991 的做法就是'投到特征脸空间、找最近的人'——PCA + 1-NN 那一行。")

    # 4. 特征脸空间里两个人的脸各在哪：前两个主成分
    Z = pca.transform(X)
    # 图 1：平均脸 + 前 15 张特征脸
    fig, axes = plt.subplots(2, 8, figsize=(7.6, 2.2))
    axes[0, 0].imshow(pca.mean_.reshape(64, 64), cmap="gray"); axes[0, 0].set_title("平均脸", fontsize=7)
    for i, ax in enumerate(axes.ravel()[1:]):
        ax.imshow(pca.components_[i].reshape(64, 64), cmap="gray"); ax.set_title(f"特征脸 {i + 1}\n{ev[i]:.1%}", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    save(fig, "case-08-eigenfaces")

    # 图 2：重建
    fig, axes = plt.subplots(1, 6, figsize=(7.6, 1.7))
    axes[0].imshow(face.reshape(64, 64), cmap="gray"); axes[0].set_title("原图\n4,096 个数", fontsize=7)
    axes[1].imshow(pca.mean_.reshape(64, 64), cmap="gray"); axes[1].set_title("k = 0\n平均脸", fontsize=7)
    for ax, k in zip(axes[2:], [5, 20, 50, 150]):
        ax.imshow(recs[k].reshape(64, 64), cmap="gray"); ax.set_title(f"k = {k}\n{k} 个数", fontsize=7)
    for ax in axes:
        ax.axis("off")
    save(fig, "case-08-reconstruction")

    # 图 3：解释方差 + 识别准确率
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    ax.plot(range(1, 151), cum * 100, c=C["blue"]); ax.set_xlabel("主成分个数 k"); ax.set_ylabel("累计解释方差 %")
    for k in [10, 50]:
        ax.axvline(k, ls=":", c=C["gray"], lw=0.8); ax.text(k + 2, 30, f"k={k}: {cum[k - 1]:.0%}", fontsize=7, color=C["gray"])
    ax.set_title("4,096 维里 50 个方向就够了")
    ax = axes[1]
    for cname, col in [("1-NN", C["orange"]), ("SVM(RBF)", C["green"]), ("逻辑回归", C["purple"])]:
        ax.plot(["像素", "PCA 10", "PCA 50", "PCA 150"], [a for f, c, a in rows if c == cname], "o-", c=col, label=cname)
    ax.set_ylabel("识别准确率（100 张测试）"); ax.set_ylim(0.5, 1.02); ax.legend(fontsize=7); ax.set_title("先降维再分类")
    save(fig, "case-08-accuracy")

    # 图 4：前两个主成分上 5 个人的分布
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for pid, col in zip([0, 7, 13, 21, 35], [C["red"], C["blue"], C["green"], C["orange"], C["purple"]]):
        m = y == pid
        ax.scatter(Z[m, 0], Z[m, 1], c=col, s=18, label=f"第 {pid} 人")
    ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2"); ax.legend(fontsize=7); ax.set_title("5 个人的 50 张脸投到前两个主成分上")
    save(fig, "case-08-pc-scatter")


if __name__ == "__main__":
    main()
