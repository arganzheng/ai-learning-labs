"""SVM 与核方法（经典 ML 05）：最大间隔、hinge loss、手写线性 SVM、软间隔的 C、核技巧（把圆环升到三维）、RBF 与 γ、
核 = 相似度（与 attention 的对应）、以及 SVM 的训练复杂度为什么让它在大数据上退场。
https://arganzheng.life/svm-and-kernel-methods.html

    python 05_svm_and_kernels.py          # 全部：margin hinge primal softc kernel rbf attention scale
    python 05_svm_and_kernels.py kernel

图输出到 out/05-*.svg。
"""
import sys
import time

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from _plot import C, plt, save


# ---------------- 手写：线性 SVM 的原问题（hinge loss + L2），次梯度下降 ----------------
def fit_linear_svm(X, y, C_=1.0, lr=0.01, epochs=200, seed=0):
    """y ∈ {−1, +1}。最小化 ½||w||² + C·Σ max(0, 1 − y(wᵀx + b))。"""
    r = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for ep in range(epochs):
        for i in r.permutation(n):                                   # ① 逐个样本（SGD）
            margin = y[i] * (X[i] @ w + b)                           # ② 这个样本的间隔 y·f(x)
            if margin < 1:                                           # ③ 间隔不足 1：hinge 有梯度
                w -= lr * (w - C_ * y[i] * X[i]); b += lr * C_ * y[i]
            else:                                                    # ④ 间隔够：只有正则项在拉 w
                w -= lr * w
    return w, b


def plot_boundary(ax, decision, X, y, title, lim=None, res=80, sv=None):
    lim = lim or (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    xx, yy = np.meshgrid(np.linspace(lim[0], lim[1], res), np.linspace(lim[2], lim[3], res))
    Z = decision(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]], alpha=0.15)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["k", "k", "k"], linestyles=["--", "-", "--"], linewidths=[0.7, 1.3, 0.7])
    ax.scatter(X[y <= 0, 0], X[y <= 0, 1], s=9, color=C["blue"]); ax.scatter(X[y > 0, 0], X[y > 0, 1], s=9, color=C["red"])
    if sv is not None:
        ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolors="none", edgecolors="k", lw=0.8)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


# ---------------- 1. 最大间隔：很多条线都能分开，选离两边都最远的 ----------------
def exp_margin():
    print("=== 1. 最大间隔：可分数据上，哪条线最好 ===")
    X, y = make_blobs(n_samples=60, centers=[[-1.5, -1.5], [1.5, 1.5]], cluster_std=0.7, random_state=3)
    clf = SVC(kernel="linear", C=1e6).fit(X, y)                       # 硬间隔
    w, b = clf.coef_[0], clf.intercept_[0]
    margin = 1 / np.linalg.norm(w)
    print(f"  最大间隔直线：w = {np.round(w, 3)}, b = {b:.3f}；间隔宽度（到边界的距离）= 1/||w|| = {margin:.3f}")
    print(f"  支持向量：{len(clf.support_vectors_)} 个（{len(X)} 个训练点里只有它们决定了这条线）")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    ax = axes[0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, color=C["blue"]); ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, color=C["red"])
    xs = np.linspace(-4, 4, 2)
    for (ww, bb), col in ((((1.0, 0.45), 0.3), C["gray"]), (((0.45, 1.0), -0.3), C["gray"]), (((1.0, 1.0), 0.0), C["orange"])):
        sep = ((X @ np.array(ww) + bb > 0) == (y == 1)).mean()
        assert sep == 1.0, f"直线 {ww}, {bb} 没有完全分开：{sep}"
        ax.plot(xs, -(ww[0] * xs + bb) / ww[1], color=col, lw=1.2)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_title("三条都能把训练点分开的直线", fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
    plot_boundary(axes[1], clf.decision_function, X, y, f"SVM 选间隔最大的那条；虚线是间隔边界，圈出的是支持向量（{len(clf.support_vectors_)} 个）",
                  lim=(-4, 4, -4, 4), sv=clf.support_vectors_)
    save(fig, "05-max-margin")
    print()


# ---------------- 2. hinge loss vs 逻辑回归的 loss ----------------
def exp_hinge():
    print("=== 2. hinge loss：间隔够 1 就不罚，不够才罚 ===")
    m = np.linspace(-2, 3, 300)
    hinge = np.maximum(0, 1 - m)
    logistic = np.log1p(np.exp(-m))
    zero_one = (m < 0).astype(float)
    for mm in (-1, 0, 0.5, 1, 2):
        print(f"  间隔 y·f(x) = {mm:+.1f}: hinge {max(0, 1-mm):.2f}  逻辑回归 {np.log1p(np.exp(-mm)):.2f}  0-1 {int(mm < 0)}")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.plot(m, zero_one, color=C["gray"], lw=1.2, ls=":", label="0-1 损失（分错就是 1，不可优化）")
    ax.plot(m, hinge, color=C["red"], lw=1.8, label="hinge：max(0, 1 − y·f(x))，SVM")
    ax.plot(m, logistic, color=C["blue"], lw=1.8, label="逻辑回归：log(1 + e^(−y·f(x)))")
    ax.axvline(1, color=C["gray"], lw=0.6, ls="--"); ax.axvline(0, color=C["gray"], lw=0.6)
    ax.set_xlabel("间隔 y · f(x)（正 = 分对，越大越自信）"); ax.set_ylabel("损失"); ax.set_ylim(0, 3.2)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "05-hinge-vs-logistic")
    print("  hinge 在间隔 ≥ 1 处恰好为零：分对且够远的点对模型没有影响——它们不是支持向量")
    print()


# ---------------- 3. 手写线性 SVM 对 sklearn ----------------
def exp_primal():
    print("=== 3. 手写线性 SVM（hinge + L2 的次梯度下降）vs sklearn LinearSVC ===")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=0, class_sep=1.2, random_state=1)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ypm = 2 * ytr - 1                                                 # 标签换成 ±1
    t0 = time.time(); w, b = fit_linear_svm(Xtr, ypm, C_=1.0, lr=0.001, epochs=100); t_hand = time.time() - t0
    acc = ((Xte @ w + b > 0) == yte).mean()
    sk = LinearSVC(C=1.0 / len(Xtr) * 1.0, loss="hinge", max_iter=20000).fit(Xtr, ytr)   # sklearn 的 C 对应 Σ 而非均值，换算一下
    cos = (w @ sk.coef_[0]) / np.linalg.norm(w) / np.linalg.norm(sk.coef_[0])
    print(f"  手写：{t_hand:.2f}s，测试准确率 {acc:.3f}；LinearSVC：{sk.score(Xte, yte):.3f}；两个 w 方向的余弦相似度 {cos:.4f}")
    lr = LogisticRegression().fit(Xtr, ytr)
    print(f"  同一份数据逻辑回归 {lr.score(Xte, yte):.3f}——线性可分程度高的数据上，两种线性分类器差别很小")
    print()


# ---------------- 4. 软间隔：C 的作用 ----------------
def exp_softc():
    print("=== 4. 软间隔：C 小 = 允许更多点越过间隔（边界更平、更稳），C 大 = 尽量不犯错（边界更贴数据）===")
    X, y = make_blobs(n_samples=120, centers=[[-1, -1], [1, 1]], cluster_std=1.1, random_state=5)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for ax, c in zip(axes, (0.01, 1, 100)):
        clf = SVC(kernel="linear", C=c).fit(X, y)
        n_sv = len(clf.support_vectors_)
        print(f"  C = {c:<5}: 支持向量 {n_sv:>3} 个，间隔宽度 {1/np.linalg.norm(clf.coef_[0]):.2f}，训练准确率 {clf.score(X, y):.3f}")
        plot_boundary(ax, clf.decision_function, X, y, f"C = {c}：{n_sv} 个支持向量，间隔 {1/np.linalg.norm(clf.coef_[0]):.2f}", lim=(-4, 4, -4, 4), sv=clf.support_vectors_)
    save(fig, "05-soft-margin-c")
    print("  C 是正则化的倒数：C 小 → 更看重 ||w|| 小（间隔宽）→ 更多点落在间隔内成为支持向量")
    print()


# ---------------- 5. 核技巧：二维不可分，升到三维就可分 ----------------
def exp_kernel():
    print("=== 5. 核技巧：圆环数据在二维分不开，加一个特征 x₁² + x₂² 就是一个平面能分 ===")
    X, y = make_circles(n_samples=300, factor=0.4, noise=0.08, random_state=0)
    lin = SVC(kernel="linear").fit(X, y)
    Z = np.c_[X, (X ** 2).sum(1)]                                     # 显式升维：第三个特征是到原点距离的平方
    lin3 = SVC(kernel="linear").fit(Z, y)
    rbf = SVC(kernel="rbf", gamma=1.0).fit(X, y)
    print(f"  二维线性 SVM 训练准确率 {lin.score(X, y):.3f}（一条直线切圆环，只能对一半）")
    print(f"  加第三维 r² 后线性 SVM {lin3.score(Z, y):.3f}——三维里一个水平的平面就把内圈与外圈分开")
    print(f"  不显式升维、直接用 RBF 核的 SVM {rbf.score(X, y):.3f}（核函数在算「升维后的内积」，但从不真的构造那些特征）")
    fig = plt.figure(figsize=(7.6, 2.8))
    ax = fig.add_subplot(1, 3, 1)
    plot_boundary(ax, lin.decision_function, X, y, f"二维：线性 SVM {lin.score(X, y):.2f}", lim=(-1.5, 1.5, -1.5, 1.5))
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], Z[y == 0, 2], s=6, color=C["blue"]); ax.scatter(Z[y == 1, 0], Z[y == 1, 1], Z[y == 1, 2], s=6, color=C["red"])
    xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 8), np.linspace(-1.3, 1.3, 8))
    w3, b3 = lin3.coef_[0], lin3.intercept_[0]
    ax.plot_surface(xx, yy, -(w3[0] * xx + w3[1] * yy + b3) / w3[2], alpha=0.25, color=C["gray"])
    ax.set_title("加一维 z = x₁² + x₂²：一个平面分开", fontsize=8); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=15, azim=-60)
    ax = fig.add_subplot(1, 3, 3)
    plot_boundary(ax, rbf.decision_function, X, y, f"RBF 核 SVM：{rbf.score(X, y):.2f}，边界是圆", lim=(-1.5, 1.5, -1.5, 1.5))
    save(fig, "05-kernel-trick-circles")
    print()


# ---------------- 6. RBF 核的 γ ----------------
def exp_rbf():
    print("=== 6. RBF 核：K(x, x') = exp(−γ||x − x'||²)；γ 大 = 每个点的影响范围小 = 边界更弯 ===")
    X, y = make_moons(n_samples=400, noise=0.22, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for ax, g in zip(axes, (0.1, 1, 200)):
        clf = SVC(kernel="rbf", gamma=g, C=1.0).fit(Xtr, ytr)
        print(f"  γ = {g:<4}: 训练 {clf.score(Xtr, ytr):.3f} / 测试 {clf.score(Xte, yte):.3f}，支持向量 {len(clf.support_vectors_)} 个")
        plot_boundary(ax, clf.decision_function, Xtr, ytr, f"γ = {g}：训练 {clf.score(Xtr, ytr):.2f} / 测试 {clf.score(Xte, yte):.2f}", lim=(-2, 3, -1.5, 2))
    save(fig, "05-rbf-gamma")
    print("  γ 太大 → 每个训练点周围一个小岛，训练 100% 测试掉——又是过拟合；γ 与 C 一起用验证集选")
    print()


# ---------------- 7. 核 = 相似度：Nadaraya-Watson 核回归 与 attention ----------------
def exp_attention():
    print("=== 7. 核 = 相似度加权：核回归的公式与 attention 的公式是同一个 ===")
    r = np.random.default_rng(0)
    xk = np.sort(r.uniform(0, 6, 30)); vk = np.sin(xk) + r.normal(0, 0.15, 30)      # 30 个「键」及其「值」
    xq = np.linspace(0, 6, 200)                                                    # 200 个「查询」
    def kernel_regression(xq, xk, vk, gamma):
        K = np.exp(-gamma * (xq[:, None] - xk[None, :]) ** 2)      # ① 查询与每个键的相似度（RBF 核）
        W = K / K.sum(1, keepdims=True)                             # ② 每行归一化成权重——这一步就是 softmax 的作用
        return W @ vk                                               # ③ 用权重加权「值」
    def attention(q, k, v, scale):
        S = q @ k.T / scale                                         # ① 查询与键的点积相似度
        W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)   # ② softmax
        return W @ v                                                # ③ 加权值
    # 点积注意力在一维上等价于 RBF 核（差一个只依赖 q、k 自身范数的因子；把 ||q||² 与 ||k||² 补进去就完全相同）
    gamma = 2.0
    q = np.c_[xq, -gamma * xq ** 2 / 1.0, np.ones_like(xq)]                       # 把 exp(−γ(q−k)²) 拆成 exp(2γqk − γq² − γk²)
    k = np.c_[2 * gamma * xk, np.ones_like(xk), -gamma * xk ** 2]
    y_kr = kernel_regression(xq, xk, vk, gamma)
    y_att = attention(q, k, vk, scale=1.0)
    print(f"  RBF 核回归与用点积 + softmax 写的 attention，在 200 个查询点上的最大差 {np.abs(y_kr - y_att).max():.1e}")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.scatter(xk, vk, s=18, color=C["blue"], zorder=3, label="30 个键值对 (x, v)")
    for g, col in ((0.3, C["gray"]), (2, C["red"]), (30, C["orange"])):
        ax.plot(xq, kernel_regression(xq, xk, vk, g), color=col, lw=1.4, label=f"γ = {g}")
    ax.plot(xq, np.sin(xq), ls="--", color=C["gray"], lw=0.8)
    ax.set_xlabel("查询 x"); ax.set_ylabel("加权后的值"); ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.set_title("核回归：查询与每个键算相似度、归一化成权重、加权求值——attention 的三步", fontsize=8.5)
    save(fig, "05-kernel-regression-attention")
    print("  attention(Q, K, V) = softmax(QKᵀ/√d) V：相似度（点积核）→ 归一化 → 加权值。区别是 Q、K、V 都是学出来的投影")
    print()


# ---------------- 8. 为什么大数据上 SVM 退场：训练时间随 n 的增长 ----------------
def exp_scale():
    print("=== 8. 训练时间随样本数增长：核 SVM 是 n² 到 n³，线性模型是 n ===")
    print(f"  {'n':>7} {'RBF SVM':>10} {'线性 SVM':>10} {'逻辑回归':>10}")
    for n in (1000, 4000, 16000, 64000):
        X, y = make_classification(n_samples=n, n_features=20, n_informative=8, random_state=0)
        X = StandardScaler().fit_transform(X)
        ts = []
        for m in (SVC(kernel="rbf"), LinearSVC(max_iter=5000), LogisticRegression(max_iter=2000)):
            t0 = time.time(); m.fit(X, y); ts.append(time.time() - t0)
        print(f"  {n:>7} {ts[0]:>9.2f}s {ts[1]:>9.2f}s {ts[2]:>9.2f}s")
    print("  n 乘 4，RBF SVM 的时间乘十几倍；预测也要与所有支持向量算核——样本到百万级只剩线性模型与树，深度学习时代核 SVM 因此退场")
    print()


EXPS = {"margin": exp_margin, "hinge": exp_hinge, "primal": exp_primal, "softc": exp_softc, "kernel": exp_kernel,
        "rbf": exp_rbf, "attention": exp_attention, "scale": exp_scale}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
