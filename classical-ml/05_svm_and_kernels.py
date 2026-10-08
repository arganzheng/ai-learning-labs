"""SVM 与核方法（经典 ML 05）：最大间隔、hinge loss、手写线性 SVM、软间隔的 C、核技巧（把圆环升到三维）、RBF 与 γ、
核 = 相似度（与 attention 的对应）、以及 SVM 的训练复杂度为什么让它在大数据上退场。
https://arganzheng.life/svm-and-kernel-methods.html

    python 05_svm_and_kernels.py          # 全部：margin hinge primal softc kernel rbf attention scale dual cscale cost analogy
    python 05_svm_and_kernels.py kernel

图输出到 out/05-*.svg。
"""
import sys
import time

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


# ---------------- 9. 从原问题到「只依赖内积」：w 是样本的加权和，核化只换一个矩阵 ----------------
def kernel_svm_sgd(K, y, C_=1.0, lr=0.001, epochs=100, seed=0):
    """fit_linear_svm 的核化版：不存 w，只存每个样本的系数 α_i，f(x_j) = Σ_i α_i y_i K(x_i, x_j) + b。
    w ← (1 − lr)·w 变成 α ← (1 − lr)·α；w += lr·C·y_i·x_i 变成 α_i += lr·C。同一 seed 下与原问题逐步相同。"""
    r = np.random.default_rng(seed)
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    for _ in range(epochs):
        for i in r.permutation(n):
            # !ref dual-margin
            margin = y[i] * ((alpha * y) @ K[:, i] + b)
            # !ref dual-shrink
            alpha *= 1 - lr
            if margin < 1:
                # !ref dual-add
                alpha[i] += lr * C_
                b += lr * C_ * y[i]
    return alpha, b


def kernel_svm_decision(K_new, alpha, y, b):
    """K_new[j, i] = K(x_new_j, x_i)：新点与全部训练点的核，只有 α_i ≠ 0 的训练点真的参与。"""
    return K_new @ (alpha * y) + b


def rbf_kernel(A, B, gamma):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0))


def exp_dual():
    print("=== 9. 从原问题到对偶：w = Σ αᵢyᵢxᵢ，训练与预测只用到样本两两的内积 ===")
    X, y = make_blobs(n_samples=60, centers=[[-1.5, -1.5], [1.5, 1.5]], cluster_std=0.7, random_state=3)
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    sv = clf.support_vectors_
    coef = clf.dual_coef_[0]
    w_from_alpha = coef @ sv
    print(f"  第 1 节那 60 个点：支持向量 {np.round(sv, 3).tolist()}，系数 αᵢyᵢ = {np.round(coef, 3).tolist()}")
    print(f"  w = Σ αᵢyᵢxᵢ = {np.round(w_from_alpha, 3)}；sklearn 的 coef_ = {np.round(clf.coef_[0], 3)}；最大差 {np.abs(w_from_alpha - clf.coef_[0]).max():.1e}")
    print(f"  两个支持向量的 αᵢyᵢ 大小相等符号相反（Σ αᵢyᵢ = {coef.sum():.1e}），所以 w 平行于两点的连线 x₊ − x₋ = {np.round(sv[1] - sv[0], 3)}")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=0, class_sep=1.2, random_state=1)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    ypm = 2 * ytr - 1
    w, b = fit_linear_svm(Xtr, ypm, C_=1.0, lr=0.001, epochs=100)
    K = Xtr @ Xtr.T
    alpha, b_k = kernel_svm_sgd(K, ypm, C_=1.0, lr=0.001, epochs=100)
    w_k = (alpha * ypm) @ Xtr
    f_primal = Xte @ w + b
    f_dual = kernel_svm_decision(Xte @ Xtr.T, alpha, ypm, b_k)
    n_touched = int((alpha > 1e-12 * alpha.max()).sum())
    print(f"  第 3 节的 1000 个样本：原问题 SGD 的 w 与核化版还原的 Σ αᵢyᵢxᵢ 最大差 {np.abs(w - w_k).max():.1e}，300 个测试点决策值最大差 {np.abs(f_primal - f_dual).max():.1e}")
    print(f"  核化版只存 α（{len(alpha)} 个数），其中 {n_touched} 个非零——这些就是训练中至少一次间隔不足 1 的点；其余 {len(alpha) - n_touched} 个点从未进入 w")
    Xc, yc = make_circles(n_samples=300, factor=0.4, noise=0.08, random_state=0)
    ycpm = 2 * yc - 1
    gamma = 1.0
    Kc = rbf_kernel(Xc, Xc, gamma)
    ac, bc = kernel_svm_sgd(Kc, ycpm, C_=100.0, lr=0.001, epochs=100)
    acc_c = ((kernel_svm_decision(Kc, ac, ycpm, bc) > 0) == (ycpm > 0)).mean()
    n_sv_c = int((ac > 1e-12 * ac.max()).sum())
    sk_c = SVC(kernel="rbf", gamma=gamma, C=100.0 / len(Xc)).fit(Xc, yc)
    print(f"  同一段代码、把 K = XXᵀ 换成 RBF 核矩阵（C′ = 100，即 C_sum = 100/300）：圆环数据训练准确率 {acc_c:.3f}，非零 α {n_sv_c} 个；sklearn SVC(rbf, C=100/n) 支持向量 {len(sk_c.support_vectors_)} 个、准确率 {sk_c.score(Xc, yc):.3f}")
    print("  代码里从头到尾没有出现 φ(x)：升维只发生在「K 是某个 φ 的内积」这个事实里，算法只看 K")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    lim = (-1.5, 1.5, -1.5, 1.5)
    plot_boundary(axes[0], lambda P: kernel_svm_decision(rbf_kernel(P, Xc, gamma), ac, ycpm, bc), Xc, yc,
                  f"手写核化 SGD（RBF γ=1）：训练 {acc_c:.2f}，非零 α {n_sv_c} 个", lim=lim, sv=Xc[ac > 1e-12 * ac.max()])
    order = np.argsort(-ac)
    axes[1].bar(range(len(ac)), ac[order] / ac.max(), color=C["blue"], width=1.0)
    axes[1].set_xlabel("300 个训练点（按 α 从大到小排）"); axes[1].set_ylabel("α / max α")
    axes[1].set_title(f"α 的分布：{n_sv_c} 个非零，其余恰好为 0", fontsize=8)
    save(fig, "05-dual-kernel-sgd")
    print()


# ---------------- 10. C 的尺度：损失按求和还是按平均，C 差一个 n ----------------
def exp_cscale():
    print("=== 10. C 的尺度：½||w||² + C·Σ hinge（LinearSVC/SVC）vs ½||w||² + C'·mean hinge（手写 SGD）vs λ/2·||w||² + mean hinge（SGDClassifier）===")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=0, class_sep=1.2, random_state=1)
    X = StandardScaler().fit_transform(X)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    ypm = 2 * ytr - 1
    n = len(Xtr)

    def j_sum(w, b, c_sum):
        return 0.5 * w @ w + c_sum * np.maximum(0, 1 - ypm * (Xtr @ w + b)).sum()

    def cos(u, v):
        return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))

    print(f"  n = {n}。三种写法同一个最优解的换算：C_sum = C'/n，λ = 1/(n·C_sum) = 1/C'")
    print(f"  {'手写 C′':>8} {'||w|| 手写':>10} {'LinearSVC(C′/n)':>16} {'余弦':>6} {'LinearSVC(C′)':>14} {'余弦':>6} {'SGD(α=1/C′)':>12} {'余弦':>6}")
    for c_hand in (0.01, 1.0, 100.0):
        w, b = fit_linear_svm(Xtr, ypm, C_=c_hand, lr=0.001, epochs=100)
        right = LinearSVC(C=c_hand / n, loss="hinge", max_iter=50000).fit(Xtr, ytr)
        wrong = LinearSVC(C=c_hand, loss="hinge", max_iter=50000).fit(Xtr, ytr)
        sgd = SGDClassifier(loss="hinge", alpha=1.0 / c_hand, max_iter=2000, tol=1e-6, random_state=0).fit(Xtr, ytr)
        print(f"  {c_hand:>8g} {np.linalg.norm(w):>10.3f} {np.linalg.norm(right.coef_[0]):>16.3f} {cos(w, right.coef_[0]):>6.3f}"
              f" {np.linalg.norm(wrong.coef_[0]):>14.3f} {cos(w, wrong.coef_[0]):>6.3f} {np.linalg.norm(sgd.coef_[0]):>12.3f} {cos(w, sgd.coef_[0]):>6.3f}")
    w, b = fit_linear_svm(Xtr, ypm, C_=1.0, lr=0.001, epochs=100)
    right = LinearSVC(C=1.0 / n, loss="hinge", max_iter=50000).fit(Xtr, ytr)
    print(f"  C′ = 1 时按「求和」目标 ½||w||² + (1/n)·Σ hinge 打分：手写解 {j_sum(w, b, 1.0 / n):.4f}，LinearSVC(C=1/n) 的解 {j_sum(right.coef_[0], right.intercept_[0], 1.0 / n):.4f}（同一个目标，后者是精确求解器）")
    for m in (1, 2, 4):
        Xm = np.tile(Xtr, (m, 1))
        ym = np.tile(ytr, m)
        clf = LinearSVC(C=0.01, loss="hinge", max_iter=50000).fit(Xm, ym)
        print(f"  把训练集复制 {m} 份（n = {len(Xm)}）、C_sum 固定 0.01：||w|| = {np.linalg.norm(clf.coef_[0]):.3f}，间隔 {1 / np.linalg.norm(clf.coef_[0]):.3f}——按求和的 C 随 n 变大等于正则在变弱")
    print("  所以换数据量、换库、换论文时先问一句：它的 C 乘的是 Σ 还是 mean？差的那个 n 就是正则强度差的倍数")
    print()


# ---------------- 11. 核化不免费：省掉的是 φ 的维数，付出的是 n² 的核矩阵与随 n 增长的支持向量 ----------------
def exp_cost():
    print("=== 11. 核化不免费：核矩阵 n²、支持向量数随 n 增长，预测时间随支持向量数增长 ===")
    Xall, yall = make_classification(n_samples=21000, n_features=20, n_informative=8, flip_y=0.1, random_state=0)
    Xall = StandardScaler().fit_transform(Xall)
    Xte, yte = Xall[:5000], yall[:5000]
    print(f"  {'n':>6} {'支持向量':>8} {'占比':>6} {'核矩阵 n²·8B':>12} {'训练':>8} {'预测 5000 点':>12}")
    ns = [1000, 2000, 4000, 8000, 16000]
    nsv, t_fit, t_pred = [], [], []
    for n in ns:
        X, y = Xall[5000:5000 + n], yall[5000:5000 + n]
        m = SVC(kernel="rbf")
        t0 = time.time()
        m.fit(X, y)
        t_fit.append(time.time() - t0)
        t0 = time.time()
        m.predict(Xte)
        t_pred.append(time.time() - t0)
        nsv.append(int(m.n_support_.sum()))
        print(f"  {n:>6} {nsv[-1]:>8} {nsv[-1] / n:>6.0%} {n * n * 8 / 1e6:>10.0f}MB {t_fit[-1]:>7.2f}s {t_pred[-1]:>11.2f}s")
    print(f"  n 乘 16，支持向量乘 {nsv[-1] / nsv[0]:.1f}（有标签噪声时支持向量数近似随 n 线性增长），预测时间乘 {t_pred[-1] / t_pred[0]:.1f}；模型大小 = 支持向量数 × d")
    X, y = Xall[5000:], yall[5000:]
    t0 = time.time()
    poly = SVC(kernel="poly", degree=2, coef0=1, gamma=1.0 / X.shape[1]).fit(X, y)
    t_poly = time.time() - t0
    t0 = time.time()
    expl = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearSVC(C=0.01, max_iter=20000)).fit(X, y)
    t_expl = time.time() - t0
    d_expl = expl[0].n_output_features_
    print(f"  同一个二次多项式空间：核 SVC(poly, degree=2) 训练 {t_poly:.1f}s、支持向量 {poly.n_support_.sum()} 个、测试 {poly.score(Xte, yte):.3f}；"
          f"显式造 {d_expl} 维特征 + LinearSVC 训练 {t_expl:.1f}s、测试 {expl.score(Xte, yte):.3f}")
    print("  d = 20 时显式二次特征只有 231 维，线性求解是 O(n·231)；核化省的是 φ 的维数，付的是 O(n²) 的核计算与随 n 增长的支持向量——维数低、样本多时显式升维反而便宜")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    axes[0].plot(ns, nsv, "o-", color=C["red"], label="支持向量数")
    axes[0].plot(ns, ns, ls="--", color=C["gray"], lw=0.8, label="n（上限）")
    axes[0].set_xlabel("训练样本数 n"); axes[0].set_ylabel("支持向量数"); axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("带 10% 标签噪声：支持向量随 n 近似线性增长", fontsize=8)
    axes[1].loglog(ns, t_fit, "o-", color=C["blue"], label="训练时间")
    axes[1].loglog(ns, t_pred, "s-", color=C["orange"], label="预测 5000 点的时间")
    axes[1].set_xlabel("训练样本数 n（对数轴）"); axes[1].set_ylabel("秒（对数轴）"); axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("RBF SVC：训练与预测时间", fontsize=8)
    save(fig, "05-kernel-cost")
    print()


# ---------------- 12. attention 与核平滑的类比在哪一层成立 ----------------
def exp_analogy():
    print("=== 12. attention 与核平滑：三步公式一样；相似度是不是核、有没有「训练集」，不一样 ===")
    r = np.random.default_rng(0)
    X = r.normal(size=(40, 8))
    K = rbf_kernel(X, X, gamma=0.5)
    eig_k = np.linalg.eigvalsh(K)
    print(f"  40 个 8 维点的 RBF 核矩阵：对称 {np.allclose(K, K.T)}，最小特征值 {eig_k.min():.2e}（≥ 0：是一个合法的核 / 内积）")
    Wq, Wk = r.normal(size=(8, 8)) / 8 ** 0.5, r.normal(size=(8, 8)) / 8 ** 0.5
    S = (X @ Wq) @ (X @ Wk).T / 8 ** 0.5
    eig_s = np.linalg.eigvalsh((S + S.T) / 2)
    print(f"  attention 的打分 S = (XW_Q)(XW_K)ᵀ/√d，W_Q ≠ W_K：|S − Sᵀ| 最大 {np.abs(S - S.T).max():.2f}（不对称：i 看 j 与 j 看 i 不一样），对称部分最小特征值 {eig_s.min():.2f}（< 0：不是任何 φ 的内积）")
    S2 = (X @ Wq) @ (X @ Wq).T / 8 ** 0.5
    eig_s2 = np.linalg.eigvalsh(S2)
    print(f"  若令 W_K = W_Q：对称 {np.allclose(S2, S2.T)}，最小特征值 {eig_s2.min():.2e}——这时 exp(S) 才是一个核（exp 保持半正定），类比才能下到「核」这一层")
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    Wn = K / K.sum(1, keepdims=True)
    print(f"  但第三步一样：两者的权重矩阵每行和都是 1（attention {A.sum(1).min():.3f}–{A.sum(1).max():.3f}，核回归 {Wn.sum(1).min():.3f}–{Wn.sum(1).max():.3f}），输出都是「值」的凸组合")
    print(f"  参数：核回归 0 个可学参数（只有 γ）、键是训练集本身（n = 40 行要存下来）；attention 这一层 3 个 8×8 矩阵共 {3 * 64} 个参数，键是同一次前向里的输入，不是训练集")
    print("  所以类比成立在「相似度 → 归一化 → 加权值」这一层；「相似度是核」只在 Q、K 共享投影时成立；「它是 SVM」不成立——没有间隔目标、没有逐样本的 α")
    print()


EXPS = {"margin": exp_margin, "hinge": exp_hinge, "primal": exp_primal, "softc": exp_softc, "kernel": exp_kernel,
        "rbf": exp_rbf, "attention": exp_attention, "scale": exp_scale,
        "dual": exp_dual, "cscale": exp_cscale, "cost": exp_cost, "analogy": exp_analogy}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
