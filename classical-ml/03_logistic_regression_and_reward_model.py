"""逻辑回归与奖励模型（经典 ML 03）：sigmoid、交叉熵与它的梯度、手写 GD 对 sklearn、决策边界、softmax 回归、
以及"奖励模型 = 逻辑回归作用在两个回答的特征差上"。
https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html

    python 03_logistic_regression_and_reward_model.py          # 全部：sigmoid gradient boundary cancer softmax reward noise
    python 03_logistic_regression_and_reward_model.py reward

图输出到 out/03-*.svg。
"""
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _plot import C, plt, save


# ---------------- 手写实现 ----------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce(p, y):
    """二元交叉熵（每样本平均）。"""
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def fit_logistic(X, y, lr=0.1, steps=2000, l2=0.0):
    """逻辑回归的梯度下降。X 已含一列 1（偏置）。返回 w 与每步的 loss。"""
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(steps):
        p = sigmoid(X @ w)                              # ① 预测概率
        grad = X.T @ (p - y) / len(y) + l2 * w          # ② 梯度：(p − y) 乘特征，再平均
        w -= lr * grad                                  # ③ 走一步
        hist.append(bce(p, y))
    return w, np.array(hist)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)                     # 数值稳定：先减每行最大值
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_softmax(X, y, n_classes, lr=0.5, steps=500):
    """softmax 回归（多类逻辑回归）。W 是 [d, K]。"""
    Y = np.eye(n_classes)[y]                            # one-hot [n, K]
    W = np.zeros((X.shape[1], n_classes))
    for _ in range(steps):
        P = softmax(X @ W)                              # [n, K]
        W -= lr * X.T @ (P - Y) / len(y)                # 梯度形式与二元完全相同：(P − Y) 乘特征
    return W


def add_bias(X):
    return np.c_[np.ones(len(X)), X]


# ---------------- 1. sigmoid：把任意实数压到 (0, 1) ----------------
def exp_sigmoid():
    print("=== 1. sigmoid 与交叉熵 ===")
    for z in (-4, -2, 0, 2, 4):
        print(f"  z = {z:+d} → σ(z) = {sigmoid(z):.3f}")
    print(f"  σ(z) 的导数 σ(z)(1−σ(z)) 在 z=0 最大 = {sigmoid(0)*(1-sigmoid(0)):.2f}，z=±4 时只有 {sigmoid(4)*(1-sigmoid(4)):.3f}")
    z = np.linspace(-8, 8, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    ax.plot(z, sigmoid(z), color=C["blue"], lw=1.8)
    ax.axhline(0.5, color=C["gray"], ls="--", lw=0.8); ax.axvline(0, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("z = wᵀx + b（线性部分，任意实数）"); ax.set_ylabel("σ(z) = P(y = 1)")
    ax.set_title("sigmoid：z > 0 → 概率 > 0.5 → 判正类")
    ax = axes[1]
    p = np.linspace(0.001, 0.999, 300)
    ax.plot(p, -np.log(p), color=C["red"], lw=1.8, label="y = 1：−log p")
    ax.plot(p, -np.log(1 - p), color=C["blue"], lw=1.8, label="y = 0：−log(1 − p)")
    ax.plot(p, (p - 1) ** 2, color=C["gray"], ls="--", lw=1.2, label="对比：y = 1 时的平方误差 (p − 1)²")
    ax.set_ylim(0, 6); ax.set_xlabel("预测概率 p"); ax.set_ylabel("损失")
    ax.set_title("交叉熵：越自信地错，罚得越重（对数轴级）")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "03-sigmoid-and-cross-entropy")
    print()


# ---------------- 2. 梯度 (p − y)x 的数值验证 ----------------
def exp_gradient():
    print("=== 2. 交叉熵对 w 的梯度 = (p − y)·x：用有限差分验证 ===")
    r = np.random.default_rng(0)
    X = add_bias(r.normal(size=(50, 3))); y = (r.random(50) < 0.5).astype(float)
    w = r.normal(size=4)
    p = sigmoid(X @ w)
    analytic = X.T @ (p - y) / len(y)
    numeric = np.zeros_like(w)
    h = 1e-6
    for j in range(4):
        e = np.zeros(4); e[j] = h
        numeric[j] = (bce(sigmoid(X @ (w + e)), y) - bce(sigmoid(X @ (w - e)), y)) / (2 * h)
    print(f"  公式算的梯度   {np.round(analytic, 6)}")
    print(f"  有限差分算的   {np.round(numeric, 6)}   最大差 {np.abs(analytic - numeric).max():.1e}")
    print("  形式与线性回归的梯度 Xᵀ(Xw − y) 一模一样：预测减真实，乘特征——sigmoid 与 log 的导数恰好互相抵消")
    print()


# ---------------- 3. 决策边界：二维数据上画出来 ----------------
def exp_boundary():
    print("=== 3. 二维数据上的决策边界（手写 GD vs sklearn）===")
    X, y = make_blobs(n_samples=200, centers=[[-1.5, -1], [1.5, 1]], cluster_std=1.2, random_state=0)
    Xb = add_bias(X)
    w, hist = fit_logistic(Xb, y.astype(float), lr=0.5, steps=500)
    clf = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)          # C 很大 = 几乎无正则，与手写版对齐
    acc_hand = ((sigmoid(Xb @ w) > 0.5) == y).mean()
    print(f"  手写 GD 500 步：w = {np.round(w, 3)}，训练准确率 {acc_hand:.3f}，最终交叉熵 {hist[-1]:.3f}")
    print(f"  sklearn：       w = {np.round(np.r_[clf.intercept_, clf.coef_[0]], 3)}，训练准确率 {clf.score(X, y):.3f}")
    print(f"  两者系数最大差 {np.abs(w - np.r_[clf.intercept_, clf.coef_[0]]).max():.3f}（GD 还没完全收敛；边界方向已一致）")
    xx, yy = np.meshgrid(np.linspace(-5, 5, 60), np.linspace(-5, 5, 60))
    P = sigmoid(add_bias(np.c_[xx.ravel(), yy.ravel()]) @ w).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    ax = axes[0]
    ax.plot(hist, color=C["blue"]); ax.set_xlabel("步数"); ax.set_ylabel("训练交叉熵"); ax.set_title("手写梯度下降的 loss")
    ax = axes[1]
    cf = ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
    ax.contour(xx, yy, P, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, color=C["blue"], edgecolor="w", lw=0.3, label="类 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, color=C["red"], edgecolor="w", lw=0.3, label="类 1")
    ax.set_title("P(y = 1 | x) 的等高线；黑线是 p = 0.5 的决策边界（一条直线）", fontsize=8.5)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.colorbar(cf, ax=ax, shrink=0.8, label="P(y = 1)")
    save(fig, "03-decision-boundary")
    print()


# ---------------- 4. 真实数据：乳腺癌 ----------------
def exp_cancer():
    print("=== 4. 逻辑回归：乳腺癌数据 30 个特征 → 良性 / 恶性 ===")
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    acc = ((p > 0.5) == yte).mean()
    print(f"  {len(Xtr)} 训练 / {len(Xte)} 测试；准确率 {acc:.3f}；测试集每样本交叉熵 {bce(p, yte):.3f}")
    # 手写版对照
    w, _ = fit_logistic(add_bias(sc.transform(Xtr)), ytr.astype(float), lr=0.1, steps=3000, l2=1 / len(Xtr))
    p_hand = sigmoid(add_bias(sc.transform(Xte)) @ w)
    print(f"  手写 GD（同样的 L2 强度）：准确率 {((p_hand > 0.5) == yte).mean():.3f}，与 sklearn 预测概率最大差 {np.abs(p_hand - p).max():.3f}")
    top = np.argsort(-np.abs(clf.coef_[0]))[:3]
    print("  |w| 最大的三个特征:", [f"{data.feature_names[i]} ({clf.coef_[0][i]:+.2f})" for i in top])
    print(f"  一个样本: 线性部分 wᵀx+b = {clf.decision_function(sc.transform(Xte[:1]))[0]:+.2f} → σ(·) = {p[0]:.3f} → 预测 {'良性' if p[0] > 0.5 else '恶性'}，真实 {'良性' if yte[0] else '恶性'}")
    order = np.argsort(clf.coef_[0])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = [C["red"] if c < 0 else C["green"] for c in clf.coef_[0][order]]
    ax.barh(range(30), clf.coef_[0][order], color=colors)
    ax.set_yticks(range(30)); ax.set_yticklabels(data.feature_names[order], fontsize=7)
    ax.axvline(0, color=C["gray"], lw=0.8)
    ax.set_xlabel("系数 w（特征已标准化；正 → 偏向良性，负 → 偏向恶性）")
    ax.set_title("30 个特征的系数：模型是可读的")
    save(fig, "03-cancer-coefficients")
    print("  任何神经网络分类头 = 前面所有层做特征 x + 最后这一行 σ(wᵀx + b)")
    print()


# ---------------- 5. softmax 回归：10 类手写数字 ----------------
def exp_softmax():
    print("=== 5. softmax 回归：手写数字 10 类（这就是语言模型的输出层）===")
    d = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.3, random_state=0, stratify=d.target)
    W = fit_softmax(add_bias(Xtr), ytr, 10, lr=0.5, steps=500)
    P = softmax(add_bias(Xte) @ W)
    acc = (P.argmax(1) == yte).mean()
    clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    print(f"  W 的形状 {W.shape}（64 个像素 + 1 偏置 → 10 类）；手写 softmax GD 500 步测试准确率 {acc:.3f}；sklearn {clf.score(Xte, yte):.3f}")
    i = 0
    print(f"  一个测试样本：10 个类的概率 {np.round(P[i], 2)} → argmax {P[i].argmax()}，真实 {yte[i]}")
    print("  语言模型最后一层 nn.Linear(d_model, vocab) + softmax = 一个 vocab 类的 softmax 回归，特征是前面所有层算出的向量")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1, 2.2]})
    axes[0].imshow(Xte[i].reshape(8, 8), cmap="gray_r"); axes[0].set_title(f"输入：8×8 像素，真实标签 {yte[i]}"); axes[0].axis("off")
    axes[1].bar(range(10), P[i], color=[C["red"] if k == P[i].argmax() else C["blue"] for k in range(10)])
    axes[1].set_xticks(range(10)); axes[1].set_xlabel("类别"); axes[1].set_ylabel("softmax 概率"); axes[1].set_title("输出：10 个类的概率，和为 1")
    save(fig, "03-softmax-digit")
    print()


# ---------------- 6. 奖励模型 = 逻辑回归作用在两个样本的分差上 ----------------
def make_preferences(n_items=300, d=16, n_pairs=3000, noise_scale=1.0, seed=0):
    r = np.random.default_rng(seed)
    w_true = r.normal(size=d)                            # 真实的"品味"
    feats = r.normal(size=(n_items, d))                  # 每个回答一个 16 维特征（真实场景是 LLM 的 hidden state）
    score = feats @ w_true                               # 真实分数
    i, j = r.integers(0, n_items, (2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    p_win = sigmoid((score[i] - score[j]) / noise_scale)  # noise_scale 越大标注越随机
    y = (r.random(len(i)) < p_win).astype(int)           # 标注员按 Bradley-Terry 概率给出偏好
    return feats, score, w_true, i, j, y


def exp_reward():
    print("=== 6. Bradley-Terry 奖励模型：用逻辑回归拟合 P(A ≻ B) = σ(r(A) − r(B)) ===")
    feats, score, w_true, i, j, y = make_preferences()
    Xdiff = feats[i] - feats[j]                          # 关键：特征取差，无偏置
    ntr = int(0.8 * len(y))
    clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=2000).fit(Xdiff[:ntr], y[:ntr])
    acc = (clf.predict(Xdiff[ntr:]) == y[ntr:]).mean()
    corr = np.corrcoef(clf.coef_[0], w_true)[0, 1]
    oracle = ((score[i] > score[j]).astype(int)[ntr:] == y[ntr:]).mean()
    print(f"  {len(y)} 个偏好对，80% 训练；留出集准确率 {acc:.3f}（用真实分数判断的上限 {oracle:.3f}——标注本身有噪声）")
    print(f"  学到的 w 与真实 w 的相关系数 {corr:.3f}")
    # 手写版：同样的逻辑回归、无偏置
    w_hand, _ = fit_logistic(Xdiff[:ntr], y[:ntr].astype(float), lr=0.5, steps=2000)
    print(f"  手写 GD（无偏置列）：留出集准确率 {((sigmoid(Xdiff[ntr:] @ w_hand) > 0.5) == y[ntr:]).mean():.3f}，与真实 w 相关 {np.corrcoef(w_hand, w_true)[0,1]:.3f}")
    fitted = feats @ clf.coef_[0]
    # 按分差分桶看标注一致率
    diff = np.abs(score[i] - score[j])[ntr:]
    agree = ((score[i] > score[j]).astype(int) == y)[ntr:]
    bins = np.quantile(diff, [0, 0.25, 0.5, 0.75, 1.0])
    print("  按两个回答真实分差分桶，标注与真实分数一致的比例：")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (diff >= lo) & (diff <= hi)
        print(f"    分差 {lo:5.2f}–{hi:5.2f}: 一致率 {agree[m].mean():.2f}（{m.sum()} 对）")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    ax.scatter(score, fitted, s=8, color=C["blue"], alpha=0.7)
    ax.set_xlabel("真实分数 r*(x)"); ax.set_ylabel("拟合出的分数 wᵀx"); ax.set_title(f"300 个回答：拟合分 vs 真实分（相关 {np.corrcoef(score, fitted)[0,1]:.3f}）", fontsize=8.5)
    ax = axes[1]
    ds = np.linspace(-6, 6, 200)
    ax.plot(ds, sigmoid(ds), color=C["red"], lw=1.8)
    ax.set_xlabel("r(A) − r(B)"); ax.set_ylabel("P(A ≻ B)")
    ax.set_title("Bradley-Terry：分差过 sigmoid 就是偏好概率\n分差 0 → 抛硬币；分差 4 → 98%", fontsize=8.5)
    ax.axhline(0.5, color=C["gray"], ls="--", lw=0.8); ax.axvline(0, color=C["gray"], ls="--", lw=0.8)
    save(fig, "03-reward-model-fit")
    print("  奖励模型的 loss −log σ(r_w − r_l) 就是逻辑回归的 loss，只是输入从 x 换成了 x_w − x_l；准确率上限是标注的一致性")
    print()


# ---------------- 7. 标注噪声 vs 准确率上限 ----------------
def exp_noise():
    print("=== 7. 标注越不一致，奖励模型的准确率上限越低——模型准确率始终贴着上限 ===")
    scales, accs, oracles, corrs = [], [], [], []
    print(f"  {'标注噪声':>8} {'标注一致率(上限)':>14} {'模型留出准确率':>12} {'w 相关':>8}")
    for s in (0.3, 0.5, 1.0, 2.0, 4.0, 8.0):
        feats, score, w_true, i, j, y = make_preferences(noise_scale=s, seed=0)
        Xdiff = feats[i] - feats[j]; ntr = int(0.8 * len(y))
        clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=2000).fit(Xdiff[:ntr], y[:ntr])
        acc = (clf.predict(Xdiff[ntr:]) == y[ntr:]).mean()
        oracle = ((score[i] > score[j]).astype(int)[ntr:] == y[ntr:]).mean()
        corr = np.corrcoef(clf.coef_[0], w_true)[0, 1]
        scales.append(s); accs.append(acc); oracles.append(oracle); corrs.append(corr)
        print(f"  {s:>8.1f} {oracle:>14.3f} {acc:>12.3f} {corr:>8.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.plot(scales, oracles, "o-", color=C["gray"], label="标注与真实分数的一致率（能达到的上限）")
    ax.plot(scales, accs, "s-", color=C["red"], label="学到的奖励模型的留出集准确率")
    ax.set_xscale("log"); ax.set_xlabel("标注噪声（越大标注越随机）"); ax.set_ylabel("准确率")
    ax.set_ylim(0.45, 1.0); ax.legend(frameon=False)
    save(fig, "03-label-noise-ceiling")
    print("  模型准确率贴着上限走：70% 的准确率可能不是模型差，是标注只有 70% 一致；再训只会拟合噪声")
    print()


EXPS = {"sigmoid": exp_sigmoid, "gradient": exp_gradient, "boundary": exp_boundary, "cancer": exp_cancer,
        "softmax": exp_softmax, "reward": exp_reward, "noise": exp_noise}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
