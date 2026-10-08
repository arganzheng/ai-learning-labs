"""逻辑回归与奖励模型（经典 ML 03）：sigmoid、交叉熵与它的梯度、手写 GD 对 sklearn、决策边界、softmax 回归、
以及"奖励模型 = 逻辑回归作用在两个回答的特征差上"。
https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html

    python 03_logistic_regression_and_reward_model.py          # 全部：sigmoid gradient boundary cancer softmax reward noise
    python 03_logistic_regression_and_reward_model.py reward

图输出到 out/03-*.svg。
"""
import sys
from itertools import pairwise

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
    for zi in (-4, -2, 0, 2, 4):
        print(f"  z = {zi:+d} → σ(z) = {sigmoid(zi):.3f}")
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
    for lo, hi in pairwise(bins):
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


# ---------------- 8. 六个数走一遍：logit → 概率 → loss → 梯度 → 一步更新 ----------------
def stable_sigmoid(z):
    """σ(z) = exp(−log(1 + e^{−z}))：先在 log 域算，再取 exp，z = ±800 也不溢出。"""
    return np.exp(-np.logaddexp(0.0, -z))


def logistic_loss(z, y):
    """每个样本的交叉熵，直接用 logit 算：log(1 + e^z) − y·z，等价于 −[y log p + (1−y) log(1−p)]。"""
    return np.logaddexp(0.0, z) - y * z


def one_step(x, y, w, b, lr):
    """逻辑回归的一步梯度下降；返回这一步的所有中间量，方便逐行核对。"""
    z = w * x + b                                       # !ref step-logit
    p = stable_sigmoid(z)                               # !ref step-prob
    loss = logistic_loss(z, y).mean()                   # !ref step-loss
    err = p - y                                         # !ref step-err
    dw, db = (err * x).mean(), err.mean()               # !ref step-grad
    return z, p, loss, err, dw, db, w - lr * dw, b - lr * db  # !ref step-update


def exp_step():
    print("=== 8. 六个数走一遍：logit → 概率 → loss → 梯度 → 一步更新 ===")
    x = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    w, b, lr = 0.0, 0.0, 0.1
    print(f"  复习小时 x = {x.astype(int).tolist()}，是否通过 y = {y.astype(int).tolist()}；初始 w = {w}, b = {b}，学习率 {lr}")
    z, p, loss, err, dw, db, w1, b1 = one_step(x, y, w, b, lr)
    print("    x   y   z=wx+b   p=σ(z)   每样本 loss   p−y    (p−y)·x")
    for k in range(len(x)):
        print(f"    {x[k]:.0f}   {y[k]:.0f}   {z[k]:6.3f}   {p[k]:6.3f}   {logistic_loss(z[k], y[k]):11.4f}   {err[k]:+.3f}   {err[k] * x[k]:+.3f}")
    print(f"  平均 loss = {loss:.4f}（= ln 2，因为每个 p 都是 0.5）；∂L/∂w = mean((p−y)·x) = {dw:+.4f}，∂L/∂b = mean(p−y) = {db:+.4f}")
    print(f"  一步更新：w ← {w} − {lr}·({dw:+.3f}) = {w1:.3f}，b ← {b} − {lr}·({db:+.3f}) = {b1:.3f}")
    z1, p1, loss1, *_ = one_step(x, y, w1, b1, lr)
    print(f"  更新后：z = {np.round(z1, 3).tolist()}，p = {np.round(p1, 3).tolist()}，loss {loss:.4f} → {loss1:.4f}")
    # 继续走：看状态怎么变
    traj = {0: (w, b, loss)}
    wk, bk = w, b
    for step in range(1, 1001):
        _, _, lk, _, _, _, wk, bk = one_step(x, y, wk, bk, lr)
        if step in (1, 10, 100, 1000):
            traj[step] = (wk, bk, one_step(x, y, wk, bk, lr)[2])
    print("  再走下去（同样的学习率）：")
    print("    步数      w        b      loss   决策边界 x = −b/w")
    for step, (wk, bk, lk) in traj.items():
        bound = f"{-bk / wk + 0.0:6.2f}" if wk != 0 else "   —  "
        print(f"    {step:>4}   {wk:7.3f}  {bk:7.3f}  {lk:7.4f}   {bound}")
    print("  梯度是「预测减真实，乘特征」：前三个人 p=0.5>y=0 把 w 往下拉、后三个人 p=0.5<y=1 把 w 往上推，复习多的人力臂长，所以净梯度为负、w 变正")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    xs = np.linspace(0, 8, 200)
    for (step, (wk, bk, _)), col in zip(traj.items(), [C["light"], C["gray"], C["orange"], C["blue"], C["red"]]):
        ax.plot(xs, stable_sigmoid(wk * xs + bk), color=col, lw=1.6, label=f"第 {step} 步")
    ax.scatter(x, y, color="k", zorder=3, s=22)
    ax.set_xlabel("复习小时 x"); ax.set_ylabel("P(通过 | x)"); ax.set_title("六个点上 sigmoid 曲线随步数变化", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    hist = []
    wk, bk = w, b
    for step in range(1000):
        _, _, lk, _, _, _, wk, bk = one_step(x, y, wk, bk, lr)
        hist.append(lk)
    ax.plot(np.arange(1, 1001), hist, color=C["blue"])
    ax.set_xscale("log"); ax.set_xlabel("步数"); ax.set_ylabel("平均交叉熵"); ax.set_title("loss 持续下降但到不了 0：数据线性可分，‖w‖ 会继续增大", fontsize=8.5)
    save(fig, "03-one-step-update")
    print()


# ---------------- 9. 数值稳定：logaddexp 版的 sigmoid、交叉熵、softmax ----------------
def exp_stable():
    print("=== 9. 数值稳定：log(1 + e^z) 用 logaddexp，softmax 先减最大值 ===")
    zs = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
    print("  标签 y = 1 时的每样本 loss：朴素写法 −log(1/(1+e^{−z}))  vs  带 eps 的 bce  vs  logaddexp(0, −z)")
    print("      z     朴素 σ(z)   朴素 −log σ    bce(eps=1e-12)   logaddexp 版   手算 log(1+e^{−z})")
    for z in zs:
        with np.errstate(over="ignore", divide="ignore"):
            p_naive = sigmoid(z)
            l_naive = -np.log(p_naive)
            l_eps = bce(np.array([p_naive]), np.array([1.0]))
        l_stab = logistic_loss(z, 1.0)
        exact = "≈ −z" if z < -50 else ("≈ e^{−z}" if z > 50 else f"{np.log1p(np.exp(-z)):.4f}")
        print(f"    {z:6.0f}   {p_naive:9.3g}   {l_naive:11.4g}   {l_eps:14.4g}   {l_stab:12.4g}   {exact}")
    print("  z = −800、y = 1：真实 loss 是 800；朴素写法 e^{800} 溢出成 inf → σ = 0 → −log 0 = inf；")
    print("  bce 的 eps 把它截成 −log(1e-12) ≈ 27.6，数字看着正常、其实错了 30 倍；logaddexp 直接给 800.0")
    print(f"  稳定版 σ：exp(−logaddexp(0, −z))，z = −700 → {stable_sigmoid(-700.0):.3g}，z = −800 → {stable_sigmoid(-800.0):.0f}（e^−800 小于 float64 能表示的最小正数，σ 本身确实下溢；但 loss 不经过 σ，仍是 800）")
    Z = np.array([[1000.0, 1001.0, 1002.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(Z) / np.exp(Z).sum()
    print(f"  softmax([1000, 1001, 1002])：朴素 e^z/Σe^z = {naive.ravel().tolist()}；先减最大值 → {np.round(softmax(Z).ravel(), 4).tolist()}")
    logsumexp = Z.max() + np.log(np.exp(Z - Z.max()).sum())
    print(f"  log-softmax = z − logsumexp(z)，logsumexp = max + log Σ e^{{z−max}} = {logsumexp:.4f}；log-softmax = {np.round((Z - logsumexp).ravel(), 4).tolist()}")
    # 稳定版 loss 的梯度 = (p − y)·x，用有限差分在六个数的数据上核对
    x = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    theta = np.array([0.1, 0.0])                        # (w, b)：第 8 节走完一步的位置

    def loss_at(t):
        return logistic_loss(t[0] * x + t[1], y).mean()

    p = stable_sigmoid(theta[0] * x + theta[1])
    analytic = np.array([((p - y) * x).mean(), (p - y).mean()])
    print(f"  在 (w, b) = {theta.tolist()} 处，公式梯度 (∂L/∂w, ∂L/∂b) = {np.round(analytic, 6).tolist()}")
    for h in (1e-2, 1e-4, 1e-6):
        num = np.array([(loss_at(theta + h * e) - loss_at(theta - h * e)) / (2 * h) for e in np.eye(2)])
        print(f"    中心差分 h = {h:.0e}：{np.round(num, 6).tolist()}，最大差 {np.abs(num - analytic).max():.1e}")
    print("  中心差分误差随 h² 下降（1e-2 → 1e-4 差了约 1e4 倍），到 1e-6 时被浮点舍入接住；三种 h 都对上才算验证通过")
    print()


# ---------------- 10. 一条样本怎么变成一对回答 ----------------
def pair_step(xa, xb, w, y, lr):
    """奖励模型的一步：两个回答 → 两个分数 → 分差 → 偏好概率 → loss → 梯度 → 更新。"""
    ra, rb = xa @ w, xb @ w                             # !ref pair-score
    delta = ra - rb                                     # !ref pair-delta
    p = stable_sigmoid(delta)                           # !ref pair-prob
    loss = logistic_loss(delta, y)                      # !ref pair-loss
    grad_w = (p - y) * (xa - xb)                        # !ref pair-grad
    return ra, rb, delta, p, loss, grad_w, w - lr * grad_w  # !ref pair-update


def exp_pair():
    print("=== 10. 一条样本怎么变成一对回答：分差、概率、loss 与梯度 ===")
    xa = np.array([2.0, 1.0, 1.0])                      # 回答 A 的 3 个特征（比如：论据数、冗余句数、是否答对）
    xb = np.array([1.0, 3.0, 0.0])                      # 回答 B
    w = np.array([0.5, -0.2, 1.0])                      # 当前奖励模型的权重
    y, lr = 1.0, 0.5                                    # 标注员选了 A
    ra, rb, delta, p, loss, g, w1 = pair_step(xa, xb, w, y, lr)
    print(f"  x_A = {xa.tolist()}，x_B = {xb.tolist()}，w = {w.tolist()}，标注 y = 1（A 更好）")
    print(f"  r(A) = w·x_A = {ra:.2f}，r(B) = w·x_B = {rb:.2f}，分差 Δ = {delta:.2f}")
    print(f"  P(A ≻ B) = σ(Δ) = {p:.4f}；loss = log(1 + e^{{−Δ}}) = {loss:.4f}")
    print(f"  x_A − x_B = {(xa - xb).tolist()}；∂loss/∂w = (p − y)(x_A − x_B) = {p - y:+.4f} × {(xa - xb).tolist()} = {np.round(g, 4).tolist()}")
    print(f"  ∂loss/∂r(A) = p − 1 = {p - 1:+.4f}（往上推 A 的分），∂loss/∂r(B) = 1 − p = {1 - p:+.4f}（等量往下压 B 的分）")
    _ra1, _rb1, delta1, p1, loss1, *_ = pair_step(xa, xb, w1, y, lr)
    print(f"  一步更新（lr = {lr}）：w ← {np.round(w1, 4).tolist()}；分差 {delta:.2f} → {delta1:.4f}，P(A ≻ B) {p:.4f} → {p1:.4f}，loss {loss:.4f} → {loss1:.4f}")
    # 同一条样本换个摆法：(B, A, y=0) 必须给出同一个梯度
    _, _, delta_s, p_s, loss_s, g_s, _ = pair_step(xb, xa, w, 0.0, lr)
    print(f"  把同一条样本摆成 (B, A, y=0)：Δ = {delta_s:.2f}，P(B ≻ A) = {p_s:.4f}，loss = {loss_s:.4f}，梯度 {np.round(g_s, 4).tolist()} —— 与上面逐位相同")
    # 作为普通逻辑回归：输入 x_A − x_B、标签 1、无偏置
    w_lr, _ = fit_logistic((xa - xb)[None], np.array([y]), lr=lr, steps=1)
    w_hand = 0.0 - lr * (stable_sigmoid(0.0) - y) * (xa - xb)
    print(f"  当成普通逻辑回归：X = [x_A − x_B]、y = [1]、无偏置列，fit_logistic 从 w = 0 走一步得到 {np.round(w_lr, 4).tolist()}；"
          f"手算 0 − {lr}·(σ(0) − 1)·(x_A − x_B) = {np.round(w_hand, 4).tolist()}，同一条公式")
    b = 0.3
    print(f"  若硬加偏置 b = {b}：P(A ≻ B) + P(B ≻ A) = σ(Δ + b) + σ(−Δ + b) = {stable_sigmoid(delta + b) + stable_sigmoid(-delta + b):.4f} ≠ 1，"
          "「谁放前面」会影响结论，所以奖励模型不要偏置")
    # 有限差分核对 ∂loss/∂w
    def loss_at(wv):
        return logistic_loss((xa - xb) @ wv, y)

    h = 1e-6
    num = np.array([(loss_at(w + h * e) - loss_at(w - h * e)) / (2 * h) for e in np.eye(3)])
    print(f"  有限差分核对：{np.round(num, 6).tolist()}，与公式最大差 {np.abs(num - g).max():.1e}")
    print("  分差大小决定这对样本贡献多少梯度（y = 1）：")
    print("      Δ     P(A≻B)    loss    |∂loss/∂Δ| = 1 − p")
    for d in (0.0, 1.0, 2.0, 4.0, 8.0):
        pd = stable_sigmoid(d)
        print(f"    {d:4.1f}   {pd:.4f}   {logistic_loss(d, 1.0):.4f}   {1 - pd:.4f}")
    print("  已经分得很开的对几乎不再更新模型；分差接近 0 的对贡献最大——它们也正是标注最容易出错的对（见第 7 节分桶）")
    print()


# ---------------- 11. 准确率、概率校准、标注一致性：三个不同的数 ----------------
def reliability(prob, y, n_bins=5):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in pairwise(edges):
        m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if m.sum():
            rows.append((lo, hi, m.sum(), prob[m].mean(), y[m].mean()))
    ece = sum(n * abs(pm - ym) for _, _, n, pm, ym in rows) / len(y)
    return rows, ece


def exp_calib():
    print("=== 11. 准确率、概率校准、标注一致性：三个不同的数 ===")
    feats, score, _w_true, i, j, y = make_preferences()
    Xdiff = feats[i] - feats[j]
    ntr = int(0.8 * len(y))
    clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=2000).fit(Xdiff[:ntr], y[:ntr])
    yte = y[ntr:]
    logits = Xdiff[ntr:] @ clf.coef_[0]
    print(f"  同第 6 节的奖励模型，留出集 {len(yte)} 对")
    print("  把 logit 乘一个常数 T 不改变任何一对的判断方向，所以准确率完全相同，但概率变了：")
    print("    T      准确率     ECE   平均 log loss   Brier      （ECE = 各桶 |平均预测概率 − 实际胜率| 按桶大小加权）")
    curves = {}
    for T in (1.0, 3.0, 1 / 3):
        prob = stable_sigmoid(T * logits)
        acc = ((prob > 0.5) == yte).mean()
        rows, ece = reliability(prob, yte)
        ll = logistic_loss(T * logits, yte).mean()
        brier = ((prob - yte) ** 2).mean()
        curves[T] = rows
        print(f"    {T:4.2f}   {acc:.3f}   {ece:.3f}   {ll:13.3f}   {brier:.3f}")
    print("  T = 1 时的可靠性表（预测概率分桶 → 这桶里 A 真的被选中的比例）：")
    for lo, hi, n, pm, ym in curves[1.0]:
        print(f"    预测 {lo:.1f}–{hi:.1f}：{n:>3} 对，平均预测 {pm:.3f}，实际胜率 {ym:.3f}")
    # 标注一致性：同一批对再找一个「标注员」标一遍
    r2 = np.random.default_rng(1)
    p_win = sigmoid(score[i] - score[j])
    y2 = (r2.random(len(i)) < p_win).astype(int)
    truth = (score[i] > score[j]).astype(int)
    a1, a2 = (y == truth).mean(), (y2 == truth).mean()
    agree = (y == y2).mean()
    model_pred = (logits > 0).astype(int)
    print("  标注一致性（全部对）：")
    print(f"    标注员 1 与真实分数一致 {a1:.3f}，标注员 2 与真实分数一致 {a2:.3f}，两人互相一致 {agree:.3f}")
    print(f"    若两人的错误相互独立，互相一致率应是 a₁a₂ + (1−a₁)(1−a₂) = {a1 * a2 + (1 - a1) * (1 - a2):.3f}；实际更高，因为两人都在同一批分差小的对上出错——错误不独立")
    print(f"    模型在留出集上：与标注员 1 一致 {(model_pred == yte).mean():.3f}，与标注员 2 一致 {(model_pred == y2[ntr:]).mean():.3f}，与真实分数一致 {(model_pred == truth[ntr:]).mean():.3f}")
    print("  三个数回答三个问题：准确率——阈值后的判断对了多少；校准——说 80% 的那些对里是不是真有 80% 选 A；一致性——换个人标还剩多少能对上")
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color=C["gray"], lw=0.8, label="完美校准")
    for (T, rows), col, name in zip(curves.items(), [C["blue"], C["red"], C["orange"]], ["T = 1（原模型）", "T = 3（过度自信）", "T = 1/3（过度保守）"]):
        acc = ((stable_sigmoid(T * logits) > 0.5) == yte).mean()
        ax.plot([r[3] for r in rows], [r[4] for r in rows], "o-", color=col, label=f"{name}，准确率 {acc:.3f}")
    ax.set_xlabel("模型给出的 P(A ≻ B)（按桶平均）"); ax.set_ylabel("这桶里 A 实际被选中的比例")
    ax.set_title("三个模型准确率一样，校准完全不同", fontsize=8.5); ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "03-calibration-vs-accuracy")
    print()


EXPS = {"sigmoid": exp_sigmoid, "gradient": exp_gradient, "boundary": exp_boundary, "cancer": exp_cancer,
        "softmax": exp_softmax, "reward": exp_reward, "noise": exp_noise,
        "step": exp_step, "stable": exp_stable, "pair": exp_pair, "calib": exp_calib}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
