"""线性回归与逻辑回归（经典 ML 02）：闭式解 vs 梯度下降、Ridge/Lasso 的稀疏、逻辑回归、以及"奖励模型 = 逻辑回归作用在分差上"。
https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html

    python 02_linear_and_logistic_regression.py        # 全部：linear regularize logistic reward
"""
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)


# ---------------- 1. 线性回归：闭式解与梯度下降得到同一个答案 ----------------
def exp_linear():
    print("=== 1. 线性回归：最小二乘的闭式解 vs 梯度下降 ===")
    X, y, w_true = make_regression(n_samples=200, n_features=3, noise=5.0, coef=True, random_state=0)
    Xb = np.c_[np.ones(len(X)), X]                                     # 加一列 1 当偏置
    w_closed = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)                    # (XᵀX)⁻¹ Xᵀy
    w = np.zeros(4)
    for step in range(2000):                                           # L0 第七篇：θ ← θ − η ∇L
        grad = 2 * Xb.T @ (Xb @ w - y) / len(y)
        w -= 0.05 * grad
    print(f"  真实系数     {np.round(w_true, 2)}")
    print(f"  闭式解       {np.round(w_closed[1:], 2)}  (偏置 {w_closed[0]:.2f})")
    print(f"  梯度下降 2000 步 {np.round(w[1:], 2)}  (偏置 {w[0]:.2f})   两者之差 {np.abs(w - w_closed).max():.1e}")
    print(f"  sklearn      {np.round(LinearRegression().fit(X, y).coef_, 2)}")
    print()


# ---------------- 2. Ridge / Lasso：L2 压小、L1 压到零 ----------------
def exp_regularize():
    print("=== 2. Ridge (L2) vs Lasso (L1)：50 个特征里只有 5 个真的有用 ===")
    X, y, w_true = make_regression(n_samples=100, n_features=50, n_informative=5, noise=10.0, coef=True, random_state=1)
    X = StandardScaler().fit_transform(X)
    print(f"  真实非零系数 {int((w_true != 0).sum())} 个")
    for name, model in [("无正则", LinearRegression()), ("Ridge α=10", Ridge(alpha=10)), ("Lasso α=1", Lasso(alpha=1.0)), ("Lasso α=5", Lasso(alpha=5.0))]:
        m = model.fit(X, y)
        c = m.coef_
        print(f"  {name:<12} 恰好为 0 的系数 {int((np.abs(c) < 1e-6).sum()):>2}/50   |系数| 均值 {np.abs(c).mean():6.2f}   最大 {np.abs(c).max():6.2f}")
    print("  L2 把所有系数一起压小但不为零；L1 把不重要的压到恰好为零（L0 第二篇：零点附近惩罚不变小）")
    print()


# ---------------- 3. 逻辑回归：分类头的原型 ----------------
def exp_logistic():
    print("=== 3. 逻辑回归：乳腺癌数据 30 个特征 → 良性 / 恶性 ===")
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    acc = ((p > 0.5) == yte).mean()
    nll = -np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p))
    print(f"  {len(Xtr)} 训练 / {len(Xte)} 测试；准确率 {acc:.3f}；测试集每样本交叉熵 {nll:.3f}")
    top = np.argsort(-np.abs(clf.coef_[0]))[:3]
    print("  |w| 最大的三个特征:", [f"{data.feature_names[i]} ({clf.coef_[0][i]:+.2f})" for i in top])
    print(f"  一个样本: 线性部分 wᵀx+b = {clf.decision_function(sc.transform(Xte[:1]))[0]:+.2f} → σ(·) = {p[0]:.3f} → 预测 {'良性' if p[0] > 0.5 else '恶性'}，真实 {'良性' if yte[0] else '恶性'}")
    print("  任何神经网络分类头 = 前面所有层做特征 x + 最后这一行 σ(wᵀx + b)")
    print()


# ---------------- 4. 奖励模型 = 逻辑回归作用在两个样本的分差上 ----------------
def exp_reward():
    print("=== 4. Bradley-Terry 奖励模型：用逻辑回归拟合 P(A ≻ B) = σ(r(A) − r(B)) ===")
    d, n_items, n_pairs = 16, 300, 3000
    r = np.random.default_rng(0)
    w_true = r.normal(size=d)                            # 真实的"品味"
    feats = r.normal(size=(n_items, d))                  # 每个回答一个 16 维特征（真实场景是 LLM 的 hidden state）
    score = feats @ w_true                               # 真实分数
    i, j = r.integers(0, n_items, (2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    p_win = 1 / (1 + np.exp(-(score[i] - score[j])))
    y = (r.random(len(i)) < p_win).astype(int)           # 标注员按 Bradley-Terry 概率给出偏好
    Xdiff = feats[i] - feats[j]                          # 关键：特征取差，无偏置
    ntr = int(0.8 * len(y))
    clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=2000).fit(Xdiff[:ntr], y[:ntr])
    acc = (clf.predict(Xdiff[ntr:]) == y[ntr:]).mean()
    corr = np.corrcoef(clf.coef_[0], w_true)[0, 1]
    oracle = ((score[i] > score[j]).astype(int)[ntr:] == y[ntr:]).mean()
    print(f"  {len(y)} 个偏好对，80% 训练；留出集准确率 {acc:.3f}（用真实分数判断的上限 {oracle:.3f}——标注本身有噪声）")
    print(f"  学到的 w 与真实 w 的相关系数 {corr:.3f}")
    print("  奖励模型的 loss −log σ(r_w − r_l) 就是逻辑回归的 loss，只是输入从 x 换成了 x_w − x_l；准确率上限是标注的一致性")
    print()


EXPS = {"linear": exp_linear, "regularize": exp_regularize, "logistic": exp_logistic, "reward": exp_reward}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
