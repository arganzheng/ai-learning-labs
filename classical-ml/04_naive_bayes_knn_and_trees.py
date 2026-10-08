"""三个基础分类器（经典 ML 04）：朴素贝叶斯、KNN、决策树——各自怎么工作、边界长什么样、用 NumPy 手写并与 scikit-learn 对数；
外加七个分类器在同一份数据上的总表（04–06 三篇共用）。
https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html

    python 04_naive_bayes_knn_and_trees.py          # 全部：compare grid bayes nb knn curse tree depth
    python 04_naive_bayes_knn_and_trees.py tree

图输出到 out/04-*.svg。
"""
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from _plot import C, plt, save


# ---------------- 数据：表格数据（5000 × 20）与二维月牙 ----------------
def tabular():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, n_redundant=4, flip_y=0.05, class_sep=0.8, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler().fit(Xtr)
    return Xtr, Xte, ytr, yte, sc.transform(Xtr), sc.transform(Xte)


def moons():
    X, y = make_moons(n_samples=400, noise=0.22, random_state=0)
    return train_test_split(X, y, test_size=0.5, random_state=0)


def seven_models():
    return {
        "逻辑回归":     make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "朴素贝叶斯":   GaussianNB(),
        "KNN (k=15)":  make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "决策树":       DecisionTreeClassifier(max_depth=None, random_state=0),
        "SVM (RBF)":   make_pipeline(StandardScaler(), SVC(C=1.0)),
        "随机森林":     RandomForestClassifier(300, random_state=0, n_jobs=-1),
        "梯度提升":     GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0),
    }


# ---------------- 手写实现 ----------------
def knn_predict(Xtrain, ytrain, Xtest, k=15):
    d2 = ((Xtest[:, None, :] - Xtrain[None, :, :]) ** 2).sum(-1)   # ① [n_test, n_train]：每对样本的欧氏距离平方
    idx = np.argpartition(d2, k, axis=1)[:, :k]                    # ② 每个测试样本最近的 k 个训练样本的下标
    votes = ytrain[idx]                                             # ③ [n_test, k]：这 k 个邻居的标签
    return (votes.mean(1) > 0.5).astype(int)                        # ④ 多数票（两类时就是均值过半）


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.array([(y == c).mean() for c in self.classes])            # ① P(c)：各类的比例
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])             # ② [类, 特征]：每类每特征的均值
        self.var = np.array([X[y == c].var(0) + 1e-9 for c in self.classes])      # ③ 方差（加一点防止除零）
        return self

    def predict(self, X):
        ll = -0.5 * (np.log(2 * np.pi * self.var[:, None, :])
                     + (X[None] - self.mu[:, None, :]) ** 2 / self.var[:, None, :]).sum(-1)   # ④ Σ_j log N(x_j; μ, σ²)：[类, n]
        return self.classes[(np.log(self.prior)[:, None] + ll).argmax(0)]                    # ⑤ 加上 log P(c)，取最大的类


def gini(y):
    p = y.mean(); return 2 * p * (1 - p)                                          # 两类：全是一类 0，各占一半 0.5


def best_split(X, y):
    best: tuple[float, int | None, float | None] = (gini(y), None, None)
    for j in range(X.shape[1]):                                                   # ① 每个特征
        for thr in np.percentile(X[:, j], np.arange(5, 100, 5)):                  # ② 19 个分位数当候选阈值
            left = X[:, j] <= thr
            if left.sum() == 0 or left.sum() == len(y): continue
            g = left.mean() * gini(y[left]) + (1 - left.mean()) * gini(y[~left])  # ③ 切完两边不纯度的加权平均
            if g < best[0]: best = (g, j, thr)
    return best[1], best[2]


def build_tree(X, y, depth, max_depth):
    if depth == max_depth or len(np.unique(y)) == 1: return int(y.mean() > 0.5)  # ④ 到深度或已纯：叶子 = 多数类
    j, thr = best_split(X, y)
    if j is None: return int(y.mean() > 0.5)
    left = X[:, j] <= thr
    return (j, thr, build_tree(X[left], y[left], depth + 1, max_depth), build_tree(X[~left], y[~left], depth + 1, max_depth))  # ⑤ 递归


def tree_predict(node, x):
    while isinstance(node, tuple):                                                # ⑥ 从根往下走到叶子
        j, thr, l, r = node
        node = l if x[j] <= thr else r
    return node


def plot_boundary(ax, predict, X, y, title, res=80):
    xx, yy = np.meshgrid(np.linspace(-2, 3, res), np.linspace(-1.5, 2, res))
    Z = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]], alpha=0.18)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=7, color=C["blue"]); ax.scatter(X[y == 1, 0], X[y == 1, 1], s=7, color=C["red"])
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


# ---------------- 1. 七个分类器：总表 ----------------
def exp_compare():
    print("=== 1. 七个分类器：合成表格数据（5000 样本、20 特征、其中 8 个有用、含非线性）===")
    Xtr, Xte, ytr, yte, _, _ = tabular()
    notes = {"逻辑回归": "线性边界：有非线性就吃亏", "朴素贝叶斯": "假设特征独立：冗余特征让它更差", "KNN (k=15)": "不训练；预测时找 15 个邻居投票",
             "决策树": "训练 100%、测试掉一截：过拟合的教科书样子", "SVM (RBF)": "核把线性边界变弯", "随机森林": "很多棵树的 bagging：降方差",
             "梯度提升": "逐棵树拟合残差：表格数据的默认最强"}
    print(f"  {'模型':<14} {'训练准确率':>10} {'测试准确率':>10} {'训练耗时':>9}   一句话")
    for name, m in seven_models().items():
        t0 = time.time(); m.fit(Xtr, ytr); dt = time.time() - t0
        print(f"  {name:<14} {m.score(Xtr, ytr):>10.3f} {m.score(Xte, yte):>10.3f} {dt:>8.2f}s   {notes[name]}")
    print()


# ---------------- 2. 七个分类器在二维月牙上的决策边界 ----------------
def exp_grid():
    print("=== 2. 同一份二维数据（两个月牙）上七个分类器的决策边界 ===")
    Xtr, Xte, ytr, yte = moons()
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9))
    axes = axes.ravel()
    ax = axes[0]
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=7, color=C["blue"]); ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=7, color=C["red"])
    ax.set_xlim(-2, 3); ax.set_ylim(-1.5, 2); ax.set_title("数据：两个月牙（200 训练点）", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax, (name, m) in zip(axes[1:], seven_models().items()):
        m.fit(Xtr, ytr)
        acc = m.score(Xte, yte)
        print(f"  {name:<12} 测试准确率 {acc:.3f}")
        plot_boundary(ax, m.predict, Xtr, ytr, f"{name}：{acc:.2f}")
    save(fig, "04-seven-boundaries")
    print()


# ---------------- 3. 贝叶斯公式：一个算得出来的例子 ----------------
def exp_bayes():
    print("=== 3. 贝叶斯公式：垃圾邮件里的一个词 ===")
    p_spam, p_word_spam, p_word_ham = 0.3, 0.6, 0.05
    p_word = p_word_spam * p_spam + p_word_ham * (1 - p_spam)
    post = p_word_spam * p_spam / p_word
    print(f"  先验 P(垃圾) = {p_spam}；P('免费'|垃圾) = {p_word_spam}，P('免费'|正常) = {p_word_ham}")
    print(f"  P('免费') = {p_word_spam}×{p_spam} + {p_word_ham}×{1-p_spam:.1f} = {p_word:.3f}")
    print(f"  后验 P(垃圾|'免费') = {p_word_spam}×{p_spam} / {p_word:.3f} = {post:.3f}")
    # 两个词、独立假设
    p2_spam, p2_ham = 0.4, 0.2   # P('会议'|垃圾), P('会议'|正常)
    num_s = p_spam * p_word_spam * p2_spam; num_h = (1 - p_spam) * p_word_ham * p2_ham
    print(f"  再看到'会议'（P(·|垃圾)={p2_spam}, P(·|正常)={p2_ham}），假设两个词独立：")
    print(f"    垃圾：{p_spam}×{p_word_spam}×{p2_spam} = {num_s:.4f}；正常：{1-p_spam:.1f}×{p_word_ham}×{p2_ham} = {num_h:.4f} → P(垃圾|两个词) = {num_s/(num_s+num_h):.3f}")
    print("  「朴素」= 假设特征在给定类别下彼此独立，于是 P(x|c) 是各特征概率的乘积（取对数变相加）")
    print()


# ---------------- 4. 高斯朴素贝叶斯：每类每特征一个正态分布 ----------------
def exp_nb():
    print("=== 4. 高斯朴素贝叶斯：手写 vs sklearn；为什么冗余特征让它变差 ===")
    Xtr, Xte, ytr, yte, _Xtr_s, _Xte_s = tabular()
    t0 = time.time(); nb = GaussianNaiveBayes().fit(Xtr, ytr); acc = (nb.predict(Xte) == yte).mean()
    print(f"  手写：fit {time.time()-t0:.4f}s，测试准确率 {acc:.3f}；sklearn GaussianNB {GaussianNB().fit(Xtr, ytr).score(Xte, yte):.3f}")
    # 去掉冗余特征后
    X, y = make_classification(n_samples=5000, n_features=16, n_informative=8, n_redundant=0, flip_y=0.05, class_sep=0.8, random_state=0)
    a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=0)
    print(f"  同样 8 个有用特征但去掉 4 个冗余特征：朴素贝叶斯 {GaussianNB().fit(a, c).score(b, d):.3f}，逻辑回归 {make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(a, c).score(b, d):.3f}")
    # 图：乳腺癌两个特征上每类的正态曲线
    data = load_breast_cancer()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for ax, j in zip(axes, [np.where(data.feature_names == "mean radius")[0][0], np.where(data.feature_names == "mean concave points")[0][0]]):
        xs = np.linspace(data.data[:, j].min(), data.data[:, j].max(), 200)
        for c, col, name in ((0, C["red"], "恶性"), (1, C["blue"], "良性")):
            v = data.data[data.target == c, j]
            mu, var = v.mean(), v.var()
            ax.hist(v, bins=25, density=True, color=col, alpha=0.25)
            ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var), color=col, lw=1.6, label=f"{name}：μ={mu:.2f}, σ={np.sqrt(var):.2f}")
        ax.set_title(f"特征「{data.feature_names[j]}」：每类拟一个正态分布", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7); ax.set_yticks([])
    save(fig, "04-nb-gaussians")
    gnb = GaussianNB().fit(data.data, data.target)
    x = data.data[0]
    print(f"  乳腺癌数据：fit 只是算了 mu {gnb.theta_.shape} 与 var {gnb.var_.shape} 两张表；30 个特征的高斯 NB 准确率（训练集）{gnb.score(data.data, data.target):.3f}")
    print(f"  样本 0 的 log P(x|类) 两类各为 {np.round(gnb.predict_joint_log_proba(x[None])[0], 1)} → 预测 {gnb.predict(x[None])[0]}（真实 {data.target[0]}）")
    print()


# ---------------- 5. KNN：k 的作用 ----------------
def exp_knn():
    print("=== 5. KNN：手写 vs sklearn；k 小方差大、k 大偏差大 ===")
    Xtr, Xte, ytr, yte, Xtr_s, Xte_s = tabular()
    t0 = time.time(); pred = knn_predict(Xtr_s, ytr, Xte_s, 15); dt = time.time() - t0
    print(f"  手写 KNN(15)：{len(Xte)}×{len(Xtr)} 个距离，{dt:.2f}s，测试准确率 {(pred == yte).mean():.3f}；sklearn {KNeighborsClassifier(15).fit(Xtr_s, ytr).score(Xte_s, yte):.3f}")
    ks = [1, 3, 5, 9, 15, 25, 51, 101, 201, 501]
    tr_acc, te_acc = [], []
    for k in ks:
        m = KNeighborsClassifier(k).fit(Xtr_s, ytr); tr_acc.append(m.score(Xtr_s, ytr)); te_acc.append(m.score(Xte_s, yte))
    print("  k:     " + " ".join(f"{k:>6}" for k in ks))
    print("  训练:  " + " ".join(f"{a:6.3f}" for a in tr_acc))
    print("  测试:  " + " ".join(f"{a:6.3f}" for a in te_acc))
    mXtr, mXte, mytr, myte = moons()
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    for ax, k in zip(axes[:3], (1, 15, 150)):
        m = KNeighborsClassifier(k).fit(mXtr, mytr)
        plot_boundary(ax, m.predict, mXtr, mytr, f"k = {k}：测试 {m.score(mXte, myte):.2f}")
    ax = axes[3]
    ax.plot(ks, tr_acc, "o-", ms=3, color=C["blue"], label="训练"); ax.plot(ks, te_acc, "o-", ms=3, color=C["red"], label="测试")
    ax.set_xscale("log"); ax.set_xlabel("k（对数轴）"); ax.set_ylabel("准确率"); ax.set_title("表格数据：k 从 1 到 501", fontsize=8); ax.legend(frameon=False, fontsize=7)
    save(fig, "04-knn-k")
    print("  k=1：训练 100%（每个点最近的是自己）、边界锯齿（方差大）；k 太大：边界过平、连月牙形状都丢了（偏差大）")
    print()


# ---------------- 6. 维度灾难：高维里所有点的距离都差不多 ----------------
def exp_curse():
    print("=== 6. 维度灾难：随机点之间最近与最远距离之比 ===")
    r = np.random.default_rng(0)
    dims = [1, 2, 5, 10, 50, 100, 500, 1000]
    ratios = []
    for d in dims:
        P = r.uniform(size=(500, d))
        dist = np.sqrt(((P[0] - P[1:]) ** 2).sum(1))
        ratios.append(dist.min() / dist.max())
        print(f"  维度 {d:>5}: 到最近点 {dist.min():7.3f}  到最远点 {dist.max():7.3f}  比值 {dist.min()/dist.max():.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.plot(dims, ratios, "o-", color=C["blue"]); ax.set_xscale("log")
    ax.set_xlabel("维度 d（对数轴）"); ax.set_ylabel("最近距离 / 最远距离"); ax.set_ylim(0, 1)
    ax.set_title("500 个随机点：维度越高，「最近的邻居」与「最远的点」越分不开", fontsize=8.5)
    save(fig, "04-curse-of-dimensionality")
    print("  1000 维时最近与最远只差 20%——「邻居」失去意义；所以 KNN / 检索要先学出好的低维表示（embedding）")
    print()


# ---------------- 7. 决策树：第一刀怎么选 ----------------
def exp_tree():
    print("=== 7. 决策树：第一刀怎么选（月牙数据）===")
    Xtr, Xte, ytr, yte = moons()
    print(f"  根节点 Gini = {gini(ytr):.3f}（{ytr.mean():.2f} 的样本是类 1）")
    # 对特征 x₂ 扫描所有阈值
    thrs = np.linspace(Xtr[:, 1].min(), Xtr[:, 1].max(), 200)
    def weighted_gini(j, thr):
        left = Xtr[:, j] <= thr
        if left.sum() == 0 or left.sum() == len(ytr): return gini(ytr)
        return left.mean() * gini(ytr[left]) + (1 - left.mean()) * gini(ytr[~left])
    g1 = [weighted_gini(0, t) for t in np.linspace(Xtr[:, 0].min(), Xtr[:, 0].max(), 200)]
    g2 = [weighted_gini(1, t) for t in thrs]
    j, thr = best_split(Xtr, ytr)
    print(f"  手写 best_split 选出：特征 x{j+1} ≤ {thr:.3f}，切完加权 Gini = {weighted_gini(j, thr):.3f}")
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr)
    t = tree.tree_
    print(f"  sklearn 深度 2 的树：根节点 x{t.feature[0]+1} ≤ {t.threshold[0]:.3f}；左子节点 x{t.feature[1]+1} ≤ {t.threshold[1]:.3f}；右子节点 x{t.feature[4]+1} ≤ {t.threshold[4]:.3f}")
    print(f"    叶子的样本数与类别比例：{[ (int(t.n_node_samples[i]), np.round(t.value[i][0] / t.value[i][0].sum(), 2).tolist()) for i in range(t.node_count) if t.children_left[i] == -1]}")
    print(f"    训练 {tree.score(Xtr, ytr):.3f} / 测试 {tree.score(Xte, yte):.3f}")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    ax = axes[0]
    ax.plot(np.linspace(Xtr[:, 0].min(), Xtr[:, 0].max(), 200), g1, color=C["gray"], label="沿 x₁ 切")
    ax.plot(thrs, g2, color=C["blue"], label="沿 x₂ 切")
    ax.axhline(gini(ytr), color=C["gray"], ls="--", lw=0.8)
    ax.plot(thr if j == 1 else t.threshold[0], weighted_gini(j, thr), "o", color=C["red"])
    ax.set_xlabel("阈值"); ax.set_ylabel("切完的加权 Gini"); ax.set_title("扫描每个阈值，选最低点", fontsize=8.5); ax.legend(frameon=False, fontsize=7)
    plot_boundary(axes[1], DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr, ytr).predict, Xtr, ytr, f"深度 1：一刀（x{t.feature[0]+1} ≤ {t.threshold[0]:.2f}）")
    plot_boundary(axes[2], tree.predict, Xtr, ytr, f"深度 2：三刀，测试 {tree.score(Xte, yte):.2f}")
    save(fig, "04-tree-first-split")
    print()


# ---------------- 8. 深度与过拟合 ----------------
def exp_depth():
    print("=== 8. 决策树的深度：手写 vs sklearn；深度不限 = 背下训练集 ===")
    Xtr, Xte, ytr, yte, _, _ = tabular()
    print(f"  {'max_depth':>9} {'手写 训练/测试':>16} {'sklearn 训练/测试':>18}")
    for md in (1, 2, 3, 6, 10, None):
        tree = build_tree(Xtr, ytr, 0, md if md else 10**9)
        tr = np.mean([tree_predict(tree, x) for x in Xtr] == ytr); te = np.mean([tree_predict(tree, x) for x in Xte] == yte)
        sk = DecisionTreeClassifier(max_depth=md, random_state=0).fit(Xtr, ytr)
        print(f"  {md!s:>9} {tr:>7.3f} / {te:.3f} {sk.score(Xtr, ytr):>9.3f} / {sk.score(Xte, yte):.3f}")
    depths = list(range(1, 26))
    tr_acc = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xtr, ytr) for d in depths]
    te_acc = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xte, yte) for d in depths]
    mXtr, mXte, mytr, myte = moons()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    for ax, d in zip(axes[:2], (3, None)):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(mXtr, mytr)
        plot_boundary(ax, m.predict, mXtr, mytr, f"深度 {d if d else '不限'}：训练 {m.score(mXtr, mytr):.2f} / 测试 {m.score(mXte, myte):.2f}")
    ax = axes[2]
    ax.plot(depths, tr_acc, "o-", ms=3, color=C["blue"], label="训练"); ax.plot(depths, te_acc, "o-", ms=3, color=C["red"], label="测试")
    ax.set_xlabel("max_depth"); ax.set_ylabel("准确率"); ax.set_title("表格数据：深度 1 到 25", fontsize=8); ax.legend(frameon=False, fontsize=7)
    save(fig, "04-tree-depth")
    print("  深度不限时每个叶子只剩一个样本，训练 100%（含 5% 翻转的标签）、测试掉到 83%；测试准确率在深度 6–8 附近最高")
    print()


# ---------------- 9. 多项式朴素贝叶斯手算：计数、平滑、log-space ----------------
class MultinomialNaiveBayes:
    """词计数版朴素贝叶斯：log P(c | d) ∝ log P(c) + Σ_w count(w, d)·log P(w | c)。"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.log_prior = np.log([(y == c).mean() for c in self.classes])               # !ref mnb-prior
        counts = np.array([X[y == c].sum(0) for c in self.classes])                    # !ref mnb-count
        smoothed = counts + self.alpha                                                   # !ref mnb-smooth
        self.log_prob = np.log(smoothed / smoothed.sum(1, keepdims=True))               # !ref mnb-logprob
        return self

    def joint_log(self, X):
        return self.log_prior + X @ self.log_prob.T                                      # !ref mnb-joint

    def predict(self, X):
        return self.classes[self.joint_log(X).argmax(1)]

    def predict_proba(self, X):
        J = self.joint_log(X)
        J = J - J.max(1, keepdims=True)                                                  # !ref mnb-lse
        return np.exp(J) / np.exp(J).sum(1, keepdims=True)


def exp_mnb():
    print("=== 9. 多项式朴素贝叶斯手算：计数、平滑、log-space ===")
    from sklearn.naive_bayes import MultinomialNB
    vocab = ["免费", "点击", "会议", "明天", "中奖"]
    docs = ["免费 中奖 点击", "免费 免费 点击", "中奖 点击", "会议 明天", "明天 会议 会议", "免费 会议"]
    y = np.array([1, 1, 1, 0, 0, 0])
    X = np.array([[d.split().count(w) for w in vocab] for d in docs], dtype=float)
    print(f"  词表 {vocab}；6 条短信，前 3 条 spam（y=1）、后 3 条 ham（y=0）：")
    for d, row, c in zip(docs, X, y):
        print(f"    {'spam' if c else 'ham ':<4}  {d:<12} → 计数 {row.astype(int).tolist()}")
    for c, name in ((1, "spam"), (0, "ham ")):
        cnt = X[y == c].sum(0).astype(int)
        print(f"  {name} 的词计数 {cnt.tolist()}，共 {cnt.sum()} 个词")
    test = np.array([[d.split().count(w) for w in vocab] for d in ["点击 会议 会议"]], dtype=float)
    print("  待判短信「点击 会议 会议」→ 计数", test.astype(int).ravel().tolist())
    for alpha in (0.0, 1.0):
        with np.errstate(divide="ignore"):
            m = MultinomialNaiveBayes(alpha=alpha).fit(X, y)
        tag = "不平滑（α=0）" if alpha == 0 else f"拉普拉斯平滑（α={alpha:g}）"
        print(f"  {tag}：P(w | ham) = {np.round(np.exp(m.log_prob[0]), 4).tolist()}，P(w | spam) = {np.round(np.exp(m.log_prob[1]), 4).tolist()}")
        lp = np.where(test[0] > 0, m.log_prob, 0.0)                                      # 没出现的词不参与，避免 0 × log 0
        J = m.log_prior + (test[0] * lp).sum(1)
        with np.errstate(invalid="ignore"):
            p_ham = np.exp(J[0] - np.logaddexp(J[0], J[1]))
        verdict = "两边都是 −inf，argmax 无意义" if np.isinf(J).all() else f"判 {'spam' if J.argmax() else 'ham'}"
        print(f"    log P(ham) + Σ count·log P(w|ham) = {J[0]:.4f}，spam 侧 = {J[1]:.4f} → {verdict}，P(ham | d) = {p_ham:.4f}")
    print("  α=0 时「点击」在 ham 里没出现过、「会议」在 spam 里没出现过，各有一项 log 0 = −inf，两类同时被一票否决；α=1 给每个词的计数加 1：P(点击|ham) = (0+1)/(7+5) = 1/12，P(会议|spam) = (0+1)/(8+5) = 1/13")
    sk = MultinomialNB(alpha=1.0).fit(X, y)
    m1 = MultinomialNaiveBayes(alpha=1.0).fit(X, y)
    print(f"  与 sklearn MultinomialNB(alpha=1) 对照：feature_log_prob_ 最大差 {np.abs(sk.feature_log_prob_ - m1.log_prob).max():.1e}，"
          f"predict_log_proba 最大差 {np.abs(sk.predict_log_proba(test) - np.log(m1.predict_proba(test))).max():.1e}（对齐条件：同一个 α、fit_prior=True 用类频率做先验）")
    # 下溢：为什么必须在 log 域累加
    long = np.array([[120, 120, 120, 120, 120]], dtype=float)
    with np.errstate(under="ignore"):
        prod = np.exp(m1.log_prior)[None] * np.prod(np.exp(m1.log_prob)[None] ** long, axis=2)
    print(f"  一条 600 词的短信（每个词 120 次）：直接乘概率 → P(ham)·ΠP(w|ham) = {prod[0, 0]}, spam 侧 = {prod[0, 1]}——两边都下溢成 0，argmax 无从比较；"
          f"log 域累加 → {np.round(m1.joint_log(long)[0], 1).tolist()}，照常比大小")
    # 反例 1：把同一个词复制成两个特征 → 同一条证据算两次
    Xd = np.c_[X, X[:, [2]]]
    td = np.c_[test, test[:, [2]]]
    md = MultinomialNaiveBayes(alpha=1.0).fit(Xd, y)
    print(f"  反例 1——条件独立假设失效：把「会议」复制成第 6 个特征（与原列完全相关），P(ham | d) 从 {m1.predict_proba(test)[0, 0]:.4f} 变成 {md.predict_proba(td)[0, 0]:.4f}；"
          "同一条证据被算了两次，后验被推得更极端——相关特征多的文本上，朴素贝叶斯给出的概率往往比实际更「自信」")
    # 反例 2：XOR——每个特征单看都与类别无关，只有合起来才有信息
    r = np.random.default_rng(0)
    Xx = r.integers(0, 2, (400, 2))
    yx = Xx[:, 0] ^ Xx[:, 1]
    from sklearn.naive_bayes import BernoulliNB
    nb_acc = BernoulliNB().fit(Xx[:200], yx[:200]).score(Xx[200:], yx[200:])
    tree_acc = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xx[:200], yx[:200]).score(Xx[200:], yx[200:])
    print(f"  反例 2——XOR：y = x₁ XOR x₂，P(x₁=1 | y) = {Xx[yx == 1, 0].mean():.2f}/{Xx[yx == 0, 0].mean():.2f}（两类几乎一样），"
          f"朴素贝叶斯测试准确率 {nb_acc:.3f}，深度 2 的决策树 {tree_acc:.3f}；每个特征单独看都没信息，NB 的「各算各的再相加」注定失败")
    print()


# ---------------- 10. KNN：缩放、距离度量与投票规则 ----------------
def knn_vote(Xtrain, ytrain, q, k, metric="euclidean", weighted=False):
    """对一个查询点 q：算距离 → 取 k 近邻 → 投票。返回 (邻居下标, 距离, 每类票数, 预测)。"""
    if metric == "euclidean":
        d = np.sqrt(((Xtrain - q) ** 2).sum(1))                                          # !ref knn-euclid
    elif metric == "manhattan":
        d = np.abs(Xtrain - q).sum(1)                                                   # !ref knn-manhattan
    else:
        d = 1 - (Xtrain @ q) / (np.linalg.norm(Xtrain, axis=1) * np.linalg.norm(q))     # !ref knn-cosine
    idx = np.argsort(d, kind="stable")[:k]                                              # !ref knn-sort
    w = 1 / d[idx] if weighted else np.ones(k)                                          # !ref knn-weight
    votes = np.array([w[ytrain[idx] == c].sum() for c in (0, 1)])                       # !ref knn-votes
    return idx, d[idx], votes, int(votes.argmax())                                      # !ref knn-argmax


def exp_knn_rules():
    print("=== 10. KNN：缩放、距离度量与投票规则（六个训练点）===")
    X = np.array([[20, 30000], [22, 32000], [24, 31000], [50, 80000], [52, 85000], [54, 82000]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    q = np.array([51.0, 40000.0])
    print("  训练点（年龄, 年收入/元）与类别：")
    for row, c in zip(X, y):
        print(f"    ({row[0]:.0f}, {row[1]:,.0f}) → 类 {c}")
    print(f"  查询点 q = ({q[0]:.0f}, {q[1]:,.0f})，k = 3")
    idx, d, votes, pred = knn_vote(X, y, q, 3)
    print(f"  原始尺度、欧氏距离：最近 3 个是 {[f'({X[i, 0]:.0f}, {X[i, 1]:,.0f}) d={d[t]:,.0f}' for t, i in enumerate(idx)]} → 票数 {votes.astype(int).tolist()} → 判 {pred}（收入的差是几千到几万，年龄的差是几十，距离里年龄等于不存在）")
    mu, sd = X.mean(0), X.std(0)
    Z, qz = (X - mu) / sd, (q - mu) / sd
    print(f"  按训练集标准化：均值 {mu.tolist()}，标准差 {np.round(sd, 1).tolist()}；q → {np.round(qz, 3).tolist()}")
    for metric in ("euclidean", "manhattan", "cosine"):
        idx, d, votes, pred = knn_vote(Z, y, qz, 3, metric)
        print(f"  标准化后、{metric:<9}：最近 3 个 {[f'#{i} d={d[t]:.3f}' for t, i in enumerate(idx)]} → 票数 {votes.astype(int).tolist()} → 判 {pred}")
    sk = KNeighborsClassifier(n_neighbors=3).fit(Z, y)
    print(f"  sklearn KNeighborsClassifier(3) 在标准化数据上判 {sk.predict(qz[None])[0]}，邻居 {sk.kneighbors(qz[None], return_distance=False)[0].tolist()}（与手写一致）")
    # 投票规则：k 取偶数会平票（一维小例子）
    X1 = np.array([[1.0], [2.0], [8.0], [9.0]])
    y1 = np.array([1, 1, 0, 0])
    q1 = np.array([5.2])
    idx, d, votes, pred = knn_vote(X1, y1, q1, 4)
    _idx_w, _d_w, votes_w, pred_w = knn_vote(X1, y1, q1, 4, weighted=True)
    sk4 = KNeighborsClassifier(n_neighbors=4).fit(X1, y1).predict(q1[None])[0]
    sk4w = KNeighborsClassifier(n_neighbors=4, weights="distance").fit(X1, y1).predict(q1[None])[0]
    print(f"  投票规则：一维训练点 x = {X1.ravel().tolist()}，类 {y1.tolist()}，查询 {q1[0]}，k = 4，距离 {np.round(d[np.argsort(idx)], 1).tolist()}")
    print(f"    多数票：票数 {votes.astype(int).tolist()} 平票，手写 argmax 取第一个最大 → 判 {pred}；sklearn（scipy mode，平票取较小标签）→ 判 {sk4}")
    print(f"    距离加权（权重 1/d）：票数 {np.round(votes_w, 3).tolist()} → 判 {pred_w}；sklearn weights='distance' → 判 {sk4w}。k 取奇数或用距离加权，才不会把结论交给「谁的标签编号小」")
    # 反例：无关特征淹没距离
    Xtr, Xte, ytr, yte = moons()
    r = np.random.default_rng(0)
    base = make_pipeline(StandardScaler(), KNeighborsClassifier(15))
    acc0 = base.fit(Xtr, ytr).score(Xte, yte)
    print(f"  反例——「近的点同类」失效：月牙数据 2 维，标准化 + KNN(15) 测试准确率 {acc0:.3f}")
    for extra in (10, 50, 200):
        noise_tr, noise_te = r.normal(size=(len(Xtr), extra)), r.normal(size=(len(Xte), extra))
        a = base.fit(np.c_[Xtr, noise_tr], ytr).score(np.c_[Xte, noise_te], yte)
        aw = make_pipeline(StandardScaler(), KNeighborsClassifier(15, weights="distance")).fit(np.c_[Xtr, noise_tr], ytr).score(np.c_[Xte, noise_te], yte)
        print(f"    加 {extra:>3} 个与标签无关的 N(0,1) 特征：{a:.3f}（距离加权 {aw:.3f}）")
    print("  无关特征已经是单位尺度，标准化救不了它；距离被噪声维度主导，「最近」的点与标签无关——KNN 没有任何机制学会忽略一个特征")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, data, qq, title in ((axes[0], X, q, "原始尺度：收入（元）决定一切"), (axes[1], Z, qz, "标准化后：两个特征同权")):
        idx = knn_vote(data, y, qq, 3)[0]
        ax.scatter(data[:, 0], data[:, 1], c=[C["blue"] if c == 0 else C["orange"] for c in y], s=40, zorder=3)
        ax.scatter(data[idx, 0], data[idx, 1], s=160, facecolors="none", edgecolors=C["red"], lw=1.5, zorder=4, label="最近 3 个邻居")
        ax.scatter(*qq, marker="*", s=160, color="k", zorder=5, label="查询点")
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("年龄" if data is X else "年龄（标准化）"); ax.set_ylabel("年收入" if data is X else "年收入（标准化）")
        ax.legend(frameon=False, fontsize=7)
    save(fig, "04-knn-scaling")
    print()


# ---------------- 11. 候选切点：教学树扫分位点，库扫所有相邻中点 ----------------
def best_split_exact(X, y, min_samples_leaf=1):
    """库算法的做法：每个特征排序后，取相邻不同取值的中点为候选，全部扫一遍。返回 (gini, j, thr, 候选数)。"""
    best: tuple[float, int | None, float | None] = (gini(y), None, None)
    n_cand = 0
    for j in range(X.shape[1]):
        v = np.unique(X[:, j])                                                           # !ref exact-unique
        cands = (v[:-1] + v[1:]) / 2                                                     # !ref exact-mid
        n_cand += len(cands)
        for thr in cands:
            left = X[:, j] <= thr
            if min(left.sum(), (~left).sum()) < min_samples_leaf:                        # !ref exact-minleaf
                continue
            g = left.mean() * gini(y[left]) + (1 - left.mean()) * gini(y[~left])          # !ref exact-gini
            if g < best[0]:
                best = (g, j, thr)
    return best + (n_cand,)


def exp_split():
    print("=== 11. 候选切点：教学树只扫 19 个分位点，库算法扫所有相邻中点 ===")
    Xtr, _Xte, ytr, _yte = moons()
    n = len(ytr)

    def weighted_gini(j, thr):
        left = Xtr[:, j] <= thr
        return left.mean() * gini(ytr[left]) + (1 - left.mean()) * gini(ytr[~left])

    jq, tq = best_split(Xtr, ytr)
    g_exact, je, te, n_cand = best_split_exact(Xtr, ytr)
    print(f"  {n} 个训练点、2 个特征")
    print(f"  教学树 best_split：每个特征 19 个分位点（5%, 10%, …, 95%），共 38 个候选 → x{jq + 1} ≤ {tq:.4f}，加权 Gini {weighted_gini(jq, tq):.4f}")
    print(f"  精确扫描 best_split_exact：每个特征 {n - 1} 个相邻中点，共 {n_cand} 个候选 → x{je + 1} ≤ {te:.4f}，加权 Gini {g_exact:.4f}")
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr, ytr)
    t = stump.tree_
    g_sk = (t.weighted_n_node_samples[1] * t.impurity[1] + t.weighted_n_node_samples[2] * t.impurity[2]) / n
    print(f"  sklearn 深度 1：x{t.feature[0] + 1} ≤ {t.threshold[0]:.4f}，子节点加权 Gini {g_sk:.4f}（与精确扫描逐位一致：阈值差 {abs(t.threshold[0] - te):.1e}）")
    qs = np.percentile(Xtr[:, je], np.arange(5, 100, 5))
    lo, hi = qs[qs < te].max(), qs[qs > te].min()
    print(f"  精确最优阈值 {te:.4f} 落在教学树的两个分位点 {lo:.4f} 与 {hi:.4f} 之间，19 个分位点里没有一个碰到它；两者的 Gini 差 {weighted_gini(jq, tq) - g_exact:.4f}")
    print(f"  对应到数据：精确阈值左侧 {int((Xtr[:, je] <= te).sum())} 个点，分位点阈值左侧 {int((Xtr[:, jq] <= tq).sum())} 个点——差了 {int((Xtr[:, jq] <= tq).sum()) - int((Xtr[:, je] <= te).sum())} 个样本的归属")
    print("  说「两种切法 Gini 差不多」掩盖了三件事：候选集不同（38 vs 398）、复杂度不同（O(特征×19) vs O(特征×n log n)）、阈值语义不同（分位点插值 vs 相邻样本中点）")
    print()


# ---------------- 12. 递归生长：十个点的完整生长记录 + 停止 + 剪枝 ----------------
def grow(X, y, depth, max_depth, min_samples_leaf, log, name="根"):
    """带日志的递归生长。叶子记为 dict(n, counts, gini)，内部节点再带 feature/thr/left/right。"""
    counts = np.bincount(y, minlength=2)
    node = {"n": len(y), "counts": counts, "gini": gini(y), "name": name}
    pad = "    " + "  " * depth
    head = f"{pad}{name}（深度 {depth}）：{len(y)} 个点，类计数 {counts.tolist()}，Gini {node['gini']:.3f}"
    if node["gini"] == 0:                                                                # !ref grow-pure
        log.append(head + " → 纯节点，停")
        return node
    if depth == max_depth:                                                               # !ref grow-depth
        log.append(head + " → 到达 max_depth，停")
        return node
    if len(y) < 2 * min_samples_leaf:                                                    # !ref grow-minleaf
        log.append(head + f" → 样本数不足以分出两个 ≥{min_samples_leaf} 的叶子，停")
        return node
    g, j, thr, n_cand = best_split_exact(X, y, min_samples_leaf)                         # !ref grow-split
    if j is None:
        log.append(head + " → 没有任何（满足叶子最小样本数的）切点能降低 Gini，停")
        return node
    left = X[:, j] <= thr
    log.append(head + f" → 扫 {n_cand} 个候选，选 x{j + 1} ≤ {thr:.2f}（加权 Gini {node['gini']:.3f} → {g:.3f}），左 {int(left.sum())} / 右 {int((~left).sum())}")
    node.update(feature=j, thr=thr,
                left=grow(X[left], y[left], depth + 1, max_depth, min_samples_leaf, log, name + "-左"),   # !ref grow-recurse
                right=grow(X[~left], y[~left], depth + 1, max_depth, min_samples_leaf, log, name + "-右"))
    return node


def leaves(node):
    return [node] if "left" not in node else leaves(node["left"]) + leaves(node["right"])


def internal_nodes(node):
    return [] if "left" not in node else [node] + internal_nodes(node["left"]) + internal_nodes(node["right"])


def weakest_link(root, N):
    """代价复杂度剪枝的一步：对每个内部节点算 α_eff = (R(t) − R(T_t)) / (|叶子(T_t)| − 1)，剪掉最小的那个。"""
    alphas = []
    for t in internal_nodes(root):
        R_t = t["gini"] * t["n"] / N                                                       # !ref ccp-rt
        R_T = sum(lf["gini"] * lf["n"] / N for lf in leaves(t))                            # !ref ccp-rT
        alphas.append((R_t - R_T) / (len(leaves(t)) - 1))                                 # !ref ccp-alpha
    alpha = min(alphas)
    names = []
    for t in internal_nodes(root):                                                         # α 相同的节点一起剪（sklearn 也是）
        R_t = t["gini"] * t["n"] / N
        R_T = sum(lf["gini"] * lf["n"] / N for lf in leaves(t))
        if abs((R_t - R_T) / (len(leaves(t)) - 1) - alpha) < 1e-12:
            names.append(t["name"])
            for key in ("feature", "thr", "left", "right"):
                t.pop(key)                                                                 # !ref ccp-cut
    return alpha, names


def exp_grow():
    print("=== 12. 递归生长：十个点的完整生长记录、停止条件与代价复杂度剪枝 ===")
    X = np.array([[1, 1], [2, 3], [3, 2], [2, 6], [7, 8], [6, 2], [7, 3], [8, 1], [6, 7], [3, 7]], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    print("  十个点 (x₁, x₂) → 类：" + "  ".join(f"({a:.0f},{b:.0f})→{c}" for (a, b), c in zip(X, y)))
    log: list[str] = []
    root = grow(X, y, 0, max_depth=None, min_samples_leaf=1, log=log)
    print("  不限深度、min_samples_leaf=1 的生长记录：")
    print("\n".join(log))
    print(f"  长成 {len(leaves(root))} 片叶子，其中不纯的叶子 {sum(lf['gini'] > 0 for lf in leaves(root))} 片")
    sk = DecisionTreeClassifier(random_state=0).fit(X, y)
    print("  sklearn 同参数的树（export_text）：")
    from sklearn.tree import export_text
    print("\n".join("    " + line for line in export_text(sk, feature_names=["x1", "x2"], decimals=2).rstrip().split("\n")))
    for md, msl in ((1, 1), (None, 3)):
        log = []
        r2 = grow(X, y, 0, max_depth=md, min_samples_leaf=msl, log=log)
        skp = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=0).fit(X, y)
        print(f"  预剪枝 max_depth={md}, min_samples_leaf={msl}：手写 {len(leaves(r2))} 片叶子（{[l['counts'].tolist() for l in leaves(r2)]}），sklearn {skp.get_n_leaves()} 片叶子，训练准确率 {skp.score(X, y):.3f}")
    # 代价复杂度剪枝：从满树开始一次剪掉一个「最弱的」内部节点
    N = len(y)
    alphas = [0.0]
    sizes = [len(leaves(root))]
    print("  代价复杂度剪枝（weakest link）：每步算 α_eff = (R(t) − R(T_t)) / (|叶子| − 1)，剪掉最小的：")
    while "left" in root:
        a, names = weakest_link(root, N)
        alphas.append(a)
        sizes.append(len(leaves(root)))
        print(f"    α_eff = {a:.4f}：把「{'」「'.join(names)}」剪成叶子，剩 {sizes[-1]} 片叶子")
    # 停止规则的另一处差别：零增益的切分
    Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    yx = np.array([0, 1, 1, 0])
    log = []
    rx = grow(Xx, yx, 0, None, 1, log)
    skx = DecisionTreeClassifier(random_state=0).fit(Xx, yx)
    print(f"  四个 XOR 点：任何一刀都不能降低 Gini（0.5 → 0.5），手写 grow 停在根、{len(leaves(rx))} 片叶子；"
          f"sklearn 默认 min_impurity_decrease=0.0，零增益也切，长到 {skx.get_n_leaves()} 片叶子、训练全对——「不纯就继续切」与「有增益才切」是两种停止规则")
    path = sk.cost_complexity_pruning_path(X, y)
    print(f"  sklearn cost_complexity_pruning_path 的 ccp_alphas：{np.round(path.ccp_alphas, 4).tolist()}")
    print(f"  手写序列：{np.round(alphas, 4).tolist()}（对齐条件：R(t) 用 Gini × 节点样本占比；两边都用加权不纯度，不是误分类数）")
    # 真实一点的数据：α 怎么选
    Xtr, Xte, ytr, yte, _, _ = tabular()
    full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    pth = full.cost_complexity_pruning_path(Xtr, ytr)
    cands = np.unique(np.round(pth.ccp_alphas, 5))
    cands = cands[cands < 0.05][::4]
    from sklearn.model_selection import cross_val_score
    print(f"  表格数据（{len(ytr)} 训练）：满树 {full.get_n_leaves()} 片叶子，候选 α 共 {len(cands)} 个；用 5 折交叉验证挑 α：")
    cv, tr_acc, te_acc, n_leaf = [], [], [], []
    for a in cands:
        m = DecisionTreeClassifier(random_state=0, ccp_alpha=a)
        cv.append(cross_val_score(m, Xtr, ytr, cv=5).mean())
        m.fit(Xtr, ytr)
        tr_acc.append(m.score(Xtr, ytr)); te_acc.append(m.score(Xte, yte)); n_leaf.append(m.get_n_leaves())
    best = int(np.argmax(cv))
    for k in (0, best, len(cands) - 1):
        print(f"    α = {cands[k]:.5f}：{n_leaf[k]:>4} 片叶子，交叉验证 {cv[k]:.3f}，训练 {tr_acc[k]:.3f}，测试 {te_acc[k]:.3f}" + ("   ← CV 选中" if k == best else ""))
    print("  剪枝后的树比 max_depth 限制更「不均匀」：该深的分支留着，没用的分支整枝剪掉；α 要用交叉验证挑，用测试集挑就是在测试集上训练")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    ax.plot(cands, tr_acc, color=C["gray"], label="训练"); ax.plot(cands, cv, color=C["blue"], label="5 折交叉验证"); ax.plot(cands, te_acc, color=C["red"], label="测试")
    ax.axvline(cands[best], ls="--", color=C["gray"], lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-3); ax.set_xlabel("ccp_alpha"); ax.set_ylabel("准确率"); ax.set_title("α 越大剪得越狠：训练降、测试先升后降", fontsize=8.5); ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ax.plot(cands, n_leaf, color=C["green"]); ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("log")
    ax.set_xlabel("ccp_alpha"); ax.set_ylabel("叶子数"); ax.set_title(f"叶子从 {n_leaf[0]} 剪到 {n_leaf[-1]}", fontsize=8.5)
    save(fig, "04-tree-pruning")
    print()


# ---------------- 13. 失效反例：斜着的边界 ----------------
def exp_diag():
    print("=== 13. 决策树的失效反例：斜着的边界要用楼梯去逼近 ===")
    r = np.random.default_rng(0)
    X = r.uniform(-1, 1, (2000, 2))
    y = (X[:, 0] > X[:, 1]).astype(int)
    Xtr, Xte, ytr, yte = X[:1000], X[1000:], y[:1000], y[1000:]
    R = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    Xr_tr, Xr_te = Xtr @ R.T, Xte @ R.T
    print("  2000 个均匀点，y = [x₁ > x₂]：边界是对角线；再把坐标轴转 45°，边界变成 x₁' = 0")
    print("    max_depth   原坐标：叶子数 / 测试准确率   转 45° 后：叶子数 / 测试准确率")
    for md in (1, 2, 4, 8, None):
        a = DecisionTreeClassifier(max_depth=md, random_state=0).fit(Xtr, ytr)
        b = DecisionTreeClassifier(max_depth=md, random_state=0).fit(Xr_tr, ytr)
        print(f"    {md!s:>9}   {a.get_n_leaves():>8} / {a.score(Xte, yte):.3f}            {b.get_n_leaves():>5} / {b.score(Xr_te, yte):.3f}")
    lr = LogisticRegression().fit(Xtr, ytr).score(Xte, yte)
    full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    print(f"  对照：逻辑回归在原坐标下测试准确率 {lr:.3f}——一条直线就是答案，树用 {full.get_n_leaves()} 片叶子砌楼梯也只到 {full.score(Xte, yte):.3f}；同一个问题，换坐标系后一刀就够")
    print("  树的核心假设是「边界平行于坐标轴」；特征之间的线性组合（x₁ − x₂）它表达不了，只能用很多刀逼近，而每一刀都在消耗样本")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, Xa, md, title in zip(axes, (Xtr, Xr_tr), (4, 1), ("原坐标：深度 4 的树切出楼梯", "转 45°：深度 1 的树一刀切齐")):
        m = DecisionTreeClassifier(max_depth=md, random_state=0).fit(Xa, ytr)
        xx, yy = np.meshgrid(np.linspace(-1.45, 1.45, 120), np.linspace(-1.45, 1.45, 120))
        Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]], alpha=0.18)
        ax.scatter(Xa[ytr == 0, 0], Xa[ytr == 0, 1], s=4, color=C["blue"]); ax.scatter(Xa[ytr == 1, 0], Xa[ytr == 1, 1], s=4, color=C["red"])
        ax.set_title(f"{title}（{m.get_n_leaves()} 片叶子，测试 {m.score(Xte if Xa is Xtr else Xr_te, yte):.3f}）", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "04-tree-diagonal")
    print()


EXPS = {"compare": exp_compare, "grid": exp_grid, "bayes": exp_bayes, "nb": exp_nb, "knn": exp_knn, "curse": exp_curse, "tree": exp_tree, "depth": exp_depth,
        "mnb": exp_mnb, "knn_rules": exp_knn_rules, "split": exp_split, "grow": exp_grow, "diag": exp_diag}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
