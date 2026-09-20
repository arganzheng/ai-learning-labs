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
    best = (gini(y), None, None)
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
    Xtr, Xte, ytr, yte, Xtr_s, Xte_s = tabular()
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
        print(f"  {str(md):>9} {tr:>7.3f} / {te:.3f} {sk.score(Xtr, ytr):>9.3f} / {sk.score(Xte, yte):.3f}")
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


EXPS = {"compare": exp_compare, "grid": exp_grid, "bayes": exp_bayes, "nb": exp_nb, "knn": exp_knn, "curse": exp_curse, "tree": exp_tree, "depth": exp_depth}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
