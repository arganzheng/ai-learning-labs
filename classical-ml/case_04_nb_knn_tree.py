"""案例（经典 ML 04）：三个基础分类器各一个经典问题。
https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html

    python case_04_nb_knn_tree.py            # 全部：nb knn tree
    python case_04_nb_knn_tree.py tree       # 只跑一个

nb   朴素贝叶斯做垃圾短信（与 03 篇的逻辑回归同一份数据、同一个划分）
knn  KNN 做 MNIST 手写数字（LeCun 1998 对比表里的那一行）
tree 决策树做泰坦尼克生还预测（可解释的树 + 一个泄漏的陷阱）
图输出到 out/case-04-*.svg。
"""
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from _data import mnist, sms_spam, titanic
from _plot import C, plt, save


def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return np.mean(y == pred), p, r, f


# ---------------- A. 朴素贝叶斯：垃圾短信 ----------------
def exp_nb():
    print("=== A. 朴素贝叶斯做垃圾短信（与 03 篇逻辑回归同一份数据、同一个划分）===")
    df = sms_spam()
    y = (df.label == "spam").astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(df.text, y, test_size=0.2, random_state=0, stratify=y)
    print(f"  {'模型':<40}{'准确率':>7}{'精确率':>7}{'召回率':>7}{'F1':>7}{'训练':>8}")
    rows = {}
    for name, model in [
        ("朴素贝叶斯（词计数，α=1 拉普拉斯平滑）", make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))),
        ("朴素贝叶斯，α=0.1", make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.1))),
        ("朴素贝叶斯，α=1e-9（几乎不平滑）", make_pipeline(CountVectorizer(), MultinomialNB(alpha=1e-9))),
        ("对照：TF-IDF + 逻辑回归（03 篇）", make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(C=10, max_iter=2000))),
    ]:
        t = time.time(); model.fit(Xtr, ytr); dt = time.time() - t
        a, p, r, f = prf(yte, model.predict(Xte))
        rows[name] = (a, p, r, f)
        print(f"  {name:<40}{a:>7.3f}{p:>7.3f}{r:>7.3f}{f:>7.3f}{dt * 1000:>6.0f}ms")

    nb = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(Xtr, ytr)
    vocab = nb[0].get_feature_names_out()
    lp = nb[-1].feature_log_prob_                                  # [2, V]：log P(词 | ham), log P(词 | spam)
    ratio = lp[1] - lp[0]                                          # log P(词|spam)/P(词|ham)：Paul Graham 的"spam 概率表"
    counts = np.asarray(nb[0].transform(Xtr).sum(0)).ravel()
    print(f"\n  词表 {len(vocab):,} 个词；模型 = 每个词在 ham / spam 里各一个概率（{2 * len(vocab):,} 个数）+ 两个先验")
    print(f"  先验：P(spam) = {np.exp(nb[-1].class_log_prior_[1]):.3f}")
    common = counts >= 5
    idx = np.argsort(ratio)
    print("  最'spam'的词（log P(词|spam)/P(词|ham)）：" + "  ".join(f"{vocab[i]}({ratio[i]:+.1f})" for i in idx[common[idx]][::-1][:12]))
    print("  最'ham'的词：" + "  ".join(f"{vocab[i]}({ratio[i]:+.1f})" for i in idx[common[idx]][:12]))
    # 一条短信怎么算
    msg = "Free entry in 2 a wkly comp to win FA Cup final tkts"
    x = nb[0].transform([msg])
    words = [vocab[j] for j in x.nonzero()[1]]
    contrib = [ratio[j] for j in x.nonzero()[1]]
    tot = nb[-1].class_log_prior_[1] - nb[-1].class_log_prior_[0] + sum(contrib)
    print(f"\n  一条短信怎么算：'{msg}'")
    print("   " + " ".join(f"{w}({c:+.1f})" for w, c in zip(words, contrib)))
    print(f"   先验 log(P(spam)/P(ham)) = {nb[-1].class_log_prior_[1] - nb[-1].class_log_prior_[0]:+.2f}，加上每个词的贡献 = {tot:+.1f} → P(spam) = {1 / (1 + np.exp(-tot)):.4f}")
    print("  解读：每个词独立投票、把 log 比值加起来——这就是'朴素'（假设词之间独立）；和逻辑回归的 wᵀx 形式一样，"
          "区别在权重怎么来：NB 数频率（一遍扫过就好），LR 用梯度下降学。")

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    sel = np.r_[idx[common[idx]][:12], idx[common[idx]][::-1][:12][::-1]]
    ax.barh(range(len(sel)), ratio[sel], color=[C["green"] if ratio[i] < 0 else C["red"] for i in sel])
    ax.set_yticks(range(len(sel))); ax.set_yticklabels([vocab[i] for i in sel], fontsize=8)
    ax.set_xlabel("log P(词 | spam) − log P(词 | ham)（训练集里出现 ≥ 5 次的词）")
    ax.set_title("朴素贝叶斯的「spam 概率表」：每个词一个数，数出来的")
    save(fig, "case-04-nb-word-ratios")


# ---------------- B. KNN：MNIST ----------------
def exp_knn():
    print("\n=== B. KNN 做 MNIST 手写数字 ===")
    X, y = mnist("train"); Xt, yt = mnist("test")
    print(f"  训练 {len(X):,} 张、测试 {len(Xt):,} 张，每张 28×28 = 784 个像素 → 一个 784 维的点；'模型' = 存下全部训练点")
    # 先在 10k 子集上扫 k
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), 10000, replace=False)
    Xs, ys = X[sub], y[sub]
    Xv, yv = Xt[:2000], yt[:2000]
    print(f"\n  在 10,000 张训练子集上扫 k（验证用 2,000 张测试图）：")
    ks = [1, 3, 5, 7, 11, 21, 51]
    errs = []
    for k in ks:
        m = KNeighborsClassifier(k, n_jobs=-1).fit(Xs, ys)
        e = 1 - m.score(Xv, yv); errs.append(e)
        print(f"    k = {k:<3} 错误率 {e:.3%}")
    # 全量
    print(f"\n  全量 60,000 张训练、10,000 张测试：")
    for k, metric in [(3, "euclidean"), (3, "cosine")]:
        m = KNeighborsClassifier(k, metric=metric, algorithm="brute", n_jobs=-1).fit(X, y)
        t = time.time(); pred = m.predict(Xt); dt = time.time() - t
        e = np.mean(pred != yt)
        print(f"    k = {k}，{metric:<10} 错误率 {e:.2%}（{int(e * len(yt))} / 10,000 张错），预测 10,000 张耗时 {dt:.1f}s")
        if metric == "euclidean":
            pred_e = pred
    print("  对照 LeCun 等 1998 的表：线性分类器 12.0%、K-NN 欧氏距离 5.0%（他们用了不同的预处理）、"
          "去斜（deskew）后 K-NN 2.4%、LeNet-5 0.95%；05 篇会加上 SVM 那一行。")
    print("  注意：KNN 训练 0 秒，预测每张要算 60,000 个距离——'训练便宜、预测贵'，和别的模型反着。")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), width_ratios=[1, 1.3])
    axes[0].plot(ks, [e * 100 for e in errs], "o-", c=C["blue"])
    axes[0].set_xscale("log"); axes[0].set_xticks(ks); axes[0].set_xticklabels(ks)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("错误率 %"); axes[0].set_title("MNIST 标签干净：k 越大越模糊，k=1 反而最好（10k 训练子集）")
    wrong = np.where(pred_e != yt)[0][:24]
    ax = axes[1]; ax.axis("off"); ax.set_title(f"全量 k=3 错分的前 24 张（真实→预测）")
    for i, j in enumerate(wrong):
        r, c = divmod(i, 8)
        ax.imshow(Xt[j].reshape(28, 28), cmap="gray_r", extent=(c, c + 0.9, -r - 0.9, -r))
        ax.text(c + 0.45, -r - 1.0, f"{yt[j]}→{pred_e[j]}", ha="center", va="top", fontsize=6.5, color=C["red"])
    ax.set_xlim(0, 8); ax.set_ylim(-3.4, 0); ax.set_aspect("equal")
    save(fig, "case-04-knn-mnist")


# ---------------- C. 决策树：泰坦尼克 ----------------
FEATS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]


def prep(df):
    X = df[FEATS].copy()
    X["sex"] = (X.sex == "female").astype(int)                     # 女 = 1
    X["age"] = X.age.fillna(X.age.median())                         # 263 个缺失年龄用中位数补
    X["fare"] = X.fare.fillna(X.fare.median())
    X["embarked"] = X.embarked.map({"S": 0, "C": 1, "Q": 2}).fillna(0).astype(int)
    return X


def exp_tree():
    print("\n=== C. 决策树做泰坦尼克生还预测 ===")
    df = titanic()
    y = df.survived.values
    print(f"  {len(df)} 名乘客，生还 {y.sum()}（{y.mean():.1%}）；特征：{FEATS}；age 缺 {df.age.isna().sum()} 个")
    X = prep(df)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=0, stratify=y)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]

    print(f"\n  {'模型':<36}{'训练准确率':>9}{'测试准确率':>9}")
    print(f"  {'基线：全部判死亡':<36}{np.mean(ytr == 0):>9.3f}{np.mean(yte == 0):>9.3f}")
    print(f"  {'一条规则：女的活、男的死':<36}{np.mean((Xtr.sex == 1) == ytr):>9.3f}{np.mean((Xte.sex == 1) == yte):>9.3f}")
    res = {}
    for d in [1, 2, 3, 4, 5, 8, None]:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        res[d] = (m.score(Xtr, ytr), m.score(Xte, yte), m.get_n_leaves())
        print(f"  {'决策树 max_depth=' + str(d):<36}{res[d][0]:>9.3f}{res[d][1]:>9.3f}   叶子 {res[d][2]}")
    cv = {d: cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), Xtr, ytr, cv=5).mean() for d in [1, 2, 3, 4, 5, 6, 8, 10, None]}
    best = max(cv, key=cv.get)
    print(f"  5 折交叉验证（只用训练集）选深度：" + "  ".join(f"{d}:{v:.3f}" for d, v in cv.items()) + f" → 选 {best}")
    print("  解读：深度不限时训练 98% 测试 76%——树把每个乘客背下来了（01 篇的过拟合）；深度 3 的树测试最好，而且能整棵画出来。")

    # 泄漏陷阱：把 boat（救生艇号）放进去
    Xl = X.copy(); Xl["boat"] = df.boat.notna().astype(int)
    m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xl.iloc[tr], ytr)
    print(f"\n  陷阱：加一列 boat（有没有救生艇记录），深度 3 的树测试准确率 {m.score(Xl.iloc[te], yte):.3f}——"
          f"根节点直接问 boat。这是泄漏：上了救生艇才有记录，它是'生还'的另一种写法，不是能在事前知道的特征（01 篇第五章）。")

    # 画深度 3 的树（自己画：sklearn 的 plot_tree 在这个宽度下字太小）
    m3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
    draw_tree(m3, ["舱等", "是女性", "年龄", "兄弟姐妹/配偶数", "父母/子女数", "票价", "登船港"], len(ytr))
    print("  根节点：是女性？→ 女性 → 舱等 ≤ 2.5？→ 头等 / 二等舱女性生还率最高；男性 → 年龄 ≤ 9.5？→ 小男孩生还率明显高于成年男性。")


def draw_tree(model, names, n_total):
    t = model.tree_
    fig, ax = plt.subplots(figsize=(7.6, 3.9)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(-3.5, 0.3)

    def rec(node, x0, x1, depth):
        x, y = (x0 + x1) / 2, -depth
        n = int(t.n_node_samples[node]); p1 = t.value[node][0][1] / t.value[node][0].sum()
        if t.children_left[node] == -1:
            col = C["blue"] if p1 >= 0.5 else C["orange"]
            ax.text(x, y, f"{'生还' if p1 >= 0.5 else '死亡'}\n生还率 {p1:.0%}\n{n} 人（{n / n_total:.0%}）", ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none", alpha=0.85), color="white")
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        cond = "是女性？" if f == "是女性" else f"{f} ≤ {thr:.3g}？"
        ax.text(x, y, f"{cond}\n{n} 人，生还率 {p1:.0%}", ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec=C["gray"]))
        for child, (a, b), lab in [(t.children_left[node], (x0, x), "否" if f == "是女性" else "是"),
                                   (t.children_right[node], (x, x1), "是" if f == "是女性" else "否")]:
            cx = (a + b) / 2
            ax.annotate("", xy=(cx, y - 1 + 0.32), xytext=(x, y - 0.3), arrowprops=dict(arrowstyle="->", color=C["gray"], lw=0.8))
            ax.text((x + cx) / 2 + (-0.03 if cx < x else 0.03), y - 0.45, lab, fontsize=7, color=C["gray"], ha="center")
            rec(child, a, b, depth + 1)

    rec(0, 0, 1, 0)
    ax.set_title("深度 3 的决策树：先问性别，男性再问年龄，女性再问舱等——「妇女与儿童优先」被数据学出来了", fontsize=9)
    save(fig, "case-04-titanic-tree")


EXPS = {"nb": exp_nb, "knn": exp_knn, "tree": exp_tree}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
