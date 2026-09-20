"""集成：随机森林与梯度提升（经典 ML 06）：bagging 降方差、随机森林、梯度提升的逐步残差拟合（手写）、学习率与棵数、
特征重要性、表格数据上 GBDT vs 神经网络、以及数据质量分类器的算力账。
https://arganzheng.life/ensembles-random-forest-and-gradient-boosting.html

    python 06_ensembles_and_gradient_boosting.py          # 全部：bagging forest boost_steps boost_hand lr importance tabular budget
    python 06_ensembles_and_gradient_boosting.py boost_steps

图输出到 out/06-*.svg。
"""
import sys
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer, make_classification, make_moons
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from _plot import C, plt, save


def tabular():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, n_redundant=4, flip_y=0.05, class_sep=0.8, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def moons():
    X, y = make_moons(n_samples=400, noise=0.22, random_state=0)
    return train_test_split(X, y, test_size=0.5, random_state=0)


def plot_boundary(ax, predict, X, y, title, res=70):
    xx, yy = np.meshgrid(np.linspace(-2, 3, res), np.linspace(-1.5, 2, res))
    Z = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["blue"], C["red"]], alpha=0.18)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=7, color=C["blue"]); ax.scatter(X[y == 1, 0], X[y == 1, 1], s=7, color=C["red"])
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


# ---------------- 手写：梯度提升回归（平方损失 → 残差） ----------------
def fit_gbdt(X, y, n_trees=100, lr=0.1, max_depth=2):
    f0 = y.mean()                                                  # ① 初始预测：常数（均值）
    pred = np.full(len(y), f0)
    trees = []
    for _ in range(n_trees):
        resid = y - pred                                           # ② 残差 = 平方损失的负梯度 −∂L/∂f
        t = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, resid)   # ③ 用一棵浅树拟合残差
        pred += lr * t.predict(X)                                  # ④ 加进去，乘学习率
        trees.append(t)
    return f0, trees


def predict_gbdt(model, X, lr=0.1, n_trees=None):
    f0, trees = model
    pred = np.full(len(X), f0)
    for t in trees[:n_trees]:
        pred += lr * t.predict(X)
    return pred


# ---------------- 1. bagging：很多棵树平均，方差降下来 ----------------
def exp_bagging():
    print("=== 1. bagging：一棵树 vs 很多棵（各在重采样的数据上训）取平均 ===")
    Xtr, Xte, ytr, yte = tabular()
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    print(f"  一棵不限深度的树：训练 {tree.score(Xtr, ytr):.3f} / 测试 {tree.score(Xte, yte):.3f}")
    ns = [1, 2, 5, 10, 20, 50, 100, 200]
    accs = []
    for n in ns:
        bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        accs.append(bag.score(Xte, yte))
    print("  棵数:  " + " ".join(f"{n:>6}" for n in ns))
    print("  测试:  " + " ".join(f"{a:6.3f}" for a in accs))
    # 方差：换 20 批训练数据，看单棵树 vs 50 棵 bagging 的预测在测试集上的方差
    preds_single, preds_bag = [], []
    for s in range(20):
        idx = np.random.default_rng(s).choice(len(Xtr), len(Xtr), replace=True)
        preds_single.append(DecisionTreeClassifier(random_state=s).fit(Xtr[idx], ytr[idx]).predict_proba(Xte)[:, 1])
        preds_bag.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=s, n_jobs=-1).fit(Xtr[idx], ytr[idx]).predict_proba(Xte)[:, 1])
    print(f"  换 20 批训练数据：单棵树预测概率的方差 {np.mean(np.var(preds_single, 0)):.3f}；50 棵 bagging 的 {np.mean(np.var(preds_bag, 0)):.3f}")
    mXtr, mXte, mytr, myte = moons()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    t = DecisionTreeClassifier(random_state=0).fit(mXtr, mytr)
    plot_boundary(axes[0], t.predict, mXtr, mytr, f"一棵树：训练 {t.score(mXtr, mytr):.2f} / 测试 {t.score(mXte, myte):.2f}")
    b = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0).fit(mXtr, mytr)
    plot_boundary(axes[1], b.predict, mXtr, mytr, f"100 棵 bagging：{b.score(mXtr, mytr):.2f} / {b.score(mXte, myte):.2f}")
    axes[2].plot(ns, accs, "o-", color=C["red"]); axes[2].axhline(tree.score(Xte, yte), color=C["gray"], ls="--", lw=0.8, label="一棵树")
    axes[2].set_xscale("log"); axes[2].set_xlabel("树的棵数（对数轴）"); axes[2].set_ylabel("测试准确率"); axes[2].set_title("表格数据：越多越好，50 棵后饱和", fontsize=8); axes[2].legend(frameon=False, fontsize=7)
    save(fig, "06-bagging")
    print("  月牙上的边界仍由横竖线段拼成，但单棵树圈出的孤岛被抹掉；表格数据上训练准确率仍是 100%（每棵树都背下了自己的数据），但测试从 0.83 升到 0.90——方差被平均掉了")
    print()


# ---------------- 2. 随机森林：再加一层随机——每次分裂只看部分特征 ----------------
def exp_forest():
    print("=== 2. 随机森林 = bagging + 每个节点只在随机的特征子集里选切分 ===")
    Xtr, Xte, ytr, yte = tabular()
    print(f"  {'max_features':>13} {'测试准确率':>10} {'树之间预测的平均相关':>20}")
    for mf in (20, 10, 4, 2, 1):
        rf = RandomForestClassifier(200, max_features=mf, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        P = np.array([t.predict_proba(Xte)[:, 1] for t in rf.estimators_[:50]])   # [50 棵, n_test]
        corr = np.corrcoef(P)
        mean_corr = corr[np.triu_indices(50, 1)].mean()
        print(f"  {mf:>13} {rf.score(Xte, yte):>10.3f} {mean_corr:>20.3f}")
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1, oob_score=True).fit(Xtr, ytr)
    print(f"  300 棵、默认 max_features=√20≈4：测试 {rf.score(Xte, yte):.3f}；袋外（OOB）估计 {rf.oob_score_:.3f}——不用验证集就有一个泛化估计")
    print("  只看部分特征让树彼此更不像（相关更低），平均后方差降得更多——这是随机森林比纯 bagging 更好的原因")
    print()


# ---------------- 3. 梯度提升：一步一步拟合残差（一维回归，能画出来）----------------
def exp_boost_steps():
    print("=== 3. 梯度提升的每一步：拟合残差 ===")
    r = np.random.default_rng(0)
    x = np.sort(r.uniform(0, 6, 80)); y = np.sin(x) + 0.3 * x + r.normal(0, 0.25, 80)
    X = x[:, None]
    xs = np.linspace(0, 6, 300)[:, None]
    lr = 1.0                                                          # 学习率 1：每棵树的修正全额加进去，最容易看
    model = fit_gbdt(X, y, n_trees=50, lr=lr, max_depth=1)            # 深度 1 的树 = 一个台阶
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.8))
    for k, ax in zip((0, 1, 2, 3), axes[0]):
        pred = predict_gbdt(model, xs, lr=lr, n_trees=k)
        pred_tr = predict_gbdt(model, X, lr=lr, n_trees=k)
        ax.scatter(x, y, s=6, color=C["blue"]); ax.plot(xs[:, 0], pred, color=C["red"], lw=1.5)
        ax.set_title(f"{k} 棵树后的预测 F{k}(x)\n训练 MSE {np.mean((pred_tr - y)**2):.3f}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        print(f"  {k} 棵树后：训练 MSE {np.mean((pred_tr - y)**2):.3f}")
    for k, ax in zip((0, 1, 2), axes[1]):
        pred_tr = predict_gbdt(model, X, lr=lr, n_trees=k)
        resid = y - pred_tr
        t = model[1][k]
        ax.scatter(x, resid, s=6, color=C["gray"]); ax.plot(xs[:, 0], t.predict(xs), color=C["orange"], lw=1.5)
        ax.axhline(0, color=C["gray"], lw=0.5)
        ax.set_title(f"残差 y − F{k}(x)，第 {k+1} 棵树拟合它", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1][3]
    pred50 = predict_gbdt(model, xs, lr=lr); pred50_tr = predict_gbdt(model, X, lr=lr)
    ax.scatter(x, y, s=6, color=C["blue"]); ax.plot(xs[:, 0], pred50, color=C["red"], lw=1.5)
    ax.set_title(f"50 棵树后的预测\n训练 MSE {np.mean((pred50_tr - y)**2):.3f}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "06-boosting-steps")
    print(f"  50 棵树后：训练 MSE {np.mean((pred50_tr - y)**2):.3f}（噪声方差 0.0625）")
    print("  每棵树只负责修正前面所有树加起来还没解释的部分（残差）；深度 1 的树是一个台阶，50 个台阶叠出一条曲线")
    print()


# ---------------- 4. 手写 GBDT vs sklearn ----------------
def exp_boost_hand():
    print("=== 4. 手写梯度提升（15 行）vs sklearn GradientBoostingRegressor ===")
    r = np.random.default_rng(1)
    X = r.uniform(-3, 3, (500, 4)); y = np.sin(X[:, 0]) + X[:, 1] ** 2 / 4 + X[:, 2] * X[:, 3] / 3 + r.normal(0, 0.2, 500)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    model = fit_gbdt(Xtr, ytr, n_trees=300, lr=0.1, max_depth=3)
    mse_hand = np.mean((predict_gbdt(model, Xte) - yte) ** 2)
    sk = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, ytr)
    mse_sk = np.mean((sk.predict(Xte) - yte) ** 2)
    print(f"  手写 GBDT（300 棵深度 3、lr 0.1）测试 MSE {mse_hand:.4f}；sklearn {mse_sk:.4f}；两者预测最大差 {np.abs(predict_gbdt(model, Xte) - sk.predict(Xte)).max():.1e}")
    print(f"  对比：一棵深度 8 的回归树 {np.mean((DecisionTreeRegressor(max_depth=8, random_state=0).fit(Xtr, ytr).predict(Xte) - yte)**2):.4f}；噪声方差 0.04")
    print("  分类版只改一处：残差换成 log-loss 的负梯度 y − p（第三篇的逻辑回归梯度），树拟合它，累加的是 logit")
    print()


# ---------------- 5. 学习率与棵数 ----------------
def exp_lr():
    print("=== 5. 学习率 × 棵数：小学习率 + 多棵树更稳；棵数太多会过拟合 ===")
    Xtr, Xte, ytr, yte = tabular()
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    for lr, col in ((1.0, C["orange"]), (0.3, C["red"]), (0.1, C["blue"]), (0.03, C["green"])):
        gb = GradientBoostingClassifier(n_estimators=600, learning_rate=lr, max_depth=3, random_state=0).fit(Xtr, ytr)
        te = [np.mean(p == yte) for p in gb.staged_predict(Xte)]
        best = int(np.argmax(te))
        print(f"  学习率 {lr:<4}: 最好测试准确率 {max(te):.3f}（第 {best+1} 棵）；600 棵时 {te[-1]:.3f}")
        ax.plot(range(1, 601), te, color=col, lw=1.2, label=f"学习率 {lr}")
    ax.set_xscale("log"); ax.set_xlabel("树的棵数（对数轴）"); ax.set_ylabel("测试准确率"); ax.set_ylim(0.8, 0.93); ax.legend(frameon=False, fontsize=7)
    save(fig, "06-learning-rate-vs-trees")
    print("  学习率大：几十棵就到顶、然后开始过拟合下滑；学习率小：慢但顶更高更平。实践：lr 0.05–0.1，棵数用验证集早停")
    print()


# ---------------- 6. 特征重要性 ----------------
def exp_importance():
    print("=== 6. 梯度提升的特征重要性（乳腺癌 30 个特征）===")
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xtr, ytr)
    print(f"  测试准确率 {gb.score(Xte, yte):.3f}")
    order = np.argsort(-gb.feature_importances_)
    for i in order[:6]:
        print(f"    {data.feature_names[i]:<26} {gb.feature_importances_[i]:.3f}")
    print(f"  前 6 个特征占重要性 {gb.feature_importances_[order[:6]].sum():.0%}")
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    top = order[:12]
    ax.barh(range(12)[::-1], gb.feature_importances_[top], color=C["blue"])
    ax.set_yticks(range(12)[::-1]); ax.set_yticklabels(data.feature_names[top], fontsize=7)
    ax.set_xlabel("特征重要性（该特征在所有分裂里贡献的不纯度下降，归一化）"); ax.set_title("30 个特征里前 12 个；前 6 个占 91%", fontsize=8.5)
    save(fig, "06-feature-importance")
    print()


# ---------------- 7. 表格数据：GBDT vs 神经网络 ----------------
def exp_tabular():
    print("=== 7. 表格数据上 GBDT vs 一个两层神经网络（同样不调参）===")
    Xtr, Xte, ytr, yte = tabular()
    for name, m in [("梯度提升 300 棵", GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=0)),
                    ("随机森林 300 棵", RandomForestClassifier(300, random_state=0, n_jobs=-1)),
                    ("MLP 两层 128", make_pipeline(StandardScaler(), MLPClassifier((128, 128), max_iter=500, random_state=0))),
                    ("逻辑回归", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))]:
        t0 = time.time(); m.fit(Xtr, ytr)
        print(f"  {name:<14} 测试 {m.score(Xte, yte):.3f}  训练 {time.time()-t0:.1f}s")
    print("  表格数据上树的集成默认最强：不用标准化、对无关特征鲁棒、小数据不易过拟合；神经网络要调很多才追平")
    print()


# ---------------- 8. 文本分类器 + 给 15T token 打分的算力账 ----------------
def exp_budget():
    print("=== 8. 词袋 + 线性分类器（fastText 的形态）与'给 15T token 打分'的算力账 ===")
    try:
        cats = ["sci.space", "rec.sport.hockey", "talk.politics.misc", "comp.graphics"]
        train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
        test = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        Xtr, Xte = vec.fit_transform(train.data), vec.transform(test.data)
        for name, m in [("MultinomialNB", MultinomialNB()), ("逻辑回归", LogisticRegression(max_iter=3000, C=5))]:
            t0 = time.time(); m.fit(Xtr, train.target)
            print(f"  20newsgroups 4 类 {Xtr.shape[0]} 篇 → {Xtr.shape[1]} 个 1/2-gram 特征；{name:<14} 测试准确率 {m.score(Xte, test.target):.3f}  训练 {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  （20newsgroups 需要联网下载，跳过：{type(e).__name__}）")
    print("  给 15T token 打分的账（L0 第一篇：一个 token 前向 ≈ 2N FLOP）：")
    D = 15e12
    train_8b = 6 * 8e9 * D
    for name, N in [("8B LLM 逐段打分", 8e9), ("1B 小 LLM", 1e9), ("BERT 级 1 亿参数", 1e8), ("fastText / 线性模型 ~1M", 1e6)]:
        score = 2 * N * D
        print(f"    {name:<22} {score:.1e} FLOP = 训练 8B 模型 ({train_8b:.1e}) 的 {score/train_8b*100:6.3f}%")
    print("  → 两级做法：大模型标几十万段，训小分类器，小分类器过全部。不是效果更好，是只有它跑得起")
    print()


EXPS = {"bagging": exp_bagging, "forest": exp_forest, "boost_steps": exp_boost_steps, "boost_hand": exp_boost_hand,
        "lr": exp_lr, "importance": exp_importance, "tabular": exp_tabular, "budget": exp_budget}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
