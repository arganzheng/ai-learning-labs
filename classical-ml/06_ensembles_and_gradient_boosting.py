"""集成：随机森林与梯度提升（经典 ML 06）：bagging 降方差、随机森林、梯度提升的逐步残差拟合（手写）、学习率与棵数、
特征重要性、表格数据上 GBDT vs 神经网络、以及数据质量分类器的算力账。
https://arganzheng.life/ensembles-random-forest-and-gradient-boosting.html

    python 06_ensembles_and_gradient_boosting.py          # 全部：bagging forest boost_steps boost_hand lr importance tabular budget variance logit early forest_cost
    python 06_ensembles_and_gradient_boosting.py boost_steps

图输出到 out/06-*.svg。
"""
import pickle
import sys
import time

import numpy as np
from sklearn.datasets import (
    fetch_20newsgroups,
    load_breast_cancer,
    make_classification,
    make_moons,
)
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
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
    except Exception as e:  # noqa: BLE001
        print(f"  （20newsgroups 需要联网下载，跳过：{type(e).__name__}）")
    print("  给 15T token 打分的账（L0 第一篇：一个 token 前向 ≈ 2N FLOP）：")
    D = 15e12
    train_8b = 6 * 8e9 * D
    for name, N in [("8B LLM 逐段打分", 8e9), ("1B 小 LLM", 1e9), ("BERT 级 1 亿参数", 1e8), ("fastText / 线性模型 ~1M", 1e6)]:
        score = 2 * N * D
        print(f"    {name:<22} {score:.1e} FLOP = 训练 8B 模型 ({train_8b:.1e}) 的 {score/train_8b*100:6.3f}%")
    print("  → 两级做法：大模型标几十万段，训小分类器，小分类器过全部。不是效果更好，是只有它跑得起")
    print()


# ---------------- 9. 方差公式：ρσ² + (1 − ρ)σ²/B——相关 ρ 由 max_features 控制，B 再大也压不掉 ρσ² ----------------
def exp_variance():
    print("=== 9. 平均 B 棵相关系数 ρ 的树：方差 = ρσ² + (1 − ρ)σ²/B；随机特征选择降的是 ρ ===")
    S, n, n_te, pairs = 50, 1000, 300, 5
    Xall, yall = make_classification(n_samples=S * n + n_te, n_features=20, n_informative=8, n_redundant=4, flip_y=0.05, class_sep=0.8, random_state=0)
    Xte, yte = Xall[:n_te], yall[:n_te]
    sets = [(Xall[n_te + s * n: n_te + (s + 1) * n], yall[n_te + s * n: n_te + (s + 1) * n]) for s in range(S)]

    def one_tree(X, y, m, seed):
        return RandomForestClassifier(n_estimators=1, max_features=m, bootstrap=True, random_state=seed).fit(X, y).predict_proba(Xte)[:, 1]

    def forest_var(m, B):
        F = np.array([RandomForestClassifier(n_estimators=B, max_features=m, random_state=s, n_jobs=-1).fit(X, y).predict_proba(Xte)[:, 1] for s, (X, y) in enumerate(sets)])
        return float(F.var(0).mean())

    print(f"  从同一个分布抽 {S} 份训练集（各 {n} 个样本），固定 {n_te} 个测试点；每份上训 {pairs} 对只差随机种子的树 (h, h′) 和一片 B 棵的森林")
    print(f"  {'m':>3} {'单棵树方差 σ²':>12} {'两棵树相关 ρ':>12} {'公式 B=100':>10} {'实测 B=100':>10} {'公式 B=∞ (ρσ²)':>14} {'实测 B=1000':>10} {'森林测试准确率':>12}")
    rows = {}
    for m in (20, 4, 1):
        var, cov = np.zeros(n_te), np.zeros(n_te)
        for k in range(pairs):
            H1 = np.array([one_tree(X, y, m, 1000 * k + 2 * s) for s, (X, y) in enumerate(sets)])
            H2 = np.array([one_tree(X, y, m, 1000 * k + 2 * s + 1) for s, (X, y) in enumerate(sets)])
            var += 0.5 * (H1.var(0) + H2.var(0)) / pairs
            cov += ((H1 - H1.mean(0)) * (H2 - H2.mean(0))).mean(0) / pairs
        sigma2, rho = float(var.mean()), float(cov.mean() / var.mean())
        acc = np.mean([(RandomForestClassifier(n_estimators=100, max_features=m, random_state=0, n_jobs=-1).fit(X, y).score(Xte, yte)) for X, y in sets[:5]])
        pred = rho * sigma2 + (1 - rho) * sigma2 / 100
        rows[m] = (sigma2, rho)
        print(f"  {m:>3} {sigma2:>12.4f} {rho:>12.3f} {pred:>10.4f} {forest_var(m, 100):>10.4f} {rho * sigma2:>14.4f} {forest_var(m, 1000):>10.4f} {acc:>12.3f}")
    sigma2, rho = rows[4]
    print(f"  m = 4 时手算：ρσ² + (1 − ρ)σ²/B = {rho:.3f}×{sigma2:.4f} + {1 - rho:.3f}×{sigma2:.4f}/100 = {rho * sigma2:.4f} + {(1 - rho) * sigma2 / 100:.5f} = {rho * sigma2 + (1 - rho) * sigma2 / 100:.4f}")
    print(f"  {'B':>5} {'公式 (m=4)':>10} {'实测 (m=4)':>10} {'训练 50 片森林':>12}")
    Bs = [1, 10, 100, 1000]
    meas, form = [], []
    for B in Bs:
        t0 = time.time()
        meas.append(forest_var(4, B))
        form.append(rho * sigma2 + (1 - rho) * sigma2 / B)
        print(f"  {B:>5} {form[-1]:>10.4f} {meas[-1]:>10.4f} {time.time() - t0:>11.1f}s")
    print("  条件读法：方差公式要求各棵树同分布（bootstrap + 同一算法保证）、ρ 是任意两棵树的相关；B → ∞ 只能压掉 (1 − ρ)σ²/B，剩下的 ρσ² 只能靠降 ρ——这就是每个节点只看 m 个特征的理由；m 太小单棵树偏差上升，准确率反而掉")
    print(f"  ρ 是用 {S} 份数据 × {pairs} 对树估的，小到 0.0x 时本身带有 ±0.01 量级的抽样误差，所以「公式」列在 m = 4、1 时只能看量级，不能看第四位小数")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    for m, col in ((20, C["gray"]), (4, C["red"]), (1, C["blue"])):
        s2, r_ = rows[m]
        bb = np.logspace(0, 3.5, 50)
        ax.plot(bb, r_ * s2 + (1 - r_) * s2 / bb, color=col, lw=1.4, label=f"m = {m}：ρ = {r_:.2f}，ρσ² = {r_ * s2:.4f}")
    ax.plot(Bs, meas, "o", color=C["red"], label="实测（m = 4）")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("树的棵数 B（对数轴）"); ax.set_ylabel("森林预测概率的方差（对数轴）")
    ax.legend(frameon=False, fontsize=6.5); ax.set_title("方差 = ρσ² + (1 − ρ)σ²/B：B 大了只剩 ρσ²", fontsize=8)
    ax = axes[1]
    ms = [20, 4, 1]
    ax.bar([f"m = {m}" for m in ms], [rows[m][1] for m in ms], color=[C["gray"], C["red"], C["blue"]])
    ax.set_ylabel("两棵树预测的相关 ρ"); ax.set_title("每个节点只看 m 个特征：m 越小 ρ 越低", fontsize=8)
    save(fig, "06-variance-formula")
    print()


# ---------------- 10. 二分类 boosting：树拟合 y − p，累加的是 logit，最后才过 sigmoid ----------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_gbdt_logloss(X, y, n_trees=100, lr=0.1, max_depth=2):
    """y ∈ {0, 1}。loss = −[y log p + (1 − y) log(1 − p)]，p = sigmoid(F)；树拟合负梯度 y − p，叶子值用一步牛顿法。"""
    p0 = y.mean()
    # !ref init-logit
    f0 = np.log(p0 / (1 - p0))
    F = np.full(len(y), f0)
    trees = []
    for _ in range(n_trees):
        # !ref prob-from-logit
        p = sigmoid(F)
        # !ref neg-grad
        g = y - p
        t = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, g)
        leaf = t.apply(X)
        for j in np.unique(leaf):
            mask = leaf == j
            # !ref newton-leaf
            t.tree_.value[j, 0, 0] = g[mask].sum() / (p[mask] * (1 - p[mask])).sum()
        # !ref add-logit
        F += lr * t.predict(X)
        trees.append(t)
    return f0, trees


def predict_logit(model, X, lr=0.1, n_trees=None):
    f0, trees = model
    F = np.full(len(X), f0)
    for t in trees[:n_trees]:
        F += lr * t.predict(X)
    return F


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def exp_logit():
    print("=== 10. 二分类 boosting：负梯度 y − p，叶子值 Σ(y − p)/Σ p(1 − p)，累加 logit，最后 sigmoid ===")
    X4 = np.array([[1.0], [2.0], [3.0], [4.0]])
    y4 = np.array([0, 0, 1, 1])
    model = fit_gbdt_logloss(X4, y4, n_trees=2, lr=1.0, max_depth=1)
    print(f"  4 个点 y = {y4.tolist()}，学习率 1：F₀ = log(p̄/(1 − p̄)) = log(0.5/0.5) = {model[0]:.3f}")
    for k in (1, 2):
        F = predict_logit(model, X4, lr=1.0, n_trees=k)
        p = sigmoid(F)
        print(f"  第 {k} 棵树后：F = {np.round(F, 3).tolist()}，p = sigmoid(F) = {np.round(p, 3).tolist()}，log-loss {log_loss(y4, p):.4f}")
    print("  手算第 1 棵：p = 0.5，g = y − p = [−0.5, −0.5, 0.5, 0.5]，切在中间；左叶 Σg/Σp(1−p) = −1/(2×0.25) = −2，右叶 +2 → F = [−2, −2, 2, 2]，p = 0.119 / 0.881")
    print("  第 2 棵：g = [−0.119, −0.119, 0.119, 0.119]，Σp(1−p) = 2×0.119×0.881 = 0.210，叶子值 ∓0.238/0.210 = ∓1.135 → F = ∓3.135，p = 0.042 / 0.958")
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    n_trees, lr = 100, 0.1
    model = fit_gbdt_logloss(Xtr, ytr, n_trees=n_trees, lr=lr, max_depth=2)
    sk = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=lr, max_depth=2, random_state=0).fit(Xtr, ytr)
    F_hand = predict_logit(model, Xte, lr=lr)
    F_sk = sk.decision_function(Xte)
    print(f"  乳腺癌数据、{n_trees} 棵深度 2、lr {lr}：手写与 sklearn GradientBoostingClassifier 的 logit 最大差 {np.abs(F_hand - F_sk).max():.1e}，概率最大差 {np.abs(sigmoid(F_hand) - sk.predict_proba(Xte)[:, 1]).max():.1e}")
    print(f"  测试 log-loss：手写 {log_loss(yte, sigmoid(F_hand)):.4f}，sklearn {log_loss(yte, sk.predict_proba(Xte)[:, 1]):.4f}；准确率 {((F_hand > 0) == yte).mean():.3f} / {sk.score(Xte, yte):.3f}")
    # 反例：把每棵树的输出直接加到概率上
    p_naive = np.full(len(ytr), ytr.mean())
    p_naive_te = np.full(len(yte), ytr.mean())
    curve_naive, curve_logit = [], []
    for k in range(n_trees):
        g = ytr - p_naive
        t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(Xtr, g)
        p_naive = p_naive + lr * t.predict(Xtr)
        p_naive_te = p_naive_te + lr * t.predict(Xte)
        curve_naive.append(log_loss(yte, p_naive_te))
        curve_logit.append(log_loss(yte, sigmoid(predict_logit(model, Xte, lr=lr, n_trees=k + 1))))
    out = ((p_naive_te < 0) | (p_naive_te > 1)).mean()
    print(f"  反例——直接累加概率 p ← p + η·tree(y − p)：{n_trees} 棵后 {out:.0%} 的测试点「概率」跑出 [0, 1]（最小 {p_naive_te.min():.2f}，最大 {p_naive_te.max():.2f}）；裁剪到 [0, 1] 后 log-loss {curve_naive[-1]:.4f} vs 累加 logit 的 {curve_logit[-1]:.4f}")
    print("  在这份干净的数据上两者的损失差不多——直接累加概率的问题不是「准确率低」，而是输出不再是概率：一半样本越界、越界的点梯度 y − p 仍非零会继续推、也没有 p(1 − p) 这个二阶量可用")
    print("  累加 logit：F 可以是任何实数，sigmoid 最后才把它压回 (0, 1)；每棵树的叶子值由负梯度与二阶导 p(1 − p) 一起决定，这就是 XGBoost「二阶近似」的最简形式")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    xs = np.linspace(0.5, 4.5, 200)[:, None]
    model_4 = fit_gbdt_logloss(X4, y4, n_trees=2, lr=1.0, max_depth=1)
    for k, col in ((1, C["orange"]), (2, C["red"])):
        ax.plot(xs[:, 0], sigmoid(predict_logit(model_4, xs, lr=1.0, n_trees=k)), color=col, lw=1.4, label=f"{k} 棵树后的 p = sigmoid(F)")
        ax.plot(xs[:, 0], predict_logit(model_4, xs, lr=1.0, n_trees=k) / 4, color=col, lw=0.8, ls="--", label=f"{k} 棵树后的 F / 4")
    ax.scatter(X4[:, 0], y4, s=30, color=C["blue"], zorder=3, label="4 个样本")
    ax.axhline(0, color=C["gray"], lw=0.5); ax.set_xlabel("x"); ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("树累加的是 F（虚线），概率是最后一步 sigmoid(F)", fontsize=8)
    ax = axes[1]
    ax.plot(range(1, n_trees + 1), curve_logit, color=C["red"], lw=1.4, label="累加 logit，最后 sigmoid")
    ax.plot(range(1, n_trees + 1), curve_naive, color=C["gray"], lw=1.4, label="直接累加概率（裁剪到 [0,1]）")
    ax.set_xlabel("树的棵数"); ax.set_ylabel("测试 log-loss"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("乳腺癌数据：两种累加方式的测试 log-loss", fontsize=8)
    save(fig, "06-logit-boosting")
    print()


# ---------------- 11. 早停与数据划分：用哪份数据选轮数；OOB 对群组 / 时间泄漏无能为力 ----------------
def exp_early():
    print("=== 11. 早停选轮数要用独立的验证集；OOB 是随机留出，处理不了群组与时间泄漏 ===")
    Xtr, Xte, ytr, yte = tabular()
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=1, stratify=ytr)
    gb = GradientBoostingClassifier(n_estimators=600, learning_rate=0.1, max_depth=3, random_state=0).fit(Xa, ya)
    val = np.array([np.mean(p == yv) for p in gb.staged_predict(Xv)])
    te = np.array([np.mean(p == yte) for p in gb.staged_predict(Xte)])
    k_val, k_te = int(val.argmax()), int(te.argmax())
    print(f"  训练集再切 20% 做验证：验证集最好在第 {k_val + 1} 棵（验证 {val[k_val]:.3f}），此时测试 {te[k_val]:.3f}；600 棵全用时测试 {te[-1]:.3f}")
    print(f"  若直接在测试集上挑轮数：第 {k_te + 1} 棵、测试 {te[k_te]:.3f}——比用验证集挑高 {te[k_te] - te[k_val]:.3f}，这部分是「在测试集上调参」的乐观偏差，不是泛化")
    es = GradientBoostingClassifier(n_estimators=600, learning_rate=0.1, max_depth=3, random_state=0, n_iter_no_change=20, validation_fraction=0.2).fit(Xtr, ytr)
    print(f"  sklearn 内置早停（n_iter_no_change=20, validation_fraction=0.2）：停在第 {es.n_estimators_} 棵，测试 {es.score(Xte, yte):.3f}；HistGradientBoosting 的 early_stopping='auto' 只在 n > 10000 时自动开启")
    # 群组泄漏：每个「群组」是同一来源的 8 个近似重复样本
    r = np.random.default_rng(0)
    G, per = 400, 8
    centers, labels = make_classification(n_samples=G, n_features=20, n_informative=8, n_redundant=4, flip_y=0.05, class_sep=0.8, random_state=3)
    Xg = np.repeat(centers, per, axis=0) + r.normal(0, 0.15, (G * per, 20))
    yg = np.repeat(labels, per)
    groups = np.repeat(np.arange(G), per)
    gtr = groups < 300
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1, oob_score=True).fit(Xg[gtr], yg[gtr])
    random_cv = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), Xg[gtr], yg[gtr], cv=KFold(5, shuffle=True, random_state=0)).mean()
    group_cv = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), Xg[gtr], yg[gtr], cv=GroupKFold(5), groups=groups[gtr]).mean()
    fresh = rf.score(Xg[~gtr], yg[~gtr])
    print(f"  群组数据（{G} 个来源 × {per} 条近似重复，前 300 个来源训练）：OOB {rf.oob_score_:.3f}，随机 5 折 {random_cv:.3f}，按来源分组 5 折 {group_cv:.3f}，100 个全新来源 {fresh:.3f}")
    # 时间漂移：决策边界随时间旋转
    T = 4000
    t = np.linspace(0, 1, T)
    Xt_ = r.normal(size=(T, 2))
    theta = np.pi / 2 * t
    yt_ = ((np.cos(theta) * Xt_[:, 0] + np.sin(theta) * Xt_[:, 1] + r.normal(0, 0.3, T)) > 0).astype(int)
    Xtime = np.c_[Xt_, t]
    first = np.arange(T) < int(0.7 * T)
    rf_t = RandomForestClassifier(300, random_state=0, n_jobs=-1, oob_score=True).fit(Xtime[first], yt_[first])
    shuffled = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), Xtime[first], yt_[first], cv=KFold(5, shuffle=True, random_state=0)).mean()
    forward = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), Xtime[first], yt_[first], cv=TimeSeriesSplit(5)).mean()
    future = rf_t.score(Xtime[~first], yt_[~first])
    print(f"  时间漂移数据（边界随时间转 90°，前 70% 训练）：OOB {rf_t.oob_score_:.3f}，打乱 5 折 {shuffled:.3f}，按时间向前 5 折 {forward:.3f}，之后 30% {future:.3f}")
    print("  OOB 只是「每棵树没抽到的样本」——抽样是逐条随机的，同来源的另一条、同一时刻前后的样本仍在袋内；它对随机划分诚实，对群组与时间泄漏和随机 K 折一样乐观")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    names = ["OOB", "随机 5 折", "按群组 / 时间划分", "真实留出"]
    g_vals = [rf.oob_score_, random_cv, group_cv, fresh]
    t_vals = [rf_t.oob_score_, shuffled, forward, future]
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, g_vals, w, color=C["blue"], label="群组数据（近似重复）")
    ax.bar(x + w / 2, t_vals, w, color=C["orange"], label="时间漂移数据")
    for i in range(4):
        ax.text(i - w / 2, g_vals[i] + 0.004, f"{g_vals[i]:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, t_vals[i] + 0.004, f"{t_vals[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.5, 1.02); ax.set_ylabel("准确率"); ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("OOB 与随机 K 折一样乐观；按群组 / 时间划分才接近真实留出", fontsize=8.5)
    save(fig, "06-oob-leakage")
    print()


# ---------------- 12. 「森林不用调参」「树越多越好」的条件与代价 ----------------
def exp_forest_cost():
    print("=== 12. 树越多：准确率饱和、训练 / 预测 / 内存线性增长；标签噪声大或信号稀疏时默认参数不是最好 ===")
    Xtr, Xte, ytr, yte = tabular()
    print(f"  {'棵数':>5} {'测试准确率':>8} {'训练':>7} {'预测 1500 点':>10} {'节点总数':>9} {'pickle 大小':>9}")
    Bs = [10, 50, 100, 300, 1000, 3000]
    accs, fits, sizes = [], [], []
    for B in Bs:
        rf = RandomForestClassifier(B, random_state=0, n_jobs=-1)
        t0 = time.time()
        rf.fit(Xtr, ytr)
        fits.append(time.time() - t0)
        t0 = time.time()
        acc = rf.score(Xte, yte)
        t_pred = time.time() - t0
        nodes = sum(t.tree_.node_count for t in rf.estimators_)
        size = len(pickle.dumps(rf)) / 1e6
        accs.append(acc); sizes.append(size)
        print(f"  {B:>5} {acc:>8.3f} {fits[-1]:>6.2f}s {t_pred:>9.2f}s {nodes:>9,} {size:>7.1f}MB")
    print(f"  {Bs[2]} → {Bs[-1]} 棵：准确率 {accs[-1] - accs[2]:+.3f}，训练时间 ×{fits[-1] / fits[2]:.0f}、模型 ×{sizes[-1] / sizes[2]:.0f}——「越多越好」只在方差项上成立（第 9 节的 1/B 项），代价是线性的")
    Xn, yn = make_classification(n_samples=3000, n_features=20, n_informative=8, n_redundant=4, flip_y=0.3, class_sep=0.8, random_state=0)
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(Xn, yn, test_size=0.3, random_state=0)
    print("  标签噪声 30%（flip_y=0.3）：", end="")
    print("  ".join(f"min_samples_leaf={leaf}: {RandomForestClassifier(300, min_samples_leaf=leaf, random_state=0, n_jobs=-1).fit(Xn_tr, yn_tr).score(Xn_te, yn_te):.3f}" for leaf in (1, 5, 20, 50)))
    Xs, ys = make_classification(n_samples=2000, n_features=300, n_informative=5, n_redundant=0, class_sep=1.0, random_state=0)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3, random_state=0)
    print("  300 个特征只有 5 个有信息：", end="")
    print("  ".join(f"max_features={mf}: {RandomForestClassifier(300, max_features=mf, random_state=0, n_jobs=-1).fit(Xs_tr, ys_tr).score(Xs_te, ys_te):.3f}" for mf in ("sqrt", 60, 150, None)))
    print("  默认值（不限深度、叶子 1 个样本、m = √d）在信号强、噪声小、特征数适中时够用；噪声大要加 min_samples_leaf，有信息特征稀疏时 m = √d 太小，每次分裂常抽不到有用特征")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    axes[0].plot(Bs, accs, "o-", color=C["red"]); axes[0].set_xscale("log"); axes[0].set_xlabel("棵数 B（对数轴）"); axes[0].set_ylabel("测试准确率")
    axes[0].set_title("准确率：100 棵后基本饱和", fontsize=8)
    axes[1].loglog(Bs, fits, "o-", color=C["blue"], label="训练时间（秒）"); axes[1].loglog(Bs, sizes, "s-", color=C["orange"], label="模型大小（MB）")
    axes[1].set_xlabel("棵数 B（对数轴）"); axes[1].legend(frameon=False, fontsize=7); axes[1].set_title("代价：随 B 线性增长", fontsize=8)
    save(fig, "06-forest-cost")
    print()


EXPS = {"bagging": exp_bagging, "forest": exp_forest, "boost_steps": exp_boost_steps, "boost_hand": exp_boost_hand,
        "lr": exp_lr, "importance": exp_importance, "tabular": exp_tabular, "budget": exp_budget,
        "variance": exp_variance, "logit": exp_logit, "early": exp_early, "forest_cost": exp_forest_cost}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
