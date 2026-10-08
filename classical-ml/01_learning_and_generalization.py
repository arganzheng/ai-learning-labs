"""什么是学习（经典 ML 01）：划分、过拟合与欠拟合、偏差-方差、测试集泄漏——全部用一个能画出来的例子。
https://arganzheng.life/what-is-learning-splits-generalization-and-bias-variance.html

    python 01_learning_and_generalization.py          # 全部：fit learncurve split groups contamination leak biasvar
    python 01_learning_and_generalization.py fit      # 只跑一个

图输出到 out/01-*.svg。
"""
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from _plot import C, plt, save

rng = np.random.default_rng(0)


def truth(x):
    return np.sin(2 * np.pi * x)


def make_data(n, noise=0.3, seed=0):
    r = np.random.default_rng(seed)
    x = r.uniform(0, 1, n)
    return x[:, None], truth(x) + r.normal(0, noise, n)


def poly_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


def mse(model, X, y):
    return float(np.mean((model.predict(X) - y) ** 2))


# 手写版：多项式拟合就是解一个最小二乘（第二篇详述），这里只为说明 fit 在做什么
def fit_poly(x, y, degree):
    A = np.vander(x, degree + 1, increasing=True)      # ① 设计矩阵 [n, d+1]：每行 1, x, x², …, x^d
    w, *_ = np.linalg.lstsq(A, y, rcond=None)          # ② 最小二乘：让 ||A w − y||² 最小的 w
    return w


def predict_poly(w, x):
    return np.vander(x, len(w), increasing=True) @ w   # ③ 预测：同样的特征乘系数


# ---------------- 1. 欠拟合 / 过拟合：多项式次数扫描 ----------------
def exp_fit():
    print("=== 1. 模型容量 vs 训练误差 / 验证误差（30 个点拟合 sin）===")
    X, y = make_data(30, seed=1)
    Xv, yv = make_data(1000, seed=2)
    w3 = fit_poly(X[:, 0], y, 3)
    m3 = poly_model(3).fit(X, y)
    print(f"  手写 lstsq 与 sklearn 的 3 次多项式预测最大差 {np.abs(predict_poly(w3, Xv[:, 0]) - m3.predict(Xv)).max():.1e}")
    degrees = list(range(1, 26))
    tr_all, va_all = [], []
    for d in degrees:
        m = poly_model(d).fit(X, y)
        tr_all.append(mse(m, X, y))
        va_all.append(mse(m, Xv, yv))
    print(f"  {'次数':>4} {'训练 MSE':>10} {'验证 MSE':>10}   判断")
    for d in (1, 3, 5, 9, 15, 25):
        tr, va = tr_all[d - 1], va_all[d - 1]
        tag = "欠拟合（两个都高）" if d <= 1 else ("过拟合（训练低、验证高）" if va > 2.5 * tr else "合适")
        print(f"  {d:>4} {tr:>10.3f} {va:>10.3f}   {tag}")
    print("  噪声方差 0.09 是验证 MSE 的下限：再好的模型也降不到它以下（不可约误差）")

    # 图 1：三种容量的拟合曲线
    xs = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
    for ax, d, title in zip(axes, (1, 4, 15), ("次数 1：欠拟合", "次数 4：合适", "次数 15：过拟合")):
        m = poly_model(d).fit(X, y)
        ax.plot(xs, truth(xs), color=C["gray"], lw=1, ls="--", label="真实函数 sin 2πx")
        ax.scatter(X[:, 0], y, s=14, color=C["blue"], zorder=3, label="30 个训练点")
        ax.plot(xs, m.predict(xs[:, None]), color=C["red"], lw=1.6, label=f"{d} 次多项式")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(f"{title}\n训练 MSE {tr_all[d-1]:.3f} / 验证 MSE {va_all[d-1]:.3f}")
        ax.set_xlabel("x")
    axes[0].legend(loc="lower left", frameon=False)
    save(fig, "01-fit-three-degrees")

    # 图 2：训练 / 验证误差 vs 次数
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(degrees, tr_all, "o-", ms=3, color=C["blue"], label="训练 MSE")
    ax.plot(degrees, va_all, "o-", ms=3, color=C["red"], label="验证 MSE（1000 个新点）")
    ax.axhline(0.09, color=C["gray"], ls="--", lw=1, label="噪声方差 0.09（不可约误差）")
    ax.set_yscale("log")
    ax.set_xlabel("多项式次数（模型容量）")
    ax.set_ylabel("MSE（对数轴）")
    ax.axvspan(0.5, 2.5, color=C["light"], alpha=0.5)
    ax.axvspan(10.5, 25.5, color=C["light"], alpha=0.5)
    ax.set_ylim(0.03, 0.5)
    ax.text(1.5, 0.42, "欠拟合", ha="center")
    ax.text(6.5, 0.42, "合适", ha="center")
    ax.text(18, 0.42, "过拟合", ha="center")
    ax.legend(frameon=False, loc="center right")
    save(fig, "01-error-vs-degree")
    print()


# ---------------- 1b. 学习曲线：样本量 vs 误差 ----------------
def exp_learncurve():
    print("=== 1b. 学习曲线：同一个模型，数据越多泛化越好；高容量模型需要更多数据 ===")
    Xv, yv = make_data(2000, seed=2)
    ns = [15, 20, 30, 50, 100, 200, 500, 1000]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    for d, color in ((4, C["blue"]), (15, C["red"])):
        tr_c, va_c = [], []
        for n in ns:
            trs, vas = [], []
            for s in range(20):
                X, y = make_data(n, seed=1000 + s)
                m = poly_model(d).fit(X, y)
                trs.append(mse(m, X, y)); vas.append(mse(m, Xv, yv))
            tr_c.append(np.median(trs)); va_c.append(np.median(vas))
        print(f"  次数 {d:>2}: " + "  ".join(f"n={n}: 训练 {t:.3f} / 验证 {v:.3f}" for n, t, v in zip(ns, tr_c, va_c) if n in (15, 30, 100, 1000)))
        ax.plot(ns, va_c, "o-", ms=3, color=color, label=f"{d} 次多项式：验证 MSE")
        ax.plot(ns, tr_c, "o--", ms=3, color=color, alpha=0.5, label=f"{d} 次多项式：训练 MSE")
    ax.axhline(0.09, color=C["gray"], ls="--", lw=1, label="噪声方差 0.09")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.02, 3)
    ax.set_xlabel("训练样本数 n（对数轴）"); ax.set_ylabel("MSE（对数轴，20 次重复的中位数）")
    ax.legend(frameon=False, ncol=2)
    save(fig, "01-learning-curve")
    print("  数据一多，15 次多项式也不过拟合了：过拟合是「容量相对于数据太大」，不是容量本身的罪")
    print()


# ---------------- 2. 划分：验证集选超参数，测试集只看一次 ----------------
def exp_split():
    print("=== 2. 训练 / 验证 / 测试三份各干什么 ===")
    X, y = make_data(300, seed=3)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0)
    print(f"  训练 {len(Xtr)} / 验证 {len(Xva)} / 测试 {len(Xte)}")
    best = None
    for d in range(1, 16):
        m = poly_model(d).fit(Xtr, ytr)
        va = mse(m, Xva, yva)
        if best is None or va < best[1]:
            best = (d, va, m)
    assert best is not None
    d, va, m = best
    print(f"  用验证集选出次数 {d}（验证 MSE {va:.3f}）；测试集只在最后看一次：测试 MSE {mse(m, Xte, yte):.3f}")
    print()


# ---------------- 2b. 怎么切：随机切 vs 按组切（近重复样本会漏进验证集）----------------
def exp_groups():
    print("=== 2b. 随机划分 vs 按组划分：同一来源的近重复样本必须在同一侧 ===")
    # 60 个「源」点，每个源复制 5 份加微小抖动 → 300 个样本；模拟同一篇文档切出的多段 / 同一用户的多条记录
    r = np.random.default_rng(7)
    src_x = r.uniform(0, 1, 60)
    src_y = truth(src_x) + r.normal(0, 0.3, 60)             # 噪声属于「源」，复制品共享它
    X = np.repeat(src_x, 5)[:, None] + r.normal(0, 0.005, 300)[:, None]
    y = np.repeat(src_y, 5) + r.normal(0, 0.02, 300)
    groups = np.repeat(np.arange(60), 5)
    Xfresh, yfresh = make_data(2000, seed=99)
    print(f"  {'次数':>4} {'随机 5 折 CV':>12} {'按组 5 折 CV':>12} {'全新数据':>10}")
    for d in (3, 9, 15):
        m = poly_model(d)
        rand = -cross_val_score(m, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").mean()
        grp = -cross_val_score(m, X, y, cv=GroupKFold(5), groups=groups, scoring="neg_mean_squared_error").mean()
        fresh = mse(m.fit(X, y), Xfresh, yfresh)
        print(f"  {d:>4} {rand:>12.3f} {grp:>12.3f} {fresh:>10.3f}")
    print("  随机切：每个验证样本的 4 个「兄弟」都在训练集里，高次多项式背下它们，CV 分数虚高；按组切的分数才接近全新数据")
    print()


# ---------------- 2c. 污染检测：n-gram 重叠 ----------------
def ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def contamination_rate(test_text, corpus_ngrams, n=8):
    """测试题里有多大比例的 n-gram 出现在训练语料里。"""
    toks = test_text.lower().split()
    grams = ngrams(toks, n)
    return len(grams & corpus_ngrams) / max(1, len(grams))


def exp_contamination():
    print("=== 2c. 污染检测：测试题的 n-gram 在训练语料里查 ===")
    corpus = [
        "the quick brown fox jumps over the lazy dog while the farmer counts his sheep in the field",
        "janet has 16 ducks and they lay 3 eggs each per day she eats 3 for breakfast every morning and bakes muffins with 4",
        "in 1969 apollo 11 landed on the moon and neil armstrong became the first person to walk on it",
    ]
    n = 8
    corpus_grams = set().union(*(ngrams(doc.lower().split(), n) for doc in corpus))
    tests = {
        "原题（GSM8K 风格，被抓进语料）": "janet has 16 ducks and they lay 3 eggs each per day she eats 3 for breakfast every morning and bakes muffins with 4 how many does she sell",
        "改写（数字换了）": "janet has 12 ducks and they lay 2 eggs each per day she eats 1 for breakfast every morning and bakes muffins with 3 how many does she sell",
        "无关题": "a train leaves the station at 9 am traveling 60 miles per hour how far has it gone by noon",
    }
    for name, t in tests.items():
        print(f"  {name:<24} {n}-gram 重叠率 {contamination_rate(t, corpus_grams, n):.2f}")
    print("  重叠率接近 1 → 原题在语料里；换了数字的改写重叠率骤降——n-gram 检测抓得住原文、抓不住改写（GPT-4 报告用 50 字符子串，Llama 3 用 8-gram 的 token 重叠）")
    print()


# ---------------- 3. 泄漏：在测试集上选模型，分数就不算数 ----------------
def exp_leak():
    print("=== 3. 测试集泄漏：用测试集选超参数，报出来的数字偏乐观 ===")
    gaps = []
    for seed in range(200):
        X, y = make_data(40, seed=100 + seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed)
        Xfresh, yfresh = make_data(2000, seed=10_000 + seed)
        # 作弊：在测试集上挑最好的次数
        cheat = min((mse(poly_model(d).fit(Xtr, ytr), Xte, yte), d) for d in range(1, 13))
        honest = mse(poly_model(cheat[1]).fit(Xtr, ytr), Xfresh, yfresh)
        gaps.append(honest - cheat[0])
    g = np.array(gaps)
    print(f"  200 次重复：'在测试集上选出的最好分数' 比 '同一模型在全新数据上的真实分数' 平均乐观 {g.mean():.3f}（MSE），"
          f"{(g > 0).mean()*100:.0f}% 的情况下真实更差；中位数 {np.median(g):.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.hist(np.clip(g, -0.2, 1.5), bins=40, color=C["blue"], alpha=0.85)
    ax.axvline(0, color=C["gray"], lw=1)
    ax.axvline(g.mean(), color=C["red"], ls="--", lw=1.2, label=f"平均 {g.mean():.3f}")
    ax.set_xlabel("真实 MSE − 在测试集上挑出的最好 MSE（正 = 报出的分数偏乐观）")
    ax.set_ylabel("次数（共 200 次）")
    ax.legend(frameon=False)
    save(fig, "01-test-set-selection-bias")
    print("  这就是 benchmark 上反复调 prompt / 选 checkpoint 之后，分数不再可信的原因")
    print()


# ---------------- 4. 偏差-方差：多次重采样看预测的均值与抖动；集成降方差 ----------------
def exp_biasvar():
    print("=== 4. 偏差-方差分解与集成（bagging）===")
    xs = np.linspace(0.05, 0.95, 200)[:, None]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for ax, d in zip(axes, (1, 4, 15)):
        preds = np.array([poly_model(d).fit(*make_data(30, seed=s)).predict(xs) for s in range(100)])  # [100, 200]
        bias2 = np.mean((preds.mean(0) - truth(xs[:, 0])) ** 2)
        var = np.mean(preds.var(0))
        # bagging：每次用 20 个重采样模型的平均
        bag = np.array([np.mean([poly_model(d).fit(*make_data(30, seed=1000 * s + k)).predict(xs) for k in range(20)], 0) for s in range(30)])
        print(f"  次数 {d:>2}: 偏差² {bias2:.3f}  方差 {var:.3f}   → 20 个模型平均后方差 {np.mean(bag.var(0)):.3f}（偏差² {np.mean((bag.mean(0) - truth(xs[:, 0]))**2):.3f} 基本不变）")
        for p in preds[:20]:
            ax.plot(xs[::4, 0], p[::4], color=C["blue"], alpha=0.14, lw=0.8)
        ax.plot(xs[:, 0], truth(xs[:, 0]), color=C["gray"], ls="--", lw=1.2, label="真实函数")
        ax.plot(xs[:, 0], preds.mean(0), color=C["red"], lw=1.6, label="100 个模型的平均预测")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(f"次数 {d}\n偏差² {bias2:.3f}  方差 {var:.3f}")
        ax.set_xlabel("x")
    axes[0].legend(loc="lower left", frameon=False)
    save(fig, "01-bias-variance-bands")
    print("  低次：偏差大（模型太简单，系统性地错）；高次：方差大（对哪 30 个点太敏感）；平均多个模型只降方差")
    print("  LLM 里的对应：self-consistency 多次采样投票、多个 judge 取平均、model soup —— 都是在降方差")
    print()


# ---------------- 5. 从一次划分到训练集内交叉验证：选超参数只碰训练集 ----------------
def kfold_cv_mse(X, y, degree, k=5, seed=0):
    """在 (X, y) 内部做 k 折：每一折都是「在 k-1 份上 fit、在剩下 1 份上算 MSE」，k 个数取平均。"""
    scores = []
    for tr, va in KFold(k, shuffle=True, random_state=seed).split(X):     # !ref folds
        m = poly_model(degree).fit(X[tr], y[tr])                             # !ref fit_fold
        scores.append(mse(m, X[va], y[va]))                                  # !ref score_fold
    return float(np.mean(scores)), float(np.std(scores))


def exp_cv():
    print("=== 5. 从一次划分到训练集内交叉验证：选次数只碰训练集，测试集到最后只看一次 ===")
    X, y = make_data(300, seed=3)
    Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=0.2, random_state=0)     # !ref holdout
    Xtr, Xva, ytr, yva = train_test_split(Xdev, ydev, test_size=0.25, random_state=0)  # !ref single
    print(f"  开发集 {len(Xdev)}（其中一次划分：训练 {len(Xtr)} / 验证 {len(Xva)}）/ 测试 {len(Xte)}")
    print(f"  {'次数':>4} {'一次划分验证 MSE':>14} {'5 折 CV 均值':>10} {'5 折 CV 标准差':>10}")
    single, cv = {}, {}
    for d in range(1, 16):
        single[d] = mse(poly_model(d).fit(Xtr, ytr), Xva, yva)
        cv[d] = kfold_cv_mse(Xdev, ydev, d)
        if d in (1, 2, 3, 4, 5, 6, 8, 10, 15):
            print(f"  {d:>4} {single[d]:>18.3f} {cv[d][0]:>14.3f} {cv[d][1]:>14.3f}")
    d_single = min(single, key=lambda d: single[d])
    d_cv = min(cv, key=lambda d: cv[d][0])
    final = poly_model(d_cv).fit(Xdev, ydev)                                            # !ref refit
    print(f"  一次划分选出次数 {d_single}；5 折 CV 选出次数 {d_cv}（CV MSE {cv[d_cv][0]:.3f} ± {cv[d_cv][1]:.3f}）")
    print(f"  用全部 {len(Xdev)} 个开发样本重新拟合 {d_cv} 次多项式，测试集只看这一次：测试 MSE {mse(final, Xte, yte):.3f}")
    # 一次划分 vs 5 折 CV：换 30 个随机种子重做「切开发集 → 选次数」，看选出来的次数有多稳
    picks_single, picks_cv = [], []
    for seed in range(30):
        Xtr_s, Xva_s, ytr_s, yva_s = train_test_split(Xdev, ydev, test_size=0.25, random_state=seed)
        picks_single.append(min(range(1, 16), key=lambda d: mse(poly_model(d).fit(Xtr_s, ytr_s), Xva_s, yva_s)))
        picks_cv.append(min(range(1, 16), key=lambda d: kfold_cv_mse(Xdev, ydev, d, seed=seed)[0]))
    fresh = make_data(20_000, seed=999)
    true_mse = {d: mse(poly_model(d).fit(Xdev, ydev), *fresh) for d in range(1, 16)}
    print(f"  换 30 个随机种子重选：一次划分选出的次数 {sorted(set(picks_single))}，"
          f"对应全新数据 MSE {min(true_mse[d] for d in picks_single):.3f}–{max(true_mse[d] for d in picks_single):.3f}")
    print(f"                       5 折 CV 选出的次数 {sorted(set(picks_cv))}，"
          f"对应全新数据 MSE {min(true_mse[d] for d in picks_cv):.3f}–{max(true_mse[d] for d in picks_cv):.3f}")
    print(f"  全新数据上真正最好的次数是 {min(true_mse, key=lambda d: true_mse[d])}（MSE {min(true_mse.values()):.3f}）")
    print("  测试集从头到尾只参与了最后一行的那一次评估；再拿它选次数，它就变成了第二个验证集（见第 3 节）")
    print()


# ---------------- 6. 完整流程：填补 / 标准化 / 特征选择各在哪份数据上 fit；泄漏对照 ----------------
def make_wide_data(n, d=1000, noise=1.0, nan_rate=0.1, seed=0):
    """n 个样本、d 个特征，只有前两个特征与 y 有关；随机挖掉 nan_rate 的格子模拟缺失值。"""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    y = X[:, 0] + X[:, 1] + r.normal(0, noise, n)
    X[r.random(X.shape) < nan_rate] = np.nan
    return X, y


def preprocess_fit(Xtr, ytr, k=20):
    """只看训练数据，把三步预处理各自要记住的量算出来。"""
    fill = np.nanmean(Xtr, axis=0)                                           # !ref fill
    Xf = np.where(np.isnan(Xtr), fill, Xtr)
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0)                                 # !ref scale
    Z = (Xf - mu) / sd
    corr = np.abs(Z.T @ (ytr - ytr.mean())) / len(ytr)                      # !ref corr
    keep = np.sort(np.argsort(corr)[-k:])                                    # !ref select
    w = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z[:, keep]], ytr, rcond=None)[0]   # !ref fitlr
    return {"fill": fill, "mu": mu, "sd": sd, "keep": keep, "w": w}


def preprocess_apply(state, X):
    """用训练时记住的量变换任何一份数据：这里没有任何从 X 本身算出来的统计量。"""
    Xf = np.where(np.isnan(X), state["fill"], X)                             # !ref apply_fill
    Z = (Xf - state["mu"]) / state["sd"]                                     # !ref apply_scale
    return np.c_[np.ones(len(Z)), Z[:, state["keep"]]] @ state["w"]           # !ref apply_rest


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def cv_r2(X, y, k, leak=None, seed=0):
    """5 折 CV 的 R²。leak 指定哪一步在「全部数据」上 fit（None = 每一步都只看训练折）。"""
    scores = []
    full = preprocess_fit(np.where(np.isnan(X), np.nanmean(X, axis=0), X), y, k)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        st = preprocess_fit(X[tr], y[tr], k)
        if leak == "fill":
            Xtr = np.where(np.isnan(X[tr]), np.nanmean(X, axis=0), X[tr])    # 用全量均值填训练折
            st = preprocess_fit(Xtr, y[tr], k)
            st["fill"] = np.nanmean(X, axis=0)
        elif leak == "scale":
            Xall = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
            st["mu"], st["sd"] = Xall.mean(axis=0), Xall.std(axis=0)
            Z = (np.where(np.isnan(X[tr]), st["fill"], X[tr]) - st["mu"]) / st["sd"]
            st["w"] = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z[:, st["keep"]]], y[tr], rcond=None)[0]
        elif leak == "select":
            st["keep"] = full["keep"]                                         # 用全量数据选出来的列
            Z = (np.where(np.isnan(X[tr]), st["fill"], X[tr]) - st["mu"]) / st["sd"]
            st["w"] = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z[:, st["keep"]]], y[tr], rcond=None)[0]
        scores.append(r2(y[va], preprocess_apply(st, X[va])))
    return float(np.mean(scores))


def exp_pipeline():
    print("=== 6. 完整流程：填补 / 标准化 / 特征选择各在哪份数据上 fit —— 有泄漏 / 无泄漏对照 ===")
    X, y = make_wide_data(120, d=1000, seed=0)
    Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=20, random_state=0)
    Xfresh, yfresh = make_wide_data(20_000, d=1000, seed=1)
    print(f"  {len(Xdev)} 个开发样本、{X.shape[1]} 个特征（只有前 2 个有用），{np.isnan(X).mean()*100:.0f}% 的格子缺失；测试 {len(Xte)}")
    st = preprocess_fit(Xdev, ydev)
    print(f"  训练上记住的量：fill 形状 {st['fill'].shape}、mu/sd 形状 {st['mu'].shape}、保留列 {st['keep'].size} 个、系数 {st['w'].shape}")
    has01 = {0, 1} <= set(st['keep'].tolist())
    print(f"  第 0 列：训练均值 {st['fill'][0]:+.3f}（全量 {np.nanmean(X[:, 0]):+.3f}）；保留的列里含 0、1 两列：{has01}")
    print(f"  {'哪一步在全部数据上 fit':<24} {'5 折 CV R²':>10}")
    rows = [("都只在训练折上 fit（无泄漏）", None), ("缺失值填补", "fill"), ("标准化", "scale"), ("特征选择（选 20 列）", "select")]
    res = {}
    for name, leak in rows:
        res[name] = cv_r2(Xdev, ydev, 20, leak)
        print(f"  {name:<28} {res[name]:>10.3f}")
    final = preprocess_fit(Xdev, ydev)
    print(f"  无泄漏流程在测试集上：R² {r2(yte, preprocess_apply(final, Xte)):.3f}；在 20,000 个全新样本上：R² {r2(yfresh, preprocess_apply(final, Xfresh)):.3f}")
    leaked_keep = preprocess_fit(np.where(np.isnan(X), np.nanmean(X, axis=0), X), y)["keep"]
    print(f"  全量上选出的 20 列与只在开发集上选出的 20 列重合 {len(set(leaked_keep.tolist()) & set(final['keep'].tolist()))} 个"
          f"（其中 998 个纯噪声列里，谁与 y 碰巧相关，取决于看到了哪些样本）")
    # 与 scikit-learn Pipeline 对照：同一套步骤装进 Pipeline，cross_val_score 会在每一折内部重新 fit 全部步骤
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), SelectKBest(f_regression, k=20), LinearRegression())  # !ref pipe
    sk = cross_val_score(pipe, Xdev, ydev, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()         # !ref cvs
    keep_all = SelectKBest(f_regression, k=20).fit(SimpleImputer().fit_transform(X), y).get_support()        # !ref leakfit
    Xsel = SimpleImputer().fit_transform(Xdev)[:, keep_all]
    sk_leak = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), Xsel, ydev,
                              cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
    print(f"  sklearn Pipeline 装进 cross_val_score：R² {sk:.3f}；先在全量上 SelectKBest 再 CV：R² {sk_leak:.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    names = list(res) + ["无泄漏流程\n在全新数据上"]
    vals = list(res.values()) + [r2(yfresh, preprocess_apply(final, Xfresh))]
    colors = [C["green"], C["orange"], C["orange"], C["red"], C["blue"]]
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)), [n.replace("（", "\n（") for n in names], fontsize=8)
    ax.set_ylabel("5 折 CV R²")
    ax.axhline(0, color=C["gray"], lw=0.8)
    save(fig, "01-leak-contrast")
    print("  特征选择在全量数据上 fit 时，验证折的 y 已经参与了「选哪些列」，CV 分数就不再是泛化估计")
    print()


EXPS = {"fit": exp_fit, "learncurve": exp_learncurve, "split": exp_split, "groups": exp_groups,
        "contamination": exp_contamination, "leak": exp_leak, "biasvar": exp_biasvar,
        "cv": exp_cv, "pipeline": exp_pipeline}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
