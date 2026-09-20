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
    gaps = np.array(gaps)
    print(f"  200 次重复：'在测试集上选出的最好分数' 比 '同一模型在全新数据上的真实分数' 平均乐观 {gaps.mean():.3f}（MSE），"
          f"{(gaps > 0).mean()*100:.0f}% 的情况下真实更差；中位数 {np.median(gaps):.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.hist(np.clip(gaps, -0.2, 1.5), bins=40, color=C["blue"], alpha=0.85)
    ax.axvline(0, color=C["gray"], lw=1)
    ax.axvline(gaps.mean(), color=C["red"], ls="--", lw=1.2, label=f"平均 {gaps.mean():.3f}")
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


EXPS = {"fit": exp_fit, "learncurve": exp_learncurve, "split": exp_split, "groups": exp_groups,
        "contamination": exp_contamination, "leak": exp_leak, "biasvar": exp_biasvar}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
