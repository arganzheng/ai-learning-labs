"""什么是学习（经典 ML 01）：划分、过拟合与欠拟合、偏差-方差、测试集泄漏——全部用一个能画出来的例子。
https://arganzheng.life/what-is-learning-splits-generalization-and-bias-variance.html

    python 01_learning_and_generalization.py          # 全部：fit split leak biasvar
"""
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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


# ---------------- 1. 欠拟合 / 过拟合：多项式次数扫描 ----------------
def exp_fit():
    print("=== 1. 模型容量 vs 训练误差 / 验证误差（30 个点拟合 sin）===")
    X, y = make_data(30, seed=1)
    Xv, yv = make_data(1000, seed=2)
    print(f"  {'次数':>4} {'训练 MSE':>10} {'验证 MSE':>10}   判断")
    for d in (1, 3, 5, 9, 15, 25):
        m = poly_model(d).fit(X, y)
        tr, va = mse(m, X, y), mse(m, Xv, yv)
        tag = "欠拟合（两个都高）" if d <= 1 else ("过拟合（训练低、验证高）" if va > 2.5 * tr else "合适")
        print(f"  {d:>4} {tr:>10.3f} {va:>10.3f}   {tag}")
    print("  噪声方差 0.09 是验证 MSE 的下限：再好的模型也降不到它以下（不可约误差）")
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
          f"{(gaps > 0).mean()*100:.0f}% 的情况下真实更差")
    print("  这就是 benchmark 上反复调 prompt / 选 checkpoint 之后，分数不再可信的原因")
    print()


# ---------------- 4. 偏差-方差：多次重采样看预测的均值与抖动；集成降方差 ----------------
def exp_biasvar():
    print("=== 4. 偏差-方差分解与集成（bagging）===")
    xs = np.linspace(0.05, 0.95, 200)[:, None]
    for d in (1, 4, 9):
        preds = np.array([poly_model(d).fit(*make_data(30, seed=s)).predict(xs) for s in range(100)])  # [100, 200]
        bias2 = np.mean((preds.mean(0) - truth(xs[:, 0])) ** 2)
        var = np.mean(preds.var(0))
        # bagging：每次用 20 个重采样模型的平均
        bag = np.array([np.mean([poly_model(d).fit(*make_data(30, seed=1000 * s + k)).predict(xs) for k in range(20)], 0) for s in range(30)])
        print(f"  次数 {d:>2}: 偏差² {bias2:.3f}  方差 {var:.3f}   → 20 个模型平均后方差 {np.mean(bag.var(0)):.3f}（偏差² {np.mean((bag.mean(0) - truth(xs[:, 0]))**2):.3f} 基本不变）")
    print("  低次：偏差大（模型太简单，系统性地错）；高次：方差大（对哪 30 个点太敏感）；平均多个模型只降方差")
    print("  LLM 里的对应：self-consistency 多次采样投票、多个 judge 取平均、model soup —— 都是在降方差")
    print()


EXPS = {"fit": exp_fit, "split": exp_split, "leak": exp_leak, "biasvar": exp_biasvar}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
