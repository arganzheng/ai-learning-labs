"""线性回归（经典 ML 02）：最小二乘、闭式解 vs 梯度下降、特征缩放、共线、Ridge / Lasso、Ridge = weight decay。
https://arganzheng.life/linear-regression-least-squares-ridge-and-lasso.html

    python 02_linear_regression.py          # 全部：line surface gd scaling collinear paths geometry wd smooth robust
    python 02_linear_regression.py paths    # 只跑一个

图输出到 out/02-*.svg。
"""
import sys

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from _plot import C, plt, save


# ---------------- 手写实现：全文的代码都在这几行里 ----------------
def fit_closed_form(X, y):
    """最小二乘闭式解（正规方程）。X 已含一列 1。"""
    return np.linalg.solve(X.T @ X, X.T @ y)                 # (XᵀX) w = Xᵀy


def fit_gd(X, y, lr=0.1, steps=200, batch=None, seed=0, wd=0.0):
    """梯度下降；batch=None 是全量 GD，给了 batch 就是小批量 SGD；wd 是 L2 正则系数。"""
    r = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(steps):
        idx = slice(None) if batch is None else r.choice(len(y), batch, replace=False)
        Xb, yb = X[idx], y[idx]
        grad = 2 * Xb.T @ (Xb @ w - yb) / len(yb) + 2 * wd * w    # ∂/∂w [ mean (Xw − y)² + wd·||w||² ]
        w -= lr * grad
        hist.append(np.mean((X @ w - y) ** 2))
    return w, np.array(hist)


def add_bias(X):
    return np.c_[np.ones(len(X)), X]


# ---------------- 1. 一条直线、20 个点、残差 ----------------
def make_line_data(n=20, seed=0):
    r = np.random.default_rng(seed)
    x = np.sort(r.uniform(0, 10, n))
    y = 1.5 * x + 2 + r.normal(0, 2.0, n)           # 真实：y = 1.5x + 2，噪声 σ=2
    return x, y


def exp_line():
    print("=== 1. 一条直线拟合 20 个点：残差、平方和 ===")
    x, y = make_line_data()
    X = add_bias(x[:, None])
    b, w = fit_closed_form(X, y)
    resid = y - (w * x + b)
    print(f"  闭式解：w = {w:.3f}, b = {b:.3f}（真实 1.5, 2）；残差平方和 {np.sum(resid**2):.2f}，MSE {np.mean(resid**2):.3f}")
    for name, (ww, bb) in {"猜的直线 A (w=1, b=5)": (1, 5), "猜的直线 B (w=2, b=0)": (2, 0), "最小二乘": (w, b)}.items():
        print(f"    {name:<24} 残差平方和 {np.sum((y - (ww*x+bb))**2):8.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    ax.scatter(x, y, s=18, color=C["blue"], zorder=3, label="20 个数据点")
    for name, (ww, bb), col in (("直线 A", (1, 5), C["gray"]), ("直线 B", (2, 0), C["orange"]), ("最小二乘直线", (w, b), C["red"])):
        ax.plot([0, 10], [bb, ww * 10 + bb], color=col, lw=1.5, label=f"{name}：残差平方和 {np.sum((y-(ww*x+bb))**2):.0f}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("三条候选直线")
    ax = axes[1]
    ax.scatter(x, y, s=18, color=C["blue"], zorder=3)
    ax.plot([0, 10], [b, w * 10 + b], color=C["red"], lw=1.5)
    for xi, yi, ri in zip(x, y, resid):
        ax.plot([xi, xi], [yi, yi - ri], color=C["red"], lw=0.8, alpha=0.6)
    ax.set_xlabel("x"); ax.set_title("最小二乘：每条竖线是一个残差，平方和最小")
    save(fig, "02-line-and-residuals")
    print()


# ---------------- 2. 损失曲面与梯度下降的路径 ----------------
def exp_surface():
    print("=== 2. 损失是 (w, b) 的一个碗：梯度下降沿着碗壁往下走 ===")
    x, y = make_line_data()
    X = add_bias(x[:, None])
    Xs = add_bias((x[:, None] - x.mean()) / x.std())         # 标准化后的版本，碗是圆的
    for name, XX in (("原始 x（0–10）", X), ("标准化 x", Xs)):
        w_star = fit_closed_form(XX, y)
        # 学习率上限：2 / 最大特征值（Hessian 2XᵀX/n）
        lam = np.linalg.eigvalsh(2 * XX.T @ XX / len(y))
        print(f"  {name:<12} Hessian 特征值 {lam.min():.2f} / {lam.max():.2f}（条件数 {lam.max()/lam.min():.0f}），稳定学习率上限 {2/lam.max():.3f}")
    ws = np.linspace(-1, 4, 70); bs = np.linspace(-6, 10, 70)
    W, B = np.meshgrid(ws, bs)
    L = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, lr, steps, title in ((axes[0], 0.005, 400, "学习率 0.005：400 步，先冲到沟底再沿沟慢慢爬"), (axes[1], 0.0265, 60, "学习率 0.0265：接近上限 0.027，在沟两壁间来回震荡")):
        w = np.zeros(2); path = [w.copy()]
        for _ in range(steps):
            w = w - lr * 2 * X.T @ (X @ w - y) / len(y); path.append(w.copy())
        path = np.array(path)
        cs = ax.contour(W, B, L, levels=np.geomspace(L.min() + 0.5, L.max(), 14), colors=C["gray"], linewidths=0.6)
        ax.plot(path[:, 1], path[:, 0], "o-", ms=2, lw=0.8, color=C["red"], alpha=0.8)
        ax.plot(*fit_closed_form(X, y)[::-1], "*", ms=10, color=C["orange"], label="闭式解")
        ax.set_xlabel("w（斜率）"); ax.set_ylabel("b（截距）"); ax.set_title(title, fontsize=8.5)
        ax.set_xlim(ws[0], ws[-1]); ax.set_ylim(bs[0], bs[-1])
    save(fig, "02-loss-surface-gd-path")
    print()


# ---------------- 3. 闭式解 vs GD vs SGD：同一个答案 ----------------
def exp_gd():
    print("=== 3. 闭式解 vs 梯度下降 vs 小批量 SGD ===")
    X, y, w_true = make_regression(n_samples=200, n_features=3, noise=5.0, coef=True, random_state=0)
    Xb = add_bias(X)
    w_closed = fit_closed_form(Xb, y)
    w_gd, h_gd = fit_gd(Xb, y, lr=0.05, steps=300)
    w_sgd, h_sgd = fit_gd(Xb, y, lr=0.05, steps=300, batch=16)
    w_sgd_decay, h_sgd_decay = fit_gd(Xb, y, lr=0.05, steps=300, batch=16, seed=1)
    mse_star = np.mean((Xb @ w_closed - y) ** 2)
    print(f"  真实系数        {np.round(w_true, 2)}")
    print(f"  闭式解          {np.round(w_closed[1:], 2)}  (偏置 {w_closed[0]:.2f})  MSE {mse_star:.2f}")
    print(f"  GD 300 步       {np.round(w_gd[1:], 2)}  (偏置 {w_gd[0]:.2f})  与闭式解最大差 {np.abs(w_gd - w_closed).max():.1e}")
    print(f"  SGD(16) 300 步  {np.round(w_sgd[1:], 2)}  (偏置 {w_sgd[0]:.2f})  与闭式解最大差 {np.abs(w_sgd - w_closed).max():.1e}")
    print(f"  sklearn         {np.round(LinearRegression().fit(X, y).coef_, 2)}")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.plot(h_gd - mse_star, color=C["blue"], label="全量 GD（每步用 200 个样本）")
    ax.plot(h_sgd - mse_star, color=C["orange"], alpha=0.9, lw=0.9, label="小批量 SGD（每步 16 个样本）")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("步数"); ax.set_ylabel("MSE − 闭式解的 MSE（对数轴）")
    ax.legend(frameon=False)
    save(fig, "02-gd-vs-sgd")
    print("  GD 单调下降到闭式解；SGD 用 1/12 的算量到达同一个邻域，但在最优点附近抖（噪声 ∝ 学习率 / batch）")
    print()


# ---------------- 4. 特征缩放：量纲不同，梯度下降就很难走 ----------------
def exp_scaling():
    print("=== 4. 特征缩放：一个特征 0–1、一个 0–1000，GD 怎么走 ===")
    r = np.random.default_rng(3)
    n = 300
    x1 = r.uniform(0, 1, n); x2 = r.uniform(0, 1000, n)
    y = 3 * x1 + 0.002 * x2 + r.normal(0, 0.1, n)
    X_raw = add_bias(np.c_[x1, x2])
    X_std = add_bias(StandardScaler().fit_transform(np.c_[x1, x2]))
    for name, X in (("原始特征", X_raw), ("标准化后", X_std)):
        lam = np.linalg.eigvalsh(2 * X.T @ X / n)
        lr = 0.9 * 2 / lam.max()
        w, h = fit_gd(X, y, lr=lr, steps=500)
        mse_star = np.mean((X @ fit_closed_form(X, y) - y) ** 2)
        print(f"  {name}: 条件数 {lam.max()/lam.min():.0f}，最大安全学习率 {lr:.1e}，500 步后 MSE 比最优多 {h[-1]-mse_star:.4f}")
    print("  量纲差 1000 倍 → 碗是一条极窄的沟，安全学习率被大特征限制，小特征的方向几乎不动；标准化让碗变圆，几十步收敛")
    print()


# ---------------- 5. 共线：两个几乎相同的特征 → 系数爆炸，Ridge 救 ----------------
def exp_collinear():
    print("=== 5. 共线：x₂ ≈ x₁ 时最小二乘的系数不稳定 ===")
    r = np.random.default_rng(5)
    n = 100
    x1 = r.normal(size=n)
    y_true = 2 * x1
    print(f"  {'x₂ 与 x₁ 的相关':>14} {'无正则 w₁, w₂':>22} {'Ridge α=1 w₁, w₂':>22}")
    for eps in (1.0, 0.1, 0.01):
        x2 = x1 + eps * r.normal(size=n)
        X = np.c_[x1, x2]
        y = y_true + r.normal(0, 0.5, n)
        w_ols = LinearRegression().fit(X, y).coef_
        w_r = Ridge(alpha=1.0).fit(X, y).coef_
        print(f"  {np.corrcoef(x1, x2)[0,1]:>14.4f} {w_ols[0]:>10.2f}, {w_ols[1]:>9.2f} {w_r[0]:>11.2f}, {w_r[1]:>9.2f}")
    print("  两列几乎相同时 XᵀX 接近不可逆：w₁ 大正、w₂ 大负也能给出同样的预测（它们的和才是确定的）；Ridge 让解唯一、把两个系数各分一半")
    print()


# ---------------- 6. Ridge / Lasso 的系数路径 ----------------
def exp_paths():
    print("=== 6. Ridge (L2) vs Lasso (L1)：50 个特征里只有 5 个真的有用 ===")
    X, y, w_true = make_regression(n_samples=100, n_features=50, n_informative=5, noise=10.0, coef=True, random_state=1)
    X = StandardScaler().fit_transform(X)
    print(f"  真实非零系数 {int((w_true != 0).sum())} 个")
    for name, model in [("无正则", LinearRegression()), ("Ridge α=10", Ridge(alpha=10)), ("Lasso α=1", Lasso(alpha=1.0)), ("Lasso α=5", Lasso(alpha=5.0))]:
        c = model.fit(X, y).coef_
        print(f"  {name:<12} 恰好为 0 的系数 {int((np.abs(c) < 1e-6).sum()):>2}/50   |系数| 均值 {np.abs(c).mean():6.2f}   最大 {np.abs(c).max():6.2f}")
    alphas = np.geomspace(1e-2, 1e3, 25)
    ridge_path = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    lasso_alphas = np.geomspace(1e-2, 50, 25)
    lasso_path = np.array([Lasso(alpha=a, max_iter=20000).fit(X, y).coef_ for a in lasso_alphas])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    informative = w_true != 0
    for ax, path, al, title in ((axes[0], ridge_path, alphas, "Ridge：所有系数一起变小，没有一个恰好为零"),
                                (axes[1], lasso_path, lasso_alphas, "Lasso：无用特征先后被压到恰好为零")):
        for j in range(50):
            ax.plot(al, path[:, j], color=C["red"] if informative[j] else C["blue"], lw=1.4 if informative[j] else 0.7,
                    alpha=1 if informative[j] else 0.5)
        ax.set_xscale("log"); ax.set_xlabel("正则强度 α（对数轴）"); ax.set_title(title, fontsize=8.5)
        ax.axhline(0, color=C["gray"], lw=0.6)
    axes[0].set_ylabel("系数值")
    axes[0].plot([], [], color=C["red"], label="5 个真有用的特征"); axes[0].plot([], [], color=C["blue"], lw=0.7, label="45 个无用特征")
    axes[0].legend(frameon=False, loc="upper right", fontsize=7)
    save(fig, "02-ridge-lasso-paths")
    print()


# ---------------- 7. 几何：为什么 L1 压到零、L2 压不到 ----------------
def exp_geometry():
    print("=== 7. 几何图：损失的椭圆等高线碰到 L1 的菱形先碰到角，碰到 L2 的圆碰到任意点 ===")
    w_star = np.array([1.8, 0.5])
    A = np.array([[1.0, -0.3], [-0.3, 2.0]])                    # 椭圆形状（XᵀX）
    ws = np.linspace(-1.5, 2.5, 120); W1, W2 = np.meshgrid(ws, ws)
    D = np.stack([W1 - w_star[0], W2 - w_star[1]], -1)
    L = np.einsum("...i,ij,...j->...", D, A, D)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, kind in zip(axes, ("L1", "L2")):
        # 在约束 ||w|| ≤ c 下找损失最小的点（网格搜索）
        norm = np.abs(W1) + np.abs(W2) if kind == "L1" else np.sqrt(W1**2 + W2**2)
        c = 1.0
        mask = norm <= c
        k = np.argmin(np.where(mask, L, np.inf)); wi = (W1.flat[k], W2.flat[k])
        ax.contour(W1, W2, L, levels=np.geomspace(0.05, 8, 12), colors=C["gray"], linewidths=0.6)
        if kind == "L1":
            ax.fill([c, 0, -c, 0], [0, c, 0, -c], color=C["blue"], alpha=0.25)
            ax.set_title(f"L1（Lasso）：菱形的角先碰到等高线\n解 ({wi[0]:.2f}, {wi[1]:.2f})：w₂ 被压到零", fontsize=8.5)
        else:
            ax.fill(c * np.cos(t), c * np.sin(t), color=C["blue"], alpha=0.25)
            ax.set_title(f"L2（Ridge）：圆周上任意一点都可能相切\n解 ({wi[0]:.2f}, {wi[1]:.2f})：两个都变小、都不为零", fontsize=8.5)
        ax.plot(*w_star, "*", ms=10, color=C["orange"], label="无正则的最小二乘解")
        ax.plot(*wi, "o", ms=7, color=C["red"], label="加约束后的解")
        ax.set_aspect("equal"); ax.set_xlabel("w₁"); ax.set_ylabel("w₂")
        ax.axhline(0, color=C["gray"], lw=0.5); ax.axvline(0, color=C["gray"], lw=0.5)
        print(f"  {kind}: 约束 ||w|| ≤ 1 下的解 ({wi[0]:.2f}, {wi[1]:.2f})")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    save(fig, "02-l1-l2-geometry")
    print()


# ---------------- 8. Ridge 就是 weight decay ----------------
def exp_wd():
    print("=== 8. Ridge 的闭式解 = 每步把 w 乘 (1 − 2·lr·λ) 的梯度下降 ===")
    X, y = make_regression(n_samples=200, n_features=5, noise=5.0, random_state=2)
    X = StandardScaler().fit_transform(X)
    lam = 0.5
    w_ridge = np.linalg.solve(X.T @ X / len(y) + lam * np.eye(5), X.T @ y / len(y))     # (XᵀX/n + λI) w = Xᵀy/n
    w = np.zeros(5); lr = 0.05
    for _ in range(2000):
        grad = 2 * X.T @ (X @ w - y) / len(y)
        w = (1 - 2 * lr * lam) * w - lr * grad                 # weight decay：先缩小 w，再走一步普通梯度
    print(f"  Ridge 闭式解     {np.round(w_ridge, 3)}")
    print(f"  GD + weight decay {np.round(w, 3)}   最大差 {np.abs(w - w_ridge).max():.1e}")
    print(f"  无正则闭式解     {np.round(fit_closed_form(X, y), 3)}（每个系数都比 Ridge 大：Ridge 把它们一起往零压）")
    print()


# ---------------- 9. 回到上一篇：15 次多项式 + Ridge，曲线变平滑 ----------------
def exp_smooth():
    print("=== 9. 上一篇的 15 次多项式，加 Ridge 之后 ===")
    r = np.random.default_rng(1)
    x = r.uniform(0, 1, 30); y = np.sin(2 * np.pi * x) + r.normal(0, 0.3, 30)
    xv = r.uniform(0, 1, 1000); yv = np.sin(2 * np.pi * xv) + r.normal(0, 0.3, 1000)
    xs = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
    for ax, alpha in zip(axes, (0, 1e-2, 10)):
        reg = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
        m = make_pipeline(PolynomialFeatures(15), StandardScaler(), reg).fit(x[:, None], y)
        tr = np.mean((m.predict(x[:, None]) - y) ** 2); va = np.mean((m.predict(xv[:, None]) - yv) ** 2)
        coef = m[-1].coef_
        print(f"  α = {alpha:<6g} 训练 MSE {tr:.3f}  验证 MSE {va:.3f}  |系数| 最大 {np.abs(coef).max():8.1f}")
        ax.plot(xs, np.sin(2 * np.pi * xs), "--", color=C["gray"], lw=1)
        ax.scatter(x, y, s=12, color=C["blue"], zorder=3)
        ax.plot(xs, m.predict(xs[:, None]), color=C["red"], lw=1.5)
        ax.set_ylim(-1.8, 1.8); ax.set_xlabel("x")
        ax.set_title(f"15 次多项式，α = {alpha:g}\n验证 MSE {va:.3f}，|系数| 最大 {np.abs(coef).max():.0f}", fontsize=8.5)
    save(fig, "02-ridge-smooths-polynomial")
    print("  正则项不改变模型容量（还是 15 次），只是不让高次系数变大——曲线于是平滑")
    print()


# ---------------- 10. 为什么是平方：高斯噪声 vs 离群点 ----------------
def exp_robust():
    print("=== 10. 平方误差对应高斯噪声；有离群点时它被拉偏，绝对值误差（拉普拉斯）不怕 ===")
    x, y = make_line_data(seed=4)
    y_out = y.copy(); y_out[[3, 15]] += np.array([25, -20])                # 两个离群点
    X = add_bias(x[:, None])
    b2, w2 = fit_closed_form(X, y_out)
    # 绝对值误差没有闭式解：用 IRLS（迭代加权最小二乘）近似 L1 回归
    wts = np.ones(len(y)); wb = np.zeros(2)
    for _ in range(50):
        Wm = X * wts[:, None]
        wb = np.linalg.solve(Wm.T @ X, Wm.T @ y_out)
        wts = 1 / np.maximum(np.abs(y_out - X @ wb), 1e-3)
    b1, w1 = wb
    print(f"  无离群点的最小二乘        w = {fit_closed_form(X, y)[1]:.2f}")
    print(f"  两个离群点 + 平方误差     w = {w2:.2f}, b = {b2:.2f}（被拉偏）")
    print(f"  两个离群点 + 绝对值误差   w = {w1:.2f}, b = {b1:.2f}（几乎不受影响）")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.scatter(x, y_out, s=18, color=C["blue"], zorder=3)
    ax.scatter(x[[3, 15]], y_out[[3, 15]], s=60, facecolors="none", edgecolors=C["red"], zorder=4, label="离群点")
    ax.plot([0, 10], [b2, w2 * 10 + b2], color=C["orange"], lw=1.5, label=f"平方误差（MSE）：w = {w2:.2f}")
    ax.plot([0, 10], [b1, w1 * 10 + b1], color=C["green"], lw=1.5, label=f"绝对值误差（MAE）：w = {w1:.2f}")
    ax.plot([0, 10], [2, 17], "--", color=C["gray"], lw=1, label="真实：w = 1.5")
    ax.legend(frameon=False, fontsize=7); ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "02-mse-vs-mae-outliers")
    print()


EXPS = {"line": exp_line, "surface": exp_surface, "gd": exp_gd, "scaling": exp_scaling, "collinear": exp_collinear,
        "paths": exp_paths, "geometry": exp_geometry, "wd": exp_wd, "smooth": exp_smooth, "robust": exp_robust}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
