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
        _w_star = fit_closed_form(XX, y)
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
        path_arr = np.array(path)
        ax.contour(W, B, L, levels=np.geomspace(L.min() + 0.5, L.max(), 14), colors=C["gray"], linewidths=0.6)
        ax.plot(path_arr[:, 1], path_arr[:, 0], "o-", ms=2, lw=0.8, color=C["red"], alpha=0.8)
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
    _w_sgd_decay, _h_sgd_decay = fit_gd(Xb, y, lr=0.05, steps=300, batch=16, seed=1)
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
        _w, h = fit_gd(X, y, lr=lr, steps=500)
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


# ---------------- 11. 矩阵形状与一步梯度：3 个点，每个中间量的形状和数值都能手算 ----------------
def exp_step():
    print("=== 11. 矩阵形状与一步梯度：3 个点上把 GD 的一步拆开 ===")
    X = add_bias(np.array([1.0, 2.0, 3.0]))                                  # !ref X
    y = np.array([3.0, 5.0, 8.0])
    n = len(y)
    w = np.zeros(X.shape[1])                                                 # !ref w0
    pred = X @ w                                                             # !ref pred
    resid = pred - y                                                         # !ref resid
    grad = 2 * X.T @ resid / n                                               # !ref grad
    print(f"  X {X.shape}  w {w.shape}  X@w {pred.shape}  残差 {resid.shape}  Xᵀ@残差 {(X.T @ resid).shape}  梯度 {grad.shape}")
    print(f"  w = 0 时：残差 {resid}，Xᵀ·残差 = {X.T @ resid}，梯度 = 2/3 × 那个 = {np.round(grad, 3)}，MSE {np.mean(resid**2):.3f}")
    lr = 0.05
    w1 = w - lr * grad                                                       # !ref step
    print(f"  学习率 {lr}：w ← 0 − {lr} × 梯度 = {np.round(w1, 3)}，MSE {np.mean((X @ w1 - y)**2):.3f}")
    H = 2 * X.T @ X / n                                                      # !ref hess
    lam = np.linalg.eigvalsh(H)
    print(f"  Hessian H = 2/n·XᵀX =\n{np.round(H, 3)}\n  特征值 {np.round(lam, 2)}，稳定上限 2/λmax = {2/lam.max():.3f}，条件数 {lam.max()/lam.min():.1f}")
    w_star = fit_closed_form(X, y)
    for lr in (0.05, 0.17, 0.19):
        w_gd, hist = fit_gd(X, y, lr=lr, steps=200)
        print(f"  lr={lr}: 200 步后 w = {np.round(w_gd, 3)}，与闭式解 {np.round(w_star, 3)} 最大差 {np.abs(w_gd - w_star).max():.1e}，MSE {hist[-1]:.3g}")
    print()


# ---------------- 12. 为什么不显式求逆：条件数、inv / solve / lstsq 三种算法 ----------------
def exp_solvers():
    print("=== 12. 为什么实现不显式求逆：条件数把误差放大多少，三种求解器各错多少 ===")
    A = np.array([[1.0, 1.0], [1.0, 1.0001]])
    b = np.array([2.0, 2.0001])
    b2 = b + np.array([0.0, 1e-4])
    print(f"  2×2 例子：A = {A.tolist()}，cond(A) = {np.linalg.cond(A):.1e}")
    print(f"    b = {b} → 解 {np.round(np.linalg.solve(A, b), 4)}；b 的第二项加 1e-4 → 解 {np.round(np.linalg.solve(A, b2), 4)}")
    print("    右边动了万分之一，解动了 100%：条件数 ≈ 4×10⁴ 就是这个放大倍数的上界")
    r = np.random.default_rng(0)
    x = r.uniform(0, 1, 30)
    y = np.sin(2 * np.pi * x) + r.normal(0, 0.3, 30)
    print("  上一篇的 30 个点、多项式特征 1, x, …, x^d：")
    print(f"  {'次数':>4} {'cond(X)':>9} {'cond(XᵀX)':>10} {'inv 残差平方和':>12} {'solve 残差平方和':>13} {'lstsq 残差平方和':>13} {'sklearn':>10} {'inv 与 lstsq 系数最大差':>14}")
    rows = []
    for d in (3, 6, 9, 12, 15, 20):
        X = np.vander(x, d + 1, increasing=True)
        G, c = X.T @ X, X.T @ y
        w_inv = np.linalg.inv(G) @ c                                         # !ref inv
        w_solve = np.linalg.solve(G, c)                                      # !ref solve
        w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]                       # !ref lstsq
        w_sk = LinearRegression(fit_intercept=False).fit(X, y).coef_
        rss = [float(np.sum((X @ w - y) ** 2)) for w in (w_inv, w_solve, w_lstsq, w_sk)]
        rows.append((d, np.linalg.cond(X), np.linalg.cond(G), *rss, float(np.abs(w_inv - w_lstsq).max())))
        print(f"  {d:>4} {rows[-1][1]:>9.1e} {rows[-1][2]:>10.1e} {rss[0]:>14.4f} {rss[1]:>16.4f} {rss[2]:>16.4f} {rss[3]:>10.4f} {rows[-1][-1]:>20.1e}")
    print(f"  机器精度 eps = {np.finfo(float).eps:.1e}：cond(XᵀX) 超过 1/eps ≈ 4.5e15 后，正规方程里的 XᵀX 在浮点里已经分不清是否可逆")
    print("  精确算术里 cond(XᵀX) = cond(X)²，先算 XᵀX 再解等于主动把条件数平方（表里 1e17 附近是浮点算出来的饱和值）；")
    print("  lstsq / sklearn 直接对 X 做 SVD（LAPACK gelsd），只付 cond(X) 的代价")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ds = [row[0] for row in rows]
    for i, (label, color) in enumerate(((" inv(XᵀX)·Xᵀy", C["red"]), ("solve(XᵀX, Xᵀy)", C["orange"]), ("lstsq(X, y)", C["blue"]))):
        ax.plot(ds, [row[3 + i] for row in rows], "o-", color=color, label=label, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("多项式次数 d")
    ax.set_ylabel("训练残差平方和（对数轴）")
    ax.set_title("30 个点拟合 d 次多项式：先算 XᵀX 再求解，d 越大错得越多", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(ds, [row[2] for row in rows], "s--", color=C["gray"], lw=1, label="cond(XᵀX)")
    ax2.set_yscale("log")
    ax2.set_ylabel("cond(XᵀX)", color=C["gray"])
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "02-solver-conditioning")
    print()


# ---------------- 13. Lasso：一维软阈值 + 坐标下降，稀疏从哪来 ----------------
def soft_threshold(rho, t):
    """软阈值：|rho| ≤ t 时直接归零，否则向零收缩 t。"""
    return np.sign(rho) * max(abs(rho) - t, 0.0)                              # !ref soft


def lasso_cd(X, y, alpha, sweeps=100, tol=1e-10, log=None):
    """坐标下降解 sklearn 形式的 Lasso：(1/2n)‖y − Xw‖² + α‖w‖₁，无截距。每次只动一个 w_j。"""
    n, d = X.shape
    w = np.zeros(d)
    q = (X ** 2).sum(axis=0) / n                                               # !ref q
    for sweep in range(1, sweeps + 1):
        w_old = w.copy()
        for j in range(d):
            r_j = y - X @ w + X[:, j] * w[j]                                   # !ref partial
            rho = X[:, j] @ r_j / n                                            # !ref rho
            w[j] = soft_threshold(rho, alpha) / q[j]                           # !ref update
        if log is not None:
            log.append(w.copy())
        if np.abs(w - w_old).max() < tol:
            break
    return w, sweep


def exp_soft():
    print("=== 13. Lasso 的稀疏从哪来：一维软阈值手算，再用坐标下降对上 sklearn ===")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 5.0, 8.0])
    n = len(y)
    rho, q = x @ y / n, x @ x / n
    print(f"  一个特征、无截距：ρ = xᵀy/n = {rho:.3f}，q = xᵀx/n = {q:.3f}，最小二乘 w = ρ/q = {rho/q:.4f}")
    print(f"  {'α':>6} {'软阈值 S(ρ, α)/q':>16} {'sklearn Lasso':>14}")
    for a in (0.0, 1.0, 4.0, 10.0, rho, 13.0):
        w_hand = soft_threshold(rho, a) / q
        w_sk = Lasso(alpha=a, fit_intercept=False).fit(x[:, None], y).coef_[0] if a > 0 else LinearRegression(fit_intercept=False).fit(x[:, None], y).coef_[0]
        print(f"  {a:>6.3f} {w_hand:>16.4f} {w_sk:>14.4f}")
    print(f"  α ≥ |ρ| = {rho:.3f} 时 w 恰好是 0——不是「很小」，是零；Ridge 对应的收缩 ρ/(q + α) 永远不为零")
    X, y5, w_true = make_regression(n_samples=60, n_features=5, n_informative=2, noise=5.0, coef=True, random_state=4)
    X = StandardScaler().fit_transform(X)
    y5 = y5 - y5.mean()
    alpha = 5.0
    log: list = []
    w_cd, sweeps = lasso_cd(X, y5, alpha, log=log)
    print(f"  5 个特征（真实非零 {int((w_true != 0).sum())} 个）、α={alpha}，坐标下降逐轮的 w：")
    for k in (0, 1, 2, 4, len(log) - 1):
        print(f"    第 {k+1:>2} 轮  {np.round(log[k], 4)}")
    sk = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=100000).fit(X, y5).coef_
    print(f"  {sweeps} 轮后收敛；sklearn Lasso(α={alpha}) 的系数 {np.round(sk, 4)}，最大差 {np.abs(w_cd - sk).max():.1e}")
    print(f"  恰好为 0 的系数：坐标下降 {int((w_cd == 0).sum())} 个，sklearn {int((sk == 0).sum())} 个；真实为 0 的 {int((w_true == 0).sum())} 个")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    rr = np.linspace(-15, 15, 301)
    axes[0].plot(rr, rr / q, color=C["gray"], lw=1, ls="--", label="最小二乘 ρ/q")
    axes[0].plot(rr, rr / (q + 4), color=C["blue"], lw=1.4, label="Ridge ρ/(q+α)，α=4")
    axes[0].plot(rr, [soft_threshold(v, 4.0) / q for v in rr], color=C["red"], lw=1.6, label="Lasso S(ρ,α)/q，α=4")
    axes[0].axvspan(-4, 4, color=C["light"], alpha=0.6)
    axes[0].plot([rho], [soft_threshold(rho, 4.0) / q], "o", color=C["red"])
    axes[0].set_xlabel("ρ = xᵀy/n（特征与目标的相关量）")
    axes[0].set_ylabel("解出的 w")
    axes[0].set_title("一维：|ρ| ≤ α 的灰色区间里 Lasso 的解恰好为 0", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7)
    path = np.array(log)
    for j in range(5):
        axes[1].plot(range(1, len(path) + 1), path[:, j], "o-", ms=3, lw=1.2,
                     color=C["red"] if w_true[j] != 0 else C["blue"], label=f"w{j+1}" + ("（真有用）" if w_true[j] != 0 else ""))
    axes[1].set_xlabel("坐标下降轮数")
    axes[1].set_title("5 个特征：无用特征的系数在前几轮就被压到 0", fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "02-soft-threshold-cd")
    print()


# ---------------- 14. 正则系数到底乘在什么上：求和 / 平均两种写法，手算、NumPy、sklearn 对齐 ----------------
def exp_align():
    print("=== 14. 正则系数在「按求和」与「按平均」的损失里如何对应：手算、NumPy、sklearn 同一目标才可比 ===")
    X, y = make_regression(n_samples=200, n_features=5, noise=5.0, random_state=2)
    X = StandardScaler().fit_transform(X)
    y = y - y.mean()
    n = len(y)
    lam = 0.5
    w_mean = np.linalg.solve(X.T @ X / n + lam * np.eye(5), X.T @ y / n)          # !ref mean_form
    w_sum = np.linalg.solve(X.T @ X + n * lam * np.eye(5), X.T @ y)               # !ref sum_form
    w_sk = Ridge(alpha=n * lam, fit_intercept=False).fit(X, y).coef_               # !ref sk_ridge
    w_sk_wrong = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
    w_gd, _ = fit_gd(X, y, lr=0.05, steps=3000, wd=lam)                           # !ref gd_wd
    print(f"  Ridge，n={n}，平均式 λ={lam}：")
    print(f"    (XᵀX/n + λI)w = Xᵀy/n       {np.round(w_mean, 4)}")
    print(f"    (XᵀX + nλI)w = Xᵀy          {np.round(w_sum, 4)}   与上行最大差 {np.abs(w_sum - w_mean).max():.1e}")
    print(f"    sklearn Ridge(alpha=n·λ={n*lam:g}) {np.round(w_sk, 4)}   最大差 {np.abs(w_sk - w_mean).max():.1e}")
    print(f"    fit_gd(wd=λ) 3000 步          {np.round(w_gd, 4)}   最大差 {np.abs(w_gd - w_mean).max():.1e}")
    print(f"    sklearn Ridge(alpha=λ={lam})      {np.round(w_sk_wrong, 4)}   最大差 {np.abs(w_sk_wrong - w_mean).max():.1e}  ← 把 λ 直接当 alpha 传，正则弱了 {n} 倍")
    beta = 400.0
    w_cd, _ = lasso_cd(X, y, beta / (2 * n))                                        # !ref lasso_align
    w_lsk = Lasso(alpha=beta / (2 * n), fit_intercept=False, tol=1e-10, max_iter=100000).fit(X, y).coef_
    print(f"  Lasso，求和式 ‖y−Xw‖² + β‖w‖₁ 里 β={beta:g}：sklearn 的目标是 (1/2n)‖y−Xw‖² + α‖w‖₁，两边同除 2n 得 α = β/(2n) = {beta/(2*n):g}")
    print(f"    坐标下降(α={beta/(2*n):g}) {np.round(w_cd, 4)}；sklearn Lasso(alpha={beta/(2*n):g}) {np.round(w_lsk, 4)}；最大差 {np.abs(w_cd - w_lsk).max():.1e}")
    Xr, yr = make_regression(n_samples=200, n_features=5, noise=5.0, random_state=2)
    Xr = Xr + np.array([10.0, -3.0, 0.0, 5.0, 1.0])
    yr = yr + 100.0
    alpha = 100.0
    D = np.eye(6)
    D[0, 0] = 0.0                                                                   # !ref no_pen_bias
    Xb = add_bias(Xr)
    w_hand = np.linalg.solve(Xb.T @ Xb + alpha * D, Xb.T @ yr)
    m = Ridge(alpha=alpha).fit(Xr, yr)
    print("  带截距：sklearn Ridge 先把 X、y 中心化、不惩罚截距；手写时惩罚矩阵的截距那一格写 0——")
    print(f"    手写 [b, w] = {np.round(w_hand, 4)}")
    print(f"    sklearn [intercept_, coef_] = {np.round(np.r_[m.intercept_, m.coef_], 4)}   最大差 {np.abs(w_hand - np.r_[m.intercept_, m.coef_]).max():.1e}")
    w_pen_bias = np.linalg.solve(Xb.T @ Xb + alpha * np.eye(6), Xb.T @ yr)
    print(f"    若连截距一起惩罚：b = {w_pen_bias[0]:.4f}（sklearn 的 {m.intercept_:.4f}），系数最大差 {np.abs(w_pen_bias - np.r_[m.intercept_, m.coef_]).max():.1e}")
    print()


EXPS = {"line": exp_line, "surface": exp_surface, "gd": exp_gd, "scaling": exp_scaling, "collinear": exp_collinear,
        "paths": exp_paths, "geometry": exp_geometry, "wd": exp_wd, "smooth": exp_smooth, "robust": exp_robust,
        "step": exp_step, "solvers": exp_solvers, "soft": exp_soft, "align": exp_align}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
