"""算法工程师的数学（07）：导数、梯度与链式法则——softmax 的梯度与策略梯度。文中全部数字与图。
https://arganzheng.life/derivatives-gradients-chain-rule-and-policy-gradient.html

    python 07_gradients_and_policy_gradient.py            # 全部，约 1 分钟（CPU）
    python 07_gradients_and_policy_gradient.py pg gd      # 只跑指定的段

段：deriv（导数 = 斜率：割线逼近切线）· grad（梯度：等高线与箭头场）· chain（链式法则：手算 / autograd / 有限差分三方对拍）
    · smgrad（softmax + 交叉熵的梯度 = p − y）· pg（策略梯度：10 个 token 的玩具策略——精确梯度 vs 采样估计、baseline、GRPO 的偏差、三种方法的训练曲线）
    · gd（梯度下降：三种学习率的轨迹；随机梯度的抖动）· lagrange（拉格朗日乘子：等高线与约束线相切）
"""
import math
import sys

import numpy as np
import torch

from _plot import C, plt, save

torch.manual_seed(0)
rng = np.random.default_rng(0)


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ---------------------------------------------------------------- deriv
def deriv():
    section("导数 = 斜率：f(x) = x² 在 x = 1 处，割线越来越短，斜率逼近 2")
    f = lambda x: x ** 2
    x0 = 1.0
    print(f"  {'ε':>8} {'(f(1+ε) − f(1)) / ε':>22}")
    for eps in (1.0, 0.5, 0.1, 0.01, 0.001):
        print(f"  {eps:>8} {(f(x0 + eps) - f(x0)) / eps:>22.4f}")
    print("  → 趋向 2 = f'(1) = 2·1。导数就是「割线短到看不见时的斜率」。")
    xs = np.linspace(-0.5, 2.5, 200)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
    for ax, eps in zip(axes, (1.0, 0.5, 0.1)):
        ax.plot(xs, f(xs), color=C["blue"])
        x1 = x0 + eps
        slope = (f(x1) - f(x0)) / eps
        ax.plot([x0, x1], [f(x0), f(x1)], "o", color=C["red"])
        ax.plot(xs, f(x0) + slope * (xs - x0), color=C["red"], lw=1.2, label=f"割线 斜率 {slope:.2f}")
        ax.plot(xs, f(x0) + 2 * (xs - x0), color=C["green"], ls="--", lw=1, label="切线 斜率 2")
        ax.set(title=f"ε = {eps}", xlabel="x", ylim=(-1, 6))
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("f(x) = x²")
    save(fig, "07-derivative-secant")


# ---------------------------------------------------------------- grad
def L_bowl(t1, t2):
    return t1 ** 2 + 3 * t2 ** 2


def grad_bowl(t1, t2):
    return np.array([2 * t1, 6 * t2])


def grad():
    section("梯度：L(θ₁, θ₂) = θ₁² + 3θ₂²，∇L = (2θ₁, 6θ₂)")
    for pt in ((1, 1), (2, 0), (0, 1), (-1, 0.5)):
        g = grad_bowl(*pt)
        print(f"  在 {pt}：L = {L_bowl(*pt):.2f}，∇L = ({g[0]:.0f}, {g[1]:.0f})，长度 {np.linalg.norm(g):.2f}；沿 −∇L 走 L 降得最快")
    t1, t2 = np.meshgrid(np.linspace(-2.6, 2.6, 200), np.linspace(-1.6, 2.1, 200))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    cs = ax.contour(t1, t2, L_bowl(t1, t2), levels=[0.25, 1, 2, 4, 7, 10], colors=C["gray"], linewidths=0.8)
    ax.clabel(cs, fmt="L=%g", fontsize=7)
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-1.2, 1.2, 7))
    gx, gy = 2 * g1, 6 * g2
    ax.quiver(g1, g2, gx, gy, color=C["blue"], scale=60, width=0.003, alpha=0.8)
    p = (1.0, 1.0)
    g = grad_bowl(*p)
    ax.annotate("", (p[0] + g[0] * 0.12, p[1] + g[1] * 0.12), p, arrowprops=dict(arrowstyle="->", color=C["red"], lw=2))
    ax.annotate("", (p[0] - g[0] * 0.12, p[1] - g[1] * 0.12), p, arrowprops=dict(arrowstyle="->", color=C["green"], lw=2))
    ax.plot(*p, "o", color="k", ms=4)
    ax.text(1.35, 1.65, "∇L = (2, 6)\n上坡最快", color=C["red"], fontsize=8)
    ax.text(0.2, 0.05, "−∇L\n下坡最快", color=C["green"], fontsize=8)
    ax.set(xlabel="θ₁", ylabel="θ₂", title="等高线是 L 相等的点；梯度箭头垂直于等高线、指向 L 增长最快的方向", aspect="equal", xlim=(-2.6, 2.6), ylim=(-1.6, 2.1))
    save(fig, "07-gradient-field")


# ---------------------------------------------------------------- chain
def chain():
    section("链式法则：L = g₁ + 2g₂，g₁ = θ₁θ₂，g₂ = θ₁ + θ₂，在 θ = (2, 3) 处")
    th = torch.tensor([2.0, 3.0], requires_grad=True)
    g1, g2 = th[0] * th[1], th[0] + th[1]
    L = g1 + 2 * g2
    L.backward()
    print(f"  手算：∂L/∂g = (1, 2)，Jacobian = [[θ₂, θ₁], [1, 1]] = [[3, 2], [1, 1]]，(1, 2)·J = (5, 4)")
    print(f"  torch.autograd：{th.grad.tolist()}")
    eps = 1e-4
    Lf = lambda a, b: a * b + 2 * (a + b)
    fd = [(Lf(2 + eps, 3) - Lf(2 - eps, 3)) / (2 * eps), (Lf(2, 3 + eps) - Lf(2, 3 - eps)) / (2 * eps)]
    print(f"  有限差分（θ 各挪 ±1e-4）：({fd[0]:.4f}, {fd[1]:.4f})   ← gradient check：三者一致")

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    nodes = {"θ₁ = 2": (1, 3.8), "θ₂ = 3": (1, 1.2), "g₁ = θ₁θ₂ = 6": (4.5, 3.8), "g₂ = θ₁+θ₂ = 5": (4.5, 1.2), "L = g₁+2g₂ = 16": (8.3, 2.5)}
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.9, y - 0.4), 1.8 if not name.startswith("L") else 2.2, 0.8, fc="#eef3fa", ec=C["blue"]))
        ax.text(x + (0.2 if name.startswith("L") else 0), y, name, ha="center", va="center", fontsize=8.5)
    edges = [((1.9, 3.8), (3.6, 3.8), "∂g₁/∂θ₁ = θ₂ = 3", (2.75, 4.15)),
             ((1.9, 1.2), (3.6, 3.6), "∂g₁/∂θ₂ = θ₁ = 2", (1.75, 2.75)),
             ((1.9, 3.6), (3.6, 1.4), "∂g₂/∂θ₁ = 1", (3.75, 2.3)),
             ((1.9, 1.2), (3.6, 1.2), "∂g₂/∂θ₂ = 1", (2.75, 0.85)),
             ((5.4, 3.8), (7.3, 2.7), "∂L/∂g₁ = 1", (6.3, 3.6)),
             ((5.4, 1.2), (7.3, 2.3), "∂L/∂g₂ = 2", (6.3, 1.4))]
    for (x0, y0), (x1, y1), lab, (lx, ly) in edges:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=C["gray"]))
        ax.text(lx, ly, lab, fontsize=7.5, color=C["gray"], ha="center")
    ax.text(1, 4.55, "∂L/∂θ₁ = 1·3 + 2·1 = 5", color=C["red"], fontsize=9, ha="center", fontweight="bold")
    ax.text(1, 0.4, "∂L/∂θ₂ = 1·2 + 2·1 = 4", color=C["red"], fontsize=9, ha="center", fontweight="bold")
    ax.text(5, 0.15, "前向：沿箭头算值。反向：从 L 出发，沿每条路径把边上的局部导数相乘，汇到同一个 θ 的路径相加", fontsize=8, color=C["gray"], ha="center")
    save(fig, "07-chain-rule-graph")


# ---------------------------------------------------------------- smgrad
def smgrad():
    section("softmax + 交叉熵的梯度 = p − y：z = (2, 1, 0)，真实 token 是第 2 个")
    z = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    p = torch.softmax(z, 0)
    loss = -torch.log(p[1])
    loss.backward()
    y = torch.tensor([0.0, 1.0, 0.0])
    print(f"  p = {[round(v, 3) for v in p.tolist()]}，p − y = {[round(v, 3) for v in (p.detach() - y).tolist()]}")
    print(f"  autograd 的 ∂loss/∂z = {[round(v, 3) for v in z.grad.tolist()]}   ← 一致")
    print("  第 2 位为负（梯度下降走 −∇，所以它的 logit 被推高 0.755）；第 1、3 位为正、被推低 0.665 与 0.090——概率越高推得越狠")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    xs = np.arange(3)
    ax.bar(xs - 0.18, p.detach().numpy(), 0.34, color=C["light"], label="当前概率 p")
    ax.bar(xs + 0.18, y.numpy(), 0.34, color="#f3d6a6", label="真实 y (one-hot)")
    for i, gval in enumerate((p.detach() - y).tolist()):
        top = max(p[i].item(), y[i].item()) + 0.06
        col = C["green"] if gval < 0 else C["red"]
        ax.annotate("", (i, top + min(max(abs(gval), 0.15) * 0.5, 0.22) * (-1 if gval > 0 else 1)), (i, top + 0.02 if gval > 0 else top), arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.text(i, 1.28, f"更新 −(p−y) = {-gval:+.3f}", ha="center", fontsize=8, color=col)
    ax.set_xticks(xs, ["token 1\nz=2", "token 2（真实）\nz=1", "token 3\nz=0"])
    ax.set(ylim=(0, 1.42), ylabel="概率", title="梯度 p − y：真实 token 往上推 (1 − p)，其余往下推各自的 p")
    ax.legend(loc="center left", fontsize=7, bbox_to_anchor=(0.0, 0.6))
    save(fig, "07-softmax-ce-gradient")


# ---------------------------------------------------------------- pg
V = 10
R = np.array([1.0 if k % 2 == 0 else 0.0 for k in range(V)])   # 偶数 token 得 1 分


def softmax_np(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def exact_grad(z):
    """∂J/∂z_k = π_k (R_k − J)，J = Σ π_k R_k。V 小，直接算。"""
    pi = softmax_np(z)
    J = pi @ R
    return pi * (R - J), J


def grad_logpi(z, k):
    """∇_z log π_k = e_k − π"""
    g = -softmax_np(z)
    g[k] += 1
    return g


def pg():
    section("策略梯度：10 个 token 的策略，奖励 = 偶数得 1")
    z0 = np.zeros(V)                       # 初始：均匀
    g_exact, J = exact_grad(z0)
    print(f"  初始均匀策略：J = E[R] = {J:.2f}；精确梯度 ∂J/∂z = {np.round(g_exact, 3).tolist()}")
    print(f"  读法：偶数 token 的 logit 该增加（+{g_exact[0]:.2f}），奇数的该减少（{g_exact[1]:.2f}）——梯度 = π_k (R_k − J)：比平均好的往上、比平均差的往下")

    # 一次估计 = 采 n 条、平均 (R − b)·∇log π
    def estimate(z, n, b_fn, trials=20000):
        pi = softmax_np(z)
        out = np.empty((trials, V))
        for t in range(trials):
            ks = rng.choice(V, size=n, p=pi)
            rs = R[ks]
            bs = b_fn(rs)
            out[t] = np.mean([(rs[i] - bs[i]) * grad_logpi(z, ks[i]) for i in range(n)], axis=0)
        return out

    n = 4
    variants = {
        "REINFORCE（b = 0）": lambda rs: np.zeros_like(rs),
        "baseline b = 0.5（= 平均奖励）": lambda rs: np.full_like(rs, 0.5),
        "离谱的 baseline b = −5": lambda rs: np.full_like(rs, -5.0),
        "GRPO：b = 组内均值（含自己）": lambda rs: np.full_like(rs, rs.mean()),
        "RLOO：b = 其余几条的均值": lambda rs: (rs.sum() - rs) / (len(rs) - 1),
    }
    print(f"\n  每次估计采 n = {n} 条回答，重复 20000 次，看 ∂J/∂z₀（token 0，偶数）这一个分量：精确值 {g_exact[0]:.4f}")
    print(f"  {'方法':<30} {'估计的均值':>10} {'标准差':>8}")
    hist = {}
    for name, bf in variants.items():
        est = estimate(z0, n, bf)
        hist[name] = est[:, 0]
        print(f"  {name:<30} {est[:, 0].mean():>10.4f} {est[:, 0].std():>8.4f}")
    print("  → 前三种均值都 ≈ 精确值（无偏），标准差：好 baseline 最小、离谱 baseline 反而比不减更大；GRPO 均值缩成 (1 − 1/n) = 0.75 倍（有偏），RLOO 无偏")

    # GRPO 的偏差用精确枚举验证：G = 2
    print("\n  GRPO 偏差的精确验证（G = 2，枚举全部 100 种采样组合）：")
    pi = softmax_np(z0)
    E_grpo = np.zeros(V); E_rloo = np.zeros(V)
    for a in range(V):
        for b in range(V):
            w = pi[a] * pi[b]
            m = (R[a] + R[b]) / 2
            E_grpo += w * 0.5 * ((R[a] - m) * grad_logpi(z0, a) + (R[b] - m) * grad_logpi(z0, b))
            E_rloo += w * 0.5 * ((R[a] - R[b]) * grad_logpi(z0, a) + (R[b] - R[a]) * grad_logpi(z0, b))
    print(f"    精确梯度 ∂J/∂z₀ = {g_exact[0]:.4f}；GRPO 估计的期望 = {E_grpo[0]:.4f} = {E_grpo[0]/g_exact[0]:.2f} × 精确值；RLOO 估计的期望 = {E_rloo[0]:.4f}")

    # 训练曲线：三种方法各 300 步，20 个种子
    def train(method, steps=300, lr=0.5, G=4, seed=0, shift=0.0):
        r = np.random.default_rng(seed)
        z = np.zeros(V); curve = []
        for _ in range(steps):
            pi = softmax_np(z)
            curve.append(pi[R == 1].sum())
            ks = r.choice(V, size=G, p=pi); rs = R[ks] + shift
            if method == "reinforce":
                adv = rs
            elif method == "baseline":
                adv = rs - rs.mean()
            else:                                                   # grpo：再除以组内标准差
                adv = (rs - rs.mean()) / (rs.std() + 1e-8)
            g = np.mean([adv[i] * grad_logpi(z, ks[i]) for i in range(G)], axis=0)
            z += lr * g                                             # 最大化 J：沿 +∇ 走
        return np.array(curve)

    runs = [("reinforce", 0.0, "REINFORCE，奖励 0 / 1"), ("reinforce", 10.0, "REINFORCE，奖励 10 / 11"),
            ("baseline", 10.0, "减组内均值（奖励 10 / 11，与 0 / 1 一样）"), ("grpo", 10.0, "GRPO：再除以组内 std")]
    curves = {lab: np.stack([train(m, seed=s, shift=sh) for s in range(20)]) for m, sh, lab in runs}
    print("\n  训练 300 步（G = 4，lr = 0.5，20 个种子），P(偶数) 从 0.50 出发：")
    for lab, c in curves.items():
        reach = [np.argmax(cc >= 0.9) if (cc >= 0.9).any() else 300 for cc in c]
        bad = int((c[:, -1] < 0.1).sum())
        print(f"    {lab:<36} 300 步后均值 {c[:, -1].mean():.3f}，最差种子 {c[:, -1].min():.3f}，{bad} 个种子坍缩到奇数 token，到 0.9 平均 {np.mean(reach):.0f} 步（没到的按 300 算）")
    print("  → 奖励是 0/1 时不减 baseline 也能学（采到奇数不推、采到偶数推高）；奖励整体抬到 10/11，不减 baseline 就出事——每条采到的回答都被「奖励 10」猛推，前几步碰巧多采了哪个 token 就把概率全押上去，押到奇数就再也学不回来；减了均值就与抬不抬无关")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    bins = np.linspace(-0.6, 0.8, 60)
    for name, col in zip(list(variants)[:3], (C["blue"], C["green"], C["red"])):
        axes[0].hist(hist[name], bins=bins, histtype="step", lw=1.4, color=col, label=f"{name}  std {hist[name].std():.2f}")
    axes[0].axvline(g_exact[0], color="k", ls="--", lw=1)
    axes[0].text(g_exact[0] + 0.02, axes[0].get_ylim()[1] * 0.45, f"精确值 {g_exact[0]:.3f}", fontsize=8)
    axes[0].set(xlabel="∂J/∂z₀ 的一次估计（n = 4 条采样）", ylabel="次数（共 20000 次）", title="同一个梯度：三种估计的分布")
    axes[0].legend(fontsize=6.5, loc="upper left")
    for (lab, c), col in zip(curves.items(), (C["blue"], C["red"], C["green"], C["purple"])):
        mu = c.mean(0); sd = c.std(0)
        axes[1].plot(mu, color=col, label=lab)
        axes[1].fill_between(range(len(mu)), mu - sd, mu + sd, color=col, alpha=0.06 if "10 / 11" in lab and "REINFORCE" in lab else 0.15)
    axes[1].set(xlabel="步", ylabel="P(偶数 token)", title="训练曲线（20 个种子的均值 ± 标准差）", ylim=(0.45, 1.02))
    axes[1].legend(fontsize=6.5, loc="center right")
    save(fig, "07-policy-gradient-toy")


# ---------------------------------------------------------------- gd
def gd():
    section("梯度下降：L = θ₁² + 3θ₂²，从 (2, 1) 出发，三种学习率")
    def run(lr, steps=12, noise=0.0, seed=0):
        r = np.random.default_rng(seed)
        th = np.array([2.0, 1.0]); path = [th.copy()]
        for _ in range(steps):
            g = grad_bowl(*th) + noise * r.standard_normal(2)
            th = th - lr * g; path.append(th.copy())
        return np.array(path)
    for lr in (0.02, 0.15, 0.34):
        p = run(lr)
        print(f"  lr = {lr}: 12 步后 θ = ({p[-1, 0]:.3f}, {p[-1, 1]:.3f})，L = {L_bowl(*p[-1]):.3f}   θ₂ 轨迹前 4 步 {np.round(p[:4, 1], 2).tolist()}")
    print("  θ₂ 方向曲率 6，稳定要求 lr < 2/6 = 0.333：0.34 时 θ₂ 每步乘 (1 − 6·0.34) = −1.04，越跳越远")
    x = 1.0
    seq = [x]
    for _ in range(5):
        x = x - 2 * 2 * x; seq.append(x)
    print(f"  文中 L = x², η = 2 的例子：{seq}")

    t1, t2 = np.meshgrid(np.linspace(-2.6, 2.6, 200), np.linspace(-1.8, 1.8, 200))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax in axes:
        ax.contour(t1, t2, L_bowl(t1, t2), levels=[0.1, 0.5, 1, 2, 4, 7, 10], colors=C["light"], linewidths=0.8)
        ax.set(xlabel="θ₁", ylabel="θ₂", aspect="equal", xlim=(-2.6, 2.6), ylim=(-1.8, 1.8))
    for lr, col, lab in ((0.02, C["blue"], "lr 0.02：太小，12 步只走了一点"), (0.15, C["green"], "lr 0.15：合适"), (0.34, C["red"], "lr 0.34：太大，θ₂ 方向发散")):
        p = run(lr)
        axes[0].plot(p[:, 0], p[:, 1], "o-", color=col, ms=3, lw=1.2, label=lab)
    axes[0].plot(0, 0, "*", color="k", ms=8)
    axes[0].set_title("同一个起点，三种学习率")
    axes[0].legend(fontsize=6.5, loc="lower left")
    for noise, col, lab in ((0.0, C["green"], "全量梯度（无噪声）"), (1.5, C["orange"], "小 batch：噪声 σ=1.5"), (4.0, C["red"], "更小 batch：噪声 σ=4")):
        p = run(0.1, steps=40, noise=noise, seed=1)
        axes[1].plot(p[:, 0], p[:, 1], "-", color=col, lw=1.1, label=lab, alpha=0.9)
    axes[1].plot(0, 0, "*", color="k", ms=8)
    axes[1].set_title("随机梯度：batch 越小，每步的梯度越抖")
    axes[1].legend(fontsize=6.5, loc="lower left")
    save(fig, "07-gd-trajectories")


# ---------------------------------------------------------------- lagrange
def lagrange():
    section("拉格朗日乘子：在 x + y = 10 下最大化 xy")
    xs = np.linspace(0, 10, 11)
    print("  沿约束线走：x =", xs.astype(int).tolist())
    print("             xy =", (xs * (10 - xs)).astype(int).tolist(), "  ← x = y = 5 时最大")
    print("  最优点 (5, 5)：∇f = (y, x) = (5, 5)，∇g = (1, 1)，平行，λ = 5")
    x, y = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    cs = ax.contour(x, y, x * y, levels=[5, 10, 16, 21, 25, 30, 36], colors=C["gray"], linewidths=0.8)
    ax.clabel(cs, fmt="xy=%g", fontsize=7)
    ax.plot([0, 10], [10, 0], color=C["blue"], lw=2, label="约束 x + y = 10")
    ax.plot(5, 5, "o", color=C["red"], ms=6)
    ax.annotate("", (6.2, 6.2), (5, 5), arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.6))
    ax.annotate("", (5.8, 5.8), (5, 5), arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.6))
    ax.text(6.3, 6.4, "∇f = (5, 5)", color=C["red"], fontsize=8)
    ax.text(4.4, 6.5, "∇g = (1, 1)", color=C["blue"], fontsize=8)
    ax.text(5.3, 4.2, "(5, 5)：等高线 xy = 25\n与约束线相切，∇f ∥ ∇g", fontsize=8)
    ax.plot(2, 8, "s", color=C["orange"], ms=5)
    ax.text(2.5, 8.7, "(2, 8)：xy = 16，等高线穿过约束线\n沿线走还能变大", fontsize=7.5, color=C["orange"])
    ax.set(xlabel="x", ylabel="y", aspect="equal", xlim=(0, 10), ylim=(0, 10), title="带约束的极值：最优点处目标函数的等高线与约束线相切")
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "07-lagrange")


ALL = {"deriv": deriv, "grad": grad, "chain": chain, "smgrad": smgrad, "pg": pg, "gd": gd, "lagrange": lagrange}

if __name__ == "__main__":
    picks = [a for a in sys.argv[1:] if a in ALL] or list(ALL)
    for k in picks:
        ALL[k]()
