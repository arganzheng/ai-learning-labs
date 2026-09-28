"""算法工程师的数学（08）：统计推断与拟合——文中全部数字与图（模拟部分；实跑多种子见 08_multi_seed.py）。
https://arganzheng.life/statistical-inference-and-fitting-scaling-laws.html

    python 08_inference_and_fitting.py               # 全部，约 30 秒
    python 08_inference_and_fitting.py noise ci      # 只跑指定的段

段：noise（抽样噪声长什么样：同一个模型评 1000 次）· clt（中心极限定理：0/1 之和变成钟形）· ci（100 个置信区间；1.96 从哪来）
    · compare（两个模型怎么比：独立 vs 配对）· pvalue（置换检验）· lsq（最小二乘手算）· powerlaw（7 个真实模型的幂律拟合 + bootstrap 区间 + 外推）
    · chinchilla（等 loss 线与等算力线相切）
"""
import math
import os
import re
import sys

import numpy as np

from _plot import C, plt, save

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ---------------------------------------------------------------- noise
def noise():
    global rng
    rng = np.random.default_rng(0)
    section("抽样噪声：真实正确率 80% 的模型，在 n 道题上评 1000 次，每次的正确率各是多少")
    p = 0.8
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for ax, (n, name) in zip(axes, ((164, "HumanEval"), (1319, "GSM8K"), (14042, "MMLU"))):
        acc = rng.binomial(n, p, size=1000) / n
        se = math.sqrt(p * (1 - p) / n)
        print(f"  n = {n:>5}（{name}）: 1000 次评测的正确率 最低 {acc.min():.3f} 最高 {acc.max():.3f}，标准差 {acc.std():.4f}；公式 √(p(1−p)/n) = {se:.4f}")
        ax.hist(acc, bins=30, color=C["blue"], alpha=0.8)
        ax.axvline(p, color=C["red"], ls="--", lw=1)
        ax.set(title=f"{name}：n = {n}\n标准差 {acc.std():.3f}", xlabel="一次评测的正确率", xlim=(0.68, 0.92))
    axes[0].set_ylabel("次数（共 1000 次评测）")
    save(fig, "08-sampling-noise")
    print("  → 同一个模型、同样的真实水平，HumanEval 上一次评测能落在 0.71 到 0.89 之间；题数 ×4，宽度减半")


# ---------------------------------------------------------------- clt
def clt():
    global rng
    rng = np.random.default_rng(1)
    section("中心极限定理：n 个 0/1 的平均，n 越大越像钟形")
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3))
    p = 0.8
    for ax, n in zip(axes, (1, 3, 10, 100)):
        means = rng.binomial(n, p, size=20000) / n
        vals, cnts = np.unique(means, return_counts=True)
        ax.bar(vals, cnts / cnts.sum(), width=max(0.9 / n, 0.008), color=C["blue"])
        if n >= 10:
            x = np.linspace(0, 1, 300)
            sd = math.sqrt(p * (1 - p) / n)
            ax.plot(x, np.exp(-(x - p) ** 2 / (2 * sd ** 2)) / (sd * math.sqrt(2 * math.pi)) * (1 / n if n < 100 else 0.01), color=C["red"], lw=1)
        ax.set(title=f"n = {n}", xlabel="平均值", yticks=[])
        print(f"  n = {n:>3}: 平均值的标准差 {means.std():.3f}（公式 {math.sqrt(p*(1-p)/n):.3f}）")
    axes[0].set_ylabel("概率")
    save(fig, "08-clt")


# ---------------------------------------------------------------- ci
def ci():
    global rng
    rng = np.random.default_rng(2)
    section("置信区间：100 个区间里大约 95 个盖住真值；1.96 是标准正态两侧各留 2.5% 的位置")
    p, n = 0.8, 164
    hits = 0
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), width_ratios=[3, 2])
    for i in range(100):
        k = rng.binomial(n, p)
        ph = k / n
        se = math.sqrt(ph * (1 - ph) / n)
        lo, hi = ph - 1.96 * se, ph + 1.96 * se
        ok = lo <= p <= hi
        hits += ok
        axes[0].plot([lo, hi], [i, i], color=C["blue"] if ok else C["red"], lw=1)
        axes[0].plot(ph, i, ".", color=C["blue"] if ok else C["red"], ms=3)
    axes[0].axvline(p, color="k", ls="--", lw=1)
    axes[0].set(xlabel="正确率", ylabel="第几次评测", title=f"100 次评测各自的 95% 区间：{hits} 个盖住真值 0.8，{100-hits} 个（红）没盖住")
    print(f"  100 个区间里 {hits} 个盖住了真值 0.8")
    x = np.linspace(-4, 4, 400)
    pdf = np.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
    axes[1].plot(x, pdf, color=C["blue"])
    axes[1].fill_between(x, 0, pdf, where=(x > -1.96) & (x < 1.96), color=C["blue"], alpha=0.25)
    axes[1].fill_between(x, 0, pdf, where=(x <= -1.96) | (x >= 1.96), color=C["red"], alpha=0.35)
    axes[1].text(0, 0.15, "95%", ha="center", fontsize=11, color=C["blue"])
    axes[1].text(2.5, 0.06, "2.5%", fontsize=8, color=C["red"]); axes[1].text(-3.2, 0.06, "2.5%", fontsize=8, color=C["red"])
    axes[1].set_xticks([-1.96, 0, 1.96], ["−1.96", "0", "1.96"])
    axes[1].text(0, 0.02, "90% 是 ±1.64，99% 是 ±2.58", ha="center", fontsize=7, color=C["gray"])
    axes[1].set(title="标准正态：中间 95% 的边界是 ±1.96", xlabel="偏离均值几个标准差", yticks=[])
    save(fig, "08-confidence-intervals")
    for z, c in ((1.64, 90), (1.96, 95), (2.58, 99)):
        print(f"  ±{z} 个标准差盖住 {c}%")


# ---------------------------------------------------------------- compare
def simulate_pair(n, p1, p2, rho, trials, r):
    """两个模型在同一套 n 道题上的对错。rho 控制「错的题重叠多少」：用共享的题目难度实现。"""
    from scipy.stats import norm
    z1, z2 = norm.ppf(p1), norm.ppf(p2)
    shared = r.standard_normal((trials, n))
    e1 = math.sqrt(rho) * shared + math.sqrt(1 - rho) * r.standard_normal((trials, n))
    e2 = math.sqrt(rho) * shared + math.sqrt(1 - rho) * r.standard_normal((trials, n))
    return (e1 < z1), (e2 < z2)


def compare():
    global rng
    rng = np.random.default_rng(3)
    section("两个模型怎么比：真实正确率 80% vs 77%（差 3 个点），HumanEval 的 164 题")
    from scipy.stats import norm
    n, p1, p2, trials = 164, 0.80, 0.77, 5000
    a, b = simulate_pair(n, p1, p2, 0.0, trials, rng)
    d = a.mean(1) - b.mean(1)
    se_ind = np.sqrt(a.mean(1) * (1 - a.mean(1)) / n + b.mean(1) * (1 - b.mean(1)) / n)
    sig_ind = (d / se_ind > 1.96).mean()
    print(f"  独立比较（各评各的）：5000 次里观测到的差 平均 {d.mean():.3f}，标准差 {d.std():.3f}；'A 显著好于 B' 只有 {sig_ind:.0%} 的次数成立；有 {(d<0).mean():.0%} 的次数 B 反而看起来更高")
    print(f"  {'错题重叠程度 ρ':>14} {'平均分歧题数':>10} {'McNemar 判显著的比例':>18}")
    results = {}
    for rho in (0.0, 0.5, 0.8, 0.95):
        a, b = simulate_pair(n, p1, p2, rho, trials, rng)
        n01 = (a & ~b).sum(1); n10 = (~a & b).sum(1)
        disc = n01 + n10
        z = (n01 - n10) / np.sqrt(np.maximum(disc, 1))
        sig = (z > 1.96).mean()
        results[rho] = (disc.mean(), sig)
        print(f"  {rho:>14} {disc.mean():>10.1f} {sig:>18.0%}")
    print("  → 差 3 个点，独立比较在 164 题上基本分不出来；配对比较时，两个模型错的题越重叠（分歧题越少），同样 3 个点越容易显著")
    print("  McNemar 手算一例：A 独对 12 题、B 独对 7 题（都对 / 都错的题不算），z = (12 − 7) / √19 = 1.15，不显著；A 独对 8、B 独对 3：z = 5/√11 = 1.51，仍不显著；A 独对 5、B 独对 0：精确 p = 2 × 0.5^5 = 0.0625")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    axes[0].hist(d, bins=40, color=C["blue"], alpha=0.8)
    axes[0].axvline(0.03, color=C["red"], ls="--", lw=1); axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set(xlabel="一次评测里观测到的 A − B", ylabel="次数", title=f"独立比较：真实差 3 个点，观测到的差散布很宽\n{(d<0).mean():.0%} 的次数 B 看起来更高")
    rhos = list(results); axes[1].bar([str(r) for r in rhos], [results[r][1] for r in rhos], color=C["green"])
    axes[1].axhline(sig_ind, color=C["gray"], ls=":", lw=1); axes[1].text(1.5, 0.42, f"虚线：独立比较只有 {sig_ind:.0%}", fontsize=7.5, color=C["gray"], ha="center")
    for i, r in enumerate(rhos):
        axes[1].text(i, results[r][1] + 0.05, f"平均分歧\n{results[r][0]:.0f} 题", ha="center", fontsize=7)
    axes[1].set(xlabel="两个模型错题的重叠程度 ρ", ylabel="判为显著的比例", title="配对比较：分歧题越少越灵敏", ylim=(0, 1))
    save(fig, "08-independent-vs-paired")


# ---------------------------------------------------------------- pvalue
def pvalue():
    global rng
    rng = np.random.default_rng(4)
    section("p 值：置换检验——如果两个模型其实一样，随机打乱标签能有多常见地看到这么大的差")
    r = np.random.default_rng(1)
    a, b = simulate_pair(164, 0.80, 0.77, 0.8, 1, r)
    a, b = a[0], b[0]
    obs = a.mean() - b.mean()
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    print(f"  一次真实的评测：A 对 {a.sum()} 题（{a.mean():.3f}），B 对 {b.sum()} 题（{b.mean():.3f}），差 {obs:.3f}；A 独对 {n01} 题、B 独对 {n10} 题")
    perm = []
    for _ in range(10000):
        swap = r.random(164) < 0.5
        aa = np.where(swap, b, a); bb = np.where(swap, a, b)
        perm.append(aa.mean() - bb.mean())
    perm = np.array(perm)
    pv = (np.abs(perm) >= abs(obs) - 1e-12).mean()
    print(f"  随机交换每道题上两个模型的答案 10000 次（零假设：两个模型一样，谁对谁错可以互换）：差 ≥ {abs(obs):.3f} 的比例 = p 值 = {pv:.3f}")
    print(f"  读法：{'p < 0.05，差异不像噪声' if pv < 0.05 else 'p ≥ 0.05，这么大的差在「两个模型一样」时也常见，分不出'}")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.hist(perm, bins=50, color=C["light"], edgecolor=C["gray"], lw=0.3)
    ax.axvline(obs, color=C["red"], lw=1.5); ax.axvline(-obs, color=C["red"], lw=1.5, ls="--")
    ax.text(obs + 0.003, ax.get_ylim()[1] * 0.85, f"实际观测到的差 {obs:.3f}\n两侧更极端的占 {pv:.1%} = p 值", color=C["red"], fontsize=8)
    ax.set(xlabel="打乱标签后的 A − B", ylabel="次数（共 10000 次打乱）", title="置换检验：「两个模型一样好」时差值的分布")
    save(fig, "08-permutation-test")


# ---------------------------------------------------------------- lsq
def lsq():
    global rng
    rng = np.random.default_rng(5)
    section("最小二乘：5 个点手算直线")
    x = np.array([1, 2, 3, 4, 5.0]); y = np.array([2.5, 3.6, 6.4, 7.5, 10.4])
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    b = ym - a * xm
    print(f"  x = {x.tolist()}，y = {y.tolist()}")
    print(f"  x̄ = {xm}，ȳ = {ym:.2f}；Σ(x−x̄)(y−ȳ) = {((x-xm)*(y-ym)).sum():.2f}，Σ(x−x̄)² = {((x-xm)**2).sum():.2f}")
    print(f"  斜率 a = {a:.3f}，截距 b = ȳ − a·x̄ = {b:.3f}；numpy.polyfit 给 {np.polyfit(x, y, 1).round(3).tolist()}")
    res = y - (a * x + b)
    print(f"  残差 {res.round(3).tolist()}，平方和 {(res**2).sum():.4f}；换任何别的直线平方和都更大，比如 y = 1.6x + 1.4：{((y-1.6*x-1.4)**2).sum():.4f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    xs = np.linspace(0.5, 5.5, 50)
    ax.plot(xs, a * xs + b, color=C["blue"], label=f"最小二乘 y = {a:.2f}x + {b:.2f}，残差平方和 {(res**2).sum():.3f}")
    ax.plot(xs, 1.6 * xs + 1.4, color=C["gray"], ls="--", label=f"随手画的 y = 1.6x + 1.4，残差平方和 {((y-1.6*x-1.4)**2).sum():.3f}")
    ax.plot(x, y, "o", color=C["red"], zorder=3)
    for xi, yi, ri in zip(x, y, res):
        ax.plot([xi, xi], [yi, yi - ri], color=C["red"], lw=1)
        ax.text(xi + 0.06, yi - ri / 2, f"{ri:+.2f}", fontsize=7, color=C["red"])
    ax.set(xlabel="x", ylabel="y", title="最小二乘：让所有竖直距离（残差）的平方和最小")
    ax.legend(fontsize=7)
    save(fig, "08-least-squares")


# ---------------------------------------------------------------- powerlaw
def load_scaling():
    txt = open(os.path.join(HERE, "..", "transformer-and-llm", "expected", "scaling_law_fit.txt")).read()
    rows = re.findall(r"^\s+(\d+)\s+([\d,]+)\s+[\d.e+]+\s+([\d.]+)\s+\d+s$", txt, re.M)
    return np.array([int(r[1].replace(",", "")) for r in rows], float), np.array([float(r[2]) for r in rows])


def powerlaw():
    global rng
    rng = np.random.default_rng(6)
    section("幂律拟合：7 个真实训练的字符级模型（transformer-and-llm/scaling_law_fit.py 的实测）")
    N, L = load_scaling()
    print("  N（非 embedding 参数）:", [f"{int(n):,}" for n in N])
    print("  final loss            :", L.tolist())
    # 双对数直线：log L = c − α log N（没有 E 项的 Kaplan 形式）
    fitN, fitL = N[:6], L[:6]
    slope, icpt = np.polyfit(np.log(fitN), np.log(fitL), 1)
    print(f"  用前 6 个点在双对数坐标上拟直线：斜率 = {slope:.4f}（即 α = {-slope:.4f}），N 每 ×10，loss ×10^{slope:.3f} = {10**slope:.3f}")
    pred7 = math.exp(icpt + slope * math.log(N[6]))
    print(f"  外推到第 7 个模型 N = {int(N[6]):,}：预测 {pred7:.4f}，实测 {L[6]:.4f}，误差 {pred7-L[6]:+.4f}")
    # bootstrap：对残差重采样
    resid = np.log(fitL) - (icpt + slope * np.log(fitN))
    boots = []
    for _ in range(5000):
        yb = icpt + slope * np.log(fitN) + rng.choice(resid, size=6, replace=True)
        boots.append(np.polyfit(np.log(fitN), yb, 1))
    boots = np.array(boots)
    lo, hi = np.percentile(-boots[:, 0], [2.5, 97.5])
    print(f"  bootstrap 5000 次：α 的 95% 区间 [{lo:.4f}, {hi:.4f}]（点估计 {-slope:.4f}）")
    grid = np.logspace(np.log10(N[0]), np.log10(N[-1] * 10), 100)
    preds = np.exp(boots[:, 1][:, None] + boots[:, 0][:, None] * np.log(grid)[None, :])
    plo, phi = np.percentile(preds, [2.5, 97.5], axis=0)
    for target in (N[6], N[6] * 10):
        pr = np.exp(boots[:, 1] + boots[:, 0] * math.log(target))
        print(f"  外推到 N = {int(target):>10,}（{'实测点' if target == N[6] else '再 ×10，没有实测'}）：预测区间 [{np.percentile(pr, 2.5):.3f}, {np.percentile(pr, 97.5):.3f}]，宽 {np.percentile(pr, 97.5)-np.percentile(pr, 2.5):.3f}")
    print("  → 拟合范围内的区间很窄；越往外推区间越张开——scaling law 的常数是估计值，外推几个数量级时不确定性被放大")
    # 三参数形式 E + A/N^α（与 scaling_law_fit.py 一致）对照
    from scipy.optimize import curve_fit
    f3 = lambda n, E, A, al: E + A / n ** al
    (E, A, al), _ = curve_fit(f3, fitN, fitL, p0=(0.5, 5, 0.2), maxfev=20000)
    print(f"  对照：带不可约项的三参数拟合 E = {E:.3f}, A = {A:.2f}, α = {al:.3f}（scaling_law_fit.py 报 0.618 / 6.78 / 0.166）；两种形式在拟合范围内都贴合，外推时分歧——形式的选择本身也是不确定性的来源")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].plot(N, L, "o", color=C["red"], label="实测（7 个模型）")
    xs = np.logspace(4, 6.2, 100)
    axes[0].plot(xs, np.exp(icpt + slope * np.log(xs)), color=C["blue"], label=f"双对数直线拟合（前 6 点），斜率 {slope:.3f}")
    axes[0].set(xscale="log", yscale="log", xlabel="N（参数量，对数）", ylabel="loss（对数）", title="双对数坐标：幂律是直线")
    axes[0].legend(fontsize=7)
    axes[1].plot(N, L, "o", color=C["red"], zorder=3, label="实测")
    axes[1].plot(grid, np.exp(icpt + slope * np.log(grid)), color=C["blue"], label="拟合 / 外推")
    axes[1].fill_between(grid, plo, phi, color=C["blue"], alpha=0.2, label="bootstrap 95% 区间")
    axes[1].plot(grid, f3(grid, E, A, al), color=C["orange"], ls="--", lw=1.2, label=f"三参数 E + A/N^α（E = {E:.2f}）")
    axes[1].axvline(N[5], color=C["gray"], ls=":", lw=1); axes[1].text(N[5] * 0.9, 1.05, "拟合用到的范围 →|", fontsize=7, color=C["gray"], ha="right")
    axes[1].annotate(f"第 7 个模型实测 {L[6]:.3f}\n直线预测 {pred7:.3f}，区间外", (N[6], L[6]), (N[6] * 0.08, L[6] + 0.25), fontsize=7, color=C["red"], arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8))
    axes[1].set(xscale="log", xlabel="N（对数）", ylabel="loss", title="外推：噪声的区间很窄，公式形式的误差更大")
    axes[1].legend(fontsize=7, loc="upper right")
    save(fig, "08-powerlaw-bootstrap")


# ---------------------------------------------------------------- chinchilla
def chinchilla():
    global rng
    rng = np.random.default_rng(7)
    section("Chinchilla：L(N, D) = E + A/N^α + B/D^β 的等 loss 线与等算力线")
    E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
    Lf = lambda N, D: E + A / N ** al + B / D ** be
    for N, D, name in ((8e9, 160e9, "8B，160B token（20 倍）"), (8e9, 15e12, "8B，15T token（Llama-3 的量）"), (70e9, 1.4e12, "70B，1.4T（Chinchilla 自己）")):
        print(f"  {name:<26} A/N^α = {A/N**al:.3f}  B/D^β = {B/D**be:.3f}  L = {Lf(N, D):.3f}")
    n = np.logspace(8, 12, 300); d = np.logspace(9, 14, 300)
    NN, DD = np.meshgrid(n, d)
    LL = Lf(NN, DD)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    cs = ax.contour(NN, DD, LL, levels=[1.9, 1.95, 2.0, 2.1, 2.2, 2.4, 2.7, 3.0], colors=C["gray"], linewidths=0.8)
    ax.clabel(cs, fmt="L=%.2f", fontsize=7)
    for Cf, col in ((1e21, C["orange"]), (1e23, C["red"]), (1e25, C["purple"])):
        ax.plot(n, Cf / (6 * n), color=col, lw=1.5, label=f"等算力 6ND = {Cf:.0e} FLOPs")
        # 沿等算力线找最小 loss
        Ls = Lf(n, Cf / (6 * n)); k = Ls.argmin()
        ax.plot(n[k], Cf / (6 * n[k]), "o", color=col, ms=6)
        print(f"  算力 {Cf:.0e}：沿等算力线最小 loss 在 N ≈ {n[k]:.2e}, D ≈ {Cf/(6*n[k]):.2e}，D/N ≈ {Cf/(6*n[k]**2):.0f}，L = {Ls[k]:.3f}")
    ax.set(xscale="log", yscale="log", xlabel="N 参数量", ylabel="D 训练 token 数", ylim=(1e9, 1e14), title="等 loss 线（灰）与等算力线（彩）：最优点是两者相切处，各算力的最优点连成一条斜线")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "08-chinchilla-isoflop")
    print("  → 三个相切点近似在一条直线上：N 和 D 随算力同步增长（指数各约 0.5）。用这组常数解出的 D/N 远大于 20——文中「拟合结果的不确定性」一节说的就是这件事")


ALL = {"noise": noise, "clt": clt, "ci": ci, "compare": compare, "pvalue": pvalue, "lsq": lsq, "powerlaw": powerlaw, "chinchilla": chinchilla}

if __name__ == "__main__":
    picks = [a for a in sys.argv[1:] if a in ALL] or list(ALL)
    for k in picks:
        ALL[k]()
