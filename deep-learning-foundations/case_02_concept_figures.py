"""案例（深度学习基础 04）：初始化 / 归一化 / 残差的概念图与模拟数字。

https://arganzheng.life/initialization-normalization-and-residual.html

    python case_02_concept_figures.py

输出文章的七张概念图到 out/init-norm-*.svg，并打印正文引用的所有数字
（连乘因子、64 层方差模拟、LN/RMSNorm 手算例子、BN 统计量噪声）。
线型统一实线 2.2，只用 Okabe-Ito 颜色区分。
"""
import math

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from _plot import C, plt, save

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#7F7F7F"

BOX_FC = "#f4f7fb"
BOX_EC = "#7a8ca5"
LW = 2.2


def box(ax, center, text, width=6, height=5, fontsize=13, fc=BOX_FC,
        ec=BOX_EC, math=False):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.8", facecolor=fc, edgecolor=ec, linewidth=1.1,
    )
    ax.add_patch(patch)
    if math:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color=C["gray"], lw=1.2, style="->",
          connectionstyle="arc3"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": style, "color": color, "lw": lw,
                            "connectionstyle": connectionstyle})


def new_diagram(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def plot_chain():
    """FIGURE 1：一条链——前向方差连乘、反向 Jacobian 连乘、三种方案的位置。"""
    fig, ax = new_diagram((11, 6.6), (0, 108), (0, 100))

    # ---- 前向链（上）----
    y_fwd = 62
    ax.text(54, 97, "前向：方差是每层因子的连乘", ha="center", fontsize=14)
    ax.text(54, 91,
            r"$\mathrm{Var}(h_L) = \mathrm{Var}(x)\, a_1 a_2 \cdots a_L$",
            ha="center", fontsize=14)
    xs = {"x": 5, "f1": 15, "n1": 24, "h1": 31, "f2": 41, "n2": 50,
          "h2": 57, "dots": 66, "fL": 76, "nL": 85, "hL": 93}
    ax.text(xs["x"], y_fwd, "$x$", ha="center", va="center", fontsize=15)
    for key, label in (("f1", "$f_1$"), ("f2", "$f_2$"), ("fL", "$f_L$")):
        box(ax, (xs[key], y_fwd), label, width=7, height=6, fontsize=14)
    for key in ("n1", "n2", "nL"):
        box(ax, (xs[key], y_fwd), "Norm", width=7, height=6, fontsize=9,
            fc="#fdeee0", ec=ORANGE)
    ax.text(xs["h1"], y_fwd, "$h_1$", ha="center", va="center", fontsize=14)
    ax.text(xs["h2"], y_fwd, "$h_2$", ha="center", va="center", fontsize=14)
    ax.text(xs["dots"], y_fwd, "$\\cdots$", ha="center", va="center",
            fontsize=14)
    ax.text(xs["hL"], y_fwd, "$h_L$", ha="center", va="center", fontsize=15)
    half = {"x": 2, "h1": 2, "h2": 2, "dots": 2, "hL": 2.6}
    edges = [("x", "f1"), ("f1", "n1"), ("n1", "h1"), ("h1", "f2"),
             ("f2", "n2"), ("n2", "h2"), ("h2", "dots"), ("dots", "fL"),
             ("fL", "nL"), ("nL", "hL")]
    for a, b in edges:
        arrow(ax, (xs[a] + half.get(a, 3.6), y_fwd),
              (xs[b] - half.get(b, 3.6), y_fwd))
    for key, sub in (("f1", "$a_1$"), ("f2", "$a_2$"), ("fL", "$a_L$")):
        ax.text(xs[key], y_fwd - 6.4, f"$\\times$ {sub}", ha="center",
                va="top", fontsize=11, color=BLUE)

    # ---- 反向链（下）：箭头向左；每个因子被残差扩成 I + J^T ----
    y_bwd = 30
    ax.text(54, 47, "反向：梯度是 Jacobian 的连乘", ha="center", fontsize=14)
    ax.text(54, 41,
            r"$\partial L / \partial x \;=\; J_1^{\top} J_2^{\top} \cdots"
            r" J_L^{\top}\;\partial L / \partial h_L$",
            ha="center", fontsize=14)
    bx = {"gx": 5, "j1": 24, "dots": 41, "j2": 57, "jL": 76, "ghL": 93}
    for key, label in (("jL", "$J_L^{\\top}$"), ("j2", "$J_2^{\\top}$"),
                       ("j1", "$J_1^{\\top}$")):
        box(ax, (bx[key], y_bwd), "", width=9, height=10.5, fontsize=13)
        ax.text(bx[key], y_bwd + 1.9, label, ha="center", va="center",
                fontsize=13)
        ax.text(bx[key], y_bwd - 2.7,
                "→ " + label.replace("$J_", "$I+J_"),
                ha="center", va="center", fontsize=8, color=GREEN)
    ax.text(bx["ghL"], y_bwd, "$\\partial L / \\partial h_L$", ha="center",
            va="center", fontsize=12)
    ax.text(bx["dots"], y_bwd, "$\\cdots$", ha="center", va="center",
            fontsize=14)
    ax.text(bx["gx"], y_bwd, "$\\partial L / \\partial x$", ha="center",
            va="center", fontsize=12)
    bhalf = {"ghL": 5, "gx": 5, "dots": 2}
    bedges = [("ghL", "jL"), ("jL", "j2"), ("j2", "dots"), ("dots", "j1"),
              ("j1", "gx")]
    for a, b in bedges:
        arrow(ax, (bx[a] - bhalf.get(a, 4.6), y_bwd),
              (bx[b] + bhalf.get(b, 4.6), y_bwd))

    # ---- 三个方案的注解：标题与正文分行，拉开间距 ----
    ax.text(20, 82.5, "初始化", ha="center", fontsize=13, color=BLUE,
            weight="bold")
    ax.text(20, 79.3, "选 $W$ 的标准差，\n让 $t=0$ 时每个 $a_l \\approx 1$",
            ha="center", va="top", fontsize=10.5, color=BLUE)
    arrow(ax, (15.5, 73.5), (xs["f1"], y_fwd + 4.6), color=BLUE, style="->",
          connectionstyle="arc3,rad=0.2")
    arrow(ax, (24.5, 73.5), (xs["f2"], y_fwd + 4.6), color=BLUE, style="->",
          connectionstyle="arc3,rad=-0.2")

    ax.text(80, 82.5, "归一化", ha="center", fontsize=13, color=ORANGE,
            weight="bold")
    ax.text(80, 79.3, "每层之后把方差拉回 1，\n前向的连乘链每层被切断一次",
            ha="center", va="top", fontsize=10.5, color=ORANGE)
    arrow(ax, (74, 73.5), (xs["n2"], y_fwd + 4.6), color=ORANGE,
          connectionstyle="arc3,rad=0.2")
    arrow(ax, (86, 73.5), (xs["nL"], y_fwd + 4.6), color=ORANGE,
          connectionstyle="arc3,rad=-0.2")

    ax.text(54, 11, "残差", ha="center", fontsize=13, color=GREEN,
            weight="bold")
    ax.text(54, 8, "反向每个因子从 $J_l^{\\top}$ 变成 $I + J_l^{\\top}$，"
            "多一条不乘任何 $J$ 的直通路（$I$：单位矩阵）",
            ha="center", va="top", fontsize=10.5, color=GREEN)
    arrow(ax, (44, 12.5), (bx["j1"], y_bwd - 6.6), color=GREEN,
          connectionstyle="arc3,rad=-0.15")
    arrow(ax, (64, 12.5), (bx["jL"], y_bwd - 6.6), color=GREEN,
          connectionstyle="arc3,rad=0.15")

    save(fig, "init-norm-chain")


def plot_product():
    """FIGURE 2：a^L 对深度的指数敏感。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    L = np.arange(1, 129)
    curves = [
        (0.5, BLUE), (0.9, ORANGE), (1.0, GREY), (1.1, GREEN), (2.0, VERMILION),
    ]
    for a, color in curves:
        ax.plot(L, a ** L.astype(float), color=color, linewidth=LW,
                label=f"a = {a}")
    ax.set_yscale("log")
    ax.set_xlabel("Depth L", fontsize=15)
    ax.set_ylabel("$a^L$", fontsize=15)
    ax.set_title("Per-layer factor $a$: $a^L$ vs depth", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=True, framealpha=0.9, fontsize=12, loc="lower right")
    marks = [(64, 0.5, "0.5^64 ≈ 5e-20"), (128, 0.9, "0.9^128 ≈ 1.4e-6"),
             (128, 1.1, "1.1^128 ≈ 2e5")]
    color_of = {a: c for a, c in curves}
    for depth, a, text in marks:
        y = a ** depth
        ax.scatter([depth], [y], color=color_of[a], zorder=5)
        ax.annotate(text, (depth, y), xytext=(depth - 44, y * 30),
                    fontsize=11)
    print("  a^L 引用值：0.5^64 = %.2e，0.9^128 = %.2e，1.1^128 = %.2e"
          % (0.5 ** 64, 0.9 ** 128, 1.1 ** 128))
    save(fig, "init-norm-product")


def plot_init_variance():
    """FIGURE 3：64 层 Linear+ReLU，三种初始化方差下逐层激活 std。"""
    rng = np.random.default_rng(2026)
    n, depth, batch = 256, 64, 1024
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    runs = [
        ("$\\sigma_w^2 = 1/n$ (Xavier)", BLUE, 1.0),
        ("$\\sigma_w^2 = 2/n$ (Kaiming)", GREEN, 2.0),
        ("$\\sigma_w^2 = 3/n$", VERMILION, 3.0),
    ]
    results = {}
    for label, color, factor in runs:
        h = rng.standard_normal((batch, n))
        stds = []
        for _ in range(depth):
            W = rng.standard_normal((n, n)) * math.sqrt(factor / n)
            h = np.maximum(h @ W, 0.0)
            stds.append(h.std())
        results[factor] = stds
        ax.plot(np.arange(1, depth + 1), stds, color=color, linewidth=LW,
                label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=15)
    ax.set_ylabel("activation std", fontsize=15)
    ax.set_title("64-layer Linear+ReLU, width 256: std per layer", fontsize=15)
    ax.axhline(1.0, color=GREY, linewidth=1.0)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=True, framealpha=0.9, fontsize=12, loc="lower left")
    save(fig, "init-norm-init-variance")
    print("  模拟（seed 2026，width 256，batch 1024，Linear+ReLU 64 层）：")
    for factor in (1.0, 2.0, 3.0):
        stds = results[factor]
        print(f"    σ_w²={factor:g}/n：层 1 std {stds[0]:.3g}，"
              f"层 64 std {stds[-1]:.3g}")
    print(f"    理论：σ_w²=1/n 时 std 每层乘 √0.5，64 层 = 2^-32 "
          f"= {2 ** -32:.2e}")


def plot_norm_axes():
    """FIGURE 4：BN / LN / RMSNorm 各自沿哪个维度统计。"""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    rows, cols = 4, 8
    titles = ["BatchNorm", "LayerNorm", "RMSNorm"]
    notes = [
        "one feature across\nall m samples",
        "one sample across\nall d features",
        "one sample across all d features;\nno mean subtraction",
    ]
    for ax, title, note in zip(axes, titles, notes):
        ax.set_xlim(-0.6, cols + 0.6)
        ax.set_ylim(-0.6, rows + 0.9)
        ax.axis("off")
        ax.set_title(title, fontsize=15)
        for i in range(rows):
            for j in range(cols):
                ax.add_patch(Rectangle((j, rows - 1 - i), 1, 1,
                                       facecolor="white",
                                       edgecolor="#b9c4d1", linewidth=0.8))
        if title == "BatchNorm":
            ax.add_patch(Rectangle((2, 0), 1, rows, facecolor=ORANGE,
                                   alpha=0.30, edgecolor=ORANGE, linewidth=2))
        else:
            ax.add_patch(Rectangle((0, rows - 2), cols, 1, facecolor=GREEN,
                                   alpha=0.30, edgecolor=GREEN, linewidth=2))
        ax.text(cols / 2, -0.45, "features →", ha="center", va="top",
                fontsize=11)
        ax.text(-0.45, rows / 2, "samples", ha="center", va="center",
                fontsize=11, rotation=90)
        ax.text(cols / 2, -1.0, note, ha="center", va="top",
                fontsize=10, color="#555555")
    save(fig, "init-norm-axes")


def plot_ln_vs_rms():
    """FIGURE 5：x=[2,4,6,8] 上 LN 与 RMSNorm 的分组柱状图。"""
    x = np.array([2.0, 4.0, 6.0, 8.0])
    ln = (x - x.mean()) / x.std()
    rms = x / math.sqrt((x ** 2).mean())
    print("  手算 x=[2,4,6,8]：LN 均值 %.3g 方差 %.3g → %s" %
          (x.mean(), x.var(), np.array2string(ln, precision=3)))
    print("    RMS = %.4g → %s" %
          (math.sqrt((x ** 2).mean()), np.array2string(rms, precision=3)))
    x0 = np.array([-3.0, -1.0, 1.0, 3.0])
    ln0 = (x0 - x0.mean()) / x0.std()
    rms0 = x0 / math.sqrt((x0 ** 2).mean())
    print("  对照 x=[-3,-1,1,3]（均值 0）：LN → %s；RMSNorm → %s（相同）"
          % (np.array2string(ln0, precision=3),
             np.array2string(rms0, precision=3)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = np.arange(4)
    width = 0.26
    ax.bar(groups - width, x, width, color=GREY, label="original x")
    ax.bar(groups, ln, width, color=ORANGE, label="LayerNorm")
    ax.bar(groups + width, rms, width, color=GREEN, label="RMSNorm")
    ax.axhline(0, color=C["gray"], linewidth=0.7)
    ax.set_xticks(groups)
    ax.set_xticklabels(["$x_1$", "$x_2$", "$x_3$", "$x_4$"], fontsize=14)
    ax.set_ylabel("value", fontsize=15)
    ax.set_title("LayerNorm vs RMSNorm on x = [2, 4, 6, 8]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(frameon=True, framealpha=0.9, fontsize=12)
    ax.grid(alpha=0.2, axis="y")
    save(fig, "init-norm-ln-vs-rms")


def plot_bn_sequence():
    """FIGURE 6：BN 统计量跨 batch×位置（含 padding）；均值估计噪声 ∝1/√m。"""
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.4])

    lengths = [6, 4, 2]
    for k, (ax_i, title, mode) in enumerate(
            [(0, "BatchNorm", "bn"), (1, "LayerNorm", "ln")]):
        ax = fig.add_subplot(gs[0, ax_i])
        rows, cols = 3, 6
        ax.set_xlim(-0.4, cols + 0.4)
        ax.set_ylim(-1.4, rows + 0.7)
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        for i in range(rows):
            for j in range(cols):
                padded = j >= lengths[i]
                ax.add_patch(Rectangle(
                    (j, rows - 1 - i), 1, 1,
                    facecolor="#e8e8e8" if padded else "white",
                    edgecolor="#b9c4d1", linewidth=0.8))
        if mode == "bn":
            ax.add_patch(Rectangle((-0.15, -0.15), cols + 0.3, rows + 0.3,
                                   fill=False, edgecolor=ORANGE,
                                   linewidth=2.4))
            ax.text(cols / 2, -0.75, "one feature pooled over\n"
                    "batch × positions, padding included",
                    ha="center", va="top", fontsize=9.5, color="#555555")
        else:
            for i in range(rows):
                for j in range(lengths[i]):
                    ax.add_patch(Rectangle(
                        (j - 0.06, rows - 1 - i - 0.06), 1.12, 1.12,
                        fill=False, edgecolor=GREEN, linewidth=1.8))
            ax.text(cols / 2, -0.75, "each token normalized\n"
                    "over its own features",
                    ha="center", va="top", fontsize=9.5, color="#555555")
        ax.text(-0.2, rows - 0.5, "s1", ha="right", va="center", fontsize=9,
                color="#555555")
        ax.text(-0.2, rows - 1.5, "s2", ha="right", va="center", fontsize=9,
                color="#555555")
        ax.text(-0.2, rows - 2.5, "s3", ha="right", va="center", fontsize=9,
                color="#555555")

    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(7)
    ms = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    sim = []
    for m in ms:
        means = rng.standard_normal((4000, m)).mean(axis=1)
        sim.append(means.std())
    ax.plot(ms, 1 / np.sqrt(ms.astype(float)), color=GREY, linewidth=LW,
            label="theory $1/\\sqrt{m}$")
    ax.plot(ms, sim, color=BLUE, linewidth=LW, label="simulated std of "
            "batch mean")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms], fontsize=10)
    ax.set_xlabel("batch size m", fontsize=13)
    ax.set_ylabel("std of batch mean", fontsize=13)
    ax.set_title("BN statistics are noisy at small m", fontsize=14)
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=True, framealpha=0.9, fontsize=10.5, loc="lower left")
    ax.annotate("inference: batch = 1, one token\n"
                "→ must reuse training running stats",
                xy=(1, 1.0), xytext=(8, 0.7), fontsize=9.5, color="#555555",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1})
    save(fig, "init-norm-bn-sequence")
    for m, s in zip(ms, sim):
        if m in (2, 32, 256):
            print(f"  batch-mean std @ m={m}: sim {s:.3f}，"
                  f"理论 1/√m = {1 / math.sqrt(m):.3f}")


def plot_residual_paths():
    """FIGURE 7：两个残差块展开成 4 条梯度路径，恒等路高亮。"""
    fig, ax = new_diagram((11, 4.6), (0, 100), (0, 100))
    y = 52
    xs = {"x0": 8, "x1": 50, "x2": 92}
    for key, label in (("x0", "$x_0$"), ("x1", "$x_1$"), ("x2", "$x_2$")):
        ax.text(xs[key], y, label, ha="center", va="center", fontsize=16)
    box(ax, (29, y - 22), "$f_1$", width=8, height=8, fontsize=15)
    box(ax, (71, y - 22), "$f_2$", width=8, height=8, fontsize=15)

    # 恒等路（上弧，绿色高亮）
    arrow(ax, (xs["x0"] + 3, y + 4), (xs["x1"] - 3, y + 4), color=GREEN,
          lw=LW, connectionstyle="arc3,rad=-0.45")
    arrow(ax, (xs["x1"] + 3, y + 4), (xs["x2"] - 3, y + 4), color=GREEN,
          lw=LW, connectionstyle="arc3,rad=-0.45")
    ax.text(29, y + 27, "$I$", ha="center", fontsize=15, color=GREEN)
    ax.text(71, y + 27, "$I$", ha="center", fontsize=15, color=GREEN)

    # 穿过 f_l 的路（下弧，灰）
    arrow(ax, (xs["x0"] + 3, y - 4), (24.5, y - 22), color=GREY, lw=LW,
          connectionstyle="arc3,rad=-0.3")
    arrow(ax, (33.5, y - 22), (xs["x1"] - 3, y - 4), color=GREY, lw=LW,
          connectionstyle="arc3,rad=-0.3")
    arrow(ax, (xs["x1"] + 3, y - 4), (66.5, y - 22), color=GREY, lw=LW,
          connectionstyle="arc3,rad=-0.3")
    arrow(ax, (75.5, y - 22), (xs["x2"] - 3, y - 4), color=GREY, lw=LW,
          connectionstyle="arc3,rad=-0.3")
    ax.text(29, y - 36, "$J_1$", ha="center", fontsize=14, color=GREY)
    ax.text(71, y - 36, "$J_2$", ha="center", fontsize=14, color=GREY)

    ax.text(50, 90, "Two residual blocks: the backward product unrolls "
            "into $2^2 = 4$ paths", ha="center", fontsize=14)
    ax.text(50, 10,
            r"$\partial x_2 / \partial x_0 = (I + J_2)(I + J_1)"
            r" = I \;+\; J_1 \;+\; J_2 \;+\; J_2 J_1$",
            ha="center", fontsize=15)
    ax.text(50, 4, "the green term is pure identity: a path that reaches "
            "$x_0$ without multiplying any $J$",
            ha="center", fontsize=11, color=GREEN)
    save(fig, "init-norm-residual-paths")


def main():
    plot_chain()
    plot_product()
    plot_init_variance()
    plot_norm_axes()
    plot_ln_vs_rms()
    plot_bn_sequence()
    plot_residual_paths()


if __name__ == "__main__":
    main()
