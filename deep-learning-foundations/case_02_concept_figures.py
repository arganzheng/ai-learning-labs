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
    label = ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return patch, label


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
    """FIGURE 1：一段前向链、一段反向链，三种方案各作用在哪一环。"""
    fig, ax = new_diagram((12, 8.6), (0, 144), (8, 131))
    X = 72  # 水平中点

    patches, inner = [], []

    def bx_(center, text, **kw):
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)

    # ---- 面板 A：前向，方差逐层乘 a_l；两行对照 ----
    ax.text(X, 129, "前向：方差逐层乘 $a_l$", ha="center", fontsize=18,
            weight="bold")
    ax.text(X, 123,
            r"$\mathrm{Var}(h_L) = \mathrm{Var}(x)\, a_1 a_2 \cdots a_L$",
            ha="center", fontsize=17)

    # 行 1：只靠初始化
    y1 = 114
    ax.text(13, y1, "只靠初始化", ha="right", va="center", fontsize=13,
            color=BLUE)
    # 与行 2 共用列网格：x | f_1 | 归一化槽位(空) | h_1 | ⋯ | f_L | 槽位(空) | h_L
    xs = {"x": 18, "f1": 28, "h1": 72, "dots": 80, "fL": 92, "hL": 136}
    ax.text(xs["x"], y1, "$x$", ha="center", va="center", fontsize=18)
    for key, label in (("f1", "$f_1$"), ("fL", "$f_L$")):
        bx_((xs[key], y1), label, width=7, height=6, fontsize=17)
        ax.text(xs[key], y1 - 8, "Linear + 非线性", ha="center",
                va="top", fontsize=13, color=GREY)
    ax.text(xs["h1"], y1, "$h_1$", ha="center", va="center", fontsize=17)
    ax.text(xs["dots"], y1, "$\\cdots$", ha="center", va="center",
            fontsize=17)
    ax.text(xs["hL"], y1, "$h_L$", ha="center", va="center", fontsize=18)
    half = {"x": 2, "h1": 2, "dots": 2, "hL": 2.6}
    edges = [("x", "f1"), ("f1", "h1"), ("h1", "dots"), ("dots", "fL"),
             ("fL", "hL")]
    for a, b in edges:
        arrow(ax, (xs[a] + half.get(a, 3.6), y1),
              (xs[b] - half.get(b, 3.6), y1))
    for a, sub in (("f1", "$a_1$"), ("fL", "$a_L$")):
        mid = (xs[a] + 3.6 + xs[{"f1": "h1", "fL": "hL"}[a]] - 2.6) / 2
        ax.text(mid, y1 - 2.6, "方差 $\\times$ " + sub, ha="center",
                va="top", fontsize=13)

    # 初始化注解（蓝）：在 a_1 标签正下方，垂直箭头指到标签底下
    a1_mid = (xs["f1"] + 3.6 + xs["h1"] - 2.6) / 2
    ax.text(a1_mid, 100, "初始化", ha="center", fontsize=16, color=BLUE,
            weight="bold")
    ax.text(a1_mid, 95.5, "选 $W$ 的标准差，\n让 $t=0$ 时每个 $a_l \\approx 1$",
            ha="center", va="top", fontsize=13, color=BLUE)
    arrow(ax, (a1_mid, 102.6), (a1_mid, y1 - 7.2), color=BLUE)

    # 行 2：加归一化
    y2 = 80
    ax.text(13, y2, "加归一化", ha="right", va="center", fontsize=13,
            color=ORANGE)
    xs2 = {"x": 18, "f1": 28, "n1": 50, "h1": 72, "dots": 80, "fL": 92,
           "nL": 114, "hL": 136}
    ax.text(xs2["x"], y2, "$x$", ha="center", va="center", fontsize=18)
    for key, label in (("f1", "$f_1$"), ("fL", "$f_L$")):
        bx_((xs2[key], y2), label, width=7, height=6, fontsize=17)
        ax.text(xs2[key], y2 - 8, "Linear + 非线性", ha="center",
                va="top", fontsize=13, color=GREY)
    for key in ("n1", "nL"):
        bx_((xs2[key], y2), "归一化", width=10, height=6, fontsize=15,
            fc="#fdeee0", ec=ORANGE)
    ax.text(xs2["n1"], y2 - 8, "BN / LN / RMSNorm", ha="center",
            va="top", fontsize=13, color=ORANGE)
    ax.text(xs2["h1"], y2, "$h_1$", ha="center", va="center", fontsize=17)
    ax.text(xs2["dots"], y2, "$\\cdots$", ha="center", va="center",
            fontsize=17)
    ax.text(xs2["hL"], y2, "$h_L$", ha="center", va="center", fontsize=18)
    h2 = {"x": 2, "h1": 2, "dots": 2, "hL": 2.6, "n1": 5.8, "nL": 5.8}
    edges2 = [("x", "f1"), ("f1", "n1"), ("n1", "h1"), ("h1", "dots"),
              ("dots", "fL"), ("fL", "nL"), ("nL", "hL")]
    for a, b in edges2:
        arrow(ax, (xs2[a] + h2.get(a, 3.6), y2),
              (xs2[b] - h2.get(b, 3.6), y2))
    for a, b in (("f1", "n1"), ("fL", "nL")):
        mid = (xs2[a] + 3.6 + xs2[b] - 5.8) / 2
        sub = "$a_1$" if a == "f1" else "$a_L$"
        ax.text(mid, y2 - 2.6, "方差 $\\times$ " + sub, ha="center",
                va="top", fontsize=13)
    for a, b in (("n1", "h1"), ("nL", "hL")):
        mid = (xs2[a] + 5.8 + xs2[b] - 2.6) / 2
        ax.text(mid, y2 - 2.6, "方差回到 1", ha="center", va="top",
                fontsize=13, color=ORANGE)

    ax.text(X, 66, "初始化是每个网络都要做的一步，只决定 $a_l$ 的起点；\n"
            "归一化是另外加进结构的部件。Transformer 实际用 Pre-Norm——"
            "归一化放在 $f_l$ 之前、残差分支里（见第六章）",
            ha="center", va="top", fontsize=13, color="#555555")

    # ---- 面板 B：反向，Jacobian 连乘；两行对照 ----
    ax.text(X, 53, "反向：梯度是 Jacobian 的连乘", ha="center",
            fontsize=18, weight="bold")
    ax.text(X, 47,
            r"$\partial L / \partial x \;=\; J_1^{\top} J_2^{\top} \cdots"
            r" J_L^{\top}\;\partial L / \partial h_L$",
            ha="center", fontsize=17)
    bxs = {"gx": 18, "j1": 50, "dots": 79, "jL": 108, "ghL": 136}
    ax.text(13, 36, "无残差", ha="right", va="center", fontsize=13,
            color=GREY)
    for key, label in (("jL", "$J_L^{\\top}$"), ("j1", "$J_1^{\\top}$")):
        bx_((bxs[key], 36), label, width=9, height=6, fontsize=16)
    ax.text(bxs["ghL"], 36, "$\\partial L / \\partial h_L$", ha="center",
            va="center", fontsize=15)
    ax.text(bxs["dots"], 36, "$\\cdots$", ha="center", va="center",
            fontsize=17)
    ax.text(bxs["gx"], 36, "$\\partial L / \\partial x$", ha="center",
            va="center", fontsize=15)
    bhalf = {"ghL": 5, "gx": 5, "dots": 2}
    bedges = [("ghL", "jL"), ("jL", "dots"), ("dots", "j1"), ("j1", "gx")]
    for a, b in bedges:
        arrow(ax, (bxs[a] - bhalf.get(a, 4.6), 36),
              (bxs[b] + bhalf.get(b, 4.6), 36))
    ax.text(13, 22, "有残差", ha="right", va="center", fontsize=13,
            color=GREEN)
    for key, label in (("jL", "$I+J_L^{\\top}$"), ("j1", "$I+J_1^{\\top}$")):
        bx_((bxs[key], 22), label, width=12, height=6, fontsize=15,
            fc="#e6f4ee", ec=GREEN)
    ax.text(bxs["ghL"], 22, "$\\partial L / \\partial h_L$", ha="center",
            va="center", fontsize=15)
    ax.text(bxs["dots"], 22, "$\\cdots$", ha="center", va="center",
            fontsize=17)
    ax.text(bxs["gx"], 22, "$\\partial L / \\partial x$", ha="center",
            va="center", fontsize=15)
    bhalf2 = {"ghL": 5, "gx": 5, "dots": 2, "j1": 6.2, "jL": 6.2}
    for a, b in bedges:
        arrow(ax, (bxs[a] - bhalf2.get(a, 4.6), 22),
              (bxs[b] + bhalf2.get(b, 4.6), 22), color=GREEN)
    ax.text(X, 13, "残差让反向每个因子多一项 $I$：连乘展开后总有一条不乘"
            "任何 $J$ 的直通路（$I$ 是单位矩阵）",
            ha="center", va="top", fontsize=13, color=GREEN)

    # ---- 布局自检：标签与框、标签与标签之间不得重叠 ----
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    box_exts = [p.get_window_extent(rend) for p in patches]
    labels = [t for t in ax.texts if t not in inner and t.get_text()]
    lab_exts = [t.get_window_extent(rend) for t in labels]

    def padded(e, pad=2):
        return Bbox.from_extents(e.x0 - pad, e.y0 - pad,
                                 e.x1 + pad, e.y1 + pad)

    for t, e in zip(labels, lab_exts):
        for b in box_exts:
            assert not padded(e).overlaps(b), (
                f"label overlaps box: {t.get_text()!r}")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert not padded(lab_exts[i]).overlaps(lab_exts[j]), (
                f"labels overlap: {labels[i].get_text()!r} / "
                f"{labels[j].get_text()!r}")
    for p, t in zip(patches, inner):
        pe = p.get_window_extent(rend)
        te = t.get_window_extent(rend)
        assert (te.x0 >= pe.x0 + 2 and te.x1 <= pe.x1 - 2
                and te.y0 >= pe.y0 + 2 and te.y1 <= pe.y1 - 2), (
                f"text overflows its box: {t.get_text()!r}")

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


def plot_one_layer_variance():
    """FIGURE 8：一层 Linear + ReLU 里方差怎么变（对应 a_l = n_in σ_w² c_φ）。"""
    rng = np.random.default_rng(20261014)
    x = rng.normal(0.0, 1.0, (4096, 256))
    W = rng.normal(0.0, math.sqrt(2.0 / 256), (256, 256))
    z = x @ W.T
    h = np.maximum(z, 0.0)

    var_x = float(x.var())
    var_z = float(z.var())
    e_h2 = float((h ** 2).mean())

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), layout="none")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.80, bottom=0.18,
                        wspace=0.55)
    data = [
        (x.ravel(), BLUE, f"输入 $x$：Var = {var_x:.2f}"),
        (z.ravel(), ORANGE, f"Linear 之后 $z$：Var = {var_z:.2f}"),
        (h.ravel(), GREEN,
         f"ReLU 之后 $h$：$\\mathbb{{E}}[h^2] = {e_h2:.2f}$"),
    ]
    bars3 = None
    for ax, (v, color, title) in zip(axes, data):
        _, _, patches = ax.hist(v, bins=61, color=color, edgecolor=color,
                                linewidth=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("取值", fontsize=9)
        if ax is axes[2]:
            bars3 = patches

    # 第 3 幅：0 处的单根柱是 ReLU 置零的那一半
    ax3 = axes[2]
    zero_bar = max(bars3, key=lambda p: p.get_height())
    ax3.annotate("一半置 0",
                 xy=(zero_bar.get_x() + zero_bar.get_width() / 2,
                     zero_bar.get_height()),
                 xytext=(0.55, 0.75), textcoords="axes fraction",
                 fontsize=9, color=GREY, ha="center", va="center",
                 arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.1})

    fig.canvas.draw()
    for a, b, lab in ((axes[0], axes[1], "$\\times\\ n_{in}\\sigma_w^2$"),
                      (axes[1], axes[2], "$\\times\\ c_{\\mathrm{ReLU}}=1/2$")):
        pa, pb = a.get_position(), b.get_position()
        y = (pa.y0 + pa.y1) / 2
        a.annotate("", xy=(pb.x0 - 0.012, y), xytext=(pa.x1 + 0.004, y),
                   xycoords=fig.transFigure, textcoords=fig.transFigure,
                   annotation_clip=False,
                   arrowprops={"arrowstyle": "->", "color": C["gray"],
                               "lw": 1.4})
        fig.text((pa.x1 + pb.x0) / 2, y + 0.045, lab,
                 ha="center", va="bottom", fontsize=9, color="#444444")

    # ---- 布局自检：文字之间不得互相压 ----
    from matplotlib.transforms import Bbox
    rend = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in axes:
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += [t for t in ax.texts if t.get_text()]
    texts = [t for t in texts if t.get_text()]
    exts = [t.get_window_extent(rend) for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            e_i = Bbox.from_extents(exts[i].x0 - 2, exts[i].y0 - 2,
                                    exts[i].x1 + 2, exts[i].y1 + 2)
            assert not e_i.overlaps(exts[j]), (
                f"labels overlap: {texts[i].get_text()!r} / "
                f"{texts[j].get_text()!r}")

    save(fig, "init-norm-one-layer")
    print(f"  一层模拟（n_in=256，σ_w²=2/256，4096×256）："
          f"Var(x) = {var_x:.2f}，Var(z) = {var_z:.2f}，"
          f"E[h²] = {e_h2:.2f}")


def plot_one_layer_steps():
    """FIGURE 9：一层方差放大的两步示意（对应 §1「1、前向方差逐层相乘」）。"""
    fig, ax = new_diagram((9.0, 3.0), (0, 120), (0, 40))
    patches, inner = [], []

    def bx_(center, text, **kw):
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)
        return p, t

    # 输入列
    in_x, sum_x = 12, 58
    ys = [32, 24, 16, 8]
    labels_in = [r"$x_1$", r"$x_2$", "$\\vdots$", r"$x_{n_{in}}$"]
    for i, (y, lab) in enumerate(zip(ys, labels_in)):
        if i == 2:
            ax.text(in_x, y, lab, ha="center", va="center", fontsize=11)
        else:
            c = plt.Circle((in_x, y), 2.0, facecolor=BOX_FC,
                           edgecolor=BOX_EC, linewidth=1.1, zorder=3)
            ax.add_patch(c)
            patches.append(c)
            inner.append(ax.text(in_x, y, lab, ha="center", va="center",
                                 fontsize=10, zorder=4))
    ax.text(in_x, 37.5, "$n_{in}$ 个输入，每个方差 $\\sigma_x^2$",
            ha="center", va="center", fontsize=9)

    # 加权连线 → Σ
    sum_y = 20
    sigma = plt.Circle((sum_x, sum_y), 2.6, facecolor=BOX_FC,
                       edgecolor=BOX_EC, linewidth=1.1, zorder=3)
    ax.add_patch(sigma)
    patches.append(sigma)
    inner.append(ax.text(sum_x, sum_y, "$\\Sigma$", ha="center",
                         va="center", fontsize=12, zorder=4))
    wlabs = [r"$W_{i1}$", r"$W_{i2}$", None, r"$W_{i,n_{in}}$"]
    for i, y in enumerate(ys):
        if i == 2:
            continue
        ax.plot([in_x + 2.1, sum_x - 2.7], [y, sum_y], color=C["gray"],
                lw=1.2, solid_capstyle="round", zorder=1)
        if wlabs[i]:
            mx, my = (in_x + 2.1 + sum_x - 2.7) / 2, (y + sum_y) / 2
            ax.text(mx, my + 1.6, wlabs[i], ha="center", va="center",
                    fontsize=9)
    ax.text(34, 36.2, "每一项 $W_{ij}x_j$：方差 $\\sigma_w^2\\sigma_x^2$",
            ha="center", va="center", fontsize=9)

    # Σ → z_i → φ → h_i
    z_x, phi_x, h_x = 72, 90, 110
    arrow(ax, (sum_x + 2.7, sum_y), (z_x - 3.6, sum_y))
    ax.text(z_x, sum_y, "$z_i$", ha="center", va="center", fontsize=11)
    ax.text((sum_x + z_x) / 2, sum_y - 5.2,
            "$n_{in}$ 项相加：Var$(z_i)=n_{in}\\sigma_w^2\\sigma_x^2$",
            ha="center", va="center", fontsize=9)
    arrow(ax, (z_x + 3.6, sum_y), (phi_x - 6.4, sum_y))
    bx_((phi_x, sum_y), "$\\phi$（如 ReLU）", width=13, height=6,
        fontsize=10, fc="#fdeee0", ec=ORANGE)
    arrow(ax, (phi_x + 6.4, sum_y), (h_x - 3.6, sum_y))
    ax.text(h_x, sum_y, "$h_i$", ha="center", va="center", fontsize=11)
    ax.text((phi_x + h_x) / 2, sum_y - 8.2,
            "二阶矩 $\\times\\,c_\\phi$（ReLU：$1/2$）",
            ha="center", va="center", fontsize=9)

    # 底部两条彩色带 + 总结
    ax.plot([8, 74], [2.5, 2.5], color=ORANGE, lw=2.2)
    ax.text(41, 0.2, "第一步 Linear：$\\times\\ n_{in}\\sigma_w^2$",
            ha="center", va="top", fontsize=10, color=ORANGE)
    ax.plot([76, 112], [2.5, 2.5], color=GREEN, lw=2.2)
    ax.text(94, 0.2, "第二步 激活：$\\times\\ c_\\phi$",
            ha="center", va="top", fontsize=10, color=GREEN)
    ax.text(60, -4.6,
            "一层合起来：$a_l=n_{in}\\sigma_w^2 c_\\phi$",
            ha="center", va="top", fontsize=10, color="black")

    ax.set_ylim(-8, 40)

    # ---- 布局自检：标签与框/圆、标签与标签之间不得重叠 ----
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    box_exts = [p.get_window_extent(rend) for p in patches]
    labels = [t for t in ax.texts if t not in inner and t.get_text()]
    lab_exts = [t.get_window_extent(rend) for t in labels]

    def padded(e, pad=2):
        return Bbox.from_extents(e.x0 - pad, e.y0 - pad,
                                 e.x1 + pad, e.y1 + pad)

    for t, e in zip(labels, lab_exts):
        for b in box_exts:
            assert not padded(e).overlaps(b), (
                f"label overlaps box: {t.get_text()!r}")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert not padded(lab_exts[i]).overlaps(lab_exts[j]), (
                f"labels overlap: {labels[i].get_text()!r} / "
                f"{labels[j].get_text()!r}")
    for p, t in zip(patches, inner):
        pe = p.get_window_extent(rend)
        te = t.get_window_extent(rend)
        assert (te.x0 >= pe.x0 + 1 and te.x1 <= pe.x1 - 1
                and te.y0 >= pe.y0 + 1 and te.y1 <= pe.y1 - 1), (
                f"text overflows its box: {t.get_text()!r}")

    save(fig, "init-norm-one-layer-steps")


def main():
    plot_chain()
    plot_product()
    plot_init_variance()
    plot_norm_axes()
    plot_ln_vs_rms()
    plot_bn_sequence()
    plot_residual_paths()
    plot_one_layer_variance()
    plot_one_layer_steps()


if __name__ == "__main__":
    main()
