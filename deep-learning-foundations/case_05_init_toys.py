"""深度学习基础（05）：初始化、归一化与残差 配套玩具与配图。
https://arganzheng.life/initialization-normalization-and-residual.html

用法：python case_05_init_toys.py layer-var|residual-var|scale-inv|rms-vs-ln|figures

四个玩具（NumPy / torch CPU，固定种子）：
  layer-var     单层 Linear 输出方差实测 vs 公式 n_in·sigma_w^2·sigma_x^2（+ReLU x1/2）
  residual-var  64 个残差块残差流 std：无归一化 / Pre-Norm / Pre-Norm + 1/sqrt(2L)
                （配置与正文第九章的 02_init_norm_residual.py 完全一致）
  scale-inv     Norm(cx)=Norm(x)；W x c 后前向不变、dL/dW 变 1/c
  rms-vs-ln     [2,4,6,8] 与 [-3,-1,1,3] 的 LN / RMSNorm 输出（正文四.3 的手算）
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as tnn
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BLOG_IMG = "/Users/devin/repos/arganzheng-github-com/img/in-post"
spec = importlib.util.spec_from_file_location(
    "lst_plot", "/Users/devin/repos/ai-learning-labs/large-scale-training/_plot.py")
lst_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lst_plot)
plt, save, text, box = (lst_plot.plt, lst_plot.save, lst_plot.text,
                        lst_plot.box)
C = lst_plot.C
lst_plot.OUT = BLOG_IMG
torch.set_num_threads(1)
plt.rcParams["axes.unicode_minus"] = False

from dlf import data, nn                       # noqa: E402
from dlf.layers import LayerNorm, RMSNorm, Residual, Sequential, make_deep_mlp  # noqa: E402


def _segment_hits_rect(a, b, rect):
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q <= 1e-4:
                return False
        else:
            r = q / p
            if p < 0:
                lo = max(lo, r)
            else:
                hi = min(hi, r)
            if lo >= hi:
                return False
    if lo >= hi:
        return False
    t = (lo + hi) / 2
    mid = (a[0] + t * dx, a[1] + t * dy)
    return x0 + 1e-4 < mid[0] < x1 - 1e-4 and y0 + 1e-4 < mid[1] < y1 - 1e-4


def assert_line_clear(ax, start, end):
    for patch in lst_plot._PATCHES:
        if patch.get_figure() is not ax.figure:
            continue
        x, y, w, h = patch.get_bbox().bounds
        assert not _segment_hits_rect(start, end, (x, y, x + w, y + h)), \
            f"line crosses box: {start} to {end}"


_SEGMENTS = []


def dashed_line(ax, start, end, color=C["orange"], lw=.9):
    assert_line_clear(ax, start, end)
    _SEGMENTS.append((ax, start, end))
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, ls="--")


def arrow(ax, start, end, color="#555555", lw=1.2, style="-|>"):
    assert_line_clear(ax, start, end)
    _SEGMENTS.append((ax, start, end))
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


def check_text_lines(fig):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    errs = []
    for t, _, _ in lst_plot._TRACKED:
        if t.get_figure() is not fig or not t.get_text().strip():
            continue
        e = t.get_window_extent(renderer=ren)
        for ax, a, b in _SEGMENTS:
            if ax.figure is not fig:
                continue
            pa, pb = ax.transData.transform(a), ax.transData.transform(b)
            if _segment_hits_rect(pa, pb, (e.x0 + 1, e.y0 + 1, e.x1 - 1, e.y1 - 1)):
                errs.append(f"线段穿过文字 {t.get_text()!r}：{a} -> {b}")
    if errs:
        raise AssertionError(f"{len(errs)} 处文字-线段相交：\n" + "\n".join(errs))


def finish(fig, name):
    check_text_lines(fig)
    save(fig, name)


def grid(ax, x, y, vals, cw=.5, ch=.5, fs=7.5, fc="#dce9f7", ec="#7a7a7a",
         fmt="{v:g}"):
    """数字格子；y 为顶边，返回底边 y。"""
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            ax.add_patch(Rectangle((x + c * cw, y - r * ch - ch), cw - .02,
                                   ch - .02, facecolor=fc, edgecolor=ec, lw=.5))
            ax.text(x + c * cw + (cw - .02) / 2, y - r * ch - ch / 2,
                    fmt.format(v=v), ha="center", va="center", fontsize=fs)
    return y - len(vals) * ch


def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    ax.set_title(title, fontweight=600, color="#111111", pad=10)
    return fig, ax


# ---------------------------------------------------------------- toys

def toy_layer_var():
    rng = np.random.default_rng(0)
    n_in, batch = 256, 4096
    X = rng.standard_normal((batch, n_in))
    print(f"单层 Linear：n_in={n_in}，batch={batch}，Var(x)=1；"
          f"公式 Var(y) = n_in * sigma_w^2 * Var(x)，ReLU 后二阶矩约 x1/2")
    for name, sw2 in (("1/n_in", 1 / n_in), ("2/n_in", 2 / n_in),
                      ("0.5/n_in", 0.5 / n_in)):
        W = rng.standard_normal((n_in, n_in)) * np.sqrt(sw2)
        y = X @ W
        relu = np.maximum(y, 0)
        formula = n_in * sw2 * X.var()
        print(f"sigma_w^2 = {name:7s} ({sw2:.5f}): Var(y) 实测 {y.var():.4f}"
              f" vs 公式 {formula:.4f}"
              f" | ReLU 后二阶矩 实测 {(relu ** 2).mean():.4f}"
              f" vs 公式 {formula / 2:.4f}")


def _block_stds(net, X):
    """每个 Residual 块出口处残差流的 std。"""
    stds, h = [], X
    for l in net.layers:
        h = l.forward(h)
        if isinstance(l, Residual):
            stds.append(float(h.std()))
    return np.array(stds)


def _prenorm_unscaled(d=256, L=64, rng=None):
    """prenorm 去掉 1/sqrt(2L) 缩放：h + W2 relu(W1 LN(h))，W2 std = sqrt(1/d)。"""
    blocks = [nn.Linear(784, d, rng)]
    for _ in range(L):
        blocks.append(Residual(Sequential(
            [LayerNorm(d), nn.Linear(d, d, rng), nn.ReLU(),
             nn.Linear(d, d, rng, std=np.sqrt(1 / d))])))
    blocks += [LayerNorm(d), nn.Linear(d, 10, rng, std=np.sqrt(1 / d))]
    return Sequential(blocks)


def residual_var_runs(X):
    """三种接法逐块残差流 std；配置与第九章 02_init_norm_residual.py 相同。"""
    runs = {}
    rng = np.random.default_rng(0)
    runs["res-nonorm"] = _block_stds(make_deep_mlp("res-nonorm", rng=rng), X)
    rng = np.random.default_rng(0)
    runs["prenorm"] = _block_stds(_prenorm_unscaled(rng=rng), X)
    rng = np.random.default_rng(0)
    runs["prenorm-scaled"] = _block_stds(make_deep_mlp("prenorm", rng=rng), X)
    return runs


def toy_residual_var():
    np.seterr(all="ignore")
    X, y, Xt, yt = data.load_mnist_standardized()
    runs = residual_var_runs(X[:256])
    pick = [1, 8, 16, 32, 64]
    print("64 个残差块（宽 256，batch 256 张 MNIST）：残差流 std 实测 vs 公式")
    print("块号 " + " ".join(f"{p:>6d}" for p in pick))
    label = {"res-nonorm": "无归一化（公式 std x2^(l/2)）",
             "prenorm": "Pre-Norm（公式 Var = Var0 + l*sig_f^2）",
             "prenorm-scaled": "Pre-Norm + out-proj /sqrt(2L)（近似平）"}
    for k in runs:
        vals = runs[k][[p - 1 for p in pick]]
        print(f"{label[k]:42s} " + " ".join(f"{v:9.2e}" for v in vals))
    sf2 = (runs["prenorm"][0] ** 2 - 1).clip(min=0)
    print(f"\n对照公式：无归一化块 64 公式 std 2^32 = {2**32:.2e}"
          f" 实测 {runs['res-nonorm'][63]:.2e}")
    print(f"Pre-Norm 无缩放块 64 公式 sqrt(1 + 64*sig_f^2)，"
          f"sig_f^2 取块1实测 {sf2:.3f} -> {np.sqrt(1 + 64 * sf2):.3f}"
          f" 实测 {runs['prenorm'][63]:.2e}")


def toy_scale_inv():
    torch.manual_seed(0)
    x = torch.tensor([2., 4., 6., 8.])
    ln = tnn.LayerNorm(4, elementwise_affine=False)
    rms = tnn.RMSNorm(4, eps=1e-5)
    print("x = [2,4,6,8] 与 2x：")
    print(f"  LN(x)  = {[round(v, 4) for v in ln(x).tolist()]}")
    print(f"  LN(2x) = {[round(v, 4) for v in ln(2 * x).tolist()]}  —— 相同")
    print(f"  RMS(x) = {[round(v, 4) for v in rms(x).tolist()]}")
    print(f"  RMS(2x)= {[round(v, 4) for v in rms(2 * x).tolist()]}  —— 相同")
    x2 = torch.randn(8, 16)
    W0 = torch.randn(16, 16)
    g = torch.randn(8, 16)
    ln2 = tnn.LayerNorm(16, elementwise_affine=False)
    outs, norms = [], []
    for c in (1.0, 2.0):
        W = (c * W0).clone().requires_grad_(True)
        y = ln2(x2 @ W)
        loss = (y * g).sum()
        loss.backward()
        outs.append(y.detach()); norms.append(W.grad.norm().item())
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    print("\nLinear -> LN：W 乘 c=2")
    print(f"  前向输出相同（max|diff| = {(outs[0] - outs[1]).abs().max():.2e}）")
    print(f"  ||dL/dW|| c=1: {norms[0]:.4f}   c=2: {norms[1]:.4f}"
          f"   比值 {norms[1] / norms[0]:.4f} = 1/c")


def toy_rms_vs_ln():
    ln4, rms4 = LayerNorm(4), RMSNorm(4)
    print("第 1 节的两个向量过 LN 与 RMSNorm（与正文四.3 的手算对照）：")
    for x in ([2., 4., 6., 8.], [-3., -1., 1., 3.]):
        a = np.array([x])
        print(f"x = {x}: LN = {[round(float(v), 3) for v in ln4.forward(a)[0]]}"
              f"   RMSNorm = {[round(float(v), 3) for v in rms4.forward(a)[0]]}")


# ---------------------------------------------------------------- figures

def figure_residual_var():
    np.seterr(all="ignore")
    X, y, Xt, yt = data.load_mnist_standardized()
    runs = residual_var_runs(X[:256])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xs = np.arange(1, 65)
    styles = [("res-nonorm", "残差、无归一化：方差每块 ×2",
               C["red"]),
              ("prenorm", "Pre-Norm：Var = Var0 + l·σf^2（线性）",
               C["blue"]),
              ("prenorm-scaled", "Pre-Norm + out-proj /sqrt(2L)（GPT-2）",
               C["green"])]
    for key, lab, col in styles:
        ax.plot(xs, runs[key], c=col, lw=1.4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("块号 l")
    ax.set_ylabel("残差流 std（对数）")
    ax.set_title("64 个残差块：残差流标准差——无归一化翻倍、Pre-Norm 线性、加缩放近似平",
                 fontweight=600, color="#111111")
    ax.axhline(1, ls=":", c=C["gray"], lw=.8)
    ax.text(17, 3e6, "2^32 = 4.3e9，实测 6.0e9",
            fontsize=8, color=C["red"])
    ax.annotate("块 64 实测 1.46", xy=(63, runs["prenorm-scaled"][63]),
                xytext=(52, 3.0), fontsize=8, color=C["green"],
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=.8))
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    finish(fig, "dl-init-residual-var")


def figure_scale_inv():
    torch.manual_seed(0)
    x = torch.tensor([2., 4., 6., 8.])
    ln = tnn.LayerNorm(4, elementwise_affine=False)
    x2 = torch.randn(8, 16)
    W0 = torch.randn(16, 16)
    g = torch.randn(8, 16)
    ln2 = tnn.LayerNorm(16, elementwise_affine=False)
    norms = []
    for c in (1.0, 2.0):
        W = (c * W0).clone().requires_grad_(True)
        loss = (ln2(x2 @ W) * g).sum()
        loss.backward()
        norms.append(W.grad.norm().item())

    fig, ax = base_fig("尺度不变性：Norm(cx) = Norm(x)，梯度随权重范数反比缩小",
                     h=3.4, xlim=(0, 12), ylim=(1.4, 5.9))
    # 左：x 与 2x 归一化到同一组格子
    text(ax, .2, 5.15, "LN(x)", fontsize=8.5)
    text(ax, .2, 4.45, "LN(2x)", fontsize=8.5)
    text(ax, 2.75, 5.55, "x = [2, 4, 6, 8]", fontsize=8.5, ha="center")
    text(ax, 5.3, 5.55, "输出相同", fontsize=8.5, ha="center",
         color=C["green"])
    out = [round(v, 3) for v in ln(x).tolist()]
    grid(ax, 1.55, 5.2, [out], cw=1.0, ch=.55, fs=8)
    grid(ax, 1.55, 4.5, [out], cw=1.0, ch=.55, fs=8)
    ax.text(1.55, 2.05, "两个输入差 2 倍，输出逐元素相同：整体尺度被归一化吃掉",
            fontsize=8, color="#555555")
    # 右：W x c 后 dL/dW 变 1/c
    bx = 7.2
    text(ax, bx, 5.55, "Linear -> LN，W 乘 c 倍", fontsize=8.5)
    bw = 4.0
    for i, (c, nrm, col) in enumerate(((1, norms[0], C["blue"]),
                                      (2, norms[1], C["orange"]))):
        yy = 4.7 - i * 0.9
        wbar = bw * nrm / norms[0]
        ax.add_patch(Rectangle((bx + .7, yy - .3), wbar, .5,
                               facecolor=col, alpha=.75, edgecolor="none"))
        text(ax, bx + .12, yy - .05, f"c={c}", fontsize=8)
        ax.text(bx + .75 + wbar, yy - .05,
                f"||dL/dW|| = {nrm:.3f}", fontsize=8, va="center")
    text(ax, bx + .12, 3.08,
         f"c=2 的梯度范数 = c=1 的 {norms[1] / norms[0]:.3f} = 1/c；"
         "相对更新缩小 c^2 倍", fontsize=8, color="#555555")
    text(ax, bx + .12, 2.68, "前向输出逐位相同；权重越大有效学习率越小",
         fontsize=8, color="#555555")
    finish(fig, "dl-init-scale-invariance")


def figure_diagnosis():
    """第八章三条曲线：健康（prenorm）vs 病态（ln 无残差 / res-nonorm）。"""
    np.seterr(all="ignore")
    X, y, Xt, yt = data.load_mnist_standardized()
    rng = np.random.default_rng(0)
    nets = {k: make_deep_mlp(k, rng=np.random.default_rng(0))
            for k in ("prenorm", "ln", "res-nonorm")}

    def init_stats(net):
        acts, h = [], X[:256]
        for l in net.layers:
            h = l.forward(h)
            if not isinstance(l, nn.ReLU):
                acts.append(h.std())
        loss, d, _ = nn.softmax_ce(h, y[:256]); net.backward(d)
        gn = [np.linalg.norm(dP) for P, dP in net.params() if P.ndim == 2]
        return np.array(acts), np.array(gn)

    def dw_ratio(net, steps=300, lr=0.005):
        """每步 ||dW||/||W||（取每层最大；dW=lr*grad）"""
        ratios = []
        for s in range(steps):
            idx = rng.integers(0, X.shape[0], 128)
            loss, d, _ = nn.softmax_ce(net.forward(X[idx]), y[idx])
            net.backward(d)
            if not np.isfinite(loss):
                ratios.append(np.nan); continue
            rs = [lr * np.linalg.norm(dP) / np.linalg.norm(P)
                  for P, dP in net.params() if P.ndim == 2]
            ratios.append(max(rs))
            for P, dP in net.params():
                P -= lr * dP
        return np.array(ratios)

    stats = {k: init_stats(nets[k]) for k in nets}
    ratios = {lr: dw_ratio(make_deep_mlp("prenorm",
                                         rng=np.random.default_rng(0)),
                           lr=lr)
              for lr in (0.005, 0.5)}

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    fig.suptitle("三条诊断曲线：健康（Pre-Norm）vs 病态", fontweight=600,
                 color="#111111")
    ax = axes[0]
    for k, col, lab in (("prenorm", C["green"], "prenorm（健康）"),
                        ("res-nonorm", C["red"], "res-nonorm（病态）")):
        ax.plot(np.linspace(0, 64, len(stats[k][0])), stats[k][0],
                c=col, lw=1.1, label=lab)
    ax.set_yscale("log"); ax.set_title("激活 RMS vs 层", fontsize=9)
    ax.set_yticks([1e-9, 1e-3, 1, 1e3, 1e9])
    ax.set_yticklabels(["1e-9", "1e-3", "1", "1e3", "1e9"], fontsize=7)
    ax.set_xticks([0, 32, 64])
    ax.set_xlabel("块", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axes[1]
    for k, col, lab in (("prenorm", C["green"], "prenorm（健康）"),
                        ("ln", C["red"], "ln 无残差（病态）")):
        g = stats[k][1]
        gg = g[1:-1:2] if len(g) > 70 else g[1:-1]
        ax.plot(np.linspace(1, 64, len(gg)), gg, c=col, lw=1.1, label=lab)
    ax.set_yscale("log"); ax.set_title("梯度范数 vs 层", fontsize=9)
    ax.set_yticks([1e-1, 1, 1e1])
    ax.set_yticklabels(["0.1", "1", "10"], fontsize=7)
    ax.set_xticks([1, 32, 64])
    ax.set_xlabel("块", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax = axes[2]
    for lr, col, lab in ((0.005, C["green"], "Pre-Norm lr 0.005（健康）"),
                         (0.5, C["red"], "Pre-Norm lr 0.5（病态）")):
        ax.plot(ratios[lr], c=col, lw=1.1, label=lab)
    ax.set_yscale("log")
    ax.set_title("max ||dW||/||W|| vs 步（同一 Pre-Norm）", fontsize=9)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["1e-3", "1e-2", "1e-1", "1"], fontsize=7)
    ax.axhline(1e-3, ls=":", c=C["gray"], lw=.8)
    ax.set_xticks([0, 150, 300])
    ax.set_xlabel("步", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=1)
    finish(fig, "dl-init-diagnosis")


def figures():
    figure_residual_var(); figure_scale_inv(); figure_diagnosis()
    print("figures written to blog img/in-post")


TOYS = {"layer-var": toy_layer_var, "residual-var": toy_residual_var,
        "scale-inv": toy_scale_inv, "rms-vs-ln": toy_rms_vs_ln}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "figures":
        figures()
    elif cmd == "all":
        for fn in TOYS.values():
            fn()
    else:
        TOYS[cmd]()
