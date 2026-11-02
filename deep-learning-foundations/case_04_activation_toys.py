"""激活函数 04 配套玩具与配图。所有数字由 CPU 实测或直接计算产生。

用法：python case_04_activation_toys.py fold|relu-fit|backward|vanish|dead-relu|glu|saved|figures
"""
import importlib.util
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
BLOG_IMG = "/Users/devin/repos/arganzheng-github-com/img/in-post"
spec = importlib.util.spec_from_file_location(
    "lst_plot", "/Users/devin/repos/ai-learning-labs/large-scale-training/_plot.py")
lst_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lst_plot)
plt, save, text, box, box_title = (lst_plot.plt, lst_plot.save, lst_plot.text,
                                   lst_plot.box, lst_plot.box_title)
C = lst_plot.C
lst_plot.OUT = BLOG_IMG
torch.set_num_threads(1)
plt.rcParams["axes.unicode_minus"] = False


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


_SEGMENTS = []          # (ax, a, b) — arrow/dashed_line 画出的全部线段


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
    """文字框与已登记线段相交 = 失败。在 collide 的盒内检查之外补一层。"""
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
                errs.append(f"线段穿过文字 {t.get_text()!r}：{a} → {b}")
    if errs:
        raise AssertionError(f"{len(errs)} 处文字-线段相交：\n" + "\n".join(errs))


def finish(fig, name):
    check_text_lines(fig)
    save(fig, name)


def grid(ax, x, y, vals, cw=.5, ch=.5, fs=7.5, fc="#dce9f7", ec="#7a7a7a",
         zero_grey=True):
    """数字矩阵格子；y 为顶边，返回底边 y。"""
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            grey = zero_grey and v == 0
            ax.add_patch(Rectangle((x + c * cw, y - r * ch - ch), cw - .02,
                                   ch - .02, facecolor="#ececec" if grey else fc,
                                   edgecolor=ec, lw=.5))
            if not grey:
                ax.text(x + c * cw + (cw - .02) / 2, y - r * ch - ch / 2,
                        f"{v:g}", ha="center", va="center", fontsize=fs)
    return y - len(vals) * ch


def cell(ax, x, y, w, h, main, sub="", fc="#f5f8fc", ec="#777777", fs=9):
    """带数字的小格：主值 + 可选小字（未登记，不参与盒内检查）。"""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=.8))
    ax.text(x + w / 2, y + h * .6 if sub else y + h / 2, main, ha="center",
            va="center", fontsize=fs)
    if sub:
        ax.text(x + w / 2, y + h * .22, sub, ha="center", va="center",
                fontsize=fs - 1.5, color=C["orange"])


def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    ax.set_title(title, fontweight=600, color="#111111", pad=10)
    return fig, ax


# ---------------------------------------------------------------- toys

def toy_fold():
    x = np.array([[1., 1.]])
    W1 = np.array([[1., 2.], [0., 1.]])
    W2 = np.array([[1., 0.], [1., 1.]])
    h = x @ W1
    y2 = h @ W2
    W12 = W1 @ W2
    y1 = x @ W12
    print("折叠的数字例子：x=(1,1)，两层无激活 vs 折叠成一层")
    print(f"W1={W1.tolist()}  W2={W2.tolist()}")
    print(f"两层：x W1 = {h.tolist()} -> (x W1) W2 = {y2.tolist()}")
    print(f"折叠：W1 W2 = {W12.tolist()} -> x (W1 W2) = {y1.tolist()}")
    assert np.allclose(y1, y2)


def toy_relu_fit():
    xs = np.linspace(-2, 2, 4001)
    cs = [-2, -1, 0, 1, 2]
    Phi = np.stack([np.maximum(xs - c, 0) for c in cs] + [np.ones_like(xs)], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, xs ** 2, rcond=None)
    fit = Phi @ coef
    lin = np.polyfit(xs, xs ** 2, 1)
    linfit = np.polyval(lin, xs)
    print("5 个 hinge ReLU(x-c) + 偏置最小二乘拟合 x^2，[-2,2]")
    print("系数 " + ", ".join(f"ReLU(x-({c:+g})):{w:+.4f}" for c, w in zip(cs, coef[:5]))
          + f"，偏置 {coef[5]:+.4f}")
    print(f"最大误差 = {np.abs(fit - xs**2).max():.4f}")
    print(f"纯线性最佳拟合 y = {lin[0]:+.4f} x {lin[1]:+.4f}，最大误差 = "
          f"{np.abs(linfit - xs**2).max():.4f}")


def toy_backward():
    H = np.array([-1., .5, 2.])
    G_A = np.array([.3, -.2, .5])
    sig = 1 / (1 + np.exp(-H)); th = np.tanh(H)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(H / math.sqrt(2)))
    phi = np.exp(-H ** 2 / 2) / math.sqrt(2 * math.pi)
    rows = [("ReLU", (H > 0).astype(float)),
            ("sigmoid", sig * (1 - sig)),
            ("tanh", 1 - th ** 2),
            ("GELU", Phi + H * phi)]
    print(f"H={H.tolist()}, G_A={G_A.tolist()}；G_H = G_A ⊙ f'(H)")
    for name, fp in rows:
        gh = G_A * fp
        print(f"{name:8s} f'(H) = [{', '.join(f'{v:.4f}' for v in fp)}]"
              f"  -> G_H = [{', '.join(f'{v:+.4f}' for v in gh)}]")


def _make_mlp(n_layers=10, width=64, act="relu"):
    layers = [nn.Linear(width, width) for _ in range(n_layers)]
    acts = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
    net = nn.Sequential(*sum(([l, acts[act]()] for l in layers), [])[:-1])
    for lin in layers:
        if act == "relu":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
    return net, layers


def toy_vanish():
    torch.manual_seed(0)
    X = torch.randn(256, 64)
    Y = torch.randint(0, 10, (256,))
    print("10 层 MLP（宽 64，batch 256）：逐层 Linear 权重梯度范数")
    for act in ("sigmoid", "tanh", "relu"):
        net, layers = _make_mlp(act=act)
        if act == "relu":
            pass
        head = nn.Linear(64, 10)
        loss = F.cross_entropy(head(net(X)), Y)
        loss.backward()
        norms = [l.weight.grad.norm().item() for l in layers]
        print(f"{act:8s} " + " ".join(f"{v:.2e}" for v in norms)
              + f"  | L1/L10 = {norms[0] / norms[-1]:.3g}")


def toy_dead_relu():
    torch.manual_seed(1)
    x = torch.randn(256)
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)
    def stats():
        z = x * w.detach() + b.detach()
        return float((z > 0).float().mean())
    p0 = stats()
    z = x * w + b
    loss = F.mse_loss(torch.relu(z), torch.ones_like(z))
    loss.backward()
    g0 = (w.grad.item(), b.grad.item())
    with torch.no_grad():
        b -= 10
    p1 = stats()
    w.grad = None; b.grad = None
    z = x * w + b
    loss = F.mse_loss(torch.relu(z), torch.ones_like(z))
    loss.backward()
    g1 = (w.grad.item(), b.grad.item())
    print(f"单个 ReLU 单元 z = x*w + b，256 个标准正态输入，w={w.item():g}")
    print(f"更新前：b=0.5，预激活>0 比例 {p0*100:.1f}%，梯度非零 ({g0[0]:.3f}, {g0[1]:.3f})")
    print(f"一次大步更新 b -= 10：预激活>0 比例 {p1*100:.1f}%，"
          f"梯度 ({g1[0]:.3f}, {g1[1]:.3f}) —— 全灭，回不来了")


def toy_glu():
    xV = np.array([2., -1., 3.]); xW = np.array([3., -3., 0.])
    gate = 1 / (1 + np.exp(-xW))
    out = xV * gate
    print(f"xV = {xV.tolist()}, xW = {xW.tolist()}")
    print(f"σ(xW) = {[round(v, 4) for v in gate.tolist()]}")
    print(f"xV ⊙ σ(xW) = {[round(v, 4) for v in out.tolist()]}")


def toy_saved():
    torch.manual_seed(0)
    x = torch.randn(4096, 1024)
    saved = {}
    def pack(t):
        saved[id(t)] = (t.shape, t.data_ptr(), t.untyped_storage().nbytes())
        return t
    ops = {
        "ReLU": lambda t: torch.relu(t),
        "sigmoid": lambda t: torch.sigmoid(t),
        "tanh": lambda t: torch.tanh(t),
        "GELU": lambda t: F.gelu(t),
        "SiLU": lambda t: F.silu(t),
        "SwiGLU": lambda t: F.silu(t[:, :512]) * t[:, 512:],
    }
    print("saved_tensors_hooks 实测：PyTorch %s，CPU，输入 [4096,1024] fp32"
          % torch.__version__)
    for name, fn in ops.items():
        saved.clear()
        inp = x.clone().requires_grad_(True)
        in_ptrs = {inp.data_ptr(), inp[:, :512].data_ptr(), inp[:, 512:].data_ptr()}
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = fn(inp)
            out_ptr = out.data_ptr()
            out.sum().backward()
        n_bytes = sum(nb for _, _, nb in saved.values())
        kinds = []
        for shp, ptr, nb in saved.values():
            tag = ("输入/输入切片" if ptr in in_ptrs else
                   "输出" if ptr == out_ptr else "中间结果")
            kinds.append(f"{tag} {list(shp)}")
        print(f"{name:8s} 保存 {len(saved)} 个张量 共 {n_bytes/1024/1024:.1f} MB："
              + "，".join(kinds))


# ---------------------------------------------------------------- figures

def figure_fold():
    W12 = np.array([[3., 2.], [1., 1.]])
    fig, ax = base_fig("没有激活：两层 Linear 折叠成一层，结果一模一样", 7.6, 3.7,
                       (0, 12), (1.0, 6.4))
    # 上排：x -> W1 -> h -> W2 -> out
    row1 = [("x", [[1, 1]]), ("W1", [[1, 2], [0, 1]]), ("h=xW1", [[1, 3]]),
            ("W2", [[1, 0], [1, 1]]), ("out", [[4, 3]])]
    xs = [.3, 2.7, 5.1, 7.5, 9.9]
    for (lab, vals), gx in zip(row1, xs):
        top = 5.5 if len(vals) == 2 else 5.25
        grid(ax, gx, top, vals, cw=.55, fc="#dce9f7", zero_grey=False)
        text(ax, gx + .55, 5.85, lab, ha="center", va="center", fontsize=8.5)
    for gx in xs[:-1]:
        arrow(ax, (gx + 1.15, 5.0), (gx + 2.25, 5.0), color=C["blue"], lw=.9)
    # 下排：x -> W1W2 -> out
    row2 = [("x", [[1, 1]]), ("W1W2", [[3, 2], [1, 1]]), ("out", [[4, 3]])]
    xs2 = [3.3, 5.7, 8.1]
    for (lab, vals), gx in zip(row2, xs2):
        top = 2.9 if len(vals) == 2 else 2.65
        grid(ax, gx, top, vals, cw=.55, fc="#dce9f7", zero_grey=False)
        text(ax, gx + .55, 3.25, lab, ha="center", va="center", fontsize=8.5)
    for gx in xs2[:-1]:
        arrow(ax, (gx + 1.15, 2.4), (gx + 2.25, 2.4), color=C["blue"], lw=.9)
    text(ax, 6, 1.4, "两层链 (4,3) 与一层折叠 (4,3) 完全相同：中间没有非线性，"
         "多叠的层不增加表达能力。", ha="center", va="center", fontsize=9)
    finish(fig, "activation-fold")


def figure_relu_fit():
    xs = np.linspace(-2, 2, 4001)
    cs = [-2, -1, 0, 1, 2]
    Phi = np.stack([np.maximum(xs - c, 0) for c in cs] + [np.ones_like(xs)], axis=1)
    coef, *_ = np.linalg.lstsq(Phi, xs ** 2, rcond=None)
    fit = Phi @ coef
    lin = np.polyfit(xs, xs ** 2, 1)
    linfit = np.polyval(lin, xs)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(xs, xs ** 2, color="#111111", lw=1.8, label="目标 x^2")
    for c, w in zip(cs, coef[:5]):
        ax.plot(xs, w * np.maximum(xs - c, 0), color="#bbbbbb", lw=.8)
    ax.plot(xs, fit, color=C["orange"], lw=1.6, label="5 个加权 hinge + 偏置")
    ax.plot(xs, linfit, color=C["blue"], lw=1.2, ls="--", label="最佳线性拟合")
    ax.plot([], [], color="#bbbbbb", lw=.8, label="各加权 hinge 分量")
    ax.set_ylim(-1, 4.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5,
              frameon=False)
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([0, 1, 2, 3, 4])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("ReLU 单元各贡献一个折点，5 个折点就贴近整条曲线",
                 fontweight=600, color="#111111")
    ax.grid(alpha=.15)
    finish(fig, "activation-relu-fit")


def figure_backward():
    H = np.array([-1., .5, 2.])
    G_A = np.array([.3, -.2, .5])
    sig = 1 / (1 + np.exp(-H)); th = np.tanh(H)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(H / math.sqrt(2)))
    phi = np.exp(-H ** 2 / 2) / math.sqrt(2 * math.pi)
    rows = [("ReLU", (H > 0).astype(float)),
            ("sigmoid", sig * (1 - sig)),
            ("tanh", 1 - th ** 2),
            ("GELU", Phi + H * phi)]
    fig, ax = base_fig("反向：G_H = G_A ⊙ f'(H)，f 的导数决定梯度怎么传",
                       7.6, 3.7, (0, 12), (0.4, 6.2))
    text(ax, 2.3, 5.6, "f'(H)", ha="center", va="center", fontsize=8.5)
    text(ax, 4.6, 5.6, "⊙", ha="center", va="center", fontsize=11)
    text(ax, 6.4, 5.6, "G_A", ha="center", va="center", fontsize=8.5)
    text(ax, 8.5, 5.6, "=", ha="center", va="center", fontsize=11)
    text(ax, 10.3, 5.6, "G_H", ha="center", va="center", fontsize=8.5)
    y = 5.05
    for name, fp in rows:
        text(ax, .15, y - .25, name, ha="left", va="center", fontsize=8)
        grid(ax, 1.05, y, [[round(v, 3) for v in fp]], cw=.85, fs=6.5,
             fc="#f5f8fc", zero_grey=False)
        grid(ax, 5.15, y, [G_A.tolist()], cw=.85, fs=7, fc="#dce9f7",
             zero_grey=False)
        grid(ax, 9.0, y, [[round(v, 3) for v in G_A * fp]], cw=.85, fs=6.5,
             fc="#fae6d6", ec=C["orange"], zero_grey=False)
        y -= 1.0
    text(ax, 6, 0.75, "H = (-1, 0.5, 2)，G_A = (0.3, -0.2, 0.5)；"
         "ReLU 把负位置的梯度直接挡成 0。", ha="center", va="center", fontsize=8.5)
    finish(fig, "activation-backward-fprime")


def figure_vanish():
    torch.manual_seed(0)
    X = torch.randn(256, 64)
    Y = torch.randint(0, 10, (256,))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for act, c in (("sigmoid", C["red"]), ("tanh", C["blue"]), ("relu", C["orange"])):
        net, layers = _make_mlp(act=act)
        head = nn.Linear(64, 10)
        loss = F.cross_entropy(head(net(X)), Y)
        loss.backward()
        norms = [l.weight.grad.norm().item() for l in layers]
        ax.plot(range(1, 11), norms, "o-", ms=4, lw=1.3, color=c, label=act)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 11))
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
    ax.set_yticklabels(["1e-6", "1e-4", "1e-2", "1e0"])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("层号（1 = 靠近输入）"); ax.set_ylabel("||dW||")
    ax.set_title("10 层 MLP 逐层梯度范数：sigmoid 一路衰减，ReLU/tanh 基本平",
                 fontweight=600, color="#111111")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.15)
    finish(fig, "activation-vanishing")


def figure_dead_relu():
    torch.manual_seed(1)
    x = torch.randn(256)
    z0 = x * 1.0 + 0.5
    z1 = x * 1.0 - 9.5
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, z, t, c, xt, xl in ((axs[0], z0, "更新前 b=0.5：67.2% 的输入能激活",
                                 C["blue"], [-2, 0, 2], None),
                                (axs[1], z1, "b -= 10 后：预激活全为负，输出与梯度全 0",
                                 C["orange"], [-12, -8, -4, 0], (-13, 1))):
        ax.hist(z.numpy(), bins=24, color=c, alpha=.75)
        ax.axvline(0, color="#111111", lw=1)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("预激活 z", fontsize=9); ax.set_xticks(xt)
        ax.tick_params(labelsize=9)
        if xl:
            ax.set_xlim(*xl)
        p = float((z > 0).float().mean())
        ax.text(0.96, 0.92, f"z>0：{p*100:.1f}%", transform=ax.transAxes,
                ha="right", fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    axs[0].set_ylabel("输入个数", fontsize=9)
    fig.suptitle("dead ReLU 如何发生：一次过大的更新把预激活整体推过零点",
                 fontweight=600, color="#111111")
    fig.tight_layout(rect=[0, 0, 1, .92])
    finish(fig, "activation-dead-relu")


def figure_glu_example():
    fig, ax = base_fig("GLU 的逐维放行：门分支先过 σ，再与值分支逐元素乘",
                       7.6, 3.4, (0, 10.9), (0.5, 6.0))
    # 值分支
    text(ax, 1.6, 5.3, "值分支 xV", ha="center", va="center", fontsize=8.5)
    grid(ax, .7, 5.0, [[2, -1, 3]], cw=.6, fc="#dce9f7", zero_grey=False)
    # 门分支 -> σ
    text(ax, 4.4, 5.3, "门分支 xW", ha="center", va="center", fontsize=8.5)
    grid(ax, 3.5, 5.0, [[3, -3, 0]], cw=.6, fc="#dce9f7", zero_grey=False)
    p = box(ax, 5.8, 4.35, 1.0, .7, "", fc="#f5f8fc", ec="#777777")
    text(ax, 6.3, 4.7, "σ", in_box=p, ha="center", va="center", fontsize=10)
    arrow(ax, (5.4, 4.7), (5.75, 4.7), color=C["blue"], lw=.9)
    text(ax, 8.5, 5.3, "σ(xW)", ha="center", va="center", fontsize=8.5)
    grid(ax, 7.6, 5.0, [[.95, .05, .50]], cw=.6, fc="#dce9f7", zero_grey=False)
    arrow(ax, (6.85, 4.7), (7.55, 4.7), color=C["blue"], lw=.9)
    # ⊙ 汇合
    text(ax, 4.3, 3.8, "xV ⊙ σ(xW)", ha="center", va="center", fontsize=8.5)
    arrow(ax, (1.6, 4.45), (3.4, 3.5), color=C["blue"], lw=.9)
    arrow(ax, (8.5, 4.45), (4.9, 3.5), color=C["blue"], lw=.9)
    grid(ax, 3.7, 3.4, [[1.90, -.05, 1.50]], cw=.85, fs=7.5,
         fc="#fae6d6", ec=C["orange"], zero_grey=False)
    text(ax, 5.45, 2.3, "第一维几乎全放行，第二维几乎关掉，第三维放一半；"
         "门和内容用的是不同权重。", ha="center", va="center", fontsize=9)
    text(ax, 5.45, 1.55, "反向沿两支分别回传：值支乘 σ(xW)，门支乘 xV·σ'(xW)。",
         ha="center", va="center", fontsize=8.5, color="#555555")
    finish(fig, "activation-glu-example")


def figure_ffn_position():
    fig, ax = base_fig("激活函数在 Transformer block 里的位置：FFN 内部那一处",
                       7.6, 3.9, (0, 12.2), (0.4, 6.6))
    chain = [("输入", .3, .8), ("LayerNorm", 1.3, 1.3), ("Attention", 2.8, 1.2),
             ("+ 残差", 4.2, 1.0), ("LayerNorm", 5.4, 1.3)]
    y = 4.7
    for lab, x, w in chain:
        p = box(ax, x, y, w, .8, "", fc="#dce9f7", ec=C["blue"])
        text(ax, x + w / 2, y + .4, lab, in_box=p, ha="center", va="center",
             fontsize=7)
    for a, b in ((1.12, 1.28), (2.62, 2.78), (4.02, 4.18), (5.22, 5.38)):
        arrow(ax, (a, y + .4), (b, y + .4), color=C["blue"], lw=.9)
    # FFN 大框（未登记容器：内部元素仍各自参与检查）
    ax.add_patch(Rectangle((6.85, 4.3), 3.75, 1.9, facecolor="#f5f8fc",
                           edgecolor=C["blue"], lw=1.0))
    text(ax, 8.72, 6.0, "FFN（SwiGLU）", ha="center", va="center",
         fontsize=8.5)
    arrow(ax, (6.72, y + .4), (6.83, y + .4), color=C["blue"], lw=.9)
    # 框内两条并行支路：gate xW→SiLU / up xV，汇合 ⊙ → W2 down
    arrow(ax, (6.87, 5.1), (7.08, 5.42), color=C["orange"], lw=.8)
    arrow(ax, (6.87, 5.1), (7.08, 4.78), color=C["orange"], lw=.8)
    p2 = box(ax, 7.1, 5.35, .8, .42, "", fc="#ffffff", ec="#888888")
    text(ax, 7.5, 5.56, "xW 门", in_box=p2, ha="center", va="center", fontsize=7)
    p2 = box(ax, 8.35, 5.35, .65, .42, "", fc="#ffffff", ec="#888888")
    text(ax, 8.675, 5.56, "SiLU", in_box=p2, ha="center", va="center", fontsize=7)
    p2 = box(ax, 7.1, 4.55, .8, .42, "", fc="#ffffff", ec="#888888")
    text(ax, 7.5, 4.76, "xV 值", in_box=p2, ha="center", va="center", fontsize=7)
    arrow(ax, (7.92, 5.56), (8.33, 5.56), color=C["orange"], lw=.8)
    arrow(ax, (9.02, 5.5), (9.18, 5.28), color=C["orange"], lw=.8)
    arrow(ax, (7.92, 4.76), (9.18, 5.0), color=C["orange"], lw=.8)
    p2 = box(ax, 9.2, 4.85, .45, .5, "", fc="#fae6d6", ec=C["orange"])
    text(ax, 9.425, 5.1, "⊙", in_box=p2, ha="center", va="center", fontsize=8)
    arrow(ax, (9.67, 5.1), (9.78, 5.1), color=C["orange"], lw=.8)
    p2 = box(ax, 9.8, 4.75, .7, .7, "", fc="#fae6d6", ec=C["orange"])
    text(ax, 10.15, 5.22, "W2", in_box=p2, ha="center", va="center", fontsize=7)
    text(ax, 10.15, 4.97, "down", in_box=p2, ha="center", va="center",
         fontsize=6)
    # 形状标注：d → d_g（两支）→ d
    text(ax, 6.95, 4.35, "d", ha="center", va="center", fontsize=7)
    text(ax, 7.5, 5.9, "d_g", ha="center", va="center", fontsize=6.5)
    text(ax, 7.5, 4.42, "d_g", ha="center", va="center", fontsize=6.5)
    # +残差 → 输出
    p = box(ax, 10.72, 4.7, .72, .8, "", fc="#dce9f7", ec=C["blue"])
    text(ax, 11.08, 5.1, "+ 残差", in_box=p, ha="center", va="center",
         fontsize=7)
    p = box(ax, 11.55, 4.7, .62, .8, "", fc="#dce9f7", ec=C["blue"])
    text(ax, 11.86, 5.1, "输出", in_box=p, ha="center", va="center", fontsize=7)
    arrow(ax, (10.52, 5.1), (10.7, 5.1), color=C["blue"], lw=.9)
    arrow(ax, (11.46, 5.1), (11.53, 5.1), color=C["blue"], lw=.9)
    text(ax, 6, 1.9, "每个 token 各自过同一个 FFN；激活函数在 gate/up 之间，"
         "形状变化 d → d_g → d。", ha="center", va="center", fontsize=9)
    text(ax, 6, 1.25, "注意力混合 token 之间信息；FFN 只在单 token 内部算。",
         ha="center", va="center", fontsize=8.5, color="#555555")
    finish(fig, "activation-ffn-position")


# ---------------------------------------------------------------- main

def figures():
    figure_fold(); figure_relu_fit(); figure_backward(); figure_vanish()
    figure_dead_relu(); figure_glu_example(); figure_ffn_position()
    print("figures written to blog img/in-post")


TOYS = {"fold": toy_fold, "relu-fit": toy_relu_fit, "backward": toy_backward,
        "vanish": toy_vanish, "dead-relu": toy_dead_relu, "glu": toy_glu,
        "saved": toy_saved}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "figures":
        figures()
    elif cmd == "all":
        for fn in TOYS.values():
            fn()
    else:
        TOYS[cmd]()
