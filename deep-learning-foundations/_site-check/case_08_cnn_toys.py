"""深度学习基础（08）：CNN 配套玩具与配图。
https://arganzheng.life/cnn-from-lenet-to-resnet-and-vit.html

用法：python case_08_cnn_toys.py conv-as-matrix|receptive-field|figures

两个玩具（NumPy / torch CPU，固定种子）：
  conv-as-matrix   4×4 输入、3×3 核 → 2×2 输出；卷积展开成 4×16 稀疏矩阵，
                   非零位置与权重共享，矩阵乘与 conv2d 一致
  receptive-field  堆叠 3×3（stride 1）1/2/3 层与加一层 stride 2 的感受野，
                   公式 r_l = r_{l-1} + (k-1)·∏stride，并用梯度法实测对照
"""
import importlib.util
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch

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


_SEGMENTS = []


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


def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    text(ax, .15, ylim[1] - .3, title, fontsize=10.5, fontweight=600,
         color="#111111")
    return fig, ax


def grid(ax, x, y, vals, cw=.5, ch=.5, fs=7.5, fc="#dce9f7", ec="#7a7a7a",
         zero_grey=True, cell_fc=None, cell_ec=None):
    """画数字格子；cell_fc/cell_ec 可选 (i,j)->颜色 的回调覆盖逐格配色。"""
    vals = np.asarray(vals)
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            grey = zero_grey and v == 0
            c = "#ececec" if grey else fc
            e = ec
            if cell_fc is not None:
                c = cell_fc(i, j)
            if cell_ec is not None:
                e = cell_ec(i, j)
            p = box(ax, x + j * cw, y - (i + 1) * ch, cw - .02, ch - .02,
                    fc=c, ec=e, lw=.4, r=.005)
            text(ax, x + j * cw + (cw - .02) / 2,
                 y - (i + 1) * ch + (ch - .02) / 2,
                 f"{v:g}", in_box=p, ha="center", va="center", fontsize=fs)


# ---------------------------------------------------------------- toys

X4 = np.arange(1, 17, dtype=float).reshape(4, 4)
K3 = np.arange(1, 10, dtype=float).reshape(3, 3)   # 9 个不同的整数


def conv_matrix():
    """3×3 核在 4×4 输入上的 4×16 稀疏矩阵与逐位置乘加。"""
    out = np.zeros((2, 2))
    M = np.zeros((4, 16))
    for oi in range(2):
        for oj in range(2):
            acc = 0.0
            terms = []
            for a in range(3):
                for b in range(3):
                    xi, xj, k = oi + a, oj + b, K3[a, b]
                    acc += k * X4[xi, xj]
                    terms.append((k, X4[xi, xj]))
                    M[oi * 2 + oj, xi * 4 + xj] = k
            out[oi, oj] = acc
            print(f"输出({oi},{oj}): " + " + ".join(
                f"{k:g}×{x:g}" for k, x in terms) + f" = {acc:g}")
    print("输出 y =", out.reshape(-1).tolist())
    import torch
    import torch.nn.functional as Fn
    y_ref = Fn.conv2d(torch.tensor(X4[None, None]),
                      torch.tensor(K3[None, None])).numpy().reshape(-1)
    print(f"max|conv - M@x| = {np.abs(M @ X4.reshape(-1) - y_ref).max():.1e}")
    print(f"矩阵形状 {M.shape}，元素 {M.size}，非零 {int((M != 0).sum())}，"
          f"不同自由参数 {len(set(M[M != 0]))}")
    print(f"等价全连接参数量 4×16={4 * 16}，卷积只要 {K3.size}")
    return out, M


def cmd_conv_as_matrix():
    print("== 4×4 输入、3×3 核（数字 1..9）、stride 1、无 padding ==")
    print("输入 x =\n", X4.astype(int))
    print("核 k =\n", K3)
    conv_matrix()


def rf_formula(layers):
    """layers: [(k, s)] 逐个卷积，r_0=1、jump_0=1；r_l=r_{l-1}+(k-1)·jump。"""
    r, jump = 1, 1
    for k, s in layers:
        r += (k - 1) * jump
        jump *= s
    return r, jump


def rf_measured(strides):
    """梯度法实测：1×N 输入、L 层 1×3 conv，输出中心像素的非零梯度宽度。"""
    import torch
    import torch.nn as nn
    n = 41
    x = torch.zeros(1, 1, n, requires_grad=True)
    h = x
    for s in strides:
        w = torch.ones(1, 1, 3) / 3
        h = torch.nn.functional.conv1d(h, w, stride=s, padding=1)
    c = h.shape[-1]
    h[0, 0, c // 2].backward()
    nz = np.nonzero(x.grad.numpy().reshape(-1))[0]
    return int(nz.max() - nz.min() + 1)


def cmd_receptive_field():
    print("== 感受野：公式 r_l = r_{l-1} + (k-1)·∏stride vs 梯度法实测 ==")
    for L in (1, 2, 3):
        layers = [(3, 1)] * L
        f, _ = rf_formula(layers)
        m = rf_measured([1] * L)
        print(f"  3×3 stride1 ×{L} 层：公式 RF={f}，梯度实测={m}（一致={f == m}）")
    layers = [(3, 1), (3, 2), (3, 1)]
    f, _ = rf_formula(layers)
    m = rf_measured([1, 2, 1])
    print("  3×3 s1 + 3×3 s2 + 3×3 s1："
          f"公式 RF={f}（1+2+4+4），梯度实测={m}（一致={f == m}）")
    print("  stride1 要看到 224：1+2L≥224 → L=112 层")


# ---------------------------------------------------------------- figures

def figure_conv_matrix():
    out, M = conv_matrix()
    fig, ax = base_fig(r"卷积 $=$ 稀疏矩阵 $y=Mx$："
                       r"局部性（每行 9 个非零）$+$ 参数共享（同色同一权重）",
                       h=4.6, xlim=(0, 12.6), ylim=(0.0, 7.0))
    cw = .62
    # ---- 上行：x * k = y，运算符号连接，无箭头
    text(ax, .3, 6.35, r"输入 $x$（$4\times4$）", fontsize=8.5)
    grid(ax, .3, 6.1, X4.astype(int), cw=cw, ch=cw, fs=8)
    # 橙框标出第一个 3×3 窗口（行 0–2、列 0–2）
    box(ax, .3, 6.1 - 3 * cw, 3 * cw - .02, 3 * cw - .02,
        fc="none", ec=C["orange"], lw=1.6, r=.01)
    text(ax, 3.26, 4.86, r"$\ast$", fontsize=14, va="center")
    text(ax, 3.75, 6.35, r"核 $k$（$3\times3$）", fontsize=8.5)
    grid(ax, 3.75, 6.1, K3.astype(int), cw=cw, ch=cw, fs=8,
         fc="#fae6d6", ec=C["orange"], zero_grey=False)
    text(ax, 6.32, 4.86, r"$=$", fontsize=14, va="center")
    text(ax, 7.0, 6.35, r"输出 $y$（$2\times2$）", fontsize=8.5)
    grid(ax, 7.0, 6.1, out.astype(int), cw=.72, ch=.62, fs=6.5,
         fc="#e8f3e8", ec=C["green"], zero_grey=False)
    # 348 格与上行窗口同色描边
    box(ax, 7.0, 6.1 - .62, .7, .6, fc="none", ec=C["orange"], lw=1.6, r=.01)
    # ---- 下行：y = Mx 的矩阵乘排版
    # x 拉直成横条放 M 上方（列对齐），y 拉直成竖条放 M 右侧（行对齐）
    text(ax, .3, 3.4, r"拉直后就是矩阵乘 $y=Mx$：", fontsize=8.5)
    cmap = {v: c for v, c in zip(
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        ("#d6e6f5", "#c2d8f0", "#aecdee", "#9ac2ea",
         "#f5c98f", "#f0b570", "#f0dfc0", "#e8e0f5", "#cbe8c5"))}
    cmap[0.0] = "#f0f0f0"
    mx, mcw, mch = .3, .42, .45
    # x 横条（1..16），与 M 的列对齐
    for j in range(16):
        p = box(ax, mx + j * mcw, 2.62, mcw - .02, .42,
                fc="#dce9f7", ec="#7a7a7a", lw=.4, r=.003)
        text(ax, mx + j * mcw + (mcw - .02) / 2, 2.62 + .21,
             f"{j + 1:g}", in_box=p, ha="center", va="center", fontsize=5.5)
    text(ax, mx + 16 * mcw + .38, 2.62 + .21, r"$x$（拉直 $16$）",
         fontsize=8, va="center", color="#555555")
    # M（4×16）
    for i in range(4):
        for j in range(16):
            v = M[i, j]
            p = box(ax, mx + j * mcw, 2.48 - (i + 1) * mch, mcw - .02,
                    mch - .02, fc=cmap[v],
                    ec="#bbbbbb" if v == 0 else "#888888", lw=.35, r=.003)
            if v != 0:
                text(ax, mx + j * mcw + (mcw - .02) / 2,
                     2.48 - (i + 1) * mch + (mch - .02) / 2,
                     f"{v:g}", in_box=p, ha="center", va="center", fontsize=5.5)
    # M 第 0 行同色描边 —— 对应上行那个窗口
    box(ax, mx, 2.48 - mch, 16 * mcw - .02, mch - .02,
        fc="none", ec=C["orange"], lw=1.6, r=.01)
    # y 竖条（348,393,528,573），与 M 的行对齐
    ys = mx + 16 * mcw + .6
    for i, v in enumerate(out.reshape(-1)):
        p = box(ax, ys, 2.48 - (i + 1) * mch, .58, mch - .02,
                fc="#e8f3e8", ec=C["green"], lw=.5, r=.003)
        text(ax, ys + .29, 2.48 - (i + 1) * mch + (mch - .02) / 2,
             f"{v:g}", in_box=p, ha="center", va="center", fontsize=5.5)
    text(ax, ys + .29, 2.48 - 4 * mch - .13, r"$y$（拉直 $4$）", fontsize=8,
         ha="center", va="top", color="#555555")
    text(ax, ys + 1.0, 2.48 - mch / 2,
         "第 0 行与 $x$ 内积\n$=$ 输出 $(0,0)$ 的 $348$",
         fontsize=7.5, va="center", color="#555555")
    # ---- 底注
    text(ax, .3, .38,
         r"参数量：卷积 $=9$（核的 $3\times3$）"
         r"vs 等价全连接 $=4\times16=64$",
         fontsize=8, color="#555555")
    finish(fig, "cnn-conv-as-matrix")


def figure_receptive_field():
    fig, ax = base_fig(r"感受野：$r_l=r_{l-1}+(k_l-1)\prod_{i<l}s_i$，"
                       r"每层 $3\times3$（stride 1）加宽 2",
                       h=4.2, xlim=(0, 12), ylim=(0.4, 6.6))
    n = 13
    cw = .62
    x0 = (12 - n * cw) / 2
    center = n // 2
    # 从下到上：输入 / 第 1 层 / 第 2 层 / 第 3 层
    rows = [(1.6, 7, r"输入：影响顶端输出的格子数 $7=1+3\times2$"),
            (2.9, 5, r"第 1 层：$5$"),
            (4.2, 3, r"第 2 层：$3$"),
            (5.5, 1, r"第 3 层输出：$1$ 个位置")]
    # 半透明锥形带连接相邻层高亮区两端
    from matplotlib.patches import Polygon
    for ui in range(3):
        lo_span = rows[ui][1]
        hi_span = rows[ui + 1][1]
        lo_l = x0 + (center - lo_span // 2) * cw
        lo_r = x0 + (center + lo_span // 2 + 1) * cw
        hi_l = x0 + (center - hi_span // 2) * cw
        hi_r = x0 + (center + hi_span // 2 + 1) * cw
        lo_y = rows[ui][0]
        hi_y = rows[ui + 1][0] - cw
        ax.add_patch(Polygon([(lo_l, lo_y), (lo_r, lo_y),
                              (hi_r, hi_y), (hi_l, hi_y)],
                             closed=True, fc=C["orange"], ec="#c89030",
                             lw=.8, alpha=.25))
    for y, span, lab in rows:
        for j in range(n):
            on = abs(j - center) <= span // 2
            fc = "#fae6d6" if on else "#f0f0f0"
            box(ax, x0 + j * cw, y - cw, cw - .02, cw - .02,
                fc=fc, ec=C["orange"] if on else "#bbbbbb", lw=.5, r=.005)
        text(ax, x0 + n * cw + .3, y - cw / 2, lab, fontsize=8, va="center",
             color="#555555")
    finish(fig, "cnn-receptive-field")


def figures():
    figure_conv_matrix(); figure_receptive_field()


COMMANDS = {"conv-as-matrix": cmd_conv_as_matrix,
            "receptive-field": cmd_receptive_field,
            "figures": figures}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "conv-as-matrix"
    COMMANDS[cmd]()
