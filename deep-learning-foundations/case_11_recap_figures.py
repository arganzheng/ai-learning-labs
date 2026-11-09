"""深度学习基础（11）：系列总结与通关自测 配图。
https://arganzheng.life/deep-learning-foundations-series-recap-and-self-test.html

用法：python case_11_recap_figures.py

只生成图：dl-recap-training-step.svg——一次训练迭代的流水线，
各块标篇号与一个关键公式（数据→模型→loss→反向→优化器→回到模型；
04 激活嵌在模型里、05 初始化/归一化/残差作用在模型上、07 正则化挂在
loss 与优化器上、08/09/10 是模型的可替换结构）。
"""
import importlib.util
import os
import sys

import numpy as np
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
spec_dlf = importlib.util.spec_from_file_location(
    "dlf_plot", os.path.join(HERE, "_plot.py"))
dlf_plot = importlib.util.module_from_spec(spec_dlf)
spec_dlf.loader.exec_module(dlf_plot)   # 导入即注册 Noto + 真文字 SVG
save = dlf_plot.enable_embed(lst_plot)  # save() 后内嵌字体 + 数学检查
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


def cell(ax, x, y, w, h, body, fs=10.5, fc="#eef4fb", ec="#7a9ac0"):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=.9, r=.03)
    ls_ = body.split("\n")
    if len(ls_) == 1:
        text(ax, x + w / 2, y + h / 2, body, in_box=p, ha="center",
             va="center", fontsize=fs)
    else:
        text(ax, x + w / 2, y + h * .68, ls_[0], in_box=p, ha="center",
             va="center", fontsize=fs)
        text(ax, x + w / 2, y + h * .32, ls_[1], in_box=p, ha="center",
             va="center", fontsize=fs)
    return p


def cell2(ax, x, y, w, h, head, sub, fs_h=11, fs_s=10.5,
          fc="#eef4fb", ec="#7a9ac0"):
    """两行块：上行篇号+名称，下行关键公式。"""
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=.9, r=.03)
    text(ax, x + w / 2, y + h * .70, head, in_box=p, ha="center", va="center",
         fontsize=fs_h, color="#222222")
    text(ax, x + w / 2, y + h * .30, sub, in_box=p, ha="center", va="center",
         fontsize=fs_s, color="#444444")
    return p


# ---------------------------------------------------------------- figure

def figure_training_step():
    fig, ax = base_fig("一次训练迭代：系列各篇在这条流水线上的位置",
                       h=4.3, xlim=(0, 13.4), ylim=(0.8, 7.3))
    y, ch, cw = 4.1, 1.2, 1.9
    cy = y + ch / 2

    cell2(ax, .2, y, cw, ch, "数据", r"batch (x,y)")
    arrow(ax, (2.15, cy), (2.33, cy))
    pm = box(ax, 2.35, y - .3, 3.6, 1.8, fc="#fae6d6", ec=C["orange"],
             lw=1.1, r=.03)
    text(ax, 4.15, y + 1.28, "模型 $f_\\theta$", in_box=pm, ha="center",
         va="center", fontsize=10.5, color="#222222")
    text(ax, 4.15, y + .88, "04 激活 $\\phi$ 嵌在每层", in_box=pm,
         ha="center", va="center", fontsize=10.5, color="#444444")
    text(ax, 4.15, y + .50, "05 初始化·归一化·残差", in_box=pm,
         ha="center", va="center", fontsize=10.5, color="#444444")
    text(ax, 4.15, y + .12, "给梯度恒等通路", in_box=pm, ha="center",
         va="center", fontsize=10.5, color="#444444")
    cell2(ax, 6.0, y, cw, ch, "02 损失函数", r"$-\log p_y$")
    arrow(ax, (7.95, cy), (8.65, cy))
    cell2(ax, 8.7, y, cw, ch, "03 反向传播",
          r"$\partial L/\partial\theta$")
    arrow(ax, (10.65, cy), (11.0, cy))
    cell2(ax, 11.05, y, 2.1, ch, "06 优化器",
          r"$\theta\!\leftarrow\!\theta-\eta\hat m/\sqrt{\hat v}$")
    # 优化器回到模型：下沿回环
    ax.plot([12.1, 12.1, 4.15], [y - .05, 3.30, 3.30], color="#555555",
            lw=1.0)
    _SEGMENTS.append((ax, (12.1, y - .05), (12.1, 3.30)))
    _SEGMENTS.append((ax, (12.1, 3.30), (4.15, 3.30)))
    arrow(ax, (4.15, 3.30), (4.15, y - .35))
    text(ax, 8.2, 3.34, "更新 $\\theta$", fontsize=10.5, ha="center",
         va="bottom", color="#555555")
    # 07 正则化挂在 loss 与优化器上
    pr = box(ax, 5.4, 5.9, 5.8, .8, fc="#eef4fb", ec="#7a9ac0", lw=.9,
             r=.03)
    text(ax, 8.3, 6.3, "07 正则化：loss $+\\lambda\\|\\theta\\|^2$"
         " / dropout / wd", in_box=pr, ha="center",
         va="center", fontsize=10.5, color="#333333")
    arrow(ax, (6.95, 5.9), (6.95, 5.35), style="-")
    arrow(ax, (9.9, 5.9), (9.9, 5.35), style="-")
    # 可替换结构块：三格顶边括号 → 竖线连到 模型盒底
    cell(ax, .5, 1.1, 2.3, 1.2, "08 CNN\ny=Mx 稀疏共享", fs=10.5)
    cell(ax, 2.95, 1.1, 2.3, 1.2, "09 Embedding\n$e_i^{\\!\\top}W$ 查表",
         fs=10.5)
    cell(ax, 5.4, 1.1, 3.0, 1.2,
         "10 RNN\n$h_t=\\tanh(Wh_{t-1}\\!+\\!Ux_t)$", fs=10.5)
    for cx_ in (1.65, 4.1, 6.9):
        ax.plot([cx_, cx_], [2.3, 2.55], color="#aaaaaa", lw=.8)
        _SEGMENTS.append((ax, (cx_, 2.3), (cx_, 2.55)))
    ax.plot([1.65, 6.9], [2.55, 2.55], color="#aaaaaa", lw=.8)
    _SEGMENTS.append((ax, (1.65, 2.55), (6.9, 2.55)))
    ax.plot([4.15, 4.15], [2.55, 3.78], color="#aaaaaa", lw=.8)
    _SEGMENTS.append((ax, (4.15, 2.55), (4.15, 3.78)))
    text(ax, 4.4, 2.95, "模型块可替换的结构（08 / 09 / 10）",
         fontsize=10.5, ha="left", va="center", color="#555555")
    finish(fig, "dl-recap-training-step")


if __name__ == "__main__":
    figure_training_step()
