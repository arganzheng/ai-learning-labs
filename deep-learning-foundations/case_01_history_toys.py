#!/usr/bin/env python3
"""01《深度模型演进史》数字玩具：

  perceptron-xor  感知机学习规则（lr=1、w,b 初值 0、固定顺序）训 AND / OR / XOR：
                  打印每个任务收敛所需 epoch 与最终 (w,b)；XOR 跑 100 epoch，
                  打印末尾误分类数，说明永远学不到零误分；
                  再用手工设定的 2-2-1 网络（隐藏层 OR 与 NAND，输出 AND）解 XOR，
                  打印 4 个点的隐藏值与输出。
  figures         生成图：dl-evo-perceptron-xor.svg（4 个小图：AND/OR/XOR 的点与
                  决策线，加隐藏空间 (h1,h2) 里 XOR 变成线性可分）。
"""

import importlib.util
import os
import sys

import numpy as np
from matplotlib.lines import Line2D

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

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
LABELS = {"AND": np.array([-1, -1, -1, 1]),
          "OR": np.array([-1, 1, 1, 1]),
          "XOR": np.array([-1, 1, 1, -1])}


def perceptron(name, epochs=100, lr=1.0):
    """Rosenblatt ±1 写法：y(w·x+b) ≤ 0 时 w += yx, b += y；
    返回 (w, b, 首次零误分 epoch 或 None, 各 epoch 误分数)。"""
    y = LABELS[name]
    w = np.zeros(2)
    b = 0.0
    mistakes = []
    first_zero = None
    for ep in range(epochs):
        n_err = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                n_err += 1
                w = w + lr * yi * xi
                b = b + lr * yi
        mistakes.append(n_err)
        if n_err == 0 and first_zero is None:
            first_zero = ep + 1
    return w, b, first_zero, mistakes


def hidden_space(name):
    """手工设定的 OR / NAND 隐藏层对四个点求 (h1, h2)。"""
    H = lambda z: 1 if z >= 0 else 0
    out = []
    for xi in X:
        h1 = H(xi[0] + xi[1] - 0.5)
        h2 = H(-xi[0] - xi[1] + 1.5)
        yhat = H(h1 + h2 - 1.5)
        out.append((h1, h2, yhat))
    return out


def cmd_perceptron_xor():
    print("== 感知机学习规则：y(w·x+b)≤0 时 w+=y·x, b+=y，lr=1、固定顺序 ==")
    for name in ("AND", "OR", "XOR"):
        w, b, first, mistakes = perceptron(name)
        if first:
            print(f" {name:4s} 第 {first} epoch 首次零误分，"
                  f"最终 w=({w[0]:g},{w[1]:g}), b={b:g}")
        else:
            print(f" {name:4s} 100 epoch 未收敛，末尾 5 个 epoch"
                  f" 误分数：{mistakes[-5:]}，最终 w=({w[0]:g},{w[1]:g}), b={b:g}")
    print("== 手工 2-2-1：h1=OR, h2=NAND, out=AND(h1,h2) ==")
    for xi, (h1, h2, yhat) in zip(X, hidden_space("XOR")):
        print(f" x=({int(xi[0])},{int(xi[1])}) → (h1,h2)=({h1},{h2})"
              f" → ŷ={yhat}")


def _panel(ax, title):
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(-.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=10.5, color="#111111")
    for sp in ax.spines.values():
        sp.set_linewidth(.8)


def _points(ax, name):
    y = LABELS[name]
    for xi, yi in zip(X, y):
        ax.plot(xi[0], xi[1], marker="o" if yi < 0 else "s", ms=6,
                mfc="#eef4fb" if yi < 0 else "#fae6d6",
                mec=C["blue"] if yi < 0 else C["orange"], mew=1.4)


def _coef(v, sym, first=False):
    """3x_1 / +2x_2 / -x_2 的简写：系数 1 省略、first 处不写 +。"""
    if v == 0:
        return ""
    s = "" if first else ("+" if v > 0 else "-")
    if first and v < 0:
        s = "-"
    a = abs(v)
    return s + (f"{a:g}" if a != 1 else "") + sym


def _eq(w, b):
    return "$" + _coef(w[0], "x_1", first=True) + _coef(w[1], "x_2") \
        + (f"{b:+g}" if b != 0 else "") + "=0$"


def _line(ax, w, b):
    """画决策线 w·x+b=0，端点裁剪到坐标框内（不越出轴线，免得撞上标题）。"""
    lo, hi = -.23, 1.23
    pts = []
    if abs(w[1]) > 1e-9:
        for xx in (lo, hi):
            yy = -(w[0] * xx + b) / w[1]
            if lo <= yy <= hi:
                pts.append((xx, yy))
        for yy in (lo, hi):
            xx = -(w[1] * yy + b) / w[0] if abs(w[0]) > 1e-9 else None
            if xx is not None and lo <= xx <= hi:
                pts.append((xx, yy))
    else:
        xx = -b / w[0] if abs(w[0]) > 1e-9 else 0
        pts = [(xx, lo), (xx, hi)]
    if len(pts) >= 2:
        ax.plot([pts[0][0], pts[-1][0]], [pts[0][1], pts[-1][1]],
                color=C["blue"], lw=1.4)


def figure_xor():
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.5))
    fig.subplots_adjust(left=.04, right=.98, bottom=.16, top=.72, wspace=.35)
    fig.suptitle("单层感知机与 XOR：加一层隐藏就线性可分",
                 fontsize=10.5, fontweight=600, color="#111111", y=.97)
    for k, name in enumerate(("AND", "OR")):
        w, b, first, _ = perceptron(name)
        _panel(axes[k], f"{name}：第 {first} 轮零误分")
        _points(axes[k], name)
        _line(axes[k], w, b)
        axes[k].text(.04 if name == "AND" else .42, .50 if name == "AND" else .93, _eq(w, b),
                     transform=axes[k].transAxes, fontsize=10.5,
                     ha="left", va="center", color="#555555")
    handles = [Line2D([], [], marker="o", ls="none", ms=6,
                      mfc="#eef4fb", mec=C["blue"], mew=1.4),
               Line2D([], [], marker="s", ls="none", ms=6,
                      mfc="#fae6d6", mec=C["orange"], mew=1.4)]
    fig.legend(handles, ["输出 0", "输出 1"], loc="upper right",
               bbox_to_anchor=(.995, 1.0), fontsize=10.5, frameon=False,
               ncol=2, handlelength=1.2, columnspacing=1.0,
               handletextpad=.4)
    _panel(axes[2], "XOR：不存在一条直线")
    _points(axes[2], "XOR")
    axes[2].plot([-.1, 1.1], [1.2, 0.], color="#c05050", lw=1.2,
                 ls=(0, (4, 3)))
    axes[2].text(.95, .92, "找不出分隔线", fontsize=10.5, color="#c05050",
                 ha="right", transform=axes[2].transAxes)
    ax = axes[3]
    _panel(ax, r"隐藏空间 $(h_1,h_2)$")
    ax.set_xlabel(r"$h_1$（OR）", fontsize=10.5)
    ax.set_ylabel(r"$h_2$（NAND）", fontsize=10.5)
    for (h1, h2, yhat), xi in zip(hidden_space("XOR"), X):
        m = "o" if yhat == 0 else "s"
        col = C["blue"] if yhat == 0 else C["orange"]
        ax.plot(h1 + .03, h2 + .02, marker=m, ms=6,
                mfc="#eef4fb" if yhat == 0 else "#fae6d6",
                mec=col, mew=1.4)
    ax.text(1.02, 1.14, "两点重合", fontsize=10.5, ha="right",
            color="#555555")
    ax.plot([.27, 1.23], [1.5 - .27, 1.5 - 1.23], color=C["blue"], lw=1.4)
    ax.text(.62, .12, r"$h_1+h_2=1.5$", fontsize=10.5, ha="left",
            color=C["blue"])
    save(fig, "dl-evo-perceptron-xor")


def figures():
    figure_xor()


COMMANDS = {"perceptron-xor": cmd_perceptron_xor,
            "figures": figures}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "perceptron-xor"
    COMMANDS[cmd]()
