"""深度学习基础（10）：RNN——从 LSTM 到 attention 的诞生 配套玩具与配图。
https://arganzheng.life/rnn-lstm-and-the-birth-of-attention.html

用法：python case_10_rnn_toys.py bptt-decay|lstm-gate|figures

两个玩具（NumPy / torch CPU，固定种子）：
  bptt-decay  标量 RNN h_t=tanh(w·h_{t-1}+u·x_t)，T=30，
              w∈{0.5,0.9,1.0,1.5} 的 ∂h_T/∂h_t = ∏ w·tanh' 随 T−t 衰减；
              再加 d=16 矩阵版，打印 ‖∂h_T/∂h_t‖ 与 W 谱半径
  lstm-gate   单个 LSTM 单元一步：f=.9、i=.2、g=.5、o=.8、c_{t-1}=1.0
              逐步算 c_t、h_t；f≈1 时 ∂c_T/∂c_t=∏f 的 0.9^30 vs 0.99^30
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


def cell(ax, x, y, w, h, body, fs=8, fc="#eef4fb", ec="#7a9ac0"):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=.9, r=.03)
    text(ax, x + w / 2, y + h / 2, body, in_box=p, ha="center", va="center",
         fontsize=fs)
    return p


# ---------------------------------------------------------------- toys

def cmd_bptt_decay():
    rng = np.random.default_rng(0)
    T, u = 30, 1.0
    x = rng.normal(0, 1, T)
    print("== 标量 RNN：h_t = tanh(w·h_{t-1} + u·x_t)，T=30，u=1 ==")
    for w in (0.5, 0.9, 1.0, 1.5):
        h = np.zeros(T + 1)
        for t in range(T):
            h[t + 1] = np.tanh(w * h[t] + u * x[t])
        # ∂h_T/∂h_t = ∏_{k=t+1..T} w·tanh'(w h_{k-1}+u x_k)
        z = w * h[:T] + u * x
        dtanh = 1 - np.tanh(z) ** 2
        J = np.ones(T + 1)
        for t in range(T - 1, -1, -1):
            J[t] = J[t + 1] * w * dtanh[t]
        line = "  ".join(f"{T - t}:{J[t]:.2e}" for t in
                         (T, T - 5, T - 10, T - 20, T - 30))
        print(f" w={w:<4} ∂h_T/∂h_t 随 T−t：{line}")
    print("== 矩阵版 d=16：J_t = diag(tanh')·W，‖∂h_T/∂h_t‖ 与谱半径 ==")
    d = 16
    for sr in (0.9, 1.0, 1.5):
        W = rng.normal(0, 1, (d, d))
        W *= sr / max(abs(np.linalg.eigvals(W)).real for _ in (0,))
        W *= sr / np.max(np.abs(np.linalg.eigvals(W)))
        h = np.zeros((T + 1, d))
        U = rng.normal(0, .5, (d, d))
        for t in range(T):
            h[t + 1] = np.tanh(h[t] @ W + x[t] * U[0])
        J = np.eye(d)
        norms = {}
        for t in range(T - 1, -1, -1):
            dt = 1 - h[t + 1] ** 2
            J = (np.diag(dt) @ W) @ J
            norms[T - t] = np.linalg.norm(J)
        print(f" 谱半径={sr}: ‖J‖ 距离1={norms[1]:.2e} "
              f"10={norms[10]:.2e} 20={norms[20]:.2e} 30={norms[30]:.2e}")


def cmd_lstm_gate():
    f, i, g, o, c_prev = 0.9, 0.2, 0.5, 0.8, 1.0
    print("== LSTM 单步（门值手工设定）==")
    print(f"  f·c_{{t-1}} = {f}×{c_prev} = {f * c_prev}")
    print(f"  i·g        = {i}×{g} = {i * g}")
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    print(f"  c_t = {f * c_prev} + {i * g} = {c_t}")
    print(f"  h_t = o·tanh(c_t) = {o}×{np.tanh(c_t):.4f} = {h_t:.4f}")
    print("== f≈1 时 ∂c_T/∂c_t = ∏f，30 步：")
    print(f"  0.9^30 = {0.9 ** 30:.4f}   0.99^30 = {0.99 ** 30:.4f}")


# ---------------------------------------------------------------- figures

def figure_bptt_decay():
    rng = np.random.default_rng(0)
    T, u = 30, 1.0
    x = rng.normal(0, 1, T)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="none")
    fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.86)
    for w, col in ((0.5, C["blue"]), (0.9, C["green"]),
                   (1.0, C["orange"]), (1.5, C["red"])):
        h = np.zeros(T + 1)
        for t in range(T):
            h[t + 1] = np.tanh(w * h[t] + u * x[t])
        z = w * h[:T] + u * x
        dtanh = 1 - np.tanh(z) ** 2
        J = np.ones(T + 1)
        for t in range(T - 1, -1, -1):
            J[t] = J[t + 1] * w * dtanh[t]
        ax.semilogy(range(0, 31), J[::-1][:31] if False else
                    [J[T - d] for d in range(31)], lw=1.8, color=col,
                    label=rf"$w={w}$")
    ax.set_xlabel(r"距离 $T-t$")
    ax.set_xlim(0, 30)
    ax.set_yticks([10.0 ** (-3 * k) for k in range(7)])
    ax.set_yticklabels(
        [r"$10^{0}$" if k == 0 else rf"$10^{{-{3 * k}}}$" for k in range(7)],
        fontsize=8)
    ax.set_ylim(1e-20, 3)
    ax.set_ylabel(r"$\partial h_T/\partial h_t$")
    ax.set_title(r"标量 RNN 的 BPTT 梯度：$\prod w\cdot\tanh'$ 随距离衰减",
                 fontsize=10.5, fontweight=600, color="#111111")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.25)
    save(fig, "rnn-bptt-decay")


def figure_lstm_cell():
    f, i, g, o, c_prev = 0.9, 0.2, 0.5, 0.8, 1.0
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    fig, ax = base_fig("LSTM 一个时间步的数字流（门值固定）",
                       h=4.2, xlim=(0, 12.6), ylim=(0.6, 6.6))
    y = 4.9
    ch, cw = 1.0, 1.55
    p0 = cell(ax, .3, y, cw, ch, "$c_{t-1}=1.0$", fs=9)
    t1 = text(ax, 2.35, y - .3, r"$\times f$（$0.9$）",
              fontsize=8, ha="center", color="#555555")
    arrow(ax, (1.9, y + ch / 2), (2.8, y + ch / 2))
    p1 = cell(ax, 2.85, y, cw, ch, f"${f * c_prev:.1f}$", fs=9)
    t2 = text(ax, 4.9, y - .3, r"$+\ i\cdot g$（$0.2\times0.5$）",
              fontsize=8, ha="center", color="#555555")
    arrow(ax, (4.45, y + ch / 2), (5.35, y + ch / 2))
    p2 = cell(ax, 5.4, y, cw, ch, f"$c_t={c_t:.1f}$", fs=9,
              fc="#fae6d6", ec=C["orange"])
    arrow(ax, (7.0, y + ch / 2), (7.9, y + ch / 2))
    t3 = text(ax, 7.45, y - .3, r"$\tanh$", fontsize=8,
              ha="center", color="#555555")
    p3 = cell(ax, 7.95, y, cw, ch, f"${np.tanh(c_t):.3f}$", fs=9)
    arrow(ax, (9.55, y + ch / 2), (10.45, y + ch / 2))
    t4 = text(ax, 10.0, y - .3, r"$\times o$（$0.8$）",
              fontsize=8, ha="center", color="#555555")
    p4 = cell(ax, 10.5, y, cw, ch, f"$h_t={h_t:.2f}$", fs=9,
              fc="#e8f3e8", ec=C["green"])
    # 门来源小注
    text(ax, .3, 3.7,
         r"四门的值由 $h_{t-1}$ 与 $x_t$ 经过 sigmoid/tanh 算出；"
         r"$c$ 走加法通路：$\partial c_t/\partial c_{t-1}=\mathrm{diag}(f)+J_h$，"
         r"其中 $J_h$ 依赖 $h_{t-1}$",
         fontsize=8, color="#555555")
    # 梯度通路
    p5 = cell(ax, .3, 1.9, 2.6, .9, r"$f\approx1$ 时 $c$ 的梯度近乎不衰减",
              fs=8, fc="#eef4fb", ec="#7a9ac0")
    text(ax, 3.3, 2.35,
         r"$0.9^{30}=0.042$；$0.99^{30}=0.740$"
         "（遗忘门偏置初始化为 1 的原因）",
         fontsize=8.5, color="#555555", va="center")
    finish(fig, "rnn-lstm-cell")


def figures():
    figure_bptt_decay(); figure_lstm_cell()


COMMANDS = {"bptt-decay": cmd_bptt_decay,
            "lstm-gate": cmd_lstm_gate,
            "figures": figures}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bptt-decay"
    COMMANDS[cmd]()
