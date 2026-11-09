"""深度学习基础（02）：损失函数 配套玩具与配图。
https://arganzheng.life/loss-functions-from-mse-to-dpo.html

用法：python case_02_loss_toys.py lm-init|dpo|figures

两个玩具（NumPy，固定种子）：
  lm-init  均匀初始化时 next-token CE = ln V（V=50257 与 toy V=8）；
           一句话 4 个 token 的逐位置 -log p 与平均
  dpo      一对 (chosen, rejected) 的 log ratio、margin、DPO loss
           与两条 log p 的梯度符号
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


# ---------------------------------------------------------------- toys

# toy V=8、4 个 token 的目标类概率（固定的小数字，方便核对手算）
TOY_PROBS = [0.65, 0.40, 0.82, 0.55]
# DPO 例子：policy 与 reference 的 log p（自定小数字）
LP_W, LPR_W, LP_L, LPR_L, BETA = -1.2, -1.5, -2.0, -1.0, 0.1


def toy_lm_init():
    print("== 均匀初始化时 next-token CE = ln V ==")
    for V in (50257, 8):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 0.02, (64, V))        # 接近均匀的 logits
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        y = rng.integers(0, V, 64)
        ce = -np.log(p[np.arange(64), y]).mean()
        print(f"  V = {V:>6}: 实测 CE = {ce:.3f}，ln V = {np.log(V):.3f}")
    print("  读法：随机初始化输出接近均匀，第一步 loss ≈ ln V——"
          "GPT-2 的 10.82、Llama-3 128256 的 11.76 都是这个数；"
          "第一屏 loss 离它差得远就是初始化或算错。")
    print("\n== 一句话 4 个 token 的逐位置 -log p 与平均（toy V=8）==")
    for i, p in enumerate(TOY_PROBS):
        print(f"  位置 {i + 1}: p = {p:.2f}   -log p = {-np.log(p):.3f}")
    loss = np.mean([-np.log(p) for p in TOY_PROBS])
    print(f"  loss = 平均 = {loss:.3f}   perplexity = e^loss = {np.exp(loss):.2f}")


def toy_dpo():
    rw, rl = LP_W - LPR_W, LP_L - LPR_L
    delta = rw - rl
    z = BETA * delta
    sig = 1 / (1 + np.exp(-z))
    loss = -np.log(sig)
    gw, gl = -BETA * (1 - sig), BETA * (1 - sig)
    print("== DPO：一对 (chosen y_w, rejected y_l) ==")
    print(f"  log πθ(y_w) = {LP_W}，log πref(y_w) = {LPR_W}"
          f"   → log ratio_w = {rw:+.2f}")
    print(f"  log πθ(y_l) = {LP_L}，log πref(y_l) = {LPR_L}"
          f"   → log ratio_l = {rl:+.2f}")
    print(f"  margin = ratio_w - ratio_l = {delta:+.2f}，β = {BETA}")
    print(f"  z = β·margin = {z:.3f}，σ(z) = {sig:.4f}，"
          f"DPO loss = -log σ(z) = {loss:.4f}")
    print(f"  梯度符号：dL/d logπθ(y_w) = {gw:+.4f}（负 → 下降让 log p 升），"
          f"dL/d logπθ(y_l) = {gl:+.4f}（正 → 下降让 log p 降）")
    print("  读法：对 r_θ = β·log(πθ/πref) 做 logistic 回归——"
          "chosen 的 log p 被推高、rejected 的被压低，"
          "β 拉住整体不离 reference 太远。")


# ---------------------------------------------------------------- figures

def figure_overview():
    fig, ax = base_fig("任务 → 输出层 → loss → 对 logit 的梯度（每行一个实测数字例）",
                       h=4.6, xlim=(0, 12.25), ylim=(0, 7.6))
    heads = ("输出层", "loss", "对 logit 的梯度")
    xs = (1.9, 4.9, 8.2); ws = (2.8, 3.1, 4.0)
    for h_, x, w in zip(heads, xs, ws):
        text(ax, x + w / 2, 6.85, h_, fontsize=10.5, ha="center",
             color="#555555")
    rows = [
        ("回归", r"$\hat y$（连续）", r"MSE：$e=\hat y-y=2.0$" + "\n" + r"$L=e^2=4.0$",
         r"$dL/d\hat y=2e=4.0$（随误差线性长）"),
        ("分类", r"softmax，$p_y=0.62$", r"CE：$-\log 0.62=0.478$",
         r"$p-y$：正确类 $-0.38$，逐类分推动"),
        ("语言模型", r"$V=50257$ softmax", r"next-token CE 逐位平均" + "\n" + r"初始 $=\ln V=10.82$",
         r"同 $p-y$；每步在 $V$ 类上推"),
        ("对比学习", r"相似度 $\times\,1/\tau$", r"InfoNCE（$\tau=0.1$）" + "\n" + r"$=-\log e^{s^+}/\sum_j e^{s_j}=5.96$",
         "行内 softmax 差：正例推高、其余压低"),
        ("偏好对齐", r"$\log\pi-\log\pi_{\mathrm{ref}}$ 差", r"DPO $=-\log\sigma(\beta\cdot\mathrm{margin})$" + "\n" + r"$=-\log 0.532=0.630$",
         r"chosen ↑$\beta(1-\sigma)$、rejected ↓同量"),
    ]
    y = 5.9
    for name, c1, c2, c3 in rows:
        text(ax, .15, y + .12, name, fontsize=10.5, va="center", color="#111111")
        for s, x, w in zip((c1, c2, c3), xs, ws):
            box(ax, x, y - .45, w, 1.15, s, fs=10.5)
        y -= 1.25
    finish(fig, "dl-loss-overview")


def figure_lm_tokens():
    fig, ax = base_fig(r"一句话 4 个 token：逐位置 $-\log p$ 求平均就是 loss",
                       h=3.0, xlim=(0, 11.3), ylim=(0.7, 5.6))
    text(ax, .15, 5.0, r"模型对正确 token 的概率 $p$", fontsize=10.5, color="#555555")
    n = len(TOY_PROBS)
    cw, gap = 1.15, .55
    x0 = .3
    for i, p in enumerate(TOY_PROBS):
        x = x0 + i * (cw + gap)
        box(ax, x, 4.15, cw, .6, rf"$p_{i + 1}={p:.2f}$", fs=10.5)
        box(ax, x, 2.55, cw, .95, rf"$-\log p$" + "\n" + rf"${-np.log(p):.3f}$", fs=10.5,
            fc="#eef4fb", ec="#7a9ac0")
        arrow(ax, (x + cw / 2, 4.15), (x + cw / 2, 3.55))
        arrow(ax, (x + cw / 2, 2.55), (x + cw / 2, 2.4))
    ax.plot([x0 + cw / 2, 9.2], [2.4, 2.4], color="#555555", lw=1.0)
    arrow(ax, (9.2, 2.4), (9.2, 2.85))
    loss = np.mean([-np.log(p) for p in TOY_PROBS])
    box(ax, 8.1, 2.85, 2.2, .9, rf"loss = 平均" + "\n" + rf"$={loss:.3f}$", fs=10.5,
        fc="#fde9d9", ec="#c98a4a")
    text(ax, 8.1, 2.1, rf"perplexity $=e^{{L}}={np.exp(loss):.2f}$",
         fontsize=10.5, color="#555555")
    text(ax, .3, 1.1, r"初始：logits 接近均匀 → 每个位置 $-\log(1/V)=\ln V$"
         r"（GPT-2 $V=50257$ → $10.82$）", fontsize=10.5, color="#555555")
    finish(fig, "dl-loss-lm-tokens")


def figure_dpo():
    rw, rl = LP_W - LPR_W, LP_L - LPR_L
    delta, z = rw - rl, BETA * (rw - rl)
    sig = 1 / (1 + np.exp(-z)); loss = -np.log(sig)
    fig, ax = base_fig(r"DPO 一对样本：$\log$ ratio 差 $\times\,\beta$ 进 sigmoid 就是 loss",
                       h=3.7, xlim=(0, 12), ylim=(0.5, 6.6))
    for i, (name, lp, lpr, r, col, note) in enumerate((
            ("chosen", LP_W, LPR_W, rw, C["green"], r"梯度为负 → $\log p$ 被推高 ↑"),
            ("rejected", LP_L, LPR_L, rl, C["red"], r"梯度为正 → $\log p$ 被压低 ↓"))):
        y = 5.0 - i * 1.9
        text(ax, .1, y + .15, name, fontsize=10.5, color=col)
        box(ax, 1.55, y - .35, 3.6, .95,
            rf"$\log\pi_\theta-\log\pi_{{\mathrm{{ref}}}}$" + "\n" + rf"$={lp}-({lpr})$", fs=10.5)
        box(ax, 5.6, y - .35, 1.6, .95, "ratio" + "\n" + rf"${r:+.2f}$", fs=10.5)
        arrow(ax, (5.15, y + .12), (5.6, y + .12))
        text(ax, 7.85, y + .12, note, fontsize=10.5, color="#555555")
    box(ax, 5.6, 1.55, 2.45, .85, rf"margin $={delta:+.2f}$" + "\n" + rf"$\beta\cdot$margin $={z:.3f}$",
        fs=10.5, fc="#eef4fb", ec="#7a9ac0")
    arrow(ax, (6.9, 4.65), (8.0, 2.42))     # chosen ratio ↓ 合流
    arrow(ax, (6.4, 2.75), (6.4, 2.42))     # rejected ratio ↓ 合流（相减）
    box(ax, 8.45, 1.55, 1.5, .85, rf"$\sigma({z:.2f})$" + "\n" + rf"$={sig:.3f}$", fs=10.5)
    arrow(ax, (8.05, 1.97), (8.45, 1.97))
    box(ax, 10.2, 1.55, 1.7, .85, "loss" + "\n" + rf"$={loss:.3f}$", fs=10.5,
        fc="#fde9d9", ec="#c98a4a")
    arrow(ax, (9.95, 1.97), (10.2, 1.97))
    text(ax, 8.05, 1.0, r"$-\log\sigma$：margin 越大 loss 越小", fontsize=10.5,
         color="#555555")
    finish(fig, "dl-loss-dpo")


def figures():
    figure_overview(); figure_lm_tokens(); figure_dpo()
    print("figures written to blog img/in-post")


TOYS = {"lm-init": toy_lm_init, "dpo": toy_dpo}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures()
    elif len(sys.argv) > 1 and sys.argv[1] in TOYS:
        TOYS[sys.argv[1]]()
    else:
        for t in TOYS.values():
            t()
