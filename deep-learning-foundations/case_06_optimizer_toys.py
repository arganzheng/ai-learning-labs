"""深度学习基础（06）：优化器 配套玩具与配图。
https://arganzheng.life/optimizers-from-sgd-to-adamw.html

用法：python case_06_optimizer_toys.py one-step|bias-correction|grad-accum|warmup|figures

四个玩具（NumPy CPU，固定种子）：
  one-step        一个标量参数、四步梯度，SGD / Momentum / Adam / AdamW 逐中间量
  bias-correction beta2=0.999、常数梯度下前 20 步 v_t 与 v_hat_t 对 g^2 的比值
  grad-accum      batch 64 一步 vs 4x16 累积一步的参数差（loss 除不除 k）
  warmup          与第九章同配置（64 层 Pre-Norm + Adam b2=0.95），
                  前 20 步 max|dθ|/lr 实测——配图用第九章的 dl-case-03-warmup.svg
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
plt.rcParams["axes.unicode_minus"] = False

from dlf import data, nn                       # noqa: E402
from dlf.layers import make_deep_mlp           # noqa: E402
from dlf.optim import Adam, run                # noqa: E402


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


# ---------------------------------------------------------------- toys

LR, B1, B2, LAM, EPS = 0.1, 0.9, 0.999, 0.1, 1e-8
GRADS = [0.5, 0.5, -0.1, 0.4]


def one_step_rows():
    """四种优化器在同一个标量参数上的逐中间量；返回 {name: [step dicts]}。"""
    th = {"SGD": 1.0, "Momentum": 1.0, "Adam": 1.0, "AdamW": 1.0}
    m = {"Momentum": 0.0, "Adam": 0.0, "AdamW": 0.0}
    v = {"Adam": 0.0, "AdamW": 0.0}
    rows = {k: [] for k in th}
    for t, g in enumerate(GRADS, 1):
        # SGD
        d = -LR * g
        th["SGD"] += d
        rows["SGD"].append(dict(t=t, g=g, d=d, th=th["SGD"]))
        # Momentum: m <- beta m + g（正文第三章的形式）
        m["Momentum"] = B1 * m["Momentum"] + g
        d = -LR * m["Momentum"]
        th["Momentum"] += d
        rows["Momentum"].append(dict(t=t, g=g, m=m["Momentum"], d=d,
                                     th=th["Momentum"]))
        # Adam / AdamW
        for name in ("Adam", "AdamW"):
            m[name] = B1 * m[name] + (1 - B1) * g
            v[name] = B2 * v[name] + (1 - B2) * g * g
            mh = m[name] / (1 - B1 ** t)
            vh = v[name] / (1 - B2 ** t)
            upd = mh / (np.sqrt(vh) + EPS)
            wd = LR * LAM * th[name] if name == "AdamW" else 0.0
            d = -wd - LR * upd
            th[name] += d
            rows[name].append(dict(t=t, g=g, m=m[name], v=v[name],
                                   mh=mh, vh=vh, upd=upd, wd=wd, d=d,
                                   th=th[name]))
    return rows


def toy_one_step():
    print(f"一个标量参数 theta=1.0，梯度序列 {GRADS}，"
          f"lr {LR}，beta1 {B1}，beta2 {B2}，lambda {LAM}，eps {EPS}")
    rows = one_step_rows()
    r = rows["SGD"][0]
    print(f"\n== 第 1 步（g = {r['g']}）==")
    print(f"  SGD      : dθ = -lr·g            = {r['d']:+.4f}   θ -> {r['th']:.4f}")
    r = rows["Momentum"][0]
    print(f"  Momentum : m = b1·m0 + g = {r['m']:.4f}，dθ = -lr·m = {r['d']:+.4f}"
          f"   θ -> {r['th']:.4f}")
    r = rows["Adam"][0]
    print(f"  Adam     : m = (1-b1)g = {r['m']:.4f}，v = (1-b2)g^2 = {r['v']:.2e}")
    print(f"             m_hat = g = {r['mh']:.4f}，v_hat = g^2 = {r['vh']:.4f}，"
          f"dθ = -lr·g/|g| = {r['d']:+.4f}   θ -> {r['th']:.4f}")
    r = rows["AdamW"][0]
    print(f"  AdamW    : Adam 项 {(-LR * r['upd']):+.4f} + wd 支路 -lr·λ·θ"
          f" = {(-r['wd']):+.4f}，dθ = {r['d']:+.4f}   θ -> {r['th']:.4f}")
    print(f"  对照：无偏差修正的 Adam 第 1 步 dθ = -lr·(1-b1)/sqrt(1-b2)·sign(g)"
          f" = {(-LR * (1 - B1) / np.sqrt(1 - B2)):+.4f}（{ (1 - B1) / np.sqrt(1 - B2):.2f} 倍）")
    print("\n== 四步完整轨迹（t, g, m, v, m_hat, v_hat, dθ, θ）==")
    for name in ("SGD", "Momentum", "Adam", "AdamW"):
        print(f"  {name}:")
        for r in rows[name]:
            if name == "SGD":
                print(f"    t{r['t']}  g {r['g']:+.2f}  dθ {r['d']:+.4f}"
                      f"  θ {r['th']:.4f}")
            elif name == "Momentum":
                print(f"    t{r['t']}  g {r['g']:+.2f}  m {r['m']:+.4f}"
                      f"  dθ {r['d']:+.4f}  θ {r['th']:.4f}")
            else:
                print(f"    t{r['t']}  g {r['g']:+.2f}  m {r['m']:+.4f}"
                      f"  v {r['v']:.2e}  m_hat {r['mh']:+.4f}"
                      f"  v_hat {r['vh']:.4f}  dθ {r['d']:+.4f}  θ {r['th']:.4f}")
    a = rows["Adam"][0]
    print(f"\n  读法：第 1 步 |dθ| = lr（{abs(a['d']):.4f} ≈ {LR}）与 g 的大小无关——"
          f"偏差修正让 m_hat=g、v_hat=g^2，归一化直接把步长顶满；"
          f"这就是大学习率下 warmup 几乎不能省的原因。")


def toy_bias_correction():
    g = 1.0
    print(f"beta2={B2}、常数梯度 g={g}：前 20 步 v_t 与 v_hat_t 对 g^2 的比值")
    print(f"  v_t = g^2(1 - b2^t)，v_hat_t = g^2（修正后恒等于真值）")
    for t in (1, 2, 5, 10, 20):
        vt = g * g * (1 - B2 ** t)
        print(f"  t={t:2d}   v_t/g^2 = {vt:.4f}   v_hat_t/g^2 = 1.0000")
    print("  读法：t=1 时未修正的 v 只有 0.001·g^2——偏小 1000 倍，"
          "直接拿去归一化会把步长放大 1/sqrt(0.001) ≈ 31.6 倍；"
          "β2=0.999 下 v 要 ~1000 步才爬到真值附近，"
          "所以开头的偏差修正是 Adam 的关键一行。")


def toy_grad_accum():
    rng = np.random.default_rng(0)
    X, y = data.load_mnist("train")
    net = nn.MLP([784, 256, 10], rng)
    xb, yb = X[:64], y[:64]
    # batch 64 一步
    loss_full, d, _ = nn.softmax_ce(net.forward(xb), yb)
    net.backward(d)
    g_full = [g.copy() for _, g in net.params()]
    # 4 x 16 累积，loss / 4
    acc = [np.zeros_like(g) for g in g_full]
    for i in range(4):
        loss_i, d, _ = nn.softmax_ce(net.forward(xb[i * 16:(i + 1) * 16]),
                                     yb[i * 16:(i + 1) * 16])
        net.backward(d / 4)
        for a, (_, g) in zip(acc, net.params()):
            a += g
    # 忘了除 k 的版本
    acc_nodiv = [np.zeros_like(g) for g in g_full]
    for i in range(4):
        _, d, _ = nn.softmax_ce(net.forward(xb[i * 16:(i + 1) * 16]),
                                yb[i * 16:(i + 1) * 16])
        net.backward(d)
        for a, (_, g) in zip(acc_nodiv, net.params()):
            a += g
    d_acc = max(np.abs(a - b).max() for a, b in zip(acc, g_full))
    ratio = max(np.linalg.norm(a) / np.linalg.norm(b)
                for a, b in zip(acc_nodiv, acc))
    lr = 0.01
    p_diff = max(np.abs((P - lr * a) - (P - lr * b)).max()
                 for (P, _), a, b in zip(net.params(), acc, g_full))
    print(f"两层 MLP 784-256-10，batch 64 一次反向 vs 4×16 累积（loss/4）")
    print(f"  累积梯度 vs 一次梯度：max|diff| = {d_acc:.3e}（fp32 舍入量级）")
    print(f"  lr {lr} 一步后参数最大差 = {p_diff:.3e}")
    print(f"  忘了除以 k=4：梯度变 {ratio:.3f} 倍"
          f"（sum 而非 mean，恰好 k 倍），等价于学习率乘了 4")
    print("  读法：每个 micro-batch 的 loss 要先除以 k 再 backward，"
          "累积的才是平均梯度；DDP 各 rank 做的是同样的平均。")


def toy_warmup():
    """与第九章同一配置（64 层 Pre-Norm + Adam b2=0.95、峰值 lr 1e-2）。
    贴前 20 步的 max|dθ|/lr_t 与 loss；曲线见正文第九章 dl-case-03-warmup.svg。"""
    Xs, ys, _, _ = data.load_mnist_standardized()
    print("64 层 Pre-Norm + Adam（beta2=0.95），峰值 lr 1e-2，"
          "warmup 0 vs 100 步，前 20 步实测：")
    for wu in (0, 100):
        rng = np.random.default_rng(0)
        net = make_deep_mlp("prenorm", rng=rng)
        opt = Adam(net.params(), 1e-2, b2=0.95)
        peak = opt.lr
        upds, losses = [], []
        for s in range(200):
            opt.lr = peak if wu == 0 else peak * min(1.0, (s + 1) / wu)
            idx = rng.integers(0, Xs.shape[0], 128)
            loss, d, _ = nn.softmax_ce(net.forward(Xs[idx]), ys[idx])
            net.backward(d)
            opt.step()
            upds.append(opt.last_update_max)
            losses.append(loss)
        print(f"\n  warmup {wu} 步（lr_t 为当时的学习率）：")
        for s in (0, 1, 2, 4, 9, 19):
            print(f"    步 {s + 1:3d}   lr_t = {peak * opt_ratio(s, wu):.4f}"
                  f"   max|dθ|/lr_t = {upds[s]:.3f}"
                  f"   loss = {losses[s]:.3f}")
        print(f"    loss @20 {losses[19]:.3f}  @100 {losses[99]:.3f}"
              f"  @200 {losses[199]:.3f}")


def opt_ratio(s, warmup):
    """第 s 步 lr_t / lr_peak。"""
    return 1.0 if warmup == 0 else min(1.0, (s + 1) / warmup)


# ---------------------------------------------------------------- figures

def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    text(ax, .15, ylim[1] - .3, title, fontsize=10.5, fontweight=600,
         color="#111111")
    return fig, ax


def figure_one_step():
    rows = one_step_rows()
    fig, ax = base_fig("同一个梯度 g = 0.500 的第 1 步：四种优化器的中间量与 dθ（lr 0.1）",
                       h=4.0, xlim=(0, 12), ylim=(0, 6.9))
    cols = dict(g=(1.95, 1.5), mid=(4.2, 2.5), hat=(7.4, 2.5), d=(10.25, 1.7))
    ch = .85
    rows_y = {"SGD": 6.0, "Momentum": 4.75, "Adam": 3.5, "AdamW": 2.15}
    for name, yc in rows_y.items():
        y0 = yc - ch / 2
        text(ax, .1, yc, name, fontsize=9, color="#111111")
        box(ax, cols["g"][0], y0, cols["g"][1], ch,
            f"g = {rows[name][0]['g']:.3f}", fs=8)
        r = rows[name][0]
        if name == "SGD":
            box(ax, cols["d"][0], y0, cols["d"][1], ch,
                f"dθ = {r['d']:+.3f}", fs=8, fc="#fde9d9", ec="#c98a4a")
            arrow(ax, (cols["g"][0] + cols["g"][1], yc), (cols["d"][0], yc))
            text(ax, 6.5, yc + .18, "-lr·g：步长与梯度成正比", fontsize=8,
                 ha="center", color="#555555")
        elif name == "Momentum":
            box(ax, cols["mid"][0], y0, cols["mid"][1], ch,
                f"m = β1·m0 + g\n= {r['m']:.3f}", fs=8)
            box(ax, cols["d"][0], y0, cols["d"][1], ch,
                f"dθ = {r['d']:+.3f}", fs=8, fc="#fde9d9", ec="#c98a4a")
            arrow(ax, (cols["g"][0] + cols["g"][1], yc), (cols["mid"][0], yc))
            arrow(ax, (cols["mid"][0] + cols["mid"][1], yc), (cols["d"][0], yc))
            text(ax, 8.2, yc + .18, "-lr·m", fontsize=8,
                 ha="center", color="#555555")
        else:
            box(ax, cols["mid"][0], yc - .5, cols["mid"][1], 1.0,
                f"m = {r['m']:.3f}\nv = {r['v']:.1e}", fs=8)
            box(ax, cols["hat"][0], yc - .5, cols["hat"][1], 1.0,
                f"m_hat = {r['mh']:.3f}\nv_hat = {r['vh']:.3f}", fs=8)
            box(ax, cols["d"][0], y0, cols["d"][1], ch,
                f"dθ = {r['d']:+.3f}", fs=8, fc="#fde9d9", ec="#c98a4a")
            arrow(ax, (cols["g"][0] + cols["g"][1], yc), (cols["mid"][0], yc))
            arrow(ax, (cols["mid"][0] + cols["mid"][1], yc), (cols["hat"][0], yc))
            arrow(ax, (cols["hat"][0] + cols["hat"][1], yc), (cols["d"][0], yc))
    # AdamW 的 wd 支路：不进矩，单独一条
    yc = rows_y["AdamW"]
    box(ax, cols["mid"][0], yc - 1.5, cols["hat"][0] + cols["hat"][1]
        - cols["mid"][0], .95,
        "wd 支路：-lr·λ·θ = -0.010\n（不进矩、不被 sqrt(v) 缩放）",
        fs=7.5, fc="#eef4fb", ec="#7a9ac0")
    arrow(ax, (cols["hat"][0] + 2.2, yc - 0.53),
          (cols["d"][0] + cols["d"][1] / 2, yc - ch / 2), color="#7a9ac0")
    text(ax, cols["d"][0] + cols["d"][1] / 2, yc - 1.05,
         "dθ = Adam 项 -0.100\n+ wd 支路 -0.010", fontsize=7.5,
         ha="center", va="top", color="#555555")
    finish(fig, "dl-opt-one-step")


def figure_bias_correction():
    g = 1.0
    ts = np.arange(1, 21)
    v = 1 - B2 ** ts
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(ts, v, c=C["red"], lw=1.4, label="v_t / g^2（未修正）")
    ax.plot(ts, np.ones_like(ts), c=C["green"], lw=1.4, ls="--",
            label="v_hat_t / g^2（修正后）")
    ax.set_yscale("log")
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["1e-3", "1e-2", "1e-1", "1"], fontsize=8)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("步 t"); ax.set_ylabel("对真值 g^2 的比值（对数）")
    ax.set_title("beta2 = 0.999、常数梯度：未修正的 v 从 0.001·g^2 慢慢爬",
                 fontweight=600, color="#111111")
    ax.text(2.4, 1.2e-2, "t=1：v = 0.001·g^2", fontsize=8, color=C["red"])
    ax.add_patch(FancyArrowPatch((2.2, 1e-2), (1.15, 1.4e-3),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=C["red"], lw=.8,
                                 shrinkA=1, shrinkB=1))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    finish(fig, "dl-opt-bias-correction")


def figures():
    figure_one_step(); figure_bias_correction()
    print("figures written to blog img/in-post")


TOYS = {"one-step": toy_one_step, "bias-correction": toy_bias_correction,
        "grad-accum": toy_grad_accum, "warmup": toy_warmup}

if __name__ == "__main__":
    np.seterr(all="ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures()
    elif len(sys.argv) > 1 and sys.argv[1] in TOYS:
        TOYS[sys.argv[1]]()
    else:
        for t in TOYS.values():
            t()
