"""深度学习基础（07）：正则化与泛化 配套玩具与配图。
https://arganzheng.life/regularization-and-generalization.html

用法：python case_07_reg_toys.py wd-poly|dropout|figures

两个玩具（NumPy，固定种子）：
  wd-poly  10 个带噪点（y = sin(2πx) + N(0, 0.2^2)）的 9 次多项式
           最小二乘 + L2，λ = 0 / 1e-3 / 1e-1 三档：训练/测试 MSE 与 ‖w‖
  dropout  inverted dropout p=0.5：4×8 全 1 输入的 mask、输出（存活=2.0）、
           大样本输出均值与 eval 输出
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


# ---------------------------------------------------------------- wd-poly

def _poly_data():
    rng = np.random.default_rng(4)
    x_tr = np.sort(rng.uniform(0.02, 0.98, 10))
    y_tr = np.sin(2 * np.pi * x_tr) + rng.normal(0, 0.2, 10)
    x_te = np.linspace(0, 1, 301)
    y_te = np.sin(2 * np.pi * x_te)
    return x_tr, y_tr, x_te, y_te


def _fit_poly(x, y, deg, lam):
    A = np.vander(x, deg + 1, increasing=True)
    return np.linalg.solve(A.T @ A + lam * np.eye(deg + 1), A.T @ y)


def poly_rows():
    x_tr, y_tr, x_te, y_te = _poly_data()
    rows = []
    for lam in (0.0, 1e-3, 1e-1):
        w = _fit_poly(x_tr, y_tr, 9, lam)
        yh_tr = np.vander(x_tr, 10, increasing=True) @ w
        yh_te = np.vander(x_te, 10, increasing=True) @ w
        rows.append((lam, float(np.mean((yh_tr - y_tr) ** 2)),
                     float(np.mean((yh_te - y_te) ** 2)),
                     float(np.linalg.norm(w)), w))
    return rows


def toy_wd_poly():
    print("== 10 个带噪点、9 次多项式、L2 三档 ==")
    print("  数据：y = sin(2πx) + N(0, 0.2^2)，训练/测试 MSE 与 ‖w‖：")
    for lam, tr, te, nw, _ in poly_rows():
        print(f"  λ = {lam:<6}: 训练 MSE = {tr:.4f}，测试 MSE = {te:.4g}，"
              f"‖w‖ = {nw:.4g}")
    print("  读法：λ=0 时 9 次多项式穿过每个含噪点——训练 MSE≈0，但 w 的范数"
          "冲到 4e7、测试 MSE 爆炸；λ=1e-3 把 ‖w‖ 压到 10，测试 MSE 最小；"
          "λ=1e-1 压过头，函数太平、训练 MSE 也涨（欠拟合）。")


# ---------------------------------------------------------------- dropout

P_DROP = 0.5
MASK_SEED = 7


def dropout_mask():
    rng = np.random.default_rng(MASK_SEED)
    return (rng.random((4, 8)) >= P_DROP).astype(float)


def toy_dropout():
    x = np.ones((4, 8))
    m = dropout_mask()
    out = x * m / (1 - P_DROP)
    print("== inverted dropout：x 全 1、p = 0.5，训练时输出 = mask × 2 ==")
    print("  mask（1 = 保留）:")
    for r in m.astype(int):
        print("   ", " ".join(map(str, r)))
    print("  训练时输出 = x ⊙ mask / (1-p):")
    for r in out:
        print("   ", " ".join(f"{v:.0f}" for v in r))
    rng = np.random.default_rng(0)
    big = rng.random((20000, 8)) >= P_DROP
    print(f"  大样本训练输出均值 = {big.mean() / (1 - P_DROP):.4f}（≈ 1.0 = 输入）"
          f"   eval 输出 = x = 全 1")
    print("  读法：×1/(1-p) 让训练时的期望输出等于推理时的输入——"
          "幸存单元放大一倍、被丢的置零，推理不做 mask 也不缩放。")


# ---------------------------------------------------------------- figures

def sci(v, math=True):
    """大数科学计数法（math=True 返回 $…$ 段；mathtext 内使用时传 False），小数直接十进制。"""
    if v == 0 or (1e-2 <= abs(v) < 1e4):
        return f"{v:.3g}"
    m, e = f"{v:.3g}".split("e")
    s = rf"{m}\times10^{{{int(e)}}}"
    return f"${s}$" if math else s


def figure_weight_decay():
    x_tr, y_tr, x_te, y_te = _poly_data()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-2.0, 2.0)
    ax.set_xticks([0, .5, 1.0]); ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r"x", fontsize=10.5); ax.set_ylabel(r"y", fontsize=10.5)
    ax.tick_params(labelsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa"); ax.spines["bottom"].set_color("#aaaaaa")
    ax.plot(x_te, y_te, color="#999999", lw=1.2, ls="--",
            label=r"$\sin(2\pi x)$（真函数）")
    cols = {0.0: C["red"], 1e-3: C["green"], 1e-1: C["blue"]}
    for lam, tr, te, nw, w in poly_rows():
        yh = np.vander(x_te, 10, increasing=True) @ w
        ax.plot(x_te, np.clip(yh, -2.1, 2.1), color=cols[lam], lw=1.6,
                label=rf"$\lambda={lam:g}$：$\|w\|={sci(nw, math=False)}$，"
                      rf"测试MSE={sci(te)}")
    ax.scatter(x_tr, y_tr, s=22, color="#222222", zorder=5,
               label="10 个训练点（含噪声）")
    ax.set_title(r"9 次多项式拟合 10 个带噪点：$\lambda$ 越大曲线越平、"
                 r"$\|w\|$ 越小",
                 fontsize=10.5, fontweight=600, color="#111111", loc="left",
                 pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=10.5, frameon=False)
    save(fig, "dl-reg-weight-decay")


def figure_dropout():
    m = dropout_mask().astype(int)
    out = m * 2
    fig, ax = base_fig(r"inverted dropout：p=0.5，输入全 1 → "
                       r"幸存位置输出 2，被丢位置 0",
                       h=4.1, xlim=(0, 12), ylim=(0, 7.1))
    x0, cw, ch, pitch = 1.9, .72, .55, .62
    rows = ((3.6, "mask\n1 = 保留", m),
            (0.6, "训练时输出", out))
    for y0, name, grid in rows:
        text(ax, .15, y0 + .85, name, fontsize=10.5, color="#555555")
        for i in range(4):
            for j in range(8):
                v = grid[i, j]
                fc = "#eef4fb" if v else "#f5f5f5"
                ec = "#7a9ac0" if v else "#cccccc"
                p = box(ax, x0 + j * (cw + .06), y0 + (3 - i) * pitch, cw, ch,
                        fc=fc, ec=ec)
                text(ax, x0 + j * (cw + .06) + cw / 2,
                     y0 + (3 - i) * pitch + ch / 2, str(v),
                     in_box=p, ha="center", va="center", fontsize=10.5,
                     color="#111111" if v else "#999999")
    for j in range(8):
        arrow(ax, (x0 + j * (cw + .06) + cw / 2, 3.55),
              (x0 + j * (cw + .06) + cw / 2, 3.06), lw=.9)
    text(ax, 8.4, 1.3, "\n".join([
        r"训练：$x\odot m/(1-p)$", r"幸存单元 × 2",
        "期望不变；推理不缩放"]), fontsize=10.5, color="#555555",
        linespacing=1.5)
    box(ax, 8.4, 3.9, 3.5, 1.35, "eval 输出 = 输入\n大样本输出均值 ≈ 1.0",
        fs=10.5, fc="#fde9d9", ec="#c98a4a")
    finish(fig, "dl-reg-dropout")


def figures():
    figure_weight_decay(); figure_dropout()
    print("figures written to blog img/in-post")


TOYS = {"wd-poly": toy_wd_poly, "dropout": toy_dropout}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures()
    elif len(sys.argv) > 1 and sys.argv[1] in TOYS:
        TOYS[sys.argv[1]]()
    else:
        for t in TOYS.values():
            t()
