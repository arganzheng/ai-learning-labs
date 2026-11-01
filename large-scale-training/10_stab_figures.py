"""《训练稳定性与数据管线》配图：_plot.py 风格，collide()=0。
数据均来自 10_stab_toys.py 的实测输出（expected/*.json）。

F1 spike-rollback：loss + grad-norm 两联，坏窗口阴影，回退点与恢复后曲线（§四）
F2 clip-2d：全局范数裁剪二维示意 + toy 数字（§三.1）
F3 logit-qkln：带/不带 QK-LayerNorm 的 max|logit| 曲线（§三.3）
F4 packing-vs-padding：pad-to-max vs 贪心打包块图 + cu_seqlens（§六.5）
F5 resume-number-line：样本号数轴上的整数位置（§七.3）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import C, box, plt, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#dce9f7"
ORANGE = "#fae6d6"
GREEN = "#dcefe2"
PURPLE = "#eee1f3"
RED = "#f5dede"
GRAY = "#e8e8e8"


def _canvas(w=7.6, h=4.2, xlim=(0, 12), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _label(ax, x, y, s, size=10, color="#111111", ha="center", va="center",
           in_box=None, weight=None, **kw):
    return text(ax, x, y, s, fontsize=size, color=color, ha=ha, va=va,
                in_box=in_box, fontweight=weight, **kw)


def _card(ax, x, y, w, h, s="", fc="white", ec="#777777", size=10, lw=0.9):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if s:
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center",
             fontsize=size)
    return p


def _arrow(ax, a, b, color="#555555", lw=1.2, ms=11):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                mutation_scale=ms))


def _load(name):
    return json.load(open(os.path.join(HERE, "expected", name)))


def spike_rollback():
    """F1：run1（带 spike）与 run2（回退+跳过）的两联曲线。"""
    d = _load("10_stab_spike_curve.json")
    bad0, bad1 = d["bad"]
    ckpt = d["ckpt"]
    s1 = sorted((int(s), *v) for s, v in d["run1"].items())
    s2 = sorted((int(s), *v) for s, v in d["run2"].items())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 4.6))
    for ax, name, i in ((ax1, "loss", 1), (ax2, "grad norm（clip 前）", 2)):
        ax.axvspan(bad0, bad1 - 1, color="#f5dede", alpha=0.5, zorder=0)
        ax.plot([r[0] for r in s1], [r[i] for r in s1],
                color=C["blue"], lw=1.4, label="不处理（照常喂坏 batch）")
        ax.plot([r[0] for r in s2], [r[i] for r in s2],
                color=C["orange"], lw=1.4, ls="--",
                label="回退 + 跳过坏窗口")
        ax.axvline(ckpt, color="#8a8a8a", lw=0.9, ls=":")
        ax.set_ylabel(name, fontsize=10)
        ax.tick_params(labelsize=9.5)
        ax.set_xlim(120, 210)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    for ax, ticks in ((ax1, [0.01, 0.1, 1, 10]), (ax2, [0.01, 0.1, 1, 10, 100])):
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
    fig.subplots_adjust(hspace=0.45)
    ax1.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax1.text(ckpt - 2, ax1.get_ylim()[1] * 0.9, "step-140\nckpt", fontsize=9.5,
             ha="right", va="top", color="#555555")
    ax2.text((bad0 + bad1) / 2 - 0.5, ax2.get_ylim()[1] * 0.9, "坏窗口\n150–155",
             fontsize=9.5, ha="right", va="top", color="#8a3c3c")
    ax2.set_xlabel("step", fontsize=10)
    fig.suptitle("坏 batch 注入 → 回退 + 跳过的恢复轨迹（toy 实测，2 层 d=64）",
                 fontsize=12, fontweight=600)
    save(fig, "training-stability-spike-rollback")


def clip_2d():
    """F2：二维梯度裁剪：原始 g 长箭头，clip 后落到半径 1 的圆上。"""
    fig, ax = _canvas(7.6, 3.6, (0, 12), (0, 5.4))
    _label(ax, 6, 5.15, "全局范数裁剪：方向不变，整体缩到阈值（§三.1，max_norm=1.0）",
           12.5, weight="bold")
    cx, cy, r = 4.2, 2.5, 1.55
    circle = plt.Circle((cx, cy), r, fc="none", ec="#999999", ls="--", lw=1.0)
    ax.add_patch(circle)
    _label(ax, cx + r * 0.72, cy - r * 0.72, "‖g‖ = 1.0", 9.5, color="#555555")
    # 原始梯度（3315 → 归一化成图上的长度）
    dx, dy = 3.6, 1.15
    _arrow(ax, (cx, cy), (cx + dx, cy + dy), color=C["red"], lw=2.2, ms=15)
    _label(ax, cx + dx / 2 - 0.9, cy + dy / 2 + 0.55,
           "g 原始 ‖g‖ = 3315.4", 10, color=C["red"])
    k = r / (dx ** 2 + dy ** 2) ** 0.5
    _arrow(ax, (cx, cy), (cx + dx * k, cy + dy * k), color=C["blue"], lw=2.2, ms=15)
    _label(ax, cx + dx * k / 2 + 1.3, cy + dy * k / 2 - 0.62,
           "clip 后 ‖g‖ = 1.0", 10, color=C["blue"])
    _label(ax, 9.3, 3.4, "标量系数 c/(‖g‖+ε)", 10, ha="center")
    _label(ax, 9.3, 2.95, "= 1/3315.4 = 0.000302", 10, ha="center")
    _label(ax, 9.3, 2.4, "每个参数张量被", 9.5, ha="center", color="#555555")
    _label(ax, 9.3, 2.05, "同一个标量缩放", 9.5, ha="center", color="#555555")
    _label(ax, 9.3, 1.55, "→ 梯度方向不变", 10, ha="center", weight="bold")
    _label(ax, 6, 0.85, "范数 ‖g‖ 是全模型的：各卡先算自己分片的平方和，", 9.5)
    _label(ax, 6, 0.5, "在恰好覆盖所有分片一次的组上 all-reduce 再开方（§三.1 三框架实现）。", 9.5)
    save(fig, "training-stability-clip-2d")


def logit_qkln():
    """F3：带/不带 QK-LayerNorm 的 max|logit|。"""
    d = _load("10_stab_logit_curve.json")
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    steps = range(len(d["false"]))
    ax.plot(steps, d["false"], color=C["red"], lw=1.6,
            label="不带 QK-LayerNorm")
    ax.plot(steps, d["true"], color=C["blue"], lw=1.6,
            label="带 QK-LayerNorm")
    ax.set_xlabel("step", fontsize=10)
    ax.set_ylabel("max |q·k/√d|", fontsize=10)
    ax.tick_params(labelsize=9.5)
    ax.legend(fontsize=9.5, frameon=False, loc="center left")
    ax.set_ylim(0, 5.5)
    ax.text(58, 5.2, "（bf16 危险线 ~100 远在图上沿之上，未画出）", fontsize=9,
            ha="right", color="#777777")
    ax.set_title("注意力 logit 最大值：QK-LayerNorm 把它钉住（toy 实测，单层注意力，SGD lr=0.5）",
                 fontsize=11.5, fontweight=600)
    save(fig, "training-stability-logit-qkln")


def packing_vs_padding():
    """F4：pad-to-max vs 贪心打包，块图 + 利用率 + 行0 cu_seqlens。"""
    d = _load("10_stab_packing.json")
    lens, bins, cap = d["lens"], d["bins"], d["cap"]
    pad_used, pack_used = d["pad_used"], d["pack_used"]
    tot = sum(lens)
    fig, ax = _canvas(7.6, 4.9, (0, 12), (0, 7.6))
    _label(ax, 6, 7.35, "padding vs packing：12 条长尾序列，容量 4096（toy 实测）",
           12.5, weight="bold")
    x0, w = 1.7, 8.6

    def sx(t):
        return x0 + t / cap * w
    # --- 上：pad-to-max ---
    _label(ax, 1.38, 6.6, "pad-to-max", 10, ha="right", weight="bold")
    from matplotlib.patches import Rectangle
    for i, l in enumerate(lens[:4]):
        y = 6.35 - i * 0.62
        p = _card(ax, sx(0), y, l / cap * w, 0.5, "", fc=BLUE, ec=C["blue"],
                  lw=0.6)
        if l / cap * w > 1.4:
            _label(ax, sx(l / 2), y + 0.25, str(l), 9.5, in_box=p)
        pw = (max(lens) - l) / cap * w
        pp = _card(ax, sx(l), y, pw, 0.5, "", fc="white", ec="#bbbbbb", lw=0.6)
        ax.add_patch(Rectangle((sx(l), y), pw, 0.5, fc="none", ec="none",
                               hatch="///", lw=0))
        if pw > 1.0:
            _label(ax, sx(l) + pw / 2, y + 0.25, "pad", 9, color="#888888",
                   in_box=pp)
    _label(ax, x0, 3.72, f"斜线 = pad；…共 12 行，每行都 pad 到 {max(lens)}；"
           f"利用率 {tot}/{pad_used} = {tot/pad_used*100:.1f}%",
           10, ha="left", weight="bold", color=C["red"])
    # --- 下：贪心打包 ---
    _label(ax, 1.38, 2.7, "贪心打包", 10, ha="right", weight="bold")
    colors = [BLUE, GREEN, ORANGE, PURPLE, "#d5e8d4", "#f9e2ec"]
    for ri, b in enumerate(bins):
        y = 2.55 - ri * 1.15
        x = sx(0)
        for si, l in enumerate(b):
            wseg = l / cap * w
            p = _card(ax, x, y, wseg, 0.56, "", fc=colors[si % len(colors)],
                      ec="#777777", lw=0.6)
            if wseg > 1.05:
                _label(ax, x + wseg / 2, y + 0.28, str(l), 9, in_box=p)
            x += wseg
        rest = cap - sum(b)
        if rest > 0:
            pp = _card(ax, x, y, rest / cap * w, 0.56, "", fc="white",
                       ec="#bbbbbb", lw=0.6)
            ax.add_patch(Rectangle((x, y), rest / cap * w, 0.56, fc="none",
                                   ec="none", hatch="///", lw=0))
            if rest / cap * w > 1.6:
                _label(ax, x + rest / cap * w / 2, y + 0.28, f"pad {rest}",
                       9, color="#888888", in_box=pp)
        _label(ax, x0, y - 0.32,
               f"行 {ri}：{'+'.join(str(l) for l in b)} = {sum(b)}"
               f"（pad {cap - sum(b)}）", 9, ha="left", color="#444444")
    _label(ax, x0, 0.62, f"利用率 {tot}/{pack_used} = {tot/pack_used*100:.1f}%",
           10, ha="left", weight="bold", color=C["green"])
    row0_cu = [0]
    for l in bins[0]:
        row0_cu.append(row0_cu[-1] + l)
    _label(ax, x0, 0.3, f"行 0 的 cu_seqlens = {row0_cu}（每段 position_ids 0..len-1）",
           9.5, ha="left", color="#333333")
    save(fig, "training-stability-packing-vs-padding")


def resume_number_line():
    """F5：样本号数轴上的整数位置。"""
    fig, ax = _canvas(7.6, 3.4, (0, 12), (0, 4.9))
    _label(ax, 6, 4.62, "数据位置 = 已消费样本数（§七.3）", 12.5, weight="bold")
    x0, x1, y = 1.2, 10.8, 2.9
    ax.plot([x0, x1], [y, y], color="#444444", lw=1.2)
    for st in range(13):
        xx = x0 + st / 12 * (x1 - x0)
        ax.plot([xx, xx], [y - 0.07, y + 0.07], color="#444444", lw=0.9)
        _label(ax, xx, y - 0.32, str(st * 8), 9, color="#555555")
    _label(ax, x0, y - 0.68, "样本号（toy：total=96，B=8，每格一步）", 9, ha="left",
           color="#555555")
    xc = x0 + 56 / 96 * (x1 - x0)
    ax.plot([xc, xc], [y - 0.3, y + 0.3], color=C["red"], lw=1.6)
    _label(ax, xc, y + 0.62, "consumed = 56（7 步 × B=8）", 10,
           color=C["red"], weight="bold")
    # ckpt 标记在整数处
    p = _card(ax, xc - 0.85, y - 1.28, 1.7, 0.5, "", fc="#fff4d6",
              ec=C["orange"], lw=0.8)
    _label(ax, xc, y - 1.03, "ckpt 存这里", 9.5, ha="center", in_box=p)
    ax.plot([xc, xc], [y - 0.78, y - 0.3], color=C["orange"], lw=0.9, ls=":")
    # 恢复的下一个 batch：56..63
    xn0 = x0 + 56 / 96 * (x1 - x0)
    xn1 = x0 + 64 / 96 * (x1 - x0)
    p2 = _card(ax, xn0, y + 1.0, xn1 - xn0, 0.55, "", fc=BLUE, ec=C["blue"],
               lw=0.8)
    _arrow(ax, (xc, y + 0.9), (xn0 + 0.05, y + 1.0), color=C["blue"], lw=1.1)
    _label(ax, xn1 + 0.3, y + 1.28, "恢复后第 1 个 batch = [56..63]，与不中断相同",
           10, ha="left")
    _label(ax, xc - 3.1, y + 0.62, "已消费（不再喂）", 9.5, ha="center",
           color="#555555")
    _label(ax, 6, 1.15, "跳过 n 个 batch = 这个整数加 n·B；换 DP 度 = 分片方式变、全序不变。", 9.5)
    _label(ax, 6, 0.78, "StatefulDataLoader 存的是同一位置的迭代器状态，下一个 batch 同样 [56..63]（不可换 DP）。",
           9.5)
    save(fig, "training-stability-resume-number-line")


def main():
    for f in (spike_rollback, clip_2d, logit_qkln, packing_vs_padding,
              resume_number_line):
        f()
    print("全部 5 张图完成。")


if __name__ == "__main__":
    main()
