"""《千卡配置与 MFU 调优》配图：统一使用 _plot.py 的字体、画布与碰撞检查。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import C, box, box_title, collide, plt, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#dce9f7"
ORANGE = "#fae6d6"
GREEN = "#dcefe2"
PURPLE = "#eee1f3"
GRAY = "#e8e8e8"
RED = "#f5dede"


def _canvas(w=7.6, h=4.4, xlim=(0, 10), ylim=(0, 6)):
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
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center", fontsize=size)
    return p


def _arrow(ax, a, b, color="#555555", lw=1.2, style="->", ms=11, **kw):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, **kw))


def four_steps():
    """F1 总览：70B 放进 1024 H100 的四步台阶。"""
    fig, ax = _canvas(7.6, 3.6, (0, 12), (0, 5.5))
    _label(ax, 6, 5.2, "70B 放进 1024 张 H100：四步定下五元组", 13, weight="bold")
    steps = [
        ("① TP = 8", "层内切权重\n锁节点内 NVLink", "静态 14.0 GB"),
        ("② PP = 4", "80 层分 4 段\n交错 v = 4", "激活 27 GB"),
        ("③ DP = 32", "补满 1024\nZeRO-1 可重叠", "通信可盖"),
        ("④ CP = 1", "s=8K 用不上\n留作余量", "余量留这"),
    ]
    x0, cw, gap = 0.2, 2.1, 0.35
    for i, (t, why, mem) in enumerate(steps):
        x = x0 + i * (cw + gap)
        p = _card(ax, x, 2.5, cw, 2.0, "",
                  fc=[BLUE, GREEN, ORANGE, PURPLE][i],
                  ec=[C["blue"], C["green"], C["orange"], C["purple"]][i], lw=1.1)
        _label(ax, x + cw / 2, 4.1, t, 11.5, in_box=p, weight="bold")
        _label(ax, x + cw / 2, 3.45, why, 10, color="#333333", in_box=p)
        _label(ax, x + cw / 2, 2.85, mem, 10, color=C["red"], in_box=p)
        if i:
            _arrow(ax, (x - gap + 0.04, 3.5), (x - 0.04, 3.5), color="#888888", lw=1.2, ms=10)
    p = _card(ax, 9.95, 2.5, 1.9, 2.0, "", fc="#ffffff", ec=C["blue"], lw=1.2)
    _arrow(ax, (9.6, 3.5), (9.92, 3.5), color="#888888", lw=1.2, ms=10)
    _label(ax, 10.9, 4.1, "候选 A", 11, in_box=p, weight="bold")
    _label(ax, 10.9, 3.45, "8·4·32·1\nv4 m16", 9.5, in_box=p)
    _label(ax, 10.9, 2.8, "MFU 40–45%", 9.5, color=C["red"], in_box=p)
    _label(ax, 6, 1.85, "候选 B（TP8/PP2/DP64）也过线：每卡多 13 GB 静态、DP 通信翻倍、气泡略低——", 10)
    _label(ax, 6, 1.4, "本篇以 A 为主线，它的显存余量允许以后加 CP 或拉长 s 而不必重新设计。", 10)
    save(fig, "thousand-gpu-configuration-70b-four-steps")


def batch_blocks():
    """F2 global batch = micro-batch × 累积 × DP 的积木。"""
    fig, ax = _canvas(7.6, 3.5, (0, 12), (0, 5.3))
    _label(ax, 6, 5.0, "global batch B = micro-batch b × 梯度累积 m × DP d", 12.5, weight="bold")
    _label(ax, 1.05, 4.2, "一个 micro-batch\nb × s = 8192 token", 9.5)
    _card(ax, 0.35, 2.9, 1.4, 0.85, "s×b\n8K tok", fc=BLUE, ec=C["blue"], size=9.5)
    _arrow(ax, (1.78, 3.3), (2.3, 3.3), color="#555555", lw=1.2)
    _label(ax, 3.7, 4.2, "m = 16 个累积\n= 128K token / 副本", 9.5)
    for i in range(16):
        _card(ax, 2.4 + i * 0.16, 2.9, 0.15, 0.85, "", fc=BLUE, ec="#ffffff", lw=0.3)
    _arrow(ax, (5.0, 3.3), (5.5, 3.3), color="#555555", lw=1.2)
    _label(ax, 7.0, 4.2, "d = 32 个 DP 副本（共 32 列）", 9.5)
    for i in range(8):
        _card(ax, 5.6 + i * 0.16, 2.9, 0.15, 0.85, "", fc=GREEN, ec="#ffffff", lw=0.3)
    _label(ax, 7.15, 3.3, "…", 12, ha="left")
    _arrow(ax, (7.6, 3.3), (8.05, 3.3), color="#555555", lw=1.2)
    _card(ax, 8.15, 2.75, 2.4, 1.1, "B = 512 条\n4.2 M tok / step", fc=ORANGE, ec=C["orange"], size=10)
    _label(ax, 6, 2.15, "B 由训练配方定，卡数变了只能动 d 和 m；", 10)
    _label(ax, 6, 1.7, "PP 下 m 同时是流水 micro-batch 数与梯度累积步数（同一个数）。", 10)
    save(fig, "thousand-gpu-configuration-batch-blocks")


def microbatch_bounds():
    """F3 micro-batch 两头约束区间条。"""
    fig, ax = _canvas(7.6, 3.4, (0, 12), (0, 5.2))
    _label(ax, 6, 4.9, "micro-batch b 的两头约束（b = 1…4，候选 A）", 12.5, weight="bold")
    rows = [
        ("b = 1", 3.7, GREEN, "GEMM M=8192 才 60–70% · 气泡 4.7% · ~49 GB", "可行"),
        ("b = 2", 2.7, ORANGE, "M 翻倍 · m 8 → 气泡 9.4% · ~76 GB 太紧", "勉强"),
        ("b = 4", 1.7, RED, "气泡 18.8% · ~130 GB 爆显存", "不可行"),
    ]
    for name, y, col, why, tag in rows:
        _label(ax, 0.35, y, name, 10.5, ha="left")
        p = _card(ax, 1.45, y - 0.34, 8.3, 0.68, "", fc=col, ec="#999999", lw=0.7)
        _label(ax, 5.6, y, why, 10, in_box=p)
        _card(ax, 10.0, y - 0.32, 1.35, 0.64, tag,
              fc="#ffffff",
              ec=C["green"] if tag == "可行" else (C["orange"] if tag == "勉强" else C["red"]),
              size=9.5)
    _label(ax, 1.5, 0.95, "← b 太小：GEMM M 维不够", 10, ha="left")
    _label(ax, 10.5, 0.95, "b 太大：显存、气泡 →", 10, ha="right")
    _label(ax, 6, 0.45, "PP 下两头一夹，b 几乎总是 1 或 2；要大 M 只能减 p 或不用 PP。", 9.5)
    save(fig, "thousand-gpu-configuration-microbatch-bounds")


def recompute_bars():
    """F4 三种重计算策略：激活显存 + 额外计算。"""
    fig, ax = _canvas(7.6, 4.0, (0, 12), (0, 6.1))
    _label(ax, 6, 5.85, "三种重计算：激活显存 vs 额外计算（候选 A，每层 285 MB）", 12.5, weight="bold")
    _card(ax, 4.15, 5.25, 0.5, 0.3, "", fc=BLUE, ec=C["blue"], lw=0.6)
    _label(ax, 5.0, 5.4, "每层激活", 10, ha="left")
    _card(ax, 6.35, 5.25, 0.5, 0.3, "", fc="#f3b6b6", ec=C["red"], lw=0.6)
    _label(ax, 7.2, 5.4, "多算 FLOP", 10, ha="left")
    groups = [
        ("不重计算", 285, "285 MB", 0, "+0%"),
        ("选择性", 285, "≈ 285 MB（不省）", 4, "+3–5%"),
        ("按层 6/20", 200, "≈ 200 MB", 10, "+10%"),
        ("全量", 16.8, "16.8 MB", 33, "+33%"),
    ]
    bw = 285.0
    for i, (name, mem, mems, flop, flops) in enumerate(groups):
        y = 4.55 - i * 1.05
        _label(ax, 1.75, y, name, 11, ha="right")
        w = 6.2 * mem / bw
        _card(ax, 2.05, y + 0.16, w, 0.34, "", fc=BLUE, ec=C["blue"], lw=0.6)
        _label(ax, 2.05 + w + 0.3, y + 0.33, mems, 10, ha="left")
        w2 = 6.2 * flop / 100 * 3
        if w2 > 0.15:
            _card(ax, 2.05, y - 0.34, w2, 0.3, "", fc="#f3b6b6", ec=C["red"], lw=0.6)
            _label(ax, 2.05 + w2 + 0.3, y - 0.19, flops, 10, ha="left")
        else:
            _label(ax, 2.35, y - 0.19, flops, 10, ha="left")
    _label(ax, 6, 0.75, "FlashAttention 下 s^2 项已由 kernel 内重算拿走，selective 实际几乎不省；", 9.5)
    _label(ax, 6, 0.33, "toy recompute 实测：选择性存 36.3% 激活、+14.8% FLOP；全量存 21.2%、+33.3% FLOP。", 9.5)
    save(fig, "thousand-gpu-configuration-recompute-bars")


def mfu_waterfall():
    """F5 MFU 瀑布（横向）：42% 目标 → 七项损失 → 32% 实测。"""
    fig, ax = _canvas(7.6, 4.6, (25.5, 44), (-0.6, 10.2))
    _label(ax, 35, 9.8, "缺的 10 个 MFU 点：构造 trace 的七项拆解（§六.9）", 12.5, weight="bold")
    for g in range(30, 44):
        ax.plot([g, g], [1.3, 9.2], color="#eeeeee", lw=0.5, zorder=0)
        _label(ax, g, 0.95, f"{g}", 9.5, color="#666666")
    items = [
        ("目标", 42.0, 0, BLUE, ""),
        ("kernel 效率", -1.5, 0, GRAY, ""),
        ("未重叠通信", -3.0, 0, RED, ""),
        ("straggler", -2.0, 0, PURPLE, ""),
        ("CPU 发射", -1.5, 0, RED, ""),
        ("重计算白算", -1.0, 0, RED, ""),
        ("PP 气泡", -0.5, 0, GRAY, ""),
        ("数据等待", -0.5, 0, GRAY, ""),
        ("实测", 0, 32.0, ORANGE, ""),
    ]
    level = 42.0
    rows_y = [8.7 - i * 0.82 for i in range(9)]
    bh = 0.42
    for i, (name, d, endv, col, _) in enumerate(items):
        y = rows_y[i]
        _label(ax, 26.0, y, name, 10, ha="left")
        if i == 0:
            p = _card(ax, 28.6, y - bh, 42 - 28.6, 2 * bh, "", fc=col, ec="#888888", lw=0.7)
            _label(ax, 41.6, y, "42%", 10.5, in_box=p, ha="right", weight="bold")
        elif i == len(items) - 1:
            p = _card(ax, 28.6, y - bh, 32 - 28.6, 2 * bh, "", fc=col, ec="#888888", lw=0.7)
            _label(ax, 31.6, y, "32%", 10.5, in_box=p, ha="right", weight="bold")
        else:
            level += d
            p = _card(ax, level, y - bh, -d, 2 * bh, "", fc=col, ec="#888888", lw=0.7)
            if -d >= 0.9:  # 段够宽，点数写进段里
                _label(ax, level - d / 2, y, f"{d:+.1f}", 9.5, in_box=p)
            else:  # 窄段的点数标在段右侧
                _label(ax, level - d + 0.5, y, f"{d:+.1f}", 9.5, ha="left")
    _label(ax, 35, 0.35, "四项配置错误（通信、重计算、kernel、CPU）共 6 点先修；straggler 归第九篇；", 9.5)
    _label(ax, 35, -0.1, "气泡是结构问题；数据等待归第十篇——先修配置错误，再查硬件，最后碰结构。", 9.5)
    save(fig, "thousand-gpu-configuration-mfu-waterfall")


def main():
    for f in (four_steps, batch_blocks, microbatch_bounds, recompute_bars, mfu_waterfall):
        f()
    print("全部 5 张图完成。")


if __name__ == "__main__":
    main()
