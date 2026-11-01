"""《千卡配置与 MFU 调优》配图：统一使用 _plot.py 的字体、画布与碰撞检查。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import ARROW_LABEL_SIZE, C, box, box_title, collide, plt, save, text

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


def _label(ax, x, y, s, size=10, color="#111111", ha="center", va="center", in_box=None, **kw):
    return text(ax, x, y, s, fontsize=size, color=color, ha=ha, va=va, in_box=in_box, **kw)


def _card(ax, x, y, w, h, s, fc="white", ec="#777777", size=10, lw=0.9):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if s:
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center", fontsize=size)
    return p


def _arrow(ax, a, b, color="#555555", lw=1.15, style="->", ms=10, **kw):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, **kw))


def four_steps():
    """F1 总览：70B 放进 1024 H100 的四步台阶。"""
    fig, ax = _canvas(7.6, 4.7, (0, 12), (0, 7.2))
    _label(ax, 6, 6.9, "70B 放进 1024 张 H100：四步定下五元组", 12.5)
    steps = [
        ("① TP = 8", "层内切权重，通信最频繁\n锁在节点内 NVLink", "静态 14.0 GB"),
        ("② PP = 4", "80 层分 4 段\n气泡 → 4.7%（v=4）", "激活 27 GB"),
        ("③ DP = 32", "补满 1024\nZeRO-1，梯度可重叠", "通信可盖"),
        ("④ CP = 1", "s=8K 暂时用不上\n留作以后拉长序列", "余量留这"),
    ]
    x0, cw = 0.35, 2.15
    for i, (t, why, mem) in enumerate(steps):
        x = x0 + i * (cw + 0.18)
        y = 4.6 - i * 0.35
        p = _card(ax, x, y, cw, 1.9, "", fc=[BLUE, GREEN, ORANGE, PURPLE][i],
                  ec=[C["blue"], C["green"], C["orange"], C["purple"]][i], lw=1.1)
        _label(ax, x + cw / 2, y + 1.58, t, 10.5, in_box=p)
        _label(ax, x + cw / 2, y + 0.92, why, 6.9, color="#333333", in_box=p)
        _label(ax, x + cw / 2, y + 0.32, mem, 7.6, color=C["red"], in_box=p)
        if i:
            _arrow(ax, (x - 0.17, y + 1.3), (x - 0.02, y + 1.3), color="#888888", lw=1.1, ms=9)
    p = _card(ax, 9.15, 2.4, 2.55, 3.5, "", fc="#ffffff", ec=C["blue"], lw=1.2)
    _label(ax, 10.375, 5.45, "候选 A", 11, in_box=p)
    _label(ax, 10.375, 4.75, "TP 8 · PP 4 · DP 32\nCP 1 · v 4 · m 16", 7.6, in_box=p)
    _label(ax, 10.375, 3.85, "每卡 ≈ 49 GB / 80\nDP 通信 8.8 GB 可重叠", 7.4, in_box=p)
    _label(ax, 10.375, 3.05, "预期 MFU 40–45%", 9.5, color=C["red"], in_box=p)
    _label(ax, 6, 1.55, "B（TP8/PP2/DP64）也过线：多 13 GB 静态换一半 PP 通信；本篇以 A 为主线，", 7.8)
    _label(ax, 6, 1.12, "因为它的显存余量允许以后加 CP 或拉长 s 而不必重新设计。", 7.8)
    save(fig, "config-70b-four-steps")


def batch_blocks():
    """F2 global batch = micro-batch × 累积 × DP 的积木。"""
    fig, ax = _canvas(7.6, 4.5, (0, 12), (0, 6.8))
    _label(ax, 6, 6.45, "global batch B = micro-batch b × 梯度累积 m × DP d（正文数字：b=1, m=16, d=32, B=512）", 10.5)
    _label(ax, 1.3, 5.4, "一个 micro-batch\nb × s = 8192 token", 8.2)
    _card(ax, 0.6, 3.5, 1.5, 1.2, "s×b\n8K tok", fc=BLUE, ec=C["blue"], size=8.5)
    _arrow(ax, (2.3, 4.1), (2.95, 4.1), color="#888888")
    _label(ax, 4.4, 5.4, "m = 16 个累积成一个副本的一步\n16 × 8192 = 128K token", 8.2)
    for i in range(8):
        _card(ax, 3.0 + i * 0.38, 3.5, 0.36, 1.2, "", fc=BLUE, ec="#ffffff", lw=0.4)
    _label(ax, 4.4, 2.95, "16 格叠成一列", 7.6)
    _arrow(ax, (6.35, 4.1), (7.0, 4.1), color="#888888")
    _label(ax, 8.6, 5.4, "d = 32 个 DP 副本\n→ global batch", 8.2)
    for i in range(6):
        _card(ax, 7.1 + i * 0.62, 3.5, 0.56, 1.2, "", fc=GREEN, ec="#ffffff", lw=0.4)
    _label(ax, 7.9, 2.95, "32 列拼成 batch", 7.6)
    _card(ax, 9.2, 1.55, 2.3, 1.6, "B = 512 条\nB_tok = 512×8K\n= 4.2 M token", fc=ORANGE, ec=C["orange"], size=8.5)
    _label(ax, 3.6, 2.5, "B 由训练配方定，卡数只动 d 和 m；", 8.4)
    _label(ax, 3.6, 2.1, "PP 下 m 同时是流水 micro-batch 数与累积步数（同一个数）。", 8.4)
    save(fig, "config-batch-blocks")


def microbatch_bounds():
    """F3 micro-batch 两头约束区间条。"""
    fig, ax = _canvas(7.6, 3.9, (0, 12), (0, 6.0))
    _label(ax, 6, 5.65, "micro-batch b 的两头约束（b=1..4，候选 A）", 11.5)
    rows = [
        ("b = 1", 3.6, True, "GEMM M=8192 才 60–70% · 气泡 4.7% · ~49 GB", "可行"),
        ("b = 2", 2.6, True, "M 翻倍效率上去 · m 8 → 气泡 9.4% · ~76 GB 太紧", "勉强"),
        ("b = 4", 1.6, False, "气泡 18.8% · ~130 GB 爆显存", "不可行"),
    ]
    for name, y, ok, why, tag in rows:
        _label(ax, 0.7, y, name, 9, ha="left")
        col = GREEN if ok and tag == "可行" else (ORANGE if ok else RED)
        p = _card(ax, 1.7, y - 0.33, 8.6, 0.66, "", fc=col, ec="#999999", lw=0.6)
        _label(ax, 6.0, y, why, 7.8, in_box=p)
        _card(ax, 10.5, y - 0.3, 1.15, 0.6, tag,
              fc="#ffffff", ec=C["green"] if tag == "可行" else (C["orange"] if tag == "勉强" else C["red"]), size=8.2)
    _label(ax, 6, 0.75, "左界：b 太小 → GEMM M 维不够、CPU 发射占比升；右界：b 太大 → 激活线性涨、m 反比缩、气泡翻。", 7.6)
    _label(ax, 6, 0.32, "PP 下两头一夹，b 几乎总是 1 或 2；要大 M 只能减 p 或不用 PP。", 7.6)
    save(fig, "config-microbatch-bounds")


def recompute_bars():
    """F4 三种重计算策略：激活显存 + 额外计算。"""
    fig, ax = _canvas(7.6, 4.3, (0, 12), (0, 6.6))
    _label(ax, 6, 6.25, "三种重计算：激活显存 vs 额外计算（候选 A 每层 285 MB，正文 §四.1）", 11)
    _label(ax, 1.1, 5.55, "每层激活", 8.2, ha="left", color=C["blue"])
    _label(ax, 1.1, 5.15, "多算 FLOP", 8.2, ha="left", color=C["red"])
    groups = [
        ("不重计算", 285, "285 MB", 0, "+0%"),
        ("选择性", 285, "≈ 285 MB*", 4, "+3–5%"),
        ("全量", 16.8, "16.8 MB", 33, "+33%"),
        ("按层 6/20", 200, "≈ 200 MB", 10, "+10%"),
    ]
    bw = 285.0
    for i, (name, mem, mems, flop, flops) in enumerate(groups):
        y = 4.55 - i * 1.05
        _label(ax, 1.95, y, name, 8.8, ha="right")
        w = 6.5 * mem / bw
        _card(ax, 2.25, y + 0.15, w, 0.3, "", fc=BLUE, ec=C["blue"], lw=0.6)
        _label(ax, 2.25 + w + 0.3, y + 0.30, mems, 7.4, ha="left")
        w2 = 6.5 * flop / 100 * 3
        if w2 > 0.15:
            _card(ax, 2.25, y - 0.32, w2, 0.26, "", fc="#f3b6b6", ec=C["red"], lw=0.6)
            _label(ax, 2.25 + w2 + 0.3, y - 0.19, flops, 7.4, ha="left")
        else:
            _label(ax, 2.55, y - 0.19, flops, 7.4, ha="left")
    _label(ax, 6, 0.78, "* FlashAttention 下 s^2 项已由 kernel 内重算拿走，selective 几乎不省。", 7.6)
    _label(ax, 6, 0.42, "toy recompute 实测：选择性存 36.3% 激活、+14.8% FLOP；", 7.6)
    _label(ax, 6, 0.08, "全量存 21.2%、+33.3% FLOP——全量的 +33% 与正文一致。", 7.6)
    save(fig, "config-recompute-bars")


def mfu_waterfall():
    """F5 MFU 瀑布：理论 → 七项损失 → 实测（§六.9 构造表数字）。"""
    fig, ax = _canvas(7.6, 4.8, (0, 12), (0, 7.4))
    _label(ax, 6, 7.05, "缺的 10 个 MFU 点：构造 trace 的七项拆解（§六.9，单位 MFU 点）", 11.5)
    items = [  # (label, delta points, color, note)
        ("目标 42%", 42.0, BLUE, "4.75 s/step"),
        ("kernel 效率", -1.5, GRAY, "no-fusion 等"),
        ("PP 气泡", -0.5, GRAY, "首尾 stage 多算"),
        ("未重叠通信", -3.0, RED, "overlap 全没开"),
        ("重计算白算", -1.0, RED, "selective 白算"),
        ("数据等待", -0.5, GRAY, "在线 tokenize"),
        ("CPU 发射", -1.5, RED, "小 kernel 太多"),
        ("straggler", -2.0, PURPLE, "一卡降频"),
        ("实测 32%", 32.0, ORANGE, "6.2 s/step"),
    ]
    x0, cw = 0.4, 1.18
    top = 6.3
    level = 42.0
    scale = 4.9 / 42.0
    for i, (name, v, col, note) in enumerate(items):
        x = x0 + i * (cw + 0.06)
        if i in (0, len(items) - 1):
            y = top - abs(v) * scale
            p = _card(ax, x, y, cw, abs(v) * scale, "", fc=col, ec="#888888", lw=0.7)
            _label(ax, x + cw / 2, y + 0.24, f"{v:.0f}%", 9, in_box=p)
        else:
            y = top - level * scale
            h = abs(v) * scale
            _card(ax, x, y - h, cw, h, "", fc=col, ec="#888888", lw=0.7)
            _label(ax, x + cw / 2, y + 0.14, f"{v:+.1f}", 8)
            level += v
        _label(ax, x + cw / 2, 1.05, name, 7.2)
        _label(ax, x + cw / 2, 0.72, note, 5.9, color="#444444")
    _label(ax, 6, 0.38, "四项配置错误（通信、重计算、kernel、CPU）共 6 点先修；straggler 是硬件问题归第九篇；", 7.6)
    _label(ax, 6, 0.05, "气泡是结构问题；数据等待归第十篇——先修配置错误，再查硬件，最后碰结构。", 7.6)
    save(fig, "config-mfu-waterfall")


def main():
    for f in (four_steps, batch_blocks, microbatch_bounds, recompute_bars, mfu_waterfall):
        f()
    print("全部 5 张图完成。")


if __name__ == "__main__":
    main()
