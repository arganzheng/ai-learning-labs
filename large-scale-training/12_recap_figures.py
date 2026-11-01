"""12《系列总结与通关自测》配图：_plot.py 风格，collide()=0。
数字均来自 12_recap_toys.py ledger 的重算（与各篇一致）。

F1 ledger-chain：一个 70B / 1024 卡任务的算账链（§一）
F2 state-grid：四种状态 × 各篇的关系网格（§三.1）
"""
import os
import sys
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import C, box, plt, save, text

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


def _arrow(ax, a, b, color="#555555", lw=1.2, ms=13):
    ax.add_patch(mpatches.FancyArrowPatch(a, b, arrowstyle="-|>",
                                          color=color, lw=lw,
                                          mutation_scale=ms))


def ledger_chain():
    """F1：算账线六格链：16N → 49 GB → 2.0 s → 42% → δ/τ/G → GPU 小时。"""
    fig, ax = _canvas(h=4.4, xlim=(0, 12), ylim=(0, 6.2))
    _label(ax, 6, 5.9, "一个 70B / 1024 卡任务的一张账单（算账线，toy 重算）",
           12.5, weight="bold")
    items = [
        ("常驻状态", "16N=1.13TB", "", "第一篇", BLUE),
        ("每卡显存", "≈ 49 GB（A）", "", "第七篇", GREEN),
        ("理论 step 下限", "≈ 1.9 s", "", "第一篇·七", BLUE),
        ("MFU 目标", "42% ≈ 4.75 s", "", "第七篇", GREEN),
        ("落盘与间隔", "δ = 2 s", "τ_opt ≈ 3.5 min", "第八篇", ORANGE),
        ("有效训练时间", "G 71% → 94%", "（16K 卡场景）", "第九篇", ORANGE),
        ("成本单位", "1 MFU 点", "≈ 4.4 万美元", "第十一篇", RED),
    ]
    # 两排都左→右；MFU 肘形箭头下折到第二排第一格
    w, h, gap = 2.72, 1.7, 0.3
    layout = [(4, 5.1), (3, 2.6)]
    i = 0
    mfu_cx = mfu_y = x_end2 = y_row2_top = None
    for cnt, yt in layout:
        total = cnt * w + (cnt - 1) * gap
        x0 = (12 - total) / 2
        y = yt - h
        for j in range(cnt):
            title, val, sub, src, fc = items[i]
            x = x0 + j * (w + gap)
            p = _card(ax, x, y, w, h, "", fc=fc, ec="#777777", lw=0.9)
            _label(ax, x + w / 2, y + h - 0.3, title, 10.5, weight="bold",
                   in_box=p)
            if sub:
                _label(ax, x + w / 2, y + h - 0.72, val, 10, in_box=p)
                _label(ax, x + w / 2, y + h - 1.04, sub, 10, in_box=p)
            else:
                _label(ax, x + w / 2, y + h - 0.85, val, 10, in_box=p)
            _label(ax, x + w / 2, y + 0.26, src, 9, color="#555555",
                   in_box=p)
            if j < cnt - 1:
                _arrow(ax, (x + w + 0.05, y + h / 2),
                       (x + w + gap - 0.05, y + h / 2), ms=13)
            if i == 3:
                mfu_cx, mfu_y = x + w / 2, y
            if i == 4:
                x_end2, y_row2_top = x + w / 2, yt
            i += 1
    ymid = (mfu_y + y_row2_top) / 2
    ax.plot([mfu_cx, mfu_cx, x_end2 + 0.05],
            [mfu_y - 0.05, ymid, ymid], color="#555555", lw=1.2)
    _arrow(ax, (x_end2 + 0.05, ymid), (x_end2 + 0.05, y_row2_top + 0.05))
    _label(ax, 6, 0.55,
           "每格都从上一格推出：字节 → 配置 → 理论时间 → MFU 目标 → "
           "保存间隔 → 有效时间 → GPU 小时与美元", 9.5, color="#555555")
    save(fig, "large-scale-training-recap-ledger-chain")


def state_grid():
    """F2：四种状态（行）× 各篇（列）的关系网格。"""
    fig, ax = _canvas(h=4.4, xlim=(0, 12.9), ylim=(0, 6.6))
    _label(ax, 6.45, 6.32, "四种状态在十一篇里的落点（每格一句话）",
           12.5, weight="bold")
    cols = ["一算字节", "二怎么切", "三至六放容器", "八落不落盘",
            "九怎么重建", "十数值风险", "十一什么指标"]
    rows = [
        ("参数", ["16N 字节", "复制或切", "三种容器",
                  "落盘", "零通信分片", "bf16 主链",
                  "allocated"]),
        ("梯度", ["2 或 4 字节", "ZeRO-2 切", "三种容器",
                   "不落盘", "不落盘重算", "grad norm",
                   "clip 前范数"]),
        ("优化器状态", ["12N 字节", "ZeRO-1 切", "三种容器",
                     "落盘", "一并重分片", "fp32 链",
                     "alloc 重试"]),
        ("激活", ["34sbh / 层", "TP/SP/CP", "在途 O(p)",
                   "不落盘", "回退重算", "spike 指纹",
                  "step 时间"]),
    ]
    cw, rh = 1.62, 1.15
    x0, y0 = 1.5, 4.95
    for j, c in enumerate(cols):
        _label(ax, x0 + j * cw + cw / 2, y0 + 0.32, c, 9.5, weight="bold",
               color="#243447")
    fcs = [BLUE, GREEN, ORANGE, PURPLE]
    for i, (name, cells) in enumerate(rows):
        y = y0 - (i + 1) * rh
        _label(ax, 0.62, y + rh / 2, name, 10.5, ha="center", weight="bold")
        for j, cell in enumerate(cells):
            p = _card(ax, x0 + j * cw + 0.06, y + 0.06, cw - 0.12, rh - 0.12,
                      "", fc=fcs[i], ec="#888888", lw=0.7)
            _label(ax, x0 + j * cw + cw / 2, y + rh / 2, cell, 9.5, in_box=p)
    save(fig, "large-scale-training-recap-state-grid")


if __name__ == "__main__":
    ledger_chain()
    state_grid()
    print("全部 2 张图完成。")
