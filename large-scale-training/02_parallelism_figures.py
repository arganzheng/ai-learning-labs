"""《并行策略全景》配图：统一使用 _plot.py 的字体、画布与碰撞检查。"""
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


def _label(ax, x, y, s, size=10, color="#111111", ha="center", va="center", **kw):
    return text(ax, x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def _card(ax, x, y, w, h, s, fc="white", ec="#777777", size=10, lw=0.9):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if s:
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center", fontsize=size)
    return p


def _arrow(ax, a, b, color="#555555", lw=1.15, style="->", ms=10, **kw):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, **kw))


def _strip(ax, x, y, w, h, labels, colors, fs=10):
    cw = w / len(labels)
    for i, (label, color) in enumerate(zip(labels, colors)):
        _card(ax, x + i * cw, y, cw, h, label, fc=color, ec="#ffffff", size=fs, lw=0.5)


def states_grid():
    fig, ax = _canvas(7.6, 3.9, (0, 12), (0, 7.1))
    _label(ax, 6, 6.62, "五种并行各切哪种状态", 13)
    cols = ["DP", "ZeRO-3", "TP / SP", "CP", "PP", "EP"]
    rows = ["参数", "梯度", "优化器状态", "激活"]
    x0, y0, cw, rh = 2.05, 2.55, 1.48, 0.63
    for j, col in enumerate(cols):
        _label(ax, x0 + j * cw + cw / 2, 5.25, col, 10)
    status_rows = [
        [0, 2, 1, 0, 1, 2],
        [0, 1, 1, 0, 1, 2],
        [0, 1, 1, 0, 1, 2],
        [1, 1, 2, 1, 1, 2],
    ]
    for i, row in enumerate(rows):
        y = y0 + (3 - i) * rh
        _label(ax, 1.22, y + rh / 2, row, 10, ha="right")
        for j, status in enumerate(status_rows[i]):
            x = x0 + j * cw
            _card(ax, x, y, cw - 0.06, rh - 0.05, "", fc="#ffffff", ec="#d5d9df", lw=0.65)
            cx, cy = x + (cw - 0.06) / 2, y + (rh - 0.05) / 2
            if status == 0:
                ax.add_patch(plt.Circle((cx, cy), 0.09, fill=False, ec="#6d7782", lw=1.15))
            elif status == 1:
                ax.add_patch(plt.Circle((cx, cy), 0.09, fc=C["blue"], ec=C["blue"], lw=1))
            else:
                ax.add_patch(plt.Circle((cx, cy), 0.09, fc=C["orange"], ec=C["orange"], lw=1))
    _label(ax, 1.22, 2.05, "主通信", 10, ha="right")
    comm = [("梯度 AR", "IB"), ("AG + RS", "IB"), ("AG / RS", "NVLink"), ("K/V 收发", "IB"), ("激活收发", "IB"), ("all-to-all", "IB / NVLink")]
    for j, (prim, link) in enumerate(comm):
        _label(ax, x0 + j * cw + cw / 2, 2.22, prim, 10)
        _label(ax, x0 + j * cw + cw / 2, 1.92, link, 10, color="#555555")
    _label(ax, 6, 1.60, "主通信：上行原语、下行链路。DP 梯度 AR；ZeRO-3 参数 AG + 梯度 RS；TP 层边界 AG/RS", 10)
    _label(ax, 6, 1.12, "CP 收发 K/V 块；PP 收发激活；EP token all-to-all", 10)
    ax.add_patch(plt.Circle((2.35, 0.72), 0.07, fc=C["blue"], ec=C["blue"]))
    _label(ax, 2.52, 0.72, "切分", 10, ha="left")
    ax.add_patch(plt.Circle((3.35, 0.72), 0.07, fc=C["orange"], ec=C["orange"]))
    _label(ax, 3.52, 0.72, "部分 / 临时拼全", 10, ha="left")
    ax.add_patch(plt.Circle((5.95, 0.72), 0.07, fill=False, ec="#6d7782", lw=1.1))
    _label(ax, 6.12, 0.72, "复制", 10, ha="left")
    save(fig, "parallelism-states-grid")


def mesh_8cards():
    fig, ax = _canvas(7.6, 4.55, (0, 12), (0.7, 10))
    _label(ax, 6, 6.95, "一张 8 卡的五元组摆法", 13)
    _label(ax, 6, 6.55, "TP=2 × DP=2 × PP=2；rank = tp + 2·dp + 4·pp；节点 = rank // 4", 10.5)
    nodes = [[0, 1, 2, 3], [4, 5, 6, 7]]
    xnode = [0.8, 6.65]
    for ni, ranks in enumerate(nodes):
        ax.add_patch(plt.Rectangle((xnode[ni], 3.45), 4.55, 2.6, fc="#f8fafc", ec="#aab5c2", lw=0.9, zorder=0))
        _label(ax, xnode[ni] + 2.275, 5.75, f"节点 {ni} · 4 卡", 10.5)
        for k, rank in enumerate(ranks):
            x = xnode[ni] + 0.22 + k * 1.04
            _card(ax, x, 4.32, 0.98, 0.72, f"rank {rank}", fc=BLUE, ec=C["blue"], size=10)
            tp, dp, pp = rank % 2, (rank // 2) % 2, rank // 4
            _label(ax, x + 0.45, 4.1, f"t{tp} d{dp} p{pp}", 10)
        _label(ax, xnode[ni] + 2.275, 3.7, f"pp={ni}，节点内 TP / DP 走 NVLink", 10)
    for k in range(4):
        y = 4.48 + k * 0.17
        _arrow(ax, (xnode[0] + 4.55, y), (xnode[1], y), color=C["red"], lw=1.3, ms=9)
    _label(ax, (xnode[0] + 4.55 + xnode[1]) / 2, 5.36, "PP 组：跨节点 IB", 10)
    _label(ax, 6, 2.92, "PP 组 {0,4} · {1,5} · {2,6} · {3,7}：rank i 与 rank i+4 配对，跨节点 IB", 10)
    _label(ax, 6, 2.45, "TP 组 {0,1} · {2,3} · {4,5} · {6,7}　DP 组 {0,2} · {1,3} · {4,6} · {5,7}：节点内 NVLink", 10)
    _label(ax, 6, 1.35, "Megatron 顺序 tp-cp-ep-dp-pp：TP 变化最快，PP 最慢；8 卡/节点且 TP=8 时，DP 组会跨节点", 10)
    save(fig, "parallelism-mesh-8cards")


def bubble_curve():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(4, 64)
    ax.set_ylim(0, 4.2)
    ax.set_xticks([4, 16, 32, 48, 64]); ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("micro-batch 数 m")
    ax.set_ylabel("气泡 / 理想计算时间（倍）")
    ax.grid(axis="y", color="#d9dde2", lw=0.6)
    xs = list(range(4, 65))
    for p, color in [(4, C["blue"]), (8, C["green"]), (16, C["orange"])]:
        ax.plot(xs, [(p - 1) / m for m in xs], color=color, lw=1.7, label=f"p={p}：v=1，(p-1)/m")
        ax.plot(xs, [(p - 1) / (2 * m) for m in xs], color=color, lw=1.3, ls="--", label=f"p={p}：v=2")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=10, title="实线 v=1　虚线 v=2（气泡约减半，通信约翻倍）", title_fontsize=10)
    save(fig, "parallelism-bubble-rate")


def main():
    funcs = [states_grid, mesh_8cards, bubble_curve]
    for func in funcs:
        func()
    print(f"全部 {len(funcs)} 张图完成（其余 11 张为手工 SVG，本脚本不生成）。")


if __name__ == "__main__":
    main()
