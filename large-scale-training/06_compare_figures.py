"""大规模训练工程（06）三框架对比配图。输出 out/compare-f*.svg。"""
import os
from importlib.util import module_from_spec, spec_from_file_location

import matplotlib.patches as patches

from _plot import ARROW_LABEL_SIZE, BOX_TEXT_SIZE, C, box, box_title, plt, save, text

PALETTE = ["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3", "#e7e7e7"]


def f1():
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.3); ax.axis("off")
    rows = [
        ("Megatron\nTP2×PP2×DP2 ZeRO-1", [0.75, 1.50, 0.75, 0.75, 0.75],
         ["bf16\n参数", "fp32\n梯度", "fp32\n主参数", "Adam\nm", "Adam\nv"], "4.50 GB", C["blue"]),
        ("DeepSpeed\nZeRO-3 DP8", [0.375, 0.375, 0.749, 0.749, 0.749],
         ["bf16\n参数", "bf16\n梯度", "fp32\n主参数", "Adam\nm", "Adam\nv"], "3.00 GB", C["orange"]),
        ("torchtitan\nFSDP4×TP2 tied", [0.618, 0.618, 0.618, 0.618],
         ["fp32\n参数＝主参数", "fp32\n梯度", "Adam\nm", "Adam\nv"], "2.47 GB\n不绑定 3.00", C["green"]),
    ]
    y_positions = [3.15, 1.85, 0.55]
    x0, total_w, h = 2.20, 4.20, 0.54
    for (name, vals, segnames, total, accent), y in zip(rows, y_positions):
        text(ax, 2.05, y + h / 2, name, ha="right", va="center", fontsize=10, color=accent)
        width_sum = sum(vals)
        x = x0
        n = len(vals)
        for idx, (val, segname) in enumerate(zip(vals, segnames)):
            w = total_w * val / width_sum
            # Same palette meanings across bars: parameter, gradient, master, m, v.
            color_idx = (0, 1, 2, 3, 4)[idx] if n == 5 else (0, 1, 3, 4)[idx]
            color = PALETTE[color_idx]
            ax.add_patch(patches.Rectangle((x, y), w, h, fc=color, ec="white", lw=1.2))
            text(ax, x + w / 2, y + h + 0.06, segname, ha="center", va="bottom", fontsize=9)
            text(ax, x + w / 2, y + h / 2, f"{val:g}", ha="center", va="center", fontsize=10)
            x += w
        text(ax, 6.55, y + h / 2, total, ha="left", va="center", fontsize=10, color=accent)
    text(ax, 3.8, 0.10, "不含激活与临时 all-gather；torchtitan fp32 分片就是主参数。", ha="center", va="center", fontsize=10)
    ax.set_title("同一个模型，三种参数相关显存的常驻方式")
    save(fig, "compare-f1-resident-memory")


def f2():
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.9); ax.axis("off")
    x0, xstep = 1.30, 0.80
    def yr(y):
        return x0 + (y - 2019) * xstep
    # (year, label, below) — labels hug their own lane: above or below the line, never floating.
    lanes = [
        ("Megatron", C["blue"], "#eef3f9", [(2019, "TP", 0), (2021, "1F1B", 0), (2022, "SP", 0), (2023, "Megatron\nCore", 1), (2024, "EP / 分布式\n优化器", 0), (2026, "Megatron-\nFSDP", 0)]),
        ("DeepSpeed", C["orange"], "#fdf4ec", [(2019, "ZeRO", 0), (2020, "Stage 1–3", 1), (2021, "Offload", 0), (2022, "MoE", 1), (2023, "ZeRO++", 0), (2025, "AutoTP", 0)]),
        ("PyTorch +\ntorchtitan", C["green"], "#eef6ef", [(2022, "DTensor\nRFC", 0), (2023, "DTensor", 1), (2024, "DeviceMesh / FSDP2\ntorchtitan 论文", 0), (2026, "v0.3\nSharding\nConfig", 0)]),
    ]
    ys = [3.68, 2.28, 0.88]
    band_h = 0.60
    for y in range(2019, 2027):
        ax.plot([yr(y), yr(y)], [0.30, 4.30], color="#e4e4e4", lw=0.6, zorder=0)
        text(ax, yr(y), 4.55, str(y), ha="center", va="center", fontsize=9, color="#888888")
    for (name, color, tint, events), y in zip(lanes, ys):
        ax.add_patch(patches.Rectangle((0.85, y - band_h), 6.55, band_h * 2, fc=tint, ec="none", zorder=0))
        text(ax, 0.72, y, name, ha="right", va="center", fontsize=10, color=color)
        ax.plot([x0 - 0.10, yr(2026) + 0.28], [y, y], color=color, lw=1.2)
        for year, label, below in events:
            ax.plot([yr(year), yr(year)], [y - 0.08, y + 0.08], color=color, lw=1.4)
            if below:
                text(ax, yr(year), y - 0.16, label, ha="center", va="top", fontsize=9, color=color)
            else:
                text(ax, yr(year), y + 0.16, label, ha="center", va="bottom", fontsize=9, color=color)
    text(ax, 3.8, 0.06, "三条路线逐渐把并行布局变成参数 / 张量可描述的属性", ha="center", fontsize=10)
    ax.set_title("三条线怎么走到今天：每次演进都在解决新的规模瓶颈")
    save(fig, "compare-f2-timeline")


def f3():
    spec = spec_from_file_location("compare_toys", os.path.join(os.path.dirname(__file__), "06_compare_toys.py"))
    toy = module_from_spec(spec)
    spec.loader.exec_module(toy)
    events = toy.saw_events(16, prefetch=1)
    live_layers = [event[3] for event in events]
    steps = list(range(len(live_layers)))
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 5.0), sharex=True)
    bwd_start = next(i for i, e in enumerate(events) if e[0] == "backward")
    panels = [("Megatron ZeRO-1", 750.0, C["blue"], None, [0, 750]),
              ("DeepSpeed ZeRO-3", 121.6, C["orange"], live_layers, [0, 121.6, 243.2]),
              ("torchtitan FSDP2", 60.8, C["green"], live_layers, [0, 121.6])]
    for ax, (name, layer_mb, color, layers, ticks) in zip(axes, panels):
        ax.set_xlim(0, len(steps) - 1)
        ax.set_ylim(0, 800)
        ax.set_title(name, fontsize=11, loc="left", color=color)
        ax.set_yticks(ticks)
        ax.set_yticklabels(["" if (t == 0 and layers is not None) else f"{t:g}" for t in ticks])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.axvline(bwd_start - 0.5, color="#999999", lw=0.9, ls="--")
        if layers is None:
            ax.plot([0, len(steps) - 1], [layer_mb, layer_mb], color=color, lw=2.2)
        else:
            ax.step(steps, [count * layer_mb for count in layers], where="post", color=color, lw=1.8)
    axes[0].text(bwd_start + 0.6, 1.03, "前向结束 / 反向开始", fontsize=9, color="#333333",
                 va="bottom", ha="left", transform=axes[0].get_xaxis_transform())
    axes[-1].set_xticks([])
    fig.suptitle("完整 bf16 参数在一个 step 内的占用：平线与两条锯齿", fontsize=12)
    save(fig, "compare-f3-live-parameters")


def f4():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.4); ax.axis("off")
    row_y = [3.55, 2.30, 1.05]
    labels = ["Megatron", "DeepSpeed", "torchtitan"]
    colors = [C["blue"], C["orange"], C["green"]]
    bar_x0, bar_x1, bar_mid = 1.15, 6.35, 3.75
    for y, name, color in zip(row_y, labels, colors):
        text(ax, 1.0, y + 0.29, name, ha="right", va="center", fontsize=10, color=color)
        ax.add_patch(patches.Rectangle((bar_x0, y + 0.12), bar_mid - bar_x0, 0.34, fc="#f4f6f8", ec="#cccccc", lw=0.5))
        ax.add_patch(patches.Rectangle((bar_mid, y + 0.12), bar_x1 - bar_mid, 0.34, fc="#eceff2", ec="#cccccc", lw=0.5))
        text(ax, (bar_x0 + bar_mid) / 2, y + 0.29, "前向", ha="center", va="center", fontsize=10, color="#555555")
        text(ax, (bar_mid + bar_x1) / 2, y + 0.29, "反向", ha="center", va="center", fontsize=10, color="#555555")
    # Megatron: one reduce-scatter at the end of backward, one all-gather before next forward.
    box(ax, 4.85, row_y[0] - 0.66, 1.25, 0.62, "反向末尾\nRS 梯度", fc=C["orange"] + "22", ec=C["orange"], fs=9.5)
    box(ax, 6.42, row_y[0] - 0.66, 1.05, 0.62, "下一步前\nAG 参数", fc=C["green"] + "22", ec=C["green"], fs=9.5)
    text(ax, 6.55, row_y[0] + 0.29, "RS + AG = 2N\n1.1 GB/卡", ha="left", va="center", fontsize=9.5, color=C["blue"])
    # ZeRO-3 and FSDP2 repeat three collectives for each Transformer layer.
    for y, gb, color in [(row_y[1], "7.9 GB/卡", C["orange"]), (row_y[2], "3.51 GB/卡", C["green"])]:
        for x, label in [(2.00, "前向 AG"), (4.05, "反向 AG"), (5.30, "反向 RS")]:
            box(ax, x, y - 0.62, 1.10, 0.42, label, fc=color + "22", ec=color, fs=9.5)
        text(ax, 6.55, y + 0.29, f"每层 ×16\n{gb}", ha="left", va="center", fontsize=9.5, color=color)
    text(ax, 3.8, 0.18, "DeepSpeed / torchtitan：AG + AG + RS = 3N；若前向后不 reshard，则为 2N", ha="center", fontsize=10)
    ax.set_title("每步的 DP 通信：Megatron 两大块，ZeRO-3 / FSDP2 每层三小块")
    save(fig, "compare-f4-communication")


def f5():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.6); ax.axis("off")
    rows = [
        ("Megatron\nbuffer", [("c", 5, 0), ("B", 6, 1), ("A", 12, 2), ("pad", 1, 4)], [12, 12]),
        ("DeepSpeed\nZeRO-3", [("A", 12, 0), ("B", 6, 1), ("c", 5, 2), ("pad", 1, 4)], [12, 12]),
        ("FSDP1\nFlatParameter", [("A", 12, 0), ("B", 6, 1), ("c", 5, 2), ("pad", 1, 4)], [12, 12]),
        ("FSDP2\nDTensor", [("A", 12, 0), ("B", 6, 1), ("c", 5, 2)], [12, 11]),
    ]
    colors = {"A": PALETTE[0], "B": PALETTE[1], "c": PALETTE[2], "pad": "#bbbbbb"}
    rank_colors = ["#3b6fb6", "#e0812c"]
    x0, cell_w, cell_h = 1.75, 0.19, 0.46
    ys = [3.55, 2.65, 1.75, 0.85]
    for (name, segs, sizes), y in zip(rows, ys):
        text(ax, 1.55, y + cell_h / 2, name, ha="right", va="center", fontsize=10)
        pos = x0
        cursor = 0
        for pname, n, _ in segs:
            for j in range(n):
                # ownership indicated by alternating tint; pad cells stay gray.
                if pname == "pad":
                    fc = rank_colors[1]
                elif name.startswith("DeepSpeed"): 
                    # Each parameter partitions independently; c's final cell is rank1 padding.
                    rank = 0 if j < (n + 1) // 2 else 1
                    fc = rank_colors[rank]
                else:
                    fc = rank_colors[0] if cursor < sizes[0] else rank_colors[1]
                ax.add_patch(patches.Rectangle((pos, y), cell_w, cell_h, fc=colors[pname], ec="white", lw=0.4))
                ax.add_patch(patches.Rectangle((pos, y), cell_w, cell_h, fill=False, ec=fc, lw=1.4))
                pos += cell_w
                cursor += 1
            text(ax, pos - n * cell_w / 2, y + cell_h + 0.08, pname, ha="center", va="bottom", fontsize=10)
        text(ax, 6.5, y + cell_h / 2, f"rank0 {sizes[0]} 个｜rank1 {sizes[1]} 个", ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.24, "每格 1 个元素；填充色＝属于哪个参数（A/B/c），灰格是 padding；边框色：蓝=r0、橙=r1", ha="center", fontsize=10)
    ax.set_title("23 个元素放进两张卡：分片边界取决于容器")
    save(fig, "compare-f5-shard-layouts")


def f6():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 3.9); ax.axis("off")
    rows = [
        ("Megatron\nTP2×PP2×DP2", "megatron", "色条＝TP 组\nTP [0,1] [2,3] [4,5] [6,7]\nDP [0,2] [1,3] [4,6] [5,7]\nPP [0,4] [1,5] [2,6] [3,7]", C["blue"]),
        ("DeepSpeed\nZeRO-3 DP8", "deepspeed", "色条＝同一 DP 组\nDP [0,1,2,3,4,5,6,7]", C["orange"]),
        ("torchtitan\ndp_shard4×TP2", "torchtitan", "色条＝dp_shard 组\nTP [0,1] [2,3] [4,5] [6,7]\ndp_shard [0,2,4,6] [1,3,5,7]", C["green"]),
    ]
    ys = [2.95, 1.85, 0.75]
    for (name, grouping, desc, color), y in zip(rows, ys):
        text(ax, 0.94, y + 0.27, name, ha="right", va="center", fontsize=10, color=color)
        for r in range(8):
            x = 1.16 + r * 0.56
            p = box(ax, x, y, 0.50, 0.55, "", fc="white", ec="#444444", lw=0.8, r=0)
            if grouping == "megatron":
                segment_color = PALETTE[(r // 2) % 4]
            elif grouping == "deepspeed":
                segment_color = PALETTE[2]
            else:
                segment_color = PALETTE[0] if r % 2 == 0 else PALETTE[1]
            box(ax, x + 0.05, y + 0.38, 0.40, 0.09, "", fc=segment_color, ec=segment_color, r=0)
            text(ax, x + 0.25, y + 0.19, f"{r}", in_box=p, ha="center", va="center", fontsize=10)
        text(ax, 5.78, y + 0.27, desc, ha="left", va="center", fontsize=9.5, color=color)
    ax.set_title("8 张卡，三种分组：组名告诉你哪些 rank 一起通信")
    save(fig, "compare-f6-process-groups")


def f7():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.0); ax.axis("off")
    def node(x, y, w, h, label, fc="#ffffff", ec="#555555", fs=10):
        p = box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=0.9, r=0)
        text(ax, x + w / 2, y + h / 2, label, in_box=p, ha="center", va="center", fontsize=fs)
        return p
    q1 = node(2.55, 3.05, 2.5, 0.55, "模型能否改写成框架自己的层？")
    q2 = node(0.60, 1.95, 2.40, 0.50, "更看重什么？", "#f7f9fc", C["blue"])
    ds = node(4.35, 1.88, 3.15, 0.68, "DeepSpeed\n包裹任意 nn.Module、ZeRO-3 / offload", "#fff3e8", C["orange"])
    m = node(0.20, 0.36, 2.30, 0.68, "Megatron\n手写通信重叠、TP/PP 最成熟", "#eef4fb", C["blue"])
    t = node(2.75, 0.36, 2.55, 0.68, "torchtitan\nDTensor + FSDP2，策略写成声明", "#eff8f0", C["green"])
    def arrow(a, b, label=None, tx=None, ty=None):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2, color="#666666"))
        if label:
            text(ax, tx, ty, label, ha="center", va="center", fontsize=9.5)
    arrow((3.20, 3.05), (1.80, 2.45), "能", 2.40, 2.78)
    arrow((4.60, 3.05), (6.15, 2.50), "不能", 5.50, 2.78)
    arrow((1.30, 1.95), (1.35, 1.00), "千卡级极限 MFU", 1.62, 1.50)
    arrow((2.30, 1.95), (3.95, 1.00), "并行策略常改\nPyTorch 原生", 3.60, 1.52)
    ax.set_title("选型不是排行榜：先问模型约束，再问需求")
    save(fig, "compare-f7-decision")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5(); f6(); f7()
