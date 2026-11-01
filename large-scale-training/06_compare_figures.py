"""大规模训练工程（06）三框架对比配图。输出 out/compare-f*.svg。"""
import os
from importlib.util import module_from_spec, spec_from_file_location

import matplotlib.patches as patches

from _plot import ARROW_LABEL_SIZE, BOX_TEXT_SIZE, C, box, box_title, plt, save, text

PALETTE = ["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3", "#e7e7e7"]


def f1():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.4); ax.axis("off")
    rows = [
        ("Megatron\nTP2×PP2×DP2 ZeRO-1", [0.75, 1.50, 0.75, 0.75, 0.75], ["0.75", "1.50", "0.75", "0.75", "0.75"], "4.50 GB", C["blue"]),
        ("DeepSpeed\nZeRO-3 DP8", [0.375, 0.375, 0.749, 0.749, 0.749], ["0.375", "0.375", "0.749", "0.749", "0.749"], "3.00 GB", C["orange"]),
        ("torchtitan\nFSDP4×TP2 tied", [0.618, 0.618, 0.618, 0.618], ["0.618", "0.618", "0.618", "0.618"], "2.47 GB\n不绑定 3.00", C["green"]),
    ]
    y_positions = [3.35, 2.30, 1.25]
    x0, total_w, h = 2.20, 4.20, 0.54
    for (name, vals, labels, total, accent), y in zip(rows, y_positions):
        text(ax, 2.05, y + h / 2, name, ha="right", va="center", fontsize=10, color=accent)
        width_sum = sum(vals)
        x = x0
        for idx, (val, label) in enumerate(zip(vals, labels)):
            w = total_w * val / width_sum
            # Same palette meanings across bars: parameter, gradient, master, m, v.
            color_idx = (0, 1, 2, 3, 4)[idx] if len(labels) == 5 else (0, 1, 3, 4)[idx]
            color = PALETTE[color_idx]
            ax.add_patch(patches.Rectangle((x, y), w, h, fc=color, ec="white", lw=1.2))
            text(ax, x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
            x += w
        text(ax, 6.55, y + h / 2, total, ha="left", va="center", fontsize=10, color=accent)
    legend = [("参数", PALETTE[0]), ("梯度", PALETTE[1]), ("主参数", PALETTE[2]), ("m", PALETTE[3]), ("v", PALETTE[4])]
    for i, (label, color) in enumerate(legend):
        x = 0.42 + i * 1.45
        ax.add_patch(patches.Rectangle((x, 0.45), 0.15, 0.15, fc=color, ec="#999999", lw=0.5))
        text(ax, x + 0.22, 0.525, label, ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.20, "不含激活与临时 all-gather；torchtitan fp32 分片就是主参数。", ha="center", va="center", fontsize=10)
    ax.set_title("同一个模型，三种参数相关显存的常驻方式")
    save(fig, "compare-f1-resident-memory")


def f2():
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.1); ax.axis("off")
    lanes = [
        ("Megatron", C["blue"], [(0.8, "2019\nTP"), (1.75, "2021\n1F1B"), (2.7, "2022\nSP"), (3.65, "2023\nCore"), (4.6, "2024\nEP / DO"), (5.65, "2026\nFSDP")]),
        ("DeepSpeed", C["orange"], [(0.8, "2019\nZeRO"), (1.75, "2020\nStage 1–3"), (2.7, "2021\nOffload"), (3.65, "2022\nMoE"), (4.6, "2023\nZeRO++"), (5.65, "2025\nAutoTP")]),
        ("PyTorch +\ntorchtitan", C["green"], [(1.75, "2022\nDTensor RFC"), (2.7, "2023\nDTensor"), (3.65, "2024\nMesh + FSDP2"), (4.6, "2024\nTitan 论文"), (5.65, "2026\nv0.3 Sharding")]),
    ]
    ys = [3.15, 2.05, 0.95]
    for (name, color, events), y in zip(lanes, ys):
        text(ax, 0.65, y, name, ha="right", va="center", fontsize=10, color=color)
        ax.plot([0.85, 6.1], [y, y], color=color, lw=1.2)
        for x, label in events:
            ax.plot([x, x], [y - 0.08, y + 0.08], color=color, lw=1.4)
            text(ax, x, y + 0.16, label, ha="center", va="bottom", fontsize=10, color=color)
    text(ax, 3.8, 0.28, "三条路线逐渐把并行布局变成参数 / 张量可描述的属性", ha="center", fontsize=10)
    ax.set_title("三条线怎么走到今天：每次演进都在解决新的规模瓶颈")
    save(fig, "compare-f2-timeline")


def f3():
    spec = spec_from_file_location("compare_toys", os.path.join(os.path.dirname(__file__), "06_compare_toys.py"))
    toy = module_from_spec(spec)
    spec.loader.exec_module(toy)
    events = toy.saw_events(16, prefetch=1)
    live_layers = [event[3] for event in events]
    steps = list(range(len(live_layers)))
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 4.8), sharex=True)
    panels = [("Megatron ZeRO-1", 750.0, C["blue"], None),
              ("DeepSpeed ZeRO-3", 121.6, C["orange"], live_layers),
              ("torchtitan FSDP2", 60.8, C["green"], live_layers)]
    for ax, (name, layer_mb, color, layers) in zip(axes, panels):
        ax.set_xlim(0, len(steps) - 1)
        ax.set_title(name, fontsize=11, loc="left", color=color)
        ax.set_yticks([0, layer_mb] if layers is None else [0, layer_mb, layer_mb * 2])
        ax.set_yticklabels(["0", f"{layer_mb:g}"] if layers is None else ["0", f"{layer_mb:g}", f"{layer_mb * 2:g}"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        if layers is None:
            ax.plot([0, len(steps) - 1], [layer_mb, layer_mb], color=color, lw=2.2)
        else:
            ax.step(steps, [count * layer_mb for count in layers], where="post", color=color, lw=1.8)
    axes[-1].set_xticks([])
    fig.suptitle("完整 bf16 参数在一个 step 内的占用：平线与两条锯齿", fontsize=12)
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.045, "横轴 = toy hook 事件顺序（16 层前向 + 反向）", ha="center", fontsize=10)
    fig.text(0.5, 0.002, "只画 16 个 Transformer 层，不含 embedding / 输出层；单位 MB（十进制）", ha="center", fontsize=10)
    save(fig, "compare-f3-live-parameters")


def f4():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2); ax.axis("off")
    row_y = [3.2, 2.05, 0.9]
    labels = ["Megatron", "DeepSpeed", "torchtitan"]
    colors = [C["blue"], C["orange"], C["green"]]
    for y, name, color in zip(row_y, labels, colors):
        text(ax, 0.9, y + 0.28, name, ha="right", va="center", fontsize=10, color=color)
        ax.add_patch(patches.Rectangle((1.05, y + 0.12), 6.1, 0.32, fc="#f4f6f8", ec="#cccccc", lw=0.5))
        text(ax, 2.0, y + 0.28, "前向 + 反向计算（时间向右）", ha="left", va="center", fontsize=10, color="#555555")
    # Megatron uses two large collectives across the step boundary.
    box(ax, 4.15, 2.55, 1.12, 0.62, "反向末尾\nRS 梯度", fc=C["orange"] + "22", ec=C["orange"], fs=10)
    box(ax, 5.62, 2.55, 1.12, 0.62, "下一步前\nAG 参数", fc=C["green"] + "22", ec=C["green"], fs=10)
    text(ax, 2.45, 2.86, "RS + AG = 2N", ha="center", va="center", fontsize=10, color=C["blue"])
    # ZeRO-3 and FSDP2 repeat three collectives for each Transformer layer.
    for y, gb, color in [(row_y[1], "7.9 GB/卡", C["orange"]), (row_y[2], "3.51 GB/卡", C["green"])]:
        for i, label in enumerate(("前向 AG", "反向 AG", "反向 RS")):
            box(ax, 1.35 + i * 1.38, y - 0.45, 1.08, 0.40, label, fc=color + "22", ec=color, fs=10)
        text(ax, 5.65, y - 0.25, f"每层 ×16\n{gb}", ha="left", va="center", fontsize=10, color=color)
    text(ax, 3.8, 0.22, "DeepSpeed / torchtitan：AG + AG + RS = 3N；若前向后不 reshard，则为 2N", ha="center", fontsize=10)
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
        text(ax, 6.5, y + cell_h / 2, f"r0 {sizes[0]}  |  r1 {sizes[1]}", ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.24, "每格 1 个元素；灰格是 padding。边框颜色：蓝=r0，橙=r1", ha="center", fontsize=10)
    ax.set_title("23 个元素放进两张卡：分片边界取决于容器")
    save(fig, "compare-f5-shard-layouts")


def f6():
    fig, ax = plt.subplots(figsize=(7.6, 4.15))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.15); ax.axis("off")
    rows = [
        ("Megatron\nTP2×PP2×DP2", "megatron", "TP [0,1] [2,3] [4,5] [6,7]\nDP [0,2] [1,3] [4,6] [5,7] · PP [0,4] [1,5] [2,6] [3,7]", C["blue"]),
        ("DeepSpeed\nZeRO-3 DP8", "deepspeed", "DP [0,1,2,3,4,5,6,7]", C["orange"]),
        ("torchtitan\ndp_shard4×TP2", "torchtitan", "TP [0,1] [2,3] [4,5] [6,7]\ndp_shard [0,2,4,6] [1,3,5,7]", C["green"]),
    ]
    ys = [3.10, 2.00, 0.90]
    for (name, grouping, desc, color), y in zip(rows, ys):
        text(ax, 0.82, y + 0.24, name, ha="right", va="center", fontsize=10, color=color)
        for r in range(8):
            x = 1.22 + r * 0.48
            p = box(ax, x, y, 0.40, 0.48, "", fc="white", ec="#444444", lw=0.8, r=0)
            if grouping == "megatron":
                segment_color = PALETTE[(r // 2) % 4]
            elif grouping == "deepspeed":
                segment_color = PALETTE[2]
            else:
                segment_color = PALETTE[0] if r % 2 == 0 else PALETTE[1]
            box(ax, x + 0.04, y + 0.33, 0.32, 0.08, "", fc=segment_color, ec=segment_color, r=0)
            text(ax, x + 0.20, y + 0.17, f"{r}", in_box=p, ha="center", va="center", fontsize=10)
        text(ax, 5.40, y + 0.24, desc, ha="left", va="center", fontsize=10, color=color)
    ax.set_title("8 张卡，三种分组：组名告诉你哪些 rank 一起通信")
    save(fig, "compare-f6-process-groups")


def f7():
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.5); ax.axis("off")
    def node(x, y, w, h, label, fc="#ffffff", ec="#555555", fs=10):
        p = box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=0.9, r=0)
        text(ax, x + w / 2, y + h / 2, label, in_box=p, ha="center", va="center", fontsize=fs)
        return p
    n0 = node(2.55, 3.65, 2.5, 0.48, "模型能否改写成框架的层？")
    n1 = node(0.25, 2.58, 2.35, 0.72, "能：显存是否够放\n完整 bf16 参数？", "#f7f9fc", C["blue"])
    n2 = node(4.95, 2.58, 2.35, 0.72, "不能：已有任意 nn.Module，\n先要降低显存？", "#fffaf3", C["orange"])
    n3 = node(0.25, 1.55, 2.35, 0.55, "够：追求极限 MFU？")
    n4 = node(2.95, 1.55, 2.35, 0.55, "不够 / 想要声明式组合？")
    n5 = node(5.0, 1.55, 2.35, 0.55, "小集群 / 脚本零侵入？")
    node(0.35, 0.35, 2.1, 0.62, "Megatron\n手写重叠、性能深", "#eef4fb", C["blue"])
    node(2.95, 0.35, 2.1, 0.62, "torchtitan\nDTensor 可组合", "#eff8f0", C["green"])
    node(5.2, 0.35, 2.1, 0.62, "DeepSpeed\nZeRO-3 + offload", "#fff3e8", C["orange"])
    def arrow(a, b, label=None, tx=None, ty=None):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2, color="#666666"))
        if label:
            text(ax, tx, ty, label, ha="center", va="center", fontsize=10)
    arrow((3.15, 3.65), (1.45, 3.3), "能", 2.15, 3.48)
    arrow((4.45, 3.65), (6.1, 3.3), "不能", 5.55, 3.48)
    arrow((1.45, 2.58), (1.45, 2.1), "够", 1.65, 2.38)
    arrow((2.25, 2.58), (4.0, 2.1), "不够 / 研究", 3.25, 2.48)
    arrow((6.1, 2.58), (6.1, 2.1), "是", 6.35, 2.38)
    arrow((1.45, 1.55), (1.4, 0.97), "是", 1.7, 1.28)
    arrow((3.95, 1.55), (4.0, 0.97), "是", 4.25, 1.28)
    arrow((6.1, 1.55), (6.25, 0.97), "是", 6.4, 1.28)
    ax.set_title("选型不是排行榜：先问模型、显存和改策略的需求")
    save(fig, "compare-f7-decision")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5(); f6(); f7()
