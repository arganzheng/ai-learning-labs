"""DeepSpeed 04 配图：以 750px 正文列宽绘制，所有文字用 _plot.py 的碰撞检查。"""
from _plot import ARROW_LABEL_SIZE, C, box, plt, save, text


def f1():
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.5); ax.axis("off")
    ax.set_title("ZeRO-3：每个 rank 常驻 Wₖ 的 1/8，用时再拼完整")
    palette = ["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3",
               "#f7efc9", "#d8edef", "#f3dfe2", "#e5e5d7"]
    card_x = [0.08 + rank * 0.94 for rank in range(8)]
    top_y, top_h, card_w = 3.02, 1.26, 0.90

    for rank, x in enumerate(card_x):
        p = box(ax, x, top_y, card_w, top_h, "", fc="#ffffff", ec="#444444", lw=0.9)
        text(ax, x + card_w / 2, top_y + 1.08, f"rank {rank}", in_box=p,
             ha="center", va="center", fontsize=9.5, weight="bold")
        box(ax, x + 0.10, top_y + 0.87, card_w - 0.20, 0.12, "", fc=palette[rank], ec=palette[rank])
        text(ax, x + card_w / 2, top_y + 0.66, "ds_tensor 1/8", in_box=p,
             ha="center", va="center", fontsize=7.2)
        text(ax, x + card_w / 2, top_y + 0.45, "梯度 bf16 1/8", in_box=p,
             ha="center", va="center", fontsize=7.2)
        text(ax, x + card_w / 2, top_y + 0.24, "主参+m+v 1/8", in_box=p,
             ha="center", va="center", fontsize=7.2)

    bottom_y, bottom_h = 1.08, 1.02
    for x in card_x:
        p = box(ax, x, bottom_y, card_w, bottom_h, "", fc="#ffffff", ec="#444444", lw=0.9)
        inner_x, inner_w = x + 0.045, card_w - 0.09
        seg_gap = 0.006
        seg_w = (inner_w - 7 * seg_gap) / 8
        for rank, color in enumerate(palette):
            box(ax, inner_x + rank * (seg_w + seg_gap), bottom_y + 0.72, seg_w, 0.13,
                "", fc=color, ec=color, lw=0.4, r=0)
        text(ax, x + card_w / 2, bottom_y + 0.42, "完整 bf16", in_box=p,
             ha="center", va="center", fontsize=7.5)
        text(ax, x + card_w / 2, bottom_y + 0.20, "Wₖ（临时）", in_box=p,
             ha="center", va="center", fontsize=7.5)

    arrow_x = 3.80
    ax.annotate("", xy=(arrow_x, bottom_y + bottom_h), xytext=(arrow_x, top_y),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    text(ax, arrow_x, 2.52, "前向 pre-hook：all-gather Wₖ", ha="center", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#333333", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    text(ax, 3.8, 0.48, "算完即 release，回到上排状态", ha="center", va="center",
         fontsize=10, color="#333333")
    save(fig, "deepspeed-f1-8-gpus-zero3")


def f2():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 5.0); ax.axis("off")
    ax.set_title("DeepSpeed：训练循环在上，通信与分片基础在下")
    layers = [
        "groups.py / pipe/topology.py / comm：进程组与通信",
        "zero.Init / ds_tensor：构造时分片的参数",
        "DeepSpeedZeRoOffload hooks + PartitionedParameterCoordinator",
        "DeepSpeedZeroOptimizer / DeepSpeedZeroOptimizer_Stage3",
        "DeepSpeedEngine：包装模型，接入 optimizer 与 ZeRO",
        "用户训练循环：forward / backward / step",
    ]
    x, w, h, gap = 0.68, 5.95, 0.48, 0.23
    ys = [0.30 + i * (h + gap) for i in range(len(layers))]
    for y, label in zip(ys, layers):
        box(ax, x, y, w, h, label, fc="#ffffff", ec="#444444", fs=10.5)
    for i in range(len(ys) - 1):
        lower_top = ys[i] + h
        upper_bottom = ys[i + 1]
        ax.annotate("", xy=(3.66, upper_bottom), xytext=(3.66, lower_top),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
    text(ax, 6.83, 2.55, "依赖方向 ↑", rotation=90, ha="center", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#333333")
    save(fig, "deepspeed-f2-abstractions")


def f3():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.0); ax.axis("off")
    ax.set_title("Stage 1 / 2：param group 拍平，再按 DP rank 切段")
    x0, ybar, cell_w, gap = 0.62, 2.78, 0.47, 0.055
    colors = [C["blue"] + "55", C["orange"] + "66", C["green"] + "66", "#dddddd"]
    labels = ["A0", "A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3", "C0", "C1", "pad"]
    groups = [0] * 5 + [1] * 4 + [2] * 2 + [3]
    text(ax, 3.8, 3.66, "例：A 有 5 个元素，B 有 4 个，C 有 2 个；总数 11，补 1 个 pad 到 12",
         ha="center", va="center", fontsize=10.5, color="#333333")
    for i, (label, group) in enumerate(zip(labels, groups)):
        box(ax, x0 + i * (cell_w + gap), ybar, cell_w, 0.43, label,
            fc=colors[group], ec="#555555", fs=9.5)
    text(ax, x0 + 1.30, 3.31, "A (5)", ha="center", fontsize=9.5, color="#333333")
    text(ax, x0 + 3.74, 3.31, "B (4)", ha="center", fontsize=9.5, color="#333333")
    text(ax, x0 + 5.57, 3.31, "C + pad", ha="center", fontsize=9.5, color="#333333")

    rank_w = 1.56
    rank_gap = 0.28
    rank_x = [0.30 + i * (rank_w + rank_gap) for i in range(4)]
    for rank, x in enumerate(rank_x):
        text(ax, x + rank_w / 2, 2.36, f"rank {rank}：元素 {rank * 3}–{rank * 3 + 2}",
             row="rank-labels", ha="center", va="center", fontsize=9.5, color="#333333")
        ax.annotate("", xy=(x + rank_w / 2, 1.75), xytext=(x + rank_w / 2, 2.13),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
        box(ax, x, 1.02, rank_w, 0.74, f"fp32 主参数副本\n3 个元素", fc="#f5f8fc", ec="#444444", fs=9.5)
    text(ax, 3.8, 0.48, "每个 rank 更新自己的 fp32 段；更新后转回低精度参数分区",
         ha="center", va="center", fontsize=10.5, color="#333333")
    save(fig, "deepspeed-f3-flat-partition")


def f4():
    fig, ax = plt.subplots(figsize=(7.6, 4.55))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.55); ax.axis("off")
    ax.set_title("一个 Linear.weight：前向与反向按时间经过三个泳道")
    start, step, cell_w = 1.22, 0.88, 0.78
    compute_y, comm_y, state_y, box_h, compute_h = 3.25, 2.35, 1.16, 0.60, 0.70
    centers = [start + i * step + cell_w / 2 for i in range(7)]
    text(ax, 1.02, compute_y + compute_h / 2, "计算", ha="right", va="center", fontsize=10.5, color="#333333")
    text(ax, 1.02, comm_y + box_h / 2, "通信", ha="right", va="center", fontsize=10.5, color="#333333")
    text(ax, 1.02, state_y + box_h / 2, "param.data", ha="right", va="center", fontsize=10.5, color="#333333")

    comm_events = {0: "前向\nall-gather", 3: "反向\nall-gather", 5: "reduce-\nscatter"}
    compute_events = {1: "前向\nLinear", 4: "反向\ndW", 6: "step\nAdam fp32\n→ ds_tensor"}
    for i in range(7):
        x = start + i * step
        if i in comm_events:
            box(ax, x, comm_y, cell_w, box_h, comm_events[i], fc="#ffffff", ec="#555555", fs=9)
        if i in compute_events:
            label = compute_events[i]
            box(ax, x, compute_y, cell_w, compute_h, label, fc="#ffffff", ec="#555555",
                fs=7.5 if i == 6 else 9)

    states = [("完整 bf16", C["blue"] + "44"), ("完整 bf16", C["blue"] + "44"),
              ("空壳", "#eeeeee"), ("完整 bf16", C["blue"] + "44"),
              ("完整 bf16", C["blue"] + "44"), ("空壳", "#eeeeee"), ("空壳", "#eeeeee")]
    for i, (label, color) in enumerate(states):
        box(ax, start + i * step, state_y, cell_w, box_h, label, fc=color, ec="#555555", fs=9.5)

    release_x = (centers[1] + centers[2]) / 2
    release_bwd_x = (centers[4] + centers[5]) / 2
    text(ax, release_x, 2.02, "post-hook release", ha="center", va="center", fontsize=8.5, color="#333333")
    text(ax, release_bwd_x, 2.02, "post-hook release", ha="center", va="center", fontsize=8.5, color="#333333")
    ax.annotate("", xy=(centers[-1], 0.73), xytext=(centers[0], 0.73),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
    text(ax, 3.80, 0.31, "ds_tensor 常驻 1/N_d；完整 param.data 用前取回、用完释放",
         ha="center", va="center", fontsize=9.5, color="#333333")
    save(fig, "deepspeed-f4-param-lifecycle")


def f5():
    fig, ax = plt.subplots(figsize=(7.6, 4.25))
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(-80, 280)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([0, 121.6, 243.2]); ax.set_yticklabels(["0", "121.6", "243.2"])
    ax.set_ylabel("完整 bf16 参数常驻（MB）", color="#333333")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    layer_mb = 60_821_504 * 2 / 1_000_000
    x = list(range(12))
    no_prefetch_layers = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    prefetch_layers = [1, 2, 2, 1, 1, 0, 1, 2, 2, 1, 1, 0]
    no_prefetch = [count * layer_mb for count in no_prefetch_layers]
    prefetch = [count * layer_mb for count in prefetch_layers]
    ax.plot(x, no_prefetch, color=C["blue"], marker="o", linewidth=2, markersize=4)
    ax.plot(x, prefetch, color=C["orange"], marker="o", linewidth=2, markersize=4)
    ax.plot([3.4, 3.8], [264, 264], color=C["blue"], linewidth=2)
    text(ax, 3.9, 264, "预取深度 0", ha="left", va="center", fontsize=9.5, color=C["blue"])
    ax.plot([5.4, 5.8], [264, 264], color=C["orange"], linewidth=2)
    text(ax, 5.9, 264, "预取深度 1", ha="left", va="center", fontsize=9.5, color=C["orange"])

    ax.plot([0, 5], [-18, -18], color="#555555", linewidth=0.8)
    ax.plot([0, 0], [-18, -12], color="#555555", linewidth=0.8)
    ax.plot([5, 5], [-18, -12], color="#555555", linewidth=0.8)
    ax.plot([6, 11], [-18, -18], color="#555555", linewidth=0.8)
    ax.plot([6, 6], [-18, -12], color="#555555", linewidth=0.8)
    ax.plot([11, 11], [-18, -12], color="#555555", linewidth=0.8)
    text(ax, 2.5, -39, "前向", row="phase-labels", ha="center", va="center", fontsize=9.5, color="#333333")
    text(ax, 8.5, -39, "反向", row="phase-labels", ha="center", va="center", fontsize=9.5, color="#333333")
    text(ax, 5.5, -68, "toy hook 事件顺序", ha="center", va="center", fontsize=9.5, color="#333333")
    save(fig, "deepspeed-f5-live-params-sawtooth")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5()
