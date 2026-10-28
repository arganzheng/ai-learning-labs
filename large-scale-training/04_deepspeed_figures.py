"""DeepSpeed 04 配图：以 750px 正文列宽绘制，所有文字用 _plot.py 的碰撞检查。"""
from _plot import ARROW_LABEL_SIZE, C, box, box_title, plt, save, text


def f1():
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("ZeRO-3：每个 rank 常驻 W_k 的 1/8，用时再拼完整")
    palette = ["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3",
               "#f7efc9", "#d8edef", "#f3dfe2", "#e5e5d7"]
    card_w, card_h = 1.80, 1.50
    card_x = [0.10 + col * 1.85 for col in range(4)]
    card_y = {0: 4.62, 1: 3.02}

    for rank in range(8):
        x, y = card_x[rank % 4], card_y[rank // 4]
        p = box(ax, x, y, card_w, card_h, "", fc="#ffffff", ec="#444444", lw=0.9)
        box_title(ax, x + card_w / 2, y + 1.20, f"rank {rank}", in_box=p)
        box(ax, x + 0.16, y + 1.00, card_w - 0.32, 0.12, "", fc=palette[rank], ec=palette[rank])
        text(ax, x + card_w / 2, y + 0.76, "bf16 ds_tensor 1/8", in_box=p,
             ha="center", va="center", fontsize=11)
        text(ax, x + card_w / 2, y + 0.51, "梯度分区 1/8 (bf16)", in_box=p,
             ha="center", va="center", fontsize=11)
        text(ax, x + card_w / 2, y + 0.26, "fp32 主参+m+v 1/8", in_box=p,
             ha="center", va="center", fontsize=11)

    card_x2, card_y2, card_w2, card_h2 = 1.30, 0.72, 5.0, 1.62
    p2 = box(ax, card_x2, card_y2, card_w2, card_h2, "", fc="#ffffff", ec="#444444", lw=1.0)
    box_title(ax, card_x2 + card_w2 / 2, card_y2 + 1.36, "每张 rank 都拼出完整 bf16 W_k（临时）",
              in_box=p2)
    bar_x, bar_w = card_x2 + 0.55, card_w2 - 1.10
    seg_w = bar_w / 8
    for rank, color in enumerate(palette):
        box(ax, bar_x + rank * seg_w, card_y2 + 0.62, seg_w, 0.18, "",
            fc=color, ec="#888888", lw=0.4, r=0)
        text(ax, bar_x + rank * seg_w + seg_w / 2, card_y2 + 0.40, str(rank), in_box=p2,
             ha="center", va="center", fontsize=10)

    arrow_x = 3.80
    ax.annotate("", xy=(arrow_x, card_y2 + card_h2), xytext=(arrow_x, card_y[1]),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    text(ax, arrow_x, 2.66, "前向 pre-hook：all-gather W_k", ha="center", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    text(ax, 3.8, 0.32, "算完即 release，回到上排状态", ha="center", va="center",
         fontsize=11, color="#111111")
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
        p = box(ax, x, y, w, h, "", fc="#ffffff", ec="#444444")
        text(ax, x + w / 2, y + h / 2, label, in_box=p, ha="center", va="center",
             fontsize=11, weight="bold")
    for i in range(len(ys) - 1):
        lower_top = ys[i] + h
        upper_bottom = ys[i + 1]
        ax.annotate("", xy=(3.66, lower_top), xytext=(3.66, upper_bottom),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
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
         ha="center", va="center", fontsize=11, color="#111111")
    for i, (label, group) in enumerate(zip(labels, groups)):
        box(ax, x0 + i * (cell_w + gap), ybar, cell_w, 0.43, label,
            fc=colors[group], ec="#555555", fs=11)
    text(ax, x0 + 1.30, 3.31, "A (5)", ha="center", fontsize=11, color="#111111")
    text(ax, x0 + 3.74, 3.31, "B (4)", ha="center", fontsize=11, color="#111111")
    text(ax, x0 + 5.57, 3.31, "C + pad", ha="center", fontsize=11, color="#111111")

    rank_w = 1.56
    rank_gap = 0.28
    rank_x = [0.30 + i * (rank_w + rank_gap) for i in range(4)]
    for rank, x in enumerate(rank_x):
        text(ax, x + rank_w / 2, 2.36, f"rank {rank}：元素 {rank * 3}–{rank * 3 + 2}",
             row="rank-labels", ha="center", va="center", fontsize=11, color="#111111")
        ax.annotate("", xy=(x + rank_w / 2, 1.75), xytext=(x + rank_w / 2, 2.13),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
        p = box(ax, x, 1.02, rank_w, 0.74, "", fc="#f5f8fc", ec="#444444")
        box_title(ax, x + rank_w / 2, 1.58, "fp32 主参数副本", in_box=p)
        text(ax, x + rank_w / 2, 1.24, "3 个元素", in_box=p, ha="center", va="center", fontsize=11)
    text(ax, 3.8, 0.48, "每个 rank 更新自己的 fp32 段；更新后转回低精度参数分区",
         ha="center", va="center", fontsize=11, color="#111111")
    save(fig, "deepspeed-f3-flat-partition")


def f4():
    fig, ax = plt.subplots(figsize=(7.95, 4.55))
    ax.set_xlim(-0.02, 7.95); ax.set_ylim(0, 4.55); ax.axis("off")
    ax.set_title("一个 Linear.weight：前向与反向按时间经过三个泳道")
    start, step, cell_w = 1.06, 0.95, 0.95
    compute_y, comm_y, state_y, box_h, compute_h = 3.10, 2.30, 1.16, 0.68, 0.90
    centers = [start + i * step + cell_w / 2 for i in range(7)]
    text(ax, 0.88, compute_y + compute_h / 2, "计算", ha="right", va="center", fontsize=11, color="#111111")
    text(ax, 0.88, comm_y + box_h / 2, "通信", ha="right", va="center", fontsize=11, color="#111111")
    text(ax, 0.88, state_y + box_h / 2, "param.data", ha="right", va="center", fontsize=11, color="#111111")

    comm_events = {0: "前向\nall-gather", 3: "反向\nall-gather", 5: "reduce-\nscatter"}
    compute_events = {1: "前向\nLinear", 4: "反向\ndW", 6: "step\nAdam fp32\n→ ds_tensor"}
    for i in range(7):
        x = start + i * step
        if i in comm_events:
            box(ax, x, comm_y, cell_w, box_h, comm_events[i], fc="#ffffff", ec="#555555", fs=11)
        if i in compute_events:
            label = compute_events[i]
            event_w = 1.18 if i == 6 else cell_w
            event_h = 0.94 if i == 6 else compute_h
            p = box(ax, centers[i] - event_w / 2, compute_y + (compute_h - event_h) / 2,
                    event_w, event_h, label if i != 6 else "", fc="#ffffff", ec="#555555", fs=11)
            if i == 6:
                box_title(ax, centers[i], compute_y + 0.70, "step", in_box=p)
                text(ax, centers[i], compute_y + 0.31, "Adam fp32\n→ ds_tensor",
                     in_box=p, ha="center", va="center", fontsize=11)

    states = [("完整 bf16", C["blue"] + "44"), ("完整 bf16", C["blue"] + "44"),
              ("空壳", "#eeeeee"), ("完整 bf16", C["blue"] + "44"),
              ("完整 bf16", C["blue"] + "44"), ("空壳", "#eeeeee"), ("空壳", "#eeeeee")]
    for i, (label, color) in enumerate(states):
        box(ax, start + i * step, state_y, cell_w, box_h, label, fc=color, ec="#555555", fs=11)

    release_x = start + 2 * step
    release_bwd_x = start + 5 * step
    text(ax, release_x, 2.07, "post-hook release", ha="center", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111")
    text(ax, release_bwd_x, 2.07, "post-hook release", ha="center", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(centers[-1], 0.73), xytext=(centers[0], 0.73),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
    text(ax, 3.95, 0.31, "ds_tensor 常驻 1/N_d；完整 param.data 用前取回、用完释放",
         ha="center", va="center", fontsize=10, color="#111111")
    save(fig, "deepspeed-f4-param-lifecycle")


def f5():
    fig, ax = plt.subplots(figsize=(7.6, 4.25))
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(-80, 280)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([0, 121.6, 243.2]); ax.set_yticklabels(["0", "121.6", "243.2"])
    ax.set_ylabel("完整 bf16 参数常驻（MB）", color="#111111")
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
    text(ax, 3.9, 264, "预取深度 0", ha="left", va="center", fontsize=10, color=C["blue"])
    ax.plot([5.4, 5.8], [264, 264], color=C["orange"], linewidth=2)
    text(ax, 5.9, 264, "预取深度 1", ha="left", va="center", fontsize=10, color=C["orange"])

    ax.plot([0, 5], [-18, -18], color="#555555", linewidth=0.8)
    ax.plot([0, 0], [-18, -12], color="#555555", linewidth=0.8)
    ax.plot([5, 5], [-18, -12], color="#555555", linewidth=0.8)
    ax.plot([6, 11], [-18, -18], color="#555555", linewidth=0.8)
    ax.plot([6, 6], [-18, -12], color="#555555", linewidth=0.8)
    ax.plot([11, 11], [-18, -12], color="#555555", linewidth=0.8)
    text(ax, 2.5, -39, "前向", row="phase-labels", ha="center", va="center", fontsize=10, color="#111111")
    text(ax, 8.5, -39, "反向", row="phase-labels", ha="center", va="center", fontsize=10, color="#111111")
    text(ax, 5.5, -68, "toy hook 事件顺序", ha="center", va="center", fontsize=10, color="#111111")
    save(fig, "deepspeed-f5-live-params-sawtooth")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5()
