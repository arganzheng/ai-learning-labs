"""DeepSpeed 04 配图：以 750px 正文列宽绘制，所有文字用 _plot.py 的碰撞检查。"""
from _plot import ARROW_LABEL_SIZE, C, box, plt, save, text


def f1():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.8); ax.axis("off")
    ax.set_title("ZeRO-3：常驻分片，计算当前层时临时取回完整参数")

    for rank in range(8):
        col = rank % 4
        row = 1 - rank // 4
        x, y, w, h = 0.28 + col * 1.82, 2.42 + row * 1.10, 1.62, 0.94
        p = box(ax, x, y, w, h, "", fc="#f5f8fc", ec="#444444", lw=1.0)
        text(ax, x + w / 2, y + 0.75, f"rank {rank}", in_box=p, ha="center", va="center", fontsize=11, weight="bold")
        text(ax, x + w / 2, y + 0.55, "bf16 参数 1/8", in_box=p, ha="center", va="center", fontsize=10.5)
        text(ax, x + w / 2, y + 0.35, "梯度分区 1/8", in_box=p, ha="center", va="center", fontsize=10.5)
        text(ax, x + w / 2, y + 0.15, "fp32 + m + v 1/8", in_box=p, ha="center", va="center", fontsize=10.5)

    flow_y, flow_h = 0.45, 0.86
    flow = [
        (0.38, 1.85, "8 份 ds_tensor\n每卡保留 1/8"),
        (2.88, 1.85, "all-gather\n每卡得到完整 bf16 Wₖ"),
        (5.38, 1.85, "当前层算完\n释放完整 Wₖ"),
    ]
    patches = []
    for x, w, label in flow:
        patches.append(box(ax, x, flow_y, w, flow_h, label, fc="#ffffff", ec="#444444", fs=10.5))
    for left, right in ((2.23, 2.88), (4.73, 5.38)):
        ax.annotate("", xy=(right, flow_y + flow_h / 2), xytext=(left, flow_y + flow_h / 2),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
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
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("一个 Linear.weight：前向、反向各取回一次")
    left, start, step, cell_w = 1.15, 1.25, 0.88, 0.78
    rows = [("计算", 3.20), ("通信", 2.22), ("param.data", 0.93)]
    for label, y in rows:
        text(ax, left - 0.12, y + 0.25, label, ha="right", va="center", fontsize=10.5, color="#333333")
    events = {
        0: ("通信", "前向\nall-gather"),
        1: ("计算", "前向\nLinear"),
        2: ("通信", "release"),
        3: ("通信", "反向\nall-gather"),
        4: ("计算", "反向\ndW"),
        5: ("通信", "reduce-\nscatter"),
        6: ("计算", "step\nAdam"),
    }
    row_y = dict(rows)
    for i, (row, label) in events.items():
        box(ax, start + i * step, row_y[row], cell_w, 0.62, label, fc="#ffffff", ec="#555555", fs=9.5)
    states = [("空壳", "#eeeeee"), ("完整 bf16", C["blue"] + "44"), ("空壳", "#eeeeee"),
              ("完整 bf16", C["blue"] + "44"), ("空壳", "#eeeeee"), ("空壳", "#eeeeee"), ("空壳", "#eeeeee")]
    for i, (label, color) in enumerate(states):
        box(ax, start + i * step, 0.80, cell_w, 0.53, label, fc=color, ec="#555555", fs=9.5)
    text(ax, 4.25, 0.36, "ds_tensor 常驻为 1/N_d；完整 param.data 在 use 前取回、用后释放",
         ha="center", va="center", fontsize=10, color="#333333")
    save(fig, "deepspeed-f4-param-lifecycle")


def f5():
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(-0.4, 11.8); ax.set_ylim(-8, 106)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([0, 40, 80]); ax.set_ylabel("live full params (bytes)", color="#333333")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    x = list(range(12))
    no_prefetch = [48, 48, 0, 40, 40, 0, 40, 40, 0, 48, 48, 0]
    prefetch = [48, 88, 88, 40, 40, 0, 40, 88, 88, 48, 48, 0]
    ax.plot(x, no_prefetch, color=C["blue"], marker="o", linewidth=2, markersize=4)
    ax.plot(x, prefetch, color=C["orange"], marker="o", linewidth=2, markersize=4)
    text(ax, 8.15, 98, "prefetch=0（峰值 48 B）", ha="left", va="center", fontsize=10, color=C["blue"])
    text(ax, 8.15, 88, "prefetch=1（峰值 88 B）", ha="left", va="center", fontsize=10, color=C["orange"])
    text(ax, 2.2, -4, "forward hooks", ha="center", va="top", fontsize=9.5, color="#333333")
    text(ax, 8.1, -4, "backward hooks", ha="center", va="top", fontsize=9.5, color="#333333")
    ax.set_xlabel("toy hook 事件顺序", color="#333333")
    save(fig, "deepspeed-f5-live-params-sawtooth")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5()
