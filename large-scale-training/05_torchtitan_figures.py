"""torchtitan 05 配图：以 750px 正文列宽绘制，所有文字用 _plot.py 的碰撞检查。"""
from _plot import ARROW_LABEL_SIZE, C, box, box_title, plt, save, text


def f1():
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.set_title("FSDP 4 × TP 2：8 张 rank 的 DeviceMesh（TP 在最内维）")
    text(ax, 3.8, 6.16, "行内 TP all-reduce（双向）；列内 dp_shard all-gather（向下）",
         ha="center", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")

    card_w, card_h = 2.75, 1.03
    xs = [1.02, 4.03]
    ys = [4.95 - i * 1.13 for i in range(4)]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            rank = i * 2 + j
            p = box(ax, x, y, card_w, card_h, "", fc="#ffffff", ec="#444444", lw=0.9)
            box_title(ax, x + card_w / 2, y + 0.83, f"rank {rank}", in_box=p)
            tp_half = "前半" if j == 0 else "后半"
            text(ax, x + card_w / 2, y + 0.62, f"TP 切片{tp_half}；norm 复制",
                 in_box=p, ha="center", va="center", fontsize=11)
            text(ax, x + card_w / 2, y + 0.40, "fp32 DTensor 参数 1/8",
                 in_box=p, ha="center", va="center", fontsize=11)
            text(ax, x + card_w / 2, y + 0.20, "梯度 1/8；Adam m/v 1/8",
                 in_box=p, ha="center", va="center", fontsize=11)

    for y in ys:
        ax.annotate("", xy=(xs[1], y + card_h / 2), xytext=(xs[0] + card_w, y + card_h / 2),
                    arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=1.1))
    for x in xs:
        for row in range(3):
            ax.annotate("", xy=(x + card_w / 2, ys[row + 1] + card_h),
                        xytext=(x + card_w / 2, ys[row] + 0.01),
                        arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0))

    p2 = box(ax, 1.05, 0.24, 5.50, 0.78, "", fc="#f5f8fc", ec="#444444")
    box_title(ax, 3.8, 0.76, "该层 unshard 后的 param.data", in_box=p2)
    text(ax, 3.8, 0.46, "bf16 完整沿 dp_shard；TP rank 仍各持矩阵一半（临时）",
         in_box=p2, ha="center", va="center", fontsize=11)
    text(ax, 3.8, 0.05, "TP 复制的 norm：67,584 / N（约 0.0055%），不按 TP 再切",
         ha="center", va="bottom", fontsize=ARROW_LABEL_SIZE, color="#111111")
    save(fig, "torchtitan-f1-device-mesh")


def f2():
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.set_title("torchtitan：上层调用 / 依赖下层")
    layers = [
        ("配置函数 / Trainer.Config", "选择模型、并行度与组件", C["blue"]),
        ("Trainer", "装配、训练步、checkpoint 与日志", C["green"]),
        ("parallelize_llama / ShardingConfig / pipeline_llm", "模型级 TP、CP、FSDP 与 PP", C["orange"]),
        ("fully_shard / PipelineStage", "FSDPState、FSDPParamGroup 与 stage 调度", C["purple"]),
        ("DTensor + placement", "Shard / Replicate / Partial；redistribute", C["red"]),
        ("DeviceMesh", "ParallelDims.build_mesh()：轴名与 rank 网格", C["blue"]),
        ("ProcessGroup", "真正执行集合通信的进程组", C["green"]),
    ]
    x, w, h, gap = 0.83, 5.94, 0.57, 0.17
    top = 6.12
    for i, (title, detail, color) in enumerate(layers):
        y = top - i * (h + gap) - h
        p = box(ax, x, y, w, h, "", fc="#ffffff", ec="#444444")
        box_title(ax, x + w / 2, y + h - 0.17, title, in_box=p, color=color)
        text(ax, x + w / 2, y + 0.17, detail, in_box=p, ha="center", va="center", fontsize=11)
        if i > 0:
            ax.annotate("", xy=(x + w / 2, y + h), xytext=(x + w / 2, y + h + gap),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
    save(fig, "torchtitan-f2-abstractions")


def f3():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("DTensor 的三种 placement：张量、局部值与通信")

    x, w, h = 1.10, 3.75, 1.10
    rows = [
        ("Shard(0)", "rank0 [[0,1,2,3],[4,5,6,7]]\nrank1 [[8,9,10,11],[12,13,14,15]]", 3.78),
        ("Replicate", "rank0 / rank1 都是完整 W\n[[0,1,2,3], …, [12,13,14,15]]", 2.12),
        ("Partial", "rank0 前两行非零；rank1 后两行非零\n逐元素求和恢复完整 W", 0.46),
    ]
    for name, det, y in rows:
        p = box(ax, x, y, w, h, "", fc="#ffffff", ec="#444444")
        box_title(ax, x + w / 2, y + h - 0.21, name, in_box=p)
        text(ax, x + w / 2, y + 0.38, det, in_box=p, ha="center", va="center", fontsize=11)

    # 每条箭头只表示 placement 转换触发的集合通信。
    ax.annotate("", xy=(5.30, 3.30), xytext=(5.30, 3.78),
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.1))
    text(ax, 5.62, 3.54, "Shard(0) → Replicate\nall-gather", ha="left", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(5.30, 2.12), xytext=(5.30, 1.56),
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.1))
    text(ax, 5.62, 2.12, "Partial → Replicate\nall-reduce", ha="left", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(0.90, 3.78), xytext=(0.90, 1.56),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.1))
    text(ax, 0.78, 2.55, "Partial → Shard(0)\nreduce-scatter", ha="right", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111")
    text(ax, 3.8, 0.20, "数字张量取自 2-rank toy；箭头表示 redistribute 触发的通信",
         ha="center", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    save(fig, "torchtitan-f3-placements")


def f4():
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 3.5); ax.axis("off")
    ax.set_title("meta 构造五步走：模型先在 meta 上「画」出来，再按分片分配")

    stages = [
        ("meta 构造", "只有形状\n0 B"),
        ("fully_shard", "仍是 meta\n0 B"),
        ("to_empty", "参数分片\n14,336 B"),
        ("init_weights", "写入真实值\n14,336 B"),
        ("opt.step", "参数 + m/v\n71,680 B"),
    ]
    w, h = 1.38, 1.16
    xs = [0.30 + i * 1.44 for i in range(5)]
    y = 1.55
    for i, (title, det) in enumerate(stages):
        p = box(ax, xs[i], y, w, h, "", fc="#ffffff", ec="#444444")
        box_title(ax, xs[i] + w / 2, y + h - 0.20, title, in_box=p)
        text(ax, xs[i] + w / 2, y + 0.40, det, in_box=p, ha="center", va="center", fontsize=11)
        if i:
            ax.annotate("", xy=(xs[i], y + h / 2), xytext=(xs[i - 1] + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    text(ax, 3.8, 0.86, "2-rank toy 每 rank 实测；meta → to_empty 后才申请存储",
         ha="center", va="center", fontsize=11, color="#111111")
    text(ax, 3.8, 0.44, "真实 Llama 3.2 1B 字节账见总览；meta 设备可先构造 405B 结构",
         ha="center", va="center", fontsize=11, color="#111111")
    save(fig, "torchtitan-f4-meta-stages")


def f5():
    fig, ax = plt.subplots(figsize=(7.95, 4.55))
    ax.set_xlim(-0.02, 7.95); ax.set_ylim(0, 4.55); ax.axis("off")
    ax.set_title("一个 Linear.weight：fully_shard 下前向与反向按时间经过三个泳道")
    start, step, cell_w = 1.06, 0.95, 0.95
    compute_y, comm_y, state_y, box_h, compute_h = 3.10, 2.30, 1.16, 0.68, 0.90
    centers = [start + i * step + cell_w / 2 for i in range(7)]
    text(ax, 0.88, compute_y + compute_h / 2, "计算", ha="right", va="center", fontsize=11, color="#111111")
    text(ax, 0.88, comm_y + box_h / 2, "通信", ha="right", va="center", fontsize=11, color="#111111")
    text(ax, 0.88, state_y + box_h / 2, "param 状态", ha="right", va="center", fontsize=11, color="#111111")

    comm_events = {0: "unshard\nall-gather", 3: "unshard\nall-gather", 5: "reduce-\nscatter"}
    compute_events = {1: "前向\nLinear", 4: "反向\ndW", 6: "step"}
    for i in range(7):
        x = start + i * step
        if i in comm_events:
            box(ax, x, comm_y, cell_w, box_h, comm_events[i], fc="#ffffff", ec="#555555", fs=11)
        if i in compute_events:
            event_w = 1.18 if i == 6 else cell_w
            event_h = 0.94 if i == 6 else compute_h
            p = box(ax, centers[i] - event_w / 2, compute_y + (compute_h - event_h) / 2,
                    event_w, event_h, "" if i == 6 else compute_events[i],
                    fc="#ffffff", ec="#555555", fs=11)
            if i == 6:
                box_title(ax, centers[i], compute_y + 0.70, "step", in_box=p)
                text(ax, centers[i], compute_y + 0.31, "AdamW\nfp32 分片",
                     in_box=p, ha="center", va="center", fontsize=11)

    shard = "fp32\nShard(0)"
    full = "bf16\n完整"
    states = [(full, C["green"] + "55"), (full, C["green"] + "55"), (shard, C["blue"] + "44"),
              (full, C["green"] + "55"), (full, C["green"] + "55"), (shard, C["blue"] + "44"),
              (shard, C["blue"] + "44")]
    for i, (label, color) in enumerate(states):
        box(ax, start + i * step, state_y, cell_w, box_h, label, fc=color, ec="#555555", fs=11)

    for bx in (start + 2 * step, start + 5 * step):
        text(ax, bx, 2.07, "post-hook reshard", ha="center", va="center",
             fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(centers[-1], 0.73), xytext=(centers[0], 0.73),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
    text(ax, 3.95, 0.31, "fp32 分片常驻；完整 bf16 param 用前 unshard、用完 reshard",
         ha="center", va="center", fontsize=10, color="#111111")
    save(fig, "torchtitan-f5-param-lifecycle")


if __name__ == "__main__":
    f1(); f2(); f3(); f4(); f5()
