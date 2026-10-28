"""torchtitan 05 配图：以 750px 正文列宽绘制，所有文字用 _plot.py 的碰撞检查。"""
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from _plot import ARROW_LABEL_SIZE, C, box, box_title, plt, save, text


def f1():
    """8 卡 FSDP4×TP2（03 F1 风格）：列 = dp_shard 位，行 = tp 位。
    rank = tp + 2·dp：上排 tp0（rank 0/2/4/6），下排 tp1（rank 1/3/5/7）。"""
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("FSDP 4 × TP 2：8 张 rank 的 DeviceMesh（TP 在最内维）")

    cw, ch, gx = 1.57, 1.52, 0.33
    xs = [0.20 + j * (cw + gx) for j in range(4)]
    y0, y1 = 4.55, 2.55  # tp0 上排，tp1 下排
    for j, x in enumerate(xs):
        for tp, y in ((0, y0), (1, y1)):
            rank = 2 * j + tp
            p = box(ax, x, y, cw, ch, "", fc="#f7f9fc", ec="#444444", lw=0.9)
            box_title(ax, x + cw / 2, y + ch - 0.24, f"rank {rank} dp{j} tp{tp}", in_box=p)
            text(ax, x + cw / 2, y + ch - 0.60,
                 "矩阵前半（tp0）" if tp == 0 else "矩阵后半（tp1）",
                 in_box=p, ha="center", va="center", fontsize=11)
            text(ax, x + cw / 2, y + 0.44,
                 "fp32 参数 1/8\nfp32 梯度 1/8\nAdam m、v 1/8",
                 in_box=p, ha="center", va="center", fontsize=10)

    # TP：同一列上下两张卡，蓝色双向箭头，标签在箭头右侧、行隙中。
    for x in xs:
        xc = x + cw / 2
        ax.annotate("", xy=(xc, y1 + ch), xytext=(xc, y0),
                    arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=1.4))
        text(ax, xc + 0.16, (y1 + ch + y0) / 2, "TP", ha="left", va="center",
             fontsize=ARROW_LABEL_SIZE, color=C["blue"])

    # dp_shard：同一行相邻卡之间的橙色双向箭头，每行标一次。
    for y in (y0, y1):
        for j in range(3):
            ax.annotate("", xy=(xs[j + 1], y + ch / 2), xytext=(xs[j] + cw, y + ch / 2),
                        arrowprops=dict(arrowstyle="<->", color=C["orange"], lw=1.4))
    text(ax, 3.8, y0 + ch + 0.15, "dp_shard", ha="center", va="bottom",
         fontsize=ARROW_LABEL_SIZE, color=C["orange"])
    text(ax, (xs[0] + cw + xs[1]) / 2, y1 - 0.10, "dp_shard", ha="center", va="top",
         fontsize=ARROW_LABEL_SIZE, color=C["orange"])

    # 底部：04 F1 风格大卡，前向 pre-hook all-gather 后沿 dp_shard 拼完整。
    palette = ["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3"]
    p2 = box(ax, 1.30, 0.85, 5.0, 1.15, "", fc="#ffffff", ec="#444444", lw=1.0)
    box_title(ax, 3.8, 1.80, "某层 unshard 后（临时，bf16）", in_box=p2)
    bar_x, seg = 1.85, 3.90 / 4
    for j, (c, name) in enumerate(zip(palette, ("dp0", "dp1", "dp2", "dp3"))):
        box(ax, bar_x + j * seg, 1.36, seg, 0.18, "", fc=c, ec="#888888", lw=0.4, r=0)
        text(ax, bar_x + j * seg + seg / 2, 1.22, name, in_box=p2,
             ha="center", va="center", fontsize=10)
    text(ax, 3.8, 1.02, "DTensor 沿 dp_shard 拼完整，TP 仍只有本卡一半", in_box=p2,
         ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(3.8, 2.00), xytext=(3.8, 2.55),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color="#555555", lw=1.2))
    text(ax, 3.8, 2.28, "前向 pre-hook：all-gather", ha="center", va="center",
         fontsize=ARROW_LABEL_SIZE, color="#111111",
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

    text(ax, 3.8, 0.60, "TP = 同一列两张卡（SP：激活 all-gather / reduce-scatter）",
         ha="center", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    text(ax, 3.8, 0.37, "dp_shard = 同一行四张卡（FSDP：参数 all-gather、梯度 reduce-scatter）",
         ha="center", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    text(ax, 3.8, 0.14, "TP 组 [0,1] [2,3] [4,5] [6,7]　·　dp_shard 组 [0,2,4,6] [1,3,5,7]",
         ha="center", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
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
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("DTensor 的三种 placement：张量、局部值与通信")

    CW, RH = 0.28, 0.30
    _grids, _labels = [], []

    def grid(x0, y0, vals, in_box, shade_zero=False):
        rows, cols = len(vals), len(vals[0])
        _grids.append((x0, y0, cols * CW, rows * RH))
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                gx, gy = x0 + j * CW, y0 + (rows - 1 - i) * RH
                ax.add_patch(Rectangle((gx, gy), CW, RH,
                                       fc="#eeeeee" if (shade_zero and v == 0) else "#ffffff",
                                       ec="#999999", lw=0.5))
                text(ax, gx + CW / 2, gy + RH / 2, str(v), in_box=in_box,
                     ha="center", va="center", fontsize=11)

    def label(x, y_, s, in_box):
        t = text(ax, x, y_, s, in_box=in_box, ha="center", va="bottom",
                 fontsize=ARROW_LABEL_SIZE, color="#111111")
        _labels.append(t)
        return t

    # 所有 grid 的 bbox 与所有 rank 标签不得重叠（grid 是未登记的 Rectangle，collide 看不到）。
    def check_grid_labels():
        renderer = fig.canvas.get_renderer()
        for t in _labels:
            tb = t.get_window_extent(renderer)
            for gx, gy, gw, gh in _grids:
                gb = ax.transData.transform_bbox(Bbox([[gx, gy], [gx + gw, gy + gh]]))
                assert not tb.overlaps(gb), f"grid 与标签重叠: '{t.get_text()}'"

    y, h = 0.70, 2.40
    W4 = 4 * CW  # 1.12

    # Partial：每 rank 的 local 是 4×4，只填自己那两行的部分和，其余为 0（阴影只标这些补 0 格）。
    xp, wp = 0.10, 2.55
    p = box(ax, xp, y, wp, h, "", fc="#ffffff", ec="#444444")
    box_title(ax, xp + wp / 2, y + h - 0.22, "Partial", in_box=p)
    gx0 = xp + (wp - 2 * W4 - 0.15) / 2
    grid(gx0, 1.15, [[0, 1, 2, 3], [4, 5, 6, 7], [0, 0, 0, 0], [0, 0, 0, 0]], in_box=p, shade_zero=True)
    grid(gx0 + W4 + 0.15, 1.15, [[0, 0, 0, 0], [0, 0, 0, 0], [8, 9, 10, 11], [12, 13, 14, 15]], in_box=p, shade_zero=True)
    label(gx0 + W4 / 2, 2.43, "r0", in_box=p)
    label(gx0 + W4 + 0.15 + W4 / 2, 2.43, "r1", in_box=p)

    # Shard(0)：每 rank 的 local 是自己那两行的 2×4 分片。
    xs_, ws_ = 3.75, 1.30
    p = box(ax, xs_, y, ws_, h, "", fc="#ffffff", ec="#444444")
    box_title(ax, xs_ + ws_ / 2, y + h - 0.22, "Shard(0)", in_box=p)
    gx = xs_ + (ws_ - W4) / 2
    grid(gx, 1.82, [[0, 1, 2, 3], [4, 5, 6, 7]], in_box=p)
    label(gx + W4 / 2, 2.43, "r0", in_box=p)
    grid(gx, 0.84, [[8, 9, 10, 11], [12, 13, 14, 15]], in_box=p)
    label(gx + W4 / 2, 1.46, "r1", in_box=p)

    # Replicate：两 rank 的 local 都是完整 4×4。
    xr, wr = 6.15, 1.30
    p = box(ax, xr, y, wr, h, "", fc="#ffffff", ec="#444444")
    box_title(ax, xr + wr / 2, y + h - 0.22, "Replicate", in_box=p)
    gx = xr + (wr - W4) / 2
    grid(gx, 1.15, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], in_box=p)
    label(gx + W4 / 2, 2.43, "r0 / r1 相同", in_box=p)

    check_grid_labels()

    ymid = y + h / 2
    ax.annotate("", xy=(xs_, ymid), xytext=(xp + wp, ymid),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=C["green"], lw=1.1))
    text(ax, (xp + wp + xs_) / 2, ymid + 0.14, "reduce-\nscatter",
         ha="center", va="bottom", fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(xr, ymid), xytext=(xs_ + ws_, ymid),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=C["blue"], lw=1.1))
    text(ax, (xs_ + ws_ + xr) / 2, ymid + 0.14, "all-gather",
         ha="center", va="bottom", fontsize=ARROW_LABEL_SIZE, color="#111111")

    # Partial → Replicate 一步到位的 all-reduce，分解后就是上面两步。
    bx0, bx1 = xp + wp / 2, xr + wr / 2
    by = 3.42
    ax.plot([bx0, bx0], [y + h, by - 0.06], color=C["orange"], lw=1.1)
    ax.plot([bx1, bx1], [y + h, by - 0.06], color=C["orange"], lw=1.1)
    ax.annotate("", xy=(bx1, by), xytext=(bx0, by),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=C["orange"], lw=1.1))
    text(ax, (bx0 + bx1) / 2, by + 0.08, "all-reduce = reduce-scatter + all-gather",
         ha="center", va="bottom", fontsize=ARROW_LABEL_SIZE, color="#111111")

    text(ax, 3.8, 0.38, "数字张量取自 2-rank toy；箭头表示 redistribute 触发的通信",
         ha="center", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    save(fig, "torchtitan-f3-placements")


def f4():
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 2.7); ax.axis("off")
    ax.set_title("meta 构造五步走：模型先在 meta 上「画」出来，再按分片分配")

    stages = [
        ("meta 构造", "只有形状\n0 B", 1.15),
        ("fully_shard", "仍是 meta\n0 B", 1.15),
        ("to_empty", "参数分片\n14,336 B", 1.15),
        ("init_weights", "写入真实值\n14,336 B", 1.30),
        ("optimizer.step", "参数 + m/v\n71,680 B", 1.45),
    ]
    gap = 0.30
    xs = [0.10]
    for title, det, w in stages[:-1]:
        xs.append(xs[-1] + w + gap)
    y, h = 1.20, 1.16
    for i, ((title, det, w), x) in enumerate(zip(stages, xs)):
        p = box(ax, x, y, w, h, "", fc="#ffffff", ec="#444444")
        box_title(ax, x + w / 2, y + h - 0.20, title, in_box=p)
        text(ax, x + w / 2, y + 0.40, det, in_box=p, ha="center", va="center", fontsize=11)
        if i:
            ax.annotate("", xy=(x, y + h / 2), xytext=(xs[i - 1] + stages[i - 1][2], y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=13,
                                        color="#555555", lw=1.1))

    text(ax, 3.8, 0.50, "2-rank toy 每 rank 实测；meta → to_empty 后才申请存储",
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
