"""大规模训练工程（03）Megatron-LM 配图。输出 out/megatron-*.svg，拷进博客 img/in-post/。

    python 03_megatron_figures.py        # 四张全画

    F1  8 张卡上的 Megatron：TP2×PP2×DP2，每卡标注持有的层 / TP 半张矩阵 / 常驻状态
    F2  核心抽象的分层：TP 层搭模型 → DDP 包 buffer/bucket → DistributedOptimizer 切片 → schedules 驱动 → parallel_state 供组
    F3  一个 step 的泳道：计算流 / 通信流 / 常驻
    F4  1F1B 调度：PP=2、m=8，warmup / steady / cooldown 三段标注（序列由 03_megatron_toys.schedule_1f1b 生成）
"""
import matplotlib.patches as mpatches

from _plot import ARROW_LABEL_SIZE, C, box, box_title, plt, save, text


# ---------------------------------------------------------------- F1
def f1():
    """8 卡 TP2×PP2×DP2。rank = tp + 2·dp + 4·pp。
    布局：2 列 × 4 行。行序（上→下）：stage1·dp0、stage1·dp1、stage0·dp0、stage0·dp1。
    同行相邻两张卡 = TP 组；相邻行同列 = DP 组（同 stage）；
    PP 组走左右外侧拐角箭头，连同一 tp 位的两个 stage。"""
    fig, ax = plt.subplots(figsize=(7.4, 8.6))
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.36, 4.00); ax.axis("off")

    cw, ch = 0.375, 0.82
    colx = [0.065, 0.575]                    # tp0 列、tp1 列；外侧各留两条 PP 走线带
    rowy = [3.10, 2.15, 1.20, 0.25]          # stage1·dp0, stage1·dp1, stage0·dp0, stage0·dp1
    row_of = {4: 0, 5: 0, 6: 1, 7: 1, 0: 2, 1: 2, 2: 3, 3: 3}

    def rank_xy(rank):
        tp = rank % 2
        return colx[tp], rowy[row_of[rank]]

    card = {
        0: ("GPU 0  tp0 dp0 pp0", "layers 0–7 + embedding\n矩阵左半（tp0）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        1: ("GPU 1  tp1 dp0 pp0", "layers 0–7 + embedding\n矩阵右半（tp1）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        4: ("GPU 4  tp0 dp0 pp1", "layers 8–15 + 输出层\n矩阵左半（tp0）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        5: ("GPU 5  tp1 dp0 pp1", "layers 8–15 + 输出层\n矩阵右半（tp1）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        2: ("GPU 2  tp0 dp1 pp0", "layers 0–7 + embedding\n矩阵左半（tp0）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        3: ("GPU 3  tp1 dp1 pp0", "layers 0–7 + embedding\n矩阵右半（tp1）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        6: ("GPU 6  tp0 dp1 pp1", "layers 8–15 + 输出层\n矩阵左半（tp0）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
        7: ("GPU 7  tp1 dp1 pp1", "layers 8–15 + 输出层\n矩阵右半（tp1）", "bf16 参数（本卡一半）\nfp32 梯度（完整）\nfp32 主参+m+v（×1/2）"),
    }
    for rank, (t1, t2, t3) in card.items():
        x, y = rank_xy(rank)
        p = box(ax, x, y, cw, ch, "", fc="#f7f9fc")
        text(ax, x + cw / 2, y + ch - 0.090, t1, in_box=p, ha="center", va="center", fontsize=11, weight="bold", color="#111111")
        text(ax, x + cw / 2, y + 0.56, t2, in_box=p, ha="center", va="center", fontsize=10.5)
        text(ax, x + cw / 2, y + 0.24, t3, in_box=p, ha="center", va="center", fontsize=10.5, color="#111111")

    def edge(r, side):
        x, y = rank_xy(r)
        return {"l": (x, y + ch / 2), "r": (x + cw, y + ch / 2),
                "t": (x + cw / 2, y + ch), "b": (x + cw / 2, y)}[side]

    # TP：同一行内相邻两张卡
    for a, b in [(4, 5), (6, 7), (0, 1), (2, 3)]:
        (x1, y1), (x2, y2) = edge(a, "r"), edge(b, "l")
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=1.4))
        text(ax, 0.51, y1 + 0.055, "TP", ha="center", va="center", fontsize=10, color=C["blue"])
    # DP：相邻行同一 TP 位（同 stage 两个副本）
    for a, b in [(4, 6), (5, 7), (0, 2), (1, 3)]:
        (x1, y1), (x2, y2) = edge(a, "b"), edge(b, "t")
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="<->", color=C["orange"], lw=1.4))
        text(ax, x1 + 0.035, (y1 + y2) / 2, "DP", ha="left", va="center", fontsize=10, color=C["orange"])
    # PP：同一 TP 位的两个 stage——外侧拐角线
    def pp_elbow(a, b, gx, a_side, b_side, lx, ly, lha):
        (x1, y1) = edge(a, a_side)
        (x2, y2) = edge(b, b_side)
        ax.plot([x1, gx, gx, x2], [y1, y1, y2, y2], color=C["green"], lw=1.4)
        ax.annotate("", xy=(x2, y2), xytext=(x2 + (0.02 if b_side == "l" else -0.02), y2),
                    arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.4))
        text(ax, lx, ly, "PP", ha=lha, va="center", fontsize=10, color=C["green"])

    # 左右各两条外侧走线带：上行对（row2↔row0）走最外道，下行对（row3↔row1）走靠盒内道；
    # 标签放进只被本道占用的行隙里：上行对放顶隙、下行对放底隙，互不压线
    pp_elbow(0, 4, 0.014, "l", "l", 0.018, 3.02, "left")    # row2 → row0，外道
    pp_elbow(2, 6, 0.034, "l", "l", 0.038, 1.13, "left")    # row3 → row1，内道
    pp_elbow(1, 5, 0.986, "r", "r", 0.982, 3.02, "right")   # row2 → row0，外道
    pp_elbow(3, 7, 0.966, "r", "r", 0.970, 1.13, "left")    # row3 → row1，内道

    text(ax, 0.5, -0.10, "TP = 相邻两个 rank（同节点）　DP = 同一 stage 同一 TP 位的两张卡\nPP = 同一 DP 副本同一 TP 位的两个 stage",
            ha="center", fontsize=10, color="#111111")
    text(ax, 0.5, -0.23, "TP 组 [0,1] [2,3] [4,5] [6,7]　·　DP 组 [0,2] [1,3] [4,6] [5,7]　·　PP 组 [0,4] [1,5] [2,6] [3,7]",
            ha="center", fontsize=10, color="#111111")
    ax.set_title("train-ledger 配置：8 卡 TP2 × PP2 × DP2（Llama-3.2-1B 形状，16 层）")
    save(fig, "megatron-f1-8-gpus")


# ---------------------------------------------------------------- F2
def f2():
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.06); ax.axis("off")

    # 五层堆叠（自下而上），每层：框 + 一句话
    layers = [
        ("parallel_state（模块级全局变量 + RankGenerator）", "给上面每一层提供进程组：TP / DP / PP / …", "#444444"),
        ("tensor_parallel：ColumnParallel / RowParallel / mappings", "模型由切开的层搭成；通信是手写的 autograd.Function", C["blue"]),
        ("_ParamAndGradBuffer + Bucket(Group)", "两段连续内存；参数与 main_grad 是视图；通信单位是 bucket", C["orange"]),
        ("DistributedDataParallel", "包装模型，注册 hook：梯度就绪发 reduce-scatter、前向等 all-gather", C["green"]),
        ("DistributedOptimizer", "按 buffer 字节区间切 fp32 主参数与 m、v（ZeRO-1）", C["purple"]),
        ("schedules.py + P2PCommunicator", "驱动前向 / 反向时序；只通过 config 回调碰 DDP", C["red"]),
    ]
    y = 0.02
    for name, desc, col in layers:
        p = box(ax, 0.05, y, 0.9, 0.125, "", fc="white", ec=col, lw=1.3)
        box_title(ax, 0.075, y + 0.086, name, in_box=p, ha="left", color=col)
        text(ax, 0.075, y + 0.036, desc, in_box=p, ha="left", va="center", fontsize=11, color="#111111")
        y += 0.165
    for i in range(len(layers) - 1):
        lower_top = 0.02 + i * 0.165 + 0.125
        upper_bottom = 0.02 + (i + 1) * 0.165
        ax.annotate("", xy=(0.5, lower_top), xytext=(0.5, upper_bottom),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
    ax.set_title("Megatron Core 的六个核心抽象：上面是训练调度，下面是进程组")
    save(fig, "megatron-f2-abstractions")


# ---------------------------------------------------------------- F3
def f3():
    """一个 step（DP rank 视角）的泳道：上 = 计算流，中 = 通信流，下 = 常驻。"""
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 3.5); ax.axis("off")

    lanes = [(2.46, "计算流", "#f4f7fb"), (1.34, "通信流", "#fdf7ef"), (0.28, "常驻", "#f5f5f5")]
    lane_h = {2.46: 0.98, 1.34: 1.04, 0.28: 1.00}
    for y0, name, fc in lanes:
        ax.add_patch(mpatches.Rectangle((0, y0), 16, lane_h[y0], fc=fc, ec="#cccccc", lw=0.6))
        text(ax, -0.55, y0 + lane_h[y0] / 2, name, ha="right", va="center", fontsize=11, weight="bold")

    def cell(x, w, y0, text, fc, fs=11, h=0.62):
        box(ax, x, y0, w, h, text, fc=fc, fs=fs, r=0.03)

    # 计算流：m 个 micro-batch 前向/反向 → step
    cell(0.25, 9.2, 2.62, "8 个 micro-batch 的前向 + 反向\n（dW 直接累加进 main_grad，不通信）", C["blue"] + "22", h=0.66)
    cell(9.6, 2.9, 2.62, "finalize\nmodel_grads", "#eeeeee", h=0.66)
    cell(12.5, 3.4, 2.62, "optimizer.step\n本 rank 的 1/2 段", C["purple"] + "22", h=0.66)
    # 通信流：reduce-scatter 的盒子和前向+反向盒子的末尾在时间轴上重叠
    cell(8.0, 4.0, 1.38, "reduce-scatter\ngrad_data\n（与末尾反向重叠）", C["orange"] + "33", h=0.96)
    cell(12.3, 3.5, 1.38, "all-gather\nparam_data\n（发出，不等）", C["green"] + "33", h=0.96)
    cell(0.55, 3.9, 1.38, "all-gather\n（上一 step 发出）\n前向 pre-hook 里等", C["green"] + "33", h=0.96)
    # 常驻
    cell(0.15, 15.7, 0.42, "bf16 参数（完整）· fp32 梯度 buffer（完整）\nfp32 主参数 + m + v（1/2）——整个训练期不释放",
         "#ffffff", h=0.72)
    # 三条带标签的关系：等待（上指）、触发（下指）、发起（下指）。箭头居中对准下方通信盒
    ax.annotate("", xy=(2.5, 2.62), xytext=(2.5, 2.34), arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    text(ax, 2.65, 2.48, "前向 pre-hook 等它完成", ha="left", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(11.0, 2.34), xytext=(11.0, 2.62), arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    text(ax, 10.85, 2.48, "末个 micro-batch 反向触发", ha="right", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.annotate("", xy=(14.05, 2.34), xytext=(14.05, 2.62), arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    text(ax, 14.2, 2.48, "step 发起", ha="left", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
    ax.set_title("一个 DP rank 的一个 step：两次 DP 通信都与计算重叠（ZeRO-1 的 2N）")
    save(fig, "megatron-f3-step-swimlane")


# ---------------------------------------------------------------- F4
def f4():
    import importlib
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    toys = importlib.import_module("03_megatron_toys")
    timeline = toys.schedule_timeline(2, 8)

    n = len(timeline[0])
    half = (n + 1) // 2                      # 时间轴折成两段，格子才放得下 F1/B8
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(-2.0, half + 0.9); ax.set_ylim(-0.60, 4.55); ax.axis("off")
    labels = {0: "stage 0\n(layers 0–7)", 1: "stage 1\n(layers 8–15)"}
    phase_c = {"warmup": C["blue"] + "22", "steady": C["green"] + "22", "cooldown": C["orange"] + "33"}
    seg_names = ["第 1–9 拍", "第 10–18 拍"]

    def bracket(x0, x1, y, name, seg):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=1.0))
        ax.plot([x0, x0], [y - 0.06, y], color="#555555", lw=1.0)
        ax.plot([x1, x1], [y - 0.06, y], color="#555555", lw=1.0)
        text(ax, (x0 + x1) / 2, y + 0.10, name, row=f"phase{seg}", ha="center", fontsize=10, color="#111111")

    for seg in range(2):
        t0, t1 = seg * half, min((seg + 1) * half, n)
        rows_y = [3.25 - 2 * seg, 2.45 - 2 * seg]     # 段内上面 stage0、下面 stage1
        for stage, y in zip((0, 1), rows_y):
            for t in range(t0, t1):
                item = timeline[stage][t]
                if item is None:
                    box(ax, t - t0 + 0.05, y, 0.92, 0.62, "", fc="#f4f4f4", ec="#bbbbbb")
                else:
                    ph, event = item
                    box(ax, t - t0 + 0.05, y, 0.92, 0.62, event, fc=phase_c[ph], fs=11)
            text(ax, -0.35, y + 0.31, labels[stage], ha="right", va="center", fontsize=11)
        # 段上方的阶段括号：每拍取该拍有事件的 stage 的 phase（两段都标）
        phases = []
        for t in range(t0, t1):
            ph = None
            for stage in (0, 1):
                if timeline[stage][t] is not None:
                    ph = timeline[stage][t][0]
            phases.append(ph)
        yb = 4.02 - 2 * seg
        text(ax, -0.45, yb + 0.10, seg_names[seg], ha="right", va="center", fontsize=ARROW_LABEL_SIZE, color="#111111")
        run_start, run_ph = 0, phases[0]
        for i in range(1, len(phases) + 1):
            ph = phases[i] if i < len(phases) else None
            if ph != run_ph:
                if run_ph is not None:
                    bracket(run_start + 0.05, i - 0.05, yb,
                            run_ph + ("（1F1B 交替）" if run_ph == "steady" else ""), seg)
                run_start, run_ph = i, ph

    text(ax, half / 2, -0.42, "灰格 = 该 stage 此拍等待", ha="center", fontsize=10, color="#111111")
    ax.set_title("1F1B 调度：PP=2、m=8（全局时钟，每段 9 拍）")
    save(fig, "megatron-f4-1f1b-schedule")


if __name__ == "__main__":
    f1(); f2(); f3(); f4()
    print("out/megatron-f*.svg done")
