"""大规模训练工程（03）Megatron-LM 配图。输出 out/megatron-*.svg，拷进博客 img/in-post/。

    python 03_megatron_figures.py        # 四张全画

    F1  8 张卡上的 Megatron：TP2×PP2×DP2，每卡标注持有的层 / TP 半张矩阵 / 常驻状态
    F2  核心抽象的分层：TP 层搭模型 → DDP 包 buffer/bucket → DistributedOptimizer 切片 → schedules 驱动 → parallel_state 供组
    F3  一个 step 的泳道：计算流 / 通信流 / 常驻
    F4  1F1B 调度：PP=2、m=8，warmup / steady / cooldown 三段标注（序列由 03_megatron_toys.schedule_1f1b 生成）
"""
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from _plot import C, plt, save


def box(ax, x, y, w, h, text, fc="white", ec="#444444", fs=8.5, lw=0.9, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


# ---------------------------------------------------------------- F1
def f1():
    """8 卡 TP2×PP2×DP2。rank = tp + 2·dp + 4·pp。
    布局：行 = PP stage（上 stage1 layers8-15+输出层，下 stage0 layers0-7+embedding），
          列内每对相邻 rank = 一个 TP 组；左右两半 = 两个 DP 副本。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(-0.26, 1.12); ax.axis("off")

    cw, ch, gx, gy = 0.150, 0.30, 0.022, 0.10
    # DP 副本列：dp=0 → ranks 0,1(stage0) 4,5(stage1)；dp=1 → 2,3 / 6,7
    def rank_xy(rank):
        tp, dp, pp = rank % 2, (rank // 2) % 2, rank // 4
        x = 0.035 + dp * (2 * cw + 2 * gx + 0.06) + tp * (cw + gx)
        y = 0.16 + (1 - pp) * (ch + gy)   # stage0 在下
        return x, y

    card = {
        0: ("GPU 0  tp0 dp0 pp0", "layers 0–7 + embedding\n矩阵左半 (tp0)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        1: ("GPU 1  tp1 dp0 pp0", "layers 0–7 + embedding\n矩阵右半 (tp1)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        4: ("GPU 4  tp0 dp0 pp1", "layers 8–15 + 输出层\n矩阵左半 (tp0)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        5: ("GPU 5  tp1 dp0 pp1", "layers 8–15 + 输出层\n矩阵右半 (tp1)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        2: ("GPU 2  tp0 dp1 pp0", "layers 0–7 + embedding\n矩阵左半 (tp0)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        3: ("GPU 3  tp1 dp1 pp0", "layers 0–7 + embedding\n矩阵右半 (tp1)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        6: ("GPU 6  tp0 dp1 pp1", "layers 8–15 + 输出层\n矩阵左半 (tp0)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
        7: ("GPU 7  tp1 dp1 pp1", "layers 8–15 + 输出层\n矩阵右半 (tp1)", "bf16 参数(半层完整)\nfp32 梯度 完整\nfp32 主参+m+v ×1/2"),
    }
    for rank, (t1, t2, t3) in card.items():
        x, y = rank_xy(rank)
        box(ax, x, y, cw, ch, "", fc="#f7f9fc")
        ax.text(x + cw / 2, y + ch - 0.035, t1, ha="center", va="center", fontsize=7.5, weight="bold")
        ax.text(x + cw / 2, y + ch / 2 + 0.01, t2, ha="center", va="center", fontsize=7)
        ax.text(x + cw / 2, y + 0.055, t3, ha="center", va="center", fontsize=6.5, color="#555555")

    # 分组标注：TP（同 stage 内相邻）、DP（同 stage 同 tp 位、左右跨副本）、PP（同列上下）
    def edge(r, side):
        x, y = rank_xy(r)
        return {"l": (x, y + ch / 2), "r": (x + cw, y + ch / 2),
                "t": (x + cw / 2, y + ch), "b": (x + cw / 2, y)}[side]

    for a, b, lab in [(0, 1, "TP"), (4, 5, "TP"), (2, 3, "TP"), (6, 7, "TP")]:
        (x1, y1), (x2, y2) = edge(a, "r"), edge(b, "l")
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="<->", color=C["blue"], lw=1.4))
        ax.text((x1 + x2) / 2, y1 + 0.028, lab, ha="center", fontsize=7.5, color=C["blue"])
    # DP：同一 stage 同一 TP 位、跨左右两个副本——弧线走卡片外侧，不穿文字
    for a, b in [(0, 2), (1, 3)]:
        (x1, _), (_, yt) = edge(a, "t"), edge(b, "t")
        (x2, _) = edge(b, "t")
        ax.annotate("", xy=(x2, yt + 0.008), xytext=(x1, yt + 0.008),
                    arrowprops=dict(arrowstyle="<->", color=C["orange"], lw=1.4,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text((x1 + x2) / 2, yt + 0.075, "DP", ha="center", fontsize=7.5, color=C["orange"])
    for a, b in [(4, 6), (5, 7)]:
        (x1, _), (_, yb) = edge(a, "b"), edge(b, "b")
        (x2, _) = edge(b, "b")
        ax.annotate("", xy=(x2, yb - 0.008), xytext=(x1, yb - 0.008),
                    arrowprops=dict(arrowstyle="<->", color=C["orange"], lw=1.4,
                                    connectionstyle="arc3,rad=0.25"))
        ax.text((x1 + x2) / 2, yb - 0.085, "DP", ha="center", fontsize=7.5, color=C["orange"])
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        (x1, y1), (x2, y2) = edge(a, "b"), edge(b, "t")
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="<->", color=C["green"], lw=1.4))
        ax.text(x1 + 0.012, (y1 + y2) / 2, "PP", ha="left", va="center", fontsize=7.5, color=C["green"])

    ax.text(0.5, -0.11, "TP = 相邻两个 rank（同节点）　DP = 同一 stage 同一 TP 位的两张卡　PP = 同一 DP 副本同一 TP 位的两个 stage",
            ha="center", fontsize=8, color="#444444")
    ax.text(0.5, -0.19, "TP 组 [0,1] [2,3] [4,5] [6,7]　·　DP 组 [0,2] [1,3] [4,6] [5,7]　·　PP 组 [0,4] [1,5] [2,6] [3,7]",
            ha="center", fontsize=8, color="#444444")
    ax.set_title("train-ledger 配置：8 卡 TP2 × PP2 × DP2（Llama-3.2-1B 形状，16 层）")
    save(fig, "megatron-f1-8-gpus")


# ---------------------------------------------------------------- F2
def f2():
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.06); ax.axis("off")

    # 五层堆叠（自下而上），每层：框 + 一句话
    layers = [
        ("parallel_state（模块级全局变量 + RankGenerator）", "给下面每一层提供进程组：TP / DP / PP / …", C["gray"]),
        ("tensor_parallel：ColumnParallel / RowParallel / mappings", "模型由切开的层搭成；通信是手写的 autograd.Function", C["blue"]),
        ("_ParamAndGradBuffer + Bucket(Group)", "两段连续内存；参数与 main_grad 是视图；通信单位是 bucket", C["orange"]),
        ("DistributedDataParallel", "包装模型，注册 hook：梯度就绪发 reduce-scatter、前向等 all-gather", C["green"]),
        ("DistributedOptimizer", "按 buffer 字节区间切 fp32 主参数与 m、v（ZeRO-1）", C["purple"]),
        ("schedules.py + P2PCommunicator", "驱动前向 / 反向时序；只通过 config 回调碰 DDP", C["red"]),
    ]
    y = 0.02
    for name, desc, col in layers:
        box(ax, 0.05, y, 0.9, 0.125, "", fc="white", ec=col, lw=1.3)
        ax.text(0.075, y + 0.086, name, ha="left", va="center", fontsize=9, weight="bold", color=col)
        ax.text(0.075, y + 0.036, desc, ha="left", va="center", fontsize=8)
        y += 0.165
    ax.annotate("", xy=(0.98, 0.92), xytext=(0.98, 0.08),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
    ax.text(0.995, 0.5, "上层依赖下层", rotation=90, va="center", fontsize=8, color="#888888")
    ax.set_title("Megatron Core 的六个核心抽象：从下到上层层依赖")
    save(fig, "megatron-f2-abstractions")


# ---------------------------------------------------------------- F3
def f3():
    """一个 step（DP rank 视角）的泳道：上 = 计算流，中 = 通信流，下 = 常驻。"""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 3.2); ax.axis("off")

    lanes = [(2.35, "计算流", "#f4f7fb"), (1.30, "通信流", "#fdf7ef"), (0.30, "常驻", "#f5f5f5")]
    for y0, name, fc in lanes:
        ax.add_patch(mpatches.Rectangle((0, y0), 16, 0.9, fc=fc, ec="#cccccc", lw=0.6))
        ax.text(-0.15, y0 + 0.45, name, ha="right", va="center", fontsize=9, weight="bold")

    def cell(x, w, y0, text, fc, fs=7.5):
        box(ax, x, y0, w, 0.62, text, fc=fc, fs=fs, r=0.03)

    # 计算流：m 个 micro-batch 前向/反向 → step
    cell(0.25, 9.2, 2.45, "8 个 micro-batch 的前向 + 反向\n（dW 直接累加进 main_grad，不通信）", C["blue"] + "22")
    cell(9.7, 2.6, 2.45, "finalize\nmodel_grads", "#eeeeee")
    cell(12.5, 3.2, 2.45, "optimizer.step\n（只动本 rank 1/2 段）", C["purple"] + "22")
    # 通信流
    cell(7.6, 3.6, 1.40, "reduce-scatter grad_data\n（与末尾反向重叠）", C["orange"] + "33")
    cell(12.5, 3.2, 1.40, "all-gather param_data\n（发出，不等）", C["green"] + "33")
    cell(0.25, 3.4, 1.40, "all-gather（上一 step 发出）\n在前向 pre-hook 里等", C["green"] + "33")
    # 常驻
    cell(0.25, 15.4, 0.40, "bf16 参数（完整）· fp32 梯度 buffer（完整）· fp32 主参数 + m + v（1/2）——整个训练期不释放",
         "#ffffff")
    ax.annotate("", xy=(3.65, 2.30), xytext=(3.65, 1.80), arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.annotate("", xy=(11.2, 2.30), xytext=(11.2, 1.80), arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.annotate("", xy=(14.1, 2.30), xytext=(14.1, 1.80), arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.set_title("一个 DP rank 的一个 step：两次 DP 通信都与计算重叠（ZeRO-1 的 2N）")
    save(fig, "megatron-f3-step-swimlane")


# ---------------------------------------------------------------- F4
def f4():
    from importlib import import_module
    sched = import_module("03_megatron_toys") if False else None
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    toys = importlib.import_module("03_megatron_toys")
    seqs = toys.schedule_1f1b(2, 8)

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    # 时间轴 = 事件序号（stage0 与 stage1 各自按事件顺序排成同一长度 16 格）
    n = max(len(v) for v in seqs.values())
    ax.set_xlim(-0.6, n + 0.4); ax.set_ylim(-0.5, 2.6); ax.axis("off")
    labels = {0: "stage 0\n(layers 0–7)", 1: "stage 1\n(layers 8–15)"}
    phase_c = {"warmup": C["blue"] + "22", "steady": C["green"] + "22", "cooldown": C["orange"] + "33"}
    for stage, ev in seqs.items():
        for t, (ph, e) in enumerate(ev):
            y = 1 - stage
            box(ax, t + 0.05, y, 0.9, 0.62, e, fc=phase_c[ph], fs=8)
        ax.text(-0.45, y + 0.31, labels[stage], ha="right", va="center", fontsize=8.5)
    # 三段括号（按 stage1 的时间轴标；stage0 相位相同）
    def bracket(x0, x1, name):
        ax.annotate("", xy=(x0, 2.28), xytext=(x1, 2.28),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=1.0))
        ax.plot([x0, x0], [2.22, 2.28], color="#555555", lw=1.0)
        ax.plot([x1, x1], [2.22, 2.28], color="#555555", lw=1.0)
        ax.text((x0 + x1) / 2, 2.38, name, ha="center", fontsize=8.5)
    bracket(0.05, 0.95, "warmup")
    bracket(0.95, 14.95, "steady（1F1B 交替）")
    bracket(14.95, 15.95, "cooldown")
    ax.text(n / 2, -0.42, "三段分别对应函数里的三个 for：range(num_warmup) · range(num_microbatches_remaining) · range(num_warmup)",
            ha="center", fontsize=8, color="#444444")
    ax.set_title("1F1B 调度：PP=2、m=8（事件序列由玩具的 schedule_1f1b 生成）")
    save(fig, "megatron-f4-1f1b-schedule")


if __name__ == "__main__":
    f1(); f2(); f3(); f4()
    print("out/megatron-f*.svg done")
