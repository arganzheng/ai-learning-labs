"""训练任务的状态解剖（01）配图：正文列宽 750px，布局由 _plot.py 自检。"""
import importlib.util
import os

import numpy as np
from matplotlib.patches import Rectangle

from _plot import ARROW_LABEL_SIZE, C, box, box_title, plt, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3", "#e5e5d7", "#d8edef"]


def _arrow(ax, start, end, color="#555555", lw=1.1, style="->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=10))


def f1_budget():
    """Llama 3 8B 的状态 + 激活堆叠，对照一张 H100。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.set_title("Llama 3 8B 放进一张 80 GB H100：状态本身已越线")

    x0, scale, y, h = 0.55, 0.032, 2.55, 0.56
    parts = [
        ("bf16 参数", 16.1, PALETTE[0]),
        ("fp32 梯度", 32.1, PALETTE[1]),
        ("fp32 主参数", 32.1, PALETTE[2]),
        ("Adam m", 32.1, PALETTE[3]),
        ("Adam v", 32.1, PALETTE[4]),
        ("激活 + logits", 42.8, PALETTE[5]),
        ("非张量开销", 10.7, "#eeeeee"),
    ]
    cursor = x0
    narrow_cx = None
    for label, gb, color in parts:
        width = gb * scale
        ax.add_patch(Rectangle((cursor, y), width, h, facecolor=color, edgecolor="#ffffff", linewidth=0.8))
        if width < 0.6:
            narrow_cx = cursor + width / 2
            narrow_label = f"{label} {gb:g}"
        else:
            text(ax, cursor + width / 2, y + h / 2, f"{gb:g}", ha="center", va="center", fontsize=10)
            text(ax, cursor + width / 2, y + h + 0.14, label, ha="center", va="bottom", fontsize=10)
        cursor += width
    # 窄段：值与名字一起引出到条下
    ax.plot([narrow_cx, narrow_cx], [y, y - 0.10], color="#888888", lw=0.8)
    text(ax, narrow_cx, y - 0.14, narrow_label, ha="center", va="top", fontsize=10)
    capacity_x = x0 + 80 * scale
    ax.plot([capacity_x, capacity_x], [2.02, 3.36], color=C["red"], linewidth=1.8, linestyle="--")
    text(ax, capacity_x, 3.86, "80 GB 容量线", ha="center", va="bottom", fontsize=10, color=C["red"])
    total = sum(v for _, v, _ in parts)
    text(ax, 6.98, 2.83, f"合计\n{total:.0f} GB", ha="left", va="center", fontsize=10, color="#111111")
    text(ax, capacity_x + 0.10, 2.16, f"超出约 {total - 80:.0f} GB", ha="left", va="center", fontsize=10, color=C["red"])

    ax.plot([x0, x0 + 200 * scale], [1.86, 1.86], color="#555555", linewidth=0.8)
    for tick in range(0, 201, 50):
        tx = x0 + tick * scale
        ax.plot([tx, tx], [1.86, 1.80], color="#555555", linewidth=0.8)
        text(ax, tx, 1.68, str(tick), ha="center", va="center", fontsize=10)
    text(ax, x0 + 100 * scale, 1.38, "显存（GB）", ha="center", va="center", fontsize=10)

    text(ax, 3.8, 0.75, "$s=8192$，$b=1$；激活含 32 层与 logits。单卡装不下，下一篇看怎么切。",
         ha="center", va="center", fontsize=10.5)
    save(fig, "training-state-budget-overview")


def f2_ledgers():
    """显存字节账与训练 FLOP 账。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("训练能不能跑、跑得多快：两本账")
    left = box(ax, 0.32, 0.74, 3.30, 2.92, "", fc="#f5f8fc", ec=C["blue"])
    right = box(ax, 3.98, 0.74, 3.30, 2.92, "", fc="#fff8f0", ec=C["orange"])
    box_title(ax, 1.97, 3.34, "字节账：显存够不够？", left, color=C["blue"])
    for yy, label in [(2.78, "参数 + 梯度 + 优化器状态"), (2.31, "+ 激活 + 非张量开销"),
                      (1.78, "≤ 每卡可用显存（约 70 GiB）")]:
        text(ax, 1.97, yy, label, in_box=left, ha="center", va="center", fontsize=11)
    text(ax, 1.97, 1.15, "8B：$16N=128.5$ GB，单卡先放不下", in_box=left,
         ha="center", va="center", fontsize=10.5, color=C["red"])

    box_title(ax, 5.63, 3.34, "FLOP 账：最快要多久？", right, color=C["orange"])
    text(ax, 5.63, 2.76, "$\mathrm{tokens}/\mathrm{step} \\times \mathrm{FLOP}/\mathrm{token}$", in_box=right,
         ha="center", va="center", fontsize=11)
    text(ax, 5.63, 2.42, "$\div$（卡数 $\\times$ 峰值 FLOPS）", in_box=right,
         ha="center", va="center", fontsize=11)
    text(ax, 5.63, 1.93, "$=$ step 时间下限 1.86 s", in_box=right,
         ha="center", va="center", fontsize=11)
    text(ax, 5.63, 1.36, "70B · 4M token · 1024 H100", in_box=right,
         ha="center", va="center", fontsize=10)
    text(ax, 5.63, 0.98, "实测 4.5 s $\\to$ MFU $=41.3\%$", in_box=right,
         ha="center", va="center", fontsize=10.5, color=C["orange"])
    text(ax, 3.8, 0.28, "左边约束放得下，右边给出理论下限；实测与下限的差才是调优空间。",
         ha="center", va="center", fontsize=10)
    save(fig, "training-state-two-ledgers")


def f3_four_states():
    """保留原 SVG 的四种状态与 step 顺序。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.75))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.75); ax.axis("off")
    ax.set_title("四种状态：哪些常驻，哪些在 step 里来去")
    text(ax, 1.95, 4.18, "常驻：大小只与 $N$ 有关", ha="center", va="center", fontsize=11, color=C["blue"])
    text(ax, 5.55, 4.18, "瞬态：一个 step 内产生又释放", ha="center", va="center", fontsize=11, color=C["orange"])
    cards = [
        (0.20, 2.55, "参数 $\\theta$", "bf16 计算副本\n$2N$ 字节\n前向、反向都读", C["blue"]),
        (2.00, 2.55, "优化器状态", "$12N$：主参数 $4N$\nAdam $m$ $4N$ + $v$ $4N$", C["purple"]),
        (3.80, 2.55, "梯度", "$2N$ bf16 / $4N$ fp32\n反向逐层产生\nstep 后清零", C["orange"]),
        (5.60, 2.55, "激活 $A$", "$\\approx 34\\,s\\,b\\,h\\,L$\n字节\n前向堆高，反向释放", C["green"]),
    ]
    for x, y, title, detail, color in cards:
        p = box(ax, x, y, 1.70, 1.34, "", fc="#ffffff", ec=color, lw=1.0)
        box_title(ax, x + 0.85, y + 1.02, title, p, color=color)
        text(ax, x + 0.85, y + 0.46, detail, in_box=p, ha="center", va="center", fontsize=10.5)
    for xlo, xhi in [(0.20, 3.70), (3.80, 7.30)]:
        yb = 2.50
        ax.plot([xlo, xhi], [yb, yb], color="#555555", linewidth=1.0)
        ax.plot([xlo, xlo], [yb, yb + 0.08], color="#555555", linewidth=1.0)
        ax.plot([xhi, xhi], [yb, yb + 0.08], color="#555555", linewidth=1.0)
    steps = [
        (0.42, "① 前向", "读参数\n逐层留下激活\n末端算 loss"),
        (2.17, "② 反向", "从 loss 往回算\n读参数 + 激活\n产梯度、释放激活"),
        (3.92, "③ 更新", "读梯度\n更新 m、v、主参数"),
        (5.67, "④ cast 回 bf16", "主参数\n$\\to$ 计算参数"),
    ]
    for x, title, detail in steps:
        p = box(ax, x, 0.78, 1.48, 1.22, "", fc="#f7f8fa", ec="#777777")
        box_title(ax, x + 0.74, 1.74, title, p)
        text(ax, x + 0.74, 1.22, detail, in_box=p, ha="center", va="center", fontsize=10.5)
    for x in [1.90, 3.65, 5.40]:
        _arrow(ax, (x + 0.02, 1.40), (x + 0.25, 1.40), lw=1.4)
    text(ax, 3.8, 0.38, "70B、$s=8192$、$b=1$：参数 141 GB，优化器 847 GB，梯度 141 / 282 GB，激活 182 GB。",
         ha="center", va="center", fontsize=10.5)
    save(fig, "training-state-four-states-overview")


def f4_lifecycle():
    """保留原 SVG 的四层前向/反向生命周期与逐层文字。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.25))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.25); ax.axis("off")
    ax.set_title("激活先堆高，梯度后到齐：生命周期正好错开")
    xs = [2.05 + i * 0.55 for i in range(8)]
    time_labels = ["F1", "F2", "F3", "F4", "B4", "B3", "B2", "B1"]
    for xlo, xhi, name, color in [(xs[0] - 0.24, xs[3] + 0.24, "前向", C["blue"]),
                                  (xs[4] - 0.24, xs[7] + 0.24, "反向", C["orange"])]:
        yb = 3.92
        ax.plot([xlo, xhi], [yb, yb], color=color, linewidth=1.1)
        ax.plot([xlo, xlo], [yb, yb - 0.07], color=color, linewidth=1.1)
        ax.plot([xhi, xhi], [yb, yb - 0.07], color=color, linewidth=1.1)
        text(ax, (xlo + xhi) / 2, yb + 0.14, name, ha="center", va="center", fontsize=10, color=color)
    layer_ys = [3.05, 2.50, 1.95, 1.40]
    for layer, y in enumerate(layer_ys):
        text(ax, 0.82, y + 0.14, f"第 {layer + 1} 层", ha="center", va="center", fontsize=10)
        forward_start = layer
        backward_end = 7 - layer
        for i, x in enumerate(xs):
            if forward_start <= i < backward_end:
                ax.add_patch(Rectangle((x - 0.20, y), 0.40, 0.25,
                                       facecolor=C["blue"] + "66", edgecolor="#ffffff", linewidth=0.5))
            if i == backward_end:
                ax.add_patch(Rectangle((x - 0.20, y), 0.40, 0.25,
                                       facecolor=C["orange"] + "88", edgecolor="#ffffff", linewidth=0.5))
            elif i > backward_end:
                ax.add_patch(Rectangle((x - 0.20, y), 0.40, 0.25,
                                       facecolor="#d8d8d8", edgecolor="#ffffff", linewidth=0.5))
    for x, name in zip(xs, time_labels):
        text(ax, x, 1.05, name, ha="center", va="center", fontsize=10)
    legend = [
        (C["blue"] + "66", "激活还在"),
        (C["orange"] + "88", "该层正在反向"),
        ("#d8d8d8", "激活已释放、梯度已产生"),
    ]
    for i, (color, label) in enumerate(legend):
        lx = 0.85 + i * 2.45
        ax.add_patch(Rectangle((lx, 0.62), 0.16, 0.16, facecolor=color, edgecolor="#888888", linewidth=0.4))
        text(ax, lx + 0.24, 0.70, label, ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.34, "前向末：4 层激活都在，梯度为 0。反向末：激活已释放完，梯度全部到齐。",
         ha="center", va="center", fontsize=10)
    text(ax, 3.8, 0.10, "激活：第 i 层在 Fi 产生，Bi 用完释放；先产生的后释放，第 1 层留得最久。",
         ha="center", va="center", fontsize=10)
    save(fig, "training-state-activation-grad-lifecycle")


def f5_timeline():
    """Toy hooks 给出曲线形状，按 8B 参数与激活尺寸线性换算纵轴。"""
    spec = importlib.util.spec_from_file_location("state_toys", os.path.join(HERE, "01_state_toys.py"))
    toys = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toys)
    events, toy_n, _ = toys.timeline_curve()
    act = np.array([e[2] for e in events], dtype=float)
    grad = np.array([e[3] for e in events], dtype=float)
    # 8B 表格：参数 16.1GB、梯度 16.1GB（bf16）、优化器状态 96.4GB、激活 36.5GB 峰值。
    params_gb, grads_gb, optim_gb, act_peak_gb = 16.1, 16.1, 96.4, 36.5
    act_scale = act_peak_gb / max(act.max(), 1)
    grad_scale = grads_gb / max(grad.max(), 1)
    phases = [e[1] for e in events]
    fwd_end = max(i for i, ph in enumerate(phases) if ph == "前向")
    x = np.linspace(0, 1, len(events))
    y0 = params_gb
    y1 = y0 + optim_gb
    y2 = y1 + act * act_scale
    y3 = y2 + grad * grad_scale

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 200)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("显存（GB，十进制）")
    ax.set_xticks([])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.fill_between(x, 0, y0, color=PALETTE[0], label="bf16 参数 2N")
    ax.fill_between(x, y0, y1, color=PALETTE[3], label="fp32 主参数 + Adam m/v 12N")
    ax.fill_between(x, y1, y2, color=PALETTE[5], label="激活（toy hook 曲线）")
    ax.fill_between(x, y2, y3, color="#8fc9a8", label="bf16 梯度 2N")
    ax.plot(x, y3, color="#444444", linewidth=1.1)
    ax.set_title("一个 step 的显存：激活先到峰，梯度后到峰")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=10)
    xb = x[fwd_end]
    for xlo, xhi, name in [(0.0, xb, "前向"), (xb, 1.0, "反向")]:
        yline = -9
        ax.plot([xlo, xhi], [yline, yline], color="#555555", linewidth=0.8, clip_on=False)
        ax.plot([xlo, xlo], [yline, yline + 2], color="#555555", linewidth=0.8, clip_on=False)
        ax.plot([xhi, xhi], [yline, yline + 2], color="#555555", linewidth=0.8, clip_on=False)
        text(ax, (xlo + xhi) / 2, -22, name, ha="center", va="center", fontsize=10)
    i_act = int(np.argmax(y2))
    i_grad = int(np.argmax(grad))
    text(ax, x[i_act], y2[i_act] + 8, "激活峰值\n（前向末）", ha="left", va="bottom", fontsize=10)
    text(ax, x[i_grad], y3[i_grad] + 8, "梯度峰值（反向末）", ha="right", va="bottom", fontsize=10)
    ax.text(0.02, 192, "optimizer.step 后梯度清零",
            fontsize=10, color="#333333", ha="left", va="center")
    save(fig, "training-state-step-timeline")


def f6_bytes():
    """保留原 SVG 的四种精度布局，每段长度与字节数成正比。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.05))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 5.05); ax.axis("off")
    ax.set_title("每参数多少字节：dtype 决定哪些项能切")
    rows = [
        ("bf16 参数 + bf16 梯度", [("p16", 2, PALETTE[0]), ("g16", 2, PALETTE[1]), ("master", 4, PALETTE[2]), ("m", 4, PALETTE[3]), ("v", 4, PALETTE[4])], "= 16  →  4 + 12/d"),
        ("bf16 参数 + fp32 梯度", [("p16", 2, PALETTE[0]), ("g32", 4, PALETTE[1]), ("master", 4, PALETTE[2]), ("m", 4, PALETTE[3]), ("v", 4, PALETTE[4])], "= 18  →  6 + 12/d"),
        ("fp16 参数 + fp16 梯度", [("p16", 2, PALETTE[0]), ("g16", 2, PALETTE[1]), ("g32", 4, "#cfe4ea"), ("master", 4, PALETTE[2]), ("m", 4, PALETTE[3]), ("v", 4, PALETTE[4])], "= 20  →  4 + 16/d"),
        ("fp32 参数 + fp32 梯度", [("p32", 4, PALETTE[0]), ("g32", 4, PALETTE[1]), ("m", 4, PALETTE[3]), ("v", 4, PALETTE[4])], "= 16  →  8 + 8/d"),
    ]
    max_bytes = 20
    for ri, (label, segs, total) in enumerate(rows):
        y = 3.72 - ri * 0.92
        text(ax, 0.20, y + 0.56, label, ha="left", va="center", fontsize=10)
        x0, width = 0.20, 5.35
        cursor = x0
        for name, value, color in segs:
            w = width * value / max_bytes
            p = box(ax, cursor, y, w, 0.36, "", fc=color, ec="#888888", lw=0.45, r=0)
            if w >= 0.35:
                text(ax, cursor + w / 2, y + 0.18, name, in_box=p, ha="center", va="center", fontsize=10)
            cursor += w
        text(ax, 5.75, y + 0.18, total, ha="left", va="center", fontsize=10)
    text(ax, 0.55, 0.56, "p16/p32 参数 · g16/g32 梯度 · master 主参数 · m/v Adam 两矩；格宽 ∝ 字节数",
         ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.18, "Adam 的两个矩始终 8 B 且可切；变化的是梯度 dtype、主参数是否单独一份，以及 fp16 主梯度是否进分片。",
         ha="center", va="center", fontsize=10)
    save(fig, "training-state-bytes-per-parameter")


def f7_activations():
    """34sbh + 5as^2b 的逐项来源；灰色斜线表示 FlashAttention 不物化分数矩阵。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.9); ax.axis("off")
    ax.set_title("一层 Transformer 的激活：大头来自注意力分数矩阵")
    text(ax, 3.8, 4.56, "公式：$sbh(34 + 5as/h) = 34sbh + 5as^2b$", ha="center", va="center", fontsize=11)
    # 70B, s=8192,b=1: 34sbh=2.28 GB, 5as^2b=21.5 GB.
    x0, y, w, h = 0.55, 3.55, 6.45, 0.62
    base, score = 2.28, 21.5
    total = base + score
    base_w = w * base / total
    # 34sbh = 注意力 11 + MLP 19 + LayerNorm 4（单位 sbh），与下面三行一一对应
    subs = [(11, "注意力 11", C["blue"] + "55"), (19, "MLP 19", C["green"] + "55"), (4, "LN 4", C["orange"] + "55")]
    cursor = x0
    seg_labels = []
    for units, name, color in subs:
        seg_w = base_w * units / 34
        ax.add_patch(Rectangle((cursor, y), seg_w, h, facecolor=color, edgecolor="#ffffff", linewidth=1.0))
        seg_labels.append((cursor + seg_w / 2, name))
        cursor += seg_w
    text(ax, x0 + base_w / 2, y + h + 0.06, "$34sbh = 2.28$ GB", ha="center", va="bottom", fontsize=10.5)
    ax.add_patch(Rectangle((x0 + base_w, y), w - base_w, h, facecolor="#eeeeee", edgecolor="#888888", linewidth=1.0, hatch="///"))
    text(ax, x0 + base_w + (w - base_w) / 2, y + h / 2, "$5as^2b = 21.5$ GB（不使用 FlashAttention）", ha="center", va="center", fontsize=10.5)
    # 窄段标签引出到条下，交错两行；MLP 引线取段右端，绕开左边「注意力 11」的宽度
    for (cx, name), (fx, lx, ly) in zip(seg_labels, [(0.65, 0.55, 3.32), (1.06, 0.92, 3.10), (1.10, 1.30, 3.32)]):
        ax.plot([fx, fx], [y, ly + 0.16], color="#888888", lw=0.8)
        text(ax, lx, ly, name, ha="center", va="center", fontsize=10)

    rows = [
        ("注意力", "Q/K/V 投影输入 $2h$ · $QK^{\\!\\!\\top}$ 输入 $4h$ · $V$ / 输出投影输入 $4h$ · dropout mask $h$", C["blue"]),
        ("MLP", "第一线性输入 $2h$ · GELU 输入 $8h$ · 第二线性输入 $8h$ · dropout mask $h$", C["green"]),
        ("LayerNorm", "两份输入 $4h$", C["orange"]),
        ("注意力分数", "softmax 输出 $2as$ · dropout mask $as$ · dropout 输出 $2as$", C["purple"]),
    ]
    for i, (name, detail, color) in enumerate(rows):
        yy = 2.32 - i * 0.47
        p = box(ax, 0.55, yy, 1.26, 0.34, name, fc="#ffffff", ec=color, fs=10)
        text(ax, 2.02, yy + 0.17, detail, ha="left", va="center", fontsize=10.5)
    text(ax, 3.8, 0.42, "Llama 3 70B，$s=8192$、$b=1$；FlashAttention 不保存灰色斜线部分，反向时分块重算。",
         ha="center", va="center", fontsize=10.5)
    save(fig, "training-state-activation-breakdown")


def f8_memory_nesting():
    """重画 allocator / device 的嵌套关系，保留原数值与术语。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.9); ax.axis("off")
    ax.set_title("nvidia-smi 看到的占用，比张量总和多在哪？")
    outer = box(ax, 0.28, 0.43, 7.02, 4.0, "", fc="#f5f5f5", ec="#555555", lw=1.1)
    box_title(ax, 3.79, 4.15, "驱动看到的本进程占用（约 79.6 GiB 上限）", outer)
    text(ax, 3.79, 3.82, "CUDA context + 全部库 kernel：0.5–1.0 GiB  ·  NCCL communicators：1–3 GiB",
         in_box=outer, ha="center", va="center", fontsize=10)
    reserved = box(ax, 0.55, 0.78, 6.45, 2.75, "", fc="#ffffff", ec=C["blue"], lw=1.0)
    box_title(ax, 3.78, 3.25, "PyTorch allocator reserved（分配器向驱动申请的 segment）", reserved, color=C["blue"])
    alloc = box(ax, 0.82, 1.56, 4.05, 1.26, "", fc="#eff5fc", ec=C["green"], lw=0.9)
    box_title(ax, 2.84, 2.55, "allocated：活着的张量 block", alloc, color=C["green"])
    text(ax, 2.84, 2.11, "参数 $2N$ · 梯度 $2N/4N$ · 优化器状态 $12N$", in_box=alloc,
         ha="center", va="center", fontsize=10.5)
    text(ax, 2.84, 1.76, "激活 $\\approx 34\,s\,b\,h\,L$ + logits $6\,s\,b\,V$", in_box=alloc,
         ha="center", va="center", fontsize=10.5)
    text(ax, 5.82, 2.57, "库 workspace\n0.2–0.5 GiB", in_box=reserved,
         ha="center", va="center", fontsize=10)
    text(ax, 5.82, 1.88, "缓存空闲 block\n上个 step 的激活留池复用", in_box=reserved,
         ha="center", va="center", fontsize=10)
    text(ax, 5.82, 1.30, "碎片：inactive_split_bytes\n空闲但与相邻 block 合不回\n通常 reserved 的 5–15%", in_box=reserved,
         ha="center", va="center", fontsize=10)
    text(ax, 3.78, 0.60, "另留 OOM 安全边际 2–4 GiB；context 与 NCCL 由 cudaMalloc 直接分配，不在 reserved 里。",
         in_box=outer, ha="center", va="center", fontsize=10)
    save(fig, "training-state-memory-nesting")


def f9_flops():
    """Linear 的 6 FLOP/参数/token 与注意力 s^2 项。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("6N 从哪来：前向一个 GEMM，反向两个")
    cols = [
        (0.55, "前向", "$y = xW$", ["$x$ ($\\mathrm{tokens}, n$)", "$W$ ($n, k$)"], C["blue"]),
        (2.95, "dgrad（求 dX）", "$dX = dY\,W^{\\!\\!\\top}$", ["$dY$ ($\\mathrm{tokens}, k$)", "$W$ ($n, k$)"], C["blue"]),
        (5.35, "wgrad（求 dW）", "$dW = x^{\\!\\!\\top}\,dY$", ["$x$ ($\\mathrm{tokens}, n$)", "$dY$ ($\\mathrm{tokens}, k$)"], C["orange"]),
    ]
    for cx, title, formula, inputs, color in cols:
        gemm = box(ax, cx - 0.04, 1.55, 1.78, 1.30, "", fc="#ffffff", ec=color, lw=1.1)
        box_title(ax, cx + 0.85, 2.55, title, gemm, color=color)
        text(ax, cx + 0.85, 2.14, formula, in_box=gemm, ha="center", va="center", fontsize=10.5)
        text(ax, cx + 0.85, 1.78, "2 FLOP / 参数 / token", in_box=gemm, ha="center", va="center", fontsize=10)
        for j, inp in enumerate(inputs):
            chip = box(ax, cx - 0.03 + j * 1.15, 3.10, 1.12, 0.34, "", fc="#f4f6f8", ec="#888888", lw=0.6)
            text(ax, cx + 0.555 + j * 1.13, 3.27, inp, in_box=chip, ha="center", va="center", fontsize=10.5)
            _arrow(ax, (cx + 0.555 + j * 1.13, 3.08), (cx + 0.555 + j * 1.13, 2.88), lw=0.9)
    text(ax, 3.8, 0.75, "前向 2 + 反向 dgrad 2 + wgrad 2 = 6 FLOP / 参数 / token", ha="center", va="center", fontsize=11)
    text(ax, 3.8, 0.34, "注意力另有 $QK^{\\!\\!\\top}$ 与 $PV$：每层每 token 全 mask $12sh$，因果 mask 减半为 $6sh$。",
         ha="center", va="center", fontsize=10.5)
    save(fig, "training-state-flops-per-parameter")


def f10_mfu_hfu():
    """相同 step 下 MFU / HFU 分子和重计算之间的差异。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.0); ax.axis("off")
    ax.set_title("MFU 看模型做了多少有用计算，HFU 还数上重算")
    # 分子长度 ∝ MFU/HFU 百分数：41.3 / 42.3 / 55.1（同一 step 时间与卡数）
    rows = [
        ("无重计算", 41.3, 0.0, "MFU 41.3%"),
        ("选择性重计算", 41.3, 42.3 - 41.3, "HFU 42.3%"),
        ("全量重计算", 41.3, 55.1 - 41.3, "HFU 55.1%"),
    ]
    x0, scale = 0.55, 0.075
    for i, (name, base_u, re_u, pct) in enumerate(rows):
        y = 3.05 - i * 0.78
        text(ax, x0, y + 0.18, name, ha="left", va="center", fontsize=10)
        bw = base_u * scale
        rw = re_u * scale
        ax.add_patch(Rectangle((2.42, y), bw, 0.36, facecolor=C["blue"] + "55", edgecolor=C["blue"], linewidth=0.7))
        if rw > 0:
            ax.add_patch(Rectangle((2.42 + bw, y), rw, 0.36, facecolor=C["red"] + "55", edgecolor=C["red"], linewidth=0.7))
        text(ax, 2.55 + bw + rw, y + 0.18, pct, ha="left", va="center", fontsize=10)
    lx = 2.42
    ax.add_patch(Rectangle((lx, 1.06), 0.16, 0.16, facecolor=C["blue"] + "55", edgecolor=C["blue"], linewidth=0.7))
    text(ax, lx + 0.24, 1.14, "模型需要的计算", ha="left", va="center", fontsize=10)
    ax.add_patch(Rectangle((lx + 1.85, 1.06), 0.16, 0.16, facecolor=C["red"] + "55", edgecolor=C["red"], linewidth=0.7))
    text(ax, lx + 2.09, 1.14, "重计算", ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.62, "70B · 1024 H100 · 4M token/step · 实测 4.5 s", ha="center", va="center", fontsize=10)
    text(ax, 3.8, 0.34, "MFU 41.3%  ·  HFU（选择性）42.3%  ·  HFU（全量）55.1%", ha="center", va="center", fontsize=11)
    text(ax, 3.8, 0.08, "重算让 GPU 更忙，不代表每秒处理的 token 更多。", ha="center", va="center", fontsize=10)
    save(fig, "training-state-mfu-vs-hfu")




def f11_activation_who_uses():
    """反向要用前向的哪些量：前向流 / 保存 / 链式梯度 / 参数梯度 四层。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.0); ax.axis("off")
    ax.set_title("反向要用前向的哪些量：保存的只有三个，参数单独算",
                 fontweight=600, color="#111111")

    rects = []          # (x0, y0, x1, y1)，用来断言虚线不穿盒

    def chip(xc, y, w, h, s, fc, ec="#555555", fs=10, dash=False):
        rects.append((xc - w / 2, y, xc + w / 2, y + h))
        return box(ax, xc - w / 2, y, w, h, s, fc=fc, ec=ec, lw=0.9, fs=fs,
                   linestyle="--" if dash else "-")

    cols = [1.95, 4.05, 6.15, 8.25, 10.35]

    def dashed_poly(pts, color):
        """折线虚线箭头：前面是直线段，最后一段带箭头。"""
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            ax.plot([x1, x2], [y1, y2], color=color, lw=1.0,
                    linestyle=(0, (4, 3)), zorder=2)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0,
                                    linestyle=(0, (4, 3)), mutation_scale=9))

    def seg_free(pts, skip=()):
        """断言折线不穿任何盒（源盒、目标盒除外）。"""
        src, dst = pts[0], pts[-1]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            for t in np.linspace(0.08, 0.92, 40):
                px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                for (rx0, ry0, rx1, ry1) in rects:
                    if (rx0 - 0.02 < px < rx1 + 0.02
                            and ry0 - 0.02 < py < ry1 + 0.02):
                        inside_src = (abs(px - src[0]) < 0.3
                                      and abs(py - src[1]) < 0.3)
                        inside_dst = (abs(px - dst[0]) < 0.3
                                      and abs(py - dst[1]) < 0.3)
                        assert inside_src or inside_dst, (
                            f"虚线穿过盒：段 {pts} 在 ({px:.2f},{py:.2f})"
                            f" 撞上 ({rx0:.2f},{ry0:.2f})-({rx1:.2f},{ry1:.2f})")

    # ---- 前向行 ----
    yf, fw, fh = 7.9, 1.35, 0.8
    for xc, name in zip(cols, ["$x$", "$H$", "$A$", "$Z$", "loss"]):
        chip(xc, yf, fw, fh, name, "#dce9f7", C["blue"], 10.5)
    ops = ["Linear $W_1$", "ReLU", "Linear $W_2$", "CE"]
    for i, op in enumerate(ops):
        x1 = cols[i] + fw / 2; x2 = cols[i + 1] - fw / 2
        _arrow(ax, (x1 + 0.04, yf + fh / 2), (x2 - 0.04, yf + fh / 2), lw=1.2)
        text(ax, (x1 + x2) / 2, yf + fh + 0.22, op, ha="center", va="center",
             fontsize=10.5)
    text(ax, 0.12, yf + fh / 2, "前向", ha="left", va="center",
         fontsize=10.5, color=C["blue"])

    # ---- 保存行 ----
    ys = 6.0
    chip(cols[0], ys, 1.35, 0.66, "存 $x$", "#fae6d6", C["orange"], 10.5)
    chip(cols[1], ys, 2.55, 0.66, "存 $H$ 的符号", "#fae6d6", C["orange"], 10.5)
    chip(cols[2], ys, 1.35, 0.66, "存 $A$", "#fae6d6", C["orange"], 10.5)
    text(ax, 0.12, ys + 0.33, "保存", ha="left", va="center",
         fontsize=10.5, color=C["orange"])

    # W2：常驻参数，放在 col3 的空列，箭头到 G_A 不穿任何盒
    chip(8.6, 4.95, 2.5, 0.62, "$W_2$（参数，常驻）", "#eeeeee", "#888888", 10.5,
         dash=True)
    text(ax, 10.7, 4.95 + 0.31, "灰色虚线框 = 常驻参数",
         ha="left", va="center", fontsize=10, color="#555555")

    # ---- 链式梯度行（右→左，每个梯度排在它求导的张量正下方）----
    yc, cw, ch = 3.6, 1.7, 1.16
    for xc, label in [(cols[1], "$G_H$\n$G_A \\odot \\mathrm{mask}$"),
                      (cols[2], "$G_A$\n$G_Z \\cdot W_2^{\\!\\!\\top}$"),
                      (cols[3], "$G_Z$\n$(P-Y)/m$")]:
        chip(xc, yc, cw, ch, label, "#fff3e0", C["orange"], 10.5)
    _arrow(ax, (cols[3] - cw / 2 - 0.04, yc + ch / 2),
           (cols[2] + cw / 2 + 0.04, yc + ch / 2), lw=1.2)
    _arrow(ax, (cols[2] - cw / 2 - 0.04, yc + ch / 2),
           (cols[1] + cw / 2 + 0.04, yc + ch / 2), lw=1.2)
    text(ax, 0.12, yc + ch / 2, "反向", ha="left", va="center",
         fontsize=10.5, color=C["orange"])

    # ---- 参数梯度行：从链上分叉 ----
    yp, pw, ph = 1.8, 2.0, 1.10
    chip(cols[3], yp, pw, ph, "$dW_2$\n$A^{\\!\\!\\top} G_Z$", "#fff3e0", C["orange"], 10.5)
    chip(cols[1], yp, pw, ph, "$dW_1$\n$x^{\\!\\!\\top} G_H$", "#fff3e0", C["orange"], 10.5)
    _arrow(ax, (cols[3], yc - 0.04), (cols[3], yp + ph + 0.04), lw=1.2)
    _arrow(ax, (cols[1], yc - 0.04), (cols[1], yp + ph + 0.04), lw=1.2)
    text(ax, 0.12, yp + ph / 2, "参数\n梯度", ha="left", va="center",
         fontsize=10, color=C["orange"])

    # ---- 保存 → 消耗（橙色虚线，走空当）----
    # 存 x → dW1：先直下，再右拐进 dW1 左缘
    dashed_poly([(cols[0], ys), (cols[0], yp + ph / 2),
                 (cols[1] - pw / 2 - 0.03, yp + ph / 2)], C["orange"])
    # 存 H 的符号 → G_H：正下方直线
    dashed_poly([(cols[1], ys), (cols[1], yc + ch + 0.03)], C["orange"])
    # 存 A → dW2：右拐出 存A，穿过 G_A/G_Z 间的缝，直下进 dW2 顶缘
    dashed_poly([(cols[2] + 1.35 / 2, ys + 0.33), (7.22, ys + 0.33),
                 (7.22, yp + ph + 0.03)], C["orange"])
    # W2 → G_A（灰色虚线）
    dashed_poly([(7.35, 4.95 + 0.31), (cols[2] + cw / 2 - 0.05, yc + ch - 0.05)],
                "#888888")

    for pts in [[(cols[0], ys), (cols[0], yp + ph / 2),
                 (cols[1] - pw / 2 - 0.03, yp + ph / 2)],
                [(cols[1], ys), (cols[1], yc + ch + 0.03)],
                [(cols[2] + 1.35 / 2, ys + 0.33), (7.22, ys + 0.33),
                 (7.22, yp + ph + 0.03)],
                [(7.35, 4.95 + 0.31),
                 (cols[2] + cw / 2 - 0.05, yc + ch - 0.05)]]:
        seg_free(pts)

    text(ax, 6.0, 0.95,
         "反向走到哪一步，就用掉并释放对应的保存量："
         "$G_H$ 用掉 mask，$dW_2$ 用掉 $A$，$dW_1$ 用掉 $x$",
         ha="center", va="center", fontsize=10.5)
    text(ax, 6.0, 0.52,
         "$dX = G_H \\cdot W_1^{\\!\\!\\top}$ 在数学上存在，但 $x$ 是数据不求导，框架跳过"
         "（W1 被那一步读，图中未画）",
         ha="center", va="center", fontsize=10.5, color="#555555")
    save(fig, "training-state-activation-who-uses")


def f12_activation_three_meanings():
    """「激活」的三个意思；第三幅用 01 toy 的真实逐层字节画比例条。"""
    import re
    exp = os.path.join(HERE, "expected", "01_state_toys.txt")
    rows = {}
    for line in open(exp):
        m = re.match(r"\s+(\S+)\s+保存 \d+ 个\s+激活\s+([\d.]+) KB", line)
        if m:
            rows[m.group(1)] = float(m.group(2))
    total = sum(rows.values())
    groups = [
        ("$a=f(z)$：狭义激活值", ["fc2"], C["orange"] + "88"),
        ("$z$：$f$ 的输入", ["gelu"], "#f0d98a"),
        ("线性层输入", ["qkv", "out_proj", "fc1"], C["blue"] + "66"),
        ("注意力中间量", ["qk^t", "pv"], C["purple"] + "55"),
        ("softmax 输出", ["softmax"], C["green"] + "55"),
        ("dropout mask", ["attn_dropout", "out_dropout", "mlp_dropout"],
         "#d0d0d0"),
        ("LN 输入与统计量", ["ln1", "ln2"], C["red"] + "44"),
    ]
    narrow = rows["fc2"] / total * 100

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.3); ax.axis("off")
    ax.set_title("「激活」的三个意思：函数、函数值、训练显存里的保存量",
                 fontweight=600, color="#111111")

    # ① 激活函数：一个函数
    p1 = box(ax, 0.3, 6.75, 3.6, 1.85, "", fc="#f5f8fc", ec=C["blue"], lw=1.0)
    box_title(ax, 2.1, 8.25, "① 函数 $f$", p1, color=C["blue"])
    xs = np.linspace(-1, 1, 60)
    ax.plot(1.35 + xs * 0.65, 7.42 + np.maximum(xs, 0) * 0.45, color=C["blue"],
            lw=1.6)
    text(ax, 2.1, 7.05, "一个函数", in_box=p1, ha="center",
         va="center", fontsize=10)

    # ② 激活值 a = f(z)
    p2 = box(ax, 4.2, 6.75, 3.6, 1.85, "", fc="#f7f6ec", ec="#a08c1f", lw=1.0)
    box_title(ax, 6.0, 8.25, "② 值 $a=f(z)$", p2, color="#a08c1f")
    text(ax, 6.0, 7.4, "某层过 f 的输出\n也是下一层输入", in_box=p2,
         ha="center", va="center", fontsize=10)

    # ③ 训练显存里的「激活」
    p3 = box(ax, 8.1, 6.75, 3.6, 1.85, "", fc="#fff8f0", ec=C["orange"], lw=1.0)
    box_title(ax, 9.9, 8.25, "③ 显存激活", p3, color=C["orange"])
    text(ax, 9.9, 7.4, "为反向保存的\n全部张量", in_box=p3,
         ha="center", va="center", fontsize=10)

    # ③ → 比例条的连接箭头 + 标签
    _arrow(ax, (9.9, 6.68), (9.9, 5.85), lw=1.2, color=C["orange"])
    text(ax, 9.72, 6.28, "③ 拆开看：一层 Transformer 的保存清单",
         ha="right", va="center", fontsize=10, color=C["orange"])

    # 比例条：每段一个张量，宽 ∝ KB
    y0, bh = 4.75, 0.95
    x0, wmax = 0.45, 11.1
    cursor = x0
    seg_label = []
    for gname, keys, color in groups:
        for k in keys:
            kw_ = rows[k]
            w = wmax * kw_ / total
            ax.add_patch(Rectangle((cursor, y0), w, bh, facecolor=color,
                                   edgecolor="#ffffff", linewidth=0.8))
            seg_label.append((cursor + w / 2, k, kw_, w, gname))
            cursor += w
    # 段内标签：宽 ≥1.3 的画原名
    seg_disp = {"qk^t": "$qk^{\\!\\!\\top}$"}
    for xc, k, kb, w, gname in seg_label:
        if w >= 1.3:
            text(ax, xc, y0 + bh / 2, seg_disp.get(k, k), ha="center", va="center", fontsize=10)
    # 颜色分组：每格一个小色块 + 组名 + KB，左右两列避免长名互撞
    ginfo = []
    for gname, keys, color in groups:
        kb = sum(rows[k] for k in keys)
        ginfo.append((gname, keys, color,
                      f"{kb/1024:.2f}MB" if kb >= 1024 else f"{kb:.0f}KB"))
    for gi, (gname, keys, color, kb) in enumerate(ginfo):
        gx = 0.5 if gi < 4 else 6.2
        gy = 4.18 - (gi if gi < 4 else gi - 4) * 0.34
        ax.add_patch(Rectangle((gx, gy), 0.22, 0.2, facecolor=color,
                               edgecolor="#888888", lw=0.5))
        text(ax, gx + 0.32, gy + 0.10, f"{gname} {kb}",
             ha="left", va="center", fontsize=10.5)
    text(ax, 6.0, 2.75,
         f"实测一层共 {total/1024:.2f} MB（$s=64$、$b=2$、$h=256$、$f=1024$，bf16）；"
         f"狭义的 $a=f(z)$ 只占 {narrow:.0f}%",
         ha="center", va="center", fontsize=10.5)
    text(ax, 6.0, 2.32, "本系列说的「激活」是第 ③ 种：把所有段一起算",
         ha="center", va="center", fontsize=10)
    text(ax, 6.0, 1.92, "「激活 $=a=f(z)$」只是第 ② 种狭义口径，会漏掉 8 成",
         ha="center", va="center", fontsize=10.5)
    save(fig, "training-state-activation-three-meanings")


if __name__ == "__main__":
    f1_budget(); f2_ledgers(); f3_four_states(); f4_lifecycle(); f5_timeline()
    f11_activation_who_uses(); f12_activation_three_meanings()
    f6_bytes(); f7_activations(); f8_memory_nesting(); f9_flops(); f10_mfu_hfu()
