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

    x0, scale, y, h = 0.55, 0.032, 2.65, 0.56
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
    for label, gb, color in parts:
        width = gb * scale
        ax.add_patch(Rectangle((cursor, y), width, h, facecolor=color, edgecolor="#ffffff", linewidth=0.8))
        cursor += width
    capacity_x = x0 + 80 * scale
    ax.plot([capacity_x, capacity_x], [2.18, 3.46], color=C["red"], linewidth=1.8, linestyle="--")
    text(ax, capacity_x, 3.55, "80 GB 容量线", ha="center", va="bottom", fontsize=10, color=C["red"])
    total = sum(v for _, v, _ in parts)
    text(ax, 6.82, 3.02, f"合计 {total:.0f} GB", ha="left", va="center", fontsize=8.5, color="#111111")
    text(ax, capacity_x + 0.17, 2.35, f"超出约 {total - 80:.0f} GB", ha="left", va="center", fontsize=10, color=C["red"])

    legend = [
        ("bf16 参数 2N", PALETTE[0]), ("fp32 梯度 4N", PALETTE[1]),
        ("fp32 主参数 4N", PALETTE[2]), ("Adam m 4N", PALETTE[3]),
        ("Adam v 4N", PALETTE[4]), ("激活 + logits 42.8 GB", PALETTE[5]),
        ("非张量开销约 10 GiB", "#eeeeee"),
    ]
    for i, (label, color) in enumerate(legend):
        col, row = i % 2, i // 2
        lx, ly = 0.85 + col * 3.25, 1.66 - row * 0.36
        ax.add_patch(Rectangle((lx, ly - 0.08), 0.16, 0.16, facecolor=color, edgecolor="#888888", linewidth=0.4))
        text(ax, lx + 0.25, ly, label, ha="left", va="center", fontsize=10)
    text(ax, 3.8, 0.35, "s = 8192，b = 1；激活含 32 层与 logits。单卡装不下，下一篇看怎么切。",
         ha="center", va="center", fontsize=10)
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
    text(ax, 1.97, 1.15, "8B：16N = 128.5 GB，单卡先放不下", in_box=left,
         ha="center", va="center", fontsize=10, color=C["red"])

    box_title(ax, 5.63, 3.34, "FLOP 账：最快要多久？", right, color=C["orange"])
    text(ax, 5.63, 2.76, "tokens/step × FLOP/token", in_box=right,
         ha="center", va="center", fontsize=11)
    text(ax, 5.63, 2.42, "÷（卡数 × 峰值 FLOPS）", in_box=right,
         ha="center", va="center", fontsize=11)
    text(ax, 5.63, 1.93, "= step 时间下限 1.86 s", in_box=right,
         ha="center", va="center", fontsize=11)
    text(ax, 5.63, 1.36, "70B · 4M token · 1024 H100", in_box=right,
         ha="center", va="center", fontsize=10)
    text(ax, 5.63, 0.98, "实测 4.5 s → MFU = 41.3%", in_box=right,
         ha="center", va="center", fontsize=10, color=C["orange"])
    text(ax, 3.8, 0.28, "左边约束放得下，右边给出理论下限；实测与下限的差才是调优空间。",
         ha="center", va="center", fontsize=10)
    save(fig, "training-state-two-ledgers")


def f3_four_states():
    """保留原 SVG 的四种状态与 step 顺序。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.75))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.75); ax.axis("off")
    ax.set_title("四种状态：哪些常驻，哪些在 step 里来去")
    text(ax, 1.95, 4.18, "常驻：大小只与 N 有关", ha="center", va="center", fontsize=11, color=C["blue"])
    text(ax, 5.55, 4.18, "瞬态：一个 step 内产生又释放", ha="center", va="center", fontsize=11, color=C["orange"])
    cards = [
        (0.20, 2.65, "参数 θ", "bf16 计算副本\n2N 字节\n前向、反向都读", C["blue"]),
        (2.00, 2.65, "梯度", "2N bf16 / 4N fp32\n反向逐层产生\nstep 后清零", C["orange"]),
        (3.80, 2.65, "激活 A", "≈ 34·s·b·h·L 字节\n前向堆高，反向释放", C["green"]),
        (5.60, 2.65, "优化器状态", "12N：主参数 4N\nAdam m 4N + v 4N", C["purple"]),
    ]
    for x, y, title, detail, color in cards:
        p = box(ax, x, y, 1.70, 1.18, "", fc="#ffffff", ec=color, lw=1.0)
        box_title(ax, x + 0.85, y + 0.88, title, p, color=color)
        text(ax, x + 0.85, y + 0.38, detail, in_box=p, ha="center", va="center", fontsize=8.5)
    steps = [
        (0.42, "① 前向", "读参数\n逐层留下激活"),
        (2.17, "② 反向", "读参数 + 激活\n产梯度、释放激活"),
        (3.92, "③ 更新", "读梯度\n更新 m、v、主参数"),
        (5.67, "④ cast 回 bf16", "主参数\n→ 计算参数"),
    ]
    for x, title, detail in steps:
        p = box(ax, x, 0.93, 1.48, 0.95, "", fc="#f7f8fa", ec="#777777")
        box_title(ax, x + 0.74, 1.63, title, p)
        text(ax, x + 0.74, 1.22, detail, in_box=p, ha="center", va="center", fontsize=9.5)
    for x in [1.98, 3.73, 5.48]:
        _arrow(ax, (x - 0.06, 1.40), (x + 0.08, 1.40))
    text(ax, 3.8, 0.38, "70B、s = 8192、b = 1：参数 141 GB，优化器 847 GB，梯度 141 / 282 GB，激活 182 GB。",
         ha="center", va="center", fontsize=9.5)
    save(fig, "training-state-four-states-overview")


def f4_lifecycle():
    """保留原 SVG 的四层前向/反向生命周期与逐层文字。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.25))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.25); ax.axis("off")
    ax.set_title("激活先堆高，梯度后到齐：生命周期正好错开")
    xs = [2.05 + i * 0.55 for i in range(8)]
    time_labels = ["F1", "F2", "F3", "F4", "B4", "B3", "B2", "B1"]
    text(ax, 2.88, 3.75, "前向", ha="center", va="center", fontsize=10, color=C["blue"])
    text(ax, 5.88, 3.75, "反向", ha="center", va="center", fontsize=10, color=C["orange"])
    layer_ys = [3.05, 2.50, 1.95, 1.40]
    for layer, y in enumerate(layer_ys):
        text(ax, 0.82, y + 0.14, f"第 {layer + 1} 层", ha="center", va="center", fontsize=10)
        forward_start = layer
        backward_end = 7 - layer
        for i, x in enumerate(xs):
            if forward_start <= i <= backward_end:
                ax.add_patch(Rectangle((x - 0.20, y), 0.40, 0.25,
                                       facecolor=C["blue"] + "66", edgecolor="#ffffff", linewidth=0.5))
            if i >= backward_end:
                ax.add_patch(Rectangle((x - 0.20, y), 0.40, 0.25,
                                       facecolor=C["orange"] + "88", edgecolor="#ffffff", linewidth=0.5))
    for x, name in zip(xs, time_labels):
        text(ax, x, 1.05, name, ha="center", va="center", fontsize=9)
    text(ax, 6.86, 1.05, "step", ha="center", va="center", fontsize=9)
    text(ax, 3.8, 0.60, "前向末：4 层激活都在，梯度为 0。反向末：激活已释放完，梯度全部到齐。",
         ha="center", va="center", fontsize=9.5)
    text(ax, 3.8, 0.27, "激活：第 i 层在 Fi 产生，Bi 用完释放；先产生的后释放，第 1 层留得最久。",
         ha="center", va="center", fontsize=9.5)
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
    ax.fill_between(x, y2, y3, color=PALETTE[1], label="bf16 梯度 2N")
    ax.plot(x, y3, color="#444444", linewidth=1.1)
    ax.set_title("一个 step 的显存：激活先到峰，梯度后到峰")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    xb = x[fwd_end]
    for xlo, xhi, name in [(0.0, xb, "前向"), (xb, 1.0, "反向")]:
        yline = -9
        ax.plot([xlo, xhi], [yline, yline], color="#555555", linewidth=0.8, clip_on=False)
        ax.plot([xlo, xlo], [yline, yline + 2], color="#555555", linewidth=0.8, clip_on=False)
        ax.plot([xhi, xhi], [yline, yline + 2], color="#555555", linewidth=0.8, clip_on=False)
        text(ax, (xlo + xhi) / 2, -22, name, ha="center", va="center", fontsize=10)
    i_act = int(np.argmax(y2))
    i_grad = int(np.argmax(grad))
    text(ax, x[i_act], y2[i_act] + 8, "激活峰值\n（前向末）", ha="left", va="bottom", fontsize=9)
    text(ax, x[i_grad], y3[i_grad] + 8, "梯度峰值（反向末）", ha="right", va="bottom", fontsize=9)
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
                text(ax, cursor + w / 2, y + 0.18, name, in_box=p, ha="center", va="center", fontsize=8.5)
            cursor += w
        text(ax, 5.75, y + 0.18, total, ha="left", va="center", fontsize=10)
    text(ax, 0.55, 0.56, "p16/p32 参数 · g16/g32 梯度 · master 主参数 · m/v Adam 两矩；格宽 ∝ 字节数",
         ha="left", va="center", fontsize=8.5)
    text(ax, 3.8, 0.18, "Adam 的两个矩始终 8 B 且可切；变化的是梯度 dtype、主参数是否单独一份，以及 fp16 主梯度是否进分片。",
         ha="center", va="center", fontsize=9.5)
    save(fig, "training-state-bytes-per-parameter")


def f7_activations():
    """34sbh + 5as^2b 的逐项来源；灰色斜线表示 FlashAttention 不物化分数矩阵。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 5.0); ax.axis("off")
    ax.set_title("一层 Transformer 的激活：大头来自注意力分数矩阵")
    text(ax, 3.8, 4.45, "公式：sbh(34 + 5as/h) = 34sbh + 5as^2b", ha="center", va="center", fontsize=11)
    # 70B, s=8192,b=1: 34sbh=2.28 GB, 5as^2b=21.5 GB.
    x0, y, w, h = 0.55, 3.18, 6.45, 0.62
    base, score = 2.28, 21.5
    total = base + score
    base_w = w * base / total
    score_w = w - base_w
    ax.add_patch(Rectangle((x0, y), base_w, h, facecolor=PALETTE[0], edgecolor="#ffffff", linewidth=1.0))
    ax.add_patch(Rectangle((x0 + base_w, y), score_w, h, facecolor="#eeeeee", edgecolor="#888888", linewidth=1.0, hatch="///"))
    text(ax, x0 + base_w / 2, y + h / 2, "34sbh\n2.28 GB", ha="center", va="center", fontsize=9)
    text(ax, x0 + base_w + score_w / 2, y + h / 2, "5as^2b = 21.5 GB（不使用 FlashAttention）", ha="center", va="center", fontsize=10)

    rows = [
        ("注意力", "Q/K/V 投影输入 2h · QK^T 输入 4h · V / 输出投影输入 4h · dropout mask h", C["blue"]),
        ("MLP", "第一线性输入 2h · GELU 输入 8h · 第二线性输入 8h · dropout mask h", C["green"]),
        ("LayerNorm", "两份输入 4h", C["orange"]),
        ("注意力分数", "softmax 输出 2as · dropout mask as · dropout 输出 2as", C["purple"]),
    ]
    for i, (name, detail, color) in enumerate(rows):
        yy = 2.48 - i * 0.49
        p = box(ax, 0.55, yy, 1.26, 0.34, name, fc="#ffffff", ec=color, fs=9.5)
        text(ax, 2.02, yy + 0.17, detail, ha="left", va="center", fontsize=9.5)
    text(ax, 3.8, 0.42, "Llama 3 70B，s=8192、b=1；FlashAttention 不保存灰色斜线部分，反向时分块重算。",
         ha="center", va="center", fontsize=9.5)
    save(fig, "training-state-activation-breakdown")


def f8_memory_nesting():
    """重画 allocator / device 的嵌套关系，保留原数值与术语。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.9); ax.axis("off")
    ax.set_title("nvidia-smi 看到的占用，比张量总和多在哪？")
    outer = box(ax, 0.28, 0.43, 7.02, 4.0, "", fc="#f5f5f5", ec="#555555", lw=1.1)
    box_title(ax, 3.79, 4.15, "驱动看到的本进程占用（约 79.6 GiB 上限）", outer)
    text(ax, 3.79, 3.82, "CUDA context + 全部库 kernel：0.5–1.0 GiB  ·  NCCL communicators：1–3 GiB",
         in_box=outer, ha="center", va="center", fontsize=9.5)
    reserved = box(ax, 0.55, 0.78, 6.45, 2.75, "", fc="#ffffff", ec=C["blue"], lw=1.0)
    box_title(ax, 3.78, 3.25, "PyTorch allocator reserved（分配器向驱动申请的 segment）", reserved, color=C["blue"])
    alloc = box(ax, 0.82, 1.56, 4.05, 1.26, "", fc="#eff5fc", ec=C["green"], lw=0.9)
    box_title(ax, 2.84, 2.55, "allocated：活着的张量 block", alloc, color=C["green"])
    text(ax, 2.84, 2.11, "参数 2N · 梯度 2N / 4N · 优化器状态 12N", in_box=alloc,
         ha="center", va="center", fontsize=9)
    text(ax, 2.84, 1.76, "激活 ≈ 34·s·b·h·L + logits s·b·V·6", in_box=alloc,
         ha="center", va="center", fontsize=9)
    text(ax, 5.82, 2.57, "库 workspace\n0.2–0.5 GiB", in_box=reserved,
         ha="center", va="center", fontsize=9)
    text(ax, 5.82, 1.88, "缓存空闲 block\n上个 step 的激活留池复用", in_box=reserved,
         ha="center", va="center", fontsize=8.5)
    text(ax, 5.82, 1.30, "碎片：inactive_split_bytes\n空闲但与相邻 block 合不回\n通常 reserved 的 5–15%", in_box=reserved,
         ha="center", va="center", fontsize=8.3)
    text(ax, 3.78, 0.57, "另留 OOM 安全边际 2–4 GiB；context 与 NCCL 由 cudaMalloc 直接分配，不在 reserved 里。",
         in_box=outer, ha="center", va="center", fontsize=9)
    save(fig, "training-state-memory-nesting")


def f9_flops():
    """Linear 的 6 FLOP/参数/token 与注意力 s^2 项。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("6N 从哪来：前向一个 GEMM，反向两个")
    xin = box(ax, 0.35, 2.45, 1.40, 0.78, "输入 x\n(tokens, n)", fc="#eff5fc", ec=C["blue"])
    w = box(ax, 0.35, 1.12, 1.40, 0.78, "权重 W\n(n, k)", fc="#fff4e8", ec=C["orange"])
    fwd = box(ax, 2.45, 2.05, 1.62, 1.08, "前向 GEMM\ny = xW\n2 FLOP/参数", fc="#ffffff", ec=C["blue"])
    _arrow(ax, (1.75, 2.84), (2.45, 2.78), C["blue"])
    _arrow(ax, (1.75, 1.52), (2.45, 2.34), C["orange"])
    bwdx = box(ax, 5.18, 2.55, 1.92, 0.84, "dgrad：dX = dYW^T\n2 FLOP/参数", fc="#ffffff", ec=C["blue"])
    bwdw = box(ax, 5.18, 1.35, 1.92, 0.84, "wgrad：dW = x^T dY\n2 FLOP/参数", fc="#ffffff", ec=C["orange"])
    _arrow(ax, (4.07, 2.78), (5.18, 2.96), C["blue"])
    _arrow(ax, (4.07, 2.35), (5.18, 1.78), C["orange"])
    text(ax, 3.8, 0.75, "前向 2 + 反向 dgrad 2 + wgrad 2 = 6 FLOP / 参数 / token", ha="center", va="center", fontsize=11)
    text(ax, 3.8, 0.34, "注意力另有 QK^T 与 PV：每层每 token 全 mask 12sh，因果 mask 减半为 6sh。",
         ha="center", va="center", fontsize=9.5)
    save(fig, "training-state-flops-per-parameter")


def f10_mfu_hfu():
    """相同 step 下 MFU / HFU 分子和重计算之间的差异。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 7.6); ax.set_ylim(0, 4.0); ax.axis("off")
    ax.set_title("MFU 看模型做了多少有用计算，HFU 还数上重算")
    rows = [
        ("模型需要的计算", 2.2, C["blue"], "MFU 分子"),
        ("选择性重计算", 0.18, C["orange"], "HFU 多数一点"),
        ("全量重计算", 1.1, C["red"], "HFU 多算 1/3"),
    ]
    x0, maxw = 0.42, 5.8
    for i, (name, units, color, desc) in enumerate(rows):
        y = 3.05 - i * 0.72
        text(ax, x0, y + 0.18, name, ha="left", va="center", fontsize=10)
        width = maxw * units / 2.2
        ax.add_patch(Rectangle((2.35, y), width, 0.36, facecolor=color + "55", edgecolor=color, linewidth=0.7))
        text(ax, 2.50 + width, y + 0.18, desc, ha="left", va="center", fontsize=9.5, color=color)
    text(ax, 3.8, 0.73, "70B · 1024 H100 · 4M token/step · 实测 4.5 s", ha="center", va="center", fontsize=10)
    text(ax, 3.8, 0.42, "MFU 41.3%  ·  HFU（选择性）42.3%  ·  HFU（全量）55.1%", ha="center", va="center", fontsize=11)
    text(ax, 3.8, 0.13, "重算让 GPU 更忙，不代表每秒处理的 token 更多。", ha="center", va="center", fontsize=9.5)
    save(fig, "training-state-mfu-vs-hfu")


if __name__ == "__main__":
    f1_budget(); f2_ledgers(); f3_four_states(); f4_lifecycle(); f5_timeline()
    f6_bytes(); f7_activations(); f8_memory_nesting(); f9_flops(); f10_mfu_hfu()
