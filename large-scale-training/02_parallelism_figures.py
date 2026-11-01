"""《并行策略全景》配图：统一使用 _plot.py 的字体、画布与碰撞检查。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import ARROW_LABEL_SIZE, C, box, box_title, collide, plt, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#dce9f7"
ORANGE = "#fae6d6"
GREEN = "#dcefe2"
PURPLE = "#eee1f3"
GRAY = "#e8e8e8"
RED = "#f5dede"


def _canvas(w=7.6, h=4.4, xlim=(0, 10), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _label(ax, x, y, s, size=10, color="#111111", ha="center", va="center", **kw):
    return text(ax, x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def _card(ax, x, y, w, h, s, fc="white", ec="#777777", size=10, lw=0.9):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if s:
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center", fontsize=size)
    return p


def _arrow(ax, a, b, color="#555555", lw=1.15, style="->", ms=10, **kw):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, **kw))


def _strip(ax, x, y, w, h, labels, colors, fs=8.5):
    cw = w / len(labels)
    for i, (label, color) in enumerate(zip(labels, colors)):
        _card(ax, x + i * cw, y, cw, h, label, fc=color, ec="#ffffff", size=fs, lw=0.5)


def states_grid():
    fig, ax = _canvas(7.6, 5.7, (0, 12), (0, 8.8))
    _label(ax, 6, 8.35, "五种并行各切哪种状态", 13)
    cols = ["DP", "ZeRO-3", "TP / SP", "CP", "PP", "EP"]
    rows = ["参数", "梯度", "优化器状态", "激活"]
    x0, y0, cw, rh = 2.05, 4.05, 1.48, 0.68
    for j, col in enumerate(cols):
        _label(ax, x0 + j * cw + cw / 2, 7.62, col, 9.5)
    for i, row in enumerate(rows):
        y = y0 + (3 - i) * rh
        _label(ax, 1.22, y + rh / 2, row, 10, ha="right")
        for j in range(6):
            x = x0 + j * cw
            _card(ax, x, y, cw - 0.06, rh - 0.06, "", fc="#ffffff", ec="#d5d9df", lw=0.65)
            # 0 = replicated; 1 = cut; 2 = partly cut / reduced / transiently gathered.
            status = [
                [0, 2, 1, 0, 1, 2],  # parameter
                [2, 1, 1, 2, 1, 1],  # gradient
                [0, 1, 1, 0, 1, 1],  # optimizer
                [1, 1, 2, 1, 1, 2],  # activation
            ][i][j]
            cx, cy = x + (cw - 0.06) / 2, y + (rh - 0.06) / 2
            if status == 0:
                ax.add_patch(plt.Circle((cx, cy), 0.095, fill=False, ec="#6d7782", lw=1.15))
            elif status == 1:
                ax.add_patch(plt.Circle((cx, cy), 0.095, fc=C["blue"], ec=C["blue"], lw=1))
            else:
                ax.add_patch(plt.Circle((cx, cy), 0.095, fc=C["orange"], ec=C["orange"], lw=1))
    _label(ax, 0.55, 3.42, "图例", 9, ha="left")
    ax.add_patch(plt.Circle((1.6, 3.42), 0.075, fc=C["blue"], ec=C["blue"]))
    _label(ax, 1.78, 3.42, "切分", 8.5, ha="left")
    ax.add_patch(plt.Circle((2.65, 3.42), 0.075, fc=C["orange"], ec=C["orange"]))
    _label(ax, 2.83, 3.42, "部分切分 / 临时拼全", 8.5, ha="left")
    ax.add_patch(plt.Circle((5.2, 3.42), 0.075, fill=False, ec="#6d7782", lw=1.1))
    _label(ax, 5.38, 3.42, "复制", 8.5, ha="left")
    comm = ["梯度 AR\nIB", "RS + AG\nIB", "层边界 AG / RS\nNVLink", "K/V 块\nsend/recv\nIB", "激活\nsend/recv\nIB", "token all-to-all\nIB / NVLink"]
    for j, s in enumerate(comm):
        _label(ax, x0 + j * cw + cw / 2, 2.53, s, 8.3)
    _label(ax, 6, 1.63, "切分省常驻显存；临时拼接与归约决定通信账", 9.5, color="#333333")
    save(fig, "parallelism-states-grid")


def mesh_8cards():
    fig, ax = _canvas(7.6, 4.8, (0, 12), (0, 7.4))
    _label(ax, 6, 6.95, "一张 8 卡的五元组摆法", 13)
    _label(ax, 6, 6.55, "TP=2 × DP=2 × PP=2；rank = tp + 2·dp + 4·pp", 9.5)
    nodes = [[0, 1, 6, 7], [2, 3, 4, 5]]
    xnode = [0.8, 6.65]
    for ni, ranks in enumerate(nodes):
        ax.add_patch(plt.Rectangle((xnode[ni], 3.45), 4.55, 2.6, fc="#f8fafc", ec="#aab5c2", lw=0.9, zorder=0))
        _label(ax, xnode[ni] + 2.275, 5.75, f"节点 {ni} · 4 卡", 10.5)
        for k, rank in enumerate(ranks):
            x = xnode[ni] + 0.22 + k * 1.04
            _card(ax, x, 4.32, 0.9, 0.72, f"rank {rank}", fc=BLUE, ec=C["blue"], size=8.4)
            tp, dp, pp = rank % 2, (rank // 2) % 2, rank // 4
            _label(ax, x + 0.45, 4.1, f"t{tp} d{dp} p{pp}", 7.6)
        _label(ax, xnode[ni] + 2.275, 3.7, "相邻 rank 配成 TP 组，走节点内 NVLink", 8.5)
    # DP and PP edges cross nodes. Separate lanes keep lines clear of the cards.
    _arrow(ax, (5.45, 2.7), (6.3, 2.7), color=C["orange"], lw=1.7, ms=11)
    _arrow(ax, (5.45, 2.2), (6.3, 2.2), color=C["red"], lw=1.7, ms=11)
    _label(ax, 3.8, 2.7, "DP 组：{0,2} · {1,3} · {4,6} · {5,7}", 8.5, ha="right")
    _label(ax, 3.8, 2.2, "PP 组：{0,4} · {1,5} · {2,6} · {3,7}", 8.5, ha="right")
    _label(ax, 6.75, 2.7, "跨节点 IB", 8.5, ha="left")
    _label(ax, 6.75, 2.2, "跨节点 IB", 8.5, ha="left")
    _label(ax, 6, 1.35, "Megatron 顺序 tp-cp-ep-dp-pp：TP 变化最快，PP 最慢；节点映射让 DP、PP 都跨节点", 8.3)
    save(fig, "parallelism-mesh-8cards")


def zero_step_comm():
    fig, ax = _canvas(7.6, 4.9, (0, 12), (0, 7.2))
    _label(ax, 6, 6.8, "四级 ZeRO 在一个 step 里的通信（每卡，单位：元素）", 12.5)
    headers = ["前向", "反向", "optimizer.step()", "step 之后", "下一步前向"]
    x0, cw = 2.0, 1.75
    for j, s in enumerate(headers):
        _label(ax, x0 + j * cw + cw / 2, 6.14, s, 8.8)
    rows = [
        ("DP（ZeRO-0）", 5.15, [(1, "AR 梯度\n≈ 2N", RED)], "合计 2N"),
        ("ZeRO-1 / 2", 4.05, [(1, "RS 梯度\n≈ N", GREEN), (3, "AG 参数\n≈ N", ORANGE)], "合计 2N"),
        ("ZeRO-3", 2.95, [(0, "逐层 AG\n参数 ≈ N", BLUE), (1, "反向 AG 参数 ≈ N", ORANGE), (1, "反向 RS 梯度 ≈ N", GREEN)], "合计 3N"),
        ("ZeRO-3 不释放", 1.85, [(0, "逐层 AG\n参数 ≈ N", BLUE), (1, "逐层 RS\n梯度 ≈ N", GREEN)], "合计 2N"),
    ]
    for name, y, comms, total in rows:
        _label(ax, 1.70, y + 0.3, name, 8.8, ha="right")
        for j in range(5):
            _card(ax, x0 + j * cw, y, cw - 0.1, 0.82, "", fc="#fbfbfb", ec="#d5d9df", lw=0.6)
        by_col = {}
        for col, lab, color in comms:
            by_col.setdefault(col, []).append((lab, color))
        for col, entries in by_col.items():
            if len(entries) == 1:
                lab, color = entries[0]
                _card(ax, x0 + col * cw + 0.1, y + 0.02, cw - 0.3, 0.78, lab, fc=color, ec="#ffffff", size=7.6, lw=0.4)
            else:
                for slot, (lab, color) in enumerate(entries):
                    _card(ax, x0 + col * cw + 0.1, y + 0.03 + slot * 0.38, cw - 0.3, 0.35, lab, fc=color, ec="#ffffff", size=6.2, lw=0.4)
        _label(ax, 11.05, y + 0.31, total, 8.5, ha="left")
    _label(ax, 6, 1.04, "ZeRO-1/2 的 step 后 AG 在 ZeRO-3 中移到下一步；反向 AG 才是净增的一次通信", 8.5)
    save(fig, "parallelism-zero-step-comm")


def zero_stages():
    fig, ax = _canvas(7.6, 4.6, (0, 12), (0, 7))
    _label(ax, 6, 6.6, "四级 ZeRO：参数 P、梯度 G、优化器状态 O 的每卡常驻量", 12)
    _label(ax, 6, 6.15, "N_d = 4；混合精度 Adam 下 P=2N、G=2N、O=12N", 8.8, color="#444444")
    labels = ["DP / ZeRO-0", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    mem = ["P 2N · G 2N · O 12N", "P 2N · G 2N · O 3N", "P 2N · G N/2 · O 3N", "P N/2 · G N/2 · O 3N"]
    comm = ["AR 2N", "RS N + AG N = 2N", "RS N + AG N = 2N", "AG N + AG N + RS N = 3N"]
    for k, (name, m, c) in enumerate(zip(labels, mem, comm)):
        y = 5.0 - k * 1.08
        _label(ax, 2.15, y + 0.29, name, 9, ha="right")
        _card(ax, 2.45, y, 5.25, 0.6, m, fc="#f8fafc", ec="#d5d9df", size=9.1)
        _label(ax, 7.98, y + 0.3, c, 8.4, ha="left")
    _label(ax, 6, 0.52, "每升一级，多切一种常驻状态：优化器 → 梯度 → 参数；对应通信从 2N 到 3N", 8.5)
    save(fig, "parallelism-zero-stages")


def zero3_layer_step():
    fig, ax = _canvas(7.6, 5.2, (0, 12), (0, 7.8))
    _label(ax, 6, 7.35, "ZeRO-3 一层、4 张卡：四种状态分片，step 内通信三次", 12)
    cols = ["常驻", "① 前向前\nAG 参数", "② 前向\n用完整参数", "③ 反向前\nAG 参数", "④ 反向\n算梯度", "⑤ RS 梯度\noptimizer.step"]
    x0, cw = 1.58, 1.68
    for j, s in enumerate(cols):
        _label(ax, x0 + j * cw + cw / 2, 6.62, s, 8.1)
    rank_y = [5.45, 4.6, 3.75, 2.9]
    for r, y in enumerate(rank_y):
        _label(ax, 1.27, y + 0.25, f"GPU {r}", 8.5, ha="right")
        # each phase shows local state and transient gathered data, rather than a full tensor grid
        for j in range(6):
            fc = ["#f8fafc", BLUE, ORANGE, ORANGE, GREEN, PURPLE][j]
            labels = [f"P{r} / O{r}", "AG 参数", "完整 P", "AG 参数", "G 求和", "更新 P{r}" ]
            _card(ax, x0 + j * cw, y, cw - 0.12, 0.54, labels[j], fc=fc, ec="#ffffff", size=8)
    for j in range(5):
        _arrow(ax, (x0 + j * cw + cw - 0.1, 2.48), (x0 + (j + 1) * cw + 0.03, 2.48), lw=1.35)
    _label(ax, 6, 1.88, "每卡通信：前向 AG ≈ N；反向 AG ≈ N；反向 RS ≈ N；合计 3N", 9.3)
    _label(ax, 6, 1.35, "激活按 batch 切分；反向完成后释放。ZeRO-1/2 step 后 AG 对应下一步的前向 AG", 8.4)
    save(fig, "parallelism-zero3-layer-step")


def fsdp1_vs_fsdp2():
    fig, ax = _canvas(7.6, 5.0, (0, 12), (0, 7.6))
    _label(ax, 6, 7.1, "FSDP1 与 FSDP2 的切法", 12.5)
    _label(ax, 6, 6.67, "一个 wrap 单元：W 形状 [4,4] 共 16 个元素，b 形状 [4]；N_d=2（蓝=GPU0，橙=GPU1）", 8.6)
    x0, cw = 2.15, 0.43
    for r, name in enumerate(["FSDP1：FlatParameter 后切", "FSDP2：每个参数沿 dim 0 切"]):
        y = 4.55 - r * 1.75
        _label(ax, 1.95, y + 0.33, name, 8.4, ha="right")
        for i in range(20):
            x = x0 + i * cw
            owner = 0 if i < 10 else 1
            if r == 1:
                owner = 0 if (i < 8 or 16 <= i < 18) else 1
            fc = BLUE if owner == 0 else ORANGE
            ax.add_patch(plt.Rectangle((x, y), cw - 0.03, 0.62, fc=fc, ec="#ffffff", lw=0.5))
            # W cells get a tiny index every fourth element to keep the grid readable.
            if i < 16 and i % 4 == 0:
                _label(ax, x + cw / 2, y + 0.31, f"W{i//4}", 6.4)
            if i >= 16:
                _label(ax, x + cw / 2, y + 0.31, f"b{i-16}", 6.4)
        if r == 0:
            _label(ax, x0 + 5 * cw, y - 0.34, "切点：第 10 个元素 → GPU0 = W 前 10 个；GPU1 = W 后 6 个 + 整个 b", 7.8)
        else:
            _label(ax, x0 + 5 * cw, y - 0.34, "GPU0 = W 前两行 + b 前两个；GPU1 = W 后两行 + b 后两个", 7.8)
        _label(ax, x0 + 5 * cw, y - 0.74, "FSDP1 可在参数中间切；FSDP2 的每片对应原参数的 Shard(0) 切片", 7.9)
    _label(ax, 6, 0.62, "同一个 wrap 单元：分片边界决定本地对象是扁平切片还是 per-parameter DTensor", 8.5)
    save(fig, "parallelism-fsdp1-vs-fsdp2")


def hsdp_grid():
    fig, ax = _canvas(7.6, 4.6, (0, 12), (0, 7))
    _label(ax, 6, 6.55, "HSDP：16 张卡排成 N_r × N_s = 2 × 8", 12)
    _label(ax, 6, 6.1, "placement = (Replicate(), Shard(0))；横向分片组，纵向复制组", 9)
    x0, cw = 2.15, 0.98
    for row in range(2):
        y = 4.5 - row * 1.7
        _label(ax, 1.62, y + 0.4, f"副本 {row}\n节点 {row}", 8.3, ha="right")
        for i in range(8):
            _card(ax, x0 + i * cw, y, 0.86, 0.8, f"卡 {row*8+i}\n段 {i}", fc=BLUE if row == 0 else GREEN, ec="#ffffff", size=7.6)
            if row == 0:
                _arrow(ax, (x0 + i * cw + 0.43, y - 0.08), (x0 + i * cw + 0.43, y - 0.9), color=C["red"], lw=1.0, ms=8)
    _label(ax, 6, 1.75, "横向：ZeRO-3 的 AG + AG + RS ≈ 3N，走 NVLink", 9)
    _label(ax, 6, 1.24, "纵向：同一段梯度在副本间 all-reduce ≈ 2N/N_s，走 IB；每卡状态只省 N_s 倍", 8.5)
    save(fig, "parallelism-hsdp-grid")


def tp_column_row_split():
    fig, ax = _canvas(7.6, 5.6, (0, 12), (0, 8.4))
    _label(ax, 6, 7.95, "TP：列切与行切配对，让层内矩阵变小", 12.5)
    _label(ax, 6, 7.5, "t=4 张卡；每卡算 1/t 的矩阵乘，行切后的部分和要相加", 9)
    _label(ax, 6, 6.75, "列切（column-parallel）：A = [A1 | A2 | A3 | A4]", 10.2)
    _card(ax, 0.7, 5.3, 1.5, 0.85, "X\n完整副本", fc=BLUE, ec=C["blue"], size=8.5)
    _arrow(ax, (2.35, 5.72), (3.0, 5.72))
    for i in range(4):
        _card(ax, 3.1 + i * 1.05, 5.3, 0.88, 0.85, f"A{i+1}\n列块", fc=ORANGE, ec=C["orange"], size=8)
        _card(ax, 3.1 + i * 1.05, 4.18, 0.88, 0.66, f"Y{i+1}", fc=GREEN, ec=C["green"], size=9)
        _arrow(ax, (3.54 + i * 1.05, 5.24), (3.54 + i * 1.05, 4.9), lw=0.95)
    _label(ax, 8.9, 5.72, "Yi = X Ai\n输出按列分布", 8.6, ha="left")
    _label(ax, 6, 3.55, "行切（row-parallel）：A = [A1 ; A2 ; A3 ; A4]", 10.2)
    for i in range(4):
        _card(ax, 0.9 + i * 1.15, 2.38, 0.98, 0.78, f"X{i+1}", fc=BLUE, ec=C["blue"], size=9)
        _card(ax, 0.9 + i * 1.15, 1.32, 0.98, 0.78, f"A{i+1}\n行块", fc=ORANGE, ec=C["orange"], size=8)
        _arrow(ax, (1.39 + i * 1.15, 2.32), (1.39 + i * 1.15, 2.14), lw=0.95)
    _arrow(ax, (5.8, 1.7), (6.6, 1.7), color=C["red"], lw=1.35)
    _card(ax, 6.8, 1.3, 2.4, 0.8, "Σ Xi Ai\nall-reduce", fc=RED, ec=C["red"], size=8.5)
    _label(ax, 9.62, 1.7, "完整 Y", 8.5, ha="left")
    _label(ax, 6, 0.62, "列切输出正好是下一层行切的输入布局；层边界处做集合通信", 8.6)
    save(fig, "parallelism-tp-column-row-split")


def tp_conjugate_fg():
    fig, ax = _canvas(7.6, 4.9, (0, 12), (0, 7.4))
    _label(ax, 6, 6.95, "TP 的 f / g：前向与反向互为共轭", 12.5)
    _label(ax, 6, 6.48, "前向：f 恒等，g 做 all-reduce", 9.2)
    xs = [0.65, 2.65, 4.65, 6.65, 8.65]
    labels = ["X\n完整", "f\n恒等", "列切 W1i\nGELU", "行切 W2i", "g\nall-reduce\nY 完整"]
    colors = [BLUE, "#ffffff", ORANGE, ORANGE, GREEN]
    widths = [1.3, 1.2, 1.5, 1.5, 1.75]
    for x, lab, fc, w in zip(xs, labels, colors, widths):
        _card(ax, x, 4.92, w, 1.0, lab, fc=fc, ec="#aeb8c4", size=8)
    for i in range(4):
        _arrow(ax, (xs[i] + widths[i] + 0.04, 5.42), (xs[i+1] - 0.06, 5.42))
    _label(ax, 6, 4.16, "反向：g 恒等，f 对各卡的输入梯度 all-reduce", 9.2)
    rev = ["dX\n完整", "f\nall-reduce", "W1i^T\nGELU′", "W2i^T", "g\n恒等\ndY 完整"]
    for x, lab, fc, w in zip(xs, rev, [GREEN, RED, ORANGE, ORANGE, BLUE], widths):
        _card(ax, x, 2.55, w, 1.0, lab, fc=fc, ec="#aeb8c4", size=8)
    for i in range(4):
        _arrow(ax, (xs[i+1] - 0.06, 3.08), (xs[i] + widths[i] + 0.04, 3.08))
    _label(ax, 6, 1.72, "每个 MLP 块前向 + 反向共 2 次 all-reduce；注意力块同理", 8.7)
    _label(ax, 6, 1.2, "无 SP：集合通信在激活张量上；有 SP：把它改写为 AG + RS", 8.3)
    save(fig, "parallelism-tp-conjugate-fg")


def ring_attention():
    fig, ax = _canvas(7.6, 5.0, (0, 12), (0, 7.5))
    _label(ax, 6, 7.05, "Ring Attention，N_c=4：Q 留在本卡，K/V 块沿环流动", 12)
    # ring nodes
    pts = [(2.1, 5.7), (4.25, 5.7), (4.25, 4.75), (2.1, 4.75)]
    for i, (x, y) in enumerate(pts):
        _card(ax, x, y, 1.3, 0.55, f"卡 {i} · Q{i}", fc=BLUE, ec=C["blue"], size=8.8)
    _arrow(ax, (3.48, 5.98), (4.15, 5.98), color=C["orange"], lw=1.5)
    _arrow(ax, (4.98, 5.64), (4.98, 5.28), color=C["orange"], lw=1.5)
    _arrow(ax, (4.15, 4.48), (3.48, 4.48), color=C["orange"], lw=1.5)
    _arrow(ax, (1.85, 5.28), (1.85, 5.64), color=C["orange"], lw=1.5)
    _label(ax, 3.2, 4.02, "send/recv K/V，收到一块就合并 online softmax 的 m、l、acc", 8.4)
    _label(ax, 8.2, 6.45, "每卡每步用的 K/V 块", 9.2)
    x0, y0, cw, rh = 6.75, 3.95, 1.12, 0.5
    for j in range(4):
        _label(ax, x0 + j * cw + cw/2, 6.08, f"步 {j}", 7.8)
    for r in range(4):
        _label(ax, 6.42, y0 + (3-r)*rh + rh/2, f"卡 {r}", 7.8, ha="right")
        order = [(r - j) % 4 for j in range(4)]
        for j, b in enumerate(order):
            _card(ax, x0 + j*cw, y0 + (3-r)*rh, cw - 0.05, rh - 0.04, f"K/V {b}",
                  fc=[BLUE, ORANGE, GREEN, PURPLE][b], ec="#ffffff", size=7.8, lw=0.4)
    _label(ax, 6, 1.45, "每列四卡用不同块，通信可与分块注意力重叠；前向最大误差由 ring toy 实测", 8.2)
    save(fig, "parallelism-ring-attention")


def ulysses_all_to_all():
    fig, ax = _canvas(7.6, 4.7, (0, 12), (0, 7))
    _label(ax, 6, 6.55, "Ulysses，N_c=4：all-to-all 把序列切换成头切", 12)
    _label(ax, 2.65, 5.95, "按序列切：每卡持 1/4 token、全部 4 头", 8.7)
    _label(ax, 9.25, 5.95, "按头切：每卡持全部 token、1 个头", 8.7)
    x0, y0, cw, rh = 0.8, 3.45, 0.82, 0.46
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    for r in range(4):
        _label(ax, 0.5, y0 + (3-r)*rh + rh/2, f"卡 {r}", 7.5, ha="right")
        for c in range(4):
            _card(ax, x0 + c*cw, y0 + (3-r)*rh, cw - 0.04, rh - 0.04, f"h{c}", fc=colors[c], ec="#ffffff", size=7.3, lw=0.4)
    _arrow(ax, (4.25, 4.2), (5.55, 4.2), color=C["red"], lw=1.8, ms=12)
    _label(ax, 4.9, 4.65, "Q/K/V", 8, color="#333333")
    x1 = 6.3
    for r in range(4):
        _label(ax, 6.0, y0 + (3-r)*rh + rh/2, f"卡 {r}", 7.5, ha="right")
        for c in range(4):
            _card(ax, x1 + c*cw, y0 + (3-r)*rh, cw - 0.04, rh - 0.04, f"h{r}", fc=colors[r], ec="#ffffff", size=7.3, lw=0.4)
    _arrow(ax, (5.55, 3.0), (4.25, 3.0), color=C["red"], lw=1.8, ms=12)
    _label(ax, 4.9, 2.58, "O", 8, color="#333333")
    _label(ax, 6, 1.62, "颜色表示头；all-to-all 相当于转置 4×4 分布表。前向 4 次、反向 4 次", 8.2)
    _label(ax, 6, 1.12, "每次每卡载荷 (s/N_c)·b·h；N_c 不能超过可分配的 K/V 头数", 8.2)
    save(fig, "parallelism-ulysses-all-to-all")


def pipeline_gpipe_vs_1f1b():
    fig, ax = _canvas(7.6, 5.4, (0, 18), (0, 8.1))
    _label(ax, 9, 7.65, "流水线：GPipe 先填满再排空，1F1B 尽早消费激活", 12)
    # GPipe p=4, m=4: 4 forward columns then 4 backward columns, with stage offsets.
    _label(ax, 4.7, 7.08, "GPipe：p=4，m=4", 9.5)
    x0, cw, rh = 2.0, 0.63, 0.42
    for r in range(4):
        y = 5.45 - r * rh
        _label(ax, 1.62, y + rh/2, f"stage {r}", 7.5, ha="right")
        for t in range(10):
            act = ""
            fc = "#f7f7f7"
            if r <= t < r + 4:
                act, fc = str(t-r), BLUE
            elif r + 4 <= t < r + 8:
                act, fc = str(7-(t-r)), GREEN
            ax.add_patch(plt.Rectangle((x0 + t*cw, y), cw - 0.04, rh - 0.04, fc=fc, ec="#ffffff", lw=0.5))
            if act:
                ax.text(x0 + t*cw + cw/2, y + rh/2, act, ha="center", va="center", fontsize=6.5, color="#222222")
    _label(ax, 11.0, 5.0, "蓝 F · 绿 B · 灰气泡", 7.5, ha="left")
    _label(ax, 6.3, 4.08, "1F1B：p=4，m=8", 9.5)
    x1, cw2 = 2.0, 0.42
    # Standard non-interleaved 1F1B schedule in stage-local order: warmup, alternating, cooldown.
    schedule = [
        ["F0","F1","F2","B0","F3","B1","F4","B2","F5","B3","F6","B4","F7","B5","B6","B7"],
        [".","F0","F1","F2","B0","F3","B1","F4","B2","F5","B3","F6","B4","F7","B5","B6","B7"],
        [".",".","F0","F1","B0","F2","B1","F3","B2","F4","B3","F5","B4","F6","B5","F7","B6","B7"],
        [".",".",".","F0","B0","F1","B1","F2","B2","F3","B3","F4","B4","F5","B5","F6","B6","F7","B7"],
    ]
    for r, row in enumerate(schedule):
        y = 3.35 - r * 0.43
        _label(ax, 1.62, y + 0.17, f"stage {r}", 7.5, ha="right")
        for t, act in enumerate(row):
            fc = BLUE if act.startswith("F") else GREEN if act.startswith("B") else "#eeeeee"
            ax.add_patch(plt.Rectangle((x1 + t*cw2, y), cw2 - 0.025, 0.34, fc=fc, ec="#ffffff", lw=0.5))
            if act != ".":
                ax.text(x1 + t*cw2 + cw2/2, y + 0.17, act, ha="center", va="center", fontsize=5.6, color="#222222")
    _label(ax, 9, 0.95, "GPipe 在途激活 O(m)；1F1B 在途约 O(p)。二者气泡率相同", 8.6)
    save(fig, "parallelism-pipeline-gpipe-vs-1f1b")


def ep_dispatch_combine():
    fig, ax = _canvas(7.6, 5.0, (0, 12), (0, 7.4))
    _label(ax, 6, 6.95, "EP：token 路由到专家所在卡，计算后原路返回", 12)
    _label(ax, 2.2, 6.25, "路由前：每卡 4 个 token", 8.8)
    for r in range(4):
        _card(ax, 0.85 + r*0.9, 5.15, 0.68, 0.52, f"t{r}", fc=BLUE, ec=C["blue"], size=8.5)
        _label(ax, 1.19 + r*0.9, 4.9, f"→ E{[2,0,3,1][r]}", 7.5)
    _arrow(ax, (4.85, 5.4), (6.25, 5.4), color=C["red"], lw=1.6, ms=11)
    _label(ax, 5.55, 5.78, "dispatch all-to-all", 8)
    # Two expert owners
    for card, x, experts in [(0, 6.55, [0,1]), (1, 9.25, [2,3])]:
        _card(ax, x, 3.75, 2.0, 0.62, f"卡 {card}：专家 E{experts[0]}、E{experts[1]}", fc=GREEN, ec=C["green"], size=8.2)
    _card(ax, 6.55, 2.65, 2.0, 0.72, "E0 ← {t1}\nE1 ← {t3}", fc=BLUE, ec=C["blue"], size=8.1)
    _card(ax, 9.25, 2.65, 2.0, 0.72, "E2 ← {t0}\nE3 ← {t2}", fc=ORANGE, ec=C["orange"], size=8.1)
    _arrow(ax, (9.1, 3.98), (9.1, 3.55), color=C["green"], lw=1.1)
    _arrow(ax, (10.25, 2.52), (5.3, 1.9), color=C["red"], lw=1.5, ms=11)
    _label(ax, 7.6, 1.72, "combine all-to-all：按路由权重送回并合并", 8.3)
    _label(ax, 6, 0.87, "反向沿相反方向走；每层前向 dispatch + combine，反向再各一次", 8.4)
    save(fig, "parallelism-ep-dispatch-combine")


def bubble_curve():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(4, 64)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("micro-batch 数 m")
    ax.set_ylabel("气泡 / 理想计算时间（倍）")
    ax.grid(axis="y", color="#d9dde2", lw=0.6)
    xs = list(range(4, 65))
    for p, color in [(4, C["blue"]), (8, C["green"]), (16, C["orange"])]:
        ys = [(p - 1) / m for m in xs]
        ax.plot(xs, ys, color=color, lw=1.7, label=f"p={p}： (p-1)/m")
        ys2 = [(p - 1) / (2 * m) for m in xs]
        ax.plot(xs, ys2, color=color, lw=1.3, ls="--", label=f"p={p}：v=2")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    _label(ax, 33, 0.35, "interleaved v=2：气泡约减半，通信乘 2", 8.1)
    save(fig, "parallelism-bubble-rate")


def main():
    funcs = [states_grid, mesh_8cards, zero_step_comm, zero_stages, zero3_layer_step,
             fsdp1_vs_fsdp2, hsdp_grid, tp_column_row_split, tp_conjugate_fg,
             ring_attention, ulysses_all_to_all, pipeline_gpipe_vs_1f1b,
             ep_dispatch_combine, bubble_curve]
    for func in funcs:
        func()
    print(f"全部 {len(funcs)} 张图完成。")


if __name__ == "__main__":
    main()
