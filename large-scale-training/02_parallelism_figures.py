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
    fig, ax = _canvas(7.6, 4.55, (0, 12), (0, 7.0))
    _label(ax, 6, 6.62, "五种并行各切哪种状态", 13)
    cols = ["DP", "ZeRO-3", "TP / SP", "CP", "PP", "EP"]
    rows = ["参数", "梯度", "优化器状态", "激活"]
    x0, y0, cw, rh = 2.05, 2.55, 1.48, 0.63
    for j, col in enumerate(cols):
        _label(ax, x0 + j * cw + cw / 2, 5.25, col, 9.5)
    status_rows = [
        [0, 2, 1, 0, 1, 2],
        [0, 1, 1, 0, 1, 2],
        [0, 1, 1, 0, 1, 2],
        [1, 1, 2, 1, 1, 2],
    ]
    for i, row in enumerate(rows):
        y = y0 + (3 - i) * rh
        _label(ax, 1.22, y + rh / 2, row, 9.5, ha="right")
        for j, status in enumerate(status_rows[i]):
            x = x0 + j * cw
            _card(ax, x, y, cw - 0.06, rh - 0.05, "", fc="#ffffff", ec="#d5d9df", lw=0.65)
            cx, cy = x + (cw - 0.06) / 2, y + (rh - 0.05) / 2
            if status == 0:
                ax.add_patch(plt.Circle((cx, cy), 0.09, fill=False, ec="#6d7782", lw=1.15))
            elif status == 1:
                ax.add_patch(plt.Circle((cx, cy), 0.09, fc=C["blue"], ec=C["blue"], lw=1))
            else:
                ax.add_patch(plt.Circle((cx, cy), 0.09, fc=C["orange"], ec=C["orange"], lw=1))
    _label(ax, 1.22, 2.05, "主通信", 9, ha="right")
    comm = ["梯度 AR\nIB", "参数 AG + 梯度 RS\nIB", "层边界 AG / RS\nNVLink", "K/V send/recv\nIB", "激活 send/recv\nIB", "token all-to-all\nIB / NVLink"]
    for j, s in enumerate(comm):
        _label(ax, x0 + j * cw + cw / 2, 2.05, s, 7.8)
    ax.add_patch(plt.Circle((2.35, 0.72), 0.07, fc=C["blue"], ec=C["blue"]))
    _label(ax, 2.52, 0.72, "切分", 8, ha="left")
    ax.add_patch(plt.Circle((3.35, 0.72), 0.07, fc=C["orange"], ec=C["orange"]))
    _label(ax, 3.52, 0.72, "部分切分 / 临时拼全", 8, ha="left")
    ax.add_patch(plt.Circle((5.95, 0.72), 0.07, fill=False, ec="#6d7782", lw=1.1))
    _label(ax, 6.12, 0.72, "复制", 8, ha="left")
    save(fig, "parallelism-states-grid")


def mesh_8cards():
    fig, ax = _canvas(7.6, 4.8, (0, 12), (0, 7.4))
    _label(ax, 6, 6.95, "一张 8 卡的五元组摆法", 13)
    _label(ax, 6, 6.55, "TP=2 × DP=2 × PP=2；rank = tp + 2·dp + 4·pp；节点 = rank // 4", 9.5)
    nodes = [[0, 1, 2, 3], [4, 5, 6, 7]]
    xnode = [0.8, 6.65]
    for ni, ranks in enumerate(nodes):
        ax.add_patch(plt.Rectangle((xnode[ni], 3.45), 4.55, 2.6, fc="#f8fafc", ec="#aab5c2", lw=0.9, zorder=0))
        _label(ax, xnode[ni] + 2.275, 5.75, f"节点 {ni} · 4 卡", 10.5)
        for k, rank in enumerate(ranks):
            x = xnode[ni] + 0.22 + k * 1.04
            _card(ax, x, 4.32, 0.9, 0.72, f"rank {rank}", fc=BLUE, ec=C["blue"], size=8.4)
            tp, dp, pp = rank % 2, (rank // 2) % 2, rank // 4
            _label(ax, x + 0.45, 4.1, f"t{tp} d{dp} p{pp}", 7.6)
        _label(ax, xnode[ni] + 2.275, 3.7, f"pp={ni}，节点内 TP / DP 走 NVLink", 8.5)
    _label(ax, 6, 2.92, "TP 组 {0,1} · {2,3} · {4,5} · {6,7}　DP 组 {0,2} · {1,3} · {4,6} · {5,7}（节点内 NVLink）", 8.1)
    _arrow(ax, (5.45, 2.25), (6.3, 2.25), color=C["red"], lw=1.7, ms=11)
    _label(ax, 4.95, 2.25, "PP", 8.5, ha="right")
    _label(ax, 6.75, 2.25, "PP 组 {0,4} · {1,5} · {2,6} · {3,7}（跨节点 IB）", 8.1, ha="left")
    _label(ax, 6, 1.35, "Megatron 顺序 tp-cp-ep-dp-pp：TP 变化最快，PP 最慢；8 卡/节点且 TP=8 时，DP 组会跨节点", 8.3)
    save(fig, "parallelism-mesh-8cards")


def zero_step_comm():
    fig, ax = _canvas(7.6, 4.9, (0, 12), (0, 7.2))
    _label(ax, 6, 6.8, "四级 ZeRO 在一个 step 里的通信（每卡，单位：元素）", 12.5)
    headers = ["前向", "反向", "step", "step 之后", "下一 step 前向"]
    x0, cw = 2.0, 1.75
    for j, s in enumerate(headers):
        _label(ax, x0 + j * cw + cw / 2, 6.14, s, 7.7)
    rows = [
        ("DP（ZeRO-0）", 5.15, [(1, "AR 梯度\n≈ 2N", RED)], "合计 2N"),
        ("ZeRO-1 / 2", 4.05, [(1, "RS 梯度\n≈ N", GREEN), (3, "AG 参数\n≈ N", ORANGE)], "合计 2N"),
        ("ZeRO-3", 2.95, [(0, "逐层 AG\n参数 ≈ N", BLUE), (1, "逐层 AG ≈ N", BLUE), (1, "逐层 RS ≈ N", GREEN), (4, "逐层\nAG 参数", ORANGE)], "合计 3N"),
        ("ZeRO-3 不释放", 1.85, [(0, "逐层 AG\n参数 ≈ N", BLUE), (1, "逐层 RS\n梯度 ≈ N", GREEN)], "合计 2N"),
    ]
    for name, y, comms, total in rows:
        _label(ax, 1.70, y + 0.3, name, 8.8, ha="right")
        for j in range(5):
            _card(ax, x0 + j * cw, y, cw - 0.1, 0.9, "", fc="#fbfbfb", ec="#d5d9df", lw=0.6)
        by_col = {}
        for col, lab, color in comms:
            by_col.setdefault(col, []).append((lab, color))
        for col, entries in by_col.items():
            if len(entries) == 1:
                lab, color = entries[0]
                _card(ax, x0 + col * cw + 0.1, y + 0.02, cw - 0.3, 0.86, lab, fc=color, ec="#ffffff", size=7.3, lw=0.4)
            else:
                for slot, (lab, color) in enumerate(entries):
                    _card(ax, x0 + col * cw + 0.1, y + 0.02 + slot * 0.42, cw - 0.3, 0.4, lab, fc=color, ec="#ffffff", size=6.2, lw=0.4)
        _label(ax, 11.05, y + 0.31, total, 8.5, ha="left")
    _label(ax, 6, 1.06, "橙色两格是同一次参数 all-gather：ZeRO-1/2 更新完立刻把完整参数拼回来常驻；ZeRO-3 不常驻完整参数，", 8.2)
    _label(ax, 6, 0.66, "等下一步前向用到哪层再拼哪层，所以 step 之后那次省掉了。ZeRO-3 真正多出来的只有反向前的再一次 all-gather。", 8.2)
    _label(ax, 6, 0.26, "「不释放」= 前向后保留完整参数（reshard_after_forward=False），反向那次 all-gather 也省掉，代价是 2N 参数常驻。", 8.2)
    save(fig, "parallelism-zero-step-comm")


def zero_stages():
    fig, ax = _canvas(7.6, 5.2, (0, 12), (0, 7.9))
    _label(ax, 6, 7.55, "四级 ZeRO：N_d = 4 张卡上，参数 P（2N）、梯度 G（2N）、优化器状态 O（12N）各是整份还是 1/4 片", 10.5)
    _label(ax, 6, 7.12, "实心格 = 本卡持有的那一段；虚线格 = 不持有（在别的卡上）。每张卡的显存 = 实心部分之和。", 8.2)
    levels = [
        ("DP (ZeRO-0)", "每卡 16N", "通信 all-reduce 2N", "FFF"),
        ("ZeRO-1", "每卡 4N + 12N/N_d", "通信 RS N + AG N = 2N", "FFS"),
        ("ZeRO-2", "每卡 2N + 14N/N_d", "通信 2N，梯度即时释放", "FSS"),
        ("ZeRO-3 / FSDP", "每卡 16N/N_d", "通信 AG+AG+RS = 3N", "SSS"),
    ]
    state_colors = [BLUE, GREEN, PURPLE]
    state_bord = [C["blue"], C["green"], C["purple"]]
    state_names = ["P", "G", "O"]
    x0, cw = 2.75, 2.12
    for k, (name, mem, comm, held) in enumerate(levels):
        y = 5.55 - k * 1.42
        _label(ax, 0.55, y + 0.9, name, 9, ha="left")
        _label(ax, 0.55, y + 0.55, mem, 7.8, ha="left", color="#444444")
        _label(ax, 0.55, y + 0.28, comm, 7.8, ha="left", color="#444444")
        for g in range(4):
            x = x0 + g * cw
            _card(ax, x, y, 1.95, 1.2, "", fc="#ffffff", ec="#d5d9df", lw=0.65)
            _label(ax, x + 0.16, y + 1.02, f"GPU {g}", 7.2, ha="left", color="#555555")
            for s in range(3):
                by = y + 0.14 + (2 - s) * 0.27
                _label(ax, x + 0.1, by + 0.11, state_names[s], 6.8, ha="left", color="#666666")
                for seg in range(4):
                    sx = x + 0.34 + seg * 0.38
                    if held[s] == "F" or seg == g:
                        ax.add_patch(plt.Rectangle((sx, by), 0.34, 0.2, fc=state_colors[s], ec=state_bord[s], lw=0.6))
                    else:
                        ax.add_patch(plt.Rectangle((sx, by), 0.34, 0.2, fc="#ffffff", ec="#c9cfd8", lw=0.6, ls=(0, (2, 2))))
    _label(ax, 6, 0.28, "ZeRO-3 的参数片在用到某一层时临时 all-gather 成整份、用完释放；所以它多出一次反向前的 all-gather，通信 2N → 3N。", 8.0)
    save(fig, "parallelism-zero-stages")


def zero3_layer_step():
    fig, ax = _canvas(7.6, 7.0, (0, 14), (0, 10.4))
    _label(ax, 7, 10.05, "ZeRO-3 一层在 4 张卡上的一个 step：四种状态都切 4 段，通信三次", 11)
    legend = [("P 参数", BLUE), ("G 梯度", GREEN), ("O 优化器状态（主参数 + m + v）", PURPLE), ("A 激活（按 batch 切）", ORANGE)]
    lx = 0.55
    for label, col in legend:
        ax.add_patch(plt.Rectangle((lx, 9.55), 0.16, 0.13, fc=col, ec="#ffffff"))
        _label(ax, lx + 0.22, 9.615, label, 6.9, ha="left")
        lx += 0.4 + len(label) * 0.14
    _label(ax, 0.62, 9.18, "浅色 = 临时（AG 拼出的完整参数 / 未求和的本卡梯度）；红框 = 本步刚更新", 7.4, ha="left")
    phases = ["常驻（step 开始）", "① 前向前 AG 参数", "② 前向算完", "③ 反向前再 AG 参数", "④ 反向算完", "⑤ RS 梯度", "⑥ optimizer.step()"]
    notes = ["梯度、激活都还没有", "拼出本层完整参数", "完整参数释放，\n本卡激活留给反向", "前向后释放了，\n只能再拼一次", "激活用完释放；得到本卡\n数据的完整梯度（未求和）", "各卡梯度按段求和，\n每卡只留自己那段", "用自己那段梯度更新\n自己那段优化器状态和参数"]
    x0, colw, cardw, rowh, ytop = 3.35, 1.95, 1.8, 1.06, 8.0
    state_rows = ["P", "G", "O", "A"]
    palette = [C["blue"], C["green"], C["purple"], C["orange"]]
    for phase, (title, note) in enumerate(zip(phases, notes)):
        y = ytop - phase * rowh
        _label(ax, 3.15, y + 0.14, title, 7.0, ha="right")
        _label(ax, 3.15, y - 0.3, note, 6.0, ha="right", color="#444444")
        for gpu in range(4):
            x = x0 + gpu * colw
            if phase == 0:
                _label(ax, x + cardw / 2, y + 0.62, f"GPU {gpu}", 7.2)
            ax.add_patch(plt.Rectangle((x, y - 0.52), cardw, 1.0, fc="#ffffff", ec="#d5d9df", lw=0.55))
            states = [
                ("own", "none", "own", "none"), ("full", "none", "own", "none"),
                ("own", "none", "own", "own"), ("full", "none", "own", "own"),
                ("own", "fullg", "own", "none"), ("own", "own", "own", "none"),
                ("updated", "own", "updated", "none"),
            ][phase]
            for si, state in enumerate(states):
                ry = y + 0.32 - si * 0.22
                _label(ax, x + 0.07, ry + 0.05, state_rows[si], 5.6, ha="left", color="#555555")
                if state == "own" and si == 3:
                    ax.add_patch(plt.Rectangle((x + 0.24, ry - 0.02), cardw - 0.4, 0.14, fc=ORANGE, ec=C["orange"], lw=0.45))
                    ax.text(x + 0.24 + (cardw - 0.4) / 2, ry + 0.05, f"batch {gpu}", ha="center", va="center", fontsize=4.8, color="#222222")
                    continue
                for segment in range(4):
                    owned = segment == gpu
                    if state == "none" or (state in {"own", "updated"} and not owned):
                        fc, ec = "#ffffff", "#cbd1d9"
                    elif state == "full":
                        fc, ec = "#c6dcf2", C["blue"]
                    elif state == "fullg":
                        fc, ec = "#bfe3c8", C["green"]
                    else:
                        fc, ec = palette[si], palette[si]
                    if state == "updated" and owned:
                        ec = C["red"]
                    seg_w = (cardw - 0.44) / 4
                    ax.add_patch(plt.Rectangle((x + 0.24 + segment * seg_w, ry - 0.02), seg_w - 0.015, 0.14,
                                               fc=fc, ec=ec, lw=1.1 if state == "updated" and owned else 0.45,
                                               ls="solid" if state != "none" else (0, (2, 2))))
    _label(ax, 11.7, ytop + 0.62, "通信（每卡）", 7.4)
    for phase, (txt, col) in {1: ("① AG 参数 ≈ N", "#c6dcf2"), 3: ("③ AG 参数 ≈ N", "#c6dcf2"), 5: ("⑤ RS 梯度 ≈ N", "#bfe3c8")}.items():
        y = ytop - phase * rowh
        _card(ax, 11.45, y - 0.05, 2.2, 0.42, txt, fc=col, ec="#8899aa", size=7.0, lw=0.5)
    _label(ax, 7, 0.98, "一个 step 的通信量（N = 参数个数，按每卡收发的元素算）", 8.4)
    _card(ax, 0.9, 0.32, 1.7, 0.44, "RS 梯度 ≈ N", fc=GREEN, ec=C["green"], size=6.4)
    _card(ax, 2.75, 0.32, 2.75, 0.44, "step 后 AG 参数 ≈ N", fc=ORANGE, ec=C["orange"], size=6.4)
    _card(ax, 5.65, 0.32, 2.05, 0.44, "= 2N（ZeRO-1/2）", fc="#f0f1f3", ec="#9aa3ad", size=6.4)
    _card(ax, 8.05, 0.32, 2.75, 0.44, "①/③ AG + ⑤ RS ≈ N", fc=BLUE, ec=C["blue"], size=6.4)
    _card(ax, 10.95, 0.32, 2.0, 0.44, "= 3N（ZeRO-3）", fc="#f0f1f3", ec="#9aa3ad", size=6.4)
    _label(ax, 7, -0.06, "ZeRO-1/2 每卡常驻完整参数，step 后要 AG 一次把别人更新的部分拼回来；ZeRO-3 本来只存自己那段，这次 AG 挪到了下一 step 的 ①。", 7.6)
    save(fig, "parallelism-zero3-layer-step")


def fsdp1_vs_fsdp2():
    fig, ax = _canvas(7.6, 5.4, (0, 12), (0, 8.1))
    _label(ax, 6, 7.72, "一个 wrap 单元：W 形状 [4, 4]、b 形状 [4]，N_d = 2（蓝 = GPU 0，橙 = GPU 1）", 10)
    _label(ax, 0.55, 6.95, "FSDP1：拍平拼接成一个 FlatParameter，再均分", 9.3, ha="left")
    # FSDP1 flat strip of 20 cells
    x0, cw0 = 0.62, 0.4
    names = [f"W{i}{j}" for i in range(4) for j in range(4)] + [f"b{i}" for i in range(4)]
    for i, nm in enumerate(names):
        fc = BLUE if i < 10 else ORANGE
        ax.add_patch(plt.Rectangle((x0 + i * cw0, 6.05), cw0 - 0.025, 0.6, fc=fc, ec="#ffffff", lw=0.5))
        ax.text(x0 + i * cw0 + cw0 / 2, 6.35, nm, ha="center", va="center", fontsize=5.6, color="#222222")
    ax.plot([x0 + 10 * cw0 - 0.02, x0 + 10 * cw0 - 0.02], [5.95, 6.78], color=C["red"], lw=1.5)
    _label(ax, 3.35, 5.62, "GPU 0：W 的 10 个元素", 7.8)
    _label(ax, 5.55, 5.62, "从第 10 个元素切开", 7.8)
    _label(ax, 8.2, 5.62, "GPU 1：W 的 6 个 + 整个 b", 7.8)
    _label(ax, 6, 5.18, "分片是“一维大张量的第 k 段”：W 被切在第 2 行中间，b 整个落在 GPU 1，哪一段都不是一个完整参数。", 8.0)
    _label(ax, 0.55, 4.45, "FSDP2：每个参数各自沿第 0 维切，分片是 Shard(0) 的 DTensor", 9.3, ha="left")
    # FSDP2: W as 4x4 grid + b column
    x1, cs = 0.62, 0.4
    for i in range(4):
        for j in range(4):
            fc = BLUE if i < 2 else ORANGE
            ax.add_patch(plt.Rectangle((x1 + j * cs, 3.55 - i * cs), cs - 0.02, cs - 0.02, fc=fc, ec="#ffffff", lw=0.5))
            ax.text(x1 + j * cs + cs / 2, 3.55 - i * cs + cs / 2, f"W{i}{j}", ha="center", va="center", fontsize=5.4, color="#222222")
    _label(ax, x1 + 4 * cs + 0.12, 2.78, "W [4, 4]", 7.6, ha="left")
    x2 = 2.62
    for i in range(4):
        fc = BLUE if i < 2 else ORANGE
        ax.add_patch(plt.Rectangle((x2, 3.55 - i * cs), cs - 0.02, cs - 0.02, fc=fc, ec="#ffffff", lw=0.5))
        ax.text(x2 + cs / 2, 3.55 - i * cs + cs / 2, f"b{i}", ha="center", va="center", fontsize=5.4, color="#222222")
    _label(ax, x2 + cs + 0.12, 2.78, "b [4]", 7.6, ha="left")
    _label(ax, 4.05, 3.55, "GPU 0：W[0:2, :]（形状 [2, 4]）+ b[0:2]\nGPU 1：W[2:4, :]（形状 [2, 4]）+ b[2:4]", 8.0, ha="left", va="center")
    _label(ax, 4.05, 2.42, "每个分片仍是“某个参数的第几行”，可以单独设 dtype、单独冻结，\n也能再叠一维 TP 的 Shard(1)；checkpoint 按参数名重分片", 8.0, ha="left", va="center")
    save(fig, "parallelism-fsdp1-vs-fsdp2")


def hsdp_grid():
    fig, ax = _canvas(7.6, 4.8, (0, 12), (0, 7.3))
    _label(ax, 6, 6.95, "HSDP：N_d = 16 张卡排成 N_r × N_s = 2 × 8 的网格", 11.5)
    _label(ax, 6, 6.52, "placement = (Replicate(), Shard(0))", 8.8)
    x0, cw, cardw = 2.25, 1.06, 0.95
    _label(ax, 6.3, 5.98, "→ 分片组（一个节点 8 卡）：Shard(0)，AG + AG + RS ≈ 3N，走 NVLink", 8.2)
    _arrow(ax, (x0, 5.72), (x0 + 7*cw + cardw, 5.72), color="#555555", lw=1.0, ms=9)
    for row in range(2):
        y = 4.55 - row * 1.55
        _label(ax, 1.98, y + 0.42, f"副本 {row}\n（节点 {row}）", 8.1, ha="right")
        for i in range(8):
            _card(ax, x0 + i*cw, y, cardw, 0.76, f"卡 {row*8+i}\n段 {i}", fc=BLUE, ec=C["blue"], size=7.4)
            if row == 0:
                _arrow(ax, (x0 + i*cw + cardw/2, y - 0.08), (x0 + i*cw + cardw/2, y - 0.7), color=C["red"], lw=1.15, ms=8)
    _label(ax, 11.55, 3.75, "复制组\n{i, i+8}", 7.6, rotation=90, va="center")
    _label(ax, 6, 1.72, "竖向（红）：Replicate()，同一段梯度在两个副本之间 all-reduce，每卡约 2N/N_s，走 IB。", 8.2)
    _label(ax, 6, 1.25, "横向（蓝）：ZeRO-3 的全部 3N 通信留在节点内 NVLink 上。", 8.2)
    _label(ax, 6, 0.78, "代价：显存只按 N_s 分，而非 N_d 分；每卡状态为 16N/N_s。", 8.2)
    save(fig, "parallelism-hsdp-grid")


def tp_column_row_split():
    fig, ax = _canvas(7.6, 5.6, (0, 12), (0, 8.4))
    _label(ax, 6, 7.95, "TP：列切与行切配对，让层内矩阵变小", 12.5)
    _label(ax, 6, 7.5, "t=4 张卡；每卡算 1/t 的矩阵乘，行切后的部分和要相加", 9)
    _label(ax, 6, 6.75, "列切（column-parallel）：A（h × h'）按列切成 [A1 | A2 | A3 | A4]，给 4 张卡", 10.2)
    _card(ax, 0.7, 5.3, 1.5, 0.85, "X\n完整副本", fc=BLUE, ec=C["blue"], size=8.5)
    _label(ax, 1.45, 4.88, "X（tokens × h）\n输入：不需通信（每卡都有）", 7.0)
    _arrow(ax, (2.35, 5.72), (3.0, 5.72))
    for i in range(4):
        _card(ax, 3.1 + i * 1.05, 5.3, 0.88, 0.85, f"A{i+1}\n列块", fc=ORANGE, ec=C["orange"], size=8)
        _card(ax, 3.1 + i * 1.05, 4.18, 0.88, 0.66, f"Y{i+1}", fc=GREEN, ec=C["green"], size=9)
        _arrow(ax, (3.54 + i * 1.05, 5.24), (3.54 + i * 1.05, 4.9), lw=0.95)
    _label(ax, 8.9, 5.9, "Yi = X Ai 是 Y 的第 i 列块\n每卡持 tokens × h'/t\n输出：Y 按列分布，\n恰好是列切层的输出布局", 7.7, ha="left")
    _label(ax, 6, 3.55, "行切（row-parallel）：A 按行切成 [A1 ; A2 ; A3 ; A4]，输入 X 必须按列切成 Xi", 10.2)
    for i in range(4):
        _card(ax, 0.9 + i * 1.15, 2.38, 0.98, 0.78, f"X{i+1}", fc=BLUE, ec=C["blue"], size=9)
        _card(ax, 0.9 + i * 1.15, 1.32, 0.98, 0.78, f"A{i+1}\n行块", fc=ORANGE, ec=C["orange"], size=8)
        _arrow(ax, (1.39 + i * 1.15, 2.32), (1.39 + i * 1.15, 2.14), lw=0.95)
    _arrow(ax, (5.8, 1.7), (6.6, 1.7), color=C["red"], lw=1.35)
    _card(ax, 6.8, 1.3, 2.4, 0.8, "Σ Xi Ai\nall-reduce", fc=RED, ec=C["red"], size=8.5)
    _label(ax, 9.62, 1.7, "X1A1+X2A2+X3A3+X4A4\n一次 all-reduce → 完整 Y", 7.8, ha="left")
    _label(ax, 6, 0.85, "行切每卡算一个部分和 XiAi（tokens × h'）；输出：4 个部分和要相加；输入：X 按列分布即可。", 7.8)
    _label(ax, 6, 0.45, "配对：MLP 第一层列切、第二层行切；中间 GELU 逐元素，不需 gathering 整个向量；一个 MLP 块只在末尾 all-reduce 一次。", 8.0)
    save(fig, "parallelism-tp-column-row-split")


def tp_conjugate_fg():
    fig, ax = _canvas(7.6, 4.9, (0, 12), (0, 7.4))
    _label(ax, 6, 6.95, "TP 的 f / g：前向与反向互为共轭", 12.5)
    _label(ax, 6, 6.48, "前向：f 是恒等（X 每卡都有），g 是 all-reduce（把 t 个部分和加起来）", 8.8)
    xs = [0.55, 2.35, 4.35, 6.35, 8.35, 10.45]
    labels = ["X\n完整", "f：恒等", "[列切 W1i]\nGELU", "[行切 W2i]", "g：all-reduce", "Y\n完整"]
    colors = [BLUE, "#ffffff", ORANGE, ORANGE, GREEN, GREEN]
    widths = [1.3, 1.5, 1.5, 1.5, 1.55, 1.05]
    for x, lab, fc, w in zip(xs, labels, colors, widths):
        _card(ax, x, 4.92, w, 1.0, lab, fc=fc, ec="#aeb8c4", size=8)
    for i in range(5):
        _arrow(ax, (xs[i] + widths[i] + 0.04, 5.42), (xs[i+1] - 0.06, 5.42))
    _label(ax, 6, 4.16, "反向：g 变成恒等（dY 每卡都有，各取自己那块），f 变成 all-reduce（各卡对 X 的部分梯度求和）", 8.8)
    rev = ["dX\n完整", "f：all-reduce", "[W1i^T]\nGELU′", "[W2i^T]", "g：恒等", "dY\n完整"]
    for x, lab, fc, w in zip(xs, rev, [GREEN, RED, ORANGE, ORANGE, "#ffffff", BLUE], widths):
        _card(ax, x, 2.55, w, 1.0, lab, fc=fc, ec="#aeb8c4", size=8)
    for i in range(5):
        _arrow(ax, (xs[i+1] - 0.06, 3.08), (xs[i] + widths[i] + 0.04, 3.08))
    _label(ax, 3.8, 4.55, "各卡持 X 的完整副本", 7.6)
    _label(ax, 7.35, 4.55, "各卡持部分和 Σ_i", 7.6)
    _label(ax, 3.8, 2.22, "各卡对 X 的部分梯度相加", 7.6)
    _label(ax, 9.6, 2.22, "各卡直接取自己那块", 7.6)
    _label(ax, 6, 1.72, "f 与 g 互为共轭：前向的 all-reduce 在块尾、反向的在块头，一个 MLP 块前向 + 反向共 2 次；注意力块同理，每层 4 次。", 8.3)
    _label(ax, 6, 1.2, "各卡都持完整 X；行切 W2 后每卡得到部分和 Σ_i，g 在前向末尾 all-reduce 求完整 Y；有 SP 时改写成 AG + RS。", 8.0)
    save(fig, "parallelism-tp-conjugate-fg")


def ring_attention():
    fig, ax = _canvas(7.6, 4.9, (0, 12), (0, 7.4))
    _label(ax, 6, 7.0, "Ring Attention，N_c = 4：每张卡固定持有自己的 Q 块，K/V 块沿环传递 3 步，每步算一个局部注意力并用 online-softmax 合并", 10.2)
    pts = [(0.9, 5.75), (3.4, 5.75), (3.4, 4.35), (0.9, 4.35)]
    for i, (x, y) in enumerate(pts):
        _card(ax, x, y, 1.75, 0.5, f"卡 {i}：Q{i}", fc=BLUE, ec=C["blue"], size=6.2)
    _arrow(ax, (2.2, 6.0), (3.3, 6.0), color=C["orange"], lw=1.5)
    _arrow(ax, (4.02, 5.68), (4.02, 4.93), color=C["orange"], lw=1.5)
    _arrow(ax, (3.3, 4.6), (2.2, 4.6), color=C["orange"], lw=1.5)
    _arrow(ax, (0.55, 4.93), (0.55, 5.68), color=C["orange"], lw=1.5)
    _label(ax, 2.6, 5.3, "K/V 块\n顺时针 send/recv", 8.3)
    _label(ax, 9.4, 6.62, "每张卡每一步用的 K/V 块（颜色＝块来自哪张卡）", 8.7)
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    x0, y0, cw, rh = 7.0, 4.3, 1.3, 0.46
    for j in range(4):
        _label(ax, x0 + j * cw + cw / 2, 6.28, f"步 {j}", 7.8)
    for r in range(4):
        y = y0 + (3-r)*rh
        _label(ax, 6.45, y + rh/2, f"卡 {r}", 7.4, ha="right")
        order = [(r - j) % 4 for j in range(4)]
        for j, b in enumerate(order):
            _card(ax, x0 + j*cw, y, cw - 0.05, rh - 0.04, f"K/V{b}", fc=colors[b], ec="#ffffff", size=6.0, lw=0.4)
    _label(ax, 6, 2.9, "每一列的四个块各不相同：同一时刻四张卡在算四个不同的 K/V 块，链路上同时有四个块在传。", 8.2)
    _label(ax, 6, 2.4, "算第 j 块时收第 j+1 块——只要一块的注意力计算时间不小于一块的传输时间，通信全部藏在计算后面。", 8.2)
    _label(ax, 6, 1.9, "每卡每层收 N_c - 1 个块，总量 ≈ 整个序列的 K/V，与 N_c 无关；toy 与单进程最大误差约 1e-7。", 8.2)
    save(fig, "parallelism-ring-attention")


def ulysses_all_to_all():
    fig, ax = _canvas(7.6, 4.8, (0, 12), (0, 7.2))
    _label(ax, 6, 6.85, "Ulysses，N_c = 4：all-to-all 把按序列切变成按头切", 12)
    _label(ax, 2.75, 6.35, "按序列切：卡 i 持第 i 段 token 的全部 4 个头", 8.5)
    _label(ax, 2.75, 6.02, "每行一张卡：token 段 s/4，4 个头 → [s/4, b, a·d]", 7.3)
    _label(ax, 9.1, 6.35, "按头切：卡 i 持全部 token 的第 i 个头", 8.5)
    _label(ax, 9.1, 6.02, "每行一张卡：全部 s 个 token，1 个头 → [s, b, a/4·d]", 7.3)
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    x0, y0, cw, rh = 0.85, 3.55, 0.82, 0.48
    _label(ax, 0.55, 5.62, "头 →", 7.6, ha="right")
    for j in range(4):
        _label(ax, x0 + j*cw + cw/2, 5.62, f"h{j}", 7.6)
    for r in range(4):
        y = y0 + (3-r)*rh
        _label(ax, 0.55, y+rh/2, f"卡 {r}", 7.5, ha="right")
        for c in range(4):
            _card(ax, x0 + c*cw, y, cw-0.04, rh-0.04, f"s{r}", fc=colors[c], ec="#ffffff", size=7.2, lw=0.4)
    _arrow(ax, (4.3, 4.65), (5.55, 4.65), color=C["red"], lw=1.8, ms=12)
    _label(ax, 5.27, 4.85, "all-to-all\n（Q、K、V 各一次）", 7.0)
    x1 = 6.45
    _label(ax, 9.98, 5.62, "token 段 →", 7.6, ha="left")
    for j in range(4):
        _label(ax, x1 + j*cw + cw/2, 5.62, f"s{j}", 7.6)
    for r in range(4):
        y = y0 + (3-r)*rh
        _label(ax, 9.98, y+rh/2, f"卡 {r}", 7.5, ha="left")
        for c in range(4):
            _card(ax, x1 + c*cw, y, cw-0.04, rh-0.04, f"h{r}", fc=colors[r], ec="#ffffff", size=7.2, lw=0.4)
    _arrow(ax, (5.55, 3.05), (4.3, 3.05), color=C["red"], lw=1.8, ms=12)
    _label(ax, 5.27, 2.85, "all-to-all\n（O 一次）", 7.0)
    _label(ax, 6, 2.0, "颜色表示头；all-to-all 就是把 4×4 分布表转置：卡 i 把自己的第 j 个头发给卡 j。", 8.1)
    _label(ax, 6, 1.55, "前向 4 次、反向 4 次，每次每卡载荷 (s/N_c)·b·h；N_c 不能超过可分配的 K/V 头数。", 8.1)
    save(fig, "parallelism-ulysses-all-to-all")


def _pipeline_events(kind, p, m, tf=1.0, tb=2.0):
    from types import SimpleNamespace
    from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleGPipe
    cls = ScheduleGPipe if kind == "gpipe" else Schedule1F1B
    order = cls(SimpleNamespace(num_stages=p), n_microbatches=m).pipeline_order
    result = []
    for stage in range(p):
        clock, events = 0.0, []
        for slot, action in enumerate(order[stage]):
            if action is None:
                continue
            typ = str(action.computation_type).split(".")[-1]
            if typ == "REDUCE_GRAD":
                continue
            op = "F" if typ == "FORWARD" else "B"
            start = max(clock, slot * tf)
            end = start + (tf if op == "F" else tb)
            events.append((op, action.microbatch_index % m, start, end))
            clock = end
        result.append(events)
    return result


def _draw_schedule(ax, rows, x0, y_top, unit, rh, label_x):
    for s, row in enumerate(rows):
        y = y_top - s * (rh + 0.06)
        _label(ax, label_x, y + rh / 2, f"stage {s}", 7.6, ha="right")
        cursor = 0.0
        for op, mb, start, end in row:
            if start > cursor + 0.01:
                ax.add_patch(plt.Rectangle((x0 + cursor*unit, y), (start-cursor)*unit-0.015, rh-0.05,
                                           fc="#f0f1f3", ec="#ffffff", lw=0.5))
            x, w = x0 + start * unit, (end - start) * unit - 0.025
            fc = "#7fb2e5" if op == "F" else "#9ad29f"
            ax.add_patch(plt.Rectangle((x, y), w, rh - 0.05, fc=fc, ec="#ffffff", lw=0.5))
            ax.text(x + w/2, y + (rh-0.05)/2, str(mb), ha="center", va="center", fontsize=5.8, color="#222222")
            cursor = end
    return y_top - 3 * (rh + 0.06)


def pipeline_gpipe_vs_1f1b():
    fig, ax = _canvas(7.6, 5.7, (0, 12), (0, 8.55))
    _label(ax, 6, 8.2, "流水线调度：GPipe 先填满再排空，1F1B 尽早消费激活", 11.5)
    _label(ax, 6, 7.8, "横轴时间（反向 t_b = 2 t_f）；蓝 = 前向 F，绿 = 反向 B，灰 = 气泡；格内是 micro-batch 编号", 8.1)
    _label(ax, 1.6, 7.25, "GPipe：p = 4，m = 4", 9.2, ha="left")
    _draw_schedule(ax, _pipeline_events("gpipe", 4, 4), 1.5, 6.72, 0.3, 0.38, 1.32)
    _label(ax, 6, 5.05, "stage 0 做完 4 个前向后空等，直到反向从 stage 3 一路传回来；前向填充 p-1 格 + 反向排空 p-1 格，气泡 (p-1)(t_f + t_b)。", 7.7)
    _label(ax, 1.6, 4.48, "1F1B：p = 4，m = 8", 9.2, ha="left")
    _draw_schedule(ax, _pipeline_events("1f1b", 4, 8), 1.5, 3.95, 0.3, 0.38, 1.32)
    _label(ax, 6, 2.25, "warm-up（stage s 先做 p-s 个前向）后交替一前向、一反向，最后排完 trailing B；两头气泡不变。", 7.8)
    _label(ax, 6, 1.82, "在途 micro-batch ≤ p，激活显存 O(m) → O(p)。单元格位置来自 PyTorch Schedule1F1B.pipeline_order；GPipe 来自 ScheduleGPipe。", 7.8)
    save(fig, "parallelism-pipeline-gpipe-vs-1f1b")


def ep_dispatch_combine():
    fig, ax = _canvas(7.6, 5.5, (0, 12), (0, 8.2))
    _label(ax, 6, 7.82, "EP，N_e = 2、E = 4、k = 1：token 留在自己的卡上算注意力，进 MoE 层时被 all-to-all 送到专家所在的卡，算完再送回", 9.6)
    _label(ax, 6, 7.35, "路由后（token 仍在原卡）：每个 token 上标注它选中的专家", 8.5)
    ecol = {"E0": BLUE, "E1": RED, "E2": GREEN, "E3": ORANGE}
    ebor = {"E0": C["blue"], "E1": C["red"], "E2": C["green"], "E3": C["orange"]}
    # top two dashed card groups
    groups = [(0.4, "卡 0 · 持专家 E0、E1", [("t0", "E2"), ("t1", "E0"), ("t2", "E3")]),
              (6.1, "卡 1 · 持专家 E2、E3", [("t3", "E1"), ("t4", "E2"), ("t5", "E0")])]
    tok_pos = {}
    for gx, title, toks in groups:
        ax.add_patch(plt.Rectangle((gx, 5.55), 5.5, 1.28, fc="#fbfdff", ec="#9aa6b4", lw=0.9, ls=(0, (4, 3)), zorder=0))
        _label(ax, gx + 2.75, 6.6, title, 8.6)
        for i, (tk, ex) in enumerate(toks):
            x = gx + 0.35 + i * 1.72
            p = _card(ax, x, 5.68, 1.5, 0.6, f"{tk} → {ex}", fc=ecol[ex], ec=ebor[ex], size=8.6)
            tok_pos[tk] = (x + 0.75, 5.68, gx + 2.75, ex)
    _label(ax, 1.15, 4.95, "dispatch\nall-to-all", 8.3)
    # bottom two dashed expert groups
    bottoms = [(0.4, "卡 0 · 收到 3 个 token，按专家分组后各算一个小 GEMM",
                [("E0", "{ t1, t5 }", "t1 本卡，t5 来自卡 1"), ("E1", "{ t3 }", "来自卡 1")]),
               (6.1, "卡 1 · 收到 3 个 token",
                [("E2", "{ t0, t4 }", "t0 来自卡 0，t4 本卡"), ("E3", "{ t2 }", "来自卡 0")])]
    exp_pos = {}
    for gx, title, exps in bottoms:
        ax.add_patch(plt.Rectangle((gx, 2.5), 5.5, 1.72, fc="#fbfdff", ec="#9aa6b4", lw=0.9, ls=(0, (4, 3)), zorder=0))
        _label(ax, gx + 2.75, 3.96, title, 8.3)
        for i, (ex, ts, src) in enumerate(exps):
            x = gx + 0.28 + i * 2.62
            _card(ax, x, 2.62, 2.45, 1.05, f"{ex} ← {ts}\n{src}", fc=ecol[ex], ec=ebor[ex], size=7.4)
            exp_pos[ex] = (x + 1.225, 3.67)
    # dispatch arrows: red cross-card, grey same-card
    routes = [("t0", "E2"), ("t1", "E0"), ("t2", "E3"), ("t3", "E1"), ("t4", "E2"), ("t5", "E0")]
    for tk, ex in routes:
        x, y0, gcx, _ = tok_pos[tk]
        ex_cx, ey = exp_pos[ex]
        src_card = 0 if tk in {"t0", "t1", "t2"} else 1
        dst_card = 0 if ex in {"E0", "E1"} else 1
        same = src_card == dst_card
        color, rad = ("#9aa2ad", 0.12) if same else (C["red"], 0.28 if abs(ex_cx - x) > 3 else -0.22)
        ax.annotate("", xy=(ex_cx, ey), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.15,
                                    connectionstyle=f"arc3,rad={rad}", mutation_scale=10))
    # combine: dashed arrow back up on the right
    ax.annotate("", xy=(11.35, 6.95), xytext=(11.35, 2.3),
                arrowprops=dict(arrowstyle="->", color="#7a8794", lw=1.2, ls=(0, (4, 3)),
                                connectionstyle="arc3,rad=-0.25", mutation_scale=11))
    _label(ax, 11.62, 4.6, "combine：\n原路送回", 8.0, rotation=90)
    _label(ax, 6, 1.98, "dispatch 矩阵（卡 i 发给卡 j 的 token 数）：卡 0 → [1, 2]，卡 1 → [2, 1]；每卡发出 2 个、收到 3 个。", 8.0)
    _label(ax, 6, 1.55, "k = 2 时每个 token 复制两份、各去一个专家，回来后按路由权重加权相加。", 8.0)
    _label(ax, 6, 1.12, "反向走同一条路、方向相反：combine 的反向是 dispatch 形状的 all-to-all，反之亦然——所以每层 2 + 2 次。", 8.0)
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
        ax.plot(xs, [(p - 1) / m for m in xs], color=color, lw=1.7, label=f"p={p}：v=1，(p-1)/m")
        ax.plot(xs, [(p - 1) / (2 * m) for m in xs], color=color, lw=1.3, ls="--", label=f"p={p}：v=2")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8, title="实线 v=1　虚线 v=2（气泡约减半，通信约翻倍）", title_fontsize=8)
    save(fig, "parallelism-bubble-rate")


def zero_stages():
    fig, ax = _canvas(7.6, 5.1, (0, 12), (0, 7.8))
    _label(ax, 6, 7.48, "四级 ZeRO：N_d = 4 时，每卡常驻的参数 P、梯度 G、优化器状态 O", 11.2)
    _label(ax, 6, 7.05, "P = 2N、G = 2N、O = 12N；实色 = 本卡持有，虚线 = 这一片在别的卡上", 8.1)
    rows = [
        ("DP (ZeRO-0)", "每卡 16N", "all-reduce 2N", (2, 2, 12), False),
        ("ZeRO-1", "每卡 4N + 12N/N_d", "RS + AG = 2N", (2, 2, 3), True),
        ("ZeRO-2", "每卡 2N + 14N/N_d", "RS + AG = 2N；梯度即时释放", (2, 0.5, 3), True),
        ("ZeRO-3 / FSDP", "每卡 16N/N_d", "AG + AG + RS = 3N", (0.5, 0.5, 3), True),
    ]
    cols = [BLUE, GREEN, PURPLE]
    edges = [C["blue"], C["green"], C["purple"]]
    x0, cardw, barw, unit = 2.9, 2.1, 1.55, 1.55 / 16
    for k, (name, mem, comm, sizes, shard) in enumerate(rows):
        y = 5.78 - k * 1.38
        _label(ax, 0.48, y + 0.76, name, 8.8, ha="left")
        _label(ax, 0.48, y + 0.45, mem, 7.6, ha="left")
        _label(ax, 0.48, y + 0.17, f"通信 {comm}", 7.3, ha="left")
        for gpu in range(4):
            x = x0 + gpu * cardw
            ax.add_patch(plt.Rectangle((x, y), 1.88, 0.98, fc="#fbfbfc", ec="#aeb8c4", lw=0.65))
            _label(ax, x + 0.12, y + 0.81, f"GPU {gpu}", 7.1, ha="left")
            bx = x + 0.18
            by = y + 0.42
            offset = 0.0
            for si, size in enumerate(sizes):
                state_width = size * unit
                chunks = 4 if size in (2, 12) else 4
                chunk_w = state_width / chunks
                for chunk in range(chunks):
                    owned = not shard or chunk == gpu
                    face = cols[si] if owned else "#ffffff"
                    edge = edges[si] if owned else "#c8ced6"
                    ls = "solid" if owned else (0, (2, 2))
                    ax.add_patch(plt.Rectangle((bx + offset + chunk * chunk_w, by), chunk_w - 0.012, 0.19,
                                               fc=face, ec=edge, lw=0.55, ls=ls))
                offset += state_width
            _label(ax, bx + 0.1, y + 0.23, "P", 6.8)
            _label(ax, bx + 0.36, y + 0.23, "G", 6.8)
            _label(ax, bx + 0.62, y + 0.23, "O", 6.8)
    _label(ax, 6, 0.18, "ZeRO-3 的参数片只在用到某一层时临时 all-gather 成整份，用完释放；因此比 ZeRO-1/2 多一次反向前 AG：2N → 3N。", 7.8)
    save(fig, "parallelism-zero-stages")


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
