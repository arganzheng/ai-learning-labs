"""《容错与弹性》配图：_plot.py 风格，collide()=0。

F1 有效训练时间五项（§二.4 表 B→C，16K 卡）
F2 三种恢复方式的时间线（§四.4 / §五.4 / §六；区间取中点）
F3 torchft 副本组三格机制（§六.1–4）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import C, box, plt, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#dce9f7"
ORANGE = "#fae6d6"
GREEN = "#dcefe2"
PURPLE = "#eee1f3"
RED = "#f5dede"
GRAY = "#e8e8e8"


def _canvas(w=7.6, h=4.2, xlim=(0, 12), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _label(ax, x, y, s, size=10, color="#111111", ha="center", va="center",
           in_box=None, weight=None, **kw):
    return text(ax, x, y, s, fontsize=size, color=color, ha=ha, va=va,
                in_box=in_box, fontweight=weight, **kw)


def _card(ax, x, y, w, h, s="", fc="white", ec="#777777", size=10, lw=0.9):
    p = box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw)
    if s:
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center",
             fontsize=size)
    return p


def _arrow(ax, a, b, color="#555555", lw=1.2, ms=11):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                mutation_scale=ms))


def five_terms():
    """F1：损失时间五项，16K 卡场景 B(损失 13.3%) → C(损失 5.5%)。loss-only。"""
    fig, ax = _canvas(7.6, 4.2, (0, 12), (0, 6.4))
    _label(ax, 6, 6.1, "一次故障的损失时间五项（16K 卡，§二.4 场景 B → C）",
           12, weight="bold")
    segs = [("T_d", BLUE, C["blue"]), ("T_r", GREEN, C["green"]),
            ("T_l", ORANGE, C["orange"]), ("τ/2", PURPLE, C["purple"]),
            ("δ/τ", RED, C["red"])]
    rows = [
        ("B：T_d=600s", [5.5, 2.7, 1.1, 2.1, 1.9], "损失 13.3%，G 86.7%"),
        ("C：T_d=60s", [0.5, 0.3, 0.5, 2.1, 2.0], "损失 5.5%，G 94.5%"),
    ]
    def mx(v):
        return 1.9 + v / 14 * 8.8
    # 图例行（色块→文字间距 ≥0.25u）
    lx = 1.9
    names = {"T_d": "检测 T_d", "T_r": "重启 T_r", "T_l": "加载 T_l",
             "τ/2": "回退 τ/2", "δ/τ": "保存 δ/τ"}
    for sname, fc, ec in segs:
        _card(ax, lx, 5.32, 0.28, 0.28, "", fc=fc, ec=ec, lw=0.6)
        _label(ax, lx + 0.62, 5.46, names[sname], 9.5, ha="left")
        lx += 0.62 + len(names[sname]) * 0.16 + 0.45
    ys = [4.1, 2.3]
    for r, (name, vals, tot) in enumerate(rows):
        y = ys[r]
        _label(ax, 1.55, y + 0.28, name, 9.5, ha="right", color="#333333")
        x = mx(0)
        narrow = []
        for (sname, fc, ec), v in zip(segs, vals):
            w = v / 14 * 8.8
            p = _card(ax, x, y, w, 0.56, "", fc=fc, ec=ec, lw=0.6)
            if v >= 2.5:
                _label(ax, x + w / 2, y + 0.28, f"{sname} {v}%", 9.5, in_box=p)
            else:
                narrow.append((x + w / 2, f"{sname} {v}%"))
            x += w
        # 窄段：B 行上方引线；C 行全窄，改为条下一行
        if narrow and r == 0:
            lv_h = {0: 0.98}
            groups = [narrow]
            for lv, grp in enumerate(groups):
                pos = [c for c, _ in grp]
                for _ in range(20):
                    ok = True
                    for i in range(1, len(pos)):
                        if pos[i] < pos[i - 1] + 1.3:
                            pos[i] = pos[i - 1] + 1.3
                            ok = False
                    if ok:
                        break
                for (cx, lab), px in zip(grp, pos):
                    ax.plot([cx, px], [y + 0.56, y + lv_h[lv] - 0.12],
                            color="#999999", lw=0.7)
                    _label(ax, px, y + lv_h[lv], lab, 9)
        _label(ax, x + 0.3, y + 0.28, tot, 9.5, ha="left", weight="bold")
        if r == 1:
            _label(ax, mx(0) + 0.05, y - 0.42,
                   "T_d 0.5% · T_r 0.3% · T_l 0.5% · τ/2 2.1% · δ/τ 2.0%",
                   9, ha="left", color="#444444")
            _label(ax, mx(0) + 0.05, y - 0.78,
                   "T_d 600 s → 60 s：5.5% → 0.5%（§三.3 心跳检测）",
                   9, ha="left", color="#444444")
    for gx in range(0, 15, 2):
        ax.plot([mx(gx), mx(gx)], [0.95, 4.75], color="#eeeeee", lw=0.6, zorder=0)
        _label(ax, mx(gx), 0.8, f"{gx}", 9, color="#666666")
    _label(ax, 6, 0.42, "横轴：损失时间占墙钟时间 %（与 G 互补）；过窄的段 % 注在上方引线处。", 9.5)
    _label(ax, 6, 0.12, "85→95 的顺序：先 δ（A→B 拿走 16 个点），再压 hang 检测 T_d（B→C 的最大项）。", 9.5)
    save(fig, "fault-tolerance-five-terms")


def restart_lanes():
    """F2：三种恢复方式，同一时间轴；区间取 §四.4 表中点。"""
    fig, ax = _canvas(7.6, 4.4, (0, 12), (0, 6.6))
    _label(ax, 6, 6.3, "一次故障的三种恢复路径（同一分钟轴，区间取正文中点）",
           12, weight="bold")
    # 0–18 min → x 2.4–10.8
    def mx(v):
        return 2.4 + v / 18 * 8.4
    # 阶段类型颜色：检测 / 重建进程与通信 / 加载 / 回退重算 / 其他
    C_DET, C_REB, C_LOAD, C_REW, C_OTH = RED, BLUE, GRAY, PURPLE, ORANGE
    legend = [("检测", C_DET), ("重建进程与通信", C_REB), ("加载", C_LOAD),
              ("回退重算", C_REW), ("其他", C_OTH)]
    lx = 2.4
    for lname, lfc in legend:
        _card(ax, lx, 5.6, 0.26, 0.26, "", fc=lfc, ec="#888888", lw=0.6)
        _label(ax, lx + 0.55, 5.73, lname, 9.5, ha="left")
        lx += 0.55 + len(lname) * 0.19 + 0.62
    # 段（min）：中点估算，见文末 caption；notes 是窄段的秒数
    lanes = [
        ("进程重启", [
            ("检测", 10.0, C_DET), ("rendezv", 0.38, C_REB), ("进程+CUDA", 0.92, C_REB),
            ("comm 初始化", 1.0, C_REB), ("数据集", 0.58, C_OTH),
            ("加载", 2.0, C_LOAD), ("回退", 2.2, C_REW)],
         "rendezv 22s · 进程+CUDA 55s · comm 60s · 数据集 35s · 加载 2min · 回退 2.2min"),
        ("进程内重启", [
            ("检测", 1.0, C_DET), ("abort+清理", 0.4, C_REB), ("comm 重建", 0.35, C_REB),
            ("加载", 0.5, C_LOAD), ("回退", 2.2, C_REW)],
         "检测 60s · abort 24s · comm 21s · 加载 30s · 回退 2.2min"),
        ("torchft 副本组", [
            ("检测", 0.1, C_DET), ("quorum", 1.0, C_REB), ("heal", 0.5, C_LOAD)],
         "检测 6s · quorum ≤60s · heal 30s；其余副本组不中断"),
    ]
    for i, (lane, segs, note) in enumerate(lanes):
        y = 4.75 - i * 1.5
        _label(ax, 2.0, y + 0.28, lane, 10, ha="right", weight="bold")
        x = mx(0)
        for name, dur, fc in segs:
            w = dur / 18 * 8.4
            p = _card(ax, x, y, w, 0.56, "", fc=fc, ec="#888888", lw=0.6)
            if w >= 2.0:
                _label(ax, x + w / 2, y + 0.28, f"{name} {dur:.1f}min", 9, in_box=p)
            x += w
        _label(ax, x + 0.35, y + 0.28, f"≈{sum(d for _, d, _ in segs):.1f} min",
               9.5, ha="left")
        _label(ax, mx(0) + 0.05, y - 0.28, note, 9, ha="left", color="#555555")
    for gx in range(0, 19, 3):
        ax.plot([mx(gx), mx(gx)], [1.05, 5.5], color="#eeeeee", lw=0.6, zorder=0)
        _label(ax, mx(gx), 0.9, f"{gx}", 9, color="#666666")
    _label(ax, 6, 0.55, "横轴：分钟。段长取 §四.4 / §五.4 / §六 区间中点；回退 τ/2 ≈ 2.2 min", 9.5)
    _label(ax, 6, 0.18, "（τ_opt≈4.4min，δ=10s 异步）三行相同；torchft 不经过存储与全局重启。", 9.5)
    save(fig, "fault-tolerance-restart-lanes")


def replica_groups():
    """F3：torchft 副本组三格：正常 → 掉线缩 quorum → heal 回 3。"""
    fig, ax = _canvas(7.6, 4.3, (0, 12), (0, 6.2))
    _label(ax, 6, 5.9, "torchft 副本组：一个副本掉线，其余继续（§六.1–4）", 12.5,
           weight="bold")
    cols = [
        ("quorum = {0,1,2}", None, "梯度 all-reduce，步进提交"),
        ("副本 1 掉线", 1, "quorum = {0,2}，两组继续\n梯度按 2 归一化，步进照提"),
        ("副本 1 回归：heal", 1, "从邻居（副本 0）拉参数\nquorum 回到 {0,1,2}"),
    ]
    bw, gap = 3.2, 0.55
    for ci, (title, dead, desc) in enumerate(cols):
        x0 = 0.5 + ci * (bw + gap)
        p = _card(ax, x0, 4.15, bw, 1.5, "", fc="#f8fafc", ec="#888888", lw=0.8)
        _label(ax, x0 + bw / 2, 5.3, title, 11, in_box=p, weight="bold")
        _label(ax, x0 + bw / 2, 4.62, desc, 9.5, in_box=p)
        for r in range(3):
            y = 3.15 - r * 0.78
            dead_here = dead is not None and r == dead
            healed = ci == 2 and r == 1
            if dead_here:
                fc, ec, ls = "#f5f5f5", "#999999", "--"
            elif healed:
                fc, ec, ls = ORANGE, C["orange"], "-"
            else:
                fc, ec, ls = ["#dce9f7", "#dcefe2", "#fae6d6"][r], \
                    [C["blue"], C["green"], C["orange"]][r], "-"
            p = box(ax, x0 + 0.25, y, bw - 0.5, 0.62, fc=fc, ec=ec,
                    lw=0.8, linestyle=ls)
            lab = f"副本 {r}：{'掉线' if dead_here else '训练中'}"
            if healed:
                lab = "副本 1：heal 后重新加入"
            elif ci == 2 and r != 1:
                lab = "副本 0：参数来源" if r == 0 else "副本 2：训练中"
            _label(ax, x0 + bw / 2, y + 0.31, lab, 9.5, in_box=p)
        if ci < 2:
            _arrow(ax, (x0 + bw + 0.06, 2.36), (x0 + bw + gap - 0.06, 2.36))
        else:
            # heal：副本 0 → 副本 1 的弯曲箭头
            ax.annotate("", xy=(x0 + bw - 0.25, 2.68),
                        xytext=(x0 + bw - 0.25, 3.15),
                        arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.2,
                                        connectionstyle="arc3,rad=-0.55",
                                        mutation_scale=10))
            _label(ax, x0 + bw + 0.16, 2.92, "拉参数", 9, ha="left", va="center",
                   color=C["orange"])
    _label(ax, 6, 0.62, "没有全局重启、没有从存储读 checkpoint：DP 度 3→2 继续，", 9.5)
    _label(ax, 6, 0.22, "回归的副本经 CheckpointTransport 从活着的副本拉参数后重新加入。", 9.5)
    save(fig, "fault-tolerance-replica-groups")


def main():
    for f in (five_terms, restart_lanes, replica_groups):
        f()
    print("全部 3 张图完成。")


if __name__ == "__main__":
    main()
