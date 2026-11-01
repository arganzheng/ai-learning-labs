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
    """F1：G 的五项构成，16K 卡场景 B(86.7%) → C(94.5%)。"""
    fig, ax = _canvas(7.6, 4.4, (0, 12), (0, 6.6))
    _label(ax, 6, 6.3, "有效训练时间 G 的五项构成（16K 卡，§二.4 场景 B → C）",
           12, weight="bold")
    segs = [("检测 T_d", BLUE, C["blue"]), ("重启 T_r", GREEN, C["green"]),
            ("加载 T_l", ORANGE, C["orange"]), ("回退 τ/2", PURPLE, C["purple"]),
            ("保存 δ/τ", RED, C["red"]), ("有效 G", GRAY, "#999999")]
    rows = [
        ("B：异步保存 · T_d=600s · T_r=300s · T_l=120s",
         [5.5, 2.7, 1.1, 2.1, 1.9, 86.7]),
        ("C：异步保存 · T_d=60s · T_r=30s · T_l=60s",
         [0.5, 0.3, 0.5, 2.1, 2.0, 94.5]),
    ]
    x0, bw = 1.4, 8.6
    for r, (name, vals) in enumerate(rows):
        y = 4.55 - r * 1.5
        _label(ax, x0, y + 0.72, name, 9.5, ha="left", color="#333333")
        x = x0
        for (sname, fc, ec), v in zip(segs, vals):
            w = bw * v / 100
            p = _card(ax, x, y, w, 0.5, "", fc=fc, ec=ec, lw=0.6)
            if v >= 8:
                _label(ax, x + w / 2, y + 0.25, f"{v:.0f}%" if v < 50 else f"G {v:.1f}%",
                       9.5, in_box=p)
            x += w
        # 窄段外面注
        labs = []
        x = x0
        for (sname, fc, ec), v in zip(segs, vals):
            if v < 8:
                labs.append(f"{sname.split()[-1]} {v}%")
            x += bw * v / 100
        _label(ax, x0 + bw + 0.4, y + 0.25, " ".join(labs), 9, ha="left",
               color="#444444")
    # 箭头：B→C 缩最多的是 T_d
    p1 = _card(ax, 1.4, 1.15, 8.6 * 0.055, 0.5, "", fc=BLUE, ec=C["blue"], lw=0.6)
    ax.plot([], [])
    _arrow(ax, (2.0, 2.6), (2.0, 1.7), color=C["blue"], lw=1.4)
    _label(ax, 3.4, 1.85, "T_d 600→60 s：唯一缩两个数量级的项", 9.5, ha="left",
           color="#222222")
    _label(ax, 6, 0.55, "分子四项来自 §二.4 表的分项；G 与各项和为 100%。85→95 的答案：", 9.5)
    _label(ax, 6, 0.15, "先 δ（B 里已经做了），再把 hang 检测从 600 s 压到 60 s（§三.3、§五）。", 9.5)
    save(fig, "fault-tolerance-five-terms")


def restart_lanes():
    """F2：三种恢复方式，同一时间轴；区间取 §四.4 表中点。"""
    fig, ax = _canvas(7.6, 4.4, (0, 12), (0, 6.6))
    _label(ax, 6, 6.3, "一次故障的三种恢复路径（同一分钟轴，区间取正文中点）",
           12, weight="bold")
    # 0–18 min → x 2.4–10.8
    def mx(v):
        return 2.4 + v / 18 * 8.4
    # 段（min）：中点估算，见文末 caption；notes 是窄段的秒数
    lanes = [
        ("进程重启", [
            ("检测", 10.0, RED), ("rendezv", 0.38, ORANGE), ("进程+CUDA", 0.92, BLUE),
            ("comm 初始化", 1.0, GREEN), ("数据集", 0.58, PURPLE),
            ("加载", 2.0, GRAY), ("回退", 2.2, "#f0f0f0")],
         "rendezv 22s · 进程+CUDA 55s · comm 60s · 数据集 35s · 加载 2min · 回退 2.2min"),
        ("进程内重启", [
            ("检测", 1.0, RED), ("abort+清理", 0.4, ORANGE), ("comm 重建", 0.35, GREEN),
            ("加载", 0.5, GRAY), ("回退", 2.2, "#f0f0f0")],
         "检测 60s · abort 24s · comm 21s · 加载 30s · 回退 2.2min"),
        ("torchft 副本组", [
            ("检测", 0.1, RED), ("quorum", 1.0, GREEN), ("heal", 0.5, BLUE)],
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
        ("quorum = {0,1,2}", None, "三组梯度 all-reduce，步进提交"),
        ("副本 1 掉线", 1, "quorum = {0,2}，两组继续\n梯度按 2 归一化，步进照提"),
        ("副本 1 回归：heal", 1, "从邻居（副本 0）拉参数\nquorum 回到 {0,1,2}"),
    ]
    bw, gap = 3.2, 0.55
    for ci, (title, dead, desc) in enumerate(cols):
        x0 = 0.5 + ci * (bw + gap)
        p = _card(ax, x0, 4.05, bw, 1.6, "", fc="#f8fafc", ec="#888888", lw=0.8)
        _label(ax, x0 + bw / 2, 5.38, title, 10, in_box=p, weight="bold")
        _label(ax, x0 + bw / 2, 4.62, desc, 9, in_box=p)
        for r in range(3):
            y = 3.15 - r * 0.78
            dead_here = dead is not None and r == dead
            fc = RED if dead_here else ["#dce9f7", "#dcefe2", "#fae6d6"][r]
            ec = C["red"] if dead_here else [C["blue"], C["green"], C["orange"]][r]
            p = _card(ax, x0 + 0.25, y, bw - 0.5, 0.62, "", fc=fc, ec=ec, lw=0.8)
            lab = f"副本 {r}：{'掉线' if dead_here else '训练中'}"
            if ci == 2 and r == 1:
                lab = "副本 1：heal 后重新加入"
            elif ci == 2 and r != 1:
                lab = "副本 0：参数来源" if r == 0 else "副本 2：训练中"
            _label(ax, x0 + bw / 2, y + 0.31, lab, 9.5, in_box=p)
        if ci < 2:
            _arrow(ax, (x0 + bw + 0.06, 2.4), (x0 + bw + gap - 0.06, 2.4))
    _label(ax, 6, 0.62, "没有全局重启、没有从存储读 checkpoint：DP 度 3→2 继续，", 9.5)
    _label(ax, 6, 0.22, "回归的副本经 CheckpointTransport 从活着的副本拉参数后重新加入。", 9.5)
    save(fig, "fault-tolerance-replica-groups")


def main():
    for f in (five_terms, restart_lanes, replica_groups):
        f()
    print("全部 3 张图完成。")


if __name__ == "__main__":
    main()
