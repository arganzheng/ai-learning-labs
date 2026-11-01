"""《长时间训练的可观测性与运维》配图：_plot.py 风格，collide()=0。
数据均来自 11_obs_toys.py 的实测输出（expected/*.json）。

F1 fr-rings：4 个 rank 的 FR 环形缓冲按 seq 对齐，rank 3 缺席 seq 6（§三.2/6）
F2 step-time-phases：分阶段堆叠面积 + 回归检测器开火点（§四.1–2）
F3 rank-step-heatmap：rank × step 的 compute 时间热力图（§二.6）
F4 alert-timeline：吞吐曲线 + 告警状态泳道（§五.1–3）
"""
import json
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


def _load(name):
    return json.load(open(os.path.join(HERE, "expected", name)))


def fr_rings():
    """F1：每 rank 最后几条 FR 记录按 seq 对齐；rank 3 在 seq 6 缺席。"""
    d = _load("11_obs_fr.json")
    fig, ax = _canvas(h=4.6, xlim=(0, 12), ylim=(0, 6.4))
    _label(ax, 6, 6.15, "Flight Recorder：4 个 rank 的环形缓冲按 collective_seq_id 对齐",
           12.5, weight="bold")
    seqs = [1, 2, 3, 4, 5, 6]
    x0, cw, ch = 2.1, 1.55, 0.62
    for i, sq in enumerate(seqs):
        _label(ax, x0 + i * cw + cw / 2, 5.5, f"seq {sq}", 9.5, color="#555555")
    rank_rows = [d[f"rank {r}"] for r in range(4)]
    for r in range(4):
        y = 4.72 - r * 0.78
        _label(ax, 1.15, y + ch / 2, f"rank {r}", 10, ha="right")
        for i, sq in enumerate(seqs):
            have = sq in rank_rows[r]
            if have:
                fc = GREEN if sq <= 4 else (ORANGE if sq < 6 else RED)
                p = _card(ax, x0 + i * cw, y, cw - 0.14, ch, "", fc=fc,
                          ec="#888888", lw=0.8)
                if i == 0:
                    _label(ax, x0 + i * cw + (cw - 0.14) / 2, y + ch / 2,
                           "all_reduce", 9.5, in_box=p)
            else:
                p = _card(ax, x0 + i * cw, y, cw - 0.14, ch, "", fc="white",
                          ec=C["red"], lw=1.0)
                ax.plot([x0 + i * cw + 0.18, x0 + (i + 1) * cw - 0.32],
                        [y + 0.08, y + ch - 0.08], color=C["red"], lw=0.9)
                _label(ax, x0 + i * cw + (cw - 0.14) / 2, y + ch / 2,
                       "缺席", 9.5, color=C["red"], in_box=p)
    # 缺席原因标注：rank 3 的记录里少了一条 —— 它的 seq 5 其实是别人的 6
    _label(ax, x0 + 5 * cw + cw / 2, 1.62, "", 1)  # spacer 占位防越界
    _arrow(ax, (x0 + 5 * cw + cw / 2 - 0.07, 4.72 - 3 * 0.78 - 0.05),
           (x0 + 5 * cw + cw / 2 - 0.07, 4.72 - 3 * 0.78 - 0.55),
           color=C["red"])
    _label(ax, x0 + 4.6 * cw, 1.82,
           "rank 3 没有发起 seq 6 的 all_reduce（走了 rank 相关的分支）", 9.5,
           color=C["red"])
    diag = _card(ax, 1.6, 0.15, 8.8, 1.2, "", fc="#fff8e1", ec="#c58a00",
                 lw=1.0)
    _label(ax, 6.0, 1.04, "fr_trace 的诊断（toy 实测输出）", 10,
           weight="bold", in_box=diag)
    _label(ax, 6.0, 0.68, "Not all ranks joining collective", 10, in_box=diag)
    _label(ax, 6.0, 0.36, "seq 6，missing ranks {3} → 查 rank 3 的栈与 py-spy",
           10, in_box=diag)
    save(fig, "long-running-training-fr-rings")


def step_time_phases():
    """F2：分阶段堆叠面积 + 回归检测器开火点。"""
    d = _load("11_obs_timers.json")
    series = d["series"]
    steps = [s["step"] for s in series]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    keys = [("data_ms", "data（数据等待）", "#f5b04a"),
            ("fwd_ms", "fwd（前向）", "#7fb3e0"),
            ("bwd_ms", "bwd（反向）", "#4a7fb5"),
            ("opt_ms", "opt（优化器）", "#9c8ec4")]
    ax.stackplot(steps, [[s[k] for s in series] for k, _, _ in keys],
                 labels=[lb for _, lb, _ in keys],
                 colors=[c for _, _, c in keys], lw=0.5, edgecolor="white")
    ax.set_xlabel("step", fontsize=10)
    ax.set_ylabel("step 时间（ms，堆叠）", fontsize=10)
    ax.set_xlim(0, 121)
    ax.set_ylim(0, 240)
    ax.set_yticks(range(0, 241, 40))
    ax.set_xticks(range(0, 121, 20))
    ax.tick_params(labelsize=9.5)
    fire = d["fire_step"]
    ax.axvline(fire, color=C["red"], lw=1.1, ls="--")
    ax.text(fire + 2, 232, f"检测器开火：step {fire}\n（20 步中位数 > 基线 100ms × 1.10）",
            fontsize=9.5, color=C["red"], va="top")
    ax.annotate("data 从 step 60 起每步 +2ms：\n慢化的嫌疑直接落在数据上",
                xy=(112, 205), xytext=(52, 150), fontsize=9.5,
                color="#8a5a00",
                arrowprops=dict(arrowstyle="->", color="#8a5a00", lw=0.9))
    ax.legend(fontsize=9, frameon=False, loc="upper left", ncols=2)
    ax.set_title("分阶段计时的堆叠面积：渐变慢一眼定位（toy 实测）", fontsize=11.5,
                 fontweight=600)
    save(fig, "long-running-training-step-time-phases")


def rank_step_heatmap():
    """F3：rank × step compute 热力图；慢 rank 一眼可见。"""
    d = _load("11_obs_ranks.json")
    import numpy as np
    M = np.array(d["matrix"]).T  # rank × step
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=10, vmax=60)
    ax.set_yticks(range(4), [f"rank {r}" for r in range(4)], fontsize=10)
    ax.set_xticks(range(0, 20, 2), [str(s + 1) for s in range(0, 20, 2)],
                  fontsize=9.5)
    ax.set_xlabel("step", fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("compute（ms）", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    ax.annotate("rank 1 从 step 8 起 +30ms：step 时间被\nall_reduce 等齐，"
                "只有每 rank 计时能认出它",
                xy=(11.2, 1), xytext=(13.2, 0.05), fontsize=9.5,
                color="#8a0000",
                arrowprops=dict(arrowstyle="->", color="#8a0000", lw=0.9),
                va="center", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    ax.set_title("rank × step 的 compute 时间（每 rank 计时，toy 实测）",
                 fontsize=11.5, fontweight=600)
    fig.subplots_adjust(bottom=0.32)
    save(fig, "long-running-training-rank-step-heatmap")


def alert_timeline():
    """F4：吞吐曲线 + page/record 阈值 + 告警泳道（pending→firing / suppressed）。"""
    d = _load("11_obs_alert.json")
    T = [t / 60 for t in d["T"]]  # 分钟
    thr = d["throughput"]
    hb = [h / 60 for h in d["heartbeat"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 4.0), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1.3],
                                            "hspace": 0.18})
    a1.plot(T, thr, color=C["blue"], lw=1.6, label="吞吐 tok/s（合成）")
    a1.axhline(850, color="#8a8a8a", lw=0.9, ls=":")
    a1.text(0.6, 878, "record 阈值：-15%（850）", fontsize=9, color="#555555")
    a1.set_ylim(700, 1080)
    a1.set_yticks([700, 850, 1000])
    a1.set_ylabel("tok/s", fontsize=10)
    a1.tick_params(labelsize=9.5)
    # dip 区域
    a1.axvspan(10, 15, color="#fdecea", alpha=0.7)
    a1.text(12.5, 1050, "5 分钟 dip：\nrecord，不 page", fontsize=9.5,
            color="#8a5a00", ha="center")
    # 心跳停
    a1.axvline(23.5, color=C["red"], lw=1.1, ls="--")
    a1.text(22.3, 740, "心跳停\n(23:30)", fontsize=9.5, color=C["red"],
            ha="right", va="top")
    # 重启窗口
    a1.axvspan(25, 28, color="#dbeafe", alpha=0.8)
    a1.text(26.5, 760, "重启窗口\n(25:00–28:00)", fontsize=9, color="#1d4ed8",
            ha="center")
    a1.legend(fontsize=9.5, frameon=False, loc="lower left")

    # 泳道
    lanes = [("TrainingMfuDegraded (page·for 30m)", 3,
              [(10, 15, "pending", "#f0c8a0")], "未到 for:30m 即消失"),
             ("ThroughputDip (record·for 2m)", 2,
              [(10, 12, "pending", "#f0c8a0"), (12, 15, "firing", "#f5b04a")],
              "pending 2 分钟后 firing；record，白天看"),
             ("TrainingStepStalled (page·for 0m)", 1,
              [(23.5, 25.25, "pending", "#f0c8a0"),
               (25.25, 30, "firing", "#c4b5e0")],
              "firing 落在重启窗口 → 被抑制")]
    for name, row, segs, note in lanes:
        for x0, x1, st, fc in segs:
            a2.barh(row, x1 - x0, left=x0, height=0.55, color=fc,
                    edgecolor="#888888", lw=0.6)
            if x1 - x0 > 2.4:
                a2.text((x0 + x1) / 2, row, st, fontsize=9, ha="center",
                        va="center")
        a2.text(-0.5, row, name, fontsize=9.5, ha="right", va="center")
        a2.text(30.4, row, note, fontsize=9, ha="left", va="center",
                color="#555555")
    a2.axvspan(25, 28, color="#dbeafe", alpha=0.8)
    a2.set_yticks([])
    a2.set_xlim(-13.5, 41)
    a2.set_ylim(0.3, 3.7)
    a2.set_xticks(range(0, 31, 5))
    a2.set_xticklabels([f"{t:02d}:00" for t in range(0, 31, 5)], fontsize=9.5)
    a2.set_xlabel("时间（分钟）", fontsize=10)
    a2.spines[["left", "right", "top"]].set_visible(False)
    fig.suptitle("告警规则评估的模拟：for: 语义、page/record、抑制（toy 数据）",
                 fontsize=11.5, fontweight=600)
    save(fig, "long-running-training-alert-timeline")


if __name__ == "__main__":
    fr_rings()
    step_time_phases()
    rank_step_heatmap()
    alert_timeline()
    print("全部 4 张图完成。")
