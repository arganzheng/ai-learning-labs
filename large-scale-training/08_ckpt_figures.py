"""《分布式 checkpoint》配图：_plot.py 风格，collide()=0。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot import C, box, plt, save, text

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#dce9f7"
ORANGE = "#fae6d6"
GREEN = "#dcefe2"
PURPLE = "#eee1f3"
GRAY = "#e8e8e8"
RED = "#f5dede"


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


def six_tb():
    """F1：405B 一次 checkpoint 的 5.7 TB 去哪了 + 两条写路径。"""
    fig, ax = _canvas(7.6, 4.4, (0, 12), (0, 6.6))
    _label(ax, 6, 6.3, "405B 一次 checkpoint 的 5.7 TB：四类状态（§三.1）", 12.5,
           weight="bold")
    segs = [
        ("bf16 参数", 0.81, "810 GB", BLUE, C["blue"]),
        ("fp32 主参数", 1.62, "1.62 TB", GREEN, C["green"]),
        ("Adam m", 1.62, "1.62 TB", ORANGE, C["orange"]),
        ("Adam v", 1.62, "1.62 TB", PURPLE, C["purple"]),
    ]
    total = 5.67
    x = 0.6
    bw = 10.8
    for name, gb, lab, fc, ec in segs:
        w = bw * gb / total
        p = _card(ax, x, 5.0, w, 0.85, "", fc=fc, ec=ec, lw=0.8)
        _label(ax, x + w / 2, 5.62, name, 10, in_box=p, weight="bold")
        _label(ax, x + w / 2, 5.2, lab, 10, in_box=p)
        x += w
    _label(ax, 6, 4.45, "每参数 14 字节（2+4+4+4），梯度不存；16384 卡每卡 ≈ 350 MB", 10)

    _label(ax, 6, 3.75, "两条写路径（§三.2–3）", 11, weight="bold")
    p = _card(ax, 0.6, 2.05, 5.3, 1.5, "", fc="#f5dede", ec=C["red"], lw=0.9)
    _label(ax, 3.25, 3.3, "rank-0 汇总：不可行", 10.5, in_box=p, weight="bold")
    _label(ax, 3.25, 2.65, "单写入者 1–5 GB/s → 19–95 min\n主机内存要装 5.7 TB", 9.5,
           in_box=p)
    p = _card(ax, 6.15, 2.05, 5.3, 1.5, "", fc="#dcefe2", ec=C["green"], lw=0.9)
    _label(ax, 8.8, 3.3, "分片写：每卡写自己的 350 MB", 10.5, in_box=p,
           weight="bold")
    _label(ax, 8.8, 2.65, "聚合 500 GB/s → 11 s；1 TB/s → 6 s\nGPU→主机 <10 ms 可忽略", 9.5,
           in_box=p)
    _label(ax, 6, 1.45, "汇总的路在三个地方同时撞墙：内存、接收带宽、写入带宽——", 9.5)
    _label(ax, 6, 1.0, "δ 从分钟级压到秒级后，Young 公式的浪费才压得下去（§九）。", 9.5)
    save(fig, "distributed-checkpoint-6tb-breakdown")


def young_curve():
    """F2：Young 公式曲线 f(τ)=δ/τ + τ/(2M)，Llama 3 M=3.09 h。"""
    import numpy as np
    fig, ax = _canvas(7.6, 4.6, (0, 12), (0, 7.0))
    M = 3.09 * 3600
    delta = 60.0  # 同步分片写一行（正文表里 τ_opt=19 min）
    taus = np.linspace(30, 4000, 200) / 60.0  # min
    f_save = (delta / 60) / taus * 100
    f_lose = taus * 60 / (2 * M) * 100
    f_sum = f_save + f_lose
    t_opt = (2 * delta * M) ** 0.5 / 60
    # 坐标映射：x 0–40 min → 1.2–10.8；y 0–35% → 1.1–5.5
    def mx(v):
        return 1.2 + v / 40 * 9.6

    def my(v):
        return 1.1 + v / 35 * 4.4
    _label(ax, 6, 6.7, "Young 公式：最优 checkpoint 间隔 τ_opt = √(2δM)", 12.5,
           weight="bold")
    ax.plot([mx(t) for t in taus], [my(v) for v in f_save],
            color=C["blue"], lw=1.4)
    ax.plot([mx(t) for t in taus], [my(v) for v in f_lose],
            color=C["orange"], lw=1.4)
    ax.plot([mx(t) for t in taus], [my(v) for v in f_sum],
            color=C["red"], lw=1.8)
    _label(ax, mx(37), my((delta / 60) / 37 * 100) - 0.25, "δ/τ（保存开销）", 9.5,
           color=C["blue"])
    _label(ax, mx(37.5), my(37.5 * 60 / (2 * M) * 100) + 0.35, "τ/(2M)（回退损失）",
           9.5, color=C["orange"])
    _label(ax, mx(33), my(33 * 60 / (2 * M) * 100 + (delta / 60) / 33 * 100) + 0.35,
           "合计", 10, color=C["red"], weight="bold")
    ax.plot([mx(t_opt), mx(t_opt)], [1.1, my(f_sum.min() / 100 * 100)], "--",
            color="#888888", lw=0.9)
    _label(ax, mx(t_opt) + 0.15, 1.55, f"τ_opt ≈ {t_opt:.0f} min", 9.5, ha="left")
    _label(ax, mx(t_opt) + 0.15, my(f_sum.min()) + 0.35, "浪费 ≈ 10%", 9.5,
           color=C["red"], ha="left")
    for gx in range(0, 45, 5):
        ax.plot([mx(gx), mx(gx)], [1.1, 5.5], color="#eeeeee", lw=0.5, zorder=0)
        _label(ax, mx(gx), 0.85, f"{gx}", 9, color="#666666")
    for gy in range(0, 40, 5):
        ax.plot([1.2, 10.8], [my(gy), my(gy)], color="#eeeeee", lw=0.5, zorder=0)
        _label(ax, 1.0, my(gy), f"{gy}", 9, color="#666666", ha="right")
    _label(ax, 6, 0.35, "横轴：checkpoint 间隔 τ（min）；纵轴：期望浪费比例（%）· δ=60 s 同步分片写",
           9.5)
    _label(ax, 6, 0.05, "δ=600 s 时 τ_opt=61 min、浪费 32.8%；δ=10 s 异步时 8 min、4.2%",
           9.5)
    save(fig, "distributed-checkpoint-young-formula")


def reshard_blocks():
    """F3：4→3 rank 的重分片（玩具实测的块表）。"""
    fig, ax = _canvas(7.6, 4.2, (0, 12), (0, 6.4))
    _label(ax, 6, 6.1, "重分片示意：4 rank 存的 24 元素 → 3 rank 加载（toy 缩小例）",
           12, weight="bold")
    NEL, sw = 24, 0.36
    x0 = 1.6
    _label(ax, 1.28, 4.9, "磁盘\n4 rank", 9.5, ha="right")
    for i in range(4):
        p = _card(ax, x0 + i * 6 * sw, 4.45, 6 * sw, 0.9, "",
                  fc=[BLUE, GREEN, ORANGE, PURPLE][i],
                  ec=[C["blue"], C["green"], C["orange"], C["purple"]][i])
        _label(ax, x0 + i * 6 * sw + 3 * sw, 4.9, f"rank{i}\nw[{i*6}:{i*6+6}]",
               9.5, in_box=p)
    _label(ax, 1.28, 2.5, "加载\n3 rank", 9.5, ha="right")
    bounds = [(0, 8), (8, 16), (16, 24)]
    for i, (lo, hi) in enumerate(bounds):
        p = _card(ax, x0 + lo * sw, 2.05, (hi - lo) * sw, 0.9, "",
                  fc="#f8fafc", ec="#888888", lw=0.8)
        _label(ax, x0 + (lo + hi) / 2 * sw, 2.5, f"rank{i}  w[{lo}:{hi}]",
               9.5, in_box=p)
    # 交集标注：rank0 [0:8) 跨旧 rank0 全段 + rank1 前 2 个
    for i, (lo, hi) in enumerate(bounds):
        for j in range(4):
            o_lo, o_hi = j * 6, j * 6 + 6
            a, b = max(lo, o_lo), min(hi, o_hi)
            if a < b:
                p = _card(ax, x0 + a * sw, 3.15, (b - a) * sw, 0.5, "",
                          fc=["#dce9f7", "#dcefe2", "#fae6d6", "#eee1f3"][j],
                          ec="#bbbbbb", lw=0.5)
                if b - a >= 3:
                    _label(ax, x0 + (a + b) / 2 * sw, 3.4, f"{a}:{b}", 9, in_box=p)
    _label(ax, 6, 1.35, "rank0 读旧块 0–6 全部 + 6–8 两个元素；rank1 读 8–14（旧块 2 段）+ 14–16；",
           9.5)
    _label(ax, 6, 0.92, "rank2 读 16–24（旧块 3 全段）——交集由 planner 的 ReadItems 算出；", 9.5)
    _label(ax, 6, 0.5, "toy 实测：3 rank 各拿 8 元素、2 rank 各拿 12 元素，参数误差 0。", 9.5)
    save(fig, "distributed-checkpoint-reshard-4to3")


def main():
    for f in (six_tb, young_curve, reshard_blocks):
        f()
    print("全部 3 张图完成。")


if __name__ == "__main__":
    main()
