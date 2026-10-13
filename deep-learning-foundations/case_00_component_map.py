"""导读部件图：一个深度网络由哪些部件组成（无训练，纯示意）。

    python case_00_component_map.py   # out/component-map.svg

布局（正交网格）：
    顶行 输入 → Embedding → 层（外框内含三个子框）→ 输出头 → 损失
    底行 初始化（Embedding 下方）/ 优化器（层正下方）/ 反向传播（损失正下方）/ 正则化（损失右下方）
    箭头全部正交：损失 ↓ 反向传播；反向传播 → 优化器（水平）；
    优化器 ↑ 层底边中点；初始化 肘形进层底边左端；正则化 ↑ 损失底边右端。
"""
import matplotlib.patches as mp
import matplotlib.pyplot as plt

from _plot import C, save

plt.rcParams["font.size"] = 9

MAIN = C["blue"]; TRAIN = C["orange"]; AUX = C["gray"]; STRUCT = C["green"]

# ---- 盒子登记：{name: (x, y, w, h)}；箭头端点一律取盒边，绘制时断言命中
B = {}


def box(ax, name, x, y, w, h, lines, fc, ec, fontsize=9, lw=1.2):
    B[name] = (x, y, w, h)
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                   fc=fc, ec=ec, lw=lw))
    n = len(lines)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h * (1 - (i + 0.5) / n), t, ha="center", va="center",
                fontsize=fontsize, color="#222222")


def edge(name, side, frac=0.5):
    """盒边上的点。side: 'l','r','t','b'；frac 为沿边的分位。"""
    x, y, w, h = B[name]
    return {"l": (x, y + frac * h), "r": (x + w, y + frac * h),
            "t": (x + frac * w, y + h), "b": (x + frac * w, y)}[side]


def on_edge(pt, name, tol=0.06):
    x, y = pt
    bx, by, w, h = B[name]
    on_v = by - tol <= y <= by + h + tol and (abs(x - bx) <= tol or abs(x - bx - w) <= tol)
    on_h = bx - tol <= x <= bx + w + tol and (abs(y - by) <= tol or abs(y - by - h) <= tol)
    return on_v or on_h


def arrow(ax, p0, p1, target, color="#555555", lw=1.4):
    assert p0[0] == p1[0] or p0[1] == p1[1], f"非正交箭头 {p0}->{p1}"
    assert on_edge(p1, target), f"箭头 {p1} 不在 {target} 盒边上"
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))


def elbow(ax, p0, mid, p1, target, color="#555555", lw=1.4):
    """两段正交折线 p0 -> mid -> p1。"""
    assert p0[0] == mid[0] or p0[1] == mid[1], f"非正交段 {p0}->{mid}"
    assert mid[0] == p1[0] or mid[1] == p1[1], f"非正交段 {mid}->{p1}"
    assert on_edge(p1, target), f"箭头 {p1} 不在 {target} 盒边上"
    ax.plot([p0[0], mid[0]], [p0[1], mid[1]], color=color, lw=lw, solid_capstyle="round")
    ax.annotate("", xy=p1, xytext=mid,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    ax.set_xlim(0, 20.6); ax.set_ylim(0, 9.4); ax.axis("off")

    TY, TH = 6.8, 1.3   # 顶行盒 y / 高；顶行盒中缝 y = TY + TH/2 = 7.45
    BY, BH = 0.9, 1.4   # 底行
    CY = TY + TH / 2

    # ---- 顶行主链：输入 → Embedding → 层 → 输出头 → 损失
    box(ax, "input", 0.5, TY, 2.2, TH, ["输入", "token / 像素"], "#eef3fa", MAIN)
    box(ax, "emb", 3.4, TY, 2.6, TH, ["Embedding", "查表 $[V,d]$"], "#eef3fa", MAIN)
    # 层外框（x 7.0–12.4，y 5.0–8.9）
    B["layer"] = (7.0, 5.0, 5.4, 3.9)
    ax.add_patch(mp.FancyBboxPatch((7.0, 5.0), 5.4, 3.9, boxstyle="round,pad=0.02,rounding_size=0.06",
                                   fc="#f2f8f3", ec=STRUCT, lw=1.4))
    ax.text(9.7, 8.55, "层（堆叠 $L$ 次）", ha="center", fontsize=10, color=STRUCT, weight="bold")
    for i, t in enumerate(["线性 / 卷积 / 循环 / attention", "激活函数", "归一化 + 残差"]):
        box(ax, f"sub{i}", 7.3, 7.15 - i * 0.98, 4.8, 0.78, [t], "white", STRUCT, fontsize=8.5)
    box(ax, "head", 13.2, TY, 2.4, TH, ["输出头", "logits"], "#eef3fa", MAIN)
    box(ax, "loss", 16.4, TY, 3.4, TH, ["损失 $L$", "一个可求导的数"], "#fdf2e4", TRAIN)

    arrow(ax, edge("input", "r"), edge("emb", "l"), "emb")
    arrow(ax, edge("emb", "r"), (7.0, CY), "layer")
    arrow(ax, (12.4, CY), edge("head", "l"), "head")
    arrow(ax, edge("head", "r"), edge("loss", "l"), "loss")

    # ---- 底行训练回路
    box(ax, "init", 3.4, BY, 2.6, BH, ["初始化", "训练从哪开始"], "#f4f4f4", AUX)
    box(ax, "opt", 7.5, BY, 4.4, BH, ["优化器", "SGD / AdamW + 调度"], "#fdf2e4", TRAIN)
    box(ax, "bp", 16.1, BY, 2.6, BH, ["反向传播", "链式法则求 $\\partial L/\\partial\\theta$"],
        "#fdf2e4", TRAIN, fontsize=8.5)
    box(ax, "reg", 18.9, BY, 1.3, BH, ["正则化", "dropout", "数据增强"], "#f4f4f4", AUX, fontsize=7.5)

    # 损失 ↓ 反向传播（垂直，bp 中心 x=17.4 在 loss 底边范围内）
    arrow(ax, (17.4, TY), (17.4, BY + BH), "bp", color=TRAIN)
    # 反向传播 → 优化器（水平）
    arrow(ax, edge("bp", "l"), edge("opt", "r"), "opt", color=TRAIN)
    # 优化器 ↑ 层底边中点（opt 中心 x=9.7 = 层中心）
    arrow(ax, edge("opt", "t"), (9.7, 5.0), "layer", color=TRAIN)
    ax.text(10.9, 3.9, "更新参数 $\\theta$", ha="center", fontsize=8, color=TRAIN)
    # 初始化 肘形：右边缘 → 水平 → 上到层底边左端（x=7.2，避开优化器左缘 7.5）
    elbow(ax, edge("init", "r"), (7.2, BY + BH / 2), (7.2, 5.0), "layer", color=AUX)
    ax.text(6.6, 1.9, "$\\theta_0$", ha="center", fontsize=8, color=AUX)
    # 正则化 ↑ 损失底边右端（x=19.55 在 loss 16.4–19.8 内）
    arrow(ax, (19.55, BY + BH), (19.55, TY), "loss", color=AUX)
    ax.text(20.15, 4.4, "惩罚项 / 平滑目标", rotation=90, ha="center", va="center",
            fontsize=7.5, color=AUX)

    save(fig, "component-map")
    print("图已保存：部件地图（顶行前向链，底行训练回路；箭头正交并断言落在盒边上）")


if __name__ == "__main__":
    main()
