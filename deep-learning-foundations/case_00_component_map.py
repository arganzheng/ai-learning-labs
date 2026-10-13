"""导读部件图：一个深度网络由哪些部件组成（无训练，纯示意）。

    python case_00_component_map.py   # out/component-map.svg
"""
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

from _plot import C, save

plt.rcParams["font.size"] = 9

# 主链（横排）：输入 → Embedding → 层 → 输出头 → 损失
# 层框内：线性/卷积/循环/attention + 激活 + 归一化 + 残差
# 训练回路：损失 → 反向传播 → 优化器 → 参数
# 挂件：初始化（训练从哪开始）、正则化（能不能泛化）

MAIN = C["blue"]; TRAIN = C["orange"]; AUX = C["gray"]; STRUCT = C["green"]


def box(ax, x, y, w, h, lines, fc, ec, fontsize=9, text_color="#222222", lw=1.2):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                   fc=fc, ec=ec, lw=lw))
    n = len(lines)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h * (1 - (i + 0.5) / n), t, ha="center", va="center",
                fontsize=fontsize, color=text_color)


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.4, text=None, tx=0.0, ty=0.12, fs=8, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, ha="center", va="center",
                fontsize=fs, color=color)


def main():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 20); ax.set_ylim(0, 9.6); ax.axis("off")

    # ---- 前向主链
    box(ax, 0.4, 6.4, 2.4, 1.2, ["输入", "token / 像素"], "#eef3fa", MAIN)
    box(ax, 3.6, 6.4, 2.6, 1.2, ["Embedding", "查表 $[V,d]$"], "#eef3fa", MAIN)
    # 层 大框
    ax.add_patch(mp.FancyBboxPatch((7.0, 5.2), 5.2, 3.0, boxstyle="round,pad=0.02,rounding_size=0.06",
                                   fc="#f2f8f3", ec=STRUCT, lw=1.4))
    ax.text(9.6, 7.75, "层（堆叠 $L$ 次）", ha="center", fontsize=10, color=STRUCT, weight="bold")
    for i, t in enumerate(["线性 / 卷积 / 循环 / attention", "激活函数", "归一化 + 残差"]):
        box(ax, 7.3, 6.7 - i * 1.0, 4.6, 0.78, [t], "white", STRUCT, fontsize=8.5)
    box(ax, 13.0, 6.4, 2.4, 1.2, ["输出头", "logits"], "#eef3fa", MAIN)
    box(ax, 16.2, 6.4, 3.2, 1.2, ["损失 $L$", "一个可求导的数"], "#fdf2e4", TRAIN)

    arrow(ax, 2.8, 7.0, 3.6, 7.0)
    arrow(ax, 6.2, 7.0, 7.0, 7.0)
    arrow(ax, 12.2, 7.0, 13.0, 7.0)
    arrow(ax, 15.4, 7.0, 16.2, 7.0)

    # ---- 训练回路（底部）：损失 → 反向传播 → 优化器 → 回到参数
    box(ax, 11.6, 1.2, 3.4, 1.2, ["反向传播", "链式法则求 $\\partial L/\\partial\\theta$"], "#fdf2e4", TRAIN)
    box(ax, 6.6, 1.2, 3.4, 1.2, ["优化器", "SGD / AdamW + 调度"], "#fdf2e4", TRAIN)
    arrow(ax, 17.8, 6.4, 17.8, 1.8, color=TRAIN)
    arrow(ax, 17.8, 1.8, 15.0, 1.8, color=TRAIN)
    arrow(ax, 11.6, 1.8, 10.0, 1.8, color=TRAIN)
    # 优化器 → 参数（回到主链，指到层框底）
    arrow(ax, 6.6, 1.8, 8.1, 5.05, color=TRAIN, rad=-0.25, text="更新参数 $\\theta$", tx=-1.8, ty=0.0, fs=8)

    # ---- 挂件
    box(ax, 1.2, 2.6, 3.0, 1.2, ["初始化", "训练从哪开始"], "#f4f4f4", AUX)
    arrow(ax, 4.2, 3.2, 7.0, 5.35, color=AUX, rad=0.18)
    box(ax, 15.6, 3.4, 3.6, 1.2, ["正则化", "会不会泛化"], "#f4f4f4", AUX)
    arrow(ax, 16.8, 4.6, 15.0, 2.4, color=AUX, rad=0.15)

    save(fig, "component-map")
    print("图已保存：部件地图（主链=前向，橙=训练回路，灰=初始化/正则化两件套）")


if __name__ == "__main__":
    main()
