"""两种给位置的方法，用「猫 追 狗」/「狗 追 猫」算一遍（Transformer 与 LLM 01）。
https://arganzheng.life/transformer-architecture-from-a-sentence-to-the-next-token.html

方法一（GPT-2）：token 向量 + 位置表第 i 行；同一个「猫」在第 0 位和第 2 位是两个不同的向量。
方法二（RoPE）：向量不变，q、k 各按自己的位置转 m·θ；q_m · k_n 只依赖 n − m。

    python position_two_methods.py            # 打印两种方法的数字
    python position_two_methods.py out.svg    # 另存配图（需要 matplotlib）
"""
import sys

import numpy as np

np.set_printoptions(precision=3, suppress=True)

TOK = {"猫": np.array([2.0, 0.0]), "追": np.array([0.0, 2.0]), "狗": np.array([2.0, 1.0])}
POS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])   # 方法一的位置表 wpe：4 行 × 2 维
THETA = np.pi / 6                                                     # 方法二：每个位置转 30°


def rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def method1(sentence, offset=0):
    """返回每个 token 加位置向量后的向量：x_i + wpe[i]。"""
    return [(w, TOK[w] + POS[i + offset]) for i, w in enumerate(sentence)]


def method2_score(sentence, qi, kj, offset=0):
    """RoPE：q、k 各按自己的位置旋转，再做内积（玩具里 W_Q = W_K = I，q = k = x）。"""
    m, n = sentence.index(qi) + offset, sentence.index(kj) + offset
    q = rot(m * THETA) @ TOK[qi]
    k = rot(n * THETA) @ TOK[kj]
    return float(q @ k), n - m


if __name__ == "__main__":
    print("=== 方法一：加位置向量（GPT-2 的 wpe）===")
    for s, off in [("猫追狗", 0), ("狗追猫", 0), ("猫追狗", 1)]:
        label = s if off == 0 else f"{s}（整体后移 {off} 位）"
        vecs = method1(s, off)
        print(f"  {label}: " + "  ".join(f"{w}@{i + off}={v}" for i, (w, v) in enumerate(vecs)))
    print("  同一个「猫」：第 0 位是 (3, 0)，第 2 位是 (1, 0)，第 1 位是 (2, 1)——位置变了，向量就变了（绝对位置）")

    print("\n=== 方法二：转角度（RoPE，每个位置转 30°）===")
    for s, off in [("猫追狗", 0), ("狗追猫", 0), ("猫追狗", 1)]:
        label = s if off == 0 else f"{s}（整体后移 {off} 位）"
        sc, diff = method2_score(s, "猫", "狗", off)
        print(f"  {label}: 猫 看 狗 的分数 q·k = {sc:.3f}（位置差 {diff:+d} → 狗 相对 猫 转 {diff * 30:+d}°）")
    bare = float(TOK["猫"] @ TOK["狗"])
    print(f"  不加位置时 猫·狗 = {bare:.3f}；两种词序分数不同，整句后移分数不变（只看相对位置）")

    if len(sys.argv) > 1:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc

        plt.rcParams.update({
            "font.family": ["Heiti SC", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 9,
            "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        })
        BLUE, RED, GRN, GRAY, ORG = "#1f6fb2", "#c0392b", "#4d9a5c", "#8a8a8a", "#c98a00"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4))

        def arrow(ax, v, color, label, start=(0, 0), ls="-", lw=1.6, dy=0.12):
            ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls))
            ax.text(start[0] + v[0] + 0.08, start[1] + v[1] + dy, label, color=color, fontsize=8.5)

        # 左：方法一，「猫」在第 0 位（上一行）与第 2 位（下一行）
        ax1.set_title("方法一：加位置向量，同一个「猫」在第 0 位和第 2 位是两个向量", fontsize=8.5)
        for y, pos, color, name, sent in [(0.7, 0, BLUE, "猫@0 = (3, 0)", "「猫追狗」"), (-0.7, 2, RED, "猫@2 = (1, 0)", "「狗追猫」")]:
            arrow(ax1, TOK["猫"], GRAY, "猫 (2, 0)", start=(0, y), dy=0.12 if y > 0 else -0.4)
            sign = "+" if POS[pos][0] > 0 else "−"
            arrow(ax1, POS[pos], color, f"+ 位置 {pos} ({sign}1, 0)", start=(TOK["猫"][0], y + (0.35 if y > 0 else -0.35)), ls="--", lw=1.2, dy=0.1 if y > 0 else -0.4)
            ax1.plot([0, TOK["猫"][0] + POS[pos][0]], [y - 0.35 if y > 0 else y + 0.35] * 2, color=color, lw=2.2, solid_capstyle="round")
            ax1.text(TOK["猫"][0] + POS[pos][0] + 0.1, (y - 0.35 if y > 0 else y + 0.35) - 0.08, f"= {name}  {sent}", color=color, fontsize=8.5)
        ax1.set_xlim(-0.3, 5.2)
        ax1.set_ylim(-1.6, 1.6)
        ax1.set_aspect("equal")
        ax1.axvline(0, color="#ddd", lw=0.8, zorder=0)

        # 右：方法二，猫 看 狗：狗 相对 猫 转 +60° / −60°
        ax2.set_title("方法二：转角度，「狗」相对「猫」转 +60° 或 −60°，内积不同", fontsize=8.5)
        q = TOK["猫"]
        k_fwd = rot(2 * THETA) @ TOK["狗"]
        k_bwd = rot(-2 * THETA) @ TOK["狗"]
        arrow(ax2, q, GRAY, "q = 猫 (2, 0)", dy=-0.35)
        arrow(ax2, TOK["狗"], GRAY, "狗 (2, 1) 未转", ls=":", lw=1.2, dy=0.1)
        arrow(ax2, k_fwd, BLUE, f"k 转 +60°  「猫追狗」\nq·k = {float(q @ k_fwd):.2f}", dy=0.05)
        arrow(ax2, k_bwd, RED, f"k 转 −60°  「狗追猫」\nq·k = {float(q @ k_bwd):.2f}", dy=-0.55)
        ax2.add_patch(Arc((0, 0), 1.6, 1.6, theta1=np.degrees(np.arctan2(1, 2)),
                          theta2=np.degrees(np.arctan2(k_fwd[1], k_fwd[0])), color=BLUE, lw=1))
        ax2.add_patch(Arc((0, 0), 1.6, 1.6, theta1=np.degrees(np.arctan2(k_bwd[1], k_bwd[0])),
                          theta2=np.degrees(np.arctan2(1, 2)), color=RED, lw=1))
        ax2.set_xlim(-0.6, 4.6)
        ax2.set_ylim(-2.0, 2.9)
        ax2.set_aspect("equal")
        ax2.axhline(0, color="#ddd", lw=0.8, zorder=0)
        ax2.axvline(0, color="#ddd", lw=0.8, zorder=0)
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        fig.savefig(sys.argv[1], format="svg", bbox_inches="tight", metadata={"Date": None})
        print(f"\n  图已保存 {sys.argv[1]}")
