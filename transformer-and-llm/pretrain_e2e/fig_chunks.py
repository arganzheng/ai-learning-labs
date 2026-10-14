"""打包示意：文档流按 <|endoftext|> 相接 → 每 256 token 切一条序列 → 训练时随机起点取样。

输出 out/e2e-5-chunks.svg。一篇 1000 token 的文档被切成 4 段，落在不同的批里。

断言（与 deep-learning-foundations 的概念图同一套）：
  - 框与框不重叠、标签与框/标签与标签之间至少 2px
  - 所有 artist 都在图内，bbox_inches="tight" 收边
"""
import numpy as np
from _plot import plt, save, C

rng = np.random.default_rng(7)

DOC_A = ("一篇 1,000 token 的文档", 4, C["orange"])   # 目标文档：4 段
OTHER = [("文档 A", 2, C["light"]), ("文档 B", 3, C["light"]), ("文档 C", 1, C["light"])]


def box(ax, x, y, w, h, fc, text=None, pad=0.35, fs=8, tc="black"):
    p = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                      linewidth=0.8, joinstyle="round", zorder=2)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)
    return p


def main():
    fig = plt.figure(figsize=(7.6, 3.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    patches, texts = [], []

    # ---- row 1: document stream -------------------------------------------------
    ax.text(4, 84, "文档流（<|endoftext|> 分隔）", fontsize=8, ha="left", va="center")
    x = 4
    units = [("A", 9), ("B", 13), ("·1,000 token·", 22), ("C", 5)]
    cols = {"A": C["light"], "B": C["light"], "C": C["light"], "·1,000 token·": C["orange"]}
    for name, w in units:
        patches.append(box(ax, x, 70, w, 10, cols[name],
                           "文档" if name != "·1,000 token·" else name, fs=7.5))
        x += w
        patches.append(box(ax, x, 70, 1.6, 10, C["red"], "|", fs=7, tc="white"))
        x += 1.6
    ax.text(x + 1, 75, "……", fontsize=10, ha="left", va="center")

    # ---- row 2: fixed 256-token windows -----------------------------------------
    ax.text(4, 52, "每 256 token 切一条序列", fontsize=8, ha="left", va="center")
    x = 4
    seqs = [("①", 12.6), ("②", 12.6), ("③", 12.6), ("④", 12.6)]
    marks = [(0, False), (1, True), (2, True), (3, True)]
    # 目标文档的 4 段落在第 ②③④ 条序列与第 ⑤ 条里——用 4 段橙色表示
    for i, (lab, w) in enumerate(seqs + [("⑤", 12.6)]):
        fc = C["orange"] if i in (1, 2, 3, 4) else C["light"]
        patches.append(box(ax, x, 38, w, 10, fc, f"序列{lab}", fs=7.5))
        x += w + 1.4
    ax.text(4, 32, "（序列可跨文档边界：一条 1,000 token 的文档被切成 4 段，散在 4 条序列里，互不为上下文）",
            fontsize=7.5, ha="left", va="center", color="#a05a10")

    # ---- row 3: batches ----------------------------------------------------------
    ax.text(4, 16, "训练时从 token 流里随机起点取 256 长的窗口，32 条成一个批（它的 4 段落进不同的批）",
            fontsize=8, ha="left", va="center")
    bx = 30
    for i, label in enumerate(["批 k", "批 k+1", "批 k+2", "…"]):
        fc = C["orange"] if i < 3 else C["light"]
        patches.append(box(ax, bx + i * 17, 4, 8, 8, fc, label, fs=7.5))
    ax.text(4, 8, "32 条 / 批", fontsize=7.5,
            ha="left", va="center", color="black")

    # ---- assertions --------------------------------------------------------------
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    finv = fig.transFigure.inverted()
    def bb(a):
        b = a.get_window_extent(renderer=r)
        return finv.transform(b)
    def overlap(b1, b2, eps=1e-4):
        ix = min(b1[1, 0], b2[1, 0]) - max(b1[0, 0], b2[0, 0])
        iy = min(b1[1, 1], b2[1, 1]) - max(b1[0, 1], b2[0, 1])
        return ix > eps and iy > eps  # 共边不算重叠
    pbs = [bb(p) for p in patches]
    for i in range(len(pbs)):
        for j in range(i + 1, len(pbs)):
            assert not overlap(pbs[i], pbs[j]), f"box {i} 与 box {j} 重叠"
    for a in ax.texts:
        tb = bb(a)
        assert (tb >= 0).all() and (tb <= 1).all(), f"文本 {a.get_text()!r} 超出画布"
        for pb in pbs:
            in_x = tb[0, 0] < pb[1, 0] and tb[1, 0] > pb[0, 0]
            in_y = tb[0, 1] < pb[1, 1] and tb[1, 1] > pb[0, 1]
            if in_x and in_y:
                # 文本压在框上：只允许框内的居中文字（其包围盒完全在框内）
                inside = (tb[0, 0] >= pb[0, 0] - 0.002 and tb[1, 0] <= pb[1, 0] + 0.002
                          and tb[0, 1] >= pb[0, 1] - 0.002 and tb[1, 1] <= pb[1, 1] + 0.002)
                assert inside, f"文本 {a.get_text()!r} 与框交叠但不在框内"
    save(fig, "e2e-5-chunks")


if __name__ == "__main__":
    main()
