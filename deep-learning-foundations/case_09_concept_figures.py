"""案例（深度学习基础 09）：Embedding 与表示学习的概念图与手算数字。

https://arganzheng.life/embedding-and-representation-learning.html

    python case_09_concept_figures.py

输出三张概念图到 out/，并打印第四章 toy 语料的全部数字
（共现矩阵、PPMI、SVD 二维坐标）。
线型统一实线，只用 Okabe-Ito 颜色区分。
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch

from _plot import C, plt, save

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#7F7F7F"

BOX_FC = "#f4f7fb"
BOX_EC = "#7a8ca5"


def box(ax, center, text, width=6, height=5, fontsize=13, fc=BOX_FC,
        ec=BOX_EC):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.8", facecolor=fc, edgecolor=ec, linewidth=1.1,
    )
    ax.add_patch(patch)
    label = ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return patch, label


def arrow(ax, start, end, color=C["gray"], lw=1.2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": color, "lw": lw})


def new_diagram(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def check_layout(fig, ax, patches, inner):
    """标签不压框、不互压，框内文字留边距。"""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    box_exts = [p.get_window_extent(rend) for p in patches]
    labels = [t for t in ax.texts if t not in inner and t.get_text()]
    lab_exts = [t.get_window_extent(rend) for t in labels]

    def padded(e, pad=2):
        return Bbox.from_extents(e.x0 - pad, e.y0 - pad,
                                 e.x1 + pad, e.y1 + pad)

    for t, e in zip(labels, lab_exts):
        for b in box_exts:
            assert not padded(e).overlaps(b), (
                f"label overlaps box: {t.get_text()!r}")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert not padded(lab_exts[i]).overlaps(lab_exts[j]), (
                f"labels overlap: {labels[i].get_text()!r} / "
                f"{labels[j].get_text()!r}")
    for p, t in zip(patches, inner):
        pe = p.get_window_extent(rend)
        te = t.get_window_extent(rend)
        assert (te.x0 >= pe.x0 + 1 and te.x1 <= pe.x1 - 1
                and te.y0 >= pe.y0 + 1 and te.y1 <= pe.y1 - 1), (
                f"text overflows its box: {t.get_text()!r}")


# ---------------------------------------------------------------- toy 语料

WORDS = ["猫", "狗", "鱼", "吃", "追", "睡"]
CORPUS = ["猫 追 狗", "狗 追 猫", "狗 追 猫", "猫 吃 鱼", "猫 睡"]


def toy_counts():
    idx = {w: i for i, w in enumerate(WORDS)}
    X = np.zeros((len(WORDS), len(WORDS)))
    for sent in CORPUS:
        ws = sent.split()
        for a, b in zip(ws, ws[1:]):
            X[idx[a], idx[b]] += 1
            X[idx[b], idx[a]] += 1
    return X


def toy_ppmi(X):
    n = X.sum()
    row = X.sum(axis=1, keepdims=True)
    col = X.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore"):
        pmi = np.log2(n * X / (row @ col))
    return np.maximum(0, np.where(np.isfinite(pmi), pmi, 0))


def plot_ppmi_svd():
    """FIGURE：toy 语料的共现矩阵 → PPMI → SVD 二维词图。"""
    X = toy_counts()
    P = toy_ppmi(X)
    U, S, _ = np.linalg.svd(P)
    coords = U[:, :2] * np.sqrt(S[:2])

    # ---- 打印正文引用的数字 ----
    print("  toy 语料：", "；".join(CORPUS))
    print("  共现矩阵 X（行列：", "/".join(WORDS), "）：")
    for w, r in zip(WORDS, X):
        print("   ", w, " ".join(f"{v:4.0f}" for v in r))
    print("  PPMI（log2，负数取 0）：")
    for w, r in zip(WORDS, P):
        print("   ", w, " ".join(f"{v:5.2f}" for v in r))
    print("  SVD 二维坐标（u * sqrt(s)）：")
    for w, c in zip(WORDS, coords):
        print(f"    {w}: ({c[0]:.2f}, {c[1]:.2f})")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = {"猫": BLUE, "狗": SKY, "鱼": ORANGE,
              "吃": GREEN, "追": GREEN, "睡": GREY}
    for w, (x, y) in zip(WORDS, coords):
        ax.scatter([x], [y], s=46, color=colors[w], zorder=3)
        ax.annotate(w, (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=12, color=colors[w])
    ax.axhline(0, color=C["light"], lw=0.8)
    ax.axvline(0, color=C["light"], lw=0.8)
    ax.set_xlabel("SVD 第 1 维")
    ax.set_ylabel("SVD 第 2 维")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text()]
    exts = [t.get_window_extent(rend) for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            assert not exts[i].overlaps(exts[j]), (
                f"labels overlap: {texts[i].get_text()} / {texts[j].get_text()}")
    save(fig, "emb-ppmi-svd-toy")


# ---------------------------------------------------------------- 全景图

def plot_landscape():
    """FIGURE：embedding 全景——嵌入什么 → 怎么得到 → 向量空间。"""
    fig, ax = new_diagram((9.2, 3.4), (0, 120), (0, 46))
    patches, inner = [], []

    def bx_(center, text, **kw):
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)
        return p, t

    objs = [
        ("词", "五", 40), ("子词", "六", 34),
        ("句子 / 文本", "八", 27), ("图像", "九", 20),
        ("位置", "九", 14), ("图节点与实体", "九", 8),
        ("ID 与类别特征", "九", 2),
    ]
    fams = [
        ("计数与分解", "LSA · PPMI+SVD · GloVe", ORANGE, 36),
        ("预测", "Word2Vec · fastText · LM", BLUE, 27),
        ("上下文", "ELMo · GPT · BERT", SKY, 18),
        ("对比", "SBERT · SimCSE · CLIP", GREEN, 9),
    ]
    for name, sec, y in objs:
        bx_((14, y), f"{name}\n第{sec}章", width=19, height=5.4,
            fontsize=9)
    for name, sub, color, y in fams:
        bx_((62, y), f"{name}\n{sub}", width=30, height=6.4, fontsize=9,
            fc="#ffffff", ec=color)

    vec = bx_((102, 19), "d 维向量空间", width=19, height=30, fontsize=11)
    for fam_name, sub, color, fy in fams:
        arrow(ax, (71.5, fy), (91, fy), color=color)
    # 对象 → 家族的连线（每条一个颜色，取所在家族的颜色）
    fam_of_obj = {"词": [0, 1, 2], "子词": [1],
                  "句子 / 文本": [2, 3], "图像": [3],
                  "位置": [2], "图节点与实体": [1],
                  "ID 与类别特征": [0, 1]}
    fam_c = {0: ORANGE, 1: BLUE, 2: SKY, 3: GREEN}
    fam_y = {0: 36, 1: 27, 2: 18, 3: 9}
    for name, sec, y in objs:
        for f in fam_of_obj[name]:
            ax.plot([24, 52.5], [y, fam_y[f]], color=fam_c[f], lw=1.1,
                    zorder=1, alpha=0.55)
    ax.text(38, 44.5, "嵌入什么 → 用什么信号得到向量",
            ha="center", va="center", fontsize=10, color="#444444")

    check_layout(fig, ax, patches, inner)
    save(fig, "emb-landscape")


# ---------------------------------------------------------------- 监督信号

def plot_signals():
    """FIGURE：五种监督信号各画一格：输入 → 目标。"""
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7))
    titles = ["预测", "重建", "对比", "自蒸馏", "有监督"]
    subs = ["前文 → 下一个词", "x → z → 还原 x", "正例近、负例远",
            "无负样本，teacher 走 EMA", "x → 标签 y"]
    for ax, t, s in zip(axes, titles, subs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(t, fontsize=11)
        ax.text(5, 0.4, s, ha="center", va="bottom", fontsize=8, color=GREY)

    patches, inner = [], []

    def bx_(ax, center, text, **kw):
        kw.setdefault("fontsize", 8.5)
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)

    a = axes[0]
    bx_(a, (2.4, 6.5), "猫 追 __", width=4.6, height=2.6)
    bx_(a, (7.6, 6.5), "狗", width=3.2, height=2.6)
    arrow(a, (4.9, 6.5), (5.9, 6.5))
    bx_(a, (2.4, 3.2), "猫 __ 球", width=4.6, height=2.6)
    bx_(a, (7.6, 3.2), "追", width=3.2, height=2.6)
    arrow(a, (4.9, 3.2), (5.9, 3.2))
    a.text(5, 8.8, "下一个词 / mask 处", ha="center", fontsize=8,
           color=GREY)

    a = axes[1]
    bx_(a, (1.8, 5.5), "$x$", width=2.8, height=3.2)
    bx_(a, (5.0, 5.5), "$z$", width=2.2, height=3.2, fc="#fdeee0",
        ec=ORANGE)
    bx_(a, (8.2, 5.5), "$\\hat{x}$", width=2.8, height=3.2)
    arrow(a, (3.3, 5.5), (3.8, 5.5))
    arrow(a, (6.2, 5.5), (6.7, 5.5))
    a.text(5, 8.8, "瓶颈迫使保留信息", ha="center", fontsize=8, color=GREY)

    a = axes[2]
    bx_(a, (2.2, 6.6), "query", width=4.0, height=2.4)
    bx_(a, (7.4, 7.6), "$k^+$", width=3.0, height=2.4, fc="#e6f4ee",
        ec=GREEN)
    bx_(a, (7.4, 3.6), "$k^-$", width=3.0, height=2.4, fc="#fdeee0",
        ec=VERMILION)
    arrow(a, (4.4, 7.0), (5.7, 7.5), color=GREEN)
    arrow(a, (4.4, 6.2), (5.7, 4.0), color=VERMILION)

    a = axes[3]
    bx_(a, (2.0, 7.2), "$x$ 视图1", width=3.4, height=2.4)
    bx_(a, (2.0, 3.4), "$x$ 视图2", width=3.4, height=2.4)
    bx_(a, (6.2, 7.2), "student", width=3.8, height=2.4)
    bx_(a, (6.2, 3.4), "teacher", width=3.8, height=2.4, fc="#e6f4ee",
        ec=GREEN)
    arrow(a, (3.9, 7.2), (4.2, 7.2))
    arrow(a, (3.9, 3.4), (4.2, 3.4), color=GREEN)

    a = axes[4]
    bx_(a, (2.2, 5.5), "$x$", width=3.0, height=3.2)
    bx_(a, (7.4, 5.5), "猫", width=3.0, height=3.2, fc="#e6f4ee",
        ec=GREEN)
    arrow(a, (3.9, 5.5), (5.7, 5.5))
    a.text(5, 8.8, "ImageNet 类标注", ha="center", fontsize=8, color=GREY)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    box_exts = [p.get_window_extent(rend) for p in patches]
    labels = [t for t in fig.texts for _ in [0]]
    all_texts = []
    for ax in axes:
        all_texts += [ax.title]
        all_texts += [t for t in ax.texts if t.get_text()]
    all_texts = [t for t in all_texts if t.get_text()]
    exts = [t.get_window_extent(rend) for t in all_texts]
    for t, e in zip(all_texts, exts):
        pe = Bbox.from_extents(e.x0 - 2, e.y0 - 2, e.x1 + 2, e.y1 + 2)
        for b in box_exts:
            if t in inner:
                continue
            assert not pe.overlaps(b), f"label overlaps box: {t.get_text()!r}"
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            e1 = Bbox.from_extents(exts[i].x0 - 2, exts[i].y0 - 2,
                                   exts[i].x1 + 2, exts[i].y1 + 2)
            assert not e1.overlaps(exts[j]), (
                f"labels overlap: {all_texts[i].get_text()!r} / "
                f"{all_texts[j].get_text()!r}")
    for p, t in zip(patches, inner):
        pe = p.get_window_extent(rend)
        te = t.get_window_extent(rend)
        assert (te.x0 >= pe.x0 + 1 and te.x1 <= pe.x1 - 1
                and te.y0 >= pe.y0 + 1 and te.y1 <= pe.y1 - 1), (
                f"text overflows its box: {t.get_text()!r}")

    save(fig, "emb-supervision-signals")


def main():
    plot_ppmi_svd()
    plot_landscape()
    plot_signals()


if __name__ == "__main__":
    main()
