"""案例（深度学习基础 09）：Embedding 与表示学习的概念图与手算数字。

https://arganzheng.life/embedding-and-representation-learning.html

    python case_09_concept_figures.py

输出三张概念图到 out/，并打印第四章 toy 语料的全部数字
（共现矩阵、PPMI、SVD 二维坐标）。
线型统一实线，只用 Okabe-Ito 颜色区分。
"""
import itertools

import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

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
        ec=BOX_EC, pad=0.8):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle=f"round,pad={pad}", facecolor=fc, edgecolor=ec,
        linewidth=1.1,
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


def padded(e, pad=2):
    return Bbox.from_extents(e.x0 - pad, e.y0 - pad,
                             e.x1 + pad, e.y1 + pad)


def check_layout(fig, axes, patches, inner):
    """断言：标签不压框不互压，框不互压，框内文字留边，所有元素不出图。"""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    figbox = Bbox.from_extents(0, 0, W, H)

    box_exts = [p.get_window_extent(rend) for p in patches]
    labels = []
    for ax in np.atleast_1d(axes):
        labels += [t for t in ax.texts
                   if t not in inner and t.get_text()]
        labels += [ax.title] if ax.get_title() else []
    labels = [t for t in labels if t.get_text()]
    lab_exts = [t.get_window_extent(rend) for t in labels]

    for t, e in zip(labels, lab_exts):
        for b in box_exts:
            assert not padded(e).overlaps(b), (
                f"label overlaps box: {t.get_text()!r}")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert not padded(lab_exts[i]).overlaps(lab_exts[j]), (
                f"labels overlap: {labels[i].get_text()!r} / "
                f"{labels[j].get_text()!r}")
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            assert not padded(box_exts[i]).overlaps(box_exts[j]), (
                f"boxes overlap: #{i} / #{j}")
    for p, t in zip(patches, inner):
        pe = p.get_window_extent(rend)
        te = t.get_window_extent(rend)
        assert (te.x0 >= pe.x0 + 1 and te.x1 <= pe.x1 - 1
                and te.y0 >= pe.y0 + 1 and te.y1 <= pe.y1 - 1), (
                f"text overflows its box: {t.get_text()!r}")
    for e in box_exts + lab_exts:
        assert figbox.contains(e.x0, e.y0) and figbox.contains(e.x1, e.y1), (
            "artist leaves the figure canvas")


# ---------------------------------------------------------------- toy 语料

WORDS = ["猫", "狗", "鱼", "球", "追", "吃", "睡"]
CORPUS = ["猫 狗 吃 鱼", "狗 猫 吃 鱼", "猫 追 球", "狗 追 球",
          "狗 吃 球", "猫 睡", "狗 睡"]


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
    idx = {w: i for i, w in enumerate(WORDS)}

    def dist(a, b):
        return float(np.hypot(*(coords[idx[a]] - coords[idx[b]])))

    # 几何必须支持正文的说法：猫-狗 距离明显小于各自到物体名词的距离
    for obj in ("鱼", "球"):
        assert dist("猫", "狗") < dist("猫", obj) * 0.6
        assert dist("猫", "狗") < dist("狗", obj) * 0.6

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
    print("  距离：猫-狗 %.2f；猫-鱼 %.2f；猫-球 %.2f；狗-鱼 %.2f；狗-球 %.2f"
          % (dist("猫", "狗"), dist("猫", "鱼"), dist("猫", "球"),
             dist("狗", "鱼"), dist("狗", "球")))

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = {"猫": BLUE, "狗": SKY, "鱼": ORANGE, "球": VERMILION,
              "追": GREEN, "吃": GREEN, "睡": GREY}
    offsets = {"猫": (-16, -6), "狗": (9, 8), "鱼": (-16, 4), "球": (9, 8),
               "追": (9, -4), "吃": (9, -12), "睡": (-16, -4)}
    for w, (x, y) in zip(WORDS, coords):
        ax.scatter([x], [y], s=46, color=colors[w], zorder=3)
        ax.annotate(w, (x, y), xytext=offsets[w],
                    textcoords="offset points",
                    fontsize=12, color=colors[w])
    mx, my = np.ptp(coords[:, 0]), np.ptp(coords[:, 1])
    ax.set_xlim(coords[:, 0].min() - 0.25 * mx, coords[:, 0].max() + 0.3 * mx)
    ax.set_ylim(coords[:, 1].min() - 0.25 * my, coords[:, 1].max() + 0.25 * my)
    ax.set_xlabel("SVD 第 1 维")
    ax.set_ylabel("SVD 第 2 维")

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    texts = [t for t in ax.texts if t.get_text()] + [ax.xaxis.label,
                                                     ax.yaxis.label]
    exts = [t.get_window_extent(rend) for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            assert not padded(exts[i]).overlaps(exts[j]), (
                f"labels overlap: {texts[i].get_text()} / "
                f"{texts[j].get_text()}")
    for t, e in zip(texts, exts):
        assert (e.x0 >= 0 and e.y0 >= 0 and e.x1 <= W and e.y1 <= H), (
            f"text leaves the figure: {t.get_text()!r}")
    save(fig, "emb-ppmi-svd-toy")


# ---------------------------------------------------------------- 全景图

# 对象 → 家族索引；位置没有有意义的连线，单独标注
FAM_ORDER = ["计数与分解", "预测", "上下文", "对比"]
FAM_SUBS = ["LSA · PPMI+SVD · GloVe", "Word2Vec · fastText · LM",
            "ELMo · GPT · BERT", "SBERT · SimCSE · CLIP"]
FAM_COLORS = [ORANGE, BLUE, SKY, GREEN]
OBJS = [("词", "五", [0, 1]),
        ("子词", "六", [1]),
        ("句子 / 文本", "八", [2, 3]),
        ("图像", "九", [3]),
        ("图节点与实体", "九", [1, 3]),
        ("ID 与类别特征", "九", [0, 3]),
        ("位置", "九", [])]  # 位置固定垫底，旁边注一句


def _best_obj_order():
    """枚举 6 个有连线对象的顺序，使对象→家族连线的交叉最少。"""
    heads = OBJS[:-1]
    fam_y = {f: len(FAM_ORDER) - 1 - f for f in range(len(FAM_ORDER))}

    def crossings(order):
        segs = []
        for i, (name, _, fams) in enumerate(order):
            y = len(order) - 1 - i
            for f in fams:
                segs.append((y, fam_y[f]))
        n = 0
        for (a1, a2), (b1, b2) in itertools.combinations(segs, 2):
            if (a1 - b1) * (a2 - b2) < 0:
                n += 1
        return n

    best, best_n = None, None
    for perm in itertools.permutations(range(len(heads))):
        order = [heads[i] for i in perm]
        n = crossings(order)
        if best_n is None or n < best_n:
            best, best_n = order, n
    return best


def plot_landscape():
    """FIGURE：embedding 全景——嵌入什么 → 怎么得到 → 向量空间。"""
    ordered = _best_obj_order() + [OBJS[-1]]
    n = len(ordered)
    pitch = 8.2
    top = (n - 1) * pitch
    fam_pitch = 11.5
    fam_top = 3 * fam_pitch

    fig, ax = new_diagram((9.2, 4.6), (0, 118), (-4, top + 5))
    patches, inner = [], []

    def bx_(center, text, **kw):
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)
        return p, t

    obj_y = {name: top - i * pitch for i, (name, _, _) in enumerate(ordered)}
    fam_y = {f: fam_top - f * fam_pitch for f in range(len(FAM_ORDER))}

    for name, sec, _ in ordered:
        bx_((13, obj_y[name]), f"{name}\n第{sec}章", width=19,
            height=5.4, fontsize=9)
    for f, (fam, sub, color) in enumerate(zip(FAM_ORDER, FAM_SUBS,
                                              FAM_COLORS)):
        bx_((60, fam_y[f]), f"{fam}\n{sub}", width=30, height=6.4,
            fontsize=9, fc="#ffffff", ec=color)
    vec_y = (fam_top + fam_y[3]) / 2
    bx_((99, vec_y), "d 维向量空间", width=19, height=top * 0.62,
        fontsize=11)

    # 对象 → 家族：连线的颜色取目标家族的颜色
    for name, sec, fams in ordered:
        for f in fams:
            ax.plot([23.5, 44.5], [obj_y[name], fam_y[f]],
                    color=FAM_COLORS[f], lw=1.1, zorder=1, alpha=0.6)
    for f in range(len(FAM_ORDER)):
        arrow(ax, (76.5, fam_y[f]), (88.5, fam_y[f]),
              color=FAM_COLORS[f])
    ax.text(12.5, obj_y["位置"] - 4.6,
            "固定公式或随模型一起训", ha="center", va="top",
            fontsize=7.5, color=GREY)
    ax.text(48, top + 3.4, "嵌入什么 → 用什么信号得到向量",
            ha="center", va="center", fontsize=10, color="#444444")

    check_layout(fig, ax, patches, inner)
    save(fig, "emb-landscape")


# ---------------------------------------------------------------- 监督信号

def plot_signals():
    """FIGURE：五种监督信号各画一格：输入 → 目标。"""
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.6))
    titles = ["预测", "重建", "对比", "自蒸馏", "有监督"]
    subs = ["前文 → 下一个词 / mask 处", "x → z → 还原 x",
            "正例近、负例远", "无负样本，teacher 走 EMA", "x → 标签 y"]
    for ax, t, s in zip(axes, titles, subs):
        ax.set_xlim(0, 12)
        ax.set_ylim(0.2, 9.6)
        ax.axis("off")
        ax.set_title(t, fontsize=11)
        ax.text(6, 0.4, s, ha="center", va="bottom", fontsize=8,
                color=GREY)

    patches, inner = [], []

    def bx_(ax, center, text, **kw):
        kw.setdefault("fontsize", 8.5)
        kw.setdefault("pad", 0.35)
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)

    a = axes[0]
    bx_(a, (3.0, 7.4), "猫 追 __", width=5.0, height=2.4)
    bx_(a, (9.7, 7.4), "狗", width=3.0, height=2.4)
    arrow(a, (5.7, 7.4), (8.0, 7.4))
    bx_(a, (3.0, 3.0), "猫 __ 球", width=5.0, height=2.4)
    bx_(a, (9.7, 3.0), "追", width=3.0, height=2.4)
    arrow(a, (5.7, 3.0), (8.0, 3.0))

    a = axes[1]
    bx_(a, (2.2, 5.4), "$x$", width=2.8, height=3.0)
    bx_(a, (6.0, 5.4), "$z$", width=2.2, height=3.0, fc="#fdeee0",
        ec=ORANGE)
    bx_(a, (9.8, 5.4), "$\\hat{x}$", width=2.8, height=3.0)
    arrow(a, (3.7, 5.4), (4.8, 5.4))
    arrow(a, (7.2, 5.4), (8.3, 5.4))

    a = axes[2]
    bx_(a, (2.8, 5.4), "query", width=4.4, height=2.6)
    bx_(a, (9.7, 7.3), "$k^+$", width=3.4, height=2.3, fc="#e6f4ee",
        ec=GREEN)
    bx_(a, (9.7, 3.3), "$k^-$", width=3.4, height=2.3, fc="#fdeee0",
        ec=VERMILION)
    arrow(a, (5.2, 6.0), (7.8, 7.1), color=GREEN)
    arrow(a, (5.2, 4.8), (7.8, 3.5), color=VERMILION)

    a = axes[3]
    bx_(a, (3.0, 7.4), "$x$ 视图1", width=4.4, height=2.3)
    bx_(a, (3.0, 3.0), "$x$ 视图2", width=4.4, height=2.3)
    bx_(a, (9.5, 7.4), "student", width=3.8, height=2.3)
    bx_(a, (9.5, 3.0), "teacher", width=3.8, height=2.3,
        fc="#e6f4ee", ec=GREEN)
    arrow(a, (5.4, 7.4), (7.4, 7.4))
    arrow(a, (5.4, 3.0), (7.4, 3.0), color=GREEN)

    a = axes[4]
    bx_(a, (3.0, 5.4), "$x$", width=3.4, height=3.0)
    bx_(a, (9.5, 5.4), "猫", width=3.4, height=3.0, fc="#e6f4ee",
        ec=GREEN)
    arrow(a, (5.0, 5.4), (7.6, 5.4))
    a.text(6, 8.9, "ImageNet 类标注", ha="center", fontsize=8,
           color=GREY)

    check_layout(fig, axes, patches, inner)
    save(fig, "emb-supervision-signals")


def main():
    plot_ppmi_svd()
    plot_landscape()
    plot_signals()


if __name__ == "__main__":
    main()
