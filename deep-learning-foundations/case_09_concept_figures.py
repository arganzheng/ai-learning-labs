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

# 两条轴：嵌入什么（对象）→ 用什么信号得到向量（七种信号）
FAM_ORDER = ["计数分解", "预测", "重建", "对比", "自蒸馏", "有监督", "固定公式"]
FAM_SUBS = ["LSA · GloVe · 矩阵分解", "Word2Vec · BERT · GPT · MAE",
            "AE · VAE", "CLIP · SimCSE · TransE · 双塔",
            "BYOL · DINO", "ImageNet · NLI · CTR", "正弦 · RoPE"]
FAM_COLORS = [ORANGE, BLUE, SKY, GREEN, PURPLE, VERMILION, GREY]
OBJS = [("词", "四、五", [0, 1]),
        ("子词", "六", [1]),
        ("句子 / 文本", "八", [1, 3, 5]),
        ("图像（ViT patch）", "九", [1, 2, 3, 4, 5]),
        ("位置", "九", [6, 1]),
        ("图节点与实体", "九", [1, 3]),
        ("ID 与类别特征", "九", [0, 3, 5])]


def _best_obj_order():
    """枚举 7 个对象的顺序，使对象→信号连线的交叉最少。"""
    heads = OBJS
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
    """FIGURE：embedding 全景——嵌入什么 → 用哪种信号得到 → 向量空间。"""
    ordered = _best_obj_order()
    n = len(ordered)
    pitch = 8.2
    top = (n - 1) * pitch
    fam_pitch = pitch * n / len(FAM_ORDER)
    fam_top = (len(FAM_ORDER) - 1) * fam_pitch

    fig, ax = new_diagram((9.4, 5.4), (0, 118), (-4, top + 10))
    patches, inner = [], []

    def bx_(center, text, **kw):
        p, t = box(ax, center, text, **kw)
        patches.append(p)
        inner.append(t)
        return p, t

    obj_y = {name: top - i * pitch for i, (name, _, _) in enumerate(ordered)}
    fam_y = {f: fam_top - f * fam_pitch for f in range(len(FAM_ORDER))}

    for name, sec, _ in ordered:
        bx_((13, obj_y[name]), f"{name}\n第{sec}章", width=21,
            height=5.4, fontsize=8.5)
    for f, (fam, sub, color) in enumerate(zip(FAM_ORDER, FAM_SUBS,
                                              FAM_COLORS)):
        bx_((60, fam_y[f]), f"{fam}\n{sub}", width=31, height=5.6,
            fontsize=8, fc="#ffffff", ec=color)
    vec_y = fam_top / 2
    bx_((99, vec_y), "d 维向量空间", width=19, height=top * 0.62,
        fontsize=11)

    # 对象 → 信号：连线的颜色取目标信号的颜色
    for name, sec, fams in ordered:
        for f in fams:
            ax.plot([25, 43.5], [obj_y[name], fam_y[f]],
                    color=FAM_COLORS[f], lw=1.1, zorder=1, alpha=0.6)
    for f in range(len(FAM_ORDER)):
        arrow(ax, (77.5, fam_y[f]), (88.5, fam_y[f]),
              color=FAM_COLORS[f])
    ax.text(50, top + 5.6, "嵌入什么 → 用什么信号得到向量",
            ha="center", va="center", fontsize=10, color="#444444")

    check_layout(fig, ax, patches, inner)
    save(fig, "emb-landscape")


# ------------------------------------------------------------- 负采样手算

def negsample_numbers(k=5, d=8, steps=15, lr=0.2, seed=0):
    """中心词「追」、正样本上下文「猫」、k 个噪声词；几轮 SGD 后的 σ 分数。"""
    rng = np.random.default_rng(seed)
    center = "追"; pos = "猫"; negs = ["球", "狗", "鱼", "跑", "睡"]
    words = [center, pos] + negs
    vin = {w: rng.normal(0, 0.1, d) for w in words}   # 中心词表
    vout = {w: rng.normal(0, 0.1, d) for w in words}  # 上下文（输出）表
    sig = lambda x: 1 / (1 + np.exp(-x))
    before = {w: sig(vout[w] @ vin[center]) for w in [pos] + negs}
    for _ in range(steps):
        vc = vin[center]
        g_in = (sig(vout[pos] @ vc) - 1) * vout[pos]
        g_out = {}
        g_out[pos] = (sig(vout[pos] @ vc) - 1) * vc
        for n in negs:
            s = sig(vout[n] @ vc)
            g_in += s * vout[n]
            g_out[n] = s * vc
        vin[center] -= lr * g_in
        for w in [pos] + negs:
            vout[w] -= lr * g_out[w]
    after = {w: sig(vout[w] @ vin[center]) for w in [pos] + negs}
    for w in [pos] + negs:
        print(f"neg-sampling {center}-{w}: before={before[w]:.3f} "
              f"after={after[w]:.3f}")
    return center, pos, negs, before, after


def plot_negsample():
    """FIGURE：一个正样本对 + 5 个负样本，σ 分数一上一下。"""
    center, pos, negs, before, after = negsample_numbers()
    n = len(negs)
    pitch = 9.0
    top = n * pitch

    fig, ax = new_diagram((7.6, 3.4), (0, 100), (-8, top + 8))
    patches, inner = [], []

    def bx_(center_, text, **kw):
        p, t = box(ax, center_, text, **kw)
        patches.append(p)
        inner.append(t)
        return p, t

    mid = top / 2
    bx_((16, mid), f"中心词\n{center}", width=15, height=7,
        fontsize=10, fc="#eef3fa")
    bx_((60, top), f"正样本：{pos}\nσ = {after[pos]:.2f}",
        width=24, height=6.2, fontsize=9.5, fc="#ffffff", ec=GREEN)
    arrow(ax, (24.5, mid + 1.6), (46.5, top - 0.4), color=GREEN)
    for i, w in enumerate(negs):
        y = top - pitch * (i + 1) - 1.0
        bx_((60, y), f"负样本 {i + 1}：{w}  σ = {after[w]:.2f}",
            width=27, height=5.2, fontsize=9, fc="#ffffff", ec=VERMILION)
        ax.plot([24.5, 45.5], [mid - 1.2, y + 0.6],
                color=VERMILION, lw=1.1, zorder=1, alpha=0.6)
    bx_((88, mid), "目标：\n正样本推上去\n负样本压下来", width=19,
        height=11, fontsize=9)
    arrow(ax, (74.5, top), (78, mid + 3), color=GREEN)
    arrow(ax, (74.5, 0), (78, mid - 3), color=VERMILION)
    ax.text(50, top + 5.4, "随机初始化 + 15 步 SGD 之后的 σ 分数",
            ha="center", va="center", fontsize=8.5, color="#444444")

    check_layout(fig, ax, patches, inner)
    save(fig, "emb-negsample")


# ------------------------------------------------------------- tokenizer 流水线

def tokenizer_ids():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    sent = "Embedding vectors are surprisingly useful"
    ids = enc.encode(sent)
    pieces = [enc.decode([t]) for t in ids]
    print("tokenizer sentence:", repr(sent))
    for p_, i in zip(pieces, ids):
        print(f"token {i:>6}  {p_!r}")
    return sent, ids, pieces


def plot_tokenizer():
    """FIGURE：文本 → tokenizer → token ID → 查 embedding 表 → 向量。"""
    sent, ids, pieces = tokenizer_ids()
    n = len(ids)

    fig, ax = new_diagram((11.6, 3.6), (0, 200), (-3, 33))
    patches, inner = [], []

    def bx_(center_, text, **kw):
        p, t = box(ax, center_, text, **kw)
        patches.append(p)
        inner.append(t)
        return p, t

    bx_((16, 15), f"文本\n{sent[:22]}\n{sent[22:]}", width=30, height=9.5, fontsize=8.5)
    bx_((48, 15), "tokenizer\nBPE", width=16, height=9, fontsize=10,
        fc="#eef3fa")
    # token ID 序列：每个 token 一小格，宽度按 token 文本长度取
    gap = 3.0
    widths = [max(len(p_.strip()) * 1.55 + 3, 7) for p_ in pieces]
    x = 62
    cells = []
    for (p_, tid), w_ in zip(zip(pieces, ids), widths):
        bx_((x + w_ / 2, 15), f"{p_.strip()}\n{tid}", width=w_, height=9,
            fontsize=7.5, fc="#ffffff", ec=BLUE)
        x += w_ + gap
        cells.append(x)
    bx_((x + 12, 15), "embedding 表 E\nV × d 查行", width=22, height=9,
        fontsize=8.5, fc="#eef3fa")
    arrow(ax, (30.5, 15), (39, 15))
    arrow(ax, (57, 15), (61, 15))
    arrow(ax, (cells[-1], 15), (x + 0.5, 15))
    ax.text(50, 27, "文本 → token ID → 查表 → 每行一个 d 维向量",
            ha="center", va="center", fontsize=9.5, color="#444444")

    check_layout(fig, ax, patches, inner)
    save(fig, "emb-tokenizer-pipeline")


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
    plot_negsample()
    plot_tokenizer()


if __name__ == "__main__":
    main()
