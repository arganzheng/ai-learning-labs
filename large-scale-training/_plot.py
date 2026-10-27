"""画图的公共设置：中文字体、博客正文列宽（750 px）对应的图宽、SVG 输出到 out/。

    from _plot import plt, save
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ...
    save(fig, "01-overfit-curve")      # -> out/01-overfit-curve.svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],  # Heiti SC 默认落在 Light，笔画太细；Hiragino Sans GB W3 是 Regular
    "axes.unicode_minus": False,
    "svg.fonttype": "path",          # 文字落成轮廓，读者端渲染 = check_layout 量到的布局；none 会依赖读者本地字体，与断言脱节
    "font.size": 11,                # 正文列宽 750 px 下约等于 11 pt 正文
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "text.color": "#222222",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 一组柔和、打印也分得开的颜色
C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd"}


BOX_TEXT_SIZE = 10
ARROW_LABEL_SIZE = BOX_TEXT_SIZE - 1   # 箭头/分组标注说明文字，比方框文字小一号

_TRACKED = []          # (Text, FancyBboxPatch|None, row|None)
_PATCHES = []          # box() 画出的全部补丁，供「标注不压盒」检查


def text(ax, x, y, s, in_box=None, row=None, **kw):
    """ax.text 的登记版：in_box=patch 时断言文字四边留 ≥6px；row=key 的文本断言同一基线。"""
    t = ax.text(x, y, s, **kw)
    _TRACKED.append((t, in_box, row))
    return t


def box(ax, x, y, w, h, s=None, fc="white", ec="#444444", fs=10, lw=0.9, r=0.012, **kw):
    """圆角盒；给 s 时把文字画在正中并登记为盒内文本。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, **kw)
    ax.add_patch(p)
    _PATCHES.append(p)
    if s:
        text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center", fontsize=fs)
    return p


def collide(fig, margin=6.0):
    """布局自检：fig 内登记的文本两两不得重叠（>1px 相交即报）；
    登记了 in_box 的文本必须在盒内且四边间距 ≥ margin（figure dpi 像素）。有违规直接抛错。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    items = [(t, p, row) for t, p, row in _TRACKED if t.get_figure() is fig and t.get_text().strip()]
    # 坐标轴上的文字（标题、轴标签、刻度标签）也纳入重叠检查
    for ax in fig.axes:
        axis_texts = [ax.title, ax.xaxis.label, ax.yaxis.label,
                      *ax.get_xticklabels(), *ax.get_yticklabels()]
        if ax.get_legend() is not None:
            axis_texts.extend(ax.get_legend().get_texts())
        for t in axis_texts:
            if t.get_text().strip():
                items.append((t, None, None))
    ext = {t: t.get_window_extent(renderer=ren) for t, _, _ in items}
    errs = []
    for t, p, _ in items:
        e = ext[t]
        if p is not None:
            b = p.get_window_extent(renderer=ren)
            for side, d in (("左", e.x0 - b.x0), ("下", e.y0 - b.y0),
                            ("右", b.x1 - e.x1), ("上", b.y1 - e.y1)):
                if d < margin:
                    errs.append(f"盒内边距 {d:.0f}px<{margin:.0f}px（{side}）：{t.get_text()!r}")
    # 文本两两不重叠
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = ext[items[i][0]], ext[items[j][0]]
            iw = min(a.x1, b.x1) - max(a.x0, b.x0)
            ih = min(a.y1, b.y1) - max(a.y0, b.y0)
            if iw > 1 and ih > 1:
                errs.append(f"文本重叠：{items[i][0].get_text()!r} × {items[j][0].get_text()!r}")
    # 非盒内文本（箭头标注等）不得压到任何盒子上
    for p in _PATCHES:
        if p.get_figure() is not fig:
            continue
        b = p.get_window_extent(renderer=ren)
        for t, own, _ in items:
            if own is not None:
                continue
            e = ext[t]
            iw = min(e.x1, b.x1) - max(e.x0, b.x0)
            ih = min(e.y1, b.y1) - max(e.y0, b.y0)
            if iw > 1 and ih > 1:
                errs.append(f"标注压盒：{t.get_text()!r}")
    # 同一 row 组的标签必须在同一基线上
    groups = {}
    for t, _, row in items:
        if row is not None:
            groups.setdefault(row, []).append(t)
    for row, ts in groups.items():
        y0 = ext[ts[0]].y0
        for t in ts[1:]:
            if abs(ext[t].y0 - y0) > 1.5:
                errs.append(f"同排标签基线不齐（row={row}）：{t.get_text()!r}")
    if errs:
        raise AssertionError(f"{len(errs)} 处文字碰撞：\n" + "\n".join(errs))


def save(fig, name, dpi=150):
    """散点 / 箭头 / 大量点的折线在 SVG 里按 dpi 栅格化（嵌成 PNG），文字与坐标轴仍是矢量——几千个点的图也能保持在 100 KB 以内。"""
    os.makedirs(OUT, exist_ok=True)
    collide(fig)
    for ax in fig.axes:
        for coll in ax.collections:                                       # scatter / quiver
            coll.set_rasterized(True)
        for line in ax.lines:
            if len(line.get_xdata()) > 500:
                line.set_rasterized(True)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None}, dpi=dpi)
    if os.environ.get("PLOT_PNG"):                                   # 作者自查用：PLOT_PNG=/tmp/plots 另存一份 png
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
