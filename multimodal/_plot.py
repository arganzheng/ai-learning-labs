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

plt.rcParams.update({
    "font.family": ["Heiti SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "path",          # 文字落成轮廓，读者端渲染 = check_layout 量到的布局；none 会依赖读者本地字体，与断言脱节
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 一组柔和、打印也分得开的颜色
C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd"}

# 桌面端正文列里图按 ~741 px 渲染；路径化文字没有 hinting，小于这个字号发虚
COLUMN_PX = 741
MIN_TEXT_PX = 13        # 普通文字渲染后最小像素
MIN_MATH_PX = 14        # 含 $...$ 的 mathtext 最小像素


def _final_width_px(fig):
    """savefig(bbox_inches='tight') 之后的实际宽度，换算成 96 dpi 像素。"""
    renderer = fig.canvas.get_renderer()
    return fig.get_tightbbox(renderer).width * 96


def _check_text_floor(fig, name):
    if os.environ.get("PLOT_TEXT_FLOOR") == "off":
        return
    scale = min(1.0, COLUMN_PX / _final_width_px(fig))
    bad = []
    texts = list(fig.texts)
    if fig._suptitle is not None:
        texts.append(fig._suptitle)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:
            texts += ax.get_xticklabels() + ax.get_yticklabels()
    for t in texts:
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        px = t.get_fontsize() * 96 / 72 * scale
        floor = MIN_MATH_PX if "$" in s else MIN_TEXT_PX
        if px < floor - 0.05:
            bad.append(f"{s!r}: {px:.1f}px < {floor}px")
    if os.environ.get("PLOT_TEXT_FLOOR") == "scan":
        log = os.environ.get("PLOT_SCAN_LOG", "/tmp/textfloor-scan.jsonl")
        import json
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "scale": round(scale, 3), "bad": bad},
                               ensure_ascii=False) + "\n")
        return
    if bad:
        raise AssertionError(
            f"{name}: {len(bad)} 处文字渲染后低于最小字号（列宽 {COLUMN_PX}px 下按 {scale:.2f} 缩放）\n"
            + "\n".join(bad))


def _dump_texts(fig, name):
    """PLOT_TEXTS=<dir> 时把本图全部文字（排序）写成 JSON，供改字号前后对比。"""
    d = os.environ.get("PLOT_TEXTS")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    texts = sorted({t.get_text() for t in fig.findobj(matplotlib.text.Text)
                    if t.get_text().strip() and t.get_visible()})
    with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(texts, f, ensure_ascii=False, indent=1, sort_keys=True)


def save(fig, name, dpi=150):
    """散点 / 箭头 / 大量点的折线在 SVG 里按 dpi 栅格化（嵌成 PNG），文字与坐标轴仍是矢量——几千个点的图也能保持在 100 KB 以内。"""
    os.makedirs(OUT, exist_ok=True)
    for ax in fig.axes:
        for coll in ax.collections:                                       # scatter / quiver
            coll.set_rasterized(True)
        for line in ax.lines:
            if len(line.get_xdata()) > 500:
                line.set_rasterized(True)
    _dump_texts(fig, name)
    _check_text_floor(fig, name)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None}, dpi=dpi)
    if os.environ.get("PLOT_PNG"):                                   # 作者自查用：PLOT_PNG=/tmp/plots 另存一份 png
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
