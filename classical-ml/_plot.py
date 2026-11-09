"""画图的公共设置：中文字体、博客正文列宽（750 px）对应的图宽、SVG 输出到 out/。

    from _plot import plt, save
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ...
    save(fig, "01-overfit-curve")      # -> out/01-overfit-curve.svg
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

plt.rcParams.update({
    "font.family": ["Heiti SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "path",          # 文字与 mathtext 都落成轮廓，读者端不依赖字体
    "mathtext.fontset": "cm",        # $...$ 用 Computer Modern，与正文 KaTeX 一致
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 一组柔和、打印也分得开的颜色
C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd"}


def check_texts(fig):
    """图内文字自检：mathtext 里不允许 e± 科学计数法；每个非 ASCII 字符必须
    落在实际解析出的字体 cmap 里（逐字回退会让同一行字重不一）。"""
    fig.canvas.draw()
    errs = []
    texts = []
    for ax in fig.axes:
        texts += list(ax.texts)
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            texts += [t for t in ax.get_legend().get_texts() if t.get_visible()]
        texts += [*ax.get_xticklabels(), *ax.get_yticklabels()]
    cmaps = {}
    for t in texts:
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        for m in re.finditer(r"\$([^$]*)\$", s):
            if re.search(r"\d\s*e[+-]\d", m.group(1)):
                errs.append(f"mathtext 内科学计数法（会排成减法）：{s!r}，改用 \\times10^{{}}")
        fp = t.get_fontproperties()
        try:
            fpath = _fm.findfont(fp, fallback_to_default=False)
        except Exception:
            fpath = _fm.findfont(fp)
        if fpath not in cmaps:
            cmaps[fpath] = FT2Font(fpath).get_charmap()
        plain = re.sub(r"\$[^$]*\$", "", s)
        missing = [c for c in plain if ord(c) > 127 and ord(c) not in cmaps[fpath]]
        if missing:
            errs.append(f"字形缺失（{os.path.basename(fpath)}）：{s!r} 缺 {''.join(sorted(set(missing)))!r}")
    if errs:
        raise AssertionError("\n".join(errs))


def collide(fig):
    """布局自检：可见文本两两不得重叠、不得压到数据折线、不得出画布边。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        texts += [t for t in ax.texts if t.get_text().strip() and t.get_visible()]
        axis_texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            axis_texts += [t for t in ax.get_legend().get_texts() if t.get_visible()]
        texts += [t for t in axis_texts if t.get_text().strip() and t.get_visible()]
    ext = {t: t.get_window_extent(renderer=ren) for t in texts}
    errs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = ext[texts[i]], ext[texts[j]]
            iw = min(a.x1, b.x1) - max(a.x0, b.x0)
            ih = min(a.y1, b.y1) - max(a.y0, b.y0)
            if iw > 1 and ih > 1:
                errs.append(f"文本重叠：{texts[i].get_text()!r} × {texts[j].get_text()!r}")
    def _seg_rect(a, b, r):
        x0, y0, x1, y1 = r
        dx, dy = b[0] - a[0], b[1] - a[1]
        lo, hi = 0.0, 1.0
        for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                     (-dy, a[1] - y0), (dy, y1 - a[1])):
            if p == 0:
                if q <= 0:
                    return False
            else:
                rr = q / p
                lo, hi = (max(lo, rr) if p < 0 else lo), (min(hi, rr) if p > 0 else hi)
                if lo >= hi:
                    return False
        return True
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_linestyle() in (":", "--", "-.", "None", "none"):
                continue
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) < 2:
                continue
            pts = ax.transData.transform(list(zip(xd, yd)))
            ab = ax.get_window_extent(renderer=ren)
            for k in range(len(pts) - 1):
                if (max(pts[k][0], pts[k + 1][0]) < ab.x0 or min(pts[k][0], pts[k + 1][0]) > ab.x1
                        or max(pts[k][1], pts[k + 1][1]) < ab.y0 or min(pts[k][1], pts[k + 1][1]) > ab.y1):
                    continue
                a, b = tuple(pts[k]), tuple(pts[k + 1])
                for t in texts:
                    e = ext[t]
                    if _seg_rect(a, b, (e.x0 + 1, e.y0 + 1, e.x1 - 1, e.y1 - 1)):
                        errs.append(f"文字压折线：{t.get_text()!r}")
    fb = fig.bbox
    for t in texts:
        e = ext[t]
        if e.x0 < 4 or e.y0 < 4 or e.x1 > fb.x1 - 4 or e.y1 > fb.y1 - 4:
            errs.append(f"文字出画布边：{t.get_text()!r}")
    if errs:
        raise AssertionError(f"{len(errs)} 处文字碰撞：\n" + "\n".join(sorted(set(errs))))


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


def save(fig, name):
    check_texts(fig)
    collide(fig)
    _dump_texts(fig, name)
    _check_text_floor(fig, name)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(l.rstrip() for l in lines) + "\n")
    if os.environ.get("PLOT_PNG"):                                   # 作者自查用：PLOT_PNG=/tmp/plots 另存一份 png
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
