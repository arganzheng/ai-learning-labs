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

# ---- Noto Sans SC + SVG 内嵌字体子集（共享实现见 ../figtext.py）----
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figtext as _ft  # noqa: E402  导入即注册 Noto + svg.fonttype=none

register_noto = _ft.register_noto
embed_fonts = _ft.embed_fonts
check_plain_text = _ft.check_plain_text
check_math_needed = _ft.check_math_needed
check_math_line_spacing = _ft.check_math_line_spacing
check_text_in_rect = _ft.check_text_in_rect

plt.rcParams.update({
    "font.family": ["Noto Sans SC"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",          # 真文字 + 内嵌字体子集（embed_fonts），<img> 下也有 hinting
    "mathtext.fontset": "cm",        # $...$ 用 Computer Modern，与正文 KaTeX 一致
    "font.size": 10.5,
    "axes.titlesize": 11,
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


def _unicode_minus_ticks(fig):
    # matplotlib 这版的 ScalarFormatter 仍输出 ASCII '-' 负号；给线性轴换成输出 '−' 的包装
    import matplotlib.ticker as _tk

    class _UF(_tk.ScalarFormatter):
        def format_ticks(self, values):
            return [s.replace("-", "−") for s in super().format_ticks(values)]

        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace("-", "−")

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for attr in ("major", "minor"):
                fmt = getattr(axis, f"get_{attr}_formatter")()
                if type(fmt) is _tk.ScalarFormatter:
                    new = _UF()
                    new.create_dummy_axis()
                    getattr(axis, f"set_{attr}_formatter")(new)


def save(fig, name):
    _unicode_minus_ticks(fig)
    # 这批脚本的老字号（8/8.5）低于 13px 下限：统一抬到 9.4pt 起，再让检查复核
    for t in fig.findobj(matplotlib.text.Text):
        if "$" in t.get_text():
            if t.get_fontsize() < 10.5: t.set_fontsize(10.5)
        elif t.get_fontsize() < 10:
            t.set_fontsize(10)
    check_texts(fig)
    collide(fig)
    _dump_texts(fig, name)
    _check_text_floor(fig, name)
    check_math_needed(fig)
    check_plain_text(fig)
    check_math_line_spacing(fig)
    check_text_in_rect(fig)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(l.rstrip() for l in lines) + "\n")
    embed_fonts(path)
    if os.environ.get("PLOT_PNG"):                                   # 作者自查用：PLOT_PNG=/tmp/plots 另存一份 png
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
