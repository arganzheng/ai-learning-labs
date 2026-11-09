"""画图的公共设置：中文字体、博客正文列宽（750 px）对应的图宽、SVG 输出到 out/。

    from _plot import plt, save
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ...
    save(fig, "01-overfit-curve")      # -> out/01-overfit-curve.svg
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figtext as _ft  # noqa: E402  导入即注册 Noto

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib import patheffects as _pe  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams.update({
    "font.family": ["Noto Sans SC"],
    "axes.unicode_minus": True,
    "svg.fonttype": "none",         # 真文字 + 内嵌字体子集（embed_fonts），<img> 下也有 hinting
    "mathtext.fontset": "cm",        # $...$ 用 Computer Modern 排，与正文 KaTeX 风格一致
    "font.size": 11,                # 正文列宽 750 px 下约等于 11 pt 正文
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
})

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
        # ax.text artists render even on axis("off") axes (diagram figures rely
        # on it); tick labels do not, so only include them when axison
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:
            texts += ax.get_xticklabels() + ax.get_yticklabels()
    min_text = min_math = None
    for t in texts:
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        px = t.get_fontsize() * 96 / 72 * scale
        is_math = "$" in s
        if is_math:
            min_math = px if min_math is None else min(min_math, px)
        else:
            min_text = px if min_text is None else min(min_text, px)
        floor = MIN_MATH_PX if is_math else MIN_TEXT_PX
        if px < floor - 0.05:
            bad.append(f"{s!r}: {px:.1f}px < {floor}px")
    if os.environ.get("PLOT_TEXT_FLOOR") == "scan":
        log = os.environ.get("PLOT_SCAN_LOG", "/tmp/textfloor-scan.jsonl")
        import json
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"name": name, "scale": round(scale, 3),
                                "w_in": round(fig.get_figwidth(), 2),
                                "min_text": round(min_text, 1) if min_text else None,
                                "min_math": round(min_math, 1) if min_math else None,
                                "bad": bad},
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


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 一组柔和、打印也分得开的颜色
C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd"}


BOX_TEXT_SIZE = 11
BOX_TITLE_SIZE = BOX_TEXT_SIZE + 1   # 盒内主标题，比正文大一号
ARROW_LABEL_SIZE = BOX_TEXT_SIZE - 1   # 箭头/分组标注说明文字，比方框文字小一号

_TRACKED = []          # (Text, FancyBboxPatch|None, row|None)
_PATCHES = []          # box() 画出的全部补丁，供「标注不压盒」检查


def text(ax, x, y, s, in_box=None, row=None, **kw):
    """ax.text 的登记版：in_box=patch 时断言文字四边留 ≥6px；row=key 的文本断言同一基线。"""
    t = ax.text(x, y, s, **kw)
    _TRACKED.append((t, in_box, row))
    return t


def box_title(ax, x, y, s, in_box, color="#111111", **kw):
    """盒内主标题：BOX_TITLE_SIZE + Medium 字重（标题类文字用 Medium，正文 Regular）。"""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontname", "Noto Sans SC Medium")
    t = ax.text(x, y, s, fontsize=BOX_TITLE_SIZE, color=color, **kw)
    _TRACKED.append((t, in_box, None))
    return t


def box(ax, x, y, w, h, s=None, fc="white", ec="#444444", fs=BOX_TEXT_SIZE, lw=0.9, r=0.012, linespacing=None, **kw):
    """圆角盒；给 s 时把文字画在正中并登记为盒内文本。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, **kw)
    ax.add_patch(p)
    _PATCHES.append(p)
    if s:
        tkw = {"linespacing": linespacing} if linespacing else {}
        t = text(ax, x + w / 2, y + h / 2, s, in_box=p, ha="center", va="center", fontsize=fs, **tkw)
        t._centered_in_box = True   # 只有盒自己的居中标签才做墨迹重中
    return p


def collide(fig, margin=6.0):
    """布局自检：fig 内登记的文本两两不得重叠（>1px 相交即报）；
    登记了 in_box 的文本必须在盒内且四边间距 ≥ margin（figure dpi 像素）。有违规直接抛错。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    items = [(t, p, row) for t, p, row in _TRACKED if t.get_figure() is fig and t.get_text().strip()]
    # 坐标轴上的文字（标题、轴标签、刻度标签）也纳入重叠检查
    for ax in fig.axes:
        axis_texts = [ax.title]
        if ax.axison:                    # axis("off") 的隐藏刻度文字不参与检查
            axis_texts += [ax.xaxis.label, ax.yaxis.label,
                           *ax.get_xticklabels(), *ax.get_yticklabels()]
            if ax.get_legend() is not None:
                axis_texts.extend(ax.get_legend().get_texts())
        for t in axis_texts:
            if t.get_text().strip() and t.get_visible():
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
    # 非盒内文本（箭头标注等）不得压到任何盒子上。
    # _PATCHES 之外的裸 add_patch（图例色块等）同样检查；axvspan/barh 产生的矩形
    # 也算，但跳过撑满全轴的 band（text 与其相交不算压盒）。
    seen = {id(p) for p in _PATCHES if p.get_figure() is fig}
    all_patches = [p for p in _PATCHES if p.get_figure() is fig]
    for ax in fig.axes:
        for p in ax.patches:
            # 只补矩形（图例色块、barh、axvspan）——箭头/线段上的标注是常态
            import matplotlib.patches as _mp
            if not isinstance(p, _mp.Rectangle) or p is ax.patch:
                continue
            if id(p) not in seen:
                seen.add(id(p))
                all_patches.append(p)
    for p in all_patches:
        b = p.get_window_extent(renderer=ren)
        for t, own, _ in items:
            if own is not None:
                continue
            e = ext[t]
            iw = min(e.x1, b.x1) - max(e.x0, b.x0)
            ih = min(e.y1, b.y1) - max(e.y0, b.y0)
            if iw > 1 and ih > 1:
                errs.append(f"标注压盒：{t.get_text()!r}"
                            f"（矩形 {b.x0:.0f}–{b.x1:.0f} × {b.y0:.0f}–{b.y1:.0f}px）")
            elif ih > 1 and id(p) in {id(q) for q in _PATCHES}:
                # 同一行里不在盒内的标注（泳道名等）离盒边至少 16px；
                # 只约束 box() 画的盒——barh/axvspan 的数据矩形不套这个规则
                gap = max(b.x0 - e.x1, e.x0 - b.x1)
                if gap < 16:
                    errs.append(f"行内标注距盒 {gap:.0f}px<16px：{t.get_text()!r}")
    # 完全落在轴外的补丁会被静默裁掉（漏画图例色块这类错误），报出来
    for ax in fig.axes:
        ab = ax.get_window_extent(renderer=ren)
        for p in ax.patches:
            pb = p.get_window_extent(renderer=ren)
            if (pb.x1 < ab.x0 or pb.x0 > ab.x1 or
                    pb.y1 < ab.y0 or pb.y0 > ab.y1):
                errs.append("矩形完全在轴外不可见（图例色块被裁掉等）")
    # 文字/图例不得压到数据折线：所有文本 bbox 与 Line2D 的显示坐标线段相交即失败
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
                if p < 0:
                    lo = max(lo, rr)
                else:
                    hi = min(hi, rr)
                if lo >= hi:
                    return False
        return True
    line_texts = [t for t, _, _ in items]
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip() and t.get_visible() and t not in ext:
                line_texts.append(t)
                ext[t] = t.get_window_extent(renderer=ren)
        axbb = ax.get_window_extent(renderer=ren)
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) < 2:
                continue
            pts = ax.transData.transform(list(zip(xd, yd)))
            for k in range(len(pts) - 1):
                a, b = tuple(pts[k]), tuple(pts[k + 1])
                # 轴外的线段渲染时会被裁掉，先做 Liang–Barsky 裁剪再测
                t0, t1 = 0.0, 1.0
                ok = True
                for lo, hi, d0, d1 in ((axbb.x0, axbb.x1, a[0], b[0]),
                                       (axbb.y0, axbb.y1, a[1], b[1])):
                    if abs(d1 - d0) < 1e-9:
                        if not (lo <= d0 <= hi):
                            ok = False
                    else:
                        ta, tb = (lo - d0) / (d1 - d0), (hi - d0) / (d1 - d0)
                        if ta > tb:
                            ta, tb = tb, ta
                        t0, t1 = max(t0, ta), min(t1, tb)
                    if not ok or t0 > t1:
                        ok = False
                        break
                if not ok or t0 > t1:
                    continue
                a, b = ((a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                        (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1))
                for t in line_texts:
                    e = ext[t]
                    if _seg_rect(a, b, (e.x0 + 1, e.y0 + 1, e.x1 - 1, e.y1 - 1)):
                        errs.append(f"文字压折线：{t.get_text()!r}")
                        break
                else:
                    continue
                break
    # 线段/箭头不得穿过或贴着盒内区域：端点搭在盒边可以，盒内穿行不行
    box_patches = [p for p in _PATCHES if p.get_figure() is fig]
    bext = {p: p.get_window_extent(renderer=ren) for p in box_patches}
    is_container = {}
    for p in box_patches:
        b = bext[p]
        is_container[p] = any(
            q is not p and bext[q].x0 >= b.x0 - 1 and bext[q].y0 >= b.y0 - 1
            and bext[q].x1 <= b.x1 + 1 and bext[q].y1 <= b.y1 + 1
            for q in box_patches)

    def _inside(b, x, y, depth=3.0):
        return b.x0 + depth < x < b.x1 - depth and b.y0 + depth < y < b.y1 - depth

    def _seg_in_box(a, b, r):
        """线段中段（两端各留 5%）是否进入矩形内部 ≥3px。"""
        for t in (0.05, 0.18, 0.30, 0.43, 0.55, 0.68, 0.80, 0.95):
            if _inside(r, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])):
                return True
        return False

    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) < 2:
                continue
            pts = ax.transData.transform(list(zip(xd, yd)))
            for k in range(len(pts) - 1):
                a, b = tuple(pts[k]), tuple(pts[k + 1])
                if a == b:
                    continue
                for p in box_patches:
                    if is_container[p]:
                        continue
                    r = bext[p]
                    # 两端都在盒内的（卡片里的小示意线）不算；从外面穿进来的才算
                    if _inside(r, a[0], a[1]) and _inside(r, b[0], b[1]):
                        continue
                    if _seg_in_box(a, b, r):
                        errs.append(f"线段穿盒（{r.x0:.0f},{r.y0:.0f}–"
                                    f"{r.x1:.0f},{r.y1:.0f}）：line@({a[0]:.0f},{a[1]:.0f})")
                        break
        from matplotlib.patches import FancyArrowPatch
        for p in ax.patches:
            if not isinstance(p, FancyArrowPatch) or not p.get_visible():
                continue
            dpts = p.get_path().transformed(ax.transData).vertices
            for v in dpts[1:-1]:
                for bp in box_patches:
                    if is_container[bp]:
                        continue
                    r = bext[bp]
                    if _inside(r, dpts[0][0], dpts[0][1]) and _inside(r, dpts[-1][0], dpts[-1][1]):
                        continue
                    if _inside(r, v[0], v[1]):
                        errs.append(f"箭头穿盒（{r.x0:.0f},{r.y0:.0f}–"
                                    f"{r.x1:.0f},{r.y1:.0f}）")
                        break
    # 图例 swatch 不得与图例文字相交
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        for h in (getattr(leg, "legend_handles", None)
                  or getattr(leg, "legendHandles", []) or []):
            try:
                he = h.get_window_extent(renderer=ren)
            except Exception:
                continue
            for lt in leg.get_texts():
                e = ext.get(lt)
                if e is None:
                    e = lt.get_window_extent(renderer=ren)
                iw = min(he.x1, e.x1) - max(he.x0, e.x0)
                ih = min(he.y1, e.y1) - max(he.y0, e.y0)
                if iw > 1 and ih > 1:
                    errs.append(f"图例手柄压文字：{lt.get_text()!r}")
    # 落在某个矩形里的文字不得横向溢出该矩形（窄条段标签）
    text_items = [t for t, _, _ in items] + [t for ax in fig.axes for t in ax.texts
                                           if t.get_text().strip() and t.get_visible()]
    for t in text_items:
        e = ext.get(t) or t.get_window_extent(renderer=ren)
        cx, cy = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2
        for ax in fig.axes:
            for p in ax.patches:
                if not isinstance(p, Rectangle) or isinstance(p, FancyBboxPatch):
                    continue
                if p is ax.patch:
                    continue
                try:
                    b = p.get_window_extent(renderer=ren)
                except Exception:
                    continue
                if b.width <= 0 or b.height <= 0:
                    continue
                if b.x0 < cx < b.x1 and b.y0 < cy < b.y1:
                    if e.x0 < b.x0 - 1 or e.x1 > b.x1 + 1:
                        errs.append(f"文字溢出所在矩形（{b.x0:.0f}–{b.x1:.0f}px）：{t.get_text()!r}")
                        break
    # 矩形-矩形：不同对象的填充矩形不得相交；一个完整套在另一个里的（内方块、图例 swatch、轴背景）不算
    rects = [p for p in _PATCHES if p.get_figure() is fig]
    for ax in fig.axes:
        for p in ax.patches:
            if isinstance(p, Rectangle) and p not in rects:
                rects.append(p)
    rext = {p: p.get_window_extent(renderer=ren) for p in rects}
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rext[rects[i]], rext[rects[j]]
            iw = min(a.x1, b.x1) - max(a.x0, b.x0)
            ih = min(a.y1, b.y1) - max(a.y0, b.y0)
            if iw <= 1 or ih <= 1:
                continue
            inside = ((a.x0 >= b.x0 - 1 and a.y0 >= b.y0 - 1 and
                       a.x1 <= b.x1 + 1 and a.y1 <= b.y1 + 1) or
                      (b.x0 >= a.x0 - 1 and b.y0 >= a.y0 - 1 and
                       b.x1 <= a.x1 + 1 and b.y1 <= a.y1 + 1))
            if inside:
                continue
            # 背景色带（axvspan/axhspan/ax.patch）铺满某个轴方向，不算重叠
            span = False
            for q, qb in ((rects[i], a), (rects[j], b)):
                if q.axes is not None:
                    aq = q.axes.get_window_extent(renderer=ren)
                    if (qb.y1 - qb.y0 >= 0.85 * (aq.y1 - aq.y0)
                            or qb.x1 - qb.x0 >= 0.85 * (aq.x1 - aq.x0)):
                        span = True
            if span:
                continue
            errs.append(f"矩形重叠：({a.x0:.0f},{a.y0:.0f}–{a.x1:.0f},{a.y1:.0f}) "
                        f"× ({b.x0:.0f},{b.y0:.0f}–{b.x1:.0f},{b.y1:.0f})")
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
    # mathtext 里不允许 e± 科学计数法（会被排成减法）——写 \times10^{}
    for t, _, _ in items:
        s = t.get_text()
        for m in re.finditer(r"\$([^$]*)\$", s):
            if re.search(r"\d\s*e[+-]\d", m.group(1)):
                errs.append(f"mathtext 内科学计数法（会排成减法）：{s!r}，"
                            f"改用 \\times10^{{}}")
    # 每个非 ASCII 字符都必须落在实际解析出的字体 cmap 里——逐字回退会让同一行字重不一
    cmaps = {}
    for t, _, _ in items:
        s = t.get_text()
        fp = t.get_fontproperties()
        try:
            path = _fm.findfont(fp, fallback_to_default=False)
        except Exception:
            path = _fm.findfont(fp)
        if path not in cmaps:
            cmaps[path] = FT2Font(path).get_charmap()
        missing = [c for c in s if ord(c) > 127 and ord(c) not in cmaps[path]]
        if missing:
            errs.append(f"字形缺失（{os.path.basename(path)}）：{s!r} 缺 {''.join(sorted(set(missing)))!r}")
    # 每个登记的盒子（含描边）都必须完整落在画布内，四周 ≥4px；文字同样
    fb = fig.bbox
    for t, _, _ in items:
        e = ext[t]
        if e.x0 < 4 or e.y0 < 4 or e.x1 > fb.x1 - 4 or e.y1 > fb.y1 - 4:
            errs.append(f"文字出画布边（{e.x0:.0f},{e.y0:.0f}–{e.x1:.0f},{e.y1:.0f}，"
                        f"画布 {fb.x1:.0f}×{fb.y1:.0f}）：{t.get_text()!r}")
    for p in _PATCHES:
        if p.get_figure() is not fig:
            continue
        b = p.get_window_extent(renderer=ren).expanded(1 + p.get_linewidth() / 200, 1 + p.get_linewidth() / 200)
        if b.x0 < 4 or b.y0 < 4 or b.x1 > fb.x1 - 4 or b.y1 > fb.y1 - 4:
            errs.append(f"盒子出画布边（{b.x0:.0f},{b.y0:.0f}–{b.x1:.0f},{b.y1:.0f}，画布 {fb.x1:.0f}×{fb.y1:.0f}）")
    if errs:
        raise AssertionError(f"{len(errs)} 处文字碰撞：\n" + "\n".join(errs))


def _center_box_texts(fig, tol=1.5):
    """盒内文字按字形墨迹居中而不是按字体行盒居中：量出渲染 bbox，
    反复微调直到盒内上/下边距差 ≤1.5px（显示像素），做不到就抛错。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    errs = []
    for t, p, _row in _TRACKED:
        if p is None or t.get_figure() is not fig or not t.get_text().strip():
            continue
        if not getattr(t, "_centered_in_box", False):
            continue
        ax = p.axes
        if ax is None:
            continue
        for _ in range(4):
            fig.canvas.draw()
            ren = fig.canvas.get_renderer()
            e = t.get_window_extent(renderer=ren)
            b = p.get_window_extent(renderer=ren)
            d = ((b.y1 - e.y1) - (e.y0 - b.y0)) / 2.0   # >0：文字偏低，上移
            if abs(d) <= 0.75:
                break
            inv = ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform([(e.x0, e.y0),
                                                (e.x0, e.y0 + d)])
            t.set_position((t.get_position()[0] + x1 - x0,
                            t.get_position()[1] + y1 - y0))
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        e = t.get_window_extent(renderer=ren)
        b = p.get_window_extent(renderer=ren)
        diff = abs((b.y1 - e.y1) - (e.y0 - b.y0))
        if diff > tol:
            errs.append(f"盒内上下边距差 {diff:.1f}px>{tol}px：{t.get_text()!r}")
    if errs:
        raise AssertionError(f"{len(errs)} 处盒内文字未垂直居中：\n" + "\n".join(errs))


def save(fig, name, dpi=150):
    """散点 / 箭头 / 大量点的折线在 SVG 里按 dpi 栅格化（嵌成 PNG），文字与坐标轴仍是矢量——几千个点的图也能保持在 100 KB 以内。"""
    os.makedirs(OUT, exist_ok=True)
    _dump_texts(fig, name)
    _center_box_texts(fig)
    _check_text_floor(fig, name)
    if os.environ.get("PLOT_COLLIDE") == "off":                   # 全量扫描用；正常保存一律检查
        try:
            collide(fig)
        except AssertionError as e:
            print(f"  [PLOT_COLLIDE=off] {e}", file=sys.stderr)
    else:
        collide(fig)
    for ax in fig.axes:
        for coll in ax.collections:                                       # scatter / quiver
            coll.set_rasterized(True)
        for line in ax.lines:
            if len(line.get_xdata()) > 500:
                line.set_rasterized(True)
    _ft.checks(fig)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None}, dpi=dpi)
    _ft.strip_ws(path)
    _ft.embed_fonts(path)
    if os.environ.get("PLOT_PNG"):                                   # 作者自查用：PLOT_PNG=/tmp/plots 另存一份 png
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
