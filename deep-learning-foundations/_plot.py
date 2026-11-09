"""画图的公共设置（与 classical-ml/_plot.py 相同）：中文字体、博客列宽、SVG 输出到 out/。"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": ["Noto Sans SC Medium", "Noto Sans SC"],
    "axes.unicode_minus": True,
    "svg.fonttype": "none",         # 真文字 + 内嵌字体子集（embed_fonts），<img> 下也有 hinting
    "mathtext.fontset": "cm",       # $...$ 用 Computer Modern，与正文 KaTeX 一致
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
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
    for t in texts:
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        px = t.get_fontsize() * 96 / 72 * scale
        floor = MIN_MATH_PX if "$" in s else MIN_TEXT_PX
        if px < floor - 0.05:
            bad.append(f"{s!r}: {px:.1f}px < {floor}px")
    if os.environ.get("PLOT_TEXT_FLOOR") == "scan":
        import json
        log = os.environ.get("PLOT_SCAN_LOG", "/tmp/textfloor-scan.jsonl")
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


# ---- Noto Sans SC + SVG 内嵌字体子集 ----
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts")
_MPL_TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def register_noto():
    """注册 Noto Sans SC 并把排版字体切过去；必须在任何 Text 创建前调用。"""
    from matplotlib import font_manager
    for f in ("NotoSansSC-Medium.otf", "NotoSansSC-Regular.otf"):
        font_manager.fontManager.addfont(os.path.join(FONTS_DIR, f))
    plt.rcParams["font.family"] = ["Noto Sans SC Medium", "Noto Sans SC"]
    plt.rcParams["svg.fonttype"] = "none"


# matplotlib svg.fonttype='none' 实际写出的 family 名 → (字体文件, @font-face weight)
_FONT_FILES = {
    "Noto Sans SC Medium": [(os.path.join(FONTS_DIR, "NotoSansSC-Medium.otf"), "normal")],
    "Noto Sans SC": [(os.path.join(FONTS_DIR, "NotoSansSC-Regular.otf"), "normal"),
                     (os.path.join(FONTS_DIR, "NotoSansSC-Medium.otf"), "500")],
}
for _n in ("cmmi10", "cmr10", "cmsy10", "cmex10", "cmss10", "cmti10", "cmtt10", "cmb10"):
    _FONT_FILES[_n] = [(os.path.join(_MPL_TTF, _n + ".ttf"), "normal")]
_STIX_FILES = {
    "STIXGeneral": "STIXGeneral.ttf", "STIXNonUnicode": "STIXNonUni.ttf",
    "STIXSizeOneSym": "STIXSizOneSymReg.ttf", "STIXSizeTwoSym": "STIXSizTwoSymReg.ttf",
    "STIXSizeThreeSym": "STIXSizThreeSymReg.ttf",
    "STIXSizeFourSym": "STIXSizFourSymReg.ttf",
    "STIXSizeFiveSym": "STIXSizFiveSymReg.ttf",
    "STIXGeneral Bol": "STIXGeneralBol.ttf",
    "STIXGeneral Italic": "STIXGeneralItalic.ttf",
    "STIXNonUnicode Bol": "STIXNonUniBol.ttf",
    "STIXNonUnicode Italic": "STIXNonUniIta.ttf",
    "STIXSizeOneSym Bol": "STIXSizOneSymBol.ttf",
    "STIXSizeTwoSym Bol": "STIXSizTwoSymBol.ttf",
    "STIXSizeThreeSym Bol": "STIXSizThreeSymBol.ttf",
    "STIXSizeFourSym Bol": "STIXSizFourSymBol.ttf",
}
_STIX_BOLD = {
    "STIXGeneral": "STIXGeneralBol.ttf", "STIXNonUnicode": "STIXNonUniBol.ttf",
    "STIXSizeOneSym": "STIXSizOneSymBol.ttf", "STIXSizeTwoSym": "STIXSizTwoSymBol.ttf",
    "STIXSizeThreeSym": "STIXSizThreeSymBol.ttf",
    "STIXSizeFourSym": "STIXSizFourSymBol.ttf",
}
for _n, _f in _STIX_FILES.items():
    _FONT_FILES[_n] = [(os.path.join(_MPL_TTF, _f), "normal")]
    if _n in _STIX_BOLD:
        _FONT_FILES[_n].append((os.path.join(_MPL_TTF, _STIX_BOLD[_n]), "700"))
        _FONT_FILES[_n].append((os.path.join(_MPL_TTF, _STIX_BOLD[_n]), "bold"))
# PUA 字形可能只在 italic/bold 文件里——给常规 family 也挂上候选
_FONT_FILES["STIXNonUnicode"].append((os.path.join(_MPL_TTF, "STIXNonUniIta.ttf"), "normal"))
_FONT_FILES["STIXNonUnicode"].append((os.path.join(_MPL_TTF, "STIXNonUniBolIta.ttf"), "normal"))
_FONT_FILES["STIXGeneral"].append((os.path.join(_MPL_TTF, "STIXGeneralItalic.ttf"), "normal"))
_FONT_FILES["STIXGeneral"].append((os.path.join(_MPL_TTF, "STIXGeneralBolIta.ttf"), "normal"))


def _style_family(style):
    m = re.search(r"font-family:\s*'([^']+)'", style or "")
    return m.group(1) if m else None


def embed_fonts(path):
    """把 svg.fonttype='none' 产出的 SVG 里用到的字形按 family 子集成 WOFF2 内嵌进去。"""
    import base64
    import io
    import xml.etree.ElementTree as ET
    from fontTools import subset
    from fontTools.ttLib import TTFont

    src = open(path, encoding="utf-8").read()
    before = len(src.encode("utf-8"))
    tree = ET.fromstring(src)
    # (family, weight) -> chars
    used = {}
    for el in tree.iter():
        if not el.tag.rpartition("}")[2] in ("text", "tspan"):
            continue
        fam = _style_family(el.get("style", ""))
        content = el.text or ""
        if not fam or not content.strip():
            continue
        w = re.search(r"font-weight:\s*(\w+)", el.get("style", ""))
        weight = w.group(1) if w else "normal"
        used.setdefault((fam, weight), set()).update(content)
    if not used:
        return
    # family+weight -> 要嵌入的字体文件
    rules = []          # (family, weight, file)
    needed = {}         # (file, family, weight) -> chars
    for (fam, weight), chars in sorted(used.items()):
        entries = _FONT_FILES.get(fam)
        if entries is None:
            raise AssertionError(f"未知 font-family：{fam!r}")
        cands = [f for f, w in entries if w == weight] or [f for f, _ in entries]
        cands += [f for f, _ in entries if f not in cands]
        f = None
        for cand in cands:
            cmap = TTFont(cand).getBestCmap()
            if all(ord(c) in cmap for c in chars):
                f = cand
                break
        if f is None:
            cmap = TTFont(cands[0]).getBestCmap()
            miss = sorted(c for c in chars if ord(c) not in cmap)
            raise AssertionError(
                f"{fam!r} 的候选字体都缺字形 {miss!r}；改用 Unicode 替代或 $...$")
        needed.setdefault((f, fam, weight), set()).update(chars)
    faces = []
    for (f, fam, weight), chars in sorted(needed.items()):
        font = TTFont(f)
        cmap = font.getBestCmap()
        missing = {c for c in chars if ord(c) not in cmap}
        if missing:
            raise AssertionError(
                f"{os.path.basename(f)} 缺字形：{sorted(missing)!r}（family {fam!r}）")
        opts = subset.Options(flavor="woff2")
        ss = subset.Subsetter(opts)
        ss.populate(text="".join(sorted(chars)))
        try:
            ss.subset(font)
        except Exception:
            opts.flavor = "woff"
            font = TTFont(f)
            ss = subset.Subsetter(opts)
            ss.populate(text="".join(sorted(chars)))
            ss.subset(font)
        buf = io.BytesIO()
        font.save(buf)
        mime = "woff2" if opts.flavor == "woff2" else "woff"
        faces.append(
            f"@font-face{{font-family:'{fam}';font-weight:{weight};"
            f"src:url(data:font/{mime};base64,{base64.b64encode(buf.getvalue()).decode()})}}"
        )
    style = ("<defs><style type=\"text/css\"><![CDATA["
             + "".join(faces) + "]]></style></defs>")
    m = re.search(r"<svg[^>]*>", src)
    src = src[:m.end()] + "\n" + style + src[m.end():]
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(src)
    after = len(src.encode("utf-8"))
    print(f"  内嵌字体 {os.path.basename(path)}: {before} -> {after} 字节"
          f"（{len(faces)} 个 @font-face）")


def check_text_in_rect(fig):
    """文字中心落在某个普通矩形内时不得横向溢出该矩形（窄条段标签）。"""
    if os.environ.get("PLOT_COLLIDE") == "off":
        return
    import matplotlib.patches as mp
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    errs = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            e = t.get_window_extent(renderer=ren)
            cx, cy = (e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2
            for p in ax.patches:
                if not isinstance(p, mp.Rectangle) or isinstance(p, mp.FancyBboxPatch) \
                        or p is ax.patch:
                    continue
                try:
                    b = p.get_window_extent(renderer=ren)
                except Exception:
                    continue
                if b.width <= 0 or b.height <= 0:
                    continue
                if b.x0 < cx < b.x1 and b.y0 < cy < b.y1 and \
                        (e.x0 < b.x0 - 1 or e.x1 > b.x1 + 1):
                    errs.append(f"文字溢出所在矩形（{b.x0:.0f}–{b.x1:.0f}px）：{t.get_text()!r}")
                    break
    if errs:
        raise AssertionError(f"{len(errs)} 处文字溢出所在矩形：\n" + "\n".join(errs))


# $...$ 只允许放真正需要排版的（命令/上下标/希腊字母）；纯文字直接写 Unicode
_PLAIN_CMDS = re.compile(
    r"\\\\times|\\\\to\b|\\\\rightarrow|\\\\cdot|\\\\mathrm\{[^{}]*\}|"
    r"\\times|\\to\b|\\rightarrow|\\leftarrow|\\leq|\\geq|\\neq|\\approx|\\sim|"
    r"\\pm|\\mp|\\cdot|\\,|\\!|\\;|\\ |\\mathrm\{[^{}]*\}")


def check_math_needed(fig):
    """剥掉可平写的命令后还剩 \\、_、^ 才算需要 mathtext；否则用 Unicode 写。"""
    if os.environ.get("PLOT_MATH_CHECK") == "off":
        return
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        for m in re.finditer(r"\$([^$]*)\$", s):
            if not re.search(r"[_^\\]", _PLAIN_CMDS.sub("", m.group(1))):
                bad.append(s)
    if bad:
        raise AssertionError(
            f"{len(bad)} 处无需排版的 $...$（应写成普通文字/Unicode）：\n"
            + "\n".join(repr(b) for b in sorted(set(bad))))


def check_plain_text(fig):
    """$...$ 之外的正文必须是干净的 Unicode 文字：
    不允许出现反斜杠（\\times、\\n 漏排）、_ / ^（d_g 漏排）、
    以及当作负号/减号的 ASCII -（-1、(−1) 应写 −）。
    单词里的连字符（next-token）与代码标识符不受影响。"""
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        plain = re.sub(r"\$[^$]*\$", "", s)
        if "\\" in plain:
            bad.append(("反斜杠", s))
        for m_ in re.finditer(r"\w*[_^]\w*", plain):
            # next_functions 这类代码标识符（两侧都 ≥2 字符）豁免
            if not re.fullmatch(r"\w{2,}[_^]\w{2,}", m_.group(0)):
                bad.append(("_ 或 ^", s))
                break
        if re.search(r"(?<![A-Za-z0-9])-\d", plain):
            bad.append(("ASCII 负号/减号", s))
    if bad:
        raise AssertionError(
            f"{len(bad)} 处正文含非法字符（\\\\ / _ / ^ / ASCII 负号）：\n"
            + "\n".join(f"{k}：{s!r}" for k, s in bad))


def check_math_line_spacing(fig):
    """单行以上的 Text 里只要有任何一行含 $...$ 数学，
    linespacing 必须 ≥1.4——否则会叠到上一行。"""
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not t.get_visible() or "\n" not in s:
            continue
        ls = t.get_linespacing()
        try:
            ls = float(ls)
        except (TypeError, ValueError):
            ls = 1.2
        if "$" in s and ls < 1.4:
            bad.append(s)
    if bad:
        raise AssertionError(
            f"{len(bad)} 处含数学的多行文字 linespacing<1.4"
            "（应拆成两个 Text 或加大行距）：\n"
            + "\n".join(repr(b) for b in sorted(set(bad))))


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd", "teal": "#2a9d8f"}


def save(fig, name):
    # mathtext 里不允许 e± 科学计数法（会被排成减法）——写 \times10^{}
    for ax in fig.axes:
        for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            s = t.get_text()
            for m in re.finditer(r"\$([^$]*)\$", s):
                if re.search(r"\d\s*e[+-]\d", m.group(1)):
                    raise AssertionError(
                        f"mathtext 内科学计数法（会排成减法）：{s!r}，改用 \\times10^{{}}")
    _dump_texts(fig, name)
    check_math_needed(fig)
    check_plain_text(fig)
    check_math_line_spacing(fig)
    check_text_in_rect(fig)
    _check_text_floor(fig, name)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(l.rstrip() for l in lines) + "\n")
    embed_fonts(path)
    if os.environ.get("PLOT_PNG"):
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path


def enable_embed(mod):
    """包一层 mod.save（如 lst_plot.save）：存盘后跑文字溢出/数学检查并内嵌字体。"""
    orig = mod.save

    def wrapped(fig, name, *a, **k):
        check_math_needed(fig)
        check_plain_text(fig)
        check_math_line_spacing(fig)
        check_text_in_rect(fig)
        p = orig(fig, name, *a, **k)
        embed_fonts(p)
        return p
    mod.save = wrapped
    return wrapped


register_noto()   # 导入即生效：保证此后创建的 Text 用 Noto 度量
