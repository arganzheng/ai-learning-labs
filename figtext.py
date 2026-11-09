"""跨系列共享的 SVG 真文字+内嵌字体管线（从 deep-learning-foundations/_plot.py 拆出）。

用法（各系列 _plot.py）：
    import figtext                     # 注册 Noto + fonttype=none（须在创建任何 Text 前 import）
    figtext.checks(fig)                # 保存前检查：math 必要性/正文非法字符/多行行距/溢出矩形
    figtext.strip_ws(path); figtext.embed_fonts(path)
或包装其他模块的 save：
    wrapped = figtext.enable_embed(mod)
"""
import os
import re

import matplotlib
import matplotlib.pyplot as plt

# ---- Noto Sans SC + SVG 内嵌字体子集 ----
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_MPL_TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def register_noto():
    """注册 Noto Sans SC 并把排版字体切过去；必须在任何 Text 创建前调用。"""
    from matplotlib import font_manager
    for f in ("NotoSansSC-Medium.otf", "NotoSansSC-Regular.otf"):
        font_manager.fontManager.addfont(os.path.join(FONTS_DIR, f))
    # 默认正文用 Regular（400）——Medium 只给标题（见 style_titles）；
    # 否则正文整图发粗，旁边 CM mathtext 显得又细又虚。
    plt.rcParams["font.family"] = ["Noto Sans SC"]
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
    if "@font-face" in src:
        return  # 已内嵌过（包装 save 内部也嵌入时避免重复 defs）
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
    # CM / STIX 数学字形比 Noto Regular 细——给它们加同色描边对齐 KaTeX 的字重。
    # stroke-width 以 svg 单位写，换算到 741px 栏宽下的显示像素（~0.3px）。
    vb = re.search(r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+[\d.]+"', m.group(0))
    disp_px = float(os.environ.get("FIGTEXT_MATH_STROKE", "0.3"))
    sw = disp_px * float(vb.group(1)) / 741 if vb else disp_px
    _MATH_FAMILIES = ("cmmi10", "cmr10", "cmsy10", "cmex10", "cmss10", "cmti10",
                      "cmtt10", "cmb10", "STIXGeneral", "STIXNonUnicode",
                      "STIXSize")
    def _stroke(mo):
        st = mo.group(0)
        if not any(f in st for f in _MATH_FAMILIES):
            return st
        fill = re.search(r"fill:\s*([^;\"']+)", st)
        col = fill.group(1).strip() if fill else "black"
        return st[:-1] + (f";stroke:{col};stroke-width:{sw:.3f}px;"
                          f"paint-order:stroke fill\"") if st.endswith('"') else st
    src = re.sub(r'style="[^"]*"', _stroke, src)
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
            body = m.group(1)
            # 变量表达式允许：公式里的变量必须是 CM 斜体——$p$、$z_i$、$dW$、
            # $dL/dx$、$(P-Y)/m$。不给含 = 或数字开头的（$K=64$、$2h$ 仍拒绝）。
            if "=" not in body and re.fullmatch(
                    r"[A-Za-zΑ-Ωα-ω(_{][A-Za-z0-9_^{}\\ ()+\-*/·×]*", body.strip()):
                continue
            if not re.search(r"[_^\\]", _PLAIN_CMDS.sub("", body)):
                bad.append(s)
    if bad:
        raise AssertionError(
            f"{len(bad)} 处无需排版的 $...$（应写成普通文字/Unicode）：\n"
            + "\n".join(repr(b) for b in sorted(set(bad))))


def check_plain_text(fig):
    """$...$ 之外的正文必须是干净的 Unicode 文字：
    不允许出现反斜杠（\\times、\\n 漏排）、_ / ^（d_g 漏排）、
    以及当作负号/减号的 ASCII -（-1、(−1) 应写 −）。
    单词里的连字符（next-token）与代码标识符不受影响。
    控制字符（\\t 等，常见于非 raw 串里 \\to 被吃成 TAB+o）一律报错。"""
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        if re.search(r"[\x00-\x08\x0b-\x1f\x7f\x09]", s):
            bad.append(("控制字符", s))
        plain = re.sub(r"\$[^$]*\$", "", s)
        if "\\" in plain:
            bad.append(("反斜杠", s))
        for m_ in re.finditer(r"\w*[_^]\w*", plain):
            # 代码标识符豁免：两侧都 ≥2 字符（next_functions、to_empty），或 _ 开头的
            # 私有名（_ParamAndGradBuffer）；d_g、e^2 这类漏排仍报错
            if not re.fullmatch(r"\w{2,}[_^]\w{2,}|[_^]\w{2,}", m_.group(0)):
                bad.append(("_ 或 ^", s))
                break
        if re.search(r"(?<![A-Za-z0-9])-\d", plain):
            bad.append(("ASCII 负号/减号", s))
        # 希腊字母作为变量必须走 mathtext（$\tau$、$\delta$），不要写成平文本 Unicode——
        # 无衬线希腊字母和 CM 数学混在一起不齐。μ 例外：多半是 μs/μm 单位前缀。
        if re.search(r"[α-ωΑ-Ω]", plain.replace("μ", "")):
            bad.append(("希腊字母应为 mathtext", s))
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



def style_titles(fig):
    """标题类文字（suptitle / ax.title）换成 Medium 字重；正文一律 Regular。"""
    for t in ([fig._suptitle] if fig._suptitle is not None else []) + [
            ax.title for ax in fig.axes]:
        if t is not None and t.get_text().strip():
            t.set_fontname("Noto Sans SC Medium")


def checks(fig):
    """save() 前的全部文字检查。"""
    style_titles(fig)
    check_math_needed(fig)
    check_plain_text(fig)
    check_math_line_spacing(fig)
    check_text_in_rect(fig)


def strip_ws(path):
    """去掉 SVG 行尾空白（git diff --check / check.sh 会抓）。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(l.rstrip() for l in lines) + "\n")


def enable_embed(mod):
    """包一层 mod.save（如 lst_plot.save）：存盘后跑文字溢出/数学检查并内嵌字体。"""
    orig = mod.save

    def wrapped(fig, name, *a, **k):
        style_titles(fig)
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


# ---- 13/14 px 文字下限（各系列共用）----
COLUMN_PX = 741
MIN_TEXT_PX = 13
MIN_MATH_PX = 14


def _final_width_px(fig):
    renderer = fig.canvas.get_renderer()
    return fig.get_tightbbox(renderer).width * 96


def check_text_floor(fig, name):
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
