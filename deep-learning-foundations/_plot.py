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



# ---- Noto Sans SC + SVG 内嵌字体子集（共享实现见 ../figtext.py）----
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figtext as _ft  # noqa: E402  导入即注册 Noto + svg.fonttype=none

register_noto = _ft.register_noto
embed_fonts = _ft.embed_fonts
check_text_in_rect = _ft.check_text_in_rect
check_math_needed = _ft.check_math_needed
check_plain_text = _ft.check_plain_text
check_math_line_spacing = _ft.check_math_line_spacing
enable_embed = _ft.enable_embed


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


