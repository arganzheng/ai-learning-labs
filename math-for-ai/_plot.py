"""画图的公共设置：中文字体、博客正文列宽（750 px）对应的图宽、SVG 输出到 out/。

    from _plot import plt, save, C
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ...
    save(fig, "05-coin-likelihood")      # -> out/05-coin-likelihood.svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- Noto Sans SC + SVG 内嵌字体子集（共享实现见 ../figtext.py）----
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figtext as _ft  # noqa: E402  导入即注册 Noto + svg.fonttype=none

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

C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd"}


def _unicode_minus_ticks(fig):
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
    if os.environ.get("PLOT_PNG"):
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
