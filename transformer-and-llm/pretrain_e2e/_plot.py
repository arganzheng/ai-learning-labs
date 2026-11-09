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

plt.rcParams.update({
    "axes.unicode_minus": True,
    "svg.fonttype": "none",          # 真文字 + 内嵌字体子集（embed_fonts）
    "mathtext.fontset": "cm",        # 公式用 Computer Modern，跟 KaTeX 一致
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "figure.constrained_layout.use": True,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

C = {"blue": "#3b6fb6", "orange": "#e0812c", "green": "#4a9a5b", "red": "#c94c4c",
     "gray": "#8a8a8a", "purple": "#7b5ea7", "light": "#dddddd"}

# 真文字+内嵌字体（Noto Sans SC / CM 子集）：须在创建任何 Text 前 import
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import figtext  # noqa: E402,F401  导入即 register_noto()


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    figtext.check_text_floor(fig, name)
    figtext.checks(fig)
    path = os.path.join(OUT, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    figtext.strip_ws(path)
    figtext.embed_fonts(path)
    if os.environ.get("PLOT_PNG"):
        os.makedirs(os.environ["PLOT_PNG"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PLOT_PNG"], f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存 out/{name}.svg")
    return path
