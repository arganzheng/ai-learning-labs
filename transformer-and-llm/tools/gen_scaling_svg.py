"""Generates the two-panel scaling-law figure used in Transformer 与 LLM（10）:
left  = the CPU experiment of scaling_law_fit.py (loss vs N, log x, fitted power law, held-out point),
right = Chinchilla iso-compute curve: loss vs N at fixed C = 7.2e23 with the compute-optimal point and Llama-3 8B.
Usage: python gen_scaling_svg.py [out.svg]
"""
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from llm_cost_10_scaling import chinchilla_loss, compute_optimal  # noqa: E402

# 左图数据：scaling_law_fit.py 的完整运行（expected/scaling_law_fit.txt）
POINTS = [(14304, 2.0025), (25216, 1.8828), (56256, 1.7204), (99584, 1.6186),
          (223104, 1.5027), (395776, 1.4142)]
HELD_OUT = (888576, 1.3415)
E_, A_, AL_ = 0.618, 6.78, 0.166

W, H = 740, 330
out = []
a = out.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
  f'font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="11">')
a(f'<rect width="{W}" height="{H}" fill="#fff"/>')


def panel(x0, y0, w, h, title, xlab, ylab, xlog, xr, yr, curves, points, notes):
    a(f'<text x="{x0 + w / 2}" y="{y0 - 8}" text-anchor="middle" font-weight="bold" font-size="12">{title}</text>')
    a(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>')

    def px(x):
        t = (math.log10(x) - math.log10(xr[0])) / (math.log10(xr[1]) - math.log10(xr[0])) if xlog else (x - xr[0]) / (xr[1] - xr[0])
        return x0 + t * w

    def py(y):
        return y0 + h - (y - yr[0]) / (yr[1] - yr[0]) * h

    # x ticks (decades or linear)
    if xlog:
        d = int(math.floor(math.log10(xr[0])))
        while 10 ** d <= xr[1]:
            v = 10 ** d
            if v >= xr[0]:
                a(f'<line x1="{px(v)}" y1="{y0}" x2="{px(v)}" y2="{y0 + h}" stroke="#eee"/>')
                lab = f"10<tspan baseline-shift=\"super\" font-size=\"8\">{d}</tspan>"
                a(f'<text x="{px(v)}" y="{y0 + h + 14}" text-anchor="middle" fill="#555">{lab}</text>')
            d += 1
    # y ticks
    step = (yr[1] - yr[0]) / 4
    for i in range(5):
        v = yr[0] + i * step
        a(f'<line x1="{x0}" y1="{py(v)}" x2="{x0 + w}" y2="{py(v)}" stroke="#eee"/>')
        a(f'<text x="{x0 - 4}" y="{py(v) + 4}" text-anchor="end" fill="#555">{v:.2f}</text>')
    a(f'<text x="{x0 + w / 2}" y="{y0 + h + 30}" text-anchor="middle" fill="#333">{xlab}</text>')
    a(f'<text transform="translate({x0 - 40},{y0 + h / 2}) rotate(-90)" text-anchor="middle" fill="#333">{ylab}</text>')
    for color, dash, fn in curves:
        pts = []
        n = 120
        for i in range(n + 1):
            x = 10 ** (math.log10(xr[0]) + (math.log10(xr[1]) - math.log10(xr[0])) * i / n) if xlog else xr[0] + (xr[1] - xr[0]) * i / n
            y = fn(x)
            if yr[0] <= y <= yr[1]:
                pts.append(f"{px(x):.1f},{py(y):.1f}")
        a(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="{dash}"/>')
    for x, y, color, shape in points:
        if shape == "o":
            a(f'<circle cx="{px(x)}" cy="{py(y)}" r="4" fill="{color}"/>')
        else:
            a(f'<rect x="{px(x) - 4.5}" y="{py(y) - 4.5}" width="9" height="9" fill="none" stroke="{color}" stroke-width="2"/>')
    for x, y, dx, dy, text, color in notes:
        a(f'<text x="{px(x) + dx}" y="{py(y) + dy}" fill="{color}" font-size="10.5">{text}</text>')


# ---- left panel: CPU experiment
panel(58, 40, 290, 220,
      "CPU 实验：loss 随 N（2 层字符级模型，10M token）", "N（非 embedding 参数）", "验证 loss（nats）",
      True, (1e4, 1.2e6), (1.2, 2.1),
      [("#0085a1", "", lambda n: E_ + A_ / n ** AL_)],
      [(x, y, "#0085a1", "o") for x, y in POINTS] + [(HELD_OUT[0], HELD_OUT[1], "#d9480f", "s")],
      [(HELD_OUT[0], HELD_OUT[1], -150, 10, "外推 1.317，实测 1.342", "#d9480f"),
       (3e4, 1.35, 0, 0, "L = 0.62 + 6.78 / N^0.166", "#0085a1")])

# ---- right panel: iso-compute
C = 6 * 8e9 * 15e12
No, Do = compute_optimal(C)
panel(430, 40, 290, 220,
      "固定算力 C = 7.2e23：loss 随 N（D = C / 6N）", "N（参数量）", "Chinchilla 预测 loss（nats）",
      True, (2e9, 4e11), (1.95, 2.15),
      [("#0085a1", "", lambda n: chinchilla_loss(n, C / (6 * n)))],
      [(No, chinchilla_loss(No, Do), "#0085a1", "o"), (8e9, chinchilla_loss(8e9, 15e12), "#d9480f", "s")],
      [(No, chinchilla_loss(No, Do), -60, -12, f"最优 {No / 1e9:.0f}B / {Do / 1e12:.1f}T", "#0085a1"),
       (8e9, chinchilla_loss(8e9, 15e12), 8, -8, "Llama-3 8B：+0.054，推理便宜 10×", "#d9480f")])

a(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" fill="#666" font-size="10.5">'
  '左：用前 6 个点拟合、外推第 7 个；右：同一算力下模型越小、数据越多，loss 只在最优点附近很平，右侧上翘处 D 不够，左侧上翘处 N 不够</text>')
a("</svg>")
open(sys.argv[1] if len(sys.argv) > 1 else "scaling.svg", "w").write("\n".join(out))
