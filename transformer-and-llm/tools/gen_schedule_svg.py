"""Generates the learning-rate-schedule / batch-ramp figure used in Transformer 与 LLM（12）.
Left: lr (fraction of peak) over training tokens for cosine→10%, WSD (80/20) and DeepSeek-V3's four-stage schedule.
Right: batch size in tokens over training for Llama-3 405B and DeepSeek-V3 (log y).
Usage: python gen_schedule_svg.py [out.svg]
"""
import math
import sys

W, H = 740, 315
out = []
a = out.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
  f'font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="11">')
a(f'<rect width="{W}" height="{H}" fill="#fff"/>')


def axes(x0, y0, w, h, title, xlab, ylab, yticks, ylog=False):
    a(f'<text x="{x0 + w / 2}" y="{y0 - 8}" text-anchor="middle" font-weight="bold" font-size="12">{title}</text>')
    a(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>')
    for i in range(6):
        x = x0 + w * i / 5
        a(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + h}" stroke="#eee"/>')
        a(f'<text x="{x}" y="{y0 + h + 14}" text-anchor="middle" fill="#555">{i * 20}%</text>')
    for v, lab in yticks:
        y = ypos(v, y0, h, ylog)
        a(f'<line x1="{x0}" y1="{y}" x2="{x0 + w}" y2="{y}" stroke="#eee"/>')
        a(f'<text x="{x0 - 4}" y="{y + 4}" text-anchor="end" fill="#555">{lab}</text>')
    a(f'<text x="{x0 + w / 2}" y="{y0 + h + 30}" text-anchor="middle" fill="#333">{xlab}</text>')
    a(f'<text transform="translate({x0 - 44},{y0 + h / 2}) rotate(-90)" text-anchor="middle" fill="#333">{ylab}</text>')


def ypos(v, y0, h, ylog):
    if ylog:
        t = (math.log10(v) - math.log10(1e6)) / (math.log10(1e8) - math.log10(1e6))
    else:
        t = v
    return y0 + h - t * h


def curve(fn, x0, y0, w, h, color, ylog=False, n=400, dash=""):
    pts = [f"{x0 + w * i / n:.1f},{ypos(fn(i / n), y0, h, ylog):.1f}" for i in range(n + 1)]
    a(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="{dash}"/>')


def label(x, y, text, color):
    a(f'<text x="{x}" y="{y}" fill="{color}" font-size="10.5">{text}</text>')


# ---- left: lr schedules (t = fraction of tokens) ----
WARM = 0.01


def warm(t):
    return min(1.0, t / WARM)


def cosine(t):
    return warm(t) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def wsd(t):
    return warm(t) * (1.0 if t < 0.8 else max(0.0, (1 - t) / 0.2))


def deepseek_v3(t):
    # 14.8T total: warmup 2K steps (~0.85%), constant to 10T (67.6%), cosine to 2.2e-5 over 4.3T (to 96.6%),
    # constant 333B at 2.2e-5 (to 98.9%), then 7.29e-6 for 167B.
    tok = t * 14.8
    if tok < 10.0:
        return warm(t)
    if tok < 14.3:
        u = (tok - 10.0) / 4.3
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u))
    if tok < 14.633:
        return 0.1
    return 7.29e-6 / 2.2e-4


LX, LY, LW, LH = 60, 40, 290, 200
axes(LX, LY, LW, LH, "学习率调度：lr / 峰值 随训练进度", "训练 token 进度", "lr / 峰值",
     [(0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")])
curve(cosine, LX, LY, LW, LH, "#0085a1")
curve(wsd, LX, LY, LW, LH, "#d9480f")
curve(deepseek_v3, LX, LY, LW, LH, "#2b8a3e", dash="5,3")
label(LX + 60, LY + 62, "cosine → 10%（GPT-3、Llama 2 / 3）", "#0085a1")
label(LX + 20, LY + 22, "WSD：常数 80% + 线性衰减 20%（MiniCPM）", "#d9480f")
label(LX + 120, LY + 118, "DeepSeek-V3：常数到 10T，", "#2b8a3e")
label(LX + 120, LY + 131, "cosine 4.3T，再两级常数", "#2b8a3e")

# ---- right: batch ramps (tokens per batch, log y) ----
RX, RY, RW, RH = 430, 40, 290, 200
axes(RX, RY, RW, RH, "batch 大小（token）随训练进度", "训练 token 进度", "每个 batch 的 token 数",
     [(1e6, "1M"), (4e6, "4M"), (16e6, "16M"), (64e6, "64M")], ylog=True)


def llama3_batch(t):
    tok = t * 15.6e12
    return 4e6 if tok < 252e9 else (8e6 if tok < 2.87e12 else 16e6)


def dsv3_batch(t):
    tok = t * 14.8e12
    if tok >= 469e9:
        return 15360 * 4096
    return (3072 + (15360 - 3072) * tok / 469e9) * 4096


curve(llama3_batch, RX, RY, RW, RH, "#0085a1", ylog=True, n=800)
curve(dsv3_batch, RX, RY, RW, RH, "#2b8a3e", ylog=True, n=800, dash="5,3")
label(RX + 70, RY + 118, "Llama-3 405B：4M → 8M（1.6%）→ 16M（18%）", "#0085a1")
label(RX + 60, RY + 48, "DeepSeek-V3：12.6M → 63M，前 3.2% 线性增大", "#2b8a3e")

a(f'<text x="{W / 2}" y="{H - 22}" text-anchor="middle" fill="#666" font-size="10.5">'
  '左：cosine 从一开始就在降，WSD 与 DeepSeek 的常数段让中途 checkpoint 之间可以比较；</text>')
a(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" fill="#666" font-size="10.5">'
  '右：batch 在训练早期增大，对应最优 batch 随 loss 下降而变大（梯度噪声尺度）</text>')
a("</svg>")
open(sys.argv[1] if len(sys.argv) > 1 else "schedule.svg", "w").write("\n".join(out))
