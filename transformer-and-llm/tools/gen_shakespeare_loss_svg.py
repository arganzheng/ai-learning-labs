"""gen_shakespeare_loss_svg.py — 文章 04 的图：2 / 4 / 8 层在 shakespeare_char 上的 val loss 曲线（数据来自 expected/train_shakespeare_char_*.txt）。"""
import re, sys, math
EXP = "/Users/argan/Code/ai-learning-labs/transformer-and-llm/expected/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/Users/argan/Code/arganzheng.github.com/img/in-post/transformer-04-shakespeare-loss-by-depth.svg"
FONT = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif\""
runs = [("L2", "2 层（0.40M）", '#c98a00'), ("base", "4 层（0.80M）", '#1f6fb2'), ("L8", "8 层（1.59M）", '#c0392b')]
data = {}
for key, _, _ in runs:
    pts = []
    for line in open(EXP + f"train_shakespeare_char_{key}.txt"):
        m = re.match(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", line)
        if m: pts.append((int(m[1]), float(m[2]), float(m[3])))
    data[key] = pts
W, H = 760, 400; x0, y0, x1, y1 = 70, 50, 560, 340
xmax, ymin, ymax = 1750, 1.4, 4.4
def X(s): return x0 + (x1 - x0) * s / xmax
def Y(v): return y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)
b = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT} font-size="12">',
     f'<text x="{W/2}" y="24" text-anchor="middle" font-size="14" font-weight="600" fill="#222">只改层数：shakespeare_char 上 2000 步的 val loss（实线）与 train loss（虚线）</text>']
for v in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    b.append(f'<line x1="{x0}" y1="{Y(v)}" x2="{x1}" y2="{Y(v)}" stroke="#eee"/>'); b.append(f'<text x="{x0-8}" y="{Y(v)+4}" text-anchor="end" fill="#666" font-size="11">{v}</text>')
for s in range(0, 1751, 250):
    b.append(f'<text x="{X(s)}" y="{y1+18}" text-anchor="middle" fill="#666" font-size="11">{s}</text>')
b.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#999"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#999"/>')
b.append(f'<text x="{(x0+x1)/2}" y="{y1+36}" text-anchor="middle" fill="#333" font-size="11">训练步数（每步 12 × 64 = 768 个字符）</text>')
b.append(f'<text x="18" y="{(y0+y1)/2}" text-anchor="middle" fill="#333" font-size="11" transform="rotate(-90 18 {(y0+y1)/2})">loss（每字符交叉熵）</text>')
b.append(f'<line x1="{x0}" y1="{Y(math.log(65))}" x2="{x1}" y2="{Y(math.log(65))}" stroke="#bbb" stroke-dasharray="2 3"/><text x="{x1+16}" y="{Y(math.log(65))+4}" fill="#888" font-size="10.5">ln 65 = 4.17：随机乱猜</text>')
for i, (key, label, col) in enumerate(runs):
    pts = data[key]
    b.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (col, ' '.join(f'{X(s)},{Y(v)}' for s, _, v in pts)))
    b.append('<polyline fill="none" stroke="%s" stroke-width="1.2" stroke-dasharray="4 3" points="%s"/>' % (col, ' '.join(f'{X(s)},{Y(t)}' for s, t, _ in pts)))
    for s, _, v in pts: b.append(f'<circle cx="{X(s)}" cy="{Y(v)}" r="2.5" fill="{col}"/>')
    s, t, v = pts[-1]
    ly = Y(v) + 4 + (i - 1) * 14 if key != "L2" else Y(v) + 4 - 6
    b.append(f'<line x1="{X(s)+3}" y1="{Y(v)}" x2="{X(s)+14}" y2="{ly-4}" stroke="{col}" stroke-width="0.8"/>')
    b.append(f'<text x="{X(s)+16}" y="{ly}" fill="{col}" font-size="11" font-weight="600">{label}：val {v:.2f}</text>')
b.append(f'<text x="{x1+16}" y="{y0+90}" fill="#333" font-size="11">实线 val、虚线 train；</text>')
b.append(f'<text x="{x1+16}" y="{y0+106}" fill="#333" font-size="11">1000 步后虚线都跑到实线下面</text>')
b.append(f'<text x="{x1+16}" y="{y0+122}" fill="#333" font-size="11">——过拟合开始</text>')
b.append('</svg>')
open(OUT, 'w', encoding='utf-8').write('\n'.join(b)); print('saved', OUT)
