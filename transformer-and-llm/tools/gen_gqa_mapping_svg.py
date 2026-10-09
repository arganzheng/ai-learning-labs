"""gen_gqa_mapping_svg.py — 文章 05 的图：MHA / GQA / MQA 三种 Q head 到 KV head 的映射（n_h = 8）。纯标准库。"""
import sys
from pathlib import Path

OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "out"
FONT = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif\""
BLUE, ORG, GRAY = "#1f6fb2", "#c98a00", "#888"


def text(x, y, t, fill="#333", size=12, anchor="start", weight="normal"):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>'


def rect(x, y, w, h, fill, stroke):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.2" rx="4"/>'


def line(x1, y1, x2, y2, stroke):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="1.3"/>'


N_H = 8
W, CW, X0 = 940, 66, 290
ROWS = [
    ("MHA", 8, "n_kv = n_h = 8，g = 1", "每个 Q head 独占一组 K/V"),
    ("GQA", 2, "n_kv = 2，g = 4", "每 4 个 Q head 共用一组 K/V"),
    ("MQA", 1, "n_kv = 1，g = 8", "所有 Q head 共用一组 K/V"),
]
RH = 150
body = [text(W / 2, 26, "MHA / GQA / MQA：第 i 个 Q head 读第 ⌊i / g⌋ 组 K/V（n_h = 8，g = n_h / n_kv）", "#222", 14, "middle", "600")]
for r, (name, n_kv, cfg, desc) in enumerate(ROWS):
    y = 56 + r * RH
    g = N_H // n_kv
    body.append(text(30, y + 22, name, "#222", 16, "start", "600"))
    body.append(text(30, y + 44, cfg, "#444", 11.5))
    body.append(text(30, y + 64, desc, "#444", 11.5))
    body.append(text(30, y + 84, f"每 token 缓存 K、V 各 {n_kv} × d_head", ORG, 11.5, "start", "600"))
    body.append(text(X0 - 12, y + 22, "Q head", GRAY, 11, "end"))
    body.append(text(X0 - 12, y + 102, "KV head", GRAY, 11, "end"))
    for k in range(n_kv):
        span = g * CW
        kx = X0 + k * span + span / 2
        for i in range(k * g, (k + 1) * g):
            qx = X0 + i * CW + CW / 2
            body.append(line(qx, y + 30, kx, y + 84, "#9bb8d6"))
        kw = min(span - 10, 120)
        body.append(rect(kx - kw / 2, y + 84, kw, 26, "#fff4e0", ORG))
        body.append(text(kx, y + 102, f"K/V {k}", "#222", 12, "middle"))
    for i in range(N_H):
        qx = X0 + i * CW + 5
        body.append(rect(qx, y + 4, CW - 10, 26, "#eef4fb", BLUE))
        body.append(text(qx + (CW - 10) / 2, y + 22, f"Q {i}", "#222", 12, "middle"))
H = 56 + 3 * RH
s = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT} font-size="12">\n<rect width="100%" height="100%" fill="#fff"/>\n' + "\n".join(body) + "\n</svg>\n"
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "transformer-05-mha-gqa-mqa-mapping.svg").write_text(s, encoding="utf-8")
print("saved transformer-05-mha-gqa-mqa-mapping.svg")
