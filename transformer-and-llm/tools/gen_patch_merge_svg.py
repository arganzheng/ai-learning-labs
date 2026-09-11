"""Generates the patch -> merge -> token figure used in Transformer 与 LLM（08）. Usage: python gen_patch_merge_svg.py [out.svg]"""
import sys
W, H = 740, 425
out = []
a = out.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="12">')
a(f'<rect width="{W}" height="{H}" fill="#fff"/>')
a('<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#333"/></marker></defs>')
colors = ["#dbe9ff", "#ffe4c4", "#d9f2d9", "#f3d9f2"]
def gcol(gh, gw): return colors[(gh + gw) % 4]

top = 52
# ---- left: 8x8 patches
lx, ly, cell = 20, top, 22
a(f'<text x="{lx}" y="24" font-weight="bold">① patchify</text>')
a(f'<text x="{lx}" y="40" fill="#666">112×112 px → 8×8 个 14×14 patch</text>')
for i in range(8):
    for j in range(8):
        x, y = lx + j * cell, ly + i * cell
        a(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{gcol(i//2, j//2)}" stroke="#999" stroke-width="0.8"/>')
        a(f'<text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" font-size="9" fill="#444">{i*8+j}</text>')
a(f'<text x="{lx}" y="{ly+8*cell+18}" fill="#444" font-size="11">64 行 × d=1280，进 ViT</text>')

# arrow 1
ax = lx + 8 * cell + 12
a(f'<path d="M{ax},{ly+4*cell} L{ax+44},{ly+4*cell}" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>')
a(f'<text x="{ax+22}" y="{ly+4*cell-10}" text-anchor="middle" font-size="10" fill="#333">ViT ×32 层</text>')
a(f'<text x="{ax+22}" y="{ly+4*cell+22}" text-anchor="middle" font-size="10" fill="#333">2×2 merge</text>')

# ---- middle: 4x4 groups
mx, my, mcell = ax + 66, top, 44
a(f'<text x="{mx}" y="24" font-weight="bold">② 2×2 空间合并 → 16 个 image token</text>')
a(f'<text x="{mx}" y="40" fill="#666">4 patch 拼接(4×1280) → MLP → d_model</text>')
for gi in range(4):
    for gj in range(4):
        x, y = mx + gj * mcell, my + gi * mcell
        a(f'<rect x="{x}" y="{y}" width="{mcell}" height="{mcell}" fill="{gcol(gi, gj)}" stroke="#666" stroke-width="1"/>')
        p = [(2*gi)*8 + 2*gj, (2*gi)*8 + 2*gj+1, (2*gi+1)*8 + 2*gj, (2*gi+1)*8 + 2*gj+1]
        a(f'<text x="{x+mcell/2}" y="{y+15}" text-anchor="middle" font-size="9" fill="#555">{p[0]},{p[1]}</text>')
        a(f'<text x="{x+mcell/2}" y="{y+26}" text-anchor="middle" font-size="9" fill="#555">{p[2]},{p[3]}</text>')
        a(f'<text x="{x+mcell/2}" y="{y+39}" text-anchor="middle" font-size="10" font-weight="bold" fill="#222">t{gi*4+gj}</text>')
a(f'<text x="{mx}" y="{my+4*mcell+18}" fill="#444" font-size="11">16 行 × d_model，进 decoder</text>')

# arrow 2
ax2 = mx + 4 * mcell + 12
a(f'<path d="M{ax2},{my+2*mcell} L{ax2+40},{my+2*mcell}" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>')
a(f'<text x="{ax2+20}" y="{my+2*mcell-10}" text-anchor="middle" font-size="10" fill="#333">行优先</text>')
a(f'<text x="{ax2+20}" y="{my+2*mcell+22}" text-anchor="middle" font-size="10" fill="#333">展平</text>')

# ---- right: token sequence
rx, ry, rw, rh = ax2 + 60, top, 92, 17
a(f'<text x="{rx}" y="24" font-weight="bold">③ decoder 序列</text>')
a(f'<text x="{rx}" y="40" fill="#666">M-RoPE 位置 (t, h, w)</text>')
for k in range(16):
    gi, gj = divmod(k, 4)
    y = ry + k * rh
    a(f'<rect x="{rx}" y="{y}" width="{rw}" height="{rh}" fill="{gcol(gi, gj)}" stroke="#666" stroke-width="0.8"/>')
    a(f'<text x="{rx+6}" y="{y+12}" font-size="10" fill="#222">t{k}</text>')
    a(f'<text x="{rx+rw-5}" y="{y+12}" font-size="10" text-anchor="end" fill="#333">(t₀, {gi}, {gj})</text>')
a(f'<text x="{rx}" y="{ry+16*rh+18}" font-size="11" fill="#444">每行 1 token = 每层 1 份 KV</text>')

# ---- bottom caption
cy = top + 16 * rh + 46
a(f'<line x1="{lx}" y1="{cy-14}" x2="{W-lx}" y2="{cy-14}" stroke="#ddd"/>')
a(f'<text x="{lx}" y="{cy}" fill="#333" font-size="11">image token 数 = (H/28)·(W/28)：336² → 144，1024² → 37×37 = 1369，1920×1080 → 2691（Qwen2-VL，patch 14，merge 2）</text>')
a(f'<text x="{lx}" y="{cy+17}" fill="#333" font-size="11">LLaVA-1.5 没有 merge，336² 的 576 个 patch 就是 576 个 token；InternVL 的 pixel-shuffle 同样把 4 个 patch 折成 1 个 token</text>')
a(f'<text x="{lx}" y="{cy+34}" fill="#333" font-size="11">Qwen2.5-VL 里这 8×8 个 patch 正好是一个 attention window（112 px）：32 层中 28 层只在窗口内做 attention</text>')
a('</svg>')
out_path = sys.argv[1] if len(sys.argv) > 1 else "multimodal-vision-encoder-cost-patch-merge.svg"
open(out_path, "w").write("\n".join(out))
print("wrote", out_path)
