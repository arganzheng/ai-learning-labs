"""gen_attention_by_hand_svg.py — 文章 01 的图：d = 4、T = 3 的 attention 六步手算（数字来自 attention_by_hand.py）。

输出 SVG（根元素带 width/height）到博客 img/in-post/。纯标准库。
"""
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/Users/argan/Code/arganzheng.github.com/img/in-post/transformer-01-attention-by-hand.svg"
FONT = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif\""
BLUE, RED, ORG, GRN, PUR, GRAY = '#1f6fb2', '#c0392b', '#c98a00', '#4d9a5c', '#8a6bd1', '#999'

def T(x, y, t, fill='#333', size=12, anchor='start', weight='normal'):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>'
def R(x, y, w, h, fill='#fff', stroke='#555', sw=1.2, rx=3):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'
def L(x1, y1, x2, y2, stroke='#555', w=1.5, arrow=True):
    a = ' marker-end="url(#ar)"' if arrow else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{w}"{a}/>'

def matrix(x, y, rows, title, col, cw=44, ch=26, rowlab=None, collab=None, fmt=lambda v: v, hl=None):
    out = [T(x + len(rows[0]) * cw / 2, y - 22, title, col, 12, 'middle', '600')]
    if collab:
        for j, c in enumerate(collab): out.append(T(x + j * cw + cw / 2, y - 6, c, GRAY, 9.5, 'middle'))
    for i, r in enumerate(rows):
        if rowlab: out.append(T(x - 6, y + i * ch + ch / 2 + 4, rowlab[i], GRAY, 9.5, 'end'))
        for j, v in enumerate(r):
            f = '#f3f3f3' if v == '−∞' else ('#fff4e0' if hl and (i, j) in hl else '#fff')
            out.append(R(x + j * cw, y + i * ch, cw, ch, f, col, 1, 0))
            out.append(T(x + j * cw + cw / 2, y + i * ch + ch / 2 + 4, fmt(v), '#222' if v != '−∞' else '#aaa', 11, 'middle'))
    return ''.join(out)

toks = ['t0', 't1', 't2']
b = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="560" viewBox="0 0 760 560" {FONT} font-size="12">',
     '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>',
     T(380, 22, '一个 attention 头的六步手算：d = 4、T = 3 个 token（t0、t1、t2）', '#222', 14, 'middle', '600')]

# row 1: x -> Q, K, V
y = 70
b.append(matrix(40, y, [['1', '0', '1', '0'], ['0', '1', '0', '1'], ['1', '1', '0', '0']], '① 输入 x [3, 4]', '#555', cw=34, rowlab=toks))
b.append(T(200, y + 30, '× W_Q', BLUE, 10, 'middle')); b.append(T(200, y + 44, '× W_K', RED, 10, 'middle')); b.append(T(200, y + 58, '× W_V', GRN, 10, 'middle')); b.append(L(184, y + 70, 216, y + 70))
b.append(matrix(228, y, [['2', '0', '1', '0'], ['0', '2', '0', '1'], ['1', '1', '0', '0']], 'Q = x W_Q', BLUE, cw=34))
b.append(matrix(388, y, [['1', '0', '2', '0'], ['0', '1', '0', '2'], ['1', '1', '1', '1']], 'K = x W_K', RED, cw=34))
b.append(matrix(548, y, [['1', '0', '1', '0'], ['0', '1', '0', '1'], ['1', '1', '0', '0']], 'V = x W_V（取恒等）', GRN, cw=34))
b.append(T(400, y + 100, '每个 token 各自算出三个向量：query「我在找什么」、key「我是什么」、value「我能提供什么」；三个投影矩阵是可训练参数', '#666', 10.5, 'middle'))

# row 2: S, S/sqrt(d), mask
y = 210
b.append(matrix(60, y, [['4', '0', '3'], ['0', '4', '3'], ['1', '1', '2']], '② S = Q Kᵀ [3, 3]', PUR, rowlab=toks, collab=toks))
b.append(T(60 + 66, y + 96, 'S[i, j] = qᵢ · kⱼ', '#666', 10, 'middle'))
b.append(L(200, y + 40, 228, y + 40)); b.append(T(214, y + 32, '÷ √4 = 2', '#333', 10, 'middle'))
b.append(matrix(240, y, [['2.0', '0.0', '1.5'], ['0.0', '2.0', '1.5'], ['0.5', '0.5', '1.0']], '③ 除以 √d', PUR, rowlab=None, collab=toks))
b.append(T(240 + 66, y + 96, '压回 1 的量级', '#666', 10, 'middle'))
b.append(L(380, y + 40, 408, y + 40)); b.append(T(394, y + 32, 'mask', '#333', 10, 'middle'))
b.append(matrix(420, y, [['2.0', '−∞', '−∞'], ['0.0', '2.0', '−∞'], ['0.5', '0.5', '1.0']], '④ causal mask', PUR, rowlab=None, collab=toks))
b.append(T(420 + 66, y + 96, '看未来的位置填 −∞', '#666', 10, 'middle'))
b.append(L(560, y + 40, 588, y + 40)); b.append(T(574, y + 32, 'softmax', '#333', 10, 'middle'))
b.append(matrix(600, y, [['1.00', '0', '0'], ['0.12', '0.88', '0'], ['0.27', '0.27', '0.45']], '⑤ 权重 P（每行和 1）', ORG, rowlab=None, collab=toks, hl={(2, 0), (2, 1), (2, 2)}))
b.append(T(600 + 66, y + 96, 'e^−∞ = 0，权重恰为 0', '#666', 10, 'middle'))

# row 3: P V = out
y = 392
b.append(T(40, y - 52, '⑥ out = P V：每个 token 的输出 = 它能看到的 token 的 value 的加权平均', '#222', 12, 'start', '600'))
b.append(matrix(60, y, [['1.00', '0', '0'], ['0.12', '0.88', '0'], ['0.27', '0.27', '0.45']], 'P', ORG, rowlab=toks, collab=toks))
b.append(T(210, y + 44, '×', '#333', 16, 'middle'))
b.append(matrix(230, y, [['1', '0', '1', '0'], ['0', '1', '0', '1'], ['1', '1', '0', '0']], 'V', GRN, cw=34))
b.append(T(385, y + 44, '=', '#333', 16, 'middle'))
b.append(matrix(405, y, [['1.00', '0.00', '1.00', '0.00'], ['0.12', '0.88', '0.12', '0.88'], ['0.73', '0.73', '0.27', '0.27']], 'out [3, 4]', '#555', cw=52, rowlab=toks))
b.append(T(60, y + 110, 't2 那一行：0.27 × v0 + 0.27 × v1 + 0.45 × v2 = 0.27 × (1,0,1,0) + 0.27 × (0,1,0,1) + 0.45 × (1,1,0,0) = (0.73, 0.73, 0.27, 0.27)', '#333', 11))
b.append(T(60, y + 130, 't0 那一行只能看自己，输出就是 v0；t1 看 t0 与 t1，88% 是自己。输出形状与输入相同 [3, 4]——所以可以一层层叠。', '#666', 10.5))
b.append(T(60, y + 158, '真实模型只是把 4 换成 128（一个头的维度）、3 换成上下文长度、再并排 32 个头；每一步的公式一个字都不变。', '#333', 11, 'start', '600'))
b.append('</svg>')
open(OUT, 'w', encoding='utf-8').write('\n'.join(b)); print('saved', OUT)
