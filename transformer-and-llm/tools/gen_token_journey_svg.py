"""gen_token_journey_svg.py — 文章 02 的两张图：训练侧一步（teacher forcing → T 个 loss → 反向 → 更新）、
推理侧（prefill 一次算完 prompt，decode 每步只算一个新 token，KV cache 逐步增长）。纯标准库。"""
import sys
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/Users/argan/Code/arganzheng.github.com/img/in-post/"
FONT = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif\""
BLUE, RED, ORG, GRN, PUR, GRAY = '#1f6fb2', '#c0392b', '#c98a00', '#4d9a5c', '#8a6bd1', '#888'
def T(x, y, t, fill='#333', size=12, anchor='start', weight='normal'):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>'
def R(x, y, w, h, fill='#fff', stroke='#555', sw=1.2, rx=4, dash=''):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"{d}/>'
def L(x1, y1, x2, y2, stroke='#555', w=1.5, arrow=True, dash=''):
    a = ' marker-end="url(#ar)"' if arrow else ''; d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{w}"{a}{d}/>'
def svg(w, h, body, name):
    s = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT} font-size="12">\n<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>\n{body}\n</svg>\n'
    open(OUTDIR + name, 'w', encoding='utf-8').write(s); print('saved', name)

# ---------------- 1. training step ----------------
toks = ['The', 'cat', 'sat', 'on', 'the']; tgts = ['cat', 'sat', 'on', 'the', 'mat']
b = [T(380, 22, '训练的一步：一句话的 T 个位置同时各预测下一个 token，得到 T 个 loss，反向一次更新全部参数', '#222', 13.5, 'middle', '600')]
x0, y0, cw = 60, 60, 92
b.append(T(x0 - 8, y0 + 18, '输入 x', '#333', 11, 'end', '600'))
for i, t in enumerate(toks):
    b.append(R(x0 + i * cw, y0, cw - 6, 28, '#eef4fb', BLUE, 1.2)); b.append(T(x0 + i * cw + (cw - 6) / 2, y0 + 19, f'{t}', '#222', 12, 'middle'))
    b.append(T(x0 + i * cw + (cw - 6) / 2, y0 - 6, f'位置 {i}', GRAY, 9.5, 'middle'))
# model box
b.append(R(x0, y0 + 46, 5 * cw - 6, 54, '#fafbfd', '#999', 1.2, 6))
b.append(T(x0 + (5 * cw - 6) / 2, y0 + 68, '一次前向：embedding → 12 个 block（causal mask：位置 i 只看 0..i）→ lm_head', '#333', 11, 'middle'))
b.append(T(x0 + (5 * cw - 6) / 2, y0 + 88, '输出 logits [T, V]：每个位置一个词表分布——T 个预测同时算出', '#333', 11, 'middle'))
for i in range(5):
    b.append(L(x0 + i * cw + (cw - 6) / 2, y0 + 28, x0 + i * cw + (cw - 6) / 2, y0 + 46, BLUE, 1, True))
    b.append(L(x0 + i * cw + (cw - 6) / 2, y0 + 100, x0 + i * cw + (cw - 6) / 2, y0 + 120, BLUE, 1, True))
b.append(T(x0 - 8, y0 + 140, '预测', '#333', 11, 'end', '600'))
probs = ['p(·|The)', 'p(·|The cat)', 'p(·|…sat)', 'p(·|…on)', 'p(·|…the)']
for i in range(5):
    b.append(R(x0 + i * cw, y0 + 122, cw - 6, 28, '#f3f0fa', PUR, 1.2)); b.append(T(x0 + i * cw + (cw - 6) / 2, y0 + 141, probs[i], '#222', 10.5, 'middle'))
b.append(T(x0 - 8, y0 + 190, '目标 y', '#333', 11, 'end', '600'))
for i, t in enumerate(tgts):
    b.append(R(x0 + i * cw, y0 + 172, cw - 6, 28, '#fff4e0', ORG, 1.2)); b.append(T(x0 + i * cw + (cw - 6) / 2, y0 + 191, t, '#222', 12, 'middle'))
    b.append(L(x0 + i * cw + (cw - 6) / 2, y0 + 150, x0 + i * cw + (cw - 6) / 2, y0 + 172, '#999', 1, True, '3 2'))
b.append(T(x0 + 5 * cw + 4, y0 + 191, '= 输入右移一位', GRAY, 10.5))
b.append(T(x0 - 8, y0 + 232, 'loss', '#333', 11, 'end', '600'))
for i in range(5):
    b.append(R(x0 + i * cw, y0 + 214, cw - 6, 26, '#fdeeee', RED, 1.2)); b.append(T(x0 + i * cw + (cw - 6) / 2, y0 + 232, f'−log p{i}(y{i})', '#222', 10.5, 'middle'))
b.append(T(x0 + 5 * cw + 4, y0 + 232, '→ 平均 = 这一步的 loss', GRAY, 10.5))
# backward arrow
b.append(L(x0 + 5 * cw + 150, y0 + 214, x0 + 5 * cw + 150, y0 + 46, RED, 2, True))
b.append(T(x0 + 5 * cw + 160, y0 + 120, '反向传播', RED, 11, 'start', '600'))
b.append(T(x0 + 5 * cw + 160, y0 + 136, '沿同一条路走回去', RED, 10))
b.append(T(x0 + 5 * cw + 160, y0 + 150, '每个参数得到 .grad', RED, 10))
b.append(T(60, y0 + 270, '为什么可以一次算 T 个：causal mask 保证位置 i 的预测只用了 token 0..i，与推理时「只有前文」的情形完全一致；', '#333', 11))
b.append(T(60, y0 + 288, '目标直接取真实的下一个 token 而不是模型自己刚预测的（teacher forcing），T 个位置互不依赖，所以能并行。', '#333', 11))
b.append(T(60, y0 + 312, '之后 optimizer.step()：每个参数沿 .grad 走一小步（AdamW）；一步结束，换下一个 batch。', '#666', 11))
svg(760, y0 + 330, ''.join(b), 'transformer-02-training-step-teacher-forcing.svg')

# ---------------- 2. inference: prefill + decode ----------------
b = [T(450, 22, '推理：prefill 把 prompt 一次算完并存下每层的 K、V；decode 每步只算一个新 token，K、V 追加进 cache', '#222', 13.5, 'middle', '600')]
y = 60; x0 = 150; cw = 46
def row(y, label, computed, cached, newtok=None, note=''):
    out = [T(x0 - 10, y + 18, label, '#333', 11, 'end', '600')]
    for i, t in enumerate(cached):
        out.append(R(x0 + i * cw, y, cw - 4, 28, '#eaf5ea', GRN, 1)); out.append(T(x0 + i * cw + (cw - 4) / 2, y + 19, t, '#222', 10.5, 'middle'))
    n = len(cached)
    for j, t in enumerate(computed):
        out.append(R(x0 + (n + j) * cw, y, cw - 4, 28, '#fff4e0', ORG, 1.5)); out.append(T(x0 + (n + j) * cw + (cw - 4) / 2, y + 19, t, '#222', 10.5, 'middle', '600'))
    if newtok is not None:
        xx = x0 + (n + len(computed)) * cw + 6
        out.append(L(xx - 8, y + 14, xx + 6, y + 14, PUR, 1.2)); out.append(R(xx + 8, y, cw - 4, 28, '#f3f0fa', PUR, 1.2, dash='3 2')); out.append(T(xx + 8 + (cw - 4) / 2, y + 19, newtok, PUR, 10.5, 'middle'))
    out.append(T(x0 + 10 * cw, y + 19, note, GRAY, 10.5))
    return ''.join(out)
b.append(row(y, 'prefill', ['The', 'cat', 'sat', 'on', 'the'], [], 'mat', '算 5 个（同训练前向），只取最后一个位置的分布'))
b.append(row(y + 44, 'decode 1', ['mat'], ['The', 'cat', 'sat', 'on', 'the'], 'because', '只算 mat 的 Q/K/V；Q 去看 cache 里 5 个 K'))
b.append(row(y + 88, 'decode 2', ['because'], ['The', 'cat', 'sat', 'on', 'the', 'mat'], 'it', '再算 1 个；cache 变成 6 个'))
b.append(row(y + 132, 'decode 3', ['it'], ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because'], 'was', '……直到 EOS 或长度上限'))
ly = y + 184
b.append(R(x0, ly, 42, 16, '#eaf5ea', GRN, 1)); b.append(T(x0 + 48, ly + 12, '已在 KV cache 里，不再计算', '#333', 10.5))
b.append(R(x0 + 230, ly, 42, 16, '#fff4e0', ORG, 1.5)); b.append(T(x0 + 278, ly + 12, '这一步真正过模型的 token', '#333', 10.5))
b.append(R(x0 + 440, ly, 42, 16, '#f3f0fa', PUR, 1.2, dash='3 2')); b.append(T(x0 + 488, ly + 12, '从最后一个位置的分布抽出的新 token', '#333', 10.5))
b.append(T(60, ly + 44, '为什么旧 token 不用重算：causal attention 里第 i 个 token 的 K、V 只由 token 0..i 决定，', '#333', 11))
b.append(T(60, ly + 62, '新 token 来了它们一个数都不变（L0 第四篇）。每步计算量从「整段 T 个 token」降到「1 个 token」；', '#333', 11))
b.append(T(60, ly + 80, '代价是显存里多一份逐层、逐 token 的 K、V——Llama-3-8B 每个 token 128 KiB（第六篇）。', '#333', 11))
b.append(T(60, ly + 104, '训练 vs 推理：训练一次前向 T 个位置都「新算」且都有目标；推理只有最后一个位置需要分布，', '#666', 10.5))
b.append(T(60, ly + 120, '前面的位置只为了给后面提供 K、V。', '#666', 10.5))
svg(900, ly + 138, ''.join(b), 'transformer-02-prefill-decode-kv-cache.svg')
