"""gen_token_journey_svg.py — 文章 02 的六张图：训练侧一步（teacher forcing → T 个 loss → 反向 → 更新）、
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

# ---------------- 3. overview: two dynamic lines ----------------
def box(x, y, w, h, t1, t2, fill, stroke):
    return R(x, y, w, h, fill, stroke, 1.3, 6) + T(x + w / 2, y + 19, t1, '#222', 12, 'middle', '600') + T(x + w / 2, y + 37, t2, '#555', 10.5, 'middle')
b = [T(460, 24, '同一台模型的两条动态线：训练是「前向 → loss → 反向 → 更新」的循环，推理是「一次 prefill + 逐个 decode」', '#222', 13.5, 'middle', '600')]
lx, rx, bw, bh, gap = 40, 490, 390, 46, 22
b.append(R(lx - 16, 44, bw + 32, 420, '#fbfcfe', '#c9d6e6', 1, 8)); b.append(R(rx - 16, 44, bw + 32, 420, '#fbfdfb', '#cfe3d2', 1, 8))
b.append(T(lx + bw / 2, 66, '训练（参数在变）', BLUE, 13, 'middle', '600')); b.append(T(rx + bw / 2, 66, '推理（参数冻结）', GRN, 13, 'middle', '600'))
tr = [('① 取一个 batch', '[B, T+1] 个连续 token → 输入 x 与右移一位的目标 y', '#eef4fb', BLUE),
      ('② 前向', '[B, T] → [B, T, d] → … → logits [B, T, V]，沿路保存激活值', '#eef4fb', BLUE),
      ('③ loss', 'B × T 个位置各一个交叉熵，取平均成一个标量', '#fdeeee', RED),
      ('④ 反向', '沿原路走回去，每个参数得到同形状的 .grad，激活值用完释放', '#fdeeee', RED),
      ('⑤ AdamW 更新', '每个参数沿梯度反方向挪一小步；zero_grad', '#fff4e0', ORG)]
for i, (a, c, f, s) in enumerate(tr):
    y = 82 + i * (bh + gap)
    b.append(box(lx, y, bw, bh, a, c, f, s))
    if i < 4: b.append(L(lx + bw / 2, y + bh, lx + bw / 2, y + bh + gap - 2, '#666', 1.4))
yl = 82 + 4 * (bh + gap) + bh
b.append(f'<path d="M{lx},{yl - bh / 2} L{lx - 10},{yl - bh / 2} L{lx - 10},{82 + bh / 2} L{lx - 2},{82 + bh / 2}" fill="none" stroke="{BLUE}" stroke-width="1.5" marker-end="url(#ar)"/>')
b.append(T(lx + bw / 2, yl + 26, '换下一个 batch 回到 ①，几十万步之后就是一个语言模型', GRAY, 10.5, 'middle'))
inf = [('① 用户的 prompt', '[1, T] 个 token，一次性全部已知', '#eaf5ea', GRN),
       ('② prefill', 'T 个位置一次前向（同训练的前向），每层的 K、V 写进 cache', '#eaf5ea', GRN),
       ('③ 采样', '只取最后一个位置的分布，按温度 / top-k 抽出 1 个新 token', '#f3f0fa', PUR),
       ('④ decode', '只把这 1 个 token 喂进去 [1, 1]，读 cache 里全部 K、V，追加自己的', '#fff4e0', ORG)]
for i, (a, c, f, s) in enumerate(inf):
    y = 82 + i * (bh + gap)
    b.append(box(rx, y, bw, bh, a, c, f, s))
    if i < 3: b.append(L(rx + bw / 2, y + bh, rx + bw / 2, y + bh + gap - 2, '#666', 1.4))
y3, y4 = 82 + 2 * (bh + gap), 82 + 3 * (bh + gap)
b.append(f'<path d="M{rx + bw},{y4 + bh / 2} L{rx + bw + 10},{y4 + bh / 2} L{rx + bw + 10},{y3 + bh / 2} L{rx + bw + 2},{y3 + bh / 2}" fill="none" stroke="{ORG}" stroke-width="1.5" marker-end="url(#ar)"/>')
b.append(T(rx + bw / 2, y4 + bh + 26, '③ ↔ ④ 循环，每圈吐出 1 个 token，直到 EOS 或长度上限', GRAY, 10.5, 'middle'))
b.append(T(rx + bw / 2, y4 + bh + 44, '第一个字等 prefill（TTFT），之后每个字等一次 decode（TPOT）', GRAY, 10.5, 'middle'))
b.append(T(lx + bw / 2, 446, '显存大头：参数 + 梯度 + 优化器状态 + 激活值', '#333', 11, 'middle', '600'))
b.append(T(rx + bw / 2, 446, '显存大头：参数 + KV cache', '#333', 11, 'middle', '600'))
svg(920, 480, ''.join(b), 'transformer-02-two-dynamic-lines.svg')

# ---------------- 4. shapes along the forward pass ----------------
# 形状取自 token_journey.py 第 1 段的实际输出（expected/token_journey.txt）
st = [('输入 idx', (2, 5), None, '#ddd', '每个 token 一个整数编号'),
      ('wte 查表', (2, 5, 8), 8, BLUE, '每个 token 一个 8 维向量'),
      ('+ wpe 查表', (5, 8), 8, BLUE, '5 个位置各一个向量，广播加到每句上'),
      ('block0.attn.c_attn', (2, 5, 24), 24, PUR, 'Q、K、V 一次算出（3d）再切开'),
      ('block0.attn 输出', (2, 5, 8), 8, PUR, '加回残差流'),
      ('block0.mlp.c_fc', (2, 5, 32), 32, ORG, 'FFN 放大到 4d'),
      ('block0.mlp 输出', (2, 5, 8), 8, ORG, '压回 d，加回残差流'),
      ('block0 输出', (2, 5, 8), 8, GRN, '残差流'),
      ('block1（同上）', (2, 5, 8), 8, GRN, '24 → 8 → 32 → 8 再走一遍'),
      ('ln_f', (2, 5, 8), 8, GRN, '最后一次 LayerNorm'),
      ('lm_head → logits', (2, 5, 16), 16, RED, '每个位置 V = 16 个分数'),
      ('loss', (), None, RED, '标量 2.7805 ≈ ln 16')]
b = [T(440, 24, '训练前向每一站的输出形状（B = 2、T = 5、d = 8、V = 16）：条宽 ∝ 最后一维', '#222', 13.5, 'middle', '600')]
x0, y0, rh, u = 190, 50, 30, 12
b.append(L(x0 + 8 * u, y0 - 6, x0 + 8 * u, y0 + rh * len(st) - 6, '#bbb', 1, False, '4 3'))
b.append(T(x0 + 8 * u, y0 + rh * len(st) + 8, 'd = 8', GRAY, 10, 'middle'))
for i, (name, shape, w, col, note) in enumerate(st):
    y = y0 + i * rh
    b.append(T(x0 - 10, y + 13, name, '#333', 11, 'end'))
    if w is None:
        b.append(R(x0, y, 5 * u if shape else u, 18, '#f4f4f4', col, 1.2, 3))
    else:
        b.append(R(x0, y, w * u, 18, col + '22', col, 1.2, 3))
    s = '(' + ', '.join(map(str, shape)) + ')' if shape else '()'
    b.append(T(x0 + 400, y + 13, s, '#222', 11, 'start', '600'))
    b.append(T(x0 + 480, y + 13, note, GRAY, 10.5))
b.append(T(60, y0 + rh * len(st) + 34, '残差流的宽度从 wte 到 ln_f 一直是 d：只有 QKV（3d）和 FFN（4d）两处临时变宽又缩回；lm_head 之后每个位置都有一个词表分布。', '#333', 11))
svg(900, y0 + rh * len(st) + 52, ''.join(b), 'transformer-02-shape-flow.svg')

# ---------------- 5. forward saves activations, backward consumes them ----------------
mods = ['wte + wpe', 'block0', 'block1', 'ln_f', 'lm_head', 'loss']
acts = ['token 向量', 'LN 输入、Q/K/V、\nattention 权重、4d 中间量', 'LN 输入、Q/K/V、\nattention 权重、4d 中间量', 'LN 输入', 'ln_f 输出', 'logits、softmax']
b = [T(450, 24, '前向沿路把激活值存下来，反向从 loss 出发沿原路走回去、边用边释放，每个参数得到 .grad', '#222', 13.5, 'middle', '600')]
x0, cw, y = 30, 145, 60
for i, m in enumerate(mods):
    x = x0 + i * cw
    b.append(R(x, y, cw - 30, 40, '#eef4fb' if m != 'loss' else '#fdeeee', BLUE if m != 'loss' else RED, 1.3, 6)); b.append(T(x + (cw - 30) / 2, y + 25, m, '#222', 12, 'middle', '600'))
    if i < 5: b.append(L(x + cw - 30, y + 20, x + cw - 2, y + 20, BLUE, 1.5))
    if m != 'loss':
        b.append(R(x, y + 64, cw - 30, 48, '#fff4e0', ORG, 1, 4, '4 2'))
        for j, ln in enumerate(acts[i].split('\n')): b.append(T(x + (cw - 30) / 2, y + 82 + j * 14, ln, '#333', 9.5, 'middle'))
        b.append(L(x + (cw - 30) / 2, y + 40, x + (cw - 30) / 2, y + 62, ORG, 1, True, '3 2'))
        b.append(R(x, y + 150, cw - 30, 30, '#fdeeee', RED, 1, 4)); b.append(T(x + (cw - 30) / 2, y + 169, '.grad', RED, 11, 'middle', '600'))
        b.append(L(x + (cw - 30) / 2, y + 112, x + (cw - 30) / 2, y + 148, RED, 1, True))
lc = x0 + 5 * cw + (cw - 30) / 2
b.append(f'<path d="M{lc},{y + 40} L{lc},{y + 204} L{x0 + (cw - 30) / 2},{y + 204} L{x0 + (cw - 30) / 2},{y + 184}" fill="none" stroke="{RED}" stroke-width="1.5" marker-end="url(#ar)"/>')
b.append(T(lc + 6, y + 130, '反向起点', RED, 10.5))
b.append(T(x0 + 2.5 * cw, y + 220, '反向：loss → lm_head → ln_f → block1 → block0 → wte / wpe', RED, 10.5, 'middle'))
b.append(T(30, y + 244, '橙框（激活值）：前向时存下、反向用到才释放；它的大小 ∝ B × T，参数量与 B × T 无关——所以上下文越长，激活值越可能比参数本身还大。', '#333', 11))
b.append(T(30, y + 262, '红线（反向）：每一站用自己存下的激活值算出本站参数的 .grad，算完这一站的激活值就可以释放。', '#333', 11))
svg(900, y + 280, ''.join(b), 'transformer-02-forward-backward-activations.svg')

# ---------------- 6. prefill / decode on a time axis ----------------
b = [T(450, 24, '一次请求的时间线（示意，不按比例）：prefill 一大块算完 prompt，之后每个 decode 只算 1 个 token', '#222', 13.5, 'middle', '600')]
y, x0 = 90, 60
b.append(L(x0, y + 60, 860, y + 60, '#555', 1.2))
b.append(T(862, y + 64, '时间', GRAY, 10.5))
b.append(R(x0, y, 260, 44, '#eaf5ea', GRN, 1.4, 4)); b.append(T(x0 + 130, y + 19, 'prefill：T 个 prompt token 一次前向', '#222', 11.5, 'middle', '600')); b.append(T(x0 + 130, y + 35, '计算密集：一次读权重、算 T 个位置', '#555', 10, 'middle'))
outs = ['mat', 'because', 'it', 'was', 'tired', '…']
x = x0 + 266
for i, t in enumerate(outs):
    b.append(R(x, y, 70, 44, '#fff4e0', ORG, 1.2, 4)); b.append(T(x + 35, y + 19, f'decode {i + 1}' if t != '…' else '…', '#222', 11, 'middle', '600')); b.append(T(x + 35, y + 35, '1 个 token', '#555', 10, 'middle'))
    b.append(T(x - 3, y - 12, t, PUR, 11, 'middle', '600')); b.append(L(x - 3, y - 8, x - 3, y + 2, PUR, 1))
    x += 76
b.append(T(x0 + 130 + 130, y - 30, '每段结束吐出一个 token（紫色）', PUR, 10.5, 'middle'))
b.append(L(x0, y + 82, x0 + 263, y + 82, RED, 1.3)); b.append(L(x0 + 263, y + 82, x0, y + 82, RED, 1.3))
b.append(T(x0 + 131, y + 100, 'TTFT：用户等第一个字的时间', RED, 11, 'middle', '600'))
b.append(L(x0 + 266, y + 82, x0 + 339, y + 82, BLUE, 1.3)); b.append(L(x0 + 339, y + 82, x0 + 266, y + 82, BLUE, 1.3))
b.append(T(x0 + 303, y + 100, 'TPOT', BLUE, 11, 'middle', '600'))
b.append(T(x0 + 420, y + 100, '每个字之间的间隔：访存密集，每步都要把全部权重读一遍', BLUE, 10.5))
b.append(T(x0, y + 132, '「第一个字慢、后面快」：prefill 要把整个 prompt 算完才能出第一个字；之后每步只算 1 个 token，但每步都要读一遍全部参数。', '#333', 11))
svg(900, y + 150, ''.join(b), 'transformer-02-prefill-decode-timeline.svg')
