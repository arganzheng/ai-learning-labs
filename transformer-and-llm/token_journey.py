"""token_journey.py — 文章 02《一个 token 的旅程：训练侧与推理侧》的全部数字。

一个自带 KV cache 的极小 GPT（结构与 nanoGPT 一致），做六件事：
  1. 训练侧前向：打印每一站的形状（B=2, T=5, d=8, V=16）
  2. 一次前向算 T 个 loss：teacher forcing，逐位置 loss
  3. 反向：哪些参数拿到梯度、为反向保存了多少激活
  4. 优化器走一步：同一 batch 的 loss 下降
  5. 推理侧：prefill → decode；有 / 无 KV cache 的输出逐 token 一致
  6. 计时：T=512、d=64、4 层，无 cache 每步重算 vs 有 cache 只算一个 token

运行：python token_journey.py    依赖：torch（CPU）
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

# ---------------------------------------------------------------------------
# 模型：与 nanoGPT 同构，多一个可选的 KV cache
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(d, 3 * d)          # 一次算出 Q、K、V
        self.c_proj = nn.Linear(d, d)               # W_O
        self.n_head, self.d = n_head, d
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, cache=None):
        B, T, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=2)
        q = q.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)   # [B, h, T, d_h]
        k = k.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        if cache is not None:                       # decode：把旧 K、V 接在前面，新 token 只算自己的
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], dim=2); v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        S = k.size(2)                               # key 的个数（有 cache 时 = 历史长度 + T）
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))          # [B, h, T, S]
        att = att.masked_fill(self.mask[:, :, S - T:S, :S] == 0, float("-inf"))   # 最后 T 行的 causal mask
        y = F.softmax(att, dim=-1) @ v              # [B, h, T, d_h]
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, d))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__(); self.c_fc = nn.Linear(d, 4 * d); self.c_proj = nn.Linear(4 * d, d)
    def forward(self, x): return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, d, n_head, block_size):
        super().__init__(); self.ln_1 = nn.LayerNorm(d); self.attn = CausalSelfAttention(d, n_head, block_size); self.ln_2 = nn.LayerNorm(d); self.mlp = MLP(d)
    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab, d); self.wpe = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False); self.lm_head.weight = self.wte.weight   # tie
        self.block_size = block_size
        for m in self.modules():                       # nanoGPT 的初始化：正态 std 0.02，否则随机 logits 太大、loss 远超 ln V
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, caches=None, pos_offset=0):
        B, T = idx.shape
        pos = torch.arange(pos_offset, pos_offset + T)
        x = self.wte(idx) + self.wpe(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        logits = self.lm_head(self.ln_f(x))           # [B, T, V]
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))   # targets 是切片、不连续，view 会报错（Infra PyTorch 02）
        return logits, loss

def section(t): print(f"\n== {t}")

# ---------------------------------------------------------------------------
# 1. 训练侧前向：每一站的形状
# ---------------------------------------------------------------------------
V, d, L, H, CTX = 16, 8, 2, 2, 8
model = TinyGPT(V, d, L, H, CTX)
B, T = 2, 5
idx = torch.randint(0, V, (B, T + 1))                 # 取 T+1 个 token：前 T 个是输入，后 T 个是目标
x, y = idx[:, :-1], idx[:, 1:]
section(f"1. 训练侧前向的形状：B={B} 句、T={T} 个 token、d={d}、V={V}、{L} 层 {H} 头")
shapes = []
def hook(name):
    def f(m, inp, out): shapes.append((name, tuple(out.shape) if torch.is_tensor(out) else '-'))
    return f
hs = [model.wte.register_forward_hook(hook("wte 查表")), model.wpe.register_forward_hook(hook("wpe 查表"))]
for i, blk in enumerate(model.blocks):
    hs += [blk.attn.c_attn.register_forward_hook(hook(f"block{i}.attn.c_attn（QKV 合在一起）")),
           blk.attn.register_forward_hook(hook(f"block{i}.attn 输出")),
           blk.mlp.c_fc.register_forward_hook(hook(f"block{i}.mlp.c_fc（放大 4 倍）")),
           blk.mlp.register_forward_hook(hook(f"block{i}.mlp 输出")),
           blk.register_forward_hook(hook(f"block{i} 输出（残差流）"))]
hs += [model.ln_f.register_forward_hook(hook("ln_f")), model.lm_head.register_forward_hook(hook("lm_head → logits"))]
logits, loss = model(x, y)
for h in hs: h.remove()
print(f"输入 idx: {tuple(x.shape)}  目标 targets: {tuple(y.shape)}")
for n, s in shapes: print(f"  {n:38s} {s}")
print(f"loss（标量）: {loss.item():.4f}   ≈ ln(16) = {math.log(16):.4f}（随机初始化时每个位置在 16 个 token 里乱猜）")

# ---------------------------------------------------------------------------
# 2. 一次前向算 T 个 loss
# ---------------------------------------------------------------------------
section("2. teacher forcing：T 个位置同时各预测自己的下一个 token")
print("第 0 句的输入 :", x[0].tolist()); print("第 0 句的目标 :", y[0].tolist(), "（就是输入右移一位）")
per_pos = F.cross_entropy(logits[0], y[0], reduction="none")
for t in range(T):
    print(f"  位置 {t}: 看到 {x[0, :t+1].tolist()} → 预测第 {t+1} 个 token，正确答案 {y[0, t].item():2d}，"
          f"模型给它的概率 {F.softmax(logits[0, t], -1)[y[0, t]].item():.3f}，loss {per_pos[t].item():.3f}")
print(f"T 个 loss 的平均 = {per_pos.mean().item():.4f}（就是上面的 loss；两句话一起平均）")

# ---------------------------------------------------------------------------
# 3. 反向：谁拿到梯度、为反向保存了什么
# ---------------------------------------------------------------------------
section("3. 反向传播：每个参数都拿到一份和自己同形状的梯度")
loss.backward()
n_params = sum(p.numel() for p in model.parameters())
for name, p in list(model.named_parameters())[:6]:
    print(f"  {name:28s} 参数 {str(tuple(p.shape)):12s} 梯度范数 {p.grad.norm().item():.4f}")
print(f"  ……共 {sum(1 for _ in model.parameters())} 个参数张量、{n_params:,} 个数，每个都有 .grad")
print(f"  为了算这些梯度，前向时每层要存下：LayerNorm 的输入、Q/K/V、attention 权重 [B,h,T,T]、FFN 的 4d 维中间量……")
print(f"  这就是「激活值」显存：与 B × T 成正比，训练长上下文时它比参数本身还大（第十篇算账）")

# ---------------------------------------------------------------------------
# 4. 优化器走一步
# ---------------------------------------------------------------------------
section("4. AdamW 走一步：同一个 batch 的 loss 下降")
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
before = loss.item(); opt.step(); opt.zero_grad(set_to_none=True)
_, after = model(x, y)
print(f"  更新前 {before:.4f} → 更新后 {after.item():.4f}（同一 batch；真实训练每步换一个新 batch）")

# ---------------------------------------------------------------------------
# 5. 推理侧：prefill → decode，有 / 无 KV cache 一致
# ---------------------------------------------------------------------------
section("5. 推理：prefill 一次算完 prompt，decode 每步只算一个新 token")
model.eval()
prompt = torch.tensor([[3, 7, 1]])                    # 3 个 token 的 prompt
with torch.no_grad():
    # (a) 无 cache：每生成一个 token 就把整段重算一遍
    seq_a = prompt.clone()
    for step in range(5):
        logits, _ = model(seq_a)                       # 重算全部 [1, 当前长度]
        nxt = logits[:, -1].argmax(-1, keepdim=True)   # 只用最后一个位置的分布（贪心）
        seq_a = torch.cat([seq_a, nxt], 1)
    # (b) 有 cache：prefill 算 prompt 并存 K、V；之后每步只喂 1 个 token
    caches = [{"k": None, "v": None} for _ in range(L)]
    logits, _ = model(prompt, caches=caches, pos_offset=0)           # prefill：[1, 3]
    print(f"  prefill：输入 {tuple(prompt.shape)}，每层 cache 里 K 的形状 {tuple(caches[0]['k'].shape)}  [B, h, 已有 token 数, d_h]")
    seq_b = prompt.clone(); nxt = logits[:, -1].argmax(-1, keepdim=True)
    for step in range(5):
        seq_b = torch.cat([seq_b, nxt], 1)
        logits, _ = model(nxt, caches=caches, pos_offset=seq_b.size(1) - 1)   # decode：只算这 1 个 token
        print(f"  decode 第 {step+1} 步：输入形状 {tuple(nxt.shape)}，cache 里 K 变成 {tuple(caches[0]['k'].shape)}，新 token {nxt.item()}")
        nxt = logits[:, -1].argmax(-1, keepdim=True)
print(f"  无 cache 生成: {seq_a[0].tolist()}\n  有 cache 生成: {seq_b[0].tolist()}\n  两者一致: {torch.equal(seq_a, seq_b)}")

# ---------------------------------------------------------------------------
# 6. 计时：为什么 KV cache 让 decode 从 O(T²) 变成 O(T)
# ---------------------------------------------------------------------------
section("6. 计时：T=512、d=64、4 层 4 头，生成 256 个 token（CPU）")
big = TinyGPT(vocab=256, d=64, n_layer=4, n_head=4, block_size=1024).eval()
prompt = torch.randint(0, 256, (1, 256))
with torch.no_grad():
    t0 = time.perf_counter(); seq = prompt.clone()
    for _ in range(256):
        lg, _ = big(seq); seq = torch.cat([seq, lg[:, -1].argmax(-1, keepdim=True)], 1)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter(); caches = [{"k": None, "v": None} for _ in range(4)]
    lg, _ = big(prompt, caches=caches); nxt = lg[:, -1].argmax(-1, keepdim=True); n = prompt.size(1)
    for _ in range(256):
        lg, _ = big(nxt, caches=caches, pos_offset=n); n += 1; nxt = lg[:, -1].argmax(-1, keepdim=True)
    t_cache = time.perf_counter() - t0
kv_bytes = 4 * 2 * 512 * 64 * 4   # 4 层 × (K+V) × 512 token × d=64 × fp32 4 B
print(f"  无 cache（每步重算整段，长度 256→512）: {t_full:.2f} s")
print(f"  有 cache（每步只算 1 个 token）       : {t_cache:.2f} s   快 {t_full / t_cache:.1f} 倍")
print(f"  代价：cache 里存着 4 层 × K、V × 512 token × 64 维 × 4 B = {kv_bytes / 1024:.0f} KiB；"
      f"Llama-3-8B 每个 token 的 KV 是 128 KiB（第六篇）")
