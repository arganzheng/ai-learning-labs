"""attention_by_hand.py — 文章 01《Transformer 长什么样》的全部手算数字。

d = 4、T = 3 的一个 causal self-attention 头，从 Q/K/V 投影到加权求和，每一步打印中间矩阵；
最后与 torch.nn.functional.scaled_dot_product_attention 对拍，并做「打乱输入顺序」实验，
说明 attention 本身不知道位置。

运行：python attention_by_hand.py
依赖：torch（CPU 即可）
"""
import math
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=2, sci_mode=False)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. 三个 token 的输入向量 x（d = 4），三个投影矩阵（手写的小整数，便于口算）
# ---------------------------------------------------------------------------
x = torch.tensor([[1., 0., 1., 0.],     # token 0
                  [0., 1., 0., 1.],     # token 1
                  [1., 1., 0., 0.]])    # token 2
W_Q = torch.tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 0.],
                    [0., 1., 0., 1.]])
W_K = torch.tensor([[1., 0., 1., 0.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
W_V = torch.tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])   # V 取恒等：加权求和后能直接看出「混合了谁」

def section(title):
    print(f"\n== {title}")

section("输入 x [T=3, d=4]"); print(x)
Q, K, V = x @ W_Q, x @ W_K, x @ W_V
section("Q = x W_Q"); print(Q)
section("K = x W_K"); print(K)
section("V = x W_V（恒等，V 就是 x）"); print(V)

# ---------------------------------------------------------------------------
# 2. 分数、缩放、mask、softmax、加权求和
# ---------------------------------------------------------------------------
S = Q @ K.T
section("S = Q Kᵀ [T, T]：第 i 行是第 i 个 token 的 query 与每个 key 的内积"); print(S)
S_scaled = S / math.sqrt(4)
section("S / √d（d = 4，除以 2）"); print(S_scaled)
mask = torch.tril(torch.ones(3, 3)).bool()
S_masked = S_scaled.masked_fill(~mask, float("-inf"))
section("causal mask：右上角（看未来）填 −∞"); print(S_masked)
P = F.softmax(S_masked, dim=-1)
section("softmax 逐行 → 注意力权重 P，每行和为 1"); print(P)
out = P @ V
section("out = P V：每个 token 的输出 = 它能看到的 token 的 V 的加权平均"); print(out)

# 手算校验：第 2 行（token 2）看三个 token，权重来自 S_scaled[2] = [1.0, 0.5, 1.0]
e = torch.exp(S_scaled[2]); print("\n手算 token 2 的权重：exp([0.5, 0.5, 1.0]) =", e, "→ 归一化 =", e / e.sum())

# ---------------------------------------------------------------------------
# 3. 与 PyTorch 的实现对拍
# ---------------------------------------------------------------------------
ref = F.scaled_dot_product_attention(Q[None, None], K[None, None], V[None, None], is_causal=True)[0, 0]
section("与 F.scaled_dot_product_attention(is_causal=True) 对拍"); print("最大差:", (out - ref).abs().max().item())

# ---------------------------------------------------------------------------
# 4. attention 不知道顺序：打乱输入行，输出只是同样被打乱（不加 mask 时）
# ---------------------------------------------------------------------------
def attn_no_mask(x_):
    q, k, v = x_ @ W_Q, x_ @ W_K, x_ @ W_V
    return F.softmax(q @ k.T / 2, dim=-1) @ v

perm = torch.tensor([2, 0, 1])
o1, o2 = attn_no_mask(x), attn_no_mask(x[perm])
section("打乱输入顺序（不加 mask）：输出集合不变，只是跟着换了位置")
print("原顺序输出:\n", o1); print("打乱后输出:\n", o2)
print("o2 == o1[perm]:", torch.allclose(o2, o1[perm]))

# ---------------------------------------------------------------------------
# 5. 为什么除 √d：d 越大，点积越大，softmax 越尖
# ---------------------------------------------------------------------------
section("为什么除 √d：随机 q、k 的点积标准差随 d 增长")
for d in (4, 64, 128, 1024):
    q = torch.randn(10000, d); k = torch.randn(10000, d)
    dots = (q * k).sum(-1)
    print(f"d={d:5d}  点积标准差 {dots.std():7.2f}  ≈ √d = {math.sqrt(d):6.2f}  除 √d 后 {(dots / math.sqrt(d)).std():.2f}")
