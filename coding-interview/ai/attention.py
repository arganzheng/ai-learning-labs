"""面试手撕代码（14）：手撕 attention 家族 —— NumPy 实现 + torch 对拍。

https://arganzheng.life/coding-interview-attention-from-scratch.html

    python attention.py            # 打印形状推演与数值示例
    python attention.py --check    # 与 torch 参考实现对拍（需要 torch）
"""
from __future__ import annotations

import argparse
import math

import numpy as np


# ---------- 1. softmax ----------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定：先减去该行最大值。exp(x - max) 不会上溢，且结果与不减完全相同。"""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log softmax = x - logsumexp(x)。直接 log(softmax) 在极小概率处会得到 -inf。"""
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


# ---------- 2. scaled dot-product attention ----------

def causal_mask(T: int) -> np.ndarray:
    """下三角为 True（可见）。"""
    return np.tril(np.ones((T, T), dtype=bool))


def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """q: (..., T_q, d), k/v: (..., T_k, d). 返回 (..., T_q, d_v)。
    mask: (T_q, T_k) 的 bool，True 表示可见。"""
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(d)          # (..., T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)                 # 不可见位置 -inf → softmax 后为 0
    return softmax(scores) @ v


# ---------- 3. multi-head attention ----------

def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(B, T, D) -> (B, H, T, D/H)：先 reshape 拆最后一维，再把 H 换到前面。"""
    B, T, D = x.shape
    return x.reshape(B, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B, H, T, d) -> (B, T, H*d)：split 的逆操作。"""
    B, H, T, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * d)


def mha(x: np.ndarray, w_qkv: np.ndarray, w_o: np.ndarray, n_heads: int, causal: bool = True) -> np.ndarray:
    """x: (B, T, D)；w_qkv: (D, 3D)；w_o: (D, D)。无 bias 的 GPT 风格 MHA。"""
    B, T, D = x.shape
    qkv = x @ w_qkv                                              # (B, T, 3D)
    q, k, v = np.split(qkv, 3, axis=-1)
    q, k, v = split_heads(q, n_heads), split_heads(k, n_heads), split_heads(v, n_heads)   # (B, H, T, d)
    mask = causal_mask(T) if causal else None
    out = sdpa(q, k, v, mask)                                    # (B, H, T, d)
    return merge_heads(out) @ w_o                                # (B, T, D)


# ---------- 4. GQA：多个 query 头共享一个 kv 头 ----------

def gqa(x: np.ndarray, w_q: np.ndarray, w_kv: np.ndarray, w_o: np.ndarray, n_heads: int, n_kv_heads: int) -> np.ndarray:
    """w_q: (D, D)；w_kv: (D, 2 * n_kv_heads * d)。每 n_heads / n_kv_heads 个 q 头共享一个 kv 头。"""
    B, T, D = x.shape
    d = D // n_heads
    q = split_heads(x @ w_q, n_heads)                            # (B, H, T, d)
    kv = x @ w_kv                                                # (B, T, 2 * Hkv * d)
    k, v = np.split(kv, 2, axis=-1)
    k, v = split_heads(k, n_kv_heads), split_heads(v, n_kv_heads)   # (B, Hkv, T, d)
    rep = n_heads // n_kv_heads
    k = np.repeat(k, rep, axis=1)                                # (B, H, T, d)：把每个 kv 头复制 rep 份
    v = np.repeat(v, rep, axis=1)
    out = sdpa(q, k, v, causal_mask(T))
    return merge_heads(out) @ w_o


# ---------- 5. RoPE ----------

def rope_cos_sin(T: int, d: int, base: float = 10000.0):
    """返回 (T, d/2) 的 cos、sin。第 i 对维度的频率 base^(-2i/d)。"""
    inv_freq = base ** (-np.arange(0, d, 2) / d)                 # (d/2,)
    angles = np.arange(T)[:, None] * inv_freq[None, :]           # (T, d/2)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: (..., T, d)。把每对 (x[2i], x[2i+1]) 旋转 angle[t, i]。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ---------- 6. KV cache 增量解码 ----------

class KVCache:
    """每层一份：k、v 各 (B, H, T_max, d)，len 记录已缓存的 token 数。"""

    def __init__(self, B: int, H: int, T_max: int, d: int):
        self.k = np.zeros((B, H, T_max, d))
        self.v = np.zeros((B, H, T_max, d))
        self.len = 0

    def append(self, k_new: np.ndarray, v_new: np.ndarray):
        t = k_new.shape[2]
        self.k[:, :, self.len:self.len + t] = k_new
        self.v[:, :, self.len:self.len + t] = v_new
        self.len += t
        return self.k[:, :, :self.len], self.v[:, :, :self.len]


def decode_step(x_new: np.ndarray, w_qkv: np.ndarray, w_o: np.ndarray, n_heads: int, cache: KVCache) -> np.ndarray:
    """x_new: (B, 1, D) 新 token。只算它的 q/k/v，k/v 追加进 cache，q 对全部历史 k/v 做 attention。"""
    qkv = x_new @ w_qkv
    q, k, v = (split_heads(t, n_heads) for t in np.split(qkv, 3, axis=-1))   # (B, H, 1, d)
    k_all, v_all = cache.append(k, v)                            # (B, H, len, d)
    out = sdpa(q, k_all, v_all)                                  # 新 token 能看到全部历史，无需 mask
    return merge_heads(out) @ w_o                                # (B, 1, D)


# ---------- 7. online softmax（FlashAttention 一趟分块的核心） ----------

def attention_online(q: np.ndarray, k: np.ndarray, v: np.ndarray, block: int) -> np.ndarray:
    """单个 query 行 q: (d,)；k, v: (T, d)。按 block 分块扫一遍 k/v，只维护 (m, l, acc)，不保存整行 scores。"""
    d = q.shape[-1]
    m = -np.inf                                                  # 迄今最大分数
    l = 0.0                                                      # 迄今 exp 之和（相对 m）
    acc = np.zeros(v.shape[-1])                                  # 迄今加权和（相对 m）
    for s in range(0, k.shape[0], block):
        scores = k[s:s + block] @ q / math.sqrt(d)               # (block,)
        m_new = max(m, scores.max())
        scale = math.exp(m - m_new) if m > -np.inf else 0.0      # 旧累计值按新 max 重新缩放
        p = np.exp(scores - m_new)
        l = l * scale + p.sum()
        acc = acc * scale + p @ v[s:s + block]
        m = m_new
    return acc / l


# ---------- demo & check ----------

def demo():
    rng = np.random.default_rng(0)
    B, T, D, H = 2, 5, 8, 2
    x = rng.standard_normal((B, T, D))
    w_qkv = rng.standard_normal((D, 3 * D)) / math.sqrt(D)
    w_o = rng.standard_normal((D, D)) / math.sqrt(D)

    print("== 形状推演（B=2, T=5, D=8, H=2, d=4）")
    qkv = x @ w_qkv
    print(f"x {x.shape} @ w_qkv {w_qkv.shape} -> qkv {qkv.shape}")
    q = split_heads(np.split(qkv, 3, axis=-1)[0], H)
    print(f"split_heads: (B,T,D) -> reshape (B,T,H,d) -> transpose -> {q.shape}")
    scores = q @ np.swapaxes(q, -1, -2)
    print(f"scores = q @ k^T: {scores.shape}；mask {causal_mask(T).shape}")
    out = mha(x, w_qkv, w_o, H)
    print(f"merge_heads @ w_o -> {out.shape}")

    print("\n== softmax 数值稳定性")
    big = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    print(f"naive exp(1000..1002)/sum -> {naive}（上溢成 nan）")
    print(f"stable                     -> {softmax(big)}")

    print("\n== causal mask 的效果（第 0 行只看得到自己）")
    print(np.round(softmax(np.where(causal_mask(3), np.zeros((3, 3)), -np.inf)), 3))

    print("\n== KV cache 增量解码 == 一次性全序列 attention")
    full = mha(x, w_qkv, w_o, H, causal=True)
    cache = KVCache(B, H, T, D // H)
    steps = [decode_step(x[:, t:t + 1], w_qkv, w_o, H, cache) for t in range(T)]
    inc = np.concatenate(steps, axis=1)
    print(f"max |full - incremental| = {np.abs(full - inc).max():.2e}")

    print("\n== online softmax（block=2）== 普通 attention")
    qq, kk, vv = rng.standard_normal(4), rng.standard_normal((7, 4)), rng.standard_normal((7, 4))
    ref = sdpa(qq[None], kk, vv)[0]
    print(f"max diff = {np.abs(attention_online(qq, kk, vv, block=2) - ref).max():.2e}")

    print("\n== RoPE：旋转后内积只依赖相对位置")
    d = 8
    cos, sin = rope_cos_sin(10, d)
    a, b = rng.standard_normal(d), rng.standard_normal(d)
    dots = []
    for t in (0, 3, 6):
        qa = apply_rope(a[None], cos[t:t + 1], sin[t:t + 1])[0]
        kb = apply_rope(b[None], cos[t + 2:t + 3], sin[t + 2:t + 3])[0]
        dots.append(qa @ kb)
    print(f"q 在位置 t、k 在 t+2，t=0,3,6 的内积：{np.round(dots, 6)}（三者相等）")


def check():
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(1)
    B, T, D, H = 2, 6, 16, 4
    x = rng.standard_normal((B, T, D)).astype(np.float64)
    w_qkv = (rng.standard_normal((D, 3 * D)) / math.sqrt(D)).astype(np.float64)
    w_o = (rng.standard_normal((D, D)) / math.sqrt(D)).astype(np.float64)

    # softmax
    s = rng.standard_normal((3, 5))
    assert np.allclose(softmax(s), torch.softmax(torch.tensor(s), -1).numpy(), atol=1e-12)
    assert np.allclose(log_softmax(s), torch.log_softmax(torch.tensor(s), -1).numpy(), atol=1e-12)

    # sdpa vs F.scaled_dot_product_attention
    q = torch.tensor(split_heads(x @ w_qkv[:, :D], H))
    k = torch.tensor(split_heads(x @ w_qkv[:, D:2 * D], H))
    v = torch.tensor(split_heads(x @ w_qkv[:, 2 * D:], H))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).numpy()
    mine = sdpa(q.numpy(), k.numpy(), v.numpy(), causal_mask(T))
    assert np.allclose(mine, ref, atol=1e-10), np.abs(mine - ref).max()

    # mha vs nn.MultiheadAttention（batch_first，无 bias）
    mha_t = torch.nn.MultiheadAttention(D, H, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha_t.in_proj_weight.copy_(torch.tensor(w_qkv.T))       # torch 存的是 (3D, D)，y = x W^T
        mha_t.out_proj.weight.copy_(torch.tensor(w_o.T))
        xt = torch.tensor(x)
        ref, _ = mha_t(xt, xt, xt, attn_mask=~torch.tensor(causal_mask(T)), need_weights=False)
    mine = mha(x, w_qkv, w_o, H, causal=True)
    assert np.allclose(mine, ref.numpy(), atol=1e-10), np.abs(mine - ref.numpy()).max()

    # GQA vs 把 kv 头 repeat 后的 sdpa（enable_gqa）
    Hkv = 2
    d = D // H
    w_q = w_qkv[:, :D]
    w_kv = (rng.standard_normal((D, 2 * Hkv * d)) / math.sqrt(D)).astype(np.float64)
    kv = x @ w_kv
    k2 = torch.tensor(split_heads(kv[..., :Hkv * d], Hkv))
    v2 = torch.tensor(split_heads(kv[..., Hkv * d:], Hkv))
    ref = F.scaled_dot_product_attention(torch.tensor(split_heads(x @ w_q, H)), k2, v2, is_causal=True, enable_gqa=True)
    ref = merge_heads(ref.numpy()) @ w_o
    assert np.allclose(gqa(x, w_q, w_kv, w_o, H, Hkv), ref, atol=1e-10)

    # KV cache 增量 == 全序列
    full = mha(x, w_qkv, w_o, H)
    cache = KVCache(B, H, T, D // H)
    inc = np.concatenate([decode_step(x[:, t:t + 1], w_qkv, w_o, H, cache) for t in range(T)], axis=1)
    assert np.allclose(full, inc, atol=1e-10)

    # online softmax == 普通
    qq, kk, vv = rng.standard_normal(d), rng.standard_normal((11, d)), rng.standard_normal((11, d))
    assert np.allclose(attention_online(qq, kk, vv, block=3), sdpa(qq[None], kk, vv)[0], atol=1e-12)

    # RoPE：相对位置性质
    cos, sin = rope_cos_sin(20, d)
    a, b = rng.standard_normal(d), rng.standard_normal(d)
    dots = [apply_rope(a[None], cos[t:t + 1], sin[t:t + 1])[0] @ apply_rope(b[None], cos[t + 5:t + 6], sin[t + 5:t + 6])[0] for t in (0, 4, 9)]
    assert np.allclose(dots, dots[0], atol=1e-10)
    print("attention.py: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="与 torch 对拍")
    args = ap.parse_args()
    check() if args.check else demo()
