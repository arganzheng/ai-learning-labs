"""llm_cost.py 第四版（Transformer 与 LLM 04）：上下文长度扫描 —— KV cache、prefill 与 attention 占比。
https://arganzheng.life/positional-encoding-and-long-context.html
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    hidden: int
    layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tie_embeddings: bool = False
    mla_rank: Optional[int] = None   # MLA 的 d_c；None 表示 MHA/GQA
    rope_dim: Optional[int] = None   # MLA 解耦 RoPE 的 d_h^R

LLAMA3_8B = ModelConfig("Llama-3-8B", 4096, 32, 32, 8, 128, 14336, 128256)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)

@dataclass
class GPU:
    name: str
    hbm_bytes: float
    bandwidth: float   # bytes/s
    bf16_flops: float  # FLOP/s

H100 = GPU("H100 SXM", 80e9, 3.35e12, 989e12)
A100 = GPU("A100 80GB", 80e9, 2.0e12, 312e12)

GiB = 2 ** 30

def param_count(cfg: ModelConfig) -> dict:
    """dense 模型参数量（第一篇的公式，重给以便独立运行）。"""
    d, dh = cfg.hidden, cfg.head_dim
    attn = d * cfg.n_heads * dh + 2 * d * cfg.n_kv_heads * dh + cfg.n_heads * dh * d
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    emb = cfg.vocab * d
    head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.layers * per_layer + d + emb + head
    return {"per_layer": per_layer, "embedding": emb, "lm_head": head, "total": total}

def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    """每 token 的 KV cache 字节数（第三篇），支持 MLA。"""
    if cfg.mla_rank is not None:
        return cfg.layers * (cfg.mla_rank + cfg.rope_dim) * dtype_bytes
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

def weight_flops_per_token(cfg: ModelConfig) -> float:
    """权重 GEMM 部分：每参数每 token 2 FLOPs，embedding 查表不计。"""
    p = param_count(cfg)
    return 2.0 * (p["total"] - p["embedding"])

def attn_flops_per_token(cfg: ModelConfig, ctx: int) -> float:
    """对上下文 ctx 做一次 attention（QK^T 与 PV）：每层 4·d·ctx。"""
    return 4.0 * cfg.n_heads * cfg.head_dim * ctx * cfg.layers

def forward_flops_per_token(cfg: ModelConfig, ctx: int) -> float:
    """decode 一个 token、上下文 ctx 时的前向 FLOPs（第二篇）。"""
    return weight_flops_per_token(cfg) + attn_flops_per_token(cfg, ctx)

def prefill_flops(cfg: ModelConfig, ctx: int, causal: bool = True) -> tuple:
    """一次 ctx 长度 prefill 的总 FLOPs，返回 (权重项, attention 项)。"""
    w = weight_flops_per_token(cfg) * ctx
    a = attn_flops_per_token(cfg, ctx) * ctx
    if causal:
        a /= 2
    return w, a

def context_scan(cfg: ModelConfig, gpu: GPU, ctxs, mfu: float = 0.6) -> None:
    """上下文长度 → KV cache、prefill FLOPs、attention 占比。"""
    kv = kv_bytes_per_token(cfg)
    w_tok = weight_flops_per_token(cfg)
    print(f"{cfg.name}: KV {kv/1024:.1f} KiB/token, weights {w_tok/1e9:.1f} GFLOPs/token")
    print(f"{'ctx':>8} {'KV cache':>10} {'prefill':>12} {'prefill@'+str(int(mfu*100))+'%':>12} "
          f"{'attn/token':>12} {'attn share':>10}")
    for s in ctxs:
        w, a = prefill_flops(cfg, s)
        t = (w + a) / (gpu.bf16_flops * mfu)
        a_tok = attn_flops_per_token(cfg, s)
        share = a_tok / (w_tok + a_tok)
        print(f"{s:>8d} {kv*s/GiB:>8.1f} G {(w+a)/1e15:>9.2f} PF {t:>10.2f} s "
              f"{a_tok/1e9:>9.1f} GF {share*100:>9.1f}%")

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B):
        context_scan(cfg, H100, [8192, 32768, 131072])
        print()
