"""llm_cost.py 第七版（Transformer 与 LLM 07）：量化字节数、投机解码加速比、LoRA 参数量。
https://arganzheng.life/quantization-speculative-decoding-and-lora.html
"""
from dataclasses import dataclass

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

# ---- 前几篇的函数（dense 版本）----
def embedding_params(cfg):
    return cfg.vocab * cfg.hidden * (1 if cfg.tie_embeddings else 2)

def param_count(cfg):
    """第一篇的参数量（重给以便独立运行），返回 dict。"""
    d = cfg.hidden
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    attn = d * q + d * kv + d * kv + q * d
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = per_layer * cfg.layers + embed + lm_head + d
    return {"per_layer": per_layer, "embedding": embed, "lm_head": lm_head, "total": total}

def forward_flops_per_token(cfg, ctx=0):
    # embedding 查表不算 GEMM；lm_head 算
    # 输入 embedding 是查表不算 GEMM；tied 模型那张表兼作 lm_head，仍要算
    gemm_params = param_count(cfg)["total"] - (0 if cfg.tie_embeddings else cfg.vocab * cfg.hidden)
    return 2 * gemm_params + 4 * cfg.hidden * ctx * cfg.layers

def kv_bytes_per_token(cfg, dtype_bytes=2):
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

# ---- 第七篇新增 ----
def quantized_weight_bytes(cfg, bits=4, group_size=128, scale_bits=16,
                           zero_bits=16, keep_embed_bf16=False):
    """weight-only 量化后的权重字节数；返回 (bytes, 等效 bit/权重)。"""
    eff_bits = bits + (scale_bits + zero_bits) / group_size
    n = param_count(cfg)["total"]
    if keep_embed_bf16:
        e = embedding_params(cfg)
        return (n - e) * eff_bits / 8 + e * 2, eff_bits
    return n * eff_bits / 8, eff_bits

def roofline_step_time(cfg, gpu, rows, weight_bytes, mfu=1.0):
    """一次前向处理 rows 个 token 行的时间下界 max(访存, 计算)，只计权重部分。"""
    t_mem = weight_bytes / gpu.bandwidth
    t_cmp = forward_flops_per_token(cfg) * rows / (gpu.bf16_flops * mfu)
    return max(t_mem, t_cmp)

def speculative_speedup(alpha, gamma, c, batch, cfg, gpu, mfu=1.0):
    """投机解码相对普通 decode 的加速比；返回 (speedup, E[tokens])。"""
    assert 0.0 <= alpha <= 1.0
    # alpha == 1 时几何级数的闭式是 0/0，极限是 gamma + 1（全部接受 + bonus）
    exp_tokens = gamma + 1 if alpha == 1.0 else (1 - alpha ** (gamma + 1)) / (1 - alpha)
    w = param_count(cfg)["total"] * 2              # BF16 目标模型
    t_base = roofline_step_time(cfg, gpu, batch, w, mfu)
    t_verify = roofline_step_time(cfg, gpu, batch * (gamma + 1), w, mfu)
    t_round = gamma * c * t_base + t_verify
    return exp_tokens * t_base / t_round, exp_tokens

LORA_TARGETS_ATTN = ("q", "k", "v", "o")
LORA_TARGETS_ALL = ("q", "k", "v", "o", "gate", "up", "down")

def lora_params(cfg, rank=16, targets=LORA_TARGETS_ALL):
    d = cfg.hidden
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    shapes = {"q": (d, q), "k": (d, kv), "v": (d, kv), "o": (q, d),
              "gate": (d, cfg.d_ff), "up": (d, cfg.d_ff), "down": (cfg.d_ff, d)}
    per_layer = sum(rank * (din + dout)
                    for t in targets for din, dout in [shapes[t]])
    return per_layer * cfg.layers

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B):
        n = param_count(cfg)["total"]
        b4, eb = quantized_weight_bytes(cfg)
        print(f"{cfg.name}: {n/1e9:.2f}B  BF16 {n*2/1e9:.1f} GB  "
              f"INT4(g128) {eb:.2f} bit -> {b4/1e9:.2f} GB")
        print(f"  decode 下界 BF16 {n*2/H100.bandwidth*1e3:.2f} ms  "
              f"W4A16 {b4/H100.bandwidth*1e3:.2f} ms")
        print(f"  LoRA r=16 attn {lora_params(cfg, 16, LORA_TARGETS_ATTN)/1e6:.2f}M  "
              f"all {lora_params(cfg, 16)/1e6:.2f}M ({lora_params(cfg, 16)/n*100:.2f}%)")
    print("投机解码 alpha=0.8 gamma=4 c=0.1 (Llama-3-8B, H100):")
    for B in (1, 8, 32, 64, 128, 256):
        s, e = speculative_speedup(0.8, 4, 0.1, B, LLAMA3_8B, H100)
        s60, _ = speculative_speedup(0.8, 4, 0.1, B, LLAMA3_8B, H100, mfu=0.6)
        print(f"  B={B:4d}  peak {s:.2f}  60%MFU {s60:.2f}  E[tokens]={e:.2f}")
