"""llm_cost.py 第三版（Transformer 与 LLM 03）：MHA / GQA / MQA / MLA 的 KV cache 与并发上限。
https://arganzheng.life/attention-variants-and-kv-cache.html
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
    attn_type: str = "gqa"        # "mha" | "gqa" | "mqa" | "mla"
    kv_lora_rank: int = 0         # MLA: d_c
    qk_rope_head_dim: int = 0     # MLA: d_h^R
    n_params: float = 0.0         # 总参数量（含 embedding / lm_head）

LLAMA3_8B = ModelConfig("Llama-3-8B", 4096, 32, 32, 8, 128, 14336, 128256,
                        attn_type="gqa", n_params=8.03e9)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256,
                         attn_type="gqa", n_params=70.55e9)
DEEPSEEK_V3 = ModelConfig("DeepSeek-V3", 7168, 61, 128, 128, 128, 18432, 129280,
                          attn_type="mla", kv_lora_rank=512, qk_rope_head_dim=64,
                          n_params=671e9)

@dataclass
class GPU:
    name: str
    hbm_bytes: float
    bandwidth: float   # bytes/s
    bf16_flops: float  # FLOP/s

H100 = GPU("H100 SXM", 80e9, 3.35e12, 989e12)
A100 = GPU("A100 80GB", 80e9, 2.0e12, 312e12)

KIB, MIB, GIB = 1024, 1024 ** 2, 1024 ** 3


def kv_elems_per_token_per_layer(cfg: ModelConfig) -> int:
    """每层每 token 需要缓存的元素个数（与 dtype 无关）。"""
    if cfg.attn_type == "mha":
        return 2 * cfg.n_heads * cfg.head_dim
    if cfg.attn_type == "gqa":
        return 2 * cfg.n_kv_heads * cfg.head_dim
    if cfg.attn_type == "mqa":
        return 2 * 1 * cfg.head_dim
    if cfg.attn_type == "mla":
        # 只缓存 c_KV（d_c 维）与解耦的 RoPE key k^R（d_h^R 维），所有 head 共享
        return cfg.kv_lora_rank + cfg.qk_rope_head_dim
    raise ValueError(cfg.attn_type)


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    return cfg.layers * kv_elems_per_token_per_layer(cfg) * dtype_bytes


def kv_bytes(cfg: ModelConfig, ctx: int, dtype_bytes: int = 2) -> int:
    return kv_bytes_per_token(cfg, dtype_bytes) * ctx


def decode_attn_intensity(cfg: ModelConfig, dtype_bytes: int = 2) -> float:
    """decode 时 attention 对 KV cache 的算术强度（FLOP/byte）。
    每个 cached token：每个 query head 对 K 做一次点积、对 V 做一次加权和，每维 2 FLOPs。"""
    if cfg.attn_type == "mla":
        k_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim   # 吸收后 K 维 576
        v_dim = cfg.kv_lora_rank                          # 吸收后 V 维 512
        flops = cfg.n_heads * 2 * (k_dim + v_dim)
    else:
        flops = cfg.n_heads * 2 * (cfg.head_dim + cfg.head_dim)
    return flops / (kv_elems_per_token_per_layer(cfg) * dtype_bytes)


def weight_bytes(cfg: ModelConfig, weight_dtype_bytes: int = 2) -> float:
    return cfg.n_params * weight_dtype_bytes


def max_concurrency(cfg: ModelConfig, gpu: GPU, ctx: int, dtype_bytes: int = 2,
                    n_gpus: int = 1, weight_dtype_bytes: int = 2,
                    reserve_frac: float = 0.0) -> int:
    """给定显存预算，能同时驻留多少条上下文为 ctx 的序列。
    只算权重 + KV cache，忽略激活、碎片与框架开销；多卡按权重与 KV 均匀切分估算。"""
    budget = gpu.hbm_bytes * n_gpus * (1 - reserve_frac) - weight_bytes(cfg, weight_dtype_bytes)
    if budget <= 0:
        return 0
    return int(budget // kv_bytes(cfg, ctx, dtype_bytes))


if __name__ == "__main__":
    print(f"{'model':<14}{'attn':<6}{'KV B/token':>12}{'KiB/token':>11}"
          f"{'128K ctx':>11}{'FLOP/byte':>11}")
    for cfg in (LLAMA3_8B, LLAMA3_70B, DEEPSEEK_V3):
        b = kv_bytes_per_token(cfg)
        print(f"{cfg.name:<14}{cfg.attn_type:<6}{b:>12}{b / KIB:>11.1f}"
              f"{kv_bytes(cfg, 131072) / GIB:>9.1f} G{decode_attn_intensity(cfg):>11.0f}")

    mha8b = ModelConfig(**{**LLAMA3_8B.__dict__, "attn_type": "mha"})
    mhav3 = ModelConfig(**{**DEEPSEEK_V3.__dict__, "attn_type": "mha"})
    print(f"\nLlama-3-8B as MHA : {kv_bytes_per_token(mha8b) / KIB:.0f} KiB/token")
    print(f"DeepSeek-V3 as MHA: {kv_bytes_per_token(mhav3) / MIB:.2f} MiB/token, "
          f"ratio = {kv_bytes_per_token(mhav3) / kv_bytes_per_token(DEEPSEEK_V3):.1f}x")

    print("\nmax concurrency on H100 (weights + KV only, BF16 KV):")
    plans = [(LLAMA3_8B, 1, 2), (LLAMA3_70B, 8, 2), (DEEPSEEK_V3, 16, 1)]
    ctxs = [4096, 8192, 32768, 131072]
    print(f"{'model':<14}{'GPUs':>5}{'weights':>9}" + "".join(f"{c // 1024:>7}K" for c in ctxs))
    for cfg, n, wb in plans:
        row = f"{cfg.name:<14}{n:>5}{weight_bytes(cfg, wb) / 1e9:>7.0f}GB"
        for c in ctxs:
            row += f"{max_concurrency(cfg, H100, c, 2, n, wb):>8}"
        print(row)

    print("\nsame, with FP8 KV cache:")
    for cfg, n, wb in plans:
        row = f"{cfg.name:<14}{n:>5}{weight_bytes(cfg, wb) / 1e9:>7.0f}GB"
        for c in ctxs:
            row += f"{max_concurrency(cfg, H100, c, 1, n, wb):>8}"
        print(row)
