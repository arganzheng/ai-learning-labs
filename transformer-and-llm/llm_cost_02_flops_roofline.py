"""llm_cost.py 第二版（Transformer 与 LLM 02）：FLOPs、字节数与 Roofline 时间下界。
https://arganzheng.life/transformer-flops-bytes-and-roofline.html
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


# ---- 第一篇：参数量 ------------------------------------------------------

def param_count(cfg):
    """第一篇的逐组件参数量（重给以便独立运行），返回 dict。"""
    d, d_kv = cfg.hidden, cfg.n_kv_heads * cfg.head_dim
    attn = d * d + 2 * d * d_kv + d * d          # W_Q, W_K, W_V, W_O
    ffn = 3 * d * cfg.d_ff                         # gate, up, down
    norms = 2 * d
    per_layer = attn + ffn + norms
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.layers * per_layer + d + embed + lm_head   # + 最终 RMSNorm
    return {"per_layer": per_layer, "all_layers": cfg.layers * per_layer,
            "embedding": embed, "lm_head": lm_head, "total": total}


def gemm_params(cfg):
    """参与 GEMM 的参数量：embedding 只做查表，不算；lm_head 永远要算一次。"""
    p = param_count(cfg)
    return cfg.layers * (p["per_layer"] - 2 * cfg.hidden) + cfg.vocab * cfg.hidden


# ---- 第二篇：算量、字节数、Roofline --------------------------------------

def forward_flops_per_token(cfg, ctx):
    """每 token 前向 FLOPs = 权重项 2N_gemm + 上下文项 4·d·ctx·L。"""
    weight_flops = 2 * gemm_params(cfg)
    attn_flops = 4 * cfg.hidden * ctx * cfg.layers
    return weight_flops, attn_flops


def weight_bytes(cfg, dtype_bytes=2):
    return param_count(cfg)["total"] * dtype_bytes


def kv_bytes_per_token(cfg, dtype_bytes=2):
    """K 和 V 各一份：2 · L · n_kv · d_head · bytes/elem。第三篇扩展到 MLA。"""
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes


def decode_step_time(cfg, gpu, batch, ctx, dtype_bytes=2):
    """decode 一步的理论下界：max(访存时间, 算力时间)。返回各分项便于打印。"""
    w_flops, a_flops = forward_flops_per_token(cfg, ctx)
    flops = batch * (w_flops + a_flops)
    w_bytes = weight_bytes(cfg, dtype_bytes)
    kv_bytes = batch * ctx * kv_bytes_per_token(cfg, dtype_bytes)
    total_bytes = w_bytes + kv_bytes
    t_mem = total_bytes / gpu.bandwidth
    t_cmp = flops / gpu.bf16_flops
    return {
        "flops": flops,
        "weight_bytes": w_bytes,
        "kv_bytes": kv_bytes,
        "intensity": flops / total_bytes,
        "t_mem": t_mem,
        "t_cmp": t_cmp,
        "t": max(t_mem, t_cmp),
        "bound": "memory" if t_mem >= t_cmp else "compute",
    }


def prefill_time(cfg, gpu, seq, mfu=0.6, causal=True):
    """prefill 的理论时间：总 FLOPs / (峰值算力 × MFU)。因果掩码可让上下文项减半。"""
    w_flops, a_flops = forward_flops_per_token(cfg, seq)
    if causal:
        a_flops /= 2
    total = seq * (w_flops + a_flops)
    return total, total / (gpu.bf16_flops * mfu)


def ridge_point(gpu):
    return gpu.bf16_flops / gpu.bandwidth


def fmt(x, unit=""):
    for div, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f} {suffix}{unit}"
    return f"{x:.2f} {unit}"


def report(cfg, gpu, ctx=8192, batches=(1, 8, 64, 295)):
    total, gemm = param_count(cfg)["total"], gemm_params(cfg)
    w_flops, a_flops = forward_flops_per_token(cfg, ctx)
    print(f"== {cfg.name} on {gpu.name} (ridge = {ridge_point(gpu):.0f} FLOP/byte) ==")
    print(f"params total {total / 1e9:.2f} B, gemm {gemm / 1e9:.2f} B")
    print(f"weight FLOPs/token   {fmt(w_flops, 'FLOPs')}")
    print(f"attn FLOPs/token@{ctx} {fmt(a_flops, 'FLOPs')}")
    print(f"weight bytes (BF16)  {fmt(weight_bytes(cfg), 'B')}")
    print(f"KV bytes/token       {kv_bytes_per_token(cfg) / 1024:.0f} KiB")
    print()
    print(f"{'batch':>6} {'ctx':>6} {'KV read':>10} {'I(FLOP/B)':>10} "
          f"{'t_mem(ms)':>10} {'t_cmp(ms)':>10} {'step(ms)':>9} {'tok/s':>8} bound")
    for b in batches:
        r = decode_step_time(cfg, gpu, b, ctx)
        print(f"{b:>6} {ctx:>6} {r['kv_bytes'] / 2**30:>8.1f}Gi "
              f"{r['intensity']:>10.1f} {r['t_mem'] * 1e3:>10.2f} "
              f"{r['t_cmp'] * 1e3:>10.2f} {r['t'] * 1e3:>9.2f} "
              f"{b / r['t']:>8.0f} {r['bound']}")
    print()
    for seq in (8192, 131072):
        for causal in (False, True):
            fl, t = prefill_time(cfg, gpu, seq, mfu=0.6, causal=causal)
            print(f"prefill {seq:>6} causal={str(causal):5}  {fmt(fl, 'FLOP'):>14}  "
                  f"@60% MFU {t:.3f} s")
    print()


if __name__ == "__main__":
    report(LLAMA3_8B, H100)
    report(LLAMA3_70B, H100, batches=(1, 8, 64))
