"""llm_cost.py 第六版（Transformer 与 LLM 06）：dtype 字节表与训练状态显存。
https://arganzheng.life/floating-point-formats-and-mixed-precision.html
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
    param_override: int = 0      # MoE/MLA 模型直接给总参数量 (第五篇的 param_count 可算出)

LLAMA3_8B  = ModelConfig("Llama-3-8B",  4096, 32, 32, 8, 128, 14336, 128256)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)
DEEPSEEK_V3 = ModelConfig("DeepSeek-V3", 7168, 61, 128, 128, 192, 18432, 129280,
                          param_override=671_000_000_000)

def param_count(cfg: ModelConfig) -> dict:
    """第一篇的 dense 参数量（重给以便独立运行）；MoE 模型用 param_override 直接给总量。"""
    d, dh = cfg.hidden, cfg.head_dim
    attn = d * cfg.n_heads * dh * 2 + d * cfg.n_kv_heads * dh * 2     # W_Q, W_O, W_K, W_V
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.param_override or (per_layer * cfg.layers + embed + lm_head + d)
    return {"per_layer": per_layer, "embedding": embed, "lm_head": lm_head, "total": total}

# ---- 第六篇新增 ----
DTYPE_BYTES = {
    "fp32": 4, "tf32": 4,          # TF32 在内存中仍是 32 位
    "fp16": 2, "bf16": 2,
    "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1, "int4": 0.5,        # int4 不含 scale/zero-point 开销 (第七篇)
}

def training_state_bytes(cfg: ModelConfig, optimizer: str = "adam", mixed: bool = True,
                         weight_dtype: str = "bf16", grad_dtype: str = "bf16",
                         optim_dtype: str = "fp32") -> dict:
    """每参数训练状态字节数与总量. mixed=True: 低精度权重副本 + 低精度梯度 + FP32 master;
    mixed=False: 全 FP32 (权重即 master, 无副本)."""
    n = param_count(cfg)["total"]
    if mixed:
        per = {"weight_copy": DTYPE_BYTES[weight_dtype],
               "grad": DTYPE_BYTES[grad_dtype],
               "master": DTYPE_BYTES["fp32"]}
    else:
        per = {"weight": DTYPE_BYTES["fp32"], "grad": DTYPE_BYTES["fp32"]}
    n_moments = {"sgd": 0, "momentum": 1, "adam": 2, "adamw": 2}[optimizer]
    for i in range(n_moments):
        per[f"moment{i+1}"] = DTYPE_BYTES[optim_dtype]
    per_param = sum(per.values())
    return {"params": n, "per_param_bytes": per_param, "breakdown": per,
            "total_bytes": n * per_param}

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B, DEEPSEEK_V3):
        r = training_state_bytes(cfg)
        print(f"{cfg.name:12s} params={r['params']/1e9:7.2f}B  "
              f"{r['per_param_bytes']:>2} B/param  state={r['total_bytes']/1e9:9.1f} GB  "
              f"H100(80GB) >= {r['total_bytes']/80e9:6.1f} 张")
    # DeepSeek-V3 报告的变体: FP8 权重副本, BF16 梯度, FP32 master, BF16 m/v
    r = training_state_bytes(DEEPSEEK_V3, weight_dtype="fp8_e4m3", optim_dtype="bf16")
    print(f"{'V3-fp8-recipe':12s} {r['per_param_bytes']} B/param  state={r['total_bytes']/1e12:.2f} TB  {r['breakdown']}")
    # 全 FP32 对照: 同样 16 B/param
    r = training_state_bytes(LLAMA3_8B, mixed=False)
    print(f"{'8B-fp32':12s} {r['per_param_bytes']} B/param  state={r['total_bytes']/1e9:.1f} GB")
