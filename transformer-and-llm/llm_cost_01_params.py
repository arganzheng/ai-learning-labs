"""llm_cost.py -- 第一版（Transformer 与 LLM 01）：从超参数算出参数量。
https://arganzheng.life/transformer-anatomy-and-parameter-count.html

用法：
    python llm_cost.py                # 打印内置模型的参数表
    python llm_cost.py config.json    # 读 transformers 风格的 config.json
"""
import json
import sys
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


def param_count(cfg: ModelConfig) -> dict:
    """返回逐组件参数量（单位：个）。键的顺序即打印顺序。"""
    d, L = cfg.hidden, cfg.layers
    d_q = cfg.n_heads * cfg.head_dim          # W_Q 的输出维度，通常等于 d
    d_kv = cfg.n_kv_heads * cfg.head_dim      # W_K / W_V 的输出维度

    w_q = d * d_q
    w_k = d * d_kv
    w_v = d * d_kv
    w_o = d_q * d
    attn = w_q + w_k + w_v + w_o

    ffn = 3 * d * cfg.d_ff                    # gate + up + down
    norms = 2 * d                             # 两个 RMSNorm 的 gamma
    per_layer = attn + ffn + norms

    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    final_norm = d

    total = L * per_layer + embed + lm_head + final_norm
    return {
        "W_Q": w_q, "W_K": w_k, "W_V": w_v, "W_O": w_o,
        "attention/layer": attn,
        "FFN/layer": ffn,
        "norms/layer": norms,
        "per_layer": per_layer,
        "all_layers": L * per_layer,
        "embedding": embed,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "total": total,
    }


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"


def print_table(cfg: ModelConfig) -> None:
    p = param_count(cfg)
    total = p["total"]
    print(f"== {cfg.name}: d={cfg.hidden} L={cfg.layers} "
          f"n_h={cfg.n_heads} n_kv={cfg.n_kv_heads} d_head={cfg.head_dim} "
          f"d_ff={cfg.d_ff} V={cfg.vocab}")
    print(f"{'component':<18}{'params':>14}{'exact':>18}{'share':>9}")
    for k, v in p.items():
        share = "" if k == "total" else f"{100 * v / total:6.2f}%"
        if k in ("W_Q", "W_K", "W_V", "W_O"):
            share = ""  # 单个投影矩阵不算全局占比，避免表格噪音
        print(f"{k:<18}{fmt(v):>14}{v:>18,}{share:>9}")
    print()


def from_config_json(path: str) -> ModelConfig:
    with open(path) as f:
        c = json.load(f)
    n_heads = c["num_attention_heads"]
    return ModelConfig(
        name=path,
        hidden=c["hidden_size"],
        layers=c["num_hidden_layers"],
        n_heads=n_heads,
        n_kv_heads=c.get("num_key_value_heads", n_heads),
        head_dim=c.get("head_dim", c["hidden_size"] // n_heads),
        d_ff=c["intermediate_size"],
        vocab=c["vocab_size"],
        tie_embeddings=c.get("tie_word_embeddings", False),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_table(from_config_json(sys.argv[1]))
    else:
        for cfg in (LLAMA3_8B, LLAMA3_70B):
            print_table(cfg)
