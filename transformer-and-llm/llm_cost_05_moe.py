"""llm_cost.py 第五版（Transformer 与 LLM 05）：MoE 的总参数 / 激活参数、期望激活专家数、EP all-to-all 字节数。
https://arganzheng.life/moe-compute-and-communication.html
"""
from dataclasses import dataclass
from math import comb


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
    # 第三篇加入：MLA（mla_rank=0 表示 GQA/MHA）
    mla_rank: int = 0        # d_c
    rope_dim: int = 0        # d_h^R
    q_lora_rank: int = 0
    # 第五篇加入：MoE（n_experts=0 表示 dense）
    n_experts: int = 0       # 路由专家数 E
    top_k: int = 0           # 每 token 激活的路由专家数 k
    expert_d_ff: int = 0     # 每个专家的 d_ff
    n_shared: int = 0        # 共享专家数（永远激活）
    moe_layers: int = 0      # MoE 层数，其余 layers - moe_layers 层为 dense
    dense_d_ff: int = 0      # dense 层的 d_ff（0 表示与 d_ff 相同）


LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)

MIXTRAL_8X7B = ModelConfig(
    "Mixtral-8x7B", 4096, 32, 32, 8, 128, 14336, 32000,
    n_experts=8, top_k=2, expert_d_ff=14336, n_shared=0, moe_layers=32,
)

DEEPSEEK_V3 = ModelConfig(
    "DeepSeek-V3", 7168, 61, 128, 128, 192, 18432, 129280,
    mla_rank=512, rope_dim=64, q_lora_rank=1536,
    n_experts=256, top_k=8, expert_d_ff=2048, n_shared=1,
    moe_layers=58, dense_d_ff=18432,
)


def attn_params(cfg):
    """每层 attention 的权重参数量（GQA/MHA 或 MLA）。"""
    d, h = cfg.hidden, cfg.n_heads
    if cfg.mla_rank:
        nope = cfg.head_dim - cfg.rope_dim                 # 128
        w_dq = d * cfg.q_lora_rank                         # 7168 x 1536
        w_uq = cfg.q_lora_rank * h * cfg.head_dim          # 1536 x 24576
        w_dkv = d * (cfg.mla_rank + cfg.rope_dim)          # 7168 x 576
        w_uk = cfg.mla_rank * h * nope                     # 512 x 16384
        w_uv = cfg.mla_rank * h * nope                     # 512 x 16384
        w_o = h * nope * d                                 # 16384 x 7168
        return w_dq + w_uq + w_dkv + w_uk + w_uv + w_o
    q_o = 2 * d * h * cfg.head_dim
    k_v = 2 * d * cfg.n_kv_heads * cfg.head_dim
    return q_o + k_v


def ffn_params(d, d_ff):
    """SwiGLU FFN：gate、up、down 三个矩阵。"""
    return 3 * d * d_ff


def moe_param_count(cfg):
    """返回 (总参数量, 分项字典)。n_experts=0 时退化为 dense 模型。"""
    d = cfg.hidden
    n_dense = cfg.layers - cfg.moe_layers
    dense_ff = cfg.dense_d_ff or cfg.d_ff
    per_expert = ffn_params(d, cfg.expert_d_ff) if cfg.n_experts else 0
    parts = {
        "attention": cfg.layers * attn_params(cfg),
        "dense_ffn": n_dense * ffn_params(d, dense_ff),
        "router": cfg.moe_layers * d * cfg.n_experts,
        "routed_experts": cfg.moe_layers * cfg.n_experts * per_expert,
        "shared_experts": cfg.moe_layers * cfg.n_shared * per_expert,
        "norms": (2 * cfg.layers + 1) * d,
        "embedding": cfg.vocab * d * (1 if cfg.tie_embeddings else 2),
    }
    return sum(parts.values()), parts


def active_params(cfg):
    """每 token 激活的参数量：总参数减去未被选中的路由专家。"""
    total, parts = moe_param_count(cfg)
    if not cfg.n_experts:
        return total
    per_expert = ffn_params(cfg.hidden, cfg.expert_d_ff)
    return total - parts["routed_experts"] + cfg.moe_layers * cfg.top_k * per_expert


def expected_active_experts(cfg, batch):
    """batch 个 token 独立、均匀路由时，一层里期望被至少一个 token 选中的专家数。"""
    E, k = cfg.n_experts, cfg.top_k
    if not E:
        return 0.0
    return E * (1 - (1 - k / E) ** batch)


def params_read_per_step(cfg, batch):
    """一步 decode 期望从 HBM 读取的参数量（假设全模型在一张卡上，仅用于比较）。"""
    total, parts = moe_param_count(cfg)
    if not cfg.n_experts:
        return total
    per_expert = ffn_params(cfg.hidden, cfg.expert_d_ff)
    read_experts = cfg.moe_layers * expected_active_experts(cfg, batch) * per_expert
    return total - parts["routed_experts"] + read_experts


def ep_all_to_all_bytes_per_layer(cfg, batch, dispatch_bytes=1, combine_bytes=2,
                                  ep_size=None):
    """一个 MoE 层 dispatch + combine 的 all-to-all 字节数（batch 个 token 合计）。
    dispatch_bytes / combine_bytes 是每个元素的字节数（DeepSeek-V3：FP8 / BF16）。
    给出 ep_size 时，按均匀假设扣掉落在本卡的 1/ep_size。"""
    per_token = cfg.top_k * cfg.hidden * (dispatch_bytes + combine_bytes)
    total = batch * per_token
    if ep_size:
        total *= (ep_size - 1) / ep_size
    return total


def routing_combinations(cfg):
    return comb(cfg.n_experts, cfg.top_k) if cfg.n_experts else 1


if __name__ == "__main__":
    KiB = 1024
    for cfg in (MIXTRAL_8X7B, DEEPSEEK_V3, LLAMA3_70B):
        total, parts = moe_param_count(cfg)
        print(f"== {cfg.name}")
        for key, val in parts.items():
            print(f"  {key:16s} {val/1e9:8.3f} B")
        print(f"  total            {total/1e9:8.2f} B")
        print(f"  active           {active_params(cfg)/1e9:8.2f} B")
        print(f"  flops/token      {2*active_params(cfg)/1e9:8.1f} GFLOPs")
        if cfg.n_experts:
            # Mixtral 的 dispatch/combine 都按 BF16 算；DeepSeek-V3 按 FP8/BF16
            db = 1 if cfg.mla_rank else 2
            print(f"  C(E,k)           {routing_combinations(cfg):.3e}")
            for B in (1, 8, 32, 128, 512):
                n_act = expected_active_experts(cfg, B)
                read = params_read_per_step(cfg, B)
                a2a = ep_all_to_all_bytes_per_layer(cfg, B, dispatch_bytes=db) / KiB
                print(f"  B={B:4d}  active experts {n_act:7.1f}  read {read/1e9:7.1f} B"
                      f"  a2a/layer {a2a:9.0f} KiB")
