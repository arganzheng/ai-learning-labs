"""llm_cost.py 第八版（Transformer 与 LLM 08）：vision encoder 参数与 FLOPs、image token 数、image token 在 decoder 里的成本。ModelConfig / param_count / kv_bytes_per_token 沿用第七版。
https://arganzheng.life/multimodal-vision-encoder-cost-and-image-token-kv.html
"""
from dataclasses import dataclass
from math import ceil
from llm_cost_07_quant_specdec_lora import LLAMA3_8B, LLAMA3_70B, param_count, kv_bytes_per_token

@dataclass
class VisionConfig:
    name: str
    layers: int
    hidden: int
    patch: int = 14
    mlp_ratio: int = 4
    merge: int = 1          # 空间合并的边长：Qwen2-VL / InternVL 为 2，LLaVA 为 1
    tile: int = 0           # 固定 tile 边长（px）；0 表示原生动态分辨率
    max_tiles: int = 1
    cls_token: int = 0
    tile_attn: bool = False   # True: 各 tile 独立过 encoder（InternVL）；False: 多 tile 拼成一个序列做 attention（Llama-3.2）

CLIP_L_336 = VisionConfig("CLIP ViT-L/14-336", 24, 1024, tile=336, cls_token=1)
INTERN_VIT_300M = VisionConfig("InternViT-300M", 24, 1024, merge=2, tile=448, max_tiles=13, tile_attn=True)
QWEN2_VL_VIT = VisionConfig("Qwen2-VL ViT", 32, 1280, merge=2)
LLAMA32_VIT = VisionConfig("Llama-3.2 ViT-H/14", 40, 1280, tile=560, max_tiles=4, cls_token=1)

def vit_params(v):
    return v.layers * 12 * v.hidden ** 2 + 3 * v.patch ** 2 * v.hidden

def patches_per_tile(v):
    return (v.tile // v.patch) ** 2 + v.cls_token

def image_patches(v, h, w, tiles=1):
    """一张 h×w 图片进入 encoder 的 patch 数。"""
    if v.tile:
        return tiles * patches_per_tile(v)
    f = v.patch * v.merge
    return ceil(h / f) * ceil(w / f) * v.merge ** 2

def image_tokens(v, h, w, tiles=1):
    """connector 之后进入 decoder 的 token 数（tile 方案里 CLS 不进 decoder）。"""
    if v.tile:
        return tiles * ((v.tile // v.patch) ** 2 // v.merge ** 2)
    return image_patches(v, h, w) // v.merge ** 2

def vit_flops(v, n_patches, window=0, full_layers=0, tiles=1):
    """一张图的 encoder FLOPs；window>0 时按窗口 attention 计，full_layers 层做全图 attention。
    tiles>1 且 v.tile_attn 时各 tile 独立做 attention（二次项按每 tile 求和，而不是全部 patch 平方）。"""
    weight = 2 * vit_params(v) * n_patches
    if window:
        attn = 4 * n_patches * window * v.hidden * (v.layers - full_layers) \
             + 4 * n_patches ** 2 * v.hidden * full_layers
    elif tiles > 1 and v.tile_attn:
        per_tile = n_patches // tiles
        attn = tiles * 4 * per_tile ** 2 * v.hidden * v.layers
    else:
        attn = 4 * n_patches ** 2 * v.hidden * v.layers
    return weight, attn

def image_cost_in_decoder(cfg, n_img, dtype_bytes=2):
    """image token 在 decoder 里的三个数：prefill FLOPs、KV 字节、encoder 输出字节。"""
    # 输入 embedding 是查表不算 GEMM；tied 模型那张表兼作 lm_head，仍要算
    gemm_params = param_count(cfg)["total"] - (0 if cfg.tie_embeddings else cfg.vocab * cfg.hidden)
    return {
        "prefill_flops": 2 * gemm_params * n_img,
        "kv_bytes": n_img * kv_bytes_per_token(cfg, dtype_bytes),
        "encoder_out_bytes": n_img * cfg.hidden * dtype_bytes,
    }

if __name__ == "__main__":
    MiB = 2 ** 20
    for v, (h, w, tiles) in [(CLIP_L_336, (336, 336, 1)), (QWEN2_VL_VIT, (1024, 1024, 1)),
                             (INTERN_VIT_300M, (1024, 1024, 5)), (LLAMA32_VIT, (1024, 1024, 4))]:
        n_p, n_t = image_patches(v, h, w, tiles), image_tokens(v, h, w, tiles)
        wf, af = vit_flops(v, n_p, tiles=tiles)
        print(f"{v.name:22s} patches {n_p:5d} tokens {n_t:5d} "
              f"encoder {(wf + af) / 1e12:5.2f} TFLOP (attn {af / (wf + af):.0%})")
        for cfg in (LLAMA3_8B, LLAMA3_70B):
            c = image_cost_in_decoder(cfg, n_t)
            print(f"    {cfg.name:12s} prefill {c['prefill_flops'] / 1e12:6.1f} TFLOP  "
                  f"KV {c['kv_bytes'] / MiB:7.1f} MiB  enc-out {c['encoder_out_bytes'] / MiB:5.1f} MiB")
