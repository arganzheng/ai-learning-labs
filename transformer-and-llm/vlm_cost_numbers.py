"""多模态成本的一组理论数字（Transformer 与 LLM 08）：vision encoder 参数、tokens per image、encoder FLOPs、image token KV、cross-attention KV、Whisper、视频。BF16，H100 SXM 峰值。
https://arganzheng.life/multimodal-vision-encoder-cost-and-image-token-kv.html
"""
from math import ceil, floor

H100_FLOPS = 989e12
H100_BW = 3.35e12
MiB = 2**20; GiB = 2**30

def vit_params(L, d, mlp_ratio=4, patch=14, in_ch=3):
    per_layer = 4*d*d + 2*mlp_ratio*d*d   # qkvo + up/down (no bias, no norm)
    return L*per_layer, in_ch*patch*patch*d   # layers, patch-embed

def vit_flops(L, d, n, mlp_ratio=4):
    """FLOPs for one image with n patches (bidirectional attention)."""
    layers, _ = vit_params(L, d, mlp_ratio)
    weight = 2*layers*n
    attn = 4*n*n*d*L        # 2 GEMMs (QK^T, PV) each 2*n*n*d per layer
    return weight, attn

def kv_per_token(L, n_kv, d_head, bytes_=2):
    return 2*L*n_kv*d_head*bytes_

print("== vision encoders ==")
for name, L, d, ratio in [("CLIP ViT-L/14-336 (LLaVA-1.5)", 24, 1024, 4),
                          ("Qwen2-VL ViT", 32, 1280, 4),
                          ("InternViT-300M", 24, 1024, 4),
                          ("Llama-3.2 Vision ViT-H/14 (32 local + 8 global)", 40, 1280, 4)]:
    lp, pe = vit_params(L, d, ratio)
    print(f"{name:50s} layers {lp/1e6:6.0f} M  patch-embed {pe/1e6:.2f} M")

print("\n== tokens per image ==")
print("LLaVA-1.5  336/14 =", 336//14, "->", (336//14)**2, "tokens")
print("InternVL2  448/14 =", 448//14, "->", (448//14)**2, "patches -> /4 =", (448//14)**2//4, "tokens; 12 tiles+thumb:", 13*256)
print("Llama-3.2  560/14 =", 560//14, "->", (560//14)**2, "+1 cls per tile; 4 tiles:", 4*((560//14)**2+1))
def qwen_tokens(h, w, patch=14, merge=2):
    f = patch*merge
    hb, wb = round(h/f), round(w/f)
    return hb*wb, hb*wb*merge*merge
for hw in [(336,336),(448,448),(896,896),(1024,1024),(1920,1080)]:
    t, p = qwen_tokens(*hw)
    print(f"Qwen2-VL {hw[0]}x{hw[1]} -> {p} patches -> {t} tokens")

print("\n== encoder FLOPs (one image) ==")
for name, L, d, n in [("CLIP-L 576p", 24, 1024, 577), ("Qwen2-VL 1024^2 (5476 patches, full attn)", 32, 1280, 5476),
                      ("Qwen2.5-VL 1024^2 window 8x8 (28 layers) + 4 full", 32, 1280, 5476),
                      ("Llama-3.2 4 tiles x 1601", 40, 1280, 6404)]:
    w, a = vit_flops(L, d, n)
    if "window" in name:
        # 28 window layers: attention over 64-patch windows; 4 full layers
        a = 4*n*64*d*28 + 4*n*n*d*4
    print(f"{name:52s} weight {w/1e12:6.2f} TFLOP  attn {a/1e12:6.2f} TFLOP  total {(w+a)/1e12:6.2f}  ~{(w+a)/H100_FLOPS*1e3:5.1f} ms@H100 peak")

print("\n== image tokens in the decoder ==")
dec = {"Llama-3-8B": (32, 8, 128, 4096, 8.03e9), "Llama-3-70B": (80, 8, 128, 8192, 70.6e9),
       "Qwen2-VL-7B LLM": (28, 4, 128, 3584, 7.6e9), "Vicuna-7B (LLaVA-1.5)": (32, 32, 128, 4096, 6.7e9)}
for m, (L, nkv, dh, d, N) in dec.items():
    kv = kv_per_token(L, nkv, dh)
    print(f"{m:24s} KV/token {kv/1024:6.0f} KiB  embed/token {2*d/1024:4.0f} KiB")
    for n in [576, 1337, 3328, 6404]:
        print(f"    {n:5d} tokens: KV {n*kv/MiB:7.1f} MiB  encoder-out {n*2*d/MiB:5.1f} MiB  prefill {2*N*n/1e12:5.2f} TFLOP")

print("\n== cross-attention KV (Llama-3.2-11B) ==")
# 8 cross-attn layers, 8 kv heads x 128, over 6404 image tokens
n = 6404
xkv = 2*8*8*128*2*n
print(f"cross-attn KV for 4 tiles: {xkv/MiB:.1f} MiB (fixed, independent of text length)")
print(f"decoder-only equivalent (40-layer self-attn KV): {n*kv_per_token(40,8,128)/MiB:.1f} MiB")

print("\n== Whisper large-v3 encoder ==")
# 30 s, 16 kHz, hop 160 -> 3000 frames; conv stride 2 -> 1500; 32 layers d=1280
lp, _ = vit_params(32, 1280)
w, a = vit_flops(32, 1280, 1500)
print(f"encoder params {lp/1e6:.0f} M, 1500 positions, FLOPs {(w+a)/1e12:.2f} TFLOP, output {1500*1280*2/MiB:.1f} MiB")

print("\n== video: Qwen2-VL 720p 1 fps 60 s ==")
t, p = qwen_tokens(720, 1280)
frames = 60; temporal = 2
print(f"per-frame-pair tokens {t}, 60 frames/2 = {frames//temporal} pairs -> {t*frames//temporal} tokens")
