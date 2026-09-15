"""diffusion_ledger.py -- 扩散模型推理基础设施 01：一次生成的三本账。
https://arganzheng.life/diffusion-inference-workload-anatomy-and-cost-ledger.html

一张图 / 一段视频从 prompt 到像素要经过三段：文本编码器一次前向、去噪网络
(DiT) 乘步数乘 CFG 分支、VAE 解码一次。本脚本对给定模型、分辨率、帧数、步数、
硬件算出每段的 FLOPs、显存与时间，并与一次 LLM 推理对照。纯标准库，全部是
纸面估算（标称峰值 × 给定 MFU），用于数量级判断，不是实测。

用法：
    python diffusion_ledger.py                          # 五个内置模型的默认账 + 放大器扫描
    python diffusion_ledger.py --model flux --steps 4 --no-cfg
    python diffusion_ledger.py --model wan --height 480 --width 832 --frames 81 --gpu h100 --mfu 0.45
    python diffusion_ledger.py --model flux --gpu 4090 --sweep
    python diffusion_ledger.py -h
"""
import argparse
import math
from dataclasses import dataclass, replace

# ---------------------------------------------------------------- 模型


@dataclass
class DiTSpec:
    name: str
    params: float            # 去噪网络参数量（个）——决定权重字节数
    params_per_token: float  # 一个 token 真正经过的参数量——决定 FLOPs（去掉 adaLN 调制 MLP 与双流里另一条流的权重）
    layers: int              # 含 self-attention 的 block 数
    d_model: int             # 隐维度（attention 项用 4·L·N²·d）
    patch: int               # 空间 patch 边长（latent 像素）
    t_patch: int             # 时间 patch（视频）
    vae_f: int               # VAE 空间下采样倍数
    vae_t: int               # VAE 时间下采样倍数（图像模型为 1）
    text_tokens_in_attn: int  # 与图像 token 一起进 self-attention 的文本 token 数（cross-attn 模型为 0）
    text_encoder_params: float
    text_tokens: int         # 文本编码器处理的 token 数
    vae_params: float
    vae_decoder_max_ch: int  # 解码器最高分辨率层的通道数（估激活峰值）
    default_steps: int
    default_cfg: bool        # 是否需要无条件分支（×2）
    default_h: int
    default_w: int
    default_frames: int
    video: bool = False


MODELS = {
    # FLUX.1-dev：19 双流 + 38 单流 = 57 个 attention 层，d 3072；T5 512 token 进联合 attention；guidance 蒸馏 → 无 CFG
    #   每 token 经过：57 层 × 12·d² = 6.45B（19 个双流块里一个 token 只走自己那条流；adaLN 的 3.3B 只处理 1 个条件向量）
    "flux": DiTSpec("FLUX.1-dev", 11.9e9, 6.45e9, 57, 3072, 2, 1, 8, 1, 512,
                    4.7e9 + 0.12e9, 512, 84e6, 128, 28, False, 1024, 1024, 1),
    # SD3-medium：24 个 MMDiT block，d 1536；77 CLIP + 256 T5 = 333 文本 token 进联合 attention
    #   每 token 经过：24 × 12·d² = 0.68B
    "sd3": DiTSpec("SD3-medium", 2.0e9, 0.68e9, 24, 1536, 2, 1, 8, 1, 333,
                    4.7e9 + 0.12e9 + 0.69e9, 333, 84e6, 128, 28, True, 1024, 1024, 1),
    # Qwen-Image：60 个 MMDiT block，d 3072；文本编码器 Qwen2.5-VL-7B；true CFG ×2
    #   每 token 经过：60 × 12·d² = 6.8B
    "qwen": DiTSpec("Qwen-Image", 20.4e9, 6.8e9, 60, 3072, 2, 1, 8, 1, 512,
                    7.6e9, 512, 84e6, 128, 50, True, 1024, 1024, 1),
    # Wan2.1-14B：40 层，d 5120；文本 umT5 经 cross-attention（不进 self-attn）；3D VAE 4×8×8；CFG ×2
    #   每 token 经过：40 × (self-attn 4d² + cross-attn q/o 2d² + FFN 2·d·13824) = 12.0B
    "wan": DiTSpec("Wan2.1-T2V-14B", 14.3e9, 12.0e9, 40, 5120, 2, 1, 8, 4, 0,
                   5.7e9, 512, 127e6, 96, 50, True, 720, 1280, 81, video=True),
    # HunyuanVideo：20 双流 + 40 单流，d 3072；MLLM 文本 256 token 进联合 attention；embedded guidance → 无 CFG
    #   每 token 经过：60 × 12·d² = 6.8B
    "hunyuan": DiTSpec("HunyuanVideo-13B", 13.0e9, 6.8e9, 60, 3072, 2, 1, 8, 4, 256,
                       7.6e9 + 0.12e9, 256, 246e6, 128, 50, False, 720, 1280, 129, video=True),
}

# ---------------------------------------------------------------- 硬件


@dataclass
class GPU:
    name: str
    tflops_bf16: float   # dense，非稀疏
    hbm_gb: float
    bw_tbs: float        # HBM 带宽 TB/s


GPUS = {
    "h100": GPU("H100 SXM", 989, 80, 3.35),
    "h200": GPU("H200", 989, 141, 4.8),
    "a100": GPU("A100 SXM", 312, 80, 2.0),
    "4090": GPU("RTX 4090", 165, 24, 1.0),
    "l40s": GPU("L40S", 362, 48, 0.86),
    "b200": GPU("B200", 2250, 192, 8.0),
}

# ---------------------------------------------------------------- 账


@dataclass
class Ledger:
    spec: DiTSpec
    gpu: GPU
    h: int
    w: int
    frames: int
    steps: int
    cfg: bool
    mfu: float
    dtype_bytes: int = 2

    # ---- 形状
    @property
    def latent_hw(self):
        return self.h // self.spec.vae_f, self.w // self.spec.vae_f

    @property
    def latent_frames(self):
        if not self.spec.video:
            return 1
        return (self.frames - 1) // self.spec.vae_t + 1   # 因果 3D VAE：首帧单独编码

    @property
    def image_tokens(self):
        lh, lw = self.latent_hw
        return (lh // self.spec.patch) * (lw // self.spec.patch) * (self.latent_frames // self.spec.t_patch)

    @property
    def seq_len(self):
        return self.image_tokens + self.spec.text_tokens_in_attn

    # ---- 每步 FLOPs
    @property
    def linear_flops_per_forward(self):
        # 每个 token 过一遍"属于它的"权重：2 FLOP / 参数 / token
        return 2 * self.spec.params_per_token * self.seq_len

    @property
    def attn_flops_per_forward(self):
        # QKᵀ 与 PV 各 2·N²·d，每层一次 self-attention
        return 4 * self.spec.layers * self.seq_len ** 2 * self.spec.d_model

    @property
    def flops_per_forward(self):
        return self.linear_flops_per_forward + self.attn_flops_per_forward

    @property
    def forwards_per_step(self):
        return 2 if self.cfg else 1

    @property
    def dit_flops(self):
        return self.flops_per_forward * self.forwards_per_step * self.steps

    # ---- 另两段
    @property
    def text_flops(self):
        return 2 * self.spec.text_encoder_params * self.spec.text_tokens * self.forwards_per_step

    @property
    def vae_flops(self):
        # 卷积解码器的 FLOPs 与输出像素数成正比：SD 一族的 f8 解码器约 4.8 MFLOP / 输出像素
        # （512² 约 1.25 TFLOPs，1024² 约 5 TFLOPs，来自 diffusers 的 profile）；视频按帧数线性放大
        pixels = self.h * self.w * (self.frames if self.spec.video else 1)
        return 4.8e6 * pixels

    @property
    def total_flops(self):
        return self.dit_flops + self.text_flops + self.vae_flops

    # ---- 显存
    @property
    def dit_weights_bytes(self):
        return self.spec.params * self.dtype_bytes

    @property
    def text_weights_bytes(self):
        return self.spec.text_encoder_params * self.dtype_bytes

    @property
    def vae_weights_bytes(self):
        return self.spec.vae_params * 4   # VAE 通常 fp32

    @property
    def dit_activation_bytes(self):
        # FlashAttention 下无 N² 矩阵；峰值约为一层里同时存活的几份 [N, d] 张量：
        # 残差流 + QKV + attention 输出 + MLP 中间（4d）≈ 10 份 d 宽张量，再乘 CFG batch
        live = 10 * self.seq_len * self.spec.d_model * self.dtype_bytes
        return live * self.forwards_per_step

    @property
    def vae_decode_activation_bytes(self):
        # 解码器最高分辩率层：[frames, C_max, H, W] 的 fp32 张量，同时存活约 4 份（输入、卷积输出、残差、归一化）
        per_frame = self.h * self.w * self.spec.vae_decoder_max_ch * 4
        return 4 * per_frame * (self.frames if self.spec.video else 1)

    # ---- 时间
    def secs(self, flops, mfu=None):
        return flops / (self.gpu.tflops_bf16 * 1e12 * (mfu or self.mfu))

    @property
    def dit_step_secs(self):
        return self.secs(self.flops_per_forward * self.forwards_per_step)

    @property
    def dit_secs(self):
        return self.dit_step_secs * self.steps

    @property
    def text_secs(self):
        # 文本编码器是几百 token 的小 batch prefill，MFU 低，按一半算
        return self.secs(self.text_flops, self.mfu / 2)

    @property
    def vae_secs(self):
        # 卷积解码默认 fp32、大特征图、小通道数，是 memory-bound 的算子链，MFU 只有几个百分点；按 0.05
        return self.secs(self.vae_flops, 0.05)

    @property
    def arithmetic_intensity(self):
        # 一次 DiT 前向：FLOPs / 读一遍权重的字节数（激活忽略）——单请求就是 compute-bound 的原因
        return self.flops_per_forward / self.dit_weights_bytes


def fmt_flops(x):
    for unit, s in ((1e15, "P"), (1e12, "T"), (1e9, "G")):
        if x >= unit:
            return f"{x / unit:.1f} {s}FLOPs"
    return f"{x / 1e6:.1f} MFLOPs"


def fmt_bytes(x):
    return f"{x / 2**30:.1f} GiB" if x >= 2**30 else f"{x / 2**20:.0f} MiB"


def fmt_secs(x):
    if x >= 60:
        return f"{x / 60:.1f} min"
    if x >= 1:
        return f"{x:.2f} s"
    return f"{x * 1e3:.0f} ms"


# ---------------------------------------------------------------- LLM 对照


def llm_decode(gpu, params=7e9, tokens=1000, bw_util=0.7, kv_per_token=128 * 1024):
    """7B dense 模型 batch 1 生成 tokens 个 token：FLOPs 与 memory-bound 时间。"""
    flops = 2 * params * tokens
    weight_bytes = params * 2
    # 每步读一遍权重 + 已有 KV；KV 到 1000 token 才 128 MiB，忽略
    step = weight_bytes / (gpu.bw_tbs * 1e12 * bw_util)
    return flops, step, step * tokens


# ---------------------------------------------------------------- 打印


def print_ledger(L: Ledger):
    s, g = L.spec, L.gpu
    lh, lw = L.latent_hw
    shape = f"{L.h}×{L.w}" + (f"×{L.frames} 帧" if s.video else "")
    print("=" * 78)
    print(f"{s.name}  |  {shape}  |  {L.steps} 步  |  CFG {'×2' if L.cfg else '×1'}  |  {g.name}  |  DiT MFU {L.mfu:.2f}")
    print("=" * 78)

    print("\n[1] 形状与 token")
    print(f"  latent          : {lh}×{lw}" + (f"×{L.latent_frames}（时间 {s.vae_t}×）" if s.video else "") + f"，空间 f={s.vae_f}")
    print(f"  patch           : {s.patch}×{s.patch}" + (f"×{s.t_patch}" if s.video else ""))
    print(f"  图像 token       : {L.image_tokens:,}")
    print(f"  进 attention 的文本 token: {s.text_tokens_in_attn}  →  序列长度 N = {L.seq_len:,}")

    print("\n[2] 一次 DiT 前向的 FLOPs")
    lin, att = L.linear_flops_per_forward, L.attn_flops_per_forward
    print(f"  线性项 2·P_tok·N : {fmt_flops(lin)}   （P_tok = {s.params_per_token / 1e9:.2f}B，总参数 {s.params / 1e9:.1f}B）")
    print(f"  attention 4·L·N²·d: {fmt_flops(att)}   占 {att / (lin + att):.0%}")
    print(f"  合计             : {fmt_flops(lin + att)}   ×{L.forwards_per_step}（CFG）×{L.steps} 步")
    print(f"  算术强度         : {L.arithmetic_intensity:,.0f} FLOP/字节（{g.name} 的拐点 {g.tflops_bf16 * 1e12 / (g.bw_tbs * 1e12):.0f}）")

    print("\n[3] 三段的 FLOPs")
    for name, f in (("文本编码器", L.text_flops), ("DiT × 步数 × CFG", L.dit_flops), ("VAE 解码", L.vae_flops)):
        print(f"  {name:<16}: {fmt_flops(f):>14}   {f / L.total_flops:6.1%}")
    print(f"  {'合计':<16}: {fmt_flops(L.total_flops):>14}")

    print("\n[4] 显存")
    w = L.dit_weights_bytes + L.text_weights_bytes + L.vae_weights_bytes
    print(f"  DiT 权重 bf16    : {fmt_bytes(L.dit_weights_bytes)}")
    print(f"  文本编码器权重   : {fmt_bytes(L.text_weights_bytes)}")
    print(f"  VAE 权重 fp32    : {fmt_bytes(L.vae_weights_bytes)}")
    print(f"  三段权重合计     : {fmt_bytes(w)}   （{g.name} {g.hbm_gb} GB{'，放不下，需 offload 或分卡' if w > g.hbm_gb * 1e9 * 0.9 else ''}）")
    print(f"  DiT 激活峰值     : {fmt_bytes(L.dit_activation_bytes)}   （无 KV cache；随 N 线性）")
    print(f"  VAE 解码激活峰值 : {fmt_bytes(L.vae_decode_activation_bytes)}   （随像素×帧线性{'，需要 tiling' if L.vae_decode_activation_bytes > 20 * 2**30 else ''}）")

    print("\n[5] 时间")
    print(f"  文本编码器       : {fmt_secs(L.text_secs):>10}")
    print(f"  DiT 每步         : {fmt_secs(L.dit_step_secs):>10}   × {L.steps} = {fmt_secs(L.dit_secs)}")
    print(f"  VAE 解码         : {fmt_secs(L.vae_secs):>10}")
    total = L.text_secs + L.dit_secs + L.vae_secs
    print(f"  合计             : {fmt_secs(total):>10}   DiT 占 {L.dit_secs / total:.0%}")

    print("\n[6] 对照：7B LLM batch 1 生成 1000 token（memory-bound，带宽利用 70%）")
    lf, lstep, ltot = llm_decode(g)
    print(f"  FLOPs {fmt_flops(lf)}，每 token {lstep * 1e3:.1f} ms，合计 {fmt_secs(ltot)}")
    print(f"  本次生成 = LLM 的 {L.total_flops / lf:.0f}× FLOPs，{total / ltot:.1f}× 时间")
    print()


def print_sweep(L: Ledger):
    s = L.spec
    print("-" * 78)
    print(f"放大器扫描：{s.name} on {L.gpu.name}（MFU {L.mfu:.2f}）")
    print("-" * 78)
    if s.video:
        rows = [(L.h, L.w, f) for f in (17, 49, 81, 129)] + [(480, 832, 81), (1080, 1920, 81)]
        print(f"  {'形状':<20}{'N':>9}{'attn 占比':>10}{'每步':>10}{'DiT 总':>10}{'DiT 激活':>10}{'VAE 激活':>10}")
        for h, w, f in rows:
            M = replace(L, h=h, w=w, frames=f)
            print(f"  {f'{h}×{w}×{f}':<20}{M.seq_len:>9,}{M.attn_flops_per_forward / M.flops_per_forward:>10.0%}"
                  f"{fmt_secs(M.dit_step_secs):>10}{fmt_secs(M.dit_secs):>10}{fmt_bytes(M.dit_activation_bytes):>10}{fmt_bytes(M.vae_decode_activation_bytes):>10}")
    else:
        rows = [(512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048)]
        print(f"  {'形状':<12}{'N':>8}{'attn 占比':>10}{'每步':>10}{'DiT 总':>10}{'DiT 激活':>10}{'VAE 激活':>10}")
        for h, w in rows:
            M = replace(L, h=h, w=w)
            print(f"  {f'{h}×{w}':<12}{M.seq_len:>8,}{M.attn_flops_per_forward / M.flops_per_forward:>10.0%}"
                  f"{fmt_secs(M.dit_step_secs):>10}{fmt_secs(M.dit_secs):>10}{fmt_bytes(M.dit_activation_bytes):>10}{fmt_bytes(M.vae_decode_activation_bytes):>10}")
    print(f"\n  步数扫描（{L.h}×{L.w}{f'×{L.frames}' if s.video else ''}）：", end="")
    for st in (50, 28, 8, 4, 1):
        M = replace(L, steps=st)
        print(f"  {st} 步 {fmt_secs(M.dit_secs)}", end="")
    print()
    if L.cfg:
        M = replace(L, cfg=False)
        print(f"  去掉 CFG（guidance 蒸馏）：DiT {fmt_secs(L.dit_secs)} → {fmt_secs(M.dit_secs)}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, help="不给则跑全部内置模型的默认账")
    ap.add_argument("--gpu", choices=GPUS, default="h100")
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", dest="cfg", action="store_true", default=None, help="强制 CFG ×2")
    ap.add_argument("--no-cfg", dest="cfg", action="store_false", help="强制无 CFG（guidance 蒸馏）")
    ap.add_argument("--mfu", type=float, default=0.45, help="DiT 前向的 MFU 假设：H100 上 FLUX eager 约 0.3、torch.compile 后约 0.5（按 xDiT 实测反推）")
    ap.add_argument("--sweep", action="store_true", help="附加分辨率 / 帧数 / 步数扫描")
    a = ap.parse_args()

    gpu = GPUS[a.gpu]
    keys = [a.model] if a.model else list(MODELS)
    for k in keys:
        s = MODELS[k]
        L = Ledger(s, gpu,
                   a.height or s.default_h, a.width or s.default_w, a.frames or s.default_frames,
                   a.steps or s.default_steps, s.default_cfg if a.cfg is None else a.cfg, a.mfu)
        print_ledger(L)
        if a.sweep or not a.model:
            print_sweep(L)


if __name__ == "__main__":
    main()
