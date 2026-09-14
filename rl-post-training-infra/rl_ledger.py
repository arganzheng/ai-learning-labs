"""rl_ledger.py -- 第一版（RL 后训练基础设施 01）：一步 RL 的三本账。
https://arganzheng.life/rl-step-anatomy-rollout-reward-train.html

一步 GRPO / PPO 里生成、打分、训练三段各做多少 FLOP、占多少显存、
在给定 MFU 假设下花多少秒，以及整步的 GPU 利用率上限。纯标准库。

用法：
    python rl_ledger.py                       # 三个内置场景：对话 / 推理 / MoE 推理
    python rl_ledger.py --profile reasoning --gpus 128 --G 32
    python rl_ledger.py --model 32b --tp 2 --profile reasoning --ppo
    python rl_ledger.py -h
"""
import argparse
import math
from dataclasses import dataclass, replace

# ---------------------------------------------------------------- 模型与硬件


@dataclass
class ModelSpec:
    name: str
    n_total: float          # 总参数量（个）
    n_active: float         # 每 token 激活的参数量（dense 模型 = n_total）
    layers: int
    kv_bytes_per_token: int  # 一个 token 的 K+V 字节数（bf16）
    rollout_dtype_bytes: int = 2  # 推理引擎里权重的字节数（bf16 = 2，FP8 = 1）


def gqa_kv_bytes(layers, kv_heads, head_dim, dtype_bytes=2):
    return layers * kv_heads * head_dim * 2 * dtype_bytes


MODELS = {
    # Llama-3-8B：32 层，GQA 8 个 KV 头 × 128 → 128 KiB/token
    "8b": ModelSpec("Llama-3-8B", 8.03e9, 8.03e9, 32, gqa_kv_bytes(32, 8, 128)),
    # Qwen3-32B：64 层，GQA 8 × 128 → 256 KiB/token
    "32b": ModelSpec("Qwen3-32B", 32.8e9, 32.8e9, 64, gqa_kv_bytes(64, 8, 128)),
    # Llama-3-70B：80 层，GQA 8 × 128 → 320 KiB/token
    "70b": ModelSpec("Llama-3-70B", 70.6e9, 70.6e9, 80, gqa_kv_bytes(80, 8, 128)),
    # DeepSeek-V3：671B 总 / 37B 激活；MLA 每层缓存 c_kv(512)+k_rope(64) → 61×576×2 B ≈ 70 KB/token
    # 推理侧按官方部署用 FP8 权重
    "dsv3": ModelSpec("DeepSeek-V3", 671e9, 37e9, 61, 61 * (512 + 64) * 2, rollout_dtype_bytes=1),
}


@dataclass
class Hardware:
    name: str = "H100 SXM"
    peak_flops: float = 989e12     # bf16 dense
    hbm_bytes: float = 80e9
    hbm_bw: float = 3.35e12
    pcie_bw: float = 25e9          # 单向，训练状态 offload 走这里
    sync_bw: float = 50e9          # 跨机权重广播的有效带宽（400 Gb/s IB 一条链路）


# ---------------------------------------------------------------- 一步的配置


@dataclass
class RLConfig:
    B: int = 512            # prompt 数
    G: int = 16             # 每个 prompt 的回答数
    P: int = 500            # prompt 平均长度
    L: int = 8192           # 回答平均长度
    L_max: int = 32768      # 回答最大长度（决定长尾）
    gpus: int = 64
    tp: int = 1             # 推理引擎的张量并行度（一个实例占几张卡）
    ppo: bool = False       # PPO：多一个价值模型
    reward_model: bool = False  # 用奖励模型（否则是规则 / 验证器，不占 GPU）
    recompute_old_logprob: bool = True  # 训练器重算 log pi_old（训推不一致的默认做法）
    mfu_train: float = 0.40
    mfu_prefill: float = 0.50
    bw_util_decode: float = 0.60  # decode 时 HBM 带宽的有效利用率
    overhead_frac: float = 0.10   # 每卡显存里 CUDA context / NCCL / 碎片的份额


# ---------------------------------------------------------------- 账


def fmt(x, unit=""):
    for div, suf in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:6.2f} {suf}{unit}"
    return f"{x:6.2f} {unit}"


def ledger(m: ModelSpec, c: RLConfig, hw: Hardware):
    seqs = c.B * c.G
    resp_tokens = seqs * c.L
    prompt_tokens_unique = c.B * c.P            # 前缀缓存：一个 prompt 的 prefill 只做一次
    all_tokens = seqs * (c.P + c.L)             # 前向 / 训练要过的 token（prompt 也过前向）
    Na, Nt = m.n_active, m.n_total

    # --- FLOPs：生成 2N/token，前向 2N，训练 6N；critic 8N ---
    f_gen = 2 * Na * resp_tokens
    f_prefill = 2 * Na * prompt_tokens_unique
    f_ref = 2 * Na * all_tokens
    f_old = 2 * Na * all_tokens if c.recompute_old_logprob else 0.0
    f_rm = 2 * Na * all_tokens if c.reward_model else 0.0
    f_train = 6 * Na * all_tokens
    f_critic = 8 * Na * all_tokens if c.ppo else 0.0
    f_total = f_gen + f_prefill + f_ref + f_old + f_rm + f_train + f_critic

    # --- 显存（bf16 权重 2 字节/参数；训练状态 16 字节/参数，FSDP 均分到全部卡）---
    w_bytes = 2 * Nt
    train_state = 16 * Nt
    per_gpu_train = train_state / c.gpus
    per_gpu_ref = w_bytes / c.gpus
    per_gpu_rollout_w = m.rollout_dtype_bytes * Nt / c.tp
    kv_total = all_tokens * m.kv_bytes_per_token

    # --- 生成：decode 是 memory-bound，按每步读的字节数估时间 ---
    instances = c.gpus // c.tp
    usable = hw.hbm_bytes * c.tp * (1 - c.overhead_frac) - per_gpu_rollout_w * c.tp
    # 连续批处理下在飞的序列平均只生成到一半：按 P + L/2 估每条的 KV，得到一个实例的稳态并发
    kv_per_seq_avg = (c.P + c.L / 2) * m.kv_bytes_per_token
    concurrent = max(1, int(usable // kv_per_seq_avg))
    concurrent = min(concurrent, math.ceil(seqs / instances))    # 序列不够多时并发由序列数决定
    kv_inflight = concurrent * kv_per_seq_avg
    step_bytes = per_gpu_rollout_w * c.tp + kv_inflight
    t_decode_step = step_bytes / (hw.hbm_bw * c.tp * c.bw_util_decode)
    decode_steps_total = resp_tokens / (instances * concurrent)   # 吞吐意义下的 decode 步数
    t_gen_throughput = decode_steps_total * t_decode_step
    # 长尾：最后一波里最长的序列要多走 (L_max - L) 步，此时并发很低、每步只读权重
    t_tail = (c.L_max - c.L) * (per_gpu_rollout_w * c.tp) / (hw.hbm_bw * c.tp * c.bw_util_decode)
    t_gen = t_gen_throughput + t_tail
    mfu_decode = f_gen / (t_gen * c.gpus * hw.peak_flops)

    cluster = c.gpus * hw.peak_flops
    t_prefill = f_prefill / (cluster * c.mfu_prefill)
    t_fwd = (f_ref + f_old + f_rm) / (cluster * c.mfu_prefill)
    t_train = (f_train + f_critic) / (cluster * c.mfu_train)

    # --- 两次同步（共置形态）：显存切换 + 权重同步 ---
    t_switch = 2 * (per_gpu_train + per_gpu_ref) / hw.pcie_bw    # 训练状态出去再回来
    t_sync = w_bytes / hw.sync_bw                                 # 新权重广播到推理实例
    t_step = t_gen + t_prefill + t_fwd + t_train + t_switch + t_sync
    step_mfu = f_total / (t_step * cluster)

    return dict(
        seqs=seqs, resp_tokens=resp_tokens, all_tokens=all_tokens,
        f_gen=f_gen, f_prefill=f_prefill, f_ref=f_ref, f_old=f_old, f_rm=f_rm,
        f_train=f_train, f_critic=f_critic, f_total=f_total,
        w_bytes=w_bytes, train_state=train_state, per_gpu_train=per_gpu_train,
        per_gpu_ref=per_gpu_ref, per_gpu_rollout_w=per_gpu_rollout_w, kv_total=kv_total,
        instances=instances, concurrent=concurrent, t_decode_step=t_decode_step,
        waves=seqs / (instances * concurrent), t_gen=t_gen, t_tail=t_tail, mfu_decode=mfu_decode,
        tok_s_gpu=concurrent / t_decode_step / c.tp,
        t_prefill=t_prefill, t_fwd=t_fwd, t_train=t_train, t_switch=t_switch, t_sync=t_sync,
        t_step=t_step, step_mfu=step_mfu,
    )


def report(m: ModelSpec, c: RLConfig, hw: Hardware):
    r = ledger(m, c, hw)
    algo = "PPO" if c.ppo else "GRPO"
    print(f"== {m.name}  {algo}  B={c.B} G={c.G} P={c.P} L={c.L} L_max={c.L_max}  "
          f"{c.gpus}x{hw.name} tp={c.tp}")
    print(f"  序列 {r['seqs']:,}   回答 token {fmt(r['resp_tokens'])}   前向要过的 token {fmt(r['all_tokens'])}")
    print("  -- FLOPs")
    rows = [("生成 2N", r["f_gen"]), ("prompt prefill 2N", r["f_prefill"]), ("参考前向 2N", r["f_ref"]),
            ("旧策略前向 2N", r["f_old"]), ("奖励模型前向 2N", r["f_rm"]),
            ("策略训练 6N", r["f_train"]), ("价值模型 8N", r["f_critic"])]
    for k, v in rows:
        if v:
            print(f"     {k:<18}{fmt(v, 'FLOP')}   {100 * v / r['f_total']:5.1f}%")
    print(f"     {'合计':<18}{fmt(r['f_total'], 'FLOP')}   = {r['f_total'] / (m.n_active * r['all_tokens']):.1f} N 每 token")
    print("  -- 显存")
    print(f"     bf16 权重 {fmt(r['w_bytes'], 'B')}   训练状态(16/参数) {fmt(r['train_state'], 'B')}"
          f"   -> 每卡：训练 {fmt(r['per_gpu_train'], 'B')} + 参考 {fmt(r['per_gpu_ref'], 'B')}"
          f" + 推理权重副本 {fmt(r['per_gpu_rollout_w'], 'B')}")
    print(f"     KV cache 全部序列 {fmt(r['kv_total'], 'B')}（{m.kv_bytes_per_token / 1024:.0f} KiB/token）"
          f"   集群显存 {fmt(c.gpus * hw.hbm_bytes, 'B')}   -> {r['instances']} 个实例 × 并发 {r['concurrent']}，"
          f"分 {r['waves']:.1f} 波")
    print("  -- 时间（秒）")
    for k, v in (("生成（含长尾 %.0fs）" % r["t_tail"], r["t_gen"]), ("prompt prefill", r["t_prefill"]),
                 ("参考/旧策略/RM 前向", r["t_fwd"]), ("训练", r["t_train"]),
                 ("显存切换（共置）", r["t_switch"]), ("权重同步", r["t_sync"])):
        print(f"     {k:<22}{v:8.1f}   {100 * v / r['t_step']:5.1f}%")
    print(f"     {'一步墙钟':<22}{r['t_step']:8.1f}   decode 步 {1000 * r['t_decode_step']:.1f} ms，"
          f"每卡 {r['tok_s_gpu']:.0f} token/s，decode MFU {100 * r['mfu_decode']:.1f}%")
    print(f"  -- 全步 MFU {100 * r['step_mfu']:.1f}%   ({fmt(r['f_total'], 'FLOP')} / "
          f"({c.gpus} GPU × {r['t_step']:.0f} s × {fmt(hw.peak_flops, 'FLOPS')}))")
    print()
    return r


PROFILES = {
    "chat": RLConfig(B=512, G=8, P=300, L=1024, L_max=4096, gpus=16),
    "reasoning": RLConfig(B=512, G=16, P=500, L=8192, L_max=32768, gpus=64),
    "agent": RLConfig(B=256, G=8, P=2000, L=20000, L_max=65536, gpus=64),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default=None)
    ap.add_argument("--profile", choices=PROFILES, default=None)
    for f in ("B", "G", "P", "L", "L_max", "gpus", "tp"):
        ap.add_argument(f"--{f}", type=int)
    ap.add_argument("--ppo", action="store_true")
    ap.add_argument("--rm", action="store_true", help="用奖励模型而不是规则奖励")
    ap.add_argument("--no-recompute", action="store_true", help="不重算 log pi_old（直接用推理引擎的）")
    a = ap.parse_args()
    hw = Hardware()
    if a.model is None and a.profile is None:
        report(MODELS["8b"], PROFILES["chat"], hw)
        report(MODELS["8b"], PROFILES["reasoning"], hw)
        report(MODELS["32b"], replace(PROFILES["reasoning"], tp=2), hw)
        report(MODELS["dsv3"], replace(PROFILES["reasoning"], gpus=256, tp=32), hw)
        return
    c = PROFILES[a.profile or "reasoning"]
    over = {f: getattr(a, f) for f in ("B", "G", "P", "L", "L_max", "gpus", "tp") if getattr(a, f) is not None}
    c = replace(c, ppo=a.ppo, reward_model=a.rm, recompute_old_logprob=not a.no_recompute, **over)
    report(MODELS[a.model or "8b"], c, hw)


if __name__ == "__main__":
    main()
