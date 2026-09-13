"""llm_cost.py 第十二版（Transformer 与 LLM 12）：训练配方的账——真实模型的超参表、token 数 → 步数与每步时间、
DeepSeek 的 lr / batch 随算力的经验律、checkpoint 的字节数与写带宽、loss spike 回滚的代价、长上下文阶段的 attention 占比。
https://arganzheng.life/pretraining-recipe-and-training-stability.html
"""
from dataclasses import dataclass

H100_BF16 = 989e12


@dataclass
class Recipe:
    name: str
    N: float                 # 参数量（MoE 取激活参数）
    D: float                 # 训练 token
    batch_tokens: float      # 主阶段的 batch（token）
    seq: int
    peak_lr: float
    warmup_steps: int
    schedule: str
    betas: tuple = (0.9, 0.95)
    wd: float = 0.1
    clip: float = 1.0
    note: str = ""


RECIPES = [
    Recipe("GPT-3 175B", 175e9, 300e9, 3.2e6, 2048, 0.6e-4, 0, "cosine → 10%，warmup 375M token",
           note="warmup 按 token 给：375M / 3.2M ≈ 117 步"),
    Recipe("Llama-2 7B", 7e9, 2e12, 4e6, 4096, 3e-4, 2000, "cosine → 10%"),
    Recipe("Llama-2 70B", 70e9, 2e12, 4e6, 4096, 1.5e-4, 2000, "cosine → 10%"),
    Recipe("Llama-3 405B", 405e9, 15.6e12, 16e6, 8192, 8e-5, 8000, "cosine → 8e-7（1.2M 步）",
           note="batch 4M(seq 4K) → 8M(seq 8K, 252M token 后) → 16M(2.87T 后)"),
    Recipe("DeepSeek-V3（37B 激活）", 37e9, 14.8e12, 63e6, 4096, 2.2e-4, 2000, "常数到 10T → cosine 到 2.2e-5（4.3T）→ 常数 333B → 7.29e-6（167B）",
           note="batch 3072 → 15360 条序列（前 469B token 内线性增大）"),
]


def steps(r):
    return r.D / r.batch_tokens


def step_time(r, gpus, mfu=0.4, peak=H100_BF16):
    return 6 * r.N * r.batch_tokens / (gpus * peak * mfu)


def deepseek_hparam_law(C):
    """DeepSeek LLM（2024）在自家数据上拟合的经验律：最优 lr 与 batch（token）随算力 C 的幂律。"""
    return 0.3118 * C ** -0.125, 0.2920 * C ** 0.3271


def checkpoint_bytes(N, mode="full"):
    """full：BF16 权重 2 + FP32 主权重 4 + Adam 一阶 4 + 二阶 4 = 14 B/参数；weights：BF16 权重 2 B/参数。"""
    return N * (14 if mode == "full" else 2)


def rollback_cost(r, gpus, skipped_batches, rewind_steps, mfu=0.4):
    """PaLM 式处理 spike：回退到 rewind_steps 前的 checkpoint，跳过 skipped_batches 个 batch；代价按重算的步数计。"""
    t = step_time(r, gpus, mfu)
    return (rewind_steps + skipped_batches) * t, skipped_batches * r.batch_tokens


def attention_share(N, d, L, seq):
    """每 token 的 attention 对上下文项 4·d·s·L 与权重项 2N 之比（因果掩码取一半）。"""
    attn = 4 * d * seq * L / 2
    return attn / (2 * N + attn)


def fmtb(x):
    for unit, s in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if abs(x) >= unit:
            return f"{x / unit:.3g} {s}"
    return f"{x:.0f} B"


def fmt(x):
    for unit, s in ((1e15, "P"), (1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= unit:
            return f"{x / unit:.3g}{s}"
    return f"{x:.3g}"


def main():
    print("=== 公开配方：超参数与由它们推出的步数、每步时间 ===")
    print(f"{'model':<24}{'batch':>7}{'seq':>6}{'peak lr':>9}{'warmup':>8}{'steps':>8}{'每步(16K H100, MFU 40%)':>24}  调度")
    for r in RECIPES:
        gpus = 16384 if r.N >= 100e9 else 2048
        t = step_time(r, gpus)
        print(f"{r.name:<24}{fmt(r.batch_tokens):>7}{r.seq:>6}{r.peak_lr:>9.1e}{r.warmup_steps:>8}{steps(r):>8,.0f}"
              f"{t:>17.1f} s ({gpus} 卡)  {r.schedule}")
        if r.note:
            print(f"{'':<24}  {r.note}")
    print("  warmup 占总步数：Llama-3 405B 8000 / 975K = 0.8%；Llama-2 2000 / 500K = 0.4%；DeepSeek-V3 2000 / 235K = 0.9%。")

    print("\n=== DeepSeek LLM 的经验律：lr_opt = 0.3118·C^-0.125，B_opt = 0.2920·C^0.3271（token）===")
    print(f"{'C (FLOPs)':>12}{'lr_opt':>10}{'B_opt':>9}   对照")
    refs = {8.4e22: "Llama-2 7B：lr 3e-4，batch 4M", 8.4e23: "Llama-2 70B：lr 1.5e-4，batch 4M",
            3.3e24: "DeepSeek-V3：lr 2.2e-4，batch 63M", 3.8e25: "Llama-3 405B：lr 8e-5，batch 16M"}
    for C in [1e21, 1e22, 8.4e22, 8.4e23, 3.3e24, 3.8e25]:
        lr, B = deepseek_hparam_law(C)
        print(f"{C:>12.1e}{lr:>10.1e}{fmt(B):>9}   {refs.get(C, '')}")
    print("  量级对得上（lr 随算力缓慢下降，batch 随算力上升），但常数只对拟合它的数据与结构成立；MoE 的 C 用激活参数算。")

    print("\n=== checkpoint 的字节数与写带宽 ===")
    print(f"{'model':<16}{'BF16 权重':>10}{'完整训练状态':>13}{'每小时存一次 → 平均写带宽':>26}{'8 卡节点 × 2 GB/s 的写入需要':>24}")
    for name, N in [("Llama-3 8B", 8.03e9), ("Llama-3 70B", 70.6e9), ("Llama-3 405B", 405e9), ("DeepSeek-V3 671B", 671e9)]:
        full = checkpoint_bytes(N)
        print(f"{name:<16}{fmtb(checkpoint_bytes(N, 'weights')):>10}{fmtb(full):>13}{full / 3600 / 1e9:>19.2f} GB/s{full / 2e9 / 60:>19.0f} 分钟（单节点串行）")
    print("  完整状态 = 14 B/参数（第六篇：BF16 权重 + FP32 主权重 + Adam 两个矩）。数据读 9 MB/s，checkpoint 写 GB/s——训练的 I/O 在这里。")

    print("\n=== loss spike 的回滚代价（PaLM 的做法：回退约 100 步，跳过 200–500 个 batch）===")
    r = RECIPES[3]
    for skipped in [200, 500]:
        t, toks = rollback_cost(r, 16384, skipped, 100)
        print(f"  Llama-3 405B 规格：跳过 {skipped} 个 batch = {fmt(toks)} token，重算 {100 + skipped} 步 ≈ {t / 60:.0f} 分钟 × 16K 卡 = {t / 3600 * 16384:,.0f} GPU 小时")
    print("  一次 spike 的直接代价是几千到上万 GPU 小时；间接代价是人盯着曲线的时间。")

    print("\n=== 长上下文阶段：attention 在每 token FLOPs 里的占比（Llama-3 405B：d=16384，126 层）===")
    for seq in [8192, 32768, 131072]:
        print(f"  seq {seq:>7}：attention 占 {attention_share(405e9, 16384, 126, seq):.0%}")
    print("  800B token 的长上下文阶段占总 token 的 5%，按 FLOPs 算更多；这也是它要单独成为一个阶段、用不同数据配比的原因。")


if __name__ == "__main__":
    main()
