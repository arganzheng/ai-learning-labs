"""llm_cost.py 第十版（Transformer 与 LLM 10）：scaling law 的账——Chinchilla 参数化 loss、给定算力的最优 N / D、
真实模型的 D/N 与 GPU 小时、"过训练"的代价、推理量纳入后的最优点、数据重复的有效 token 数。
https://arganzheng.life/scaling-laws-and-compute-optimal-training.html
"""
from dataclasses import dataclass
from math import exp, log

# ---- Chinchilla 的参数化 loss ----
# L(N, D) = E + A / N^alpha + B / D^beta     N：参数量，D：训练 token 数，loss 以 nats 计
# Hoffmann 等 2022 原文 Approach 3 的常数是 E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28，
# 但它给出的最优 D/N（约 90）与同一篇论文 Approach 1/2 的实验结论（约 20）不一致。
# Besiroglu 等 2024 用原文放出的数据重新拟合，得到下面这组，与 Approach 1/2 一致（Gopher 算力下最优约 70B / 1.4T）。
E, A, B, ALPHA, BETA = 1.8172, 482.01, 2085.43, 0.3478, 0.3658
HOFFMANN = (1.69, 406.4, 410.7, 0.34, 0.28)

H100_BF16, H100_FP8 = 989e12, 1979e12


def chinchilla_loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def compute_optimal(C, consts=None):
    """固定算力 C = 6ND 下最小化 L：拉格朗日给出 N_opt = G (C/6)^a, D_opt = (C/6)^b / G。"""
    _, A_, B_, al, be = consts or (E, A, B, ALPHA, BETA)
    a = be / (al + be)
    G = (al * A_ / (be * B_)) ** (1 / (al + be))
    N = G * (C / 6) ** a
    return N, C / (6 * N)


def gpu_hours(C, peak=H100_BF16, mfu=0.4):
    return C / (peak * mfu * 3600)


def tokens_for_loss(N, target):
    """在 L(N, D) = target 的等 loss 线上，参数 N 需要多少 token；不可达返回 None。"""
    rem = target - E - A / N ** ALPHA
    return None if rem <= 0 else (B / rem) ** (1 / BETA)


def inference_aware_optimum(target, inference_tokens):
    """Sardana & Frankle 2023：最小化 训练 6ND + 推理 2N·D_inf，约束 L(N,D) = target。数值搜索。"""
    best = None
    N = 1e8
    while N < 1e13:
        D = tokens_for_loss(N, target)
        if D is not None:
            total = 6 * N * D + 2 * N * inference_tokens
            if best is None or total < best[0]:
                best = (total, N, D)
        N *= 1.01
    return best


def effective_tokens(unique, epochs, r_star=15.39):
    """Muennighoff 等 2023：重复数据的有效 token 数 D' = U + U·R*·(1 - e^{-R/R*})，R = epochs - 1。"""
    R = epochs - 1
    return unique + unique * r_star * (1 - exp(-R / r_star))


@dataclass
class Run:
    name: str
    N: float          # 参数量（MoE 取激活参数）
    D: float          # 训练 token
    year: str
    note: str = ""


RUNS = [
    Run("GPT-3 175B", 175e9, 300e9, "2020"),
    Run("Gopher 280B", 280e9, 300e9, "2021"),
    Run("Chinchilla 70B", 70e9, 1.4e12, "2022", "与 Gopher 同算力"),
    Run("Llama-1 65B", 65e9, 1.4e12, "2023"),
    Run("Llama-2 7B", 7e9, 2e12, "2023"),
    Run("Llama-2 70B", 70e9, 2e12, "2023"),
    Run("Llama-3 8B", 8e9, 15e12, "2024"),
    Run("Llama-3 70B", 70e9, 15e12, "2024"),
    Run("Llama-3.1 405B", 405e9, 15.6e12, "2024", "论文：3.8e25 FLOPs，30.8M H100 小时"),
    Run("Qwen2.5 7B", 7.6e9, 18e12, "2024"),
    Run("DeepSeek-V3（37B 激活）", 37e9, 14.8e12, "2024", "MoE 总参数 671B；2.79M H800 小时"),
    Run("Llama-4 Scout（17B 激活）", 17e9, 40e12, "2025", "MoE 总参数 109B"),
    Run("Qwen3 32B", 32e9, 36e12, "2025"),
    Run("Kimi K2（32B 激活）", 32e9, 15.5e12, "2025", "MoE 总参数 1T"),
]


def fmt(x):
    for unit, s in ((1e15, "P"), (1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= unit:
            return f"{x / unit:.3g}{s}"
    return f"{x:.3g}"


def main():
    print(f"=== Chinchilla 参数化（Besiroglu 等 2024 重拟合）：L = {E} + {A}/N^{ALPHA} + {B}/D^{BETA} ===")
    print(f"{'算力 C (FLOPs)':>16}{'N_opt':>9}{'D_opt':>9}{'D/N':>6}{'L_opt':>7}{'H100 GPU 小时 (MFU 40%)':>25}")
    for C in [1e21, 1e22, 1e23, 5.76e23, 1e24, 3.8e25, 1e26]:
        N, D = compute_optimal(C)
        print(f"{C:>16.2e}{fmt(N):>9}{fmt(D):>9}{D / N:>6.0f}{chinchilla_loss(N, D):>7.3f}{gpu_hours(C):>19,.0f}")
    print("  （5.76e23 是 Gopher / Chinchilla 的算力；3.8e25 是 Llama 3.1 405B 的）")

    print("\n=== 真实模型：D/N、算力、与同算力 Chinchilla 最优点的差距 ===")
    print(f"{'model':<26}{'N':>7}{'D':>7}{'D/N':>6}{'C=6ND':>10}{'GPU h':>12}{'L(N,D)':>8}{'N_opt(C)':>10}{'L_opt':>7}{'ΔL':>7}  备注")
    for r in RUNS:
        C = 6 * r.N * r.D
        No, Do = compute_optimal(C)
        L, Lo = chinchilla_loss(r.N, r.D), chinchilla_loss(No, Do)
        print(f"{r.name:<26}{fmt(r.N):>7}{fmt(r.D):>7}{r.D / r.N:>6.0f}{C:>10.1e}{gpu_hours(C):>12,.0f}"
              f"{L:>8.3f}{fmt(No):>10}{Lo:>7.3f}{L - Lo:>+7.3f}  {r.note}")
    print("  GPU 小时按 H100 BF16 峰值 989 TFLOPS、MFU 40% 折算，只用于量级对照。")

    print("\n=== 过训练的代价：固定算力 C = 7.2e23（Llama-3 8B 的训练量），把模型缩小 k 倍、数据放大 k 倍 ===")
    C = 6 * 8e9 * 15e12
    No, Do = compute_optimal(C)
    Lo = chinchilla_loss(No, Do)
    print(f"{'k':>4}{'N':>8}{'D':>8}{'D/N':>7}{'loss':>8}{'ΔL':>8}{'推理 FLOPs/token':>18}")
    for k in [1, 2, 4, 8, 10]:
        N, D = No / k, Do * k
        print(f"{k:>4}{fmt(N):>8}{fmt(D):>8}{D / N:>7.0f}{chinchilla_loss(N, D):>8.3f}{chinchilla_loss(N, D) - Lo:>+8.3f}{fmt(2 * N):>18}")
    print(f"  Llama-3 8B 本身：N=8B, D=15T, D/N=1875，L={chinchilla_loss(8e9, 15e12):.3f}，比最优点高 {chinchilla_loss(8e9, 15e12) - Lo:+.3f}，推理成本是最优点模型的 1/{No / 8e9:.0f}")

    print("\n=== 推理量纳入后的最优点：达到同一个 loss，预期服务多少 token 时该选多小的模型 ===")
    target = Lo
    print(f"  目标 loss = {target:.3f}（C=7.2e23 的 Chinchilla 最优 loss）")
    print(f"{'预期推理 token':>16}{'N*':>8}{'D*':>8}{'D/N':>7}{'训练 FLOPs':>12}{'推理 FLOPs':>12}{'合计':>10}")
    for inf in [0, 1e12, 1e13, 1e14, 1e15]:
        total, N, D = inference_aware_optimum(target, inf)
        print(f"{fmt(inf) if inf else '0':>16}{fmt(N):>8}{fmt(D):>8}{D / N:>7.0f}{6 * N * D:>12.2e}{2 * N * inf:>12.2e}{total:>10.2e}")

    print("\n=== 数据不够时重复用：有效 token 数（Muennighoff 等 2023，R* = 15.39）===")
    U = 1e12
    print(f"{'epochs':>7}{'名义 token':>11}{'有效 token':>11}{'折算率':>8}")
    for ep in [1, 2, 4, 8, 16, 40]:
        eff = effective_tokens(U, ep)
        print(f"{ep:>7}{fmt(U * ep):>11}{fmt(eff):>11}{eff / (U * ep):>8.0%}")

    print("\n=== 两组 Chinchilla 常数在 Gopher 算力（5.76e23）下的最优点 ===")
    for name, consts in [("Hoffmann 2022 Approach 3", HOFFMANN), ("Besiroglu 2024 重拟合", (E, A, B, ALPHA, BETA))]:
        N, D = compute_optimal(5.76e23, consts)
        print(f"  {name:<26} N_opt={fmt(N):>6}  D_opt={fmt(D):>6}  D/N={D / N:.0f}  N ∝ C^{consts[4] / (consts[3] + consts[4]):.2f}")

    print("\n=== Kaplan 等 2020 与 Chinchilla 的分配规则对比 ===")
    print("  Kaplan：    N ∝ C^0.73，D ∝ C^0.27  → 算力增加 10 倍，参数 ×5.4，数据 ×1.9")
    a = BETA / (ALPHA + BETA)
    print(f"  Chinchilla：N ∝ C^{a:.2f}，D ∝ C^{1 - a:.2f}  → 算力增加 10 倍，参数 ×{10 ** a:.1f}，数据 ×{10 ** (1 - a):.1f}")


if __name__ == "__main__":
    main()
