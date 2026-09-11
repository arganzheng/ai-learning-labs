"""GEMM 误差随 k 增长（Transformer 与 LLM 06）：FP32 / BF16 / FP16 / FP8 累加误差对比。
https://arganzheng.life/floating-point-formats-and-mixed-precision.html
"""
import torch

torch.manual_seed(0)

def rel_err(c, ref):
    return ((c.double() - ref).norm() / ref.norm()).item()

def bf16_accumulate_matmul(a16, b16):
    """模拟没有 FP32 累加器的 BF16 GEMM: 每加一项都舍回 BF16 (慢, 仅演示)."""
    m, k = a16.shape
    n = b16.shape[1]
    acc = torch.zeros(m, n, dtype=torch.bfloat16)
    for i in range(k):
        acc = acc + (a16[:, i:i+1] * b16[i:i+1, :])   # 逐项加, 结果留在 BF16
    return acc

m = n = 64
print(f"{'k':>6} {'fp32':>10} {'bf16/fp32acc':>14} {'bf16/bf16acc':>14} {'u*sqrt(k)':>10}")
for k in [64, 256, 1024, 4096, 16384]:
    a = torch.randn(m, k, dtype=torch.float64)
    b = torch.randn(k, n, dtype=torch.float64)
    ref = a @ b
    e32 = rel_err(a.float() @ b.float(), ref)
    e16 = rel_err(a.bfloat16() @ b.bfloat16(), ref)
    eacc = rel_err(bf16_accumulate_matmul(a.bfloat16(), b.bfloat16()), ref) if k <= 4096 else float("nan")
    print(f"{k:>6} {e32:>10.2e} {e16:>14.2e} {eacc:>14.2e} {2**-8 * k**0.5:>10.2e}")

# FP8 (需要 Hopper / Ada 及以上, torch >= 2.1 且 CUDA 可用)
if torch.cuda.is_available() and hasattr(torch, "_scaled_mm"):
    dev = "cuda"
    for k in [256, 1024, 4096]:
        a = torch.randn(m, k, dtype=torch.float64, device=dev)
        b = torch.randn(k, n, dtype=torch.float64, device=dev)
        ref = a @ b
        sa = 448.0 / a.abs().max()
        sb = 448.0 / b.abs().max()
        a8 = (a * sa).to(torch.float8_e4m3fn)
        b8 = (b * sb).to(torch.float8_e4m3fn)
        # _scaled_mm 要求 b 为列主序 (k×n 转置后 contiguous), scale 为 FP32 标量张量
        c = torch._scaled_mm(a8, b8.t().contiguous().t(),
                             scale_a=(1 / sa).float().reshape(()),
                             scale_b=(1 / sb).float().reshape(()),
                             out_dtype=torch.float32)
        print(f"FP8 E4M3 per-tensor, k={k}: rel_err={rel_err(c, ref):.2e}")
