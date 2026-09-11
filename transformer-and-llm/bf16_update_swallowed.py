"""BF16 权重更新被吃掉（Transformer 与 LLM 06）：为什么需要 FP32 master weights。
https://arganzheng.life/floating-point-formats-and-mixed-precision.html
"""
import torch

w32 = torch.tensor(1.0, dtype=torch.float32)
w16 = torch.tensor(1.0, dtype=torch.bfloat16)
wh  = torch.tensor(1.0, dtype=torch.float16)
lr_step = 1e-3   # 单步 |Δw|, 相对 w=1.0 为 1e-3, 低于 BF16 的 u=2^-8

for step in range(1000):
    w32 = w32 + lr_step
    w16 = w16 + torch.tensor(lr_step, dtype=torch.bfloat16)
    wh  = wh  + torch.tensor(lr_step, dtype=torch.float16)

print("FP32 :", w32.item())   # 约 2.0 (1.0 + 1000 × 0.001, 含微小舍入)
print("FP16 :", wh.item())    # 约 1.98, 每步被舍成 0.000977, 有损但更新保留
print("BF16 :", w16.item())   # 1.0, 一千步更新全部被吃掉

# BF16 在 1.0 附近的间距
one = torch.tensor(1.0, dtype=torch.bfloat16)
print("BF16 spacing at 1.0:", (torch.tensor(1.0 + 2**-7, dtype=torch.bfloat16) - one).item())  # 0.0078125
print("fl_bf16(1.0 + 0.001) =", torch.tensor(1.001, dtype=torch.bfloat16).item())              # 1.0
print("fl_bf16(1.0 + 0.004) =", torch.tensor(1.004, dtype=torch.bfloat16).item())              # 1.0078125 (超过半间距, 进位)

# 正确做法: FP32 master + BF16 副本
master = torch.tensor(1.0, dtype=torch.float32)
for step in range(1000):
    master = master + lr_step
    w_bf16_copy = master.to(torch.bfloat16)   # 前向用这份
print("master:", master.item(), " bf16 copy:", w_bf16_copy.item())  # 2.0, 2.0
