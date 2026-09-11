"""浮点格式逐位构造（Transformer 与 LLM 06）：FP32 / FP16 / BF16 / FP8 的位域、eps、max、tiny。
https://arganzheng.life/floating-point-formats-and-mixed-precision.html
"""
import struct
import numpy as np
import torch

def fp32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]

def fp16_from_bits(bits: int) -> float:
    return float(np.frombuffer(struct.pack("<H", bits), dtype=np.float16)[0])

def torch_from_bits(bits: int, storage, dtype) -> torch.Tensor:
    return torch.tensor([bits], dtype=storage).view(dtype)

# FP32: 1/8/23
print("FP32 max        ", fp32_from_bits(0x7F7FFFFF))   # 3.4028235e38
print("FP32 min normal ", fp32_from_bits(0x00800000))   # 1.1754944e-38
print("FP32 min subnorm", fp32_from_bits(0x00000001))   # 1.4e-45

# FP16: 1/5/10
print("FP16 max        ", fp16_from_bits(0x7BFF))       # 65504.0
print("FP16 min normal ", fp16_from_bits(0x0400))       # 6.104e-05
print("FP16 min subnorm", fp16_from_bits(0x0001))       # 5.96e-08
print("FP16 inf        ", fp16_from_bits(0x7C00))       # inf

# BF16: 1/8/7  (int16 位模式, 0x7F7F 为正数, 可直接用)
print("BF16 max        ", torch_from_bits(0x7F7F, torch.int16, torch.bfloat16).item())  # 3.3895e38
print("BF16 min normal ", torch_from_bits(0x0080, torch.int16, torch.bfloat16).item())  # 1.1755e-38
print("BF16 min subnorm", torch_from_bits(0x0001, torch.int16, torch.bfloat16).item())  # 9.18e-41

# FP8 E4M3 (fn): 1/4/3, 无 inf, 0x7F 为 NaN
e4m3 = torch.float8_e4m3fn
print("E4M3 max        ", torch_from_bits(0x7E, torch.uint8, e4m3).float().item())  # 448.0
print("E4M3 0x7F       ", torch_from_bits(0x7F, torch.uint8, e4m3).float().item())  # nan
print("E4M3 min normal ", torch_from_bits(0x08, torch.uint8, e4m3).float().item())  # 0.015625
print("E4M3 min subnorm", torch_from_bits(0x01, torch.uint8, e4m3).float().item())  # 0.001953125

# FP8 E5M2: 1/5/2, 有 inf
e5m2 = torch.float8_e5m2
print("E5M2 max        ", torch_from_bits(0x7B, torch.uint8, e5m2).float().item())  # 57344.0
print("E5M2 0x7C       ", torch_from_bits(0x7C, torch.uint8, e5m2).float().item())  # inf
print("E5M2 min normal ", torch_from_bits(0x04, torch.uint8, e5m2).float().item())  # 6.104e-05
print("E5M2 min subnorm", torch_from_bits(0x01, torch.uint8, e5m2).float().item())  # 1.526e-05

# 机器精度: torch.finfo 与 2^-M 对照
for dt, M in [(torch.float32, 23), (torch.float16, 10), (torch.bfloat16, 7), (e4m3, 3), (e5m2, 2)]:
    print(f"{str(dt):22s} eps={torch.finfo(dt).eps:.3e}  2^-M={2.0**-M:.3e}  "
          f"max={torch.finfo(dt).max:.3e}  tiny={torch.finfo(dt).tiny:.3e}")

# TF32 只是输入截断: 张量仍是 4 字节
x = torch.randn(1024, 1024)
print("float32 element_size:", x.element_size())  # 4, 打开 allow_tf32 后也不变
