"""PyTorch 使用层·下（工具箱 03）：显存的账——全量 / LoRA / QLoRA 的字节数、激活的估算、bf16 的实际字节、CPU 上量一量。
https://arganzheng.life/pytorch-in-use-mixed-precision-memory-ledger-and-multi-gpu.html

纯算术 + 少量 torch；不需要 GPU。有 CUDA 时最后一节会用 max_memory_allocated 对账。

    python 03_memory_ledger.py
"""
import torch
import torch.nn as nn

GB = 1e9
GiB = 2**30


def llama3_8b():
    """从 config.json 算参数量（L0 第一篇 / 第三篇的表）。"""
    d, dff, L, V = 4096, 14336, 32, 128256
    nq, nkv, dh = 32, 8, 128
    attn = d * (nq * dh) + d * (nkv * dh) * 2 + (nq * dh) * d      # Wq Wk Wv Wo
    mlp = 3 * d * dff
    per_layer = attn + mlp
    total = L * per_layer + V * d * 2                                # 词嵌入 + 输出层（不共享）
    lora_r16 = 16 * ((d + nq * dh) + (d + nkv * dh) * 2 + (nq * dh + d) + 3 * (d + dff)) * L
    return dict(d=d, dff=dff, L=L, V=V, per_layer=per_layer, total=total, lora_r16=lora_r16)


def ledger():
    m = llama3_8b()
    N = m["total"]
    print("=== 1. 训练状态：每个可训练参数 16 字节 ===")
    print("  bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + AdamW 一阶矩 4 + 二阶矩 4 = 16 字节")
    print(f"  Llama-3-8B: 每层 {m['per_layer']/1e6:.0f} M, 32 层 + 嵌入 = {N/1e9:.2f} B 参数; r=16 LoRA 可训练 {m['lora_r16']/1e6:.1f} M ({m['lora_r16']/N*100:.2f}%)")
    rows = [
        ("全量微调 (bf16 + AdamW)", N * 16, 0),
        ("LoRA r=16 (冻结 bf16 基座)", N * 2, m["lora_r16"] * 16),
        ("QLoRA (4-bit 基座 ≈ 0.5 B/参数 + 常数)", N * 0.55, m["lora_r16"] * 16),
    ]
    print(f"  {'方案':<40} {'冻结/权重':>10} {'训练状态':>10} {'合计':>10}")
    for name, frozen, state in rows:
        print(f"  {name:<40} {frozen/GB:>8.1f} GB {state/GB:>8.2f} GB {(frozen+state)/GB:>8.1f} GB")
    print("  → 80 GB 一张卡：全量放不下（要 ≥ 2 卡 + FSDP 切状态），LoRA 一张卡绰绰有余，QLoRA 消费级卡")
    print()

    print("=== 2. 激活：与参数量无关，与 batch × seq × 层数 × d 成正比 ===")
    B, T = 1, 4096
    resid = B * T * m["d"] * 2                                       # 残差流一份，bf16
    # 一层要为反向保存的主要中间量（粗略，bf16）：attention 输入 / q k v / attention 输出 / MLP 输入 / gate·up (dff) ×2 / act
    per_layer_act = B * T * (m["d"] * 6 + m["dff"] * 3) * 2
    print(f"  B={B}, T={T}: 残差流一份 {resid/2**20:.0f} MiB/层, 32 层 {resid*32/GiB:.1f} GiB")
    print(f"  反向要保存的中间量约 {per_layer_act/2**20:.0f} MiB/层, 32 层 {per_layer_act*32/GiB:.1f} GiB  ← 比参数的 16 GB 都大")
    print(f"  gradient checkpointing 只存每层输入 ({resid*32/GiB:.1f} GiB)，反向重算其余，多约 1/3 前向计算")
    print(f"  B=8 时激活 ×8 = {per_layer_act*32*8/GiB:.0f} GiB —— 长序列大 batch 下激活是显存大头，OOM 先看这里")
    print()

    print("=== 3. dtype 的实际字节：torch 上量 ===")
    for dt in (torch.float32, torch.bfloat16, torch.float16, torch.int8):
        t = torch.zeros(1000, 1000, dtype=dt)
        print(f"  {str(dt):<16} 1000×1000 = {t.numel() * t.element_size()/1e6:.1f} MB  (element_size {t.element_size()})")
    x = torch.randn(64, 256)
    w = nn.Linear(256, 256)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = w(x)
        s = y.float().softmax(-1)
    print(f"  autocast 下: Linear 输出 {y.dtype}, .float() 后 softmax {s.dtype}; 权重本身仍是 {w.weight.dtype}（主副本不变）")
    print()

    if torch.cuda.is_available():
        print("=== 4. CUDA 对账：算的 vs 量的 ===")
        torch.cuda.reset_peak_memory_stats()
        model = nn.Sequential(*[nn.Linear(2048, 2048) for _ in range(8)]).cuda()
        n = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters())
        x = torch.randn(64, 2048, device="cuda")
        model(x).sum().backward(); opt.step()
        peak = torch.cuda.max_memory_allocated()
        print(f"  {n/1e6:.1f} M 参数 fp32: 算 4+4+4+4 = 16 B/参数 → {n*16/GB:.2f} GB; 量到峰值 {peak/GB:.2f} GB")
    else:
        print("=== 4. 没有 CUDA，跳过对账；有卡时会打印 算的 16 B/参数 vs torch.cuda.max_memory_allocated() ===")


if __name__ == "__main__":
    ledger()
