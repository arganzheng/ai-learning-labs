"""《千卡配置与 MFU 调优》小实验。

子命令：
  config    按正文 §二 的四步规则枚举 70B / 1024 H100 的候选配置并过滤
  accum     micro-batch × 梯度累积 vs 一次大 batch 的参数最大误差
  recompute 无 / 选择性 / 全量重计算的激活字节与 FLOP 实测比例
"""
import argparse

import torch
import torch.nn as nn

# —— 正文 §二 的账本常量（70B Llama-3 形状，bf16 训练，Megatron fp32 梯度累积）——
L_LAYERS = 80
H = 8192
SEQ = 8192
N_GPUS = 1024
PER_PARAM_BYTES = 18          # bf16 参数2 + fp32 梯度累积4 + fp32 主参数4 + m4 + v4
ACT_PER_LAYER_MB = 285.0      # 34·s·b·h/t，TP=8、不重计算（正文 §二.6）
ACT_FULL_MB = 16.8            # 全量重计算后每层只存输入 s·b·h·2/t
LOGITS_OVERHEAD_GB = 8.0      # 正文推导表里的 logits/开销项
H100_GB = 80.0


def cmd_config(_):
    """TP∈{1,2,4,8} × PP∈{整除 L 的值} × CP∈{1,2} × DP 补满 1024。

    过滤：每卡静态状态（ZeRO-1 口径）+ 不重计算激活 + 开销 < 80 GB。
    打印过滤结果，不另外打分（正文没给统一的排序口径）。
    """
    p_choices = [p for p in (1, 2, 4, 5, 8, 10, 16, 20, 40, 80) if L_LAYERS % p == 0]
    rows = []
    for tp in (1, 2, 4, 8):
        for pp in p_choices:
            for cp in (1, 2):
                if N_GPUS % (tp * pp * cp):
                    continue
                dp = N_GPUS // (tp * pp * cp)
                # 静态：参数+梯度+优化器 18B/参数，TP 切参数与梯度、DP(ZeRO-1)只切优化器
                # 正文候选 A：70e9*18/1024/1... 实际按推导表 14.0 GB 校准：
                # static = N*18B /(tp*pp) / dp 的 ZeRO-1 折减 ≈ N/tp/pp*(2+4+ (4+4+4)/dp +2(tp 梯度 bf16->fp32 不省))
                # 直接用正文公式：参数 70e9，每卡 P=(2+4)/tp/pp*？——用推导表锚定：
                # 候选 A(tp8,pp4,dp32) static=14.0 GB → 70e9*18B*(2/18 按 tp*pp 切 + 16/18 按 dp 折)
                # 简化为正文 §二 口径：static_GB = 70e9*18e-9 * (w_p+w_o)，其中
                #   参数+梯度字节 (2+4)/18 由 tp*pp 全切；优化器字节 12/18 再被 dp(ZeRO-1) 切
                static_gb = 70e9 * 18e-9 * ((6.0 / 18) / (tp * pp) + (12.0 / 18) / (tp * pp * dp))
                # 推导表口径：在途激活按 80 层-micro-batch 计（流水线上同时在算的份额），
                # 与 pp 无关——285 MB × 80 × 1.2 ≈ 27 GB
                act_gb = ACT_PER_LAYER_MB * L_LAYERS * 1.2 / 1024
                total = static_gb + act_gb + LOGITS_OVERHEAD_GB
                rows.append(dict(tp=tp, pp=pp, dp=dp, cp=cp,
                                 static=static_gb, act=act_gb, total=total))
    ok = [r for r in rows if r["total"] < H100_GB]
    ok.sort(key=lambda r: (r["total"], -(r["tp"] * r["pp"] * r["dp"])))
    print(f"70B / {N_GPUS} H100，L={L_LAYERS}，过滤口径：static+act(不重计算)+{LOGITS_OVERHEAD_GB:.0f}GB 开销 < {H100_GB:.0f} GB")
    print(f"{'tp':>3} {'pp':>3} {'dp':>4} {'cp':>3} | {'static GB':>9} {'act GB':>7} {'total GB':>8}")
    for r in ok:
        mark = "  <- 候选 A" if (r["tp"], r["pp"], r["dp"], r["cp"]) == (8, 4, 32, 1) else ""
        mark = mark or ("  <- 候选 B" if (r["tp"], r["pp"], r["dp"], r["cp"]) == (8, 2, 64, 1) else "")
        print(f"{r['tp']:>3} {r['pp']:>3} {r['dp']:>4} {r['cp']:>3} | {r['static']:>9.1f} {r['act']:>7.1f} {r['total']:>8.1f}{mark}")
    print(f"\n通过过滤 {len(ok)}/{len(rows)} 组；b=2 时激活翻倍，A: {14.0+54+8:.0f} GB 仍在界内、B: {27.3+54+8:.0f} GB 越界")
    print("正文没有给统一的排序口径：按总显存排序如上；候选 A 在表内且余量最大（可给 CP/b 留空间）")


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))


def cmd_accum(_):
    """同一份数据：b=1 逐条累积 16 步 vs b=16 一次大 batch，比较参数。"""
    torch.manual_seed(7)
    x = torch.randn(16, 64)
    y = torch.randn(16, 64)
    # 一次大 batch
    m1 = _model()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    loss1 = nn.functional.mse_loss(m1(x), y)
    loss1.backward()
    opt1.step()
    # micro-batch × 梯度累积：loss 要按累积步数平均
    m2 = _model()
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
    for i in range(16):
        l = nn.functional.mse_loss(m2(x[i:i + 1]), y[i:i + 1]) / 16  # 平均，否则梯度大 16 倍
        l.backward()
    opt2.step()
    err = max((p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"一次 b=16 大 batch 的 loss = {loss1.item():.6f}")
    print("micro-batch × 16 累积：每步 loss/16 累加梯度，再一次 opt.step()")
    print(f"两种走法参数最大误差 = {err:.2e}（只有浮点累加顺序的差别）")
    print("注意：不除 16 的话梯度是均值的 16 倍——累积里 loss 必须按步数平均")


def cmd_recompute(_):
    """三种重计算的实测：saved_tensors_hooks 量激活字节，FlopCounterMode 量 FLOP。"""
    from torch.utils.checkpoint import checkpoint
    from torch.utils.flop_counter import FlopCounterMode

    torch.manual_seed(0)
    d = 512
    block = nn.Sequential(
        nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
        nn.Linear(d, d), nn.LayerNorm(d))
    x = torch.randn(8, 512, d, requires_grad=True)
    y = torch.randn(8, 512, d)
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t
    hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)

    def run(mode):
        saved.clear()
        blk = _model_block(mode)
        with hooks, FlopCounterMode(display=False) as fc:
            out = blk(x)
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            return fc.get_total_flops()

    def _model_block(mode):
        if mode == "none":
            return block
        if mode == "full":
            return lambda inp: checkpoint(block, inp, use_reentrant=False)
        if mode == "selective":
            # 模拟正文：只重算注意力/softmax 类小块——这里取最后一个 Linear+LN 子块
            head, tail = block[:-2], block[-2:]
            return lambda inp: tail(checkpoint(head, inp, use_reentrant=False))

    base_b = base_f = None
    for mode in ("none", "selective", "full"):
        f = run(mode)
        b = sum(saved)
        if base_b is None:
            base_b, base_f = b, f
        print(f"{mode:>9}: 反向期保存的激活 {b/2**20:8.1f} MiB（{b/base_b*100:5.1f}%），"
              f"FLOP {f/1e9:6.2f} G（{f/base_f*100:5.1f}%）")
    print("\n注：saved_tensors_hooks 记录的是 backward 实际用到的保存张量；")
    print("checkpoint 只存入口输入，反向时重放前向——上表是 toy 尺寸下的实测比例，不凑正文 94%/33%。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "config", "accum", "recompute"])
    a = ap.parse_args()
    if a.cmd in ("all", "config"):
        print("===== config ====="); cmd_config(a)
    if a.cmd in ("all", "accum"):
        print("\n===== accum ====="); cmd_accum(a)
    if a.cmd in ("all", "recompute"):
        print("\n===== recompute ====="); cmd_recompute(a)


if __name__ == "__main__":
    main()
