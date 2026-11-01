"""训练稳定性与数据管线（第 10 篇）：spike/clip/logit/bf16/packing/resume 实测。

全部 CPU 单进程；打印真实测量值，不为凑结论调参。
    spike   2 层 d=64 小模型 300 步，step 150–155 注入坏 batch（全同一 token），
            然后从 step-140 checkpoint 回退、跳过坏窗口重跑，两条 loss/grad-norm 对照
    clip    一步大梯度：clip_grad_norm_ 前后全局范数，以及各参数被同一个标量缩放
    logit   单层注意力高 LR，带 / 不带 QK-LayerNorm，逐步 max|logit|
    bf16    1.0 上累加 1 万次 1e-3：bf16 累加器 vs fp32 累加器
    packing 长尾序列长采样（固定 seed）：pad-to-max vs 贪心装进 4096 的利用率，
            并打印一条打包行的 cu_seqlens / position_ids
    resume  (a) Megatron 式整数位置重启（index sampler）
            (b) torchdata StatefulDataLoader state_dict/load_state_dict
"""
import copy
import json
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "expected")


# ---------------------------------------------------------------- spike ----

V, S, D, LAYERS, HEADS = 128, 32, 64, 2, 4
STEPS, BAD0, BAD1, CKPT_STEP = 300, 150, 156, 140


def _make_model(seed=0):
    torch.manual_seed(seed)
    emb = nn.Embedding(V, D)
    layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=HEADS, dim_feedforward=4 * D, dropout=0.0,
        batch_first=True, norm_first=True)
    enc = nn.TransformerEncoder(layer, num_layers=LAYERS)
    head = nn.Linear(D, V, bias=False)
    return nn.ModuleDict({"emb": emb, "enc": enc, "head": head})


def _batch(step, bad=False, bs=8):
    g = torch.Generator().manual_seed(10_000 + step)
    if bad:
        # 坏 batch：整批同一个 token 重复（最容易构造的"坏数据"）
        x = torch.full((bs, S), 5, dtype=torch.long)
    else:
        # 可学数据：x[i+1] = (x[i]*7+3) % V 的确定性链，起点随机
        x0 = torch.randint(0, V, (bs, 1), generator=g)
        x = torch.empty(bs, S, dtype=torch.long)
        x[:, 0] = x0[:, 0]
        for i in range(1, S):
            x[:, i] = (x[:, i - 1] * 7 + 3) % V
    return x[:, :-1], x[:, 1:]


def _causal_mask(s):
    return torch.triu(torch.full((s, s), float("-inf")), diagonal=1)


def _train_steps(model, opt, step_range, skip=()):
    """跑给定步集合；返回 {step: (loss, grad_norm)}。"""
    series = {}
    for step in step_range:
        if step in skip:
            continue
        bad = BAD0 <= step < BAD1
        x, y = _batch(step, bad=bad)
        logits = model["head"](model["enc"](model["emb"](x),
                                            mask=_causal_mask(x.shape[1])))
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        series[step] = (loss.item(), gn)
    return series


def cmd_spike():
    torch.manual_seed(0)
    model = _make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = None
    run1 = {}
    for step in range(STEPS):
        if step == CKPT_STEP:
            ckpt = {"model": copy.deepcopy(model.state_dict()),
                    "opt_sd": copy.deepcopy(opt.state_dict())}
        bad = BAD0 <= step < BAD1
        x, y = _batch(step, bad=bad)
        logits = model["head"](model["enc"](model["emb"](x),
                                            mask=_causal_mask(x.shape[1])))
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        run1[step] = (loss.item(), gn)

    # ---- 处置：回退到 step-140 ckpt，跳过坏窗口 150–155，续跑到 200 ----
    model2 = _make_model()
    model2.load_state_dict(ckpt["model"])
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    opt2.load_state_dict(ckpt["opt_sd"])
    run2 = _train_steps(model2, opt2, range(CKPT_STEP, 200),
                        skip=range(BAD0, BAD1))

    print(f"小模型：2 层 d={D}，V={V}，Adam lr=1e-3，{STEPS} 步")
    print(f"坏窗口：step {BAD0}–{BAD1-1} 每步整批全是 token 5")
    print(f"step-140 存了 checkpoint；处置 = 回退 + 跳过 {BAD0}–{BAD1-1} 的数据\n")
    print("run1（不处理，照常喂坏 batch）:")
    for s in range(BAD0 - 4, BAD1 + 10):
        l, g = run1[s]
        tag = " <- 坏 batch" if BAD0 <= s < BAD1 else ""
        print(f"  step {s:3d}  loss {l:7.3f}  grad_norm {g:6.2f}{tag}")
    print("\nrun2（从 step-140 恢复，跳过坏窗口）:")
    for s in range(CKPT_STEP, CKPT_STEP + 3):
        l, g = run2[s]
        print(f"  step {s:3d}  loss {l:7.3f}  grad_norm {g:6.2f}")
    print(f"  ...（{BAD0}–{BAD1-1} 的 batch 被丢弃，未训练）...")
    for s in list(range(BAD1, BAD1 + 6)) + [175, 199]:
        l, g = run2[s]
        print(f"  step {s:3d}  loss {l:7.3f}  grad_norm {g:6.2f}")
    med = sorted(g for _, g in run1.values() if g < 50)[len(run1) // 2]
    print(f"\nrun1 正常段 grad_norm 中位 ≈{med:.2f}；坏窗口内最大 "
          f"{max(run1[s][1] for s in range(BAD0, BAD1)):.2f}")
    print(f"run2 step {BAD1} 的 loss {run2[BAD1][0]:.3f}"
          f"（run1 同步 {run1[BAD1][0]:.3f}，说明参数轨迹被改写了）")
    with open(os.path.join(OUT_DIR, "10_stab_spike_curve.json"), "w") as f:
        json.dump({"run1": run1, "run2": run2, "bad": [BAD0, BAD1],
                   "ckpt": CKPT_STEP}, f)


# ----------------------------------------------------------------- clip ----

def cmd_clip():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 10))
    # 人为大梯度
    x = torch.randn(4, 64)
    loss = (model(x) ** 2).sum() * 100
    loss.backward()
    before = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters()]).item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters()]).item()
    print(f"clip 前全局范数  {before:.3f}")
    print(f"clip 后全局范数  {after:.3f}  (max_norm=1.0)")
    print("每个参数张量的缩放比（应全是同一个标量，即方向不变）：")
    for i, p in enumerate(model.parameters()):
        r = 1.0 / (before + 1e-6)
        n = p.grad.norm().item()
        print(f"  param[{i}] shape {tuple(p.shape)}  clip后范数 {n:.3f}  "
              f"缩放比 {n / (n / r + 1e-12):.6f}  (理论 {r:.6f})")


# ---------------------------------------------------------------- logit ----

class Attn(nn.Module):
    def __init__(self, qk_ln=False):
        super().__init__()
        self.wq, self.wk, self.wv = (nn.Linear(D, D, bias=False) for _ in range(3))
        self.wo = nn.Linear(D, D, bias=False)
        self.lnq = nn.LayerNorm(D) if qk_ln else None
        self.lnk = nn.LayerNorm(D) if qk_ln else None

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        if self.lnq is not None:
            q, k = self.lnq(q), self.lnk(k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        return self.wo(scores.softmax(-1) @ v)


def cmd_logit():
    torch.manual_seed(2)
    steps = 60
    out = {}
    for qk_ln in (False, True):
        torch.manual_seed(2)
        attn = Attn(qk_ln=qk_ln)
        opt = torch.optim.SGD(attn.parameters(), lr=0.5)  # 故意高 LR
        maxes = []
        for step in range(steps):
            x = torch.randn(4, 16, D)
            y = torch.randn(4, 16, D)
            pred = attn(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                q, k = attn.wq(x), attn.wk(x)
                if attn.lnq is not None:
                    q, k = attn.lnq(q), attn.lnk(k)
                mx = (q @ k.transpose(-1, -2) / math.sqrt(D)).abs().max().item()
            maxes.append(mx)
        out[qk_ln] = maxes
    print("单层注意力，SGD lr=0.5（故意偏高），60 步，每步 max|q·k/sqrt(d)|：")
    print("step   不带 QK-LN    带 QK-LN")
    for s in range(0, steps, 5):
        print(f" {s:3d}   {out[False][s]:10.2f}   {out[True][s]:10.2f}")
    print(f"末步    {out[False][-1]:10.2f}   {out[True][-1]:10.2f}")
    with open(os.path.join(OUT_DIR, "10_stab_logit_curve.json"), "w") as f:
        json.dump(out, f)


# ----------------------------------------------------------------- bf16 ----

def cmd_bf16():
    n, delta = 10_000, 1e-3
    acc16 = torch.tensor(1.0, dtype=torch.bfloat16)
    for _ in range(n):
        acc16 = acc16 + torch.tensor(delta, dtype=torch.bfloat16)
    acc32 = torch.tensor(1.0, dtype=torch.float32)
    for _ in range(n):
        acc32 = acc32 + torch.tensor(delta, dtype=torch.float32)
    # fp32 累加器最后转 bf16 存（精度纪律的折中点）
    print(f"真值         1.0 + {n}×{delta} = {1 + n * delta:.4f}")
    print(f"bf16 累加器  逐次加           = {acc16.item():.4f}")
    print(f"fp32 累加器  逐次加           = {acc32.item():.4f}")
    print(f"fp32 加完再转 bf16            = "
          f"{acc32.to(torch.bfloat16).item():.4f}")


# --------------------------------------------------------------- packing ----

def cmd_packing():
    import random
    rng = random.Random(0)
    CAP, N = 4096, 12
    lens = sorted((max(64, int(rng.lognormvariate(6.2, 0.9))) for _ in range(N)),
                  reverse=True)
    lens = [min(l, 3500) for l in lens]
    # pad-to-max
    mx = max(lens)
    pad_used, pad_tokens = N * mx, sum(lens)
    # 贪心 first-fit decreasing 装进 CAP
    bins = []
    for l in lens:
        for b in bins:
            if sum(b) + l <= CAP:
                b.append(l)
                break
        else:
            bins.append([l])
    pack_used = len(bins) * CAP
    print(f"{N} 条序列（长尾分布，seed=0），容量 {CAP}：{lens}")
    print(f"pad-to-max({mx})：占用 {pad_used} token 位，利用率 "
          f"{pad_tokens / pad_used * 100:.1f}%")
    print(f"贪心打包 {len(bins)} 行：占用 {pack_used} token 位，利用率 "
          f"{pad_tokens / pack_used * 100:.1f}%")
    for i, b in enumerate(bins):
        print(f"  行 {i}: {b}  共 {sum(b)} token（pad {CAP - sum(b)}）")
    # 一条打包行的 cu_seqlens / position_ids
    row = bins[0]
    cu = [0]
    pos = []
    for l in row:
        cu.append(cu[-1] + l)
        pos.extend(range(l))
    print(f"\n行 0 的 cu_seqlens = {cu}")
    print(f"行 0 的 position_ids 首尾 = {pos[:3]} ... {pos[-3:]}")
    with open(os.path.join(OUT_DIR, "10_stab_packing.json"), "w") as f:
        json.dump({"lens": lens, "bins": bins, "cap": CAP,
                   "pad_used": pad_used, "pack_used": pack_used}, f)


# ---------------------------------------------------------------- resume ----

def cmd_resume():
    print("(a) Megatron 式：位置 = 已消费样本整数")
    total, mbs, dp = 96, 2, 4
    B = mbs * dp
    ids = list(range(total))
    crash_step, resume_at = 7, None
    full = [ids[s * B:(s + 1) * B] for s in range(total // B)]
    run1 = full[:crash_step] + full[crash_step:]
    # 中断在第 7 步后（已消费 7*B=56 个样本），重启从 consumed=56 继续
    consumed = crash_step * B
    run2 = full[:crash_step] + [ids[i:i + B]
                                for i in range(consumed, total, B)]
    same = run1 == run2
    print(f"  total={total}, mbs={mbs}, dp={dp}, B={B}，第 {crash_step} 步后中断")
    print(f"  consumed_train_samples={consumed} → sampler range({consumed},{total})")
    print(f"  恢复后第 1 个 batch = {run2[crash_step]}")
    print(f"  与不中断的第 {crash_step} 步 batch {run1[crash_step]} 相同: {same}")
    print("\n(b) torchdata StatefulDataLoader")
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError:
        print("  torchdata 未安装，跳过 (b)")
        return
    ds = list(range(96))
    dl = StatefulDataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    it = iter(dl)
    for _ in range(crash_step):
        next(it)
    sd = dl.state_dict()
    dl2 = StatefulDataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    dl2.load_state_dict(sd)
    b_new = next(iter(dl2)).tolist()
    print(f"  7 步后 state_dict → 新 loader load_state_dict")
    print(f"  新 loader 第 1 个 batch = {b_new}")
    print(f"  与预期 {list(range(56, 64))} 相同: {b_new == list(range(56, 64))}")


# -------------------------------------------------------------------- main --

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cmds = {"spike": cmd_spike, "clip": cmd_clip, "logit": cmd_logit,
            "bf16": cmd_bf16, "packing": cmd_packing, "resume": cmd_resume}
    if which == "all":
        for name, fn in cmds.items():
            print(f"===== {name} =====")
            fn()
            print()
    else:
        cmds[which]()


if __name__ == "__main__":
    main()
