"""LoRA 专题（01）：低秩假设——梯度手算、四本账、初始化、全量微调的 ΔW 到底有多"低秩"。
https://arganzheng.life/lora-low-rank-hypothesis-gradients-and-accounts.html

    python 01_low_rank.py                    # 全部：hand account speed init spectrum
    python 01_low_rank.py hand account       # 不加载大模型的两个，几秒
    python 01_low_rank.py spectrum           # 需要先跑 02_knobs.py full（读 out/ckpt/full_linear.pt）
    python 01_low_rank.py init --quick       # 训练类实验缩到几步
"""
import argparse
import copy
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model

import lolab

ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------- 实验 1：一个 2×3 的手算 ----------------
def exp_hand():
    lolab.banner("实验 1：手算一步 LoRA 的梯度（W 2×3 冻结，r=1）")
    torch.manual_seed(0)
    W = torch.tensor([[1., 0., 2.], [0., 1., 0.]])                  # d_out=2, d_in=3，冻结
    A = torch.tensor([[1., -1., 0.]], requires_grad=True)          # r × d_in = 1 × 3
    B = torch.tensor([[0.], [0.]], requires_grad=True)             # d_out × r = 2 × 1，初始为 0
    x = torch.tensor([2., 1., 1.])
    target = torch.tensor([3., 0.])
    alpha, r = 2.0, 1
    s = alpha / r
    y = W @ x + s * (B @ (A @ x))
    loss = 0.5 * ((y - target) ** 2).sum()
    loss.backward()
    G = (y - target).detach()                                       # ∂L/∂y
    print(f"  x = {x.tolist()}，Wx = {(W @ x).tolist()}，B=0 所以 y = Wx，loss = {loss.item():.1f}")
    print(f"  G = ∂L/∂y = y − target = {G.tolist()}，Ax = {(A @ x).item():.0f}")
    print(f"  公式 ∂L/∂B = s·G·(Ax)ᵀ = {(s * G[:, None] * (A @ x).detach()).flatten().tolist()}   autograd B.grad = {B.grad.flatten().tolist()}")
    print(f"  公式 ∂L/∂A = s·Bᵀ·G·xᵀ = {(s * (B.detach().T @ G[:, None]) * x[None, :]).flatten().tolist()}   autograd A.grad = {A.grad.flatten().tolist()}  ← B=0 时 A 的梯度全 0")
    print(f"  W.requires_grad = {W.requires_grad}，W.grad = {W.grad}  ← 冻结的 W 没有梯度，也没人算 ∂L/∂W")
    # 第二步：B 动了之后 A 才有梯度
    with torch.no_grad():
        B -= 0.1 * B.grad
    A.grad = None; B.grad = None
    y = W @ x + s * (B @ (A @ x))
    loss = 0.5 * ((y - target) ** 2).sum()
    loss.backward()
    print(f"  一步 SGD（lr 0.1）后 B = {B.detach().flatten().tolist()}，y = {y.detach().tolist()}，loss = {loss.item():.4f}")
    print(f"  这时 ∂L/∂A = {A.grad.flatten().tolist()}（不再是 0），∂L/∂B = {B.grad.flatten().tolist()}")
    dW = s * (B.detach() @ A.detach())
    print(f"  等效的 ΔW = s·BA =\n{dW.numpy().round(3)}\n  两行成比例（秩 1）：第 2 行 / 第 1 行 = {(dW[1, 0] / dW[0, 0]).item():.3f}\n")


# ---------------- 实验 2：四本账 ----------------
def shapes_from_config(cfg):
    h, i, kv = cfg.hidden_size, cfg.intermediate_size, cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    return {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
            "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}


def exp_account():
    lolab.banner("实验 2：参数、训练状态、FLOPs 三本账（按 config.json 算，不加载权重）")
    from transformers import AutoConfig
    for name, total_b in [("Qwen/Qwen2.5-0.5B", None), ("meta-llama/Llama-3.1-8B", 8.03e9)]:
        try:
            cfg = AutoConfig.from_pretrained(name)
        except Exception:
            from transformers import LlamaConfig
            cfg = LlamaConfig(hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8, num_hidden_layers=32, vocab_size=128256)
        sh = shapes_from_config(cfg)
        L = cfg.num_hidden_layers
        lin = sum(i * o for i, o in sh.values()) * L
        emb = cfg.vocab_size * cfg.hidden_size * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)
        total = total_b or (lin + emb + L * 2 * cfg.hidden_size + cfg.hidden_size)
        print(f"  {name}：{L} 层，hidden {cfg.hidden_size}，中间层 {cfg.intermediate_size}，KV 维 {sh['k_proj'][1]}；线性层参数 {lin / 1e6:.0f}M，总参数约 {total / 1e6:.0f}M")
        print(f"    {'矩阵':<10}{'形状 in×out':>14}{'参数':>9}{'r=16 LoRA':>11}{'占比':>8}")
        for k, (i, o) in sh.items():
            print(f"    {k:<10}{f'{i}×{o}':>14}{i * o / 1e6:>8.1f}M{16 * (i + o) / 1e3:>9.0f}K{16 * (i + o) / (i * o):>8.2%}")
        for r in (4, 16, 64):
            attn = lolab.lora_params([sh[k] for k in ALL[:4]], r) * L
            allp = lolab.lora_params(list(sh.values()), r) * L
            print(f"    r={r:<3} attention 四个矩阵：{attn / 1e6:>6.2f}M（{attn / total:.2%}）   全部七个：{allp / 1e6:>6.2f}M（{allp / total:.2%}）"
                  f"   训练状态（全部七个）：{lolab.training_state_bytes(allp, total) / 2**30:.2f} GiB   全量：{lolab.training_state_bytes(total, total) / 2**30:.1f} GiB")
        r = 16
        extra = sum(2 * r * (i + o) for i, o in sh.values())
        base = sum(2 * i * o for i, o in sh.values())
        print(f"    r=16 每 token 每层多出的前向 FLOPs：{extra / 1e6:.2f}M / {base / 1e6:.0f}M = {extra / base:.2%}；每层线性层 kernel 数从 7 变成 7 + 14（每个矩阵多两次瘦 GEMM 和一次加）")
    print("  激活值那本账不在这里：它由 batch × 序列长 × hidden × 层数决定，与可训练参数无关，LoRA 一分不省（实验 3 只看得到时间与算子数，激活值的账在文章正文里算）。\n")


# ---------------- 实验 3：每步耗时与 kernel 数 ----------------
def exp_speed(tok, base):
    lolab.banner("实验 3：全量 vs LoRA 的一步耗时、优化器状态与算子数（batch 4 × 256，CPU/单卡）")
    x = tok(["hello world " * 120] * 4, return_tensors="pt", truncation=True, max_length=256).input_ids.to(base.device)
    rows = [("全量", None), ("LoRA r=16 全部线性层", LoraConfig(r=16, lora_alpha=32, target_modules=ALL, task_type="CAUSAL_LM"))]
    for name, cfg in rows:
        m = copy.deepcopy(base) if cfg is None else get_peft_model(copy.deepcopy(base), cfg)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-5)
        def step():
            opt.zero_grad(set_to_none=True)
            out = m(input_ids=x, labels=x)
            out.loss.backward()
            opt.step()
        step()                                                        # 预热（建优化器状态）
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with torch.no_grad():
                m(input_ids=x)
        n_ops = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
        n_mm = sum(e.count for e in prof.key_averages() if e.key in ("aten::mm", "aten::addmm", "aten::matmul", "aten::bmm", "aten::linear"))
        ts = []
        for _ in range(3):
            t = time.time(); step(); ts.append(time.time() - t)
        fwd_t = time.time()
        with torch.no_grad():
            m(input_ids=x)
        fwd_t = time.time() - fwd_t
        opt_bytes = sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
        grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
        print(f"  {name:<22} 一步 {min(ts):.2f} s（其中前向 {fwd_t:.2f} s）  前向 aten 算子 {n_ops} 个（矩阵乘 {n_mm}）  梯度 {grad_bytes / 2**20:.0f} MB  Adam 状态 {opt_bytes / 2**20:.0f} MB")
        m = opt = None
    print("  看什么：LoRA 省的是梯度与 Adam 状态（8.8M × 12 B 对 494M × 12 B），前向反而多了 168 × 2 个小矩阵乘；"
          "反向里对输入的梯度要穿过每一个冻结的 W，这部分算量一分不少。所以 LoRA 一步未必更快，尤其在算力受限的 CPU 上。\n")


# ---------------- 实验 4：初始化 ----------------
def exp_init(tok, base, train, evals, steps):
    lolab.banner(f"实验 4：A、B 各怎么初始化——{steps} 步的 loss 与训练前的输出变化")
    before = lolab.completion_loss(base, tok, evals)
    print(f"  训练前 base 的验证回复 loss {before:.4f}")
    variants = ["B=0, A 随机（默认）", "A=0, B 随机", "A、B 都随机", "A、B 都为 0", "gaussian（A 高斯，B=0）"]
    for name in variants:
        lolab.seed(0)
        init = {"A、B 都随机": False, "gaussian（A 高斯，B=0）": "gaussian"}.get(name, True)
        m = get_peft_model(copy.deepcopy(base), LoraConfig(r=16, lora_alpha=32, target_modules=ALL, task_type="CAUSAL_LM", init_lora_weights=init))
        with torch.no_grad():
            for mod in m.modules():
                if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                    a, b = mod.lora_A["default"].weight, mod.lora_B["default"].weight
                    if name == "A=0, B 随机":
                        torch.nn.init.kaiming_uniform_(b, a=math.sqrt(5)); a.zero_()
                    elif name == "A、B 都为 0":
                        a.zero_(); b.zero_()
        start = lolab.completion_loss(m, tok, evals)
        losses, _ = lolab.run_sft(tok, m, train, steps, 1e-4, "init")
        after = lolab.completion_loss(m, tok, evals)
        # 训练后 A、B 各动了多少
        da = db = 0.0
        for mod in m.modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                da += mod.lora_A["default"].weight.norm().item() ** 2
                db += mod.lora_B["default"].weight.norm().item() ** 2
        print(f"  {name:<22} 训练前 loss {start:.4f}（与 base 差 {start - before:+.4f}）  训练 loss {losses[0][1]:.3f} → {losses[-1][1]:.3f}  训练后验证 {after:.4f}  ‖A‖={math.sqrt(da):.1f} ‖B‖={math.sqrt(db):.2f}")
        del m
    print("  看什么：只要 A、B 有一个是 0，训练前模型就与 base 完全相同；都随机则一开始就把模型改坏了；都为 0 则梯度全 0、永远学不动。"
          "默认 B=0：第一步只有 B 动，A 的梯度随 B 长大才出现。\n")


# ---------------- 实验 5：全量微调的 ΔW 有多低秩 ----------------
def exp_spectrum(tok, base, evals, texts):
    lolab.banner("实验 5：全量微调 80 步的 ΔW——奇异值谱，以及把它截到秩 r 后还剩多少效果")
    path = os.path.join(lolab.OUT, "ckpt", "full_linear.pt")
    if not os.path.exists(path):
        print("  需要先运行 02_knobs.py full 得到 out/ckpt/full_linear.pt"); return
    full = torch.load(path)
    sd = base.state_dict()
    picks = [f"model.layers.{l}.{k}.weight" for l in (0, 12, 23) for k in ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")]
    ranks = [1, 4, 16, 64, 256]
    print(f"  {'矩阵':<34}{'形状':>11}{'‖ΔW‖/‖W‖':>10}" + "".join(f"{f'秩{r}':>8}" for r in ranks) + "   ← 前 r 个奇异值占能量的比例")
    spectra = {}
    for k in picks:
        dW = (full[k] - sd[k]).float()
        s = torch.linalg.svdvals(dW)
        e = (s ** 2).cumsum(0) / (s ** 2).sum()
        spectra[k] = e.tolist()
        rel = (dW.norm() / sd[k].norm()).item()
        print(f"  {k.replace('model.layers.', 'L').replace('self_attn.', '').replace('mlp.', ''):<34}{f'{dW.shape[0]}×{dW.shape[1]}':>11}{rel:>10.4f}" + "".join(f"{e[min(r, len(e)) - 1].item():>8.1%}" for r in ranks))
    # 把全部线性层的 ΔW 截到秩 r，装回 base，看效果剩多少
    print(f"\n  把全部 {len(full)} 个线性层的 ΔW 各截到秩 r 后装回模型：")
    base_val, base_txt = lolab.completion_loss(base, tok, evals), lolab.text_loss(base, tok, texts)
    print(f"  {'配置':<20}{'验证回复 loss':>14}{'普通文本 loss':>14}")
    print(f"  {'训练前':<20}{base_val:>14.4f}{base_txt:>14.4f}")
    res = {"ranks": {}, "base": [base_val, base_txt]}
    for r in [1, 4, 16, 64, None]:
        m = copy.deepcopy(base)
        msd = m.state_dict()
        with torch.no_grad():
            for k, w in full.items():
                dW = (w - sd[k]).float()
                if r is not None:
                    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
                    dW = (U[:, :r] * S[:r]) @ Vh[:r]
                msd[k].copy_(sd[k] + dW)
        v, t = lolab.completion_loss(m, tok, evals), lolab.text_loss(m, tok, texts)
        label = "全量 ΔW（不截）" if r is None else f"ΔW 截到秩 {r}"
        res["ranks"][str(r)] = [v, t]
        print(f"  {label:<20}{v:>14.4f}{t:>14.4f}")
        del m
    lolab.save_json("spectrum.json", {"cum_energy": spectra, "trunc": res})
    print("  看什么：全量微调学到的 ΔW 本身并不低秩（前 16 个奇异值只占一小部分能量），但把它截到秩 16 装回去，"
          "效果掉得很少——SFT 学格式所需的改变确实只在少数方向上，剩下的高秩部分大多是噪声。这正是 LoRA 赌对了的地方。\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", choices=["hand", "account", "speed", "init", "spectrum"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    exps = args.experiments or ["hand", "account", "speed", "init", "spectrum"]
    steps = 4 if args.quick else 20
    lolab.seed(0)
    if "hand" in exps:
        exp_hand()
    if "account" in exps:
        exp_account()
    if not set(exps) & {"speed", "init", "spectrum"}:
        return
    print(f"设备 {lolab.device()}，torch 线程 {torch.get_num_threads()}")
    tok, base = lolab.load()
    if "speed" in exps:
        exp_speed(tok, base)
    if "init" in exps or "spectrum" in exps:
        train, evals = lolab.load_data(64, 10) if args.quick else lolab.load_data(400, 100)
    if "init" in exps:
        exp_init(tok, base, train, evals, steps)
    if "spectrum" in exps:
        exp_spectrum(tok, base, evals, lolab.load_plain_text(6 if args.quick else 60))


if __name__ == "__main__":
    main()
