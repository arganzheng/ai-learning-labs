"""算法工程师的数学（07）案例：在一个真模型上做最小的策略梯度。
https://arganzheng.life/derivatives-gradients-chain-rule-and-policy-gradient.html

策略 = 《Transformer 与 LLM》第四篇训好的字符级莎士比亚 nanoGPT（0.8M 参数）；
奖励 = 生成的 63 个字符里元音（a e i o u）占字母的比例——一个可验证、不需要标签的分数；
算法 = REINFORCE + 组内均值 baseline（GRPO 的骨架，没有裁剪），可选一项对参考模型的 KL 惩罚。
看两件事：奖励能不能涨；涨的同时模型在真实文本上的 val loss 变差了多少（RL 为什么需要 KL 约束）。

    python 07_rl_on_nanogpt.py            # 两个 run（β = 0 与 β = 0.1），各 80 步，MPS 约 2 分钟 / CPU 约 6 分钟
    python 07_rl_on_nanogpt.py --quick    # 各 20 步
"""
import copy
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _plot import C, plt, save

HERE = os.path.dirname(os.path.abspath(__file__))
TLL = os.path.join(HERE, "..", "transformer-and-llm")
sys.path.insert(0, TLL)
from nanogpt_model import GPT, GPTConfig  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
QUICK = "--quick" in sys.argv
STEPS = 20 if QUICK else 80
G, T = 16, 63                     # 每步采 16 条、每条 63 个字符（block_size 64 减去 1 个提示符）
LR = 1e-4


def load():
    ck = torch.load(os.path.join(TLL, "nanogpt", "out-shakespeare-char-base", "ckpt.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ck["model_args"]))
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    meta = pickle.load(open(os.path.join(TLL, "nanogpt", "data", "shakespeare_char", "meta.pkl"), "rb"))
    return model.to(DEV), meta


def reward_fn(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.lower() in "aeiou" for c in letters) / max(len(letters), 1)


@torch.no_grad()
def val_loss(model, batch):
    x, y = batch
    _, loss = model(x, y)
    return loss.item()


def seq_logprobs(model, seqs, n_prompt):
    """每条序列生成部分的逐 token log π，形状 [G, T]"""
    logits, _ = model(seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous())        # 传 targets 只是为了拿到全部位置的 logits
    logp = F.log_softmax(logits.float(), -1)
    lp = logp.gather(-1, seqs[:, 1:, None]).squeeze(-1)  # [G, L-1]：位置 t 预测 token t+1
    return lp[:, n_prompt - 1:]                           # 只保留生成的 T 个 token


def run(beta, log_every=10):
    torch.manual_seed(0)
    model, meta = load()
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    itos = meta["itos"]
    decode = lambda ids: "".join(itos[i] for i in ids)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    val = np.memmap(os.path.join(TLL, "nanogpt", "data", "shakespeare_char", "val.bin"), dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(1)
    ix = torch.randint(len(val) - 65, (32,), generator=g)
    vb = (torch.stack([torch.from_numpy(val[i:i + 64].astype(np.int64)) for i in ix]).to(DEV),
          torch.stack([torch.from_numpy(val[i + 1:i + 65].astype(np.int64)) for i in ix]).to(DEV))
    prompt = torch.full((G, 1), meta["stoi"]["\n"], dtype=torch.long, device=DEV)
    hist = {"step": [], "reward": [], "val": [], "kl": []}
    print(f"\n--- β = {beta}（KL 惩罚系数）---")
    print(f"  {'step':>4} {'mean R':>7} {'val loss':>9} {'KL/token':>9}  样本")
    for step in range(STEPS + 1):
        model.eval()
        with torch.no_grad():
            seqs = model.generate(prompt, T, temperature=1.0)          # [G, 1+T]
        texts = [decode(s[1:].tolist()) for s in seqs]
        R = torch.tensor([reward_fn(t) for t in texts], device=DEV)
        model.train()
        lp = seq_logprobs(model, seqs, 1)                                # [G, T]，带梯度
        with torch.no_grad():
            lp_ref = seq_logprobs(ref, seqs, 1)
        kl_tok = (lp.detach() - lp_ref).mean(1)                          # 每条的 KL 估计（每 token）
        if step % log_every == 0 or step == STEPS:
            vl = val_loss(model.eval(), vb)
            hist["step"].append(step); hist["reward"].append(R.mean().item()); hist["val"].append(vl); hist["kl"].append(kl_tok.mean().item())
            print(f"  {step:>4} {R.mean().item():>7.3f} {vl:>9.3f} {kl_tok.mean().item():>9.3f}  {texts[0][:48]!r}")
        if step == STEPS:
            break
        R_shaped = R - beta * kl_tok                                     # KL 惩罚进奖励
        adv = (R_shaped - R_shaped.mean()) / (R_shaped.std() + 1e-6)     # 组内标准化：GRPO 的优势
        loss = -(adv[:, None] * lp).mean()                               # 最大化 E[A · log π]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return hist, texts


if __name__ == "__main__":
    model, _ = load()
    print(f"设备 {DEV}；策略 = 莎士比亚 nanoGPT {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数；每步 {G} 条 × {T} 字符；奖励 = 元音占字母的比例")
    print(f"参考：英文文本的元音比例约 0.38；莎士比亚原文样本约 {reward_fn(open(os.path.join(TLL, 'nanogpt', 'data', 'shakespeare_char', 'input.txt')).read()[:20000]):.3f}")
    runs = {0.0: run(0.0), 0.5: run(0.5)}
    for beta, (h, texts) in runs.items():
        print(f"\nβ = {beta}：奖励 {h['reward'][0]:.3f} → {h['reward'][-1]:.3f}，val loss {h['val'][0]:.3f} → {h['val'][-1]:.3f}，KL/token {h['kl'][-1]:.3f}")
        print("  最后一步的三条样本：")
        for t in texts[:3]:
            print("   ", repr(t))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for (beta, (h, _)), col in zip(runs.items(), (C["red"], C["blue"])):
        axes[0].plot(h["step"], h["reward"], "o-", color=col, ms=3, label=f"β = {beta}")
        axes[1].plot(h["step"], h["val"], "o-", color=col, ms=3, label=f"β = {beta}")
    axes[0].axhline(runs[0.0][0]["reward"][0], color=C["gray"], ls=":", lw=1)
    axes[0].set(xlabel="RL 步", ylabel="平均奖励（元音比例）", title="奖励：两个 run 都在涨")
    axes[1].set(xlabel="RL 步", ylabel="val loss（真实莎士比亚文本）", title="代价：不加 KL 惩罚，模型很快忘了怎么写英文")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    save(fig, "07-rl-nanogpt")
