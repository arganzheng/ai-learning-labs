"""训练配方与稳定性实验（Transformer 与 LLM 12）：在 CPU 上的字符级小 Transformer 上复现四件事——
学习率调度（cosine / WSD / 常数）、batch 与最优学习率的关系、loss spike 的诱发与 QK-norm 的作用、z-loss 对 logit 漂移的抑制。
https://arganzheng.life/pretraining-recipe-and-training-stability.html

依赖 PyTorch（CPU）。语料：Python 标准库源码。
    python training_recipe_lab.py                 # 全部四个实验，约 12 分钟
    python training_recipe_lab.py schedule spike  # 只跑指定的
    python training_recipe_lab.py --quick         # 每个实验缩短，约 2 分钟
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling_law_fit import load_corpus, batches, evaluate

torch.manual_seed(0)
SEQ, VOCAB_D = 128, None


# ---------------- 自己写的 attention：能加 QK-norm、能读出 attention logit 的最大值 ----------------
class Attention(nn.Module):
    def __init__(self, d, n_heads, qk_norm=False):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.qn, self.kn = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.max_logit = 0.0

    def forward(self, x, mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        self.max_logit = logits.detach().abs().max().item()
        logits = logits.masked_fill(mask, float("-inf"))
        return self.out((logits.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d, n_heads, qk_norm):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, qk_norm)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=64, n_layers=2, n_heads=4, qk_norm=False):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, qk_norm) for _ in range(n_layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.register_buffer("mask", torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        for b in self.blocks:
            h = b(h, self.mask[:T, :T])
        return self.head(self.ln(h))

    def max_attn_logit(self):
        return max(b.attn.max_logit for b in self.blocks)


# ---------------- 调度 ----------------
def make_schedule(kind, steps, warmup):
    def f(s):
        w = min(1.0, (s + 1) / warmup)
        if kind == "cosine":
            return w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))   # 衰减到 10%
        if kind == "wsd":
            decay_start = int(steps * 0.8)                                                # 最后 20% 线性衰减到 0
            return w * (1.0 if s < decay_start else max(0.0, 1 - (s - decay_start) / (steps - decay_start)))
        return w                                                                          # constant
    return f


def train(model, data, val, steps, bs, lr, sched="cosine", warmup=None, clip=1.0, z_loss=0.0,
          log_every=None, seed=1, eval_every=None):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    lr_f = make_schedule(sched, steps, warmup or max(1, steps // 20))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_f)
    log = []
    for step in range(1, steps + 1):
        x, y = batches(data, SEQ, bs, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        total = loss
        if z_loss:
            log_z = torch.logsumexp(logits, -1)                 # z-loss：把 log(Z) 往 0 拉
            total = loss + z_loss * (log_z ** 2).mean()
        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf")).item()
        opt.step()
        scheduler.step()
        if log_every and (step % log_every == 0 or step == 1):
            log.append(dict(step=step, loss=loss.item(), gnorm=gnorm, lr=scheduler.get_last_lr()[0],
                            max_logit=model.max_attn_logit(),
                            log_z=torch.logsumexp(logits.detach(), -1).abs().mean().item(),
                            val=evaluate(model, val, SEQ, gen) if eval_every and step % eval_every == 0 else None))
    return log


# ---------------- 实验 1：调度 ----------------
def exp_schedule(data, val, vocab, steps):
    print(f"=== 实验 1：三种学习率调度，同样的 {steps} 步（d=64，batch 32，峰值 lr 2e-3）===")
    print("  cosine：衰减到 10%；WSD：前 80% 常数、后 20% 线性到 0；constant：不衰减")
    res = {}
    for kind in ["cosine", "wsd", "constant"]:
        torch.manual_seed(0)
        m = GPT(vocab)
        log = train(m, data, val, steps, 32, 2e-3, sched=kind, log_every=steps // 10, eval_every=steps // 10)
        res[kind] = log
        curve = "  ".join(f"{l['step']}:{l['val']:.3f}" for l in log if l["val"] is not None)
        print(f"  {kind:<9} 验证 loss 随步数  {curve}")
    print("  看什么：WSD 在 80% 处之前与 constant 完全相同（同一条轨迹），最后 20% 的衰减把它拉到 cosine 的水平或更低；"
          "cosine 早早开始降 lr，中途的 loss 不代表'这么多 token 能训到多好'。步数太少时（--quick）衰减的收益还没显出来。\n")


# ---------------- 实验 2：batch 与最优 lr ----------------
def exp_batch_lr(data, val, vocab, steps_at_bs32):
    print("=== 实验 2：不同 batch 下的最优学习率（同样的 token 总量，d=48）===")
    tokens = steps_at_bs32 * 32 * SEQ
    lrs = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]
    print(f"  {'batch':>6}{'steps':>7}  " + "".join(f"lr={lr:.0e}".rjust(10) for lr in lrs) + "   best lr")
    for bs in [8, 16, 32, 64]:
        steps = tokens // (bs * SEQ)
        row = []
        for lr in lrs:
            torch.manual_seed(0)
            m = GPT(vocab, d=48, n_heads=3)
            train(m, data, val, steps, bs, lr, sched="cosine")
            row.append(evaluate(m, val, SEQ, torch.Generator().manual_seed(9)))
        best = lrs[min(range(len(lrs)), key=lambda i: row[i])]
        print(f"  {bs:>6}{steps:>7}  " + "".join(f"{v:>10.3f}" for v in row) + f"   {best:.0e}")
    print("  看什么：每行的最低点向右移——batch 越大最优 lr 越大；同样 token 数下大 batch 的最好 loss 更差，因为更新次数少了。"
          "真实配方里 batch 随训练增大（Llama 3 405B：4M → 8M → 16M token）是在小 batch 效率高、大 batch 并行度高之间折中。\n")


# ---------------- 实验 3：attention logit 增长与 QK-norm（Wortsman 等 2023 的"lr 敏感度"实验的迷你版）----------------
def exp_spike(data, val, vocab, steps):
    print(f"=== 实验 3：学习率从 2e-3 提到 1e-1，有无 QK-norm 的对照（d=128，4 层，常数 lr，warmup 10 步，不裁剪，{steps} 步）===")
    lrs = [2e-3, 8e-3, 3e-2, 1e-1]
    print(f"  {'':<12}" + "".join(f"lr={lr:.0e}".rjust(26) for lr in lrs))
    for qk in [False, True]:
        cells = []
        for lr in lrs:
            torch.manual_seed(0)
            m = GPT(vocab, d=128, n_layers=4, n_heads=4, qk_norm=qk)
            log = train(m, data, val, steps, 32, lr, sched="constant", warmup=10, clip=0, log_every=max(1, steps // 15))
            losses = [l["loss"] for l in log]
            spikes = sum(1 for a, b in zip(losses, losses[1:]) if b > a + 0.5)
            cells.append(f"loss {losses[-1]:.2f}  logit {max(l['max_logit'] for l in log):>6.0f}" + (f" 尖峰{spikes}" if spikes else ""))
        print(f"  {'QK-norm' if qk else '无 QK-norm':<12}" + "".join(c.rjust(26) for c in cells))
    print("  看什么：无 QK-norm 时 attention logit 随 lr 涨到几千、上万（softmax 饱和成 one-hot，这一头的梯度归零），"
          "末 loss 随 lr 上升——训练'能跑但变差'，这正是大模型 loss spike 前的状态；QK-norm 把 logit 钉在 O(10)，"
          "loss 对 lr 的敏感度大幅下降。fp32 的两百万参数模型不会真的发散，spike 本身要在低精度、大 batch、深层网络上才容易复现。\n")


# ---------------- 实验 4：z-loss ----------------
def exp_zloss(data, val, vocab, steps):
    print(f"=== 实验 4：z-loss 抑制输出 logit 的漂移（lr 4e-3，{steps} 步）===")
    for name, z in [("无 z-loss", 0.0), ("z-loss 1e-4（PaLM 的系数）", 1e-4), ("z-loss 1e-2（夸张，看效果）", 1e-2)]:
        torch.manual_seed(0)
        m = GPT(vocab)
        log = train(m, data, val, steps, 32, 4e-3, z_loss=z, log_every=max(1, steps // 6), eval_every=steps)
        traj = " ".join(f"{l['log_z']:.1f}" for l in log)
        print(f"  {name:<26} 末 val loss {log[-1]['val']:.3f}   |log Z| 轨迹: {traj}")
    print("  观察：|log Z|（logsumexp 的绝对值）是 softmax 归一化常数的对数，它自由漂移不改变 loss，却让 logits 整体变大、"
          "低精度下更容易溢出；z-loss 以极小代价把它按在 0 附近。1e-4 对 loss 几乎无影响，这是它能默认开启的原因。\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", choices=["schedule", "batch_lr", "spike", "zloss"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.experiments = args.experiments or ["schedule", "batch_lr", "spike", "zloss"]
    data, val, vocab = load_corpus()
    scale = 0.2 if args.quick else 1.0
    t0 = time.time()
    if "schedule" in args.experiments:
        exp_schedule(data, val, vocab, int(1500 * scale))
    if "batch_lr" in args.experiments:
        exp_batch_lr(data, val, vocab, int(500 * scale))
    if "spike" in args.experiments:
        exp_spike(data, val, vocab, int(300 * scale))
    if "zloss" in args.experiments:
        exp_zloss(data, val, vocab, int(600 * scale))
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
