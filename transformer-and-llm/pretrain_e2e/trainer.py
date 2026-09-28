"""step6 / step7 共用的训练循环：nanoGPT 的模型 + 一个会记录「七条曲线」的训练器。

曲线：train loss、val loss、学习率、梯度范数、参数范数、attention logit 最大值、吞吐（token/s）。
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nanogpt_model import GPT, GPTConfig  # noqa: E402

from common import DATA  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = 4096
BLOCK = 256


class Data:
    def __init__(self, block=BLOCK):
        self.train = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
        self.block = block
        self.g = torch.Generator().manual_seed(0)
        self.pos = 0                                         # 顺序扫过训练流（按 epoch），不放回

    def batch(self, split, bs):
        d = self.train if split == "train" else self.val
        if split == "train":
            n_seq = (len(d) - 1) // self.block
            if self.pos + bs > n_seq:
                self.pos = 0
            starts = torch.arange(self.pos, self.pos + bs) * self.block
            self.pos += bs
        else:
            starts = torch.randint(len(d) - self.block - 1, (bs,), generator=self.g)
        x = torch.stack([torch.from_numpy(d[s:s + self.block].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(d[s + 1:s + 1 + self.block].astype(np.int64)) for s in starts])
        return x.to(DEV), y.to(DEV)

    @property
    def n_train_tokens(self):
        return len(self.train)


def make_model(n_layer, n_embd, n_head=None, block=BLOCK, vocab=VOCAB):
    n_head = n_head or max(2, n_embd // 64)
    m = GPT(GPTConfig(block_size=block, vocab_size=vocab, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=0.0, bias=False))
    return m.to(DEV)


def n_params(model, non_embedding=True):
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    return n


def lr_at(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def max_attn_logit(model, x):
    """attention logit 最大值：从每层 c_attn 的输出里取 q、k，算 max |q·k| / sqrt(d_head)（只在记录步算）。"""
    cfg = model.config
    hs = cfg.n_embd // cfg.n_head
    tok_emb = model.transformer.wte(x); pos = torch.arange(x.size(1), device=x.device)
    h = tok_emb + model.transformer.wpe(pos)
    worst = 0.0
    for blk in model.transformer.h:
        a = blk.attn.c_attn(blk.ln_1(h))
        q, k, _ = a.split(cfg.n_embd, dim=2)
        B, T, _ = q.shape
        q = q.view(B, T, cfg.n_head, hs).transpose(1, 2); k = k.view(B, T, cfg.n_head, hs).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        worst = max(worst, s.abs().max().item())
        h = blk(h)
    return worst


@torch.no_grad()
def eval_loss(model, data, split, batches=8, bs=32):
    model.eval()
    ls = [model(*data.batch(split, bs))[1].item() for _ in range(batches)]
    model.train()
    return float(np.mean(ls))


def train(model, data, steps, bs, peak_lr, warmup, log_every=50, wd=0.1, betas=(0.9, 0.95), clip=1.0, verbose=True, eval_batches=8):
    opt = model.configure_optimizers(wd, peak_lr, betas, "cuda" if DEV == "cuda" else "cpu")
    hist = {k: [] for k in ("step", "tokens", "train", "val", "lr", "gnorm", "pnorm", "attn", "tps")}
    t0 = time.time(); tokens = 0; last_t = time.time(); last_tokens = 0
    model.train()
    for step in range(steps + 1):
        lr = lr_at(step, steps, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.batch("train", bs)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        opt.step()
        tokens += x.numel()
        if step % log_every == 0 or step == steps:
            vl = eval_loss(model, data, "val", eval_batches)
            pnorm = math.sqrt(sum((p.float() ** 2).sum().item() for p in model.parameters()))
            attn = max_attn_logit(model, x[:4])
            now = time.time(); tps = (tokens - last_tokens) / max(1e-6, now - last_t); last_t, last_tokens = now, tokens
            for k, v in zip(hist, (step, tokens, loss.item(), vl, lr, gnorm, pnorm, attn, tps)):
                hist[k].append(v)
            if verbose:
                print(f"    step {step:>5}  tok {tokens/1e6:6.2f}M  train {loss.item():.3f}  val {vl:.3f}  lr {lr:.2e}  |g| {gnorm:5.2f}  |θ| {pnorm:6.1f}  max qk {attn:5.1f}  {tps/1e3:5.1f}K tok/s  {now-t0:5.0f}s")
    return hist, opt
