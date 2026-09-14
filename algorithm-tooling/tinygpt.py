"""第二、五篇共用的字符级小 Transformer 与语料。

语料是 Python 标准库的源码（本机就有，不用下载），字符级 tokenizer，
模型是一个 4 层、d=128 的 decoder-only Transformer，CPU 上几十秒能看到 loss 下降。
"""
import math
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 128        # ASCII
    d: int = 128
    heads: int = 4
    layers: int = 4
    seq: int = 128
    lr: float = 3e-4
    batch: int = 32
    steps: int = 300
    warmup: int = 30
    seed: int = 0


def load_corpus(max_chars=2_000_000):
    """标准库 .py 文件拼成一个字符串，非 ASCII 字符丢掉。"""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    buf = []
    n = 0
    for p in sorted(stdlib.glob("*.py")):
        t = p.read_text(errors="ignore").encode("ascii", "ignore").decode()
        buf.append(t)
        n += len(t)
        if n >= max_chars:
            break
    text = "".join(buf)[:max_chars]
    ids = torch.tensor([ord(c) for c in text], dtype=torch.long)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:]


def get_batch(data, cfg, gen):
    ix = torch.randint(len(data) - cfg.seq - 1, (cfg.batch,), generator=gen)
    x = torch.stack([data[i : i + cfg.seq] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.seq + 1] for i in ix])
    return x, y


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.mlp = nn.Sequential(nn.Linear(cfg.d, 4 * cfg.d), nn.GELU(), nn.Linear(4 * cfg.d, cfg.d))
        self.heads = cfg.heads

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        # [B, T, D] -> [B, T, h, D/h] -> [B, h, T, D/h]：第一篇讲的多头形状变换
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.seq, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))          # logits [B, T, vocab]


def cosine_with_warmup(step, cfg):
    if step < cfg.warmup:
        return step / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
