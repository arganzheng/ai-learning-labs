"""在 CPU 上拟合一条 scaling law（Transformer 与 LLM 10）：训练 5 个尺寸的字符级小 Transformer，
拟合 L(N) = E + A / N^alpha，用它外推第 6 个（最大的）模型的 loss，再实际训练一次对照。
https://arganzheng.life/scaling-laws-and-compute-optimal-training.html

依赖 PyTorch（CPU 即可）。语料：Python 标准库源码（不需要下载）。
    python scaling_law_fit.py            # 约 10 分钟（8 线程笔记本 CPU）
    python scaling_law_fit.py --quick    # 约 1.5 分钟，模型更小、步数更少，趋势仍可见
"""
import argparse
import math
import sys
import sysconfig
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------- 数据：字符级标准库源码 ----------------
def load_corpus(max_bytes=4_000_000):
    root = Path(sysconfig.get_paths()["stdlib"])
    buf = []
    size = 0
    for p in sorted(root.glob("*.py")):
        s = p.read_text(errors="ignore")
        buf.append(s)
        size += len(s)
        if size >= max_bytes:
            break
    text = "".join(buf)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * 0.95)
    return data[:n], data[n:], len(chars)


def batches(data, seq, bs, gen):
    ix = torch.randint(len(data) - seq - 1, (bs,), generator=gen)
    x = torch.stack([data[i : i + seq] for i in ix])
    y = torch.stack([data[i + 1 : i + seq + 1] for i in ix])
    return x, y


# ---------------- 模型：最小的 decoder-only Transformer ----------------
class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, seq):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.register_buffer("mask", torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        for b in self.blocks:
            h = b(h, self.mask[:T, :T])
        return self.head(self.ln(h))

    def non_embedding_params(self):
        """Kaplan 的 N：不含 embedding（与 lm_head 共享）与位置编码。"""
        return sum(p.numel() for p in self.blocks.parameters()) + sum(p.numel() for p in self.ln.parameters())


# ---------------- 训练：每个模型同样的 token 数 ----------------
def train(model, train_data, val_data, steps, seq, bs, lr, gen, log_every, schedule="cosine"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, steps // 20)
    decay = (lambda s: 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))) if schedule == "cosine" else (lambda s: 1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * decay(s))
    curve = []
    for step in range(1, steps + 1):
        x, y = batches(train_data, seq, bs, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            curve.append((step * bs * seq, evaluate(model, val_data, seq, gen)))
    return curve


@torch.no_grad()
def evaluate(model, data, seq, gen, n_batches=8, bs=32):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = batches(data, seq, bs, gen)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / n_batches


# ---------------- 拟合：L = E + A / N^alpha ----------------
def fit_power_law(xs, ys):
    """三参数幂律的最小二乘：对 E 做一维网格搜索，每个 E 下 log(L - E) 对 log N 是线性回归。"""
    xs, ys = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    best = None
    for E in torch.linspace(0.0, ys.min().item() * 0.999, 2000):
        z = torch.log(ys - E)
        lx = torch.log(xs)
        # z = logA - alpha * lx
        X = torch.stack([torch.ones_like(lx), -lx], 1)
        coef = torch.linalg.lstsq(X, z.unsqueeze(1)).solution.squeeze(1)
        pred = E + torch.exp(coef[0]) * xs ** (-coef[1])
        err = ((pred - ys) ** 2).sum().item()
        if best is None or err < best[0]:
            best = (err, E.item(), torch.exp(coef[0]).item(), coef[1].item())
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train_data, val_data, vocab = load_corpus()
    seq, bs = 128, 32
    if args.quick:
        widths, steps = [16, 24, 32, 48, 64, 96], 400
    else:
        widths, steps = [24, 32, 48, 64, 96, 128, 192], 2500
    tokens = steps * bs * seq
    print(f"语料 {len(train_data) / 1e6:.1f}M 字符，vocab {vocab}；每个模型训练 {tokens / 1e6:.1f}M token（{steps} 步 × {bs} × {seq}）\n")

    results = []
    print(f"=== 训练 {len(widths)} 个尺寸（2 层，宽度递增），cosine 调度 ===")
    print(f"{'d':>5}{'N(非embedding)':>16}{'FLOPs(6ND)':>13}{'final loss':>12}{'耗时':>8}")
    for d in widths:
        gen = torch.Generator().manual_seed(1)
        model = TinyGPT(vocab, d, n_layers=2, n_heads=max(1, d // 16), seq=seq)
        N = model.non_embedding_params()
        lr = 1e-3 * (64 / d) ** 0.5  # 宽度越大学习率越小（μP 精神的粗略版）
        t0 = time.time()
        curve = train(model, train_data, val_data, steps, seq, bs, lr, gen, log_every=max(1, steps // 12))
        L = curve[-1][1]
        results.append((d, N, L, curve))
        print(f"{d:>5}{N:>16,}{6 * N * tokens:>13.2e}{L:>12.4f}{time.time() - t0:>7.0f}s")

    # 用前 n-1 个拟合，外推最后一个
    Ns = [r[1] for r in results[:-1]]
    Ls = [r[2] for r in results[:-1]]
    E, A, alpha = fit_power_law(Ns, Ls)
    print(f"\n=== 用前 {len(results) - 1} 个点拟合 L(N) = E + A / N^alpha ===")
    print(f"  E = {E:.3f}   A = {A:.3g}   alpha = {alpha:.3f}")
    print(f"  （Chinchilla 论文对真实 LLM 的拟合：E = 1.69, alpha = 0.34；Kaplan：alpha_N = 0.076，无 E 项。"
          f"字符级小模型的常数不可比，形状可比。）")
    print(f"\n{'d':>5}{'N':>12}{'实测 loss':>11}{'拟合/外推':>11}{'误差':>8}")
    for i, (d, N, L, _) in enumerate(results):
        pred = E + A / N ** alpha
        tag = "  <- 外推（未参与拟合）" if i == len(results) - 1 else ""
        print(f"{d:>5}{N:>12,}{L:>11.4f}{pred:>11.4f}{pred - L:>+8.4f}{tag}")

    # 同一个模型的 loss 随 token 数：L(D) 也是幂律。
    # 用常数学习率重训一个中等模型：cosine 调度下中途的 loss 被"还没退火"抬高，不能当 L(D) 的点——
    # 这正是 Kaplan 与 Chinchilla 结论不同的原因之一。
    d = widths[len(widths) // 2]
    gen = torch.Generator().manual_seed(2)
    model = TinyGPT(vocab, d, n_layers=2, n_heads=max(1, d // 16), seq=seq)
    N = model.non_embedding_params()
    curve = train(model, train_data, val_data, steps, seq, bs, 1e-3 * (64 / d) ** 0.5, gen,
                  log_every=max(1, steps // 24), schedule="constant")
    pts = curve[len(curve) // 4 :]  # 去掉 warmup 与最初的快速下降段
    E2, B, beta = fit_power_law([c[0] for c in pts], [c[1] for c in pts])
    print(f"\n=== 常数学习率下 d={d}（N={N:,}）的 loss 随训练 token 数：拟合 L(D) = E + B / D^beta ===")
    print(f"  E = {E2:.3f}   B = {B:.3g}   beta = {beta:.3f}   （Chinchilla：beta = 0.28）")
    for D, L in curve[:: max(1, len(curve) // 8)]:
        print(f"  {D / 1e6:>6.2f}M token  loss {L:.4f}  拟合 {E2 + B / D ** beta:.4f}")

    print("\n结论：loss 对 N 与对 D 都近似幂律；用小模型的点拟合出的曲线能预测大一档模型的 loss，"
          "这就是 scaling law 实验的全部——真实工作里 N 跨 3 个数量级、每个点是一次几十到几千 GPU 小时的训练。")


if __name__ == "__main__":
    main()
