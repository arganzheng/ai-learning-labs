"""mtp_nanogpt.py — 文章 09《MTP：改训练目标而不改主干》的实验。

在 nanoGPT 的 GPT 上挂一个 DeepSeek-V3 式的 MTP 模块（D=1：每个位置额外预测「下下个」token），
在 shakespeare_char 上与不带 MTP 的同结构模型各训同样步数，比较：
  1. 主头（next-token）的 val loss——MTP 的额外监督信号有没有帮到主干
  2. MTP 头对 t+2 的 val loss 与 top-1 命中率——它作为投机解码 draft 有多准
  3. 参数量与每步耗时的代价

运行（先 cd nanogpt && python data/shakespeare_char/prepare.py）：
  python mtp_nanogpt.py [--iters 1000] [--device mps|cpu]
依赖：torch, numpy；模型定义来自 nanogpt_model.py
"""
import argparse, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from nanogpt_model import GPT, GPTConfig, Block, LayerNorm

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--lam", type=float, default=0.3, help="MTP loss 权重 λ（DeepSeek-V3 前期 0.3）")
args = ap.parse_args()
torch.manual_seed(1337)
DATA = os.path.join(os.path.dirname(__file__), "nanogpt", "data", "shakespeare_char")
train_data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
B, T, V = 12, 64, 65

def get_batch(split, k_extra=1):
    """返回 x [B,T] 与 y [B,T+k_extra]：y[:, j] 是 x 右移 j+1 位；MTP 需要 t+2，所以多取一个"""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - T - k_extra, (B,))
    x = torch.stack([torch.from_numpy(d[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + T + k_extra].astype(np.int64)) for i in ix])
    return x.to(args.device), y.to(args.device)


class MTPModule(nn.Module):
    """DeepSeek-V3 的一个 MTP 模块：RMSNorm(h) ‖ RMSNorm(Emb(t_{i+1})) → 线性 2d→d → 一个 Transformer block。
    embedding 与 lm_head 与主干共享（在 GPTWithMTP 里传入），这里只有自己的 norm、投影和 block。"""
    def __init__(self, config):
        super().__init__()
        self.norm_h = LayerNorm(config.n_embd, bias=config.bias)
        self.norm_e = LayerNorm(config.n_embd, bias=config.bias)
        self.proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self.block = Block(config)

    def forward(self, h, emb_next):
        # h: 主干最后一层的表示 [B, T', d]；emb_next: 下一个真实 token 的 embedding [B, T', d]
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(emb_next)], dim=-1))
        return self.block(x)


class GPTWithMTP(nn.Module):
    def __init__(self, config, use_mtp):
        super().__init__()
        self.gpt = GPT(config)
        self.mtp = MTPModule(config) if use_mtp else None

    def trunk(self, idx):
        """nanoGPT forward 的前半段：返回 ln_f 之前的最后一层表示"""
        g = self.gpt.transformer
        pos = torch.arange(0, idx.size(1), device=idx.device)
        x = g.drop(g.wte(idx) + g.wpe(pos))
        for blk in g.h: x = blk(x)
        return x

    def forward(self, x, y):
        h = self.trunk(x)                                       # [B, T, d]
        logits = self.gpt.lm_head(self.gpt.transformer.ln_f(h))   # 主头：预测 t+1
        loss_main = F.cross_entropy(logits.reshape(-1, V), y[:, :T].reshape(-1))
        if self.mtp is None: return loss_main, None
        # MTP 头：位置 i 拿主干表示 h_i 和真实的 t_{i+1}（y[:, i]）的 embedding，预测 t_{i+2}（y[:, i+1]）
        # 最后一个位置没有 t+2 的目标可用……其实 y 多取了一个，所以 T 个位置都有
        emb_next = self.gpt.transformer.wte(y[:, :T])            # [B, T, d]
        h2 = self.mtp(h, emb_next)
        logits2 = self.gpt.lm_head(self.gpt.transformer.ln_f(h2))
        loss_mtp = F.cross_entropy(logits2.reshape(-1, V), y[:, 1:T + 1].reshape(-1))
        return loss_main, loss_mtp

    @torch.no_grad()
    def eval_losses(self, n=20):
        self.eval(); lm, lmtp, hit, tot = 0.0, 0.0, 0, 0
        for _ in range(n):
            x, y = get_batch("val")
            a, b = self(x, y); lm += a.item()
            if b is not None:
                lmtp += b.item()
                h = self.trunk(x); h2 = self.mtp(h, self.gpt.transformer.wte(y[:, :T]))
                pred2 = self.gpt.lm_head(self.gpt.transformer.ln_f(h2)).argmax(-1)
                hit += (pred2 == y[:, 1:T + 1]).sum().item(); tot += pred2.numel()
        self.train()
        return lm / n, (lmtp / n if self.mtp is not None else None), (hit / tot if tot else None)


def train(use_mtp):
    cfg = GPTConfig(vocab_size=V, block_size=T, n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False)
    torch.manual_seed(1337)
    m = GPTWithMTP(cfg, use_mtp).to(args.device)
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
    t0 = time.time(); log = []
    for it in range(args.iters + 1):
        lr = 1e-3 * min(1.0, (it + 1) / 100) * (0.5 * (1 + math.cos(math.pi * it / args.iters)) * 0.9 + 0.1)
        for g in opt.param_groups: g["lr"] = lr
        if it % 250 == 0:
            lm, lmtp, hit = m.eval_losses()
            log.append((it, lm, lmtp, hit))
            extra = f"  MTP 头 val loss {lmtp:.4f}  t+2 top-1 命中 {hit:.1%}" if lmtp is not None else ""
            print(f"  step {it:4d}: 主头 val loss {lm:.4f}{extra}")
        x, y = get_batch("train")
        loss_main, loss_mtp = m(x, y)
        loss = loss_main if loss_mtp is None else loss_main + args.lam * loss_mtp
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    dt = (time.time() - t0) / args.iters
    return n_params, dt, log


print(f"== 不带 MTP（基线）：4 层 4 头 d=128，{args.iters} 步，{args.device}")
p0, dt0, log0 = train(False)
print(f"== 带 MTP（D=1，λ={args.lam}）")
p1, dt1, log1 = train(True)
print("\n== 汇总")
print(f"  参数量：基线 {p0:,}，带 MTP {p1:,}（多 {p1 - p0:,}，+{(p1 / p0 - 1):.0%}：一个 block + 2d→d 投影 + 两个 norm）")
print(f"  每步耗时：基线 {dt0 * 1000:.0f} ms，带 MTP {dt1 * 1000:.0f} ms（+{(dt1 / dt0 - 1):.0%}）")
print(f"  最终主头 val loss：基线 {log0[-1][1]:.4f}，带 MTP {log1[-1][1]:.4f}")
print(f"  MTP 头对 t+2：val loss {log1[-1][2]:.4f}，top-1 命中率 {log1[-1][3]:.1%}（投机解码时 draft 被接受的大致比例）")
