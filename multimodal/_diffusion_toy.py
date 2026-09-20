"""07 / 08 篇共用：二维 toy 数据（两个月牙）、小 MLP、DDPM 调度。CPU 上几十秒训完。"""
import numpy as np
import torch
from sklearn.datasets import make_moons

torch.manual_seed(0)


def moons(n, seed=0):
    X, y = make_moons(n, noise=0.06, random_state=seed)
    X = (X - [0.5, 0.25]) / [1.1, 0.55]                                  # 归一到大约 [-1, 1]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


class MLP(torch.nn.Module):
    """输入 (x_t, t[, 类别 c])，输出与 x 同形状的向量（噪声 / 速度 / 分数）。"""
    def __init__(self, cond=False, width=256, dim=2):
        super().__init__()
        self.cond = cond
        self.emb_c = torch.nn.Embedding(3, 16) if cond else None            # 类 0 / 类 1 / 空条件 ∅（=2）
        din = dim + 16 + (16 if cond else 0)
        self.net = torch.nn.Sequential(torch.nn.Linear(din, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, dim))

    def forward(self, x, t, c=None):
        freqs = torch.exp(torch.linspace(0, 4, 8)) * 3.1416                 # 时间 t 用 8 对正弦余弦编码成 16 维（Transformer 的位置编码同款）
        te = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], 1)
        h = [x, te] + ([self.emb_c(c)] if self.cond else [])
        return self.net(torch.cat(h, 1))


def ddpm_schedule(T=1000, beta1=1e-4, beta2=0.02):
    betas = torch.linspace(beta1, beta2, T)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, 0)
    return betas, alphas, abar


def train(model, loss_fn, X, steps=4000, bs=512, lr=2e-3, y=None, log_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for it in range(steps):
        idx = torch.randint(0, len(X), (bs,))
        loss = loss_fn(model, X[idx], None if y is None else y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); hist.append(loss.item())
        if log_every and (it + 1) % log_every == 0:
            print(f"    step {it+1:>5}  loss {np.mean(hist[-200:]):.4f}")
    return hist
