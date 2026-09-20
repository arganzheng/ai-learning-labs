"""多模态（07）扩散上篇：DDPM 在二维 toy 数据上从零训练——加噪、噪声预测、逐步去噪、DDIM 跳步。

    python 06_ddpm_toy.py            # 全部：forward closedform train sample ddim（约 1 分钟 CPU）
"""
import sys

import numpy as np
import torch

from _diffusion_toy import MLP, ddpm_schedule, moons, train
from _plot import C, plt, save

T = 1000
betas, alphas, abar = ddpm_schedule(T)
X, _ = moons(4000)
model = MLP()


def q_sample(x0, t, eps):
    """闭式前向：x_t = sqrt(ᾱ_t) x_0 + sqrt(1−ᾱ_t) ε。t 是 [n] 的整数步。"""
    a = abar[t][:, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


def run_forward():
    print("=== 1. 前向加噪：同一批点在 t = 0, 100, 300, 600, 999 长什么样 ===")
    ts = [0, 100, 300, 600, 999]
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 1.8))
    eps = torch.randn_like(X)
    for ax, t in zip(axes, ts):
        xt = q_sample(X, torch.full((len(X),), t), eps) if t > 0 else X
        ax.scatter(xt[:, 0], xt[:, 1], s=1, color=C["blue"], alpha=0.5)
        ax.set_title(f"t = {t}\nᾱ = {abar[t]:.3f}", fontsize=8); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        print(f"  t = {t:>4}: ᾱ_t = {abar[t]:.4f}，信号系数 sqrt(ᾱ) = {abar[t].sqrt():.3f}，噪声系数 sqrt(1−ᾱ) = {(1-abar[t]).sqrt():.3f}")
    save(fig, "06-forward-noising")


def run_closedform():
    print("=== 2. 闭式解验算：逐步加噪 1000 次 vs 一步到位，方差是否一致 ===")
    x0 = torch.tensor([[1.0, -0.5]]).repeat(20000, 1)
    x = x0.clone()
    for t in range(T):
        x = alphas[t].sqrt() * x + betas[t].sqrt() * torch.randn_like(x)          # 逐步：x_t = sqrt(α_t) x_{t−1} + sqrt(β_t) ε
    direct = q_sample(x0, torch.full((20000,), T - 1), torch.randn_like(x0))
    print(f"  逐步 1000 次：均值 {x.mean(0).numpy().round(3)}，标准差 {x.std(0).numpy().round(3)}")
    print(f"  闭式一步：  均值 {direct.mean(0).numpy().round(3)}，标准差 {direct.std(0).numpy().round(3)}")
    print(f"  理论：均值 sqrt(ᾱ_T) x_0 = {(abar[-1].sqrt() * x0[0]).numpy().round(3)}，标准差 sqrt(1−ᾱ_T) = {(1-abar[-1]).sqrt():.3f}  → 几乎就是 N(0, I)")
    t = 300
    print(f"  手算 t = {t}：ᾱ = {abar[t]:.3f}，x_0 = (1.0, −0.5)，ε = (0.3, −1.2) → x_t = {abar[t].sqrt():.3f}·x_0 + {(1-abar[t]).sqrt():.3f}·ε = {(abar[t].sqrt()*torch.tensor([1.0,-0.5]) + (1-abar[t]).sqrt()*torch.tensor([0.3,-1.2])).numpy().round(3)}")


def ddpm_loss(model, x0, _):
    t = torch.randint(0, T, (len(x0),))                                   # ① 每个样本随机抽一个时间步
    eps = torch.randn_like(x0)                                            # ② 抽一份噪声
    xt = q_sample(x0, t, eps)                                             # ③ 闭式一步得到 x_t
    return ((model(xt, t / T) - eps) ** 2).mean()                         # ④ 让网络从 (x_t, t) 猜出 ε：MSE


def run_train():
    print("=== 3. 训练：L_simple = E ||ε − ε_θ(x_t, t)||²，12000 步 ===")
    hist = train(model, ddpm_loss, X, steps=12000, lr=1e-3, log_every=3000)
    torch.save(model.state_dict(), "out/06_ddpm_model.pt")
    with torch.no_grad():
        for t in (50, 300, 700, 990):
            tt = torch.full((len(X),), t); eps = torch.randn_like(X); xt = q_sample(X, tt, eps)
            err = ((model(xt, tt / T) - eps) ** 2).mean().item()
            print(f"  t = {t:>3}：预测噪声的 MSE {err:.3f}（ε 本身方差 1；t 大时 x_t ≈ ε 容易猜，t 小时噪声藏在数据里难猜）")
    fig, ax = plt.subplots(figsize=(4.6, 2.4)); ax.plot(hist, lw=0.5, color=C["blue"]); ax.set_xlabel("步"); ax.set_ylabel("L_simple"); ax.set_ylim(0, 1.2); ax.set_title("噪声预测的 MSE")
    save(fig, "06-train-loss")


@torch.no_grad()
def ddpm_sample(model, n, snapshots=()):
    x = torch.randn(n, 2)                                                 # ① 从纯噪声出发
    shots = {}
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps = model(x, tt / T)                                            # ② 预测噪声
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()   # ③ μ_θ = (x_t − β_t/√(1−ᾱ_t) ε_θ) / √α_t
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)      # ④ 加回一点随机噪声（最后一步不加）
        if t in snapshots:
            shots[t] = x.clone()
    return x, shots


@torch.no_grad()
def ddim_sample(model, n, steps, eta=0.0):
    ts = torch.linspace(T - 1, 0, steps + 1).long()                       # 只在 steps 个时间步上跳
    x = torch.randn(n, 2)
    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        eps = model(x, torch.full((n,), t.item()) / T)
        x0_hat = (x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()           # ① 先估 x̂_0
        a_s = abar[s] if s > 0 else torch.tensor(1.0)
        sigma = eta * ((1 - a_s) / (1 - abar[t]) * (1 - abar[t] / a_s)).sqrt()
        x = a_s.sqrt() * x0_hat + (1 - a_s - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * torch.randn_like(x)   # ② 按 s 的噪声水平「重新加噪」，用预测的方向
    return x


def run_sample():
    print("=== 4. 采样：从 N(0, I) 出发 1000 步逐步去噪 ===")
    model.load_state_dict(torch.load("out/06_ddpm_model.pt"))
    shots_t = (999, 600, 300, 150, 50, 0)
    x, shots = ddpm_sample(model, 2000, snapshots=shots_t)
    fig, axes = plt.subplots(1, 6, figsize=(7.6, 1.6))
    for ax, t in zip(axes, shots_t):
        ax.scatter(shots[t][:, 0], shots[t][:, 1], s=1, color=C["orange"], alpha=0.5)
        ax.set_title(f"t = {t}", fontsize=8); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    save(fig, "06-reverse-sampling")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X.numpy())
    d = nn.kneighbors(x.numpy())[0].mean()
    print(f"  2000 个生成样本到最近真实数据点的平均距离 {d:.3f}（真实数据点彼此之间约 {NearestNeighbors(n_neighbors=2).fit(X.numpy()).kneighbors(X.numpy())[0][:,1].mean():.3f}）")


def run_ddim():
    print("=== 5. DDIM 跳步：1000 步 → 50 / 20 / 10 / 5 步 ===")
    model.load_state_dict(torch.load("out/06_ddpm_model.pt"))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X.numpy())
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 1.8))
    x_full, _ = ddpm_sample(model, 2000)
    axes[0].scatter(x_full[:, 0], x_full[:, 1], s=1, color=C["orange"], alpha=0.5); axes[0].set_title(f"DDPM 1000 步\n距离 {nn.kneighbors(x_full.numpy())[0].mean():.3f}", fontsize=8)
    for ax, steps in zip(axes[1:], (50, 20, 10, 5)):
        x = ddim_sample(model, 2000, steps)
        d = nn.kneighbors(x.numpy())[0].mean()
        ax.scatter(x[:, 0], x[:, 1], s=1, color=C["green"], alpha=0.5); ax.set_title(f"DDIM {steps} 步\n距离 {d:.3f}", fontsize=8)
        print(f"  DDIM {steps:>3} 步：到最近真实点的平均距离 {d:.3f}")
    for ax in axes:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    save(fig, "06-ddim-steps")
    # 确定性：同一个 x_T 两次采样结果相同
    torch.manual_seed(1); a = ddim_sample(model, 5, 20); torch.manual_seed(1); b = ddim_sample(model, 5, 20)
    print(f"  η = 0 的 DDIM 是确定性映射：同一个 x_T 两次得到的 x_0 最大差 {(a-b).abs().max():.1e}")


if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    for w in (sys.argv[1:] or ["forward", "closedform", "train", "sample", "ddim"]):
        globals()[f"run_{w}"](); print()
