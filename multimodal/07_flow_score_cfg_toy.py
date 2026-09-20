"""多模态（08）扩散下篇：同一份二维 toy 数据上——分数场、flow matching 的直线路径、步数对比、classifier-free guidance。

    python 07_flow_score_cfg_toy.py            # 全部：score flow steps cfg（约 1 分钟 CPU；需要先跑 06_ddpm_toy.py train）
"""
import sys

import numpy as np
import torch

from _diffusion_toy import MLP, ddpm_schedule, moons, train
from _plot import C, plt, save

T = 1000
betas, alphas, abar = ddpm_schedule(T)
X, Y = moons(4000)


def run_score():
    print("=== 1. 分数 = −ε/σ：把 DDPM 训好的噪声预测器画成向量场 ===")
    ddpm = MLP(); ddpm.load_state_dict(torch.load("out/06_ddpm_model.pt"))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    g = torch.linspace(-2.5, 2.5, 18); gx, gy = torch.meshgrid(g, g, indexing="xy"); pts = torch.stack([gx.ravel(), gy.ravel()], 1)
    for ax, t in zip(axes, (600, 300, 80)):
        sigma = (1 - abar[t]).sqrt()
        with torch.no_grad():
            eps = ddpm(pts, torch.full((len(pts),), t / T))
        score = -eps / sigma                                                       # ① 分数 s = ∇ log p_t = −ε_θ / σ_t
        ax.scatter(X[:, 0] * abar[t].sqrt(), X[:, 1] * abar[t].sqrt(), s=1, color=C["light"])
        ax.quiver(pts[:, 0], pts[:, 1], score[:, 0], score[:, 1], color=C["blue"], scale=40 if t > 100 else 120, width=0.004)
        ax.set_title(f"t = {t}，σ = {sigma:.2f}", fontsize=9); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        print(f"  t = {t}：σ = {sigma:.2f}；网格上分数向量的平均长度 {score.norm(dim=1).mean():.2f}——箭头都指向数据（灰点）所在的地方")
    save(fig, "07-score-field", dpi=110)
    # Tweedie：x̂_0 = x_t + σ² s
    t = 300; sigma2 = 1 - abar[t]
    x0 = X[:5]; eps = torch.randn(5, 2); xt = abar[t].sqrt() * x0 + sigma2.sqrt() * eps
    with torch.no_grad():
        s = -ddpm(xt, torch.full((5,), t / T)) / sigma2.sqrt()
    x0_hat = (xt + sigma2 * s) / abar[t].sqrt()
    print(f"  Tweedie 验算（t = {t}）：真实 x_0 = {x0[0].numpy().round(2)}，带噪 x_t = {xt[0].numpy().round(2)}，x_t + σ²·s 再除以 √ᾱ = {x0_hat[0].numpy().round(2)}（去噪 = 沿分数走一步）")


def fm_loss(model, x0, _):
    t = torch.rand(len(x0))                                                    # ① t ~ U[0, 1]（0 = 数据，1 = 噪声）
    eps = torch.randn_like(x0)
    xt = (1 - t)[:, None] * x0 + t[:, None] * eps                              # ② 直线插值
    return ((model(xt, t) - (eps - x0)) ** 2).mean()                           # ③ 让网络猜速度 ε − x_0


@torch.no_grad()
def euler_sample(model, n, steps, x1=None, track=False):
    x = torch.randn(n, 2) if x1 is None else x1.clone()
    traj = [x.clone()]
    for i in range(steps):
        t = 1 - i / steps
        x = x - model(x, torch.full((n,), t)) / steps                          # ① x_{t−Δ} = x_t − v·Δ：沿速度反方向走一小步
        traj.append(x.clone())
    return (x, torch.stack(traj)) if track else x


@torch.no_grad()
def ddim_traj(model, x1, steps):
    ts = torch.linspace(T - 1, 0, steps + 1).long(); x = x1.clone(); traj = [x.clone()]
    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        eps = model(x, torch.full((len(x),), t.item()) / T)
        x0_hat = (x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
        a_s = abar[s] if s > 0 else torch.tensor(1.0)
        x = a_s.sqrt() * x0_hat + (1 - a_s).sqrt() * eps; traj.append(x.clone())
    return torch.stack(traj)


def straightness(tr):
    chord = (tr[-1] - tr[0]).norm(dim=1); path = (tr[1:] - tr[:-1]).norm(dim=2).sum(0)
    return (chord / path).mean().item()                                      # 弦长 / 路径长：1 = 完全直


def fm_pair_loss(x1_x0):
    """reflow 用：训练对 (噪声 x_1, 数据 x_0) 是固定配好的，不再随机配。"""
    def loss(model, batch, _):
        x1, x0 = batch[:, :2], batch[:, 2:]
        t = torch.rand(len(x0)); xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        return ((model(xt, t) - (x1 - x0)) ** 2).mean()
    return loss


def run_flow():
    print("=== 2. flow matching：学速度场 v = ε − x_0，直线路径；再做一轮 reflow ===")
    fm = MLP()
    train(fm, fm_loss, X, steps=12000, lr=1e-3, log_every=3000)
    torch.save(fm.state_dict(), "out/07_fm_model.pt")
    ddpm = MLP(); ddpm.load_state_dict(torch.load("out/06_ddpm_model.pt"))
    torch.manual_seed(3); x1 = torch.randn(12, 2)
    _, tr_fm = euler_sample(fm, 12, 100, x1=x1, track=True)
    tr_dd = ddim_traj(ddpm, x1, 100)
    print(f"  同一批 12 个噪声起点：DDIM（DDPM 模型）轨迹的直线度 {straightness(tr_dd):.2f}，flow matching 轨迹 {straightness(tr_fm):.2f}（1 = 完全直）")
    print("  训练时每条「条件路径」是直线，但随机配对的直线互相交叉，学到的边缘速度场把它们平均掉——轨迹弯了。这正是 reflow 要修的：")
    # reflow：用训好的模型生成 (x_1, x_0) 配对——它们由 ODE 轨迹连接、彼此不交叉——在这些配对上重训
    torch.manual_seed(0); z = torch.randn(8000, 2); x0_gen = euler_sample(fm, 8000, 100, x1=z)
    pairs = torch.cat([z, x0_gen], 1)
    fm2 = MLP(); fm2.load_state_dict(fm.state_dict())
    train(fm2, fm_pair_loss(pairs), pairs, steps=6000, lr=5e-4, log_every=3000)
    torch.save(fm2.state_dict(), "out/07_fm_reflow_model.pt")
    _, tr_rf = euler_sample(fm2, 12, 100, x1=x1, track=True)
    print(f"  reflow 一轮后同一批起点的轨迹直线度 {straightness(tr_rf):.2f}")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for ax, tr, name, col in zip(axes, (tr_dd, tr_fm, tr_rf), ("DDIM（DDPM 模型）", "flow matching", "flow matching + 1 轮 reflow"), (C["orange"], C["blue"], C["green"])):
        ax.scatter(X[:, 0], X[:, 1], s=1, color=C["light"])
        for k in range(12):
            ax.plot(tr[:, k, 0], tr[:, k, 1], color=col, lw=0.8); ax.plot(tr[0, k, 0], tr[0, k, 1], "o", color=C["gray"], ms=3); ax.plot(tr[-1, k, 0], tr[-1, k, 1], "o", color=col, ms=3)
        ax.set_title(f"{name}\n直线度 {straightness(tr):.2f}", fontsize=9); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "07-trajectories")


def run_steps():
    print("=== 3. 少步采样：DDIM vs flow matching vs reflow 在 1 / 2 / 5 / 20 步下 ===")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X.numpy())
    fm = MLP(); fm.load_state_dict(torch.load("out/07_fm_model.pt"))
    ddpm = MLP(); ddpm.load_state_dict(torch.load("out/06_ddpm_model.pt"))
    from importlib import import_module
    ddim_sample = import_module("06_ddpm_toy").ddim_sample
    fm2 = MLP(); fm2.load_state_dict(torch.load("out/07_fm_reflow_model.pt"))
    fig, axes = plt.subplots(3, 4, figsize=(7.6, 5.6))
    for j, steps in enumerate((1, 2, 5, 20)):
        for i, (name, x) in enumerate((("DDIM", ddim_sample(ddpm, 2000, steps)), ("flow matching", euler_sample(fm, 2000, steps)), ("FM + reflow", euler_sample(fm2, 2000, steps)))):
            d = nn.kneighbors(x.numpy())[0].mean()
            axes[i, j].scatter(x[:, 0], x[:, 1], s=1, color=[C["orange"], C["blue"], C["green"]][i], alpha=0.5)
            axes[i, j].set_title(f"{name} {steps} 步\n距离 {d:.3f}", fontsize=8); axes[i, j].set_xlim(-3, 3); axes[i, j].set_ylim(-3, 3); axes[i, j].set_aspect("equal"); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            print(f"  {name:<20} {steps:>2} 步：到最近真实点的平均距离 {d:.3f}")
    save(fig, "07-few-steps")


def cfg_loss(model, x0, y):
    t = torch.rand(len(x0)); eps = torch.randn_like(x0)
    xt = (1 - t)[:, None] * x0 + t[:, None] * eps
    c = torch.where(torch.rand(len(y)) < 0.15, torch.full_like(y, 2), y)          # ① 15% 的概率把条件换成 ∅（编号 2）：条件 dropout
    return ((model(xt, t, c) - (eps - x0)) ** 2).mean()


@torch.no_grad()
def cfg_sample(model, n, c, w, steps=50):
    x = torch.randn(n, 2); cc = torch.full((n,), c); nul = torch.full((n,), 2)
    for i in range(steps):
        t = torch.full((n,), 1 - i / steps)
        v_c, v_0 = model(x, t, cc), model(x, t, nul)                           # ② 两次前向：有条件、无条件
        v = v_0 + w * (v_c - v_0)                                              # ③ CFG：沿「有条件 − 无条件」的方向外推 w 倍
        x = x - v / steps
    return x


def run_cfg():
    print("=== 4. classifier-free guidance：条件 = 上月牙 / 下月牙 ===")
    cm = MLP(cond=True)
    train(cm, cfg_loss, X, steps=12000, lr=1e-3, y=Y, log_every=3000)
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 1.9))
    for ax, w in zip(axes, (0, 1, 2, 4, 8)):
        x = cfg_sample(cm, 1500, 0, w)
        frac = (x[:, 1] > -0.3).float().mean().item()                          # 类 0 是上面那个月牙的粗略判据
        spread = x.std(0).mean().item()
        ax.scatter(X[:, 0], X[:, 1], s=1, color=C["light"]); ax.scatter(x[:, 0], x[:, 1], s=1.5, color=C["red"], alpha=0.6)
        ax.set_title(f"w = {w}\n落在类 0：{frac:.0%}，散度 {spread:.2f}", fontsize=8); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        print(f"  w = {w}：条件「类 0」的 1500 个样本里落在类 0 一侧的 {frac:.1%}，样本坐标的标准差 {spread:.3f}（w 越大越集中）")
    save(fig, "07-cfg")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["score", "flow", "steps", "cfg"]):
        globals()[f"run_{w}"](); print()
