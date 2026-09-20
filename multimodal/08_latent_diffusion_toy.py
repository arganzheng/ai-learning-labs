"""多模态（08）Latent diffusion：用 PCA 当「VAE」，在 16 维 latent 里跑上篇的 DDPM，生成手写数字。

    python 08_latent_diffusion_toy.py            # 全部：vae latent cost（约 1 分钟 CPU）
"""
import sys

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from _diffusion_toy import MLP, ddpm_schedule, train
from _plot import C, plt, save

T = 1000
betas, alphas, abar = ddpm_schedule(T)
digits = load_digits()
IMG = digits.images / 16.0                                                 # [1797, 8, 8]，像素归到 [0, 1]
X64 = IMG.reshape(len(IMG), -1)
K = 16
pca = PCA(K).fit(X64)                                                       # 「VAE」：编码器 = 投影到前 16 个主成分，解码器 = 乘回去加均值
Z = pca.transform(X64); Z_STD = Z.std()
Zn = torch.tensor(Z / Z_STD, dtype=torch.float32)                           # 归一化到单位方差（SD 的 scale factor 0.18215 做的同一件事）


def run_vae():
    print("=== 1. 「VAE」：把 8×8 = 64 个像素压成 16 个数，再解回来 ===")
    for k in (4, 8, 16, 32):
        p = PCA(k).fit(X64); rec = p.inverse_transform(p.transform(X64))
        print(f"  latent {k:>2} 维：压缩 {64/k:>4.1f} 倍，重建 MSE {((rec - X64)**2).mean():.4f}，保留方差 {p.explained_variance_ratio_.sum():.1%}")
    fig, axes = plt.subplots(2, 6, figsize=(7.6, 2.7))
    for j, i in enumerate((0, 1, 2, 3, 4, 5)):
        axes[0, j].imshow(IMG[i], cmap="gray_r", vmin=0, vmax=1); axes[0, j].set_title(f"原图 {digits.target[i]}", fontsize=8)
        rec = pca.inverse_transform(pca.transform(X64[i:i + 1]))[0].reshape(8, 8)
        axes[1, j].imshow(rec, cmap="gray_r", vmin=0, vmax=1); axes[1, j].set_title("16 维 → 解码", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "08-pca-vae")
    print(f"  第 0 张图的 16 个 latent 数：{np.round(Z[0], 1)}")


def ddpm_loss(model, z0, _):
    t = torch.randint(0, T, (len(z0),)); eps = torch.randn_like(z0)
    zt = abar[t].sqrt()[:, None] * z0 + (1 - abar[t]).sqrt()[:, None] * eps
    return ((model(zt, t / T) - eps) ** 2).mean()


@torch.no_grad()
def ddim_sample(model, n, steps, dim):
    ts = torch.linspace(T - 1, 0, steps + 1).long(); x = torch.randn(n, dim)
    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        eps = model(x, torch.full((n,), t.item()) / T)
        x0_hat = (x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
        a_s = abar[s] if s > 0 else torch.tensor(1.0)
        x = a_s.sqrt() * x0_hat + (1 - a_s).sqrt() * eps
    return x


def run_latent():
    print("=== 2. 在 16 维 latent 里训 DDPM，采样后解码成 8×8 的图 ===")
    model = MLP(dim=K)
    train(model, ddpm_loss, Zn, steps=12000, lr=1e-3, log_every=4000)
    z = ddim_sample(model, 60, 50, K).numpy() * Z_STD                        # ① 在 latent 里去噪 50 步
    imgs = pca.inverse_transform(z).reshape(-1, 8, 8).clip(0, 1)              # ② 「VAE 解码器」一次前向：16 个数 → 64 个像素
    fig, axes = plt.subplots(5, 12, figsize=(7.6, 3.3))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im, cmap="gray_r", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("从 16 维噪声出发、latent 里去噪 50 步、解码——60 张生成的「手写数字」", fontsize=9)
    save(fig, "08-latent-samples")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X64)
    d = nn.kneighbors(imgs.reshape(60, -1))[0].mean()
    print(f"  生成图到最近真实数字的平均像素距离 {d:.2f}（真实数字彼此之间约 {NearestNeighbors(n_neighbors=2).fit(X64).kneighbors(X64)[0][:,1].mean():.2f}）")
    print(f"  扩散网络每步处理 {K} 个数而不是 64 个；像素级细节（笔画的模糊边缘）由 PCA 解码器一次给出")


def run_cost():
    print("=== 3. 算账：像素空间 vs latent 空间，一张 1024² 的图 ===")
    for name, h, c in (("像素空间", 1024, 3), ("latent f8 × 4ch（SD 1.x）", 128, 4), ("latent f8 × 16ch（SD3 / FLUX）", 128, 16)):
        n = h * h * c; tokens = (h // 2) ** 2
        print(f"  {name:<28} {h}×{h}×{c} = {n:>9,} 个数，压缩 {1024*1024*3/n:>5.1f} 倍；patch 2 的 DiT 序列 {tokens:>6,} 个 token")
    print("  DiT 的 attention 是 O(N²)：像素空间 262144 个 token 的 attention 是 latent 4096 个的 4096 倍——不可行；latent 让它进入可行区间")


if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    for w in (sys.argv[1:] or ["vae", "latent", "cost"]):
        globals()[f"run_{w}"](); print()
