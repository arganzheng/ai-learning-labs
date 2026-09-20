"""多模态（01）视觉编码器：patch 化、InfoNCE 手算、toy 对比学习、温度、sigmoid 损失。

    python 01_vision_encoders_and_contrastive.py            # 全部
    python 01_vision_encoders_and_contrastive.py patchify   # 只跑一个：patchify infonce toy temperature sigmoid

全部 CPU、离线，几秒。图输出到 out/01-*.svg。
"""
import sys

import numpy as np
from sklearn.datasets import load_digits

from _plot import C, plt, save

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- 1. patch 化
def patchify(img, P):
    """[H, W] 的图 → [N, P*P] 的 patch 序列（按行扫描）。"""
    H, W = img.shape
    return img.reshape(H // P, P, W // P, P).transpose(0, 2, 1, 3).reshape(-1, P * P)   # ① 切块 ② 每块拉平


def run_patchify():
    print("=== 1. patch 化：一张 8×8 的手写数字 → 16 个 patch → 16 个向量 ===")
    img = load_digits().images[0]                                  # 8×8，像素 0–16
    P = 2
    patches = patchify(img, P)                                     # [16, 4]
    W = rng.normal(0, 0.5, (P * P, 3))                             # 线性层：4 → 3 维（真实 ViT 是 588 → 1024）
    tokens = patches @ W                                           # [16, 3]
    print(f"  图 {img.shape} → {patches.shape[0]} 个 patch，每个 {P}×{P}={P*P} 个像素")
    print(f"  第 0 个 patch（左上角）的像素：{patches[0].astype(int)}；乘 W 之后的向量：{np.round(tokens[0], 2)}")
    print(f"  第 5 个 patch 的像素：{patches[5].astype(int)}；向量：{np.round(tokens[5], 2)}")
    print(f"  token 矩阵形状 {tokens.shape}（真实 ViT-L/14 在 336² 上是 [576, 1024]）")

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    axes[0].imshow(img, cmap="gray_r"); axes[0].set_title("原图 8×8")
    for k in range(1, 4):
        axes[0].axhline(k * P - 0.5, color=C["red"], lw=1); axes[0].axvline(k * P - 0.5, color=C["red"], lw=1)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for i in range(16):
        r, c = divmod(i, 4)
        axes[0].text(c * P + 0.5, r * P + 0.5, str(i), ha="center", va="center", color=C["blue"], fontsize=7)
    axes[1].imshow(patches, cmap="gray_r", aspect="auto"); axes[1].set_title("16 个 patch，各 4 个像素")
    axes[1].set_xlabel("patch 内像素"); axes[1].set_ylabel("patch 序号"); axes[1].set_xticks(range(4)); axes[1].set_yticks(range(0, 16, 3))
    axes[2].imshow(tokens, cmap="RdBu_r", aspect="auto"); axes[2].set_title("乘 W（4→3）：16 个 3 维 token")
    axes[2].set_xlabel("维"); axes[2].set_ylabel("token 序号"); axes[2].set_xticks(range(3)); axes[2].set_yticks(range(0, 16, 3))
    save(fig, "01-patchify")


# ---------------------------------------------------------------- 2. InfoNCE 手算
def infonce(S):
    """S: [B, B] 的相似度矩阵（已除温度）。返回图→文与文→图两个方向的平均损失。"""
    B = len(S)
    logp_i2t = S[np.arange(B), np.arange(B)] - np.log(np.exp(S).sum(1))    # ① 每行 softmax，取对角线的 log 概率
    logp_t2i = S[np.arange(B), np.arange(B)] - np.log(np.exp(S).sum(0))    # ② 每列 softmax
    return -(logp_i2t.mean() + logp_t2i.mean()) / 2


def run_infonce():
    print("=== 2. InfoNCE：3 对图文的手算 ===")
    S = np.array([[0.9, 0.2, 0.1],
                  [0.3, 0.8, 0.2],
                  [0.1, 0.4, 0.7]])                                 # 余弦相似度，对角线是配对
    for tau in (1.0, 0.1):
        Z = S / tau
        P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        print(f"  τ = {tau}：第一行 softmax 概率 {np.round(P[0], 3)}，正确对的概率 {P[0,0]:.3f}，−log = {-np.log(P[0,0]):.3f}")
        print(f"         三行 −log p_ii = {np.round(-np.log(np.diag(P)), 3)}，图→文平均 {(-np.log(np.diag(P))).mean():.3f}；双向平均 {infonce(Z):.3f}")

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{S[i,j]:.1f}", ha="center", va="center", color="white" if S[i, j] > 0.5 else "black",
                    fontweight="bold" if i == j else None)
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C["red"], lw=2))
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["文本 1", "文本 2", "文本 3"]); ax.set_yticklabels(["图 1", "图 2", "图 3"])
    ax.set_title("相似度矩阵 s_ij：对角线（红框）是配对")
    plt.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "01-similarity-matrix")


# ---------------------------------------------------------------- 3. toy 对比学习
def run_toy():
    print("=== 3. toy 对比学习：两个小编码器把「图」与「文」拉到同一空间 ===")
    import torch
    torch.manual_seed(0)
    n, K = 600, 4                                                    # 600 对样本，4 个隐含「概念」
    concept = rng.integers(0, K, n)
    centers_img = rng.normal(0, 3, (K, 6)); centers_txt = rng.normal(0, 3, (K, 5))
    X_img = torch.tensor(centers_img[concept] + rng.normal(0, 0.8, (n, 6)), dtype=torch.float32)   # 「图」：6 维原始特征
    X_txt = torch.tensor(centers_txt[concept] + rng.normal(0, 0.8, (n, 5)), dtype=torch.float32)   # 「文」：5 维，与图的空间完全不同
    f = torch.nn.Linear(6, 2); g = torch.nn.Linear(5, 2)                # 两个编码器：各一个线性层 → 2 维（真实 CLIP 是两个 Transformer → 768 维）
    tau, B = 0.1, 64
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=0.02)

    def encode(X, enc):
        return torch.nn.functional.normalize(enc(X), dim=1)             # ① 归一化到单位长度

    def clip_loss(u, v):
        S = u @ v.T / tau                                                # ② B×B 相似度 / 温度
        labels = torch.arange(len(u))
        return (torch.nn.functional.cross_entropy(S, labels)             # ③ 每行：图 i 在 B 条文本里选自己的（图→文）
                + torch.nn.functional.cross_entropy(S.T, labels)) / 2    # ④ 每列：文 j 在 B 张图里选自己的（文→图）

    snapshots, losses = {}, []
    for it in range(401):
        if it in (0, 400):
            with torch.no_grad():
                snapshots[it] = (encode(X_img, f).numpy(), encode(X_txt, g).numpy())
        idx = torch.tensor(rng.choice(n, B, replace=False))
        loss = clip_loss(encode(X_img[idx], f), encode(X_txt[idx], g))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    with torch.no_grad():
        u, v = encode(X_img, f).numpy(), encode(X_txt, g).numpy()
    S = u @ v.T
    top1_concept = (concept[S.argmax(1)] == concept).mean()
    same = np.mean([S[i, concept == concept[i]].mean() for i in range(n)])
    diff = np.mean([S[i, concept != concept[i]].mean() for i in range(n)])
    print(f"  训练前 loss {losses[0]:.2f}（log 64 = {np.log(64):.2f} 是随机猜的水平）；400 步后 {np.mean(losses[-20:]):.2f}")
    print(f"  batch 里同一概念约有 16 条文本互相几乎一样，所以 loss 的下限约 log 16 = {np.log(16):.2f}，不是 0")
    print(f"  图→文检索：最相似的文本与图同一概念的比例 {top1_concept:.1%}")
    print(f"  同概念图文的平均余弦 {same:.2f}，不同概念 {diff:.2f}——两个本来毫无关系的空间被拉到了一起")

    order = np.argsort(concept, kind="stable")[::5]                      # 按概念排好、抽 120 个，画相似度矩阵
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for ax, it in zip(axes[:2], (0, 400)):
        u0, v0 = snapshots[it]
        M = u0[order] @ v0[order].T
        ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        b = np.cumsum([np.sum(concept[order] == k) for k in range(K)])
        for x in b[:-1]:
            ax.axhline(x - 0.5, color="k", lw=0.5); ax.axvline(x - 0.5, color="k", lw=0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("文本（按概念排序）"); ax.set_ylabel("图（按概念排序）")
        ax.set_title("训练前：余弦无规律" if it == 0 else "400 步后：同概念块亮、其余暗", fontsize=9)
    axes[2].plot(losses, color=C["blue"], lw=0.8)
    axes[2].axhline(np.log(B), color=C["gray"], ls="--", lw=0.8); axes[2].text(200, np.log(B) + 0.12, "log 64：随机猜", fontsize=7, color=C["gray"], ha="center")
    axes[2].axhline(np.log(16), color=C["green"], ls="--", lw=0.8); axes[2].text(200, np.log(16) - 0.35, "log 16：同概念分不开的下限", fontsize=7, color=C["green"], ha="center")
    axes[2].set_xlabel("步"); axes[2].set_ylabel("InfoNCE"); axes[2].set_title("损失", fontsize=9)
    save(fig, "01-toy-contrastive", dpi=72)


# ---------------------------------------------------------------- 4. 温度
def run_temperature():
    print("=== 4. 温度：余弦 0.30 vs 0.25 在不同 τ 下的 softmax ===")
    cos = np.array([0.30, 0.25, 0.25, 0.25])                       # 正确类 0.30，三个错误类 0.25
    for tau in (1.0, 0.1, 0.01):
        z = cos / tau; p = np.exp(z) / np.exp(z).sum()
        print(f"  τ = {tau:<5} logit 差 {(0.05/tau):>5.1f}  正确类概率 {p[0]:.3f}  −log = {-np.log(p[0]):.3f}")
    taus = np.logspace(-2.3, 0, 60)
    p0 = [np.exp(0.30 / t) / np.exp(cos / t).sum() for t in taus]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.semilogx(taus, p0, color=C["blue"])
    for t in (1.0, 0.1, 0.01):
        p = np.exp(0.30 / t) / np.exp(cos / t).sum(); ax.plot(t, p, "o", color=C["red"]); ax.annotate(f"τ={t}: {p:.2f}", (t, p), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.axhline(0.25, color=C["gray"], ls="--", lw=0.8); ax.text(0.02, 0.27, "1/4 = 随机猜", fontsize=7, color=C["gray"])
    ax.set_xlabel("温度 τ（对数轴）"); ax.set_ylabel("正确类的 softmax 概率"); ax.set_title("余弦只差 0.05，τ 决定它算不算「分开了」")
    save(fig, "01-temperature")


# ---------------------------------------------------------------- 5. sigmoid 损失
def run_sigmoid():
    print("=== 5. SigLIP 的 sigmoid 损失：同一个 3×3 例子 ===")
    S = np.array([[0.9, 0.2, 0.1], [0.3, 0.8, 0.2], [0.1, 0.4, 0.7]]) / 0.1     # 除温度 0.1
    Z = np.where(np.eye(3) == 1, 1, -1)
    for b in (0.0, -10.0):
        L = -np.log(1 / (1 + np.exp(-Z * (S + b))))
        print(f"  b = {b:>5}：每对的损失\n{np.round(L, 3)}\n     正对平均 {L[np.eye(3)==1].mean():.3f}，负对平均 {L[np.eye(3)==0].mean():.3f}，总和/B = {L.sum()/3:.3f}")
    print(f"  σ(−10) = {1/(1+np.exp(10)):.1e}：b = −10 让所有对一开始都被判成「不匹配」，B²−B 个负对几乎不产生损失，正对再慢慢被拉高")


if __name__ == "__main__":
    which = sys.argv[1:] or ["patchify", "infonce", "toy", "temperature", "sigmoid"]
    for w in which:
        globals()[f"run_{w}"]()
        print()
