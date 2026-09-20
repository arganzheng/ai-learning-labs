"""多模态（09）自回归图像生成：VQ tokenizer（k-means 码本）把手写数字变成 16 个 token；FSQ；一个按位置的 next-token 计数模型生成数字。

    python 09_vq_tokenizer_and_ar_toy.py            # 全部：vq fsq ar steps（CPU 几秒）
"""
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

from _plot import C, plt, save

rng = np.random.default_rng(0)
digits = load_digits()
IMG = digits.images / 16.0                                                  # [1797, 8, 8]
P = 2                                                                       # 2×2 的 patch → 每张图 16 个 patch


def patchify(img):
    return img.reshape(8 // P, P, 8 // P, P).transpose(0, 2, 1, 3).reshape(-1, P * P)   # [16, 4]

def unpatchify(patches):
    return patches.reshape(8 // P, 8 // P, P, P).transpose(0, 2, 1, 3).reshape(8, 8)

ALL = np.stack([patchify(im) for im in IMG])                                # [1797, 16, 4]


def run_vq():
    global CODEBOOK, TOKENS
    print("=== 1. VQ tokenizer：2×2 patch（4 个数）→ 码本里最近的码字的编号 ===")
    K = 32
    km = KMeans(K, n_init=4, random_state=0).fit(ALL.reshape(-1, 4))          # ① 训练码本 = 对所有 patch 做 k-means（L2 第七篇）
    CODEBOOK = km.cluster_centers_                                             # [32, 4]
    TOKENS = km.predict(ALL.reshape(-1, 4)).reshape(-1, 16)                    # ② 每个 patch → 最近码字的编号：每张图 16 个整数
    rec = np.stack([unpatchify(CODEBOOK[t]) for t in TOKENS])                  # ③ 解码 = 查码本、拼回 8×8
    mse = ((rec - IMG) ** 2).mean()
    usage = np.bincount(TOKENS.ravel(), minlength=K)
    print(f"  码本 K = {K}（{int(np.log2(K))} bit / token）；一张图 = 16 个 token = {16 * int(np.log2(K))} bit，原图 64 像素 × 4 bit = 256 bit")
    print(f"  第 0 张图（数字 {digits.target[0]}）的 token 序列：{TOKENS[0].tolist()}")
    print(f"  重建 MSE {mse:.4f}；码字使用次数最多 {usage.max()}、最少 {usage.min()}——{(usage == 0).sum()} 个死码字")
    for K2 in (8, 128):
        km2 = KMeans(K2, n_init=2, random_state=0).fit(ALL.reshape(-1, 4)); t2 = km2.predict(ALL.reshape(-1, 4))
        rec2 = km2.cluster_centers_[t2].reshape(-1, 16, 4); print(f"  K = {K2:>3}：重建 MSE {((np.stack([unpatchify(r) for r in rec2]) - IMG) ** 2).mean():.4f}——码本越大重建越好，但 AR 模型要在越多的类里选")
    fig, axes = plt.subplots(3, 6, figsize=(7.6, 3.9))
    for j in range(6):
        axes[0, j].imshow(IMG[j], cmap="gray_r", vmin=0, vmax=1); axes[0, j].set_title(f"原图 {digits.target[j]}", fontsize=8)
        axes[1, j].imshow(TOKENS[j].reshape(4, 4), cmap="tab20", vmin=0, vmax=K - 1)
        for r in range(4):
            for c in range(4):
                axes[1, j].text(c, r, str(TOKENS[j, r * 4 + c]), ha="center", va="center", fontsize=6)
        axes[1, j].set_title("16 个 token", fontsize=8)
        axes[2, j].imshow(rec[j], cmap="gray_r", vmin=0, vmax=1); axes[2, j].set_title("查码本重建", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "09-vq-tokens")


def run_fsq():
    print("=== 2. FSQ：不学码本，每一维独立 round 到 L 个刻度 ===")
    L = 4
    z = ALL.reshape(-1, 4)                                                    # 直接把 4 个像素当 4 维特征（真实 FSQ 先投影到低维）
    q = np.round(z * (L - 1)) / (L - 1)                                       # ① 每维 round 到 {0, 1/3, 2/3, 1}
    idx = (q * (L - 1)).astype(int) @ (L ** np.arange(4))                     # ② 4 个刻度编号拼成一个整数：隐式码本 L^4 = 256
    rec = np.stack([unpatchify(q.reshape(-1, 16, 4)[i]) for i in range(len(IMG))])
    print(f"  L = {L}，4 维：隐式码本 {L}^4 = {L**4} 个码字，无需训练；用到 {len(np.unique(idx))} 个；重建 MSE {((rec - IMG) ** 2).mean():.4f}")
    print(f"  第 0 张图第 5 个 patch：像素 {np.round(z.reshape(-1,16,4)[0,5], 2)} → round 到 {q.reshape(-1,16,4)[0,5]} → 编号 {idx.reshape(-1,16)[0,5]}")


def run_ar():
    print("=== 3. 自回归生成：按栅格顺序，next-token 的条件是「左边的 token 与上边的 token」，用计数表估概率 ===")
    K = CODEBOOK.shape[0]
    # 模型：P(token_i | 左邻 token, 上邻 token, 位置 i)，计数 + 平滑；上下文没见过时退回只看左邻——一个最小的「语言模型」
    c2 = np.zeros((16, K, K, K)); c1 = np.zeros((16, K, K)); first = np.zeros(K)
    def ctx(seq, i):
        return seq[i - 1], (seq[i - 4] if i >= 4 else 0)                            # 左邻（行首时是上一行末尾）、上邻（第一行没有上邻，用 0）
    for seq in TOKENS:
        first[seq[0]] += 1
        for i in range(1, 16):
            l, u = ctx(seq, i)
            c2[i, l, u, seq[i]] += 1; c1[i, l, seq[i]] += 1                        # ① 训练 = 数数：(位置, 左, 上) → 当前
    def dist(i, l, u):
        row = c2[i, l, u]
        if row.sum() >= 3:
            return (row + 0.05) / (row + 0.05).sum()                               # ② 见过这个上下文 ≥ 3 次：用它
        return (c1[i, l] + 0.05) / (c1[i, l] + 0.05).sum()                         # ③ 否则退回只看左邻
    def sample():
        seq = [rng.choice(K, p=first / first.sum())]                                # ④ 第一个 token 按频率抽
        for i in range(1, 16):
            l, u = ctx(seq, i)
            seq.append(rng.choice(K, p=dist(i, l, u)))                              # ⑤ next-token：查表、按概率抽——16 步
        return np.array(seq)
    gen = np.stack([sample() for _ in range(60)])
    imgs = np.stack([unpatchify(CODEBOOK[t]) for t in gen])
    ll = np.mean([np.log(first[s[0]] / first.sum()) + sum(np.log(dist(i, *ctx(s, i))[s[i]]) for i in range(1, 16)) for s in TOKENS])
    print(f"  训练序列的平均对数似然 {ll:.2f}（每 token {ll/16:.2f}，即平均从约 {np.exp(-ll/16):.1f} 个候选里选）；随机猜是每 token log(1/32) = {np.log(1/32):.2f}")
    print(f"  生成一张图 = 16 次「查表 → 抽样」；一个 1024² 的图在 f16 下是 4096 次——这就是 AR 图像生成慢的原因")
    fig, axes = plt.subplots(5, 12, figsize=(7.6, 3.3))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im, cmap="gray_r", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("栅格顺序逐 token 生成 16 个 token、查码本解码——60 张「手写数字」（计数模型，不是神经网络）", fontsize=9)
    save(fig, "09-ar-samples")


def run_steps():
    print("=== 4. 生成一张图要几步：AR vs 扩散 ===")
    for name, tokens, steps in (("栅格 AR，256² f16", 256, 256), ("栅格 AR，1024² f16", 4096, 4096), ("VAR next-scale", 680, 10), ("MaskGIT 并行 mask", 256, 8), ("扩散 DDIM", 4096, 50), ("扩散 + 蒸馏", 4096, 4)):
        print(f"  {name:<22} token / patch 数 {tokens:>5}，串行步数 {steps:>5}")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["vq", "fsq", "ar", "steps"]):
        if w in ("ar",) and "CODEBOOK" not in globals():
            run_vq()
        globals()[f"run_{w}"](); print()
