"""降维（经典 ML 08）：PCA 的几何、手写 PCA（SVD）与 sklearn 对数、解释方差与重建、PCA 与 SVD / 协方差特征分解的关系、
PCA vs t-SNE 的二维图、embedding 的各向异性（真实句向量）、真实权重矩阵的奇异值谱（低秩直觉）。
https://arganzheng.life/dimensionality-reduction-pca-svd-tsne-and-umap.html

    python 08_dimensionality_reduction.py          # 全部：geometry pca reconstruct svd tsne anisotropy spectrum
    python 08_dimensionality_reduction.py anisotropy   # 需要 07 生成的 out/sentence_embeddings.npz（没有会自动生成）

图输出到 out/08-*.svg。
"""
import os
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from _plot import C, plt, save


# ---------------- 手写 PCA ----------------
def pca(X, k):
    Xc = X - X.mean(0)                                        # ① 中心化
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)         # ② 数据矩阵的 SVD：Vt 的每一行是一个主成分方向
    explained = S ** 2 / (len(X) - 1)                         # ③ 每个主成分的方差 = 奇异值² / (n−1)
    return Xc @ Vt[:k].T, Vt[:k], explained / explained.sum()  # ④ 投影到前 k 个方向 [n, k]；方向；各主成分的解释方差比


# ---------------- 1. 几何：找方差最大的方向 ----------------
def exp_geometry():
    print("=== 1. PCA 的几何：二维数据，哪个方向方差最大 ===")
    r = np.random.default_rng(0)
    X = r.normal(size=(200, 2)) @ np.array([[1.6, 0.9], [0.0, 0.7]])          # 一团拉长、倾斜的点
    Z, V, ratio = pca(X, 2)
    print(f"  两个主成分方向 {np.round(V[0], 3)}, {np.round(V[1], 3)}（互相垂直：点积 {V[0] @ V[1]:.1e}）")
    print(f"  解释方差比 {np.round(ratio, 3)}：第一主成分占 {ratio[0]:.1%}")
    Xc = X - X.mean(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    ax = axes[0]
    ax.scatter(Xc[:, 0], Xc[:, 1], s=6, color=C["blue"], alpha=0.6)
    for v, col, name in ((V[0], C["red"], "PC1"), (V[1], C["orange"], "PC2")):
        s = 2 * np.sqrt(ratio[0 if name == "PC1" else 1] * Xc.var(0).sum())         # 箭头长度 ∝ 该方向的标准差
        ax.annotate("", xy=v * s, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.text(*(v * s * 1.2), name, color=col, fontsize=8, ha="center", va="center")
    ax.set_aspect("equal"); ax.set_title("中心化后的数据与两个主成分方向", fontsize=8.5)
    for ax, v, name in ((axes[1], V[0], "PC1（方差最大的方向）"), (axes[2], np.array([0.0, 1.0]), "随手挑的方向（竖直）")):
        proj = (Xc @ v)[:, None] * v[None]
        ax.scatter(Xc[:, 0], Xc[:, 1], s=6, color=C["light"])
        ax.scatter(proj[:, 0], proj[:, 1], s=6, color=C["red"] if "PC1" in name else C["gray"])
        ax.set_aspect("equal"); ax.set_title(f"投影到{name}\n投影后方差 {np.var(Xc @ v):.2f}", fontsize=8.5)
        print(f"  投影到{name}：投影后的方差 {np.var(Xc @ v):.3f}")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    save(fig, "08-pca-geometry")
    print("  PCA 找的是投影后方差最大（= 丢掉的信息最少）的方向；第二个方向在与第一个垂直的方向里再找最大")
    print()


# ---------------- 2. 手写 vs sklearn：手写数字 ----------------
def exp_pca():
    print("=== 2. 手写 PCA（4 行 SVD）vs sklearn：手写数字 64 维 ===")
    d = load_digits()
    Z, V, ratio = pca(d.data, 64)
    sk = PCA().fit(d.data)
    Zs = sk.transform(d.data)
    sign = np.sign((Z[:, :10] * Zs[:, :10]).sum(0))              # 特征向量可以乘 −1，对齐符号后再比
    print(f"  前 10 维投影坐标与 sklearn 的最大差 {np.abs(Z[:, :10] * sign - Zs[:, :10]).max():.1e}（符号对齐后）")
    cum = np.cumsum(ratio)
    print("  前 k 个主成分解释的方差：" + "  ".join(f"k={k}: {cum[k-1]*100:.1f}%" for k in (1, 2, 5, 10, 20, 30)))
    k95 = int(np.searchsorted(cum, 0.95)) + 1
    print(f"  解释 95% 需要 {k95} 维（原 64 维）")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    ax = axes[0]
    ax.bar(range(1, 65), ratio * 100, color=C["blue"]); ax.set_xlabel("主成分序号"); ax.set_ylabel("解释方差比 (%)"); ax.set_title("每个主成分各解释多少方差", fontsize=8.5)
    ax = axes[1]
    ax.plot(range(1, 65), cum * 100, color=C["red"]); ax.axhline(95, color=C["gray"], ls="--", lw=0.8); ax.axvline(k95, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("前 k 个主成分"); ax.set_ylabel("累计解释方差 (%)"); ax.set_title(f"累计：{k95} 维到 95%，64 维到 100%", fontsize=8.5)
    save(fig, "08-explained-variance")
    print()


# ---------------- 3. 重建：用前 k 维还原图像 ----------------
def exp_reconstruct():
    print("=== 3. 用前 k 个主成分重建一张手写数字 ===")
    d = load_digits()
    X = d.data; mu = X.mean(0)
    Z, V, ratio = pca(X, 64)
    i = 7
    fig, axes = plt.subplots(1, 6, figsize=(7.6, 1.6))
    axes[0].imshow(X[i].reshape(8, 8), cmap="gray_r"); axes[0].set_title("原图（64 维）", fontsize=8)
    for ax, k in zip(axes[1:], (1, 2, 5, 10, 30)):
        rec = mu + Z[i, :k] @ V[:k]                                 # 均值 + 前 k 个坐标 × 前 k 个方向
        err = np.mean((rec - X[i]) ** 2)
        ax.imshow(rec.reshape(8, 8), cmap="gray_r"); ax.set_title(f"k = {k}\n重建误差 {err:.1f}", fontsize=8)
        print(f"  k = {k:>2}: 这张图的重建 MSE {err:6.2f}，全部 1797 张的平均 {np.mean((mu + Z[:, :k] @ V[:k] - X) ** 2):6.2f}")
    for a in axes: a.axis("off")
    save(fig, "08-reconstruction")
    print("  降维 = 只保留前 k 个坐标；重建 = 均值 + 坐标 × 方向。k 越大越像，30 维已看不出差别")
    print()


# ---------------- 4. PCA = 协方差特征分解 = SVD ----------------
def exp_svd():
    print("=== 4. 三种算法同一个答案：协方差矩阵特征分解、数据矩阵 SVD、sklearn ===")
    d = load_digits()
    Xc = d.data - d.data.mean(0)
    n = len(Xc)
    Cov = Xc.T @ Xc / (n - 1)                                     # 64×64 协方差矩阵
    evals, evecs = np.linalg.eigh(Cov)                            # 特征分解（eigh：对称矩阵）
    evals, evecs = evals[::-1], evecs[:, ::-1]                    # 从大到小
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    sk = PCA().fit(d.data)
    print(f"  协方差矩阵的特征值（前 3）     {np.round(evals[:3], 2)}")
    print(f"  奇异值² / (n−1)（前 3）         {np.round(S[:3]**2 / (n - 1), 2)}")
    print(f"  sklearn explained_variance_   {np.round(sk.explained_variance_[:3], 2)}")
    cos = np.abs((evecs[:, :3] * Vt[:3].T).sum(0))
    print(f"  特征向量与 SVD 的 V 的前 3 列：|余弦| {np.round(cos, 6)}（同一方向，可能差一个符号）")
    print(f"  64×64 协方差特征分解 vs 1797×64 数据 SVD：数值上 SVD 更稳（不用先算 XᵀX，条件数是平方关系），实现都走 SVD")
    print()


# ---------------- 5. PCA vs t-SNE 的二维图 ----------------
def exp_tsne():
    print("=== 5. 二维可视化：PCA（线性）vs t-SNE（非线性）===")
    d = load_digits()
    Z, _, ratio = pca(d.data, 2)
    print(f"  PCA 前两维只解释 {ratio[:2].sum():.1%} 的方差")
    Zt = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(d.data)
    Zt50 = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(PCA(30).fit_transform(d.data))
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    for name, ZZ in (("PCA 2 维", Z), ("t-SNE 2 维", Zt), ("PCA 30 → t-SNE 2", Zt50)):
        acc = cross_val_score(KNeighborsClassifier(5), ZZ, d.target, cv=5).mean()
        print(f"  在{name}坐标上做 KNN 分类（5 折）：{acc:.3f}——衡量二维图上同类点有多聚在一起")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, ZZ, title in ((axes[0], Z, f"PCA 前两维（解释 {ratio[:2].sum():.0%} 方差）：0 和 1 分得开，3/5/8 混在一起"),
                          (axes[1], Zt50, "PCA 30 维 → t-SNE 2 维：十类各成一团")):
        sc = ax.scatter(ZZ[::3, 0], ZZ[::3, 1], c=d.target[::3], cmap="tab10", s=5)
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, ticks=range(10), shrink=0.8, label="数字")
    save(fig, "08-pca-vs-tsne")
    print("  t-SNE 只保局部结构：图上两团的距离没有意义；有随机性、不可逆；标准做法先 PCA 到 30–50 维再 t-SNE / UMAP")
    print()


# ---------------- 6. embedding 的各向异性（07 的真实句向量） ----------------
def exp_anisotropy():
    print("=== 6. embedding 的各向异性：78 句真实句向量，任意两句余弦都很高 ===")
    from _sentences import embeddings
    E, texts, labels, names = embeddings()
    def cos_matrix(M):
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True); return Mn @ Mn.T
    iu = np.triu_indices(len(E), 1)
    same = (labels[:, None] == labels[None, :])[iu]
    S_raw = cos_matrix(E)[iu]
    Ec = E - E.mean(0)                                            # 减均值（各向异性的最简单修法）
    S_cen = cos_matrix(Ec)[iu]
    # 再去掉前 2 个主成分（all-but-the-top）：把所有向量共有的那几个大方向扣掉
    U, Sv, Vt = np.linalg.svd(Ec, full_matrices=False)
    Ew = Ec - (Ec @ Vt[:2].T) @ Vt[:2]
    S_wh = cos_matrix(Ew)[iu]
    print(f"  {'':<14} {'全部对的余弦均值':>12} {'同主题对':>8} {'不同主题对':>8} {'两者之差':>8}")
    for name, S in (("原始", S_raw), ("减均值", S_cen), ("减均值+去前 2 主成分", S_wh)):
        print(f"  {name:<14} {S.mean():>12.3f} {S[same].mean():>8.3f} {S[~same].mean():>8.3f} {S[same].mean() - S[~same].mean():>8.3f}")
    _, _, ratio = pca(E, 10)
    print(f"  原始向量的 PCA：第一主成分解释 {ratio[0]:.1%}、前 3 个 {ratio[:3].sum():.1%}——所有向量挤在一个窄锥里")
    # 检索例子
    q = 3                                                          # 一句体育
    for name, M in (("原始", E), ("减均值", Ec)):
        Sm = cos_matrix(M)[q].copy(); Sm[q] = -1
        top = np.argsort(-Sm)[:3]
        print(f"  查询「{texts[q][:16]}…」{name}余弦 top-3：" + "；".join(f"{texts[t][:12]}…({Sm[t]:.2f})" for t in top))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    bins = np.linspace(-0.6, 1.0, 50)
    for ax, S, title in ((axes[0], S_raw, "原始：全部对都在 0.5–1.0"), (axes[1], S_cen, "减均值后：同主题与不同主题分开了")):
        ax.hist(S[~same], bins=bins, color=C["gray"], alpha=0.7, label="不同主题的句子对")
        ax.hist(S[same], bins=bins, color=C["red"], alpha=0.7, label="同主题的句子对")
        ax.set_xlabel("余弦相似度"); ax.set_title(title, fontsize=8.5); ax.legend(frameon=False, fontsize=7)
    save(fig, "08-anisotropy")
    print("  各向异性让「余弦 0.8」失去含义；减均值（再去掉前几个主成分）后同主题与不同主题的差距拉开；检索分数全都很高又分不开时先做一次 PCA 看解释方差")
    print()


# ---------------- 7. 真实权重矩阵的奇异值谱：低秩直觉 ----------------
def exp_spectrum():
    print("=== 7. 低秩直觉：一个真实权重矩阵的奇异值谱 vs 同尺寸随机矩阵 ===")
    try:
        import glob
        from safetensors import safe_open
        path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/model.safetensors"))[0]
        with safe_open(path, framework="pt") as f:                                            # 权重是 bf16，用 torch 读再转 float32
            W = f.get_tensor("model.layers.12.self_attn.q_proj.weight").float().numpy()      # [896, 896]
    except Exception as e:
        print(f"  （需要本地缓存的 Qwen2.5-0.5B，跳过：{type(e).__name__}）"); return
    r = np.random.default_rng(0)
    R = r.normal(0, W.std(), W.shape)
    sW = np.linalg.svd(W, compute_uv=False); sR = np.linalg.svd(R, compute_uv=False)
    def energy_rank(s, frac=0.9):
        c = np.cumsum(s ** 2) / np.sum(s ** 2); return int(np.searchsorted(c, frac)) + 1
    print(f"  q_proj 权重 {W.shape}：前 10 个奇异值 {np.round(sW[:10], 1)}")
    print(f"  同尺寸随机矩阵：       前 10 个奇异值 {np.round(sR[:10], 1)}")
    print(f"  解释 90% 能量需要的秩：真实权重 {energy_rank(sW)} / 896，随机矩阵 {energy_rank(sR)} / 896")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(sW / sW[0], color=C["red"], label="Qwen2.5-0.5B 第 12 层 q_proj（896×896）")
    ax.plot(sR / sR[0], color=C["gray"], label="同尺寸、同标准差的随机高斯矩阵")
    ax.set_yscale("log"); ax.set_xlabel("奇异值序号"); ax.set_ylabel("奇异值 / 最大奇异值（对数轴）"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("真实权重的奇异值比随机矩阵衰减得快，但本身不低秩", fontsize=8.5)
    save(fig, "08-weight-spectrum")
    print("  真实权重的谱比随机矩阵衰减得快（90% 能量 299 维 vs 458 维），但 W 本身远不是低秩的；LoRA 的低秩假设说的是微调的改动 ΔW（r = 8–64），不是 W——那是一个更强、也已被实践验证的假设")
    print()


EXPS = {"geometry": exp_geometry, "pca": exp_pca, "reconstruct": exp_reconstruct, "svd": exp_svd, "tsne": exp_tsne,
        "anisotropy": exp_anisotropy, "spectrum": exp_spectrum}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
