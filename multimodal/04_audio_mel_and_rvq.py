"""多模态（04）语音上篇：波形 → STFT → mel 谱（手写）；向量量化与残差向量量化（RVQ）toy。

    python 04_audio_mel_and_rvq.py            # 全部：mel rvq
"""
import sys

import numpy as np

from _plot import C, plt, save

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- 1. mel 谱
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)                              # mel 刻度：低频密、高频疏

def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)

def mel_filterbank(n_mels, n_fft, sr):
    """[n_mels, n_fft//2+1]：每一行是一个三角滤波器，把 FFT 的频点合并成一个 mel 频带。"""
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))     # n_mels+2 个边界，在 mel 轴上等距
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        lo, c, hi = bins[m], bins[m + 1], bins[m + 2]
        fb[m, lo:c] = (np.arange(lo, c) - lo) / max(1, c - lo)
        fb[m, c:hi] = (hi - np.arange(c, hi)) / max(1, hi - c)
    return fb

def log_mel(wave, sr=16000, win=400, hop=160, n_mels=80):
    """① 分帧（25 ms 窗、10 ms 步）② 加窗做 FFT 取幅度平方 ③ 乘 mel 滤波器 ④ 取对数。"""
    frames = np.stack([wave[i:i + win] * np.hanning(win) for i in range(0, len(wave) - win, hop)])   # [T, 400]
    power = np.abs(np.fft.rfft(frames, n=win)) ** 2                                                   # [T, 201]
    mel = power @ mel_filterbank(n_mels, win, sr).T                                                   # [T, 80]
    return np.log(mel + 1e-10)


def run_mel():
    print("=== 1. 从波形到 log-mel 谱：一段 1 秒的合成「语音」 ===")
    sr = 16000; t = np.arange(sr) / sr
    f0 = 120 + 40 * np.sin(2 * np.pi * 1.5 * t)                                   # 基频 120 Hz 上下起伏（像说话的音调）
    wave = sum((1 / k) * np.sin(2 * np.pi * k * np.cumsum(f0) / sr) for k in range(1, 12))   # 11 个谐波：像元音
    wave[: sr // 4] *= np.linspace(0, 1, sr // 4); wave[-sr // 4:] *= np.linspace(1, 0, sr // 4)
    wave += 0.02 * rng.normal(size=sr)
    wave[int(0.55 * sr):int(0.62 * sr)] += 0.3 * rng.normal(size=int(0.07 * sr))   # 一段 70 ms 的「摩擦音」噪声
    M = log_mel(wave)
    print(f"  波形：{len(wave)} 个采样点（16 kHz × 1 s）")
    print(f"  分帧：窗 400 点 = 25 ms，步长 160 点 = 10 ms → {M.shape[0]} 帧")
    print(f"  每帧 FFT 得 201 个频点，乘 80 个 mel 三角滤波器 → 80 维；log-mel 形状 {M.shape}（真实 Whisper 输入：3000 帧 × 80）")
    print(f"  数据量：16000 个数 → {M.size} 个数，压缩 {16000 / M.size:.1f} 倍；一秒语音 = {M.shape[0]} 个 80 维向量")
    fb = mel_filterbank(80, 400, sr)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1.2, 1, 1.4]})
    axes[0].plot(t[:800], wave[:800], lw=0.6, color=C["blue"]); axes[0].set_title("波形（前 50 ms，800 个点）", fontsize=9); axes[0].set_xlabel("秒")
    axes[0].axvspan(0, 0.025, color=C["orange"], alpha=0.2); axes[0].text(0.0125, wave[:800].max() * 0.9, "一帧 25 ms", ha="center", fontsize=7, color=C["orange"])
    freqs = np.fft.rfftfreq(400, 1 / sr)
    for m in range(0, 80, 8):
        axes[1].plot(freqs, fb[m], lw=0.7)
    axes[1].set_title("mel 滤波器（每 8 个画 1 个）", fontsize=9); axes[1].set_xlabel("Hz"); axes[1].set_xlim(0, 8000)
    axes[2].imshow(M.T, origin="lower", aspect="auto", cmap="magma", extent=[0, 1, 0, 80]); axes[2].set_title(f"log-mel 谱：{M.shape[0]} 帧 × 80 维", fontsize=9)
    axes[2].set_xlabel("秒"); axes[2].set_ylabel("mel 频带")
    save(fig, "04-wave-to-mel")


# ---------------------------------------------------------------- 2. VQ 与 RVQ
def kmeans(X, K, iters=30):
    C_ = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C_[None]) ** 2).sum(-1).argmin(1)
        C_ = np.array([X[lab == k].mean(0) if np.any(lab == k) else C_[k] for k in range(K)])
    return C_

def rvq_encode(z, codebooks):
    """逐级量化残差：返回每级选中的码字下标 [N_q, n] 与重建 ẑ。"""
    r, codes, z_hat = z.copy(), [], np.zeros_like(z)
    for cb in codebooks:                                                # 每级一个码本 [K, d]
        k = ((r[:, None] - cb[None]) ** 2).sum(-1).argmin(1)            # ① 在这一级码本里找离残差最近的码字
        codes.append(k)
        z_hat += cb[k]                                                  # ② 累加到重建
        r = r - cb[k]                                                   # ③ 剩下的残差交给下一级
    return np.array(codes), z_hat

def train_rvq(Z, K, n_q):
    """训练：第 i 级码本 = 对第 i−1 级残差做 k-means。"""
    codebooks, r = [], Z.copy()
    for _ in range(n_q):
        cb = kmeans(r, K)
        codebooks.append(cb)
        k = ((r[:, None] - cb[None]) ** 2).sum(-1).argmin(1)
        r = r - cb[k]
    return codebooks


def run_rvq():
    print("=== 2. 向量量化（VQ）与残差向量量化（RVQ）：2000 个 16 维向量 ===")
    d, n, K = 16, 2000, 64
    Z = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) * 0.5 + rng.normal(size=(1, d))   # 有相关结构的「编码器输出」
    tot = (Z ** 2).mean()
    # 单个码本 K=64 的 VQ
    cb1 = kmeans(Z, K); k = ((Z[:, None] - cb1[None]) ** 2).sum(-1).argmin(1); err1 = ((Z - cb1[k]) ** 2).mean()
    print(f"  单码本 VQ，K = {K}（6 bit）：相对重建误差 {err1 / tot:.3f}")
    # 手算一个 2 维小例子
    z = np.array([3.2, 1.1]); cbA = np.array([[3.0, 1.0], [-3.0, 1.0], [0.0, -2.0]]); cbB = np.array([[0.2, 0.1], [-0.2, 0.1], [0.0, -0.1]])
    kA = ((z - cbA) ** 2).sum(1).argmin(); r1 = z - cbA[kA]; kB = ((r1 - cbB) ** 2).sum(1).argmin(); r2 = r1 - cbB[kB]
    print(f"  手算 2 维例子：z = {z}；第 1 级码本选第 {kA} 个 {cbA[kA]}，残差 {np.round(r1, 2)}；第 2 级选第 {kB} 个 {cbB[kB]}，残差 {np.round(r2, 2)}")
    print(f"     重建 ẑ = {cbA[kA]} + {cbB[kB]} = {cbA[kA] + cbB[kB]}，两级码本各 3 项就表示了 3×3 = 9 种组合；码字下标 ({kA}, {kB}) 就是这个向量的「token」")
    # RVQ 多级
    n_q = 8
    cbs = train_rvq(Z, K, n_q)
    codes, _ = rvq_encode(Z, cbs)
    errs, rnorm = [], []
    r = Z.copy()
    for i in range(n_q):
        _, zh = rvq_encode(Z, cbs[: i + 1]); errs.append(((Z - zh) ** 2).mean() / tot)
        kk = ((r[:, None] - cbs[i][None]) ** 2).sum(-1).argmin(1); r = r - cbs[i][kk]; rnorm.append(np.sqrt((r ** 2).mean()))
    print(f"  RVQ {n_q} 级、每级 K = {K}：只用前 n 级解码的相对误差 " + "  ".join(f"n={i+1}:{e:.3f}" for i, e in enumerate(errs)))
    print(f"  每级之后残差的均方根：" + "  ".join(f"{x:.2f}" for x in rnorm) + "（逐级变小：后面的码本在越来越小的尺度上精修）")
    print(f"  等效码本大小 {K}^{n_q} = 2^{n_q * 6}，每个向量 {n_q * 6} bit；一个 2^{n_q*6} 项的单码本根本存不下")
    print(f"  码字使用情况：第 1 级用到 {len(np.unique(codes[0]))}/{K} 个码字，第 {n_q} 级用到 {len(np.unique(codes[-1]))}/{K} 个")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    axes[0].plot(range(1, n_q + 1), errs, "o-", color=C["blue"]); axes[0].axhline(err1 / tot, color=C["gray"], ls="--", lw=0.8)
    axes[0].text(4.5, err1 / tot * 1.15, f"单码本 VQ（K={K}）", fontsize=7, color=C["gray"])
    axes[0].set_xlabel("解码时用的码本级数 n"); axes[0].set_ylabel("相对重建误差"); axes[0].set_title("前 n 级码本重建：级数越多越精细", fontsize=9); axes[0].set_yscale("log")
    axes[1].bar(range(1, n_q + 1), rnorm, color=C["orange"]); axes[1].set_xlabel("级"); axes[1].set_ylabel("该级之后残差的均方根"); axes[1].set_title("每级去掉一部分，残差递减", fontsize=9)
    save(fig, "04-rvq")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["mel", "rvq"]):
        globals()[f"run_{w}"](); print()
