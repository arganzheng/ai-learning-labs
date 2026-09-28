"""案例（经典 ML 07）：K-Means 的两个经典用法——客户分群（RFM）与图像颜色量化。
https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html

    python case_07_customer_segments.py          # 全部：rfm colors
    python case_07_customer_segments.py rfm

rfm     UCI Online Retail 54 万行交易 → 4,339 个客户的 R/F/M → K-Means 分 4 群 → 每群画像
colors  一张 640×427 的照片，96,615 种颜色 → K-Means 压成 2 / 4 / 16 / 64 种
图输出到 out/case-07-*.svg。
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _data import online_retail
from _plot import C, plt, save


def exp_rfm():
    print("=== A. 客户分群：RFM + K-Means ===")
    raw = online_retail()
    df = raw[raw.customer_id.notna() & (raw.quantity > 0) & (raw.unit_price > 0) & ~raw.invoice.str.startswith("C")].copy()
    df["amount"] = df.quantity * df.unit_price
    print(f"  原始 {len(raw):,} 行交易；去掉没有客户号的 {raw.customer_id.isna().sum():,} 行、退货 / 取消 {(raw.quantity <= 0).sum():,} 行 → {len(df):,} 行，"
          f"{df.customer_id.nunique():,} 个客户，{df.invoice.nunique():,} 张发票，{df.invoice_date.min():%Y-%m-%d} 到 {df.invoice_date.max():%Y-%m-%d}")

    # 每个客户三个数：R 最近一次购买距今几天、F 买过几次、M 一共花了多少
    now = df.invoice_date.max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(R=("invoice_date", lambda d: (now - d.max()).days),
                                        F=("invoice", "nunique"),
                                        M=("amount", "sum"))
    print(f"\n  RFM 表（{len(rfm):,} 个客户）：")
    print("  " + rfm.describe().loc[["mean", "50%", "min", "max"]].round(1).to_string().replace("\n", "\n  "))
    print("  三列都是长尾：M 中位数 £670、最大 £280,206——直接算欧氏距离会被几个大客户主导，所以先 log1p 再标准化。")

    Xs = StandardScaler().fit_transform(np.log1p(rfm.values))

    # 选 k
    print("\n  k 怎么选（惯性 = 各点到自己簇中心的距离平方和；轮廓系数 ∈ [-1, 1] 越大簇越分明）：")
    ks = range(2, 9); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=0).fit(Xs)
        inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
        print(f"    k = {k}  惯性 {km.inertia_:7.0f}  轮廓系数 {sil[-1]:.3f}")
    print("  轮廓系数 k=2 最高——但两群（活跃 / 不活跃）对业务没用；k=4 是惯性拐点附近且每群都能起名字，业务上取 4。")

    k = 4
    km = KMeans(k, n_init=10, random_state=0).fit(Xs)
    rfm["cluster"] = km.labels_
    prof = rfm.groupby("cluster").agg(人数=("R", "size"), R均值=("R", "mean"), F均值=("F", "mean"), M均值=("M", "mean"), M合计=("M", "sum")).round(0)
    prof["M占比"] = (prof.M合计 / prof.M合计.sum() * 100).round(1)
    # 按"价值"排序并起名
    order = prof.sort_values("M均值", ascending=False).index.tolist()
    names = {order[0]: "冠军：最近来过、买得最多", order[1]: "忠实：常来、花得不少", order[2]: "新客 / 低频：最近来过一两次", order[3]: "流失中：很久没来"}
    # 修正：按 R 判断"流失"——R 最大的那群叫流失
    r_max = prof.R均值.idxmax()
    if r_max != order[3]:
        names[r_max], names[order[3]] = names[order[3]], names[r_max]
    prof["画像"] = [names[i] for i in prof.index]
    print(f"\n  k = {k} 的四群画像：")
    print("  " + prof.loc[order].to_string().replace("\n", "\n  "))
    print("  解读：一群人数最少但贡献了一半以上的营业额（冠军），一群人数最多但很久没来（流失中）——"
          "运营动作按群分：冠军维护、忠实升级、新客促复购、流失召回。K-Means 没有'发现'这四群，"
          "它只是把三维空间切成四块；给每块起名、决定动作，是人的事。")

    # 图 1：F vs M 散点，按簇着色 + k 选择
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), width_ratios=[1.4, 1])
    ax = axes[0]
    cols = [C["red"], C["blue"], C["green"], C["orange"]]
    for i, c in enumerate(order):
        sub = rfm[rfm.cluster == c]
        ax.scatter(sub.R, sub.M, s=4, alpha=0.5, c=cols[i], label=f"{names[c].split('：')[0]}（{len(sub)}）")
    ax.set_yscale("log"); ax.set_xlabel("R：距上次购买（天）"); ax.set_ylabel("M：总消费 £")
    ax.set_title(f"{len(rfm):,} 个客户按 K-Means 分 4 群（颜色）"); ax.legend(fontsize=7, markerscale=3)
    ax = axes[1]
    ax.plot(list(ks), inertia, "o-", c=C["blue"]); ax.set_xlabel("k"); ax.set_ylabel("惯性", color=C["blue"])
    ax2 = ax.twinx(); ax2.plot(list(ks), sil, "s--", c=C["orange"]); ax2.set_ylabel("轮廓系数", color=C["orange"])
    ax2.spines["right"].set_visible(True)
    ax.set_title("k 怎么选：惯性拐点 vs 轮廓系数")
    save(fig, "case-07-rfm-clusters")

    # 图 2：四群画像
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    labels = [names[c].split("：")[0] for c in order]
    for ax, col, title in zip(axes, ["R均值", "F均值", "M均值"], ["R：距上次购买（天）", "F：购买次数", "M：总消费 £"]):
        ax.bar(labels, prof.loc[order, col], color=cols)
        ax.set_title(title); ax.tick_params(axis="x", labelsize=7)
    save(fig, "case-07-rfm-profiles")


def exp_colors():
    print("\n=== B. 颜色量化：一张照片近 10 万种颜色 → K 种 ===")
    img = load_sample_image("china.jpg")                      # [427, 640, 3] uint8，scikit-learn 自带的示例图（颐和园）
    h, w, _ = img.shape
    pixels = img.reshape(-1, 3).astype(np.float64) / 255      # 273,280 个像素，每个是 RGB 空间里一个三维的点
    print(f"  图 {w}×{h}，{len(pixels):,} 个像素，{len(np.unique(img.reshape(-1, 3), axis=0)):,} 种不同的颜色；原图 24 位/像素 = {len(pixels) * 3 / 1024:,.0f} KB")
    rng = np.random.default_rng(0)
    sample = pixels[rng.choice(len(pixels), 10000, replace=False)]  # 拟合只用 1 万个像素就够
    outs = {}
    for k in [2, 4, 16, 64]:
        km = KMeans(k, n_init=4, random_state=0).fit(sample)
        labels = km.predict(pixels)
        rec = km.cluster_centers_[labels]
        mse = np.mean((rec - pixels) ** 2) * 255 ** 2
        bits = np.ceil(np.log2(k))
        outs[k] = rec.reshape(h, w, 3)
        print(f"  k = {k:<3} 每像素 {bits:.0f} 位 + 调色板 {k}×3 字节 → {len(pixels) * bits / 8 / 1024:,.0f} KB（{bits / 24:.1%}）；均方误差 {mse:6.1f}（0–255 尺度）")
    print("  解读：每个像素是 RGB 空间的一个点，K-Means 找 k 个'代表色'（簇中心），每个像素换成离它最近的代表色——"
          "这是 GIF 调色板、早期显示器 256 色模式的算法。k=64 已经很难看出差别，文件只有原来的 1/4；k=16 是 1/6。")

    fig, axes = plt.subplots(1, 5, figsize=(7.6, 1.6))
    for ax, (k, im) in zip(axes, [("原图", img / 255)] + [(k, outs[k]) for k in [2, 4, 16, 64]]):
        ax.imshow(im); ax.axis("off"); ax.set_title(k if isinstance(k, str) else f"k = {k} 种颜色", fontsize=8)
    save(fig, "case-07-color-quantization")


EXPS = {"rfm": exp_rfm, "colors": exp_colors}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
