"""多模态（02）VLM 结构：三类 connector 的形状与信息损失、token 预算随分辨率的变化。

    python 02_connectors_and_resolution.py            # 全部：connectors budget
"""
import sys

import numpy as np
import torch

from _plot import C, plt, save

torch.manual_seed(0)
rng = np.random.default_rng(0)


def run_connectors():
    print("=== 1. 三类 connector：同一份 4×4=16 个 patch 特征（d_v = 8）进去，出来多少个 token、丢不丢信息 ===")
    H = W = 4; dv, d = 8, 12
    h = torch.randn(H * W, dv)                                          # 编码器输出：16 个 patch 特征 [16, 8]

    # ① MLP projector：逐 token，不改数量
    mlp = torch.nn.Sequential(torch.nn.Linear(dv, d), torch.nn.GELU(), torch.nn.Linear(d, d))
    z_mlp = mlp(h)
    print(f"  MLP projector：{tuple(h.shape)} → {tuple(z_mlp.shape)}；参数 {sum(p.numel() for p in mlp.parameters())}（d_v·d + d² 量级）")

    # ② 2×2 merge：相邻四个 patch 拼接成一个 4·d_v 维向量，再过 MLP
    grid = h.view(H, W, dv)                                              # 恢复成 4×4 的网格
    merged = grid.view(H // 2, 2, W // 2, 2, dv).permute(0, 2, 1, 3, 4).reshape(-1, 4 * dv)   # [4, 32]：每个新 token = 2×2 邻域的拼接
    mlp2 = torch.nn.Linear(4 * dv, d)
    z_merge = mlp2(merged)
    print(f"  2×2 merge：{tuple(h.shape)} → 拼接 {tuple(merged.shape)} → MLP {tuple(z_merge.shape)}；token 数 16 → 4，拼接不丢任何数（32 个数原样在）")
    # pixel shuffle 是同一件事：space-to-depth
    ps = torch.nn.functional.pixel_unshuffle(grid.permute(2, 0, 1)[None], 2)[0]   # [4·dv, 2, 2]
    ps = ps.permute(1, 2, 0).reshape(-1, 4 * dv)
    print(f"  pixel (un)shuffle 得到的矩阵与手工 2×2 拼接是否只差通道顺序：{torch.allclose(ps.sort(1).values, merged.sort(1).values)}")

    # ③ 平均池化：四个 patch 取平均，有损
    pooled = grid.view(H // 2, 2, W // 2, 2, dv).mean((1, 3)).reshape(-1, dv)      # [4, 8]
    lost = (grid.view(H // 2, 2, W // 2, 2, dv) - pooled.view(H // 2, 1, W // 2, 1, dv)).pow(2).mean()
    print(f"  2×2 平均池化：{tuple(h.shape)} → {tuple(pooled.shape)}；四个 patch 之间的差别被抹掉，抹掉的部分（均方）{lost:.2f}，占原方差的 {lost / h.var():.0%}")

    # ④ resampler / Q-Former：K 个可学习 query 对全部 patch 做 cross-attention
    K = 3
    queries = torch.nn.Parameter(torch.randn(K, d))                      # 与图片内容无关的 K 个 query
    Wq, Wk, Wv = torch.nn.Linear(d, d), torch.nn.Linear(dv, d), torch.nn.Linear(dv, d)
    att = torch.softmax(Wq(queries) @ Wk(h).T / d ** 0.5, dim=-1)        # [K, 16]：每个 query 在 16 个 patch 上的注意力
    z_res = att @ Wv(h)                                                  # [K, d]
    print(f"  resampler（K = {K} 个 query）：{tuple(h.shape)} → {tuple(z_res.shape)}；不论图有多少 patch，永远输出 {K} 个 token")
    print(f"     第 0 个 query 的注意力分布（16 个 patch）：{np.round(att[0].detach().numpy(), 2)}")

    # 图：16 个 patch → 三种 connector
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3))
    val = np.arange(16).reshape(4, 4)
    axes[0].imshow(val, cmap="Pastel1"); axes[0].set_title("编码器输出：4×4 个 patch 特征", fontsize=8)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, str(val[i, j]), ha="center", va="center", fontsize=8)
    axes[1].imshow(val, cmap="Pastel1"); axes[1].set_title("MLP：16 → 16，逐个映射", fontsize=8)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"z{val[i, j]}", ha="center", va="center", fontsize=7)
    axes[2].imshow(np.arange(4).reshape(2, 2), cmap="Pastel2"); axes[2].set_title("2×2 merge：16 → 4，邻域拼接", fontsize=8)
    labels = [["0,1\n4,5", "2,3\n6,7"], ["8,9\n12,13", "10,11\n14,15"]]
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, labels[i][j], ha="center", va="center", fontsize=7)
    axes[3].imshow(np.arange(3).reshape(3, 1), cmap="Set3", aspect=0.5); axes[3].set_title("resampler：16 → K=3\n每个 query 看全部 patch", fontsize=8)
    for i in range(3):
        axes[3].text(0, i, f"q{i}·全图", ha="center", va="center", fontsize=7)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "02-connectors")


def tokens_fixed(h, w, side=336, patch=14):
    return (side // patch) ** 2                                          # 不管图多大，都 resize 到 336²

def tokens_anyres(h, w, side=336, patch=14):
    """LLaVA-NeXT 的 select_best_resolution：在候选网格里选「有效像素最多、浪费最少」的那个。"""
    grids = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]             # (行, 列) 的候选网格，≤ 4 tile
    best, best_key = None, None
    for r, c in grids:
        gh, gw = r * side, c * side
        scale = min(gw / w, gh / h)                                      # 保持宽高比缩放到网格里
        eff = min(int(w * scale) * int(h * scale), w * h)                # 有效像素：放大不算
        waste = gw * gh - eff
        key = (eff, -waste)
        if best_key is None or key > best_key:
            best, best_key = (r, c), key
    return (best[0] * best[1] + 1) * (side // patch) ** 2                # tile 数 × 576 + 一张缩略图

def tokens_native(h, w, unit=28, lo=256, hi=1280):
    n = (round(h / unit) * round(w / unit))
    return int(min(max(n, lo), hi))                                       # Qwen2-VL：每 28×28 像素一个 token，夹在上下限之间


def run_budget():
    print("=== 2. 一张图占多少 token：三种分辨率策略 ===")
    cases = [("手机截图 1080×2400", 2400, 1080), ("A4 文档照片 3000×2100", 3000, 2100), ("小图标 64×64", 64, 64), ("横幅 400×1600", 400, 1600), ("普通照片 768×1024", 768, 1024)]
    print(f"  {'图':<22}{'固定 336²':>10}{'AnyRes ≤4 tile':>16}{'原生动态(28px)':>16}")
    for name, h, w in cases:
        print(f"  {name:<22}{tokens_fixed(h, w):>10}{tokens_anyres(h, w):>16}{tokens_native(h, w):>16}")
    print("  原生动态 = 像素数 / 784：64×64 只要 5 个 token（被下限 256 顶住），一页文档 3000×2100 = 8036 → 被上限 1280 截住")

    sides = np.arange(64, 2049, 16)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(sides, [tokens_fixed(s, s) for s in sides], color=C["gray"], label="固定 336²：永远 576")
    ax.plot(sides, [tokens_anyres(s, s) for s in sides], color=C["orange"], label="AnyRes：tile 数 × 576 + 缩略图（≤ 4 tile）", drawstyle="steps-post")
    ax.plot(sides, [tokens_native(s, s) for s in sides], color=C["blue"], label="原生动态：像素 / 784，夹在 [256, 1280]")
    ax.plot(sides, [(s / 28) ** 2 for s in sides], color=C["blue"], ls=":", lw=0.8, label="像素 / 784（不设上下限）")
    ax.set_xlabel("正方形图的边长（像素）"); ax.set_ylabel("进 LLM 的 token 数"); ax.set_ylim(0, 3200)
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("同一张图，三种策略给出的 token 数")
    save(fig, "02-token-budget")


def run_diagrams():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.transforms import Bbox

    L = {"blue": "#dbe7f5", "orange": "#fde8d0", "green": "#dcefd9",
         "gray": "#eeeeee", "dark": "#222222"}

    def check_layout(fig, axes, patches, inner):
        """断言：标签不压框不互压，框不互压，框内文字留边，所有元素不出图。"""
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        W, H = fig.canvas.get_width_height()
        figbox = Bbox.from_extents(0, 0, W, H)
        box_exts = [p.get_window_extent(rend) for p in patches]
        labels = []
        for a in np.atleast_1d(axes):
            labels += [t for t in a.texts if t not in inner and t.get_text()]
        lab_exts = [t.get_window_extent(rend) for t in labels]
        for t, e in zip(labels, lab_exts):
            for b in box_exts:
                assert not Bbox.from_extents(e.x0 - 2, e.y0 - 2, e.x1 + 2, e.y1 + 2).overlaps(b), (
                    f"label overlaps box: {t.get_text()!r}")
        for i in range(len(lab_exts)):
            for j in range(i + 1, len(lab_exts)):
                assert not lab_exts[i].overlaps(lab_exts[j]), "labels overlap"
        for i in range(len(box_exts)):
            for j in range(i + 1, len(box_exts)):
                a, b = box_exts[i], box_exts[j]
                nested = (a.contains(b.x0, b.y0) and a.contains(b.x1, b.y1)) or \
                         (b.contains(a.x0, a.y0) and b.contains(a.x1, a.y1))
                assert nested or not a.overlaps(b), "boxes overlap"
        for p, t in zip(patches, inner):
            pe = p.get_window_extent(rend)
            te = t.get_window_extent(rend)
            assert (te.x0 >= pe.x0 + 6 and te.x1 <= pe.x1 - 6
                    and te.y0 >= pe.y0 + 6 and te.y1 <= pe.y1 - 6), (
                        f"text within 6 px of box edge: {t.get_text()!r}")
        for e in box_exts + lab_exts:
            assert figbox.contains(e.x0, e.y0) and figbox.contains(e.x1, e.y1), (
                "artist leaves the figure canvas")

    def box(ax, patches, inner, xy, w, h, text, fc, fs=8):
        x, y = xy
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           facecolor=fc, edgecolor="#999999", linewidth=1)
        ax.add_patch(p)
        patches.append(p)
        inner.append(ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                             fontsize=fs, color=L["dark"]))

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color="#888888", linewidth=1.2))

    print("=== 3. LLaVA-1.5 的 MLP projector：真实维度 ===")
    n_patch, dv, d = 576, 1024, 4096
    p = dv * d + d + d * d + d
    print(f"  图 336×336×3 → patch 14×14 → {n_patch} 个 {dv} 维向量")
    print(f"  MLP：Linear({dv}→{d}) + GELU + Linear({d}→{d})，参数 {p:,} ≈ {p/1e6:.0f}M")
    print(f"  输出 {n_patch} 个 {d} 维 token，与文本 token 首尾相接进 decoder")

    fig, ax = plt.subplots(figsize=(10.5, 2.4))
    ax.set_xlim(0, 110); ax.set_ylim(0, 24); ax.axis("off")
    patches, inner = [], []
    box(ax, patches, inner, (1, 7), 13, 10, "一张图\n336×336×3", L["gray"])
    arrow(ax, 14.5, 12, 17.5, 12)
    box(ax, patches, inner, (18, 7), 15, 10, "ViT 切块\n24×24 = 576\n个 patch", L["blue"], fs=7.5)
    arrow(ax, 33.5, 12, 36.5, 12)
    box(ax, patches, inner, (37, 7), 12, 10, "576 行\n每行 1024 维", L["blue"], fs=7.5)
    arrow(ax, 49.5, 12, 52.5, 12)
    # MLP：外框 + 三个堆叠子框
    big = FancyBboxPatch((53, 0.5), 24, 23, boxstyle="round,pad=0.02",
                         facecolor=L["orange"], edgecolor="#999999", linewidth=1)
    ax.add_patch(big); patches.append(big)
    ax.text(65, 21.3, "同一个 MLP（每行独立，≈21M 参数）", ha="center", va="center",
            fontsize=7.5, color=L["dark"])
    inner.append(ax.texts[-1])
    for i, sub in enumerate(["Linear 1024→4096", "GELU", "Linear 4096→4096"]):
        sp = FancyBboxPatch((56, 14.5 - i * 4.7), 18, 3.6, boxstyle="round,pad=0.02",
                            facecolor="white", edgecolor="#999999", linewidth=1)
        ax.add_patch(sp); patches.append(sp)
        inner.append(ax.text(65, 16.3 - i * 4.7, sub, ha="center", va="center",
                             fontsize=7.5, color=L["dark"]))
    arrow(ax, 77.5, 12, 80.5, 12)
    box(ax, patches, inner, (81, 7), 12, 10, "576 行\n每行 4096 维", L["green"], fs=7.5)
    arrow(ax, 93.5, 12, 96.5, 12)
    box(ax, patches, inner, (97, 7), 12, 10, "与文本 token\n序列拼接", L["green"], fs=7.5)
    check_layout(fig, ax, patches, inner)
    save(fig, "02-mlp-projector")

    print("\n=== 4. resampler 的构造 toy：32 个 query 各读一块区域 ===")
    G, K, dv2 = 24, 32, 16
    yy, xx = np.mgrid[0:G, 0:G]
    keys = 0.3 * np.random.RandomState(0).randn(G * G, dv2)
    v_obj, v_txt, v_bg = np.eye(3, dv2, 0), np.eye(3, dv2, 1), np.eye(3, dv2, 2)
    obj = (yy < 9) & (xx < 9)                       # 左上角一个「物体」
    txt = (yy >= 12) & (yy <= 14)                   # 中间一条「文字带」
    bg = (yy > 17) & (xx > 17)                      # 右下角一块「背景」
    keys[obj.reshape(-1)] += 3 * v_obj[0]
    keys[txt.reshape(-1)] += 3 * v_txt[0]
    keys[bg.reshape(-1)] += 3 * v_bg[0]
    queries = 0.2 * np.random.RandomState(1).randn(K, dv2)
    for q, v in [(0, v_obj), (7, v_txt), (15, v_bg)]:
        queries[q] += 5 * v[0]                          # 构造：各对不同区域有亲和度
    att = torch.softmax(torch.tensor(queries).float() @ torch.tensor(keys).float().T / dv2 ** 0.5, -1).numpy()
    print("  三个展示 query 的前 5% patch 权重占比（均匀时应 ≈5%）：")
    for q in (0, 7, 15):
        m = np.sort(att[q])[::-1][: int(0.05 * G * G)].sum()
        print(f"    query {q}: {m:.0%}")
        assert m >= 0.30, "query 权重不够集中——热力图又会变成均匀洗色"
    print(f"  不论图有多少 patch，输出永远是 {K} 个 token")

    fig = plt.figure(figsize=(10.5, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[3.6, 1.5, 1.5, 1.5], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 116); ax.set_ylim(0, 30); ax.axis("off")
    patches, inner = [], []
    box(ax, patches, inner, (1, 17), 28, 10, "576 个 patch 特征\n作 key / value", L["blue"], fs=7.5)
    box(ax, patches, inner, (1, 3), 28, 10, "32 个可学习 query\n模型参数，与图无关", L["orange"], fs=7.5)
    box(ax, patches, inner, (37, 8), 34, 14, "cross-attention\nsoftmax(QKᵀ/√d)\n· V", L["gray"], fs=7)
    box(ax, patches, inner, (81, 10), 34, 10, "32 个输出 token\n每个 = patch\n的加权平均", L["green"], fs=7)
    arrow(ax, 29.5, 22, 36.5, 17)
    arrow(ax, 29.5, 8, 36.5, 13)
    arrow(ax, 71.5, 15, 80.5, 15)
    check_layout(fig, ax, patches, inner)
    titles = ["query 0 → 左上角物体", "query 7 → 文字带", "query 15 → 右下角背景"]
    vmax = att.max()
    for i, q in enumerate([0, 7, 15]):
        a2 = fig.add_subplot(gs[i + 1])
        im = a2.imshow(att[q].reshape(G, G), cmap="Oranges", vmin=0, vmax=vmax)
        a2.set_title(f"{titles[i]}\nmax {att[q].max():.2f}", fontsize=7.5)
        a2.set_xticks([]); a2.set_yticks([])
        fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    save(fig, "02-resampler-attention")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["connectors", "budget", "diagrams"]):
        globals()[f"run_{w}"](); print()
