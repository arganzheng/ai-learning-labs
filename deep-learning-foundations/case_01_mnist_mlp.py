"""案例（DL 01）：MNIST 手写数字，NumPy 从零——一个 784-256-10 的 MLP 训到 97.6%，并把过程画出来。
https://arganzheng.life/backpropagation-by-hand.html

    python case_01_mnist_mlp.py

复用 01_backprop.py 的 dlf.nn 基座；输出 out/case-01-*.svg。
"""
import time

import numpy as np

from _plot import C, plt, save
from dlf import data, nn


def draw_mlp(sizes=(784, 256, 10), names=("输入：28×28 像素", "隐藏层：256 个 ReLU", "输出：10 个类别的 logits")):
    """画一张 MLP 结构图：三列节点、连线、形状与参数量。"""
    fig, ax = plt.subplots(figsize=(7.6, 3.8)); ax.axis("off"); ax.set_xlim(-0.5, 5.2); ax.set_ylim(-1.55, 1.45); ax.set_aspect("equal")
    XC = [0, 1.8, 3.6]
    shown = [12, 10, 10]
    ys = [np.linspace(-1, 1, k) for k in shown]
    for col, (k, y) in enumerate(zip(shown, ys)):
        for j, yy in enumerate(y):
            if col == 0 and j in (5, 6):
                ax.text(XC[col], yy, "⋮", ha="center", va="center", fontsize=10, color=C["gray"]); continue
            if col == 1 and j in (4, 5):
                ax.text(XC[col], yy, "⋮", ha="center", va="center", fontsize=10, color=C["gray"]); continue
            color = [C["gray"], C["blue"], C["orange"]][col]
            if col == 2 and j == 7:
                color = C["red"]
            ax.add_patch(plt.Circle((XC[col], yy), 0.055, color=color, zorder=3))
            if col == 2:
                ax.text(XC[col] + 0.13, yy, str(j), va="center", fontsize=7.5, color=C["red"] if j == 7 else C["gray"])
    for c in range(2):
        for ya in ys[c]:
            for yb in ys[c + 1]:
                ax.plot([XC[c], XC[c + 1]], [ya, yb], c="#cccccc", lw=0.3, zorder=1)
    for col, (n, name) in enumerate(zip(sizes, names)):
        ax.text(XC[col], 1.18, f"{name}\n[B, {n}]", ha="center", fontsize=8.5)
    ax.text(0.9, -1.3, f"W₁ [784, 256] + b₁ [256]\n= {784 * 256 + 256:,} 个参数\n矩阵乘 + ReLU", ha="center", fontsize=8, color=C["blue"])
    ax.text(2.7, -1.3, f"W₂ [256, 10] + b₂ [10]\n= {256 * 10 + 10:,} 个参数\n矩阵乘 → softmax → 交叉熵", ha="center", fontsize=8, color=C["orange"])
    ax.text(3.95, ys[2][7], "← 标签 y = 7：\n这一格的概率要大", va="center", fontsize=7.5, color=C["red"])
    ax.set_title(f"两层 MLP：784 → 256 → 10，共 {784 * 256 + 256 + 256 * 10 + 10:,} 个参数", fontsize=10)
    save(fig, "case-01-mlp-structure")


def main():
    rng = np.random.default_rng(0)
    X, y = data.load_mnist("train"); Xt, yt = data.load_mnist("test")
    print(f"数据：训练 {len(X):,} 张、测试 {len(Xt):,} 张，每张 28×28 = 784 个像素，10 类")
    net = nn.MLP([784, 256, 10], rng)
    nparams = sum(P.size for P, _ in net.params())
    print(f"模型：784 → 256（ReLU）→ 10，{nparams:,} 个参数；初始 loss 应 ≈ ln 10 = {np.log(10):.3f}")
    draw_mlp()

    # 训练并记录每个 epoch
    epochs, bs, lr = 15, 128, 0.1
    hist = {"loss": [], "acc": [], "step_loss": []}
    n = len(X); t0 = time.time()
    loss0, _, _ = nn.softmax_ce(net.forward(X[:bs]), y[:bs]); print(f"  初始 loss {loss0:.3f}")
    for ep in range(epochs):
        perm = rng.permutation(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, d, _ = nn.softmax_ce(net.forward(X[idx]), y[idx]); tot += loss * len(idx)
            net.backward(d)
            for P, dP in net.params():
                P -= lr * dP
            if ep == 0 and (i // bs) % 5 == 0:
                hist["step_loss"].append(loss)
        acc = (net.forward(Xt).argmax(1) == yt).mean()
        hist["loss"].append(tot / n); hist["acc"].append(acc)
        if ep + 1 in (1, 2, 5, 10, 15):
            print(f"  epoch {ep + 1:<3d} train loss {tot / n:.4f}  test acc {acc * 100:.2f}%")
    print(f"  15 个 epoch 共 {time.time() - t0:.0f}s（笔记本 CPU，纯 NumPy）")

    pred = net.forward(Xt).argmax(1)
    wrong = np.where(pred != yt)[0]
    print(f"  测试集 {len(Xt):,} 张错 {len(wrong)} 张（{len(wrong) / len(Xt):.2%}）")
    cm = np.zeros((10, 10), int)
    for a, b in zip(yt, pred):
        cm[a, b] += 1
    off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    print("  最常混的三对：" + "，".join(f"{i}→{j} {c} 次" for c, i, j in sorted(off, reverse=True)[:3]))

    # 图：训练曲线 + 错分样本
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), width_ratios=[1, 1, 1.2])
    ax = axes[0]
    ax.plot(np.arange(len(hist["step_loss"])) * 5, hist["step_loss"], c=C["blue"], lw=0.8)
    ax.axhline(np.log(10), ls="--", c=C["gray"], lw=0.8); ax.text(5, np.log(10) + 0.05, "ln 10 = 2.30（瞎猜）", fontsize=7, color=C["gray"])
    ax.set_xlabel("第 1 个 epoch 的步数"); ax.set_ylabel("batch loss"); ax.set_title("前 469 步：从瞎猜到 0.3")
    ax = axes[1]
    ax.plot(range(1, epochs + 1), hist["loss"], "o-", ms=3, c=C["blue"], label="训练 loss")
    ax2 = ax.twinx(); ax2.plot(range(1, epochs + 1), [a * 100 for a in hist["acc"]], "s--", ms=3, c=C["orange"], label="测试准确率 %")
    ax2.spines["right"].set_visible(True); ax2.set_ylim(90, 100)
    ax.set_xlabel("epoch"); ax.set_title("15 个 epoch：97.6%"); ax.legend(loc="center right", fontsize=7); ax2.legend(loc="lower right", fontsize=7)
    ax = axes[2]; ax.axis("off"); ax.set_title(f"错分的 {len(wrong)} 张里的前 16 张（真实→预测）", fontsize=8.5)
    for i, j in enumerate(wrong[:16]):
        r, c = divmod(i, 8)
        ax.imshow(Xt[j].reshape(28, 28), cmap="gray_r", extent=(c, c + 0.9, -r * 1.4 - 0.9, -r * 1.4))
        ax.text(c + 0.45, -r * 1.4 - 1.0, f"{yt[j]}→{pred[j]}", ha="center", va="top", fontsize=6, color=C["red"])
    ax.set_xlim(0, 8); ax.set_ylim(-2.9, 0); ax.set_aspect("equal")
    save(fig, "case-01-training")


if __name__ == "__main__":
    main()
