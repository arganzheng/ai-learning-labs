"""案例（DL 05）：复现 LeNet-5（LeCun 等 1998）——MNIST 上第一个卷积网络，逐层形状、参数量、训练曲线、错分样本。
https://arganzheng.life/cnn-from-lenet-to-resnet-and-vit.html

    python case_05_lenet5.py

与 L2 第五篇（KNN 2.95%、SVM 1.43%）、L3 第一篇（两层 MLP 2.39%）同一份数据同一个划分。输出 out/case-05-*.svg。
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _plot import C, plt, save
from dlf import data

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class LeNet5(nn.Module):
    """1998 年原版的现代写法：tanh → ReLU，平均池化 → 最大池化，RBF 输出层 → softmax；结构（C1-S2-C3-S4-C5-F6-OUT）不变。"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)     # 28×28 补到 32×32 再卷 → 6@28×28
        self.c3 = nn.Conv2d(6, 16, 5)               # 6@14×14 → 16@10×10
        self.c5 = nn.Linear(16 * 5 * 5, 120)        # 原文 C5 是 16@5×5 → 120 的卷积，核正好覆盖全图，等价于全连接
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x, trace=None):
        def rec(name, t):
            if trace is not None:
                trace.append((name, tuple(t.shape[1:])))
            return t
        x = rec("输入", x)
        x = rec("C1 卷积 5×5, 1→6", F.relu(self.c1(x)))
        x = rec("S2 池化 2×2", F.max_pool2d(x, 2))
        x = rec("C3 卷积 5×5, 6→16", F.relu(self.c3(x)))
        x = rec("S4 池化 2×2", F.max_pool2d(x, 2))
        x = rec("拉平", x.flatten(1))
        x = rec("C5 全连接 400→120", F.relu(self.c5(x)))
        x = rec("F6 全连接 120→84", F.relu(self.f6(x)))
        return rec("输出 84→10", self.out(x))


def main():
    X, y = data.load_mnist("train"); Xt, yt = data.load_mnist("test")
    X = torch.tensor(X).view(-1, 1, 28, 28); Xt = torch.tensor(Xt).view(-1, 1, 28, 28)
    y = torch.tensor(y); yt = torch.tensor(yt)
    mu, sd = X.mean(), X.std(); X = (X - mu) / sd; Xt = (Xt - mu) / sd
    model = LeNet5().to(DEVICE)
    print(f"=== LeNet-5 复现（{DEVICE}）===")
    trace = []; model(X[:1].to(DEVICE), trace)
    print("  逐层形状与参数量：")
    params = {"C1 卷积 5×5, 1→6": model.c1, "C3 卷积 5×5, 6→16": model.c3, "C5 全连接 400→120": model.c5, "F6 全连接 120→84": model.f6, "输出 84→10": model.out}
    total = 0
    for name, shape in trace:
        p = sum(t.numel() for t in params[name].parameters()) if name in params else 0; total += p
        print(f"    {name:<22} → {str(shape):<14} {'参数 ' + f'{p:,}' if p else ''}")
    print(f"  共 {total:,} 个参数（第一篇的两层 MLP 是 203,530——LeNet-5 用 1/3 的参数）；两个卷积层只有 {sum(t.numel() for m in (model.c1, model.c3) for t in m.parameters()):,} 个参数，96% 的参数在三个全连接层")

    B, epochs = 128, 5
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    steps = epochs * (len(X) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    hist = []; t0 = time.time(); g = torch.Generator().manual_seed(0)
    Xd, yd = X.to(DEVICE), y.to(DEVICE); Xtd, ytd = Xt.to(DEVICE), yt.to(DEVICE)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X), generator=g).to(DEVICE); tot = 0.0
        for i in range(0, len(X) - B + 1, B):
            idx = perm[i:i + B]
            loss = F.cross_entropy(model(Xd[idx]), yd[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item() * B
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xtd[i:i + 2000]).argmax(1) for i in range(0, len(Xt), 2000)])
        acc = (pred == ytd).float().mean().item(); hist.append((tot / len(X), acc))
        print(f"  epoch {ep + 1}  train loss {tot / len(X):.4f}  test acc {acc * 100:.2f}%  ({time.time() - t0:.0f}s)")
    pred = pred.cpu().numpy(); err = np.where(pred != yt.numpy())[0]
    print(f"  测试集错 {len(err)} 张（{len(err) / len(yt):.2%}）；LeCun 等 1998 报的 LeNet-5 是 0.95%（他们训了 20 个 epoch、有数据增强的版本 0.8%）")
    print("  对照同一份 MNIST：KNN 2.95%（L2 第四篇）、RBF-SVM 1.43%（L2 第五篇）、两层 MLP 2.39%（第一篇）、LeNet-5 " + f"{len(err) / len(yt):.2%}——"
          "参数是 MLP 的 1/3，错误率降到 1/3。卷积赢在两处：每个 5×5 的核在整张图上共享，同一个笔画特征在哪都认得；池化让它对几个像素的平移不敏感。")

    # 图 1：第一层的 6 个 5×5 卷积核 + 一张图过 C1 后的 6 张特征图
    fig, axes = plt.subplots(2, 7, figsize=(7.6, 2.4))
    img = Xt[7:8].to(DEVICE)
    with torch.no_grad():
        fm = F.relu(model.c1(img))[0].cpu().numpy()
    axes[0, 0].imshow(Xt[7, 0], cmap="gray_r"); axes[0, 0].set_title("输入", fontsize=7)
    axes[1, 0].axis("off"); axes[1, 0].text(0.5, 0.5, "C1 的 6 个\n5×5 卷积核 ↓\n\n6 张特征图 →", ha="center", va="center", fontsize=7, transform=axes[1, 0].transAxes)
    k = model.c1.weight.detach().cpu().numpy()
    for i in range(6):
        axes[0, i + 1].imshow(fm[i], cmap="gray_r"); axes[0, i + 1].set_title(f"特征图 {i + 1}", fontsize=7)
        axes[1, i + 1].imshow(k[i, 0], cmap="RdBu", vmin=-abs(k).max(), vmax=abs(k).max())
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "case-05-lenet-filters")

    # 图 2：训练曲线 + 错分
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), width_ratios=[1, 1.4])
    ax = axes[0]
    ax.plot(range(1, epochs + 1), [h[0] for h in hist], "o-", ms=3, c=C["blue"], label="训练 loss")
    ax2 = ax.twinx(); ax2.plot(range(1, epochs + 1), [h[1] * 100 for h in hist], "s--", ms=3, c=C["orange"], label="测试准确率 %"); ax2.set_ylim(97, 100); ax2.spines["right"].set_visible(True)
    ax.set_xlabel("epoch"); ax.set_title(f"LeNet-5：5 个 epoch 到 {hist[-1][1] * 100:.2f}%"); ax.legend(loc="center right", fontsize=7); ax2.legend(loc="lower right", fontsize=7)
    ax = axes[1]; ax.axis("off"); ax.set_title(f"错分的 {len(err)} 张里的前 24 张（真实→预测）", fontsize=8.5)
    for i, j in enumerate(err[:24]):
        r, c = divmod(i, 12)
        ax.imshow(Xt[j, 0].numpy(), cmap="gray_r", extent=(c, c + 0.9, -r * 1.4 - 0.9, -r * 1.4))
        ax.text(c + 0.45, -r * 1.4 - 1.0, f"{yt[j]}→{pred[j]}", ha="center", va="top", fontsize=6, color=C["red"])
    ax.set_xlim(0, 12); ax.set_ylim(-2.9, 0); ax.set_aspect("equal")
    save(fig, "case-05-lenet-training")


if __name__ == "__main__":
    main()
