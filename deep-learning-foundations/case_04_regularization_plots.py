"""案例（DL 04）：1,000 张 MNIST 上的过拟合与三种解药；宽度扫描画出 double descent——把 04 篇的两个实验画成图。
https://arganzheng.life/regularization-and-generalization.html

    python case_04_regularization_plots.py          # 全部（double descent 约 10 分钟）
    python case_04_regularization_plots.py reg      # 只跑正则化对比（1 分钟）

输出 out/case-04-*.svg。
"""
import sys

import numpy as np

from _plot import C, plt, save
from dlf import data, nn
from dlf.layers import Dropout, Sequential
from dlf.optim import Adam


def ce_eval(net, X, y, bs=4096):
    net.set_train(False); tot = 0.0; correct = 0
    for i in range(0, len(X), bs):
        logits = net.forward(X[i:i + bs]); loss, _, _ = nn.softmax_ce(logits, y[i:i + bs])
        tot += loss * len(logits); correct += (logits.argmax(1) == y[i:i + bs]).sum()
    net.set_train(True); return tot / len(X), correct / len(X)


def exp_reg(X, y, Xt, yt):
    rng = np.random.default_rng(0); n = 1000; idx = rng.permutation(len(X))[:n]; Xs, ys = X[idx], y[idx]
    print(f"=== 1. 只用 {n} 张训练图，784-512-10，AdamW 1e-3，300 个 epoch：无正则 / weight decay / dropout / 两者 ===")
    cfgs = {"无正则": (0.0, 0.0), "weight decay 0.5": (0.5, 0.0), "dropout 0.5": (0.0, 0.5), "两者都加": (0.5, 0.5)}
    curves = {}
    for name, (wd, p) in cfgs.items():
        rng2 = np.random.default_rng(1)
        net = Sequential([nn.Linear(784, 512, rng2), nn.ReLU(), Dropout(p, seed=2), nn.Linear(512, 10, rng2, std=np.sqrt(1 / 512))])
        opt = Adam(net.params(), 1e-3, wd=wd); tr_l, te_l, te_a, eps = [], [], [], []
        for ep in range(300):
            perm = rng2.permutation(n)
            for i in range(0, n, 100):
                b = perm[i:i + 100]; loss, d, _ = nn.softmax_ce(net.forward(Xs[b]), ys[b]); net.backward(d); opt.step()
            if (ep + 1) % 5 == 0:
                trl, _ = ce_eval(net, Xs, ys); tel, tea = ce_eval(net, Xt, yt)
                tr_l.append(trl); te_l.append(tel); te_a.append(tea); eps.append(ep + 1)
        curves[name] = (eps, tr_l, te_l, te_a)
        b = int(np.argmin(te_l))
        print(f"  {name:<18} 训练 loss @300 {tr_l[-1]:.3f}  测试 loss @10 {te_l[1]:.3f} → 最低 {te_l[b]:.3f}（第 {eps[b]} 个 epoch）→ @300 {te_l[-1]:.3f}  测试准确率 @300 {te_a[-1] * 100:.1f}%")
    print("  解读：四条训练 loss 都到 0——1,000 张、40 万参数的网络把训练集背下来毫无困难；差别在测试 loss：无正则的从第 10 个 epoch 起一路上升（0.40 → 0.55），"
          "这就是过拟合的形状；weight decay 让它几乎不再上升；dropout 单独用反而更差（这个网络只有一层隐藏层，drop 掉一半直接伤容量）；"
          "早停在四条曲线上各挑最低点，都在第 10–40 个 epoch——训 300 个 epoch 是白训。测试准确率却几乎不动（88.7%–89.7%）：loss 上升是模型对错的样本越来越自信，不是错得更多。")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    cols = [C["red"], C["blue"], C["orange"], C["green"]]
    for (name, (eps, trl, tel, tea)), col in zip(curves.items(), cols):
        axes[0].plot(eps, tel, c=col, lw=1.3, label=name); axes[0].plot(eps, trl, c=col, lw=0.8, ls=":")
        axes[1].plot(eps, [a * 100 for a in tea], c=col, lw=1.2, label=name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss（实线测试，虚线训练）"); axes[0].set_ylim(0, 0.7); axes[0].legend(fontsize=7)
    axes[0].set_title("1,000 张训练图：训练 loss 归零，测试 loss 掉头")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("测试准确率 %"); axes[1].set_ylim(85, 92); axes[1].set_title("准确率几乎不动：错的样本越来越自信")
    save(fig, "case-04-regularizers")


def exp_dd(X, y, Xt, yt):
    rng = np.random.default_rng(0); n = 4000; idx = rng.permutation(len(X))[:n]; Xs, ys = X[idx], y[idx].copy()
    noisy = rng.random(n) < 0.2; ys[noisy] = rng.integers(0, 10, noisy.sum())
    widths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"\n=== 2. double descent：{n} 张、20% 标签打乱，隐藏层宽度 2 → 2048，Adam 1e-3 全批 6000 步 ===")
    print(f"  {'宽度':>6}{'参数':>9}{'参数/样本':>9}{'训练错误率':>9}{'测试错误率':>9}{'测试 loss':>9}")
    res = []
    for w in widths:
        rng2 = np.random.default_rng(1); net = Sequential([nn.Linear(784, w, rng2), nn.ReLU(), nn.Linear(w, 10, rng2, std=np.sqrt(1 / w))])
        opt = Adam(net.params(), 1e-3)
        for s in range(6000):
            loss, d, _ = nn.softmax_ce(net.forward(Xs), ys); net.backward(d); opt.step()
        trl, tra = ce_eval(net, Xs, ys); tel, tea = ce_eval(net, Xt, yt)
        P = sum(p.size for p, _ in net.params()); res.append((w, P, 1 - tra, 1 - tea, tel))
        print(f"  {w:>6}{P:>9,}{P / n:>9.1f}{100 * (1 - tra):>8.1f}%{100 * (1 - tea):>8.1f}%{tel:>9.3f}")
    print("  解读：宽度 8–16 处训练错误率刚到 0（参数 ≈ 样本数 × 1.6–3.2，插值阈值），测试 loss 最高（5.5）——模型刚好有能力背下 20% 的错标签、又没有余量；"
          "再加宽，测试误差反而一路下降到 14.5%。经典的 U 形只在阈值左边成立，右边是第二次下降——这就是 double descent。")

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ws = [r[0] for r in res]
    ax.plot(ws, [r[3] * 100 for r in res], "o-", c=C["red"], label="测试错误率 %")
    ax.plot(ws, [r[2] * 100 for r in res], "s--", c=C["blue"], label="训练错误率 %")
    ax2 = ax.twinx(); ax2.plot(ws, [r[4] for r in res], "^:", c=C["orange"], label="测试 loss"); ax2.set_ylabel("测试 loss", color=C["orange"]); ax2.spines["right"].set_visible(True)
    ax.set_xscale("log", base=2); ax.set_xticks(ws); ax.set_xticklabels(ws); ax.set_xlabel("隐藏层宽度")
    ax.set_ylabel("错误率 %"); ax.axvline(12, ls="--", c=C["gray"], lw=0.8); ax.text(13, 28, "插值阈值：\n训练错误率刚到 0\n参数 ≈ 样本数", fontsize=7.5, color=C["gray"])
    ax.legend(loc="upper right", fontsize=7); ax2.legend(loc="upper center", fontsize=7)
    ax.set_title("4,000 张、20% 错标签：测试误差先降、再升、再降")
    save(fig, "case-04-double-descent")


EXPS = {"reg": exp_reg, "dd": exp_dd}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    X, y, Xt, yt = data.load_mnist_standardized()
    for n in names:
        EXPS[n](X, y, Xt, yt)
