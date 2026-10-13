"""案例（DL 02）：损失函数——概念图 + 三个 MNIST 实验。
https://arganzheng.life/loss-functions-from-mse-to-dpo.html

    python case_02_loss_functions.py

输出 out/loss-*.svg；三个实验都在 CPU 上几分钟内跑完，seed 固定：
1. 概念图：回归三兄弟、分类为什么不选 MSE、focal、InfoNCE 相似度矩阵。
2. MSE vs CE 收敛对比（两层 MLP 784-256-10，1 epoch）。
3. focal vs CE 在类不均衡 MNIST 子集上的逐类召回。
4. one-hot vs label smoothing 的校准（ECE + 可靠性图）。
"""
import sys

import numpy as np

from _plot import C, plt, save
from dlf import data, nn
from dlf.optim import SGD

LW = 2.2


# ---------------------------------------------------------------- 概念图

def fig_regression():
    e = np.linspace(-3, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    ax.plot(e, 0.5 * e ** 2, lw=LW, c=C["blue"], label="MSE $\\frac{1}{2}e^2$")
    ax.plot(e, np.abs(e), lw=LW, c=C["orange"], label="MAE $|e|$")
    d = 1.0
    hub = np.where(np.abs(e) <= d, 0.5 * e ** 2, d * (np.abs(e) - 0.5 * d))
    ax.plot(e, hub, lw=LW, c=C["green"], label="Huber $\\delta=1$")
    ax.set_xlabel("误差 $e = \\hat{y} - y$"); ax.set_ylabel("loss")
    ax.set_title("损失"); ax.legend()
    ax = axes[1]
    ax.plot(e, e, lw=LW, c=C["blue"], label="MSE：$e$")
    ax.plot(e, np.sign(e), lw=LW, c=C["orange"], label="MAE：sign$(e)$")
    ax.plot(e, np.clip(e, -d, d), lw=LW, c=C["green"], label="Huber：clip$(e, \\pm 1)$")
    ax.set_xlabel("误差 $e$"); ax.set_ylabel("$\\partial L / \\partial e$")
    ax.set_title("梯度：离群点拿多大步长"); ax.legend()
    save(fig, "loss-regression")


def fig_classification_gradient():
    """y = 1 的样本：sigmoid 输出 p = σ(z)。MSE 的梯度过 σ' 会饱和，BCE 恰好消掉。"""
    z = np.linspace(-8, 8, 400)
    p = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    ax.plot(z, 0.5 * (p - 1) ** 2, lw=LW, c=C["orange"], label="MSE $\\frac{1}{2}(p-1)^2$")
    ax.plot(z, -np.log(p), lw=LW, c=C["blue"], label="CE $-\\log p$")
    ax.set_xlabel("logit $z$（真标签 $y=1$，越往左错得越狠）"); ax.set_ylabel("loss")
    ax.set_title("损失"); ax.legend()
    ax = axes[1]
    g_mse = np.abs(p * (1 - p) * (p - 1))                       # σ'(z)·(p−y)
    g_ce = np.abs(p - 1)                                       # p − y
    ax.plot(z, g_mse, lw=LW, c=C["orange"], label="MSE：$|\\sigma'(z)(p-1)|$")
    ax.plot(z, g_ce, lw=LW, c=C["blue"], label="CE：$|p-1|$")
    ax.set_xlabel("logit $z$"); ax.set_ylabel("$|\\partial L / \\partial z|$")
    ax.set_title("对 logit 的梯度：MSE 在饱和区 ≈ 0"); ax.legend()
    ax.annotate("$z=-6$ 时 MSE 梯度只有 CE 的\n$\\sigma'(-6)\\approx 0.0025$ 倍", xy=(-6, g_mse[np.argmin(np.abs(z + 6))]),
                xytext=(-2.5, 0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
    save(fig, "loss-classification-gradient")
    i = np.argmin(np.abs(z + 6))
    print(f"  分类梯度：z=-6 时 |MSE 梯度|={g_mse[i]:.5f}，|CE 梯度|={g_ce[i]:.5f}，"
          f"比值 {g_ce[i]/max(g_mse[i],1e-12):.0f} 倍")


def fig_focal():
    p = np.linspace(1e-3, 1, 400)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for g, c in [(0, C["gray"]), (1, C["blue"]), (2, C["orange"]), (5, C["green"])]:
        ax.plot(p, -((1 - p) ** g) * np.log(p), lw=LW, c=c, label=f"$\\gamma={g}$")
    ax.set_xlabel("$p_t$（正确类的预测概率）"); ax.set_ylabel("loss")
    ax.set_title("focal loss：$-(1-p_t)^\\gamma \\log p_t$"); ax.legend()
    save(fig, "loss-focal")
    for g in [1, 2, 5]:
        r = (1 - 0.9) ** g / (1 - 0.5) ** g
        print(f"  focal γ={g}：p_t=0.9 相对 p_t=0.5 的 loss 权重比 = {r:.2f}")


def fig_infonce():
    """8 个样本 2 类的相似度矩阵：InfoNCE 看的是每一行里正例那格的 logit。"""
    rng = np.random.default_rng(0)
    d = 16; n = 8
    mu = rng.standard_normal((2, d)); mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    z = mu[np.repeat([0, 1], n // 2)] + 0.4 * rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    sim = z @ z.T / 0.1                                       # τ = 0.1
    y = np.repeat([0, 1], n // 2)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(sim, cmap="Blues", vmin=-10, vmax=15)
    for i in range(n):
        for j in range(n):
            color = "#c94c4c" if (i != j and y[i] == y[j]) else "#444444"
            ax.text(j, i, f"{sim[i,j]:.0f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"$z_{j}$" for j in range(n)]); ax.set_yticklabels([f"$z_{i}$" for i in range(n)])
    ax.set_title("in-batch 相似度矩阵 $\\times\\,1/\\tau$（红=正例对）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "loss-infonce")
    # 第一行：z0 的 InfoNCE（正例 z1..z3，其余为负）
    i = 0; pos = sim[i][(y == y[i]) & (np.arange(n) != i)]
    lse = np.log(np.exp(sim[i] - sim[i].max()).sum()) + sim[i].max()
    print(f"  InfoNCE 样例：τ=0.1，z0 行 logsumexp={lse:.2f}，正例 logit 和={pos.sum():.2f}，"
          f"loss={(lse*3 - pos.sum())/3:.2f}（三个正例平均）")


# ---------------------------------------------------------------- 实验工具

def run_custom(net, X, y, steps, bs, rng, loss_fn, lr=0.1, lr_sched=None, log_every=None):
    """minibatch SGD，loss_fn(logits,y_batch) -> (loss, dlogits/mean)。返回 per-step 记录。"""
    losses, gns = [], []
    for s in range(steps):
        if lr_sched is not None: lr = lr_sched(s)
        idx = rng.integers(0, X.shape[0], bs)
        logits = net.forward(X[idx])
        loss, d = loss_fn(logits, y[idx])
        if not np.isfinite(loss):
            print("  NaN!"); return np.array(losses), np.array(gns)
        net.backward(d)
        gn = float(np.sqrt(sum((g * g).sum() for _, g in net.params())))
        for P, g in net.params(): P -= lr * g
        losses.append(loss); gns.append(gn)
        if log_every and (s + 1) % log_every == 0:
            print(f"    step {s+1:<4d} loss {np.mean(losses[-log_every:]):.4f}  |g| {np.mean(gns[-log_every:]):.2f}")
    return np.array(losses), np.array(gns)


def ce_loss(logits, y):
    loss, d, _ = nn.softmax_ce(logits, y)
    return loss, d


def mse_loss(logits, y):
    """one-hot MSE on sigmoid head（教学对比：分类不该这么写）。"""
    p = 1 / (1 + np.exp(-logits))
    t = np.zeros_like(p); t[np.arange(len(y)), y] = 1.0
    loss = 0.5 * ((p - t) ** 2).sum(1).mean()
    return loss, p * (1 - p) * (p - t) / len(y)


def focal_loss(logits, y, gamma=2.0):
    """FL(pt) = -(1-pt)^γ log pt；dFL/dz = dFL/dpt · pt(δ - p)"""
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    pt = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    loss = -((1 - pt) ** gamma) * np.log(pt)
    dfl = gamma * (1 - pt) ** (gamma - 1) * np.log(pt) - (1 - pt) ** gamma / pt
    d = dfl[:, None] * pt[:, None] * (-p)
    d[np.arange(len(y)), y] += dfl * pt
    return loss.mean(), d / len(y)


def smooth_loss(logits, y, eps=0.1):
    """目标 q：正确类 1-ε，其余类各 ε/(C-1)；loss = -Σ q_k log p_k，梯度 = p - q"""
    z = logits - logits.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    C_ = logits.shape[1]
    q = np.full_like(lp, eps / (C_ - 1)); q[np.arange(len(y)), y] = 1 - eps
    loss = -(q * lp).sum(1).mean()
    p = np.exp(lp)
    return loss, (p - q) / len(y)


def ece(probs, y, bins=10):
    conf = probs.max(1); pred = probs.argmax(1); acc = pred == y
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf >= lo) & (conf < hi if b < bins - 1 else conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def probs_of(net, X):
    z = net.forward(X); z -= z.max(1, keepdims=True)
    p = np.exp(z); return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- 实验

def case_mse_vs_ce(X, y, Xt, yt):
    print("=== 2. MSE(sigmoid) vs CE(softmax)：784-256-10 MLP，SGD lr 0.1，1 epoch（469 步 × 128，seed 0）===")
    res = {}
    for name, fn in [("CE", ce_loss), ("MSE", mse_loss)]:
        rng = np.random.default_rng(0)
        net = nn.MLP([784, 256, 10], rng)
        l0, _, _ = (nn.softmax_ce(net.forward(Xt), yt) if name == "CE" else (mse_loss(net.forward(Xt), yt)[0], None, None))
        losses, gns = run_custom(net, X, y, 469, 128, rng, fn, lr=0.1)
        if name == "CE":
            acc = (probs_of(net, Xt).argmax(1) == yt).mean()
        else:
            p = 1 / (1 + np.exp(-net.forward(Xt))); acc = (p.argmax(1) == yt).mean()
        res[name] = (losses, gns, acc)
        print(f"  {name}: 初始 loss {l0:.3f} → 末 100 步均值 {losses[-100:].mean():.3f}；"
              f"前 50 步平均 |g| {gns[:50].mean():.2f}；test acc {acc*100:.2f}%")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    for name, c in [("CE", C["blue"]), ("MSE", C["orange"])]:
        l = res[name][0]
        ax.plot(np.convolve(l, np.ones(20) / 20, "valid"), lw=LW, c=c, label={"CE": "softmax + 交叉熵", "MSE": "sigmoid + MSE"}[name])
    ax.set_xlabel("step"); ax.set_ylabel("train loss（20 步滑动平均）"); ax.legend()
    ax.set_title("收敛速度")
    ax = axes[1]
    for name, c in [("CE", C["blue"]), ("MSE", C["orange"])]:
        g = res[name][1]
        ax.plot(np.convolve(g, np.ones(20) / 20, "valid"), lw=LW, c=c, label={"CE": "softmax + 交叉熵", "MSE": "sigmoid + MSE"}[name])
    ax.set_xlabel("step"); ax.set_ylabel("梯度范数 $|g|$"); ax.legend()
    ax.set_title("梯度尺度：MSE 一开始就小一个量级")
    save(fig, "loss-mse-vs-ce-training")


def case_focal_imbalance(X, y, Xt, yt):
    print("=== 3. 不均衡子集：类 9 只留 2%（约 120 张），CE vs focal γ=2，1 epoch，seed 1 ===")
    rng = np.random.default_rng(1)
    keep = np.ones(len(y), bool)
    idx9 = np.where(y == 9)[0]
    drop = rng.choice(idx9, int(len(idx9) * 0.98), replace=False)
    keep[drop] = False
    Xb, yb = X[keep], y[keep]
    print(f"  训练集 {len(y)} → {len(yb)}，类 9 只剩 {int((y==9).sum()*0.02)} 张（{yb.sum()==yb.sum() and (yb==9).mean()*100:.1f}%）")
    res = {}
    for name, fn in [("CE", ce_loss), ("focal γ=2", lambda l_, y_: focal_loss(l_, y_, 2.0))]:
        rng = np.random.default_rng(1)
        net = nn.MLP([784, 256, 10], rng)
        losses, _ = run_custom(net, Xb, yb, 469, 128, rng, fn, lr=0.1)
        pred = probs_of(net, Xt).argmax(1)
        rec = [(pred[yt == k] == k).mean() for k in range(10)]
        res[name] = rec
        print(f"  {name}: 末 100 步 loss {losses[-100:].mean():.3f}；总体 acc {(pred==yt).mean()*100:.2f}%；"
              f"类 9 召回 {rec[9]*100:.1f}%；逐类召回 " + " ".join(f"{r*100:.0f}" for r in rec))
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(10); w = 0.38
    ax.bar(x - w / 2, np.array(res["CE"]) * 100, w, color=C["blue"], label="CE")
    ax.bar(x + w / 2, np.array(res["focal γ=2"]) * 100, w, color=C["orange"], label="focal $\\gamma=2$")
    ax.set_xticks(x); ax.set_xlabel("类别（9 = 只占 2% 的少数类）"); ax.set_ylabel("召回 %")
    ax.set_ylim(0, 101); ax.legend(); ax.set_title("逐类召回率（类 9 是少数类）")
    save(fig, "loss-focal-imbalance")


def case_calibration(X, y, Xt, yt):
    print("=== 4. 校准：one-hot vs label smoothing ε=0.1，1 epoch，seed 2 ===")
    res = {}
    for name, fn in [("one-hot", ce_loss), ("平滑 ε=0.1", lambda l_, y_: smooth_loss(l_, y_, 0.1))]:
        rng = np.random.default_rng(2)
        net = nn.MLP([784, 256, 10], rng)
        run_custom(net, X, y, 469, 128, rng, fn, lr=0.1)
        p = probs_of(net, Xt)
        res[name] = p
        print(f"  {name}: test acc {(p.argmax(1)==yt).mean()*100:.2f}%；ECE(10 bin) {ece(p, yt)*100:.2f}%；"
              f"平均置信度 {p.max(1).mean()*100:.1f}%")
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    bins = np.linspace(0, 1, 11)
    for name, c in [("one-hot", C["blue"]), ("平滑 ε=0.1", C["orange"])]:
        conf = res[name].max(1); acc = (res[name].argmax(1) == yt)
        mids, accs = [], []
        for b in range(10):
            m = (conf >= bins[b]) & (conf < bins[b + 1] if b < 9 else conf <= bins[b + 1])
            if m.any(): mids.append(conf[m].mean()); accs.append(acc[m].mean())
        ax.plot(mids, np.array(accs), "o-", lw=LW, ms=4, c=c, label=name)
    ax.plot([0, 1], [0, 1], lw=1.0, c="#bbbbbb")
    ax.set_xlabel("置信度（bin 均值）"); ax.set_ylabel("准确率"); ax.legend()
    ax.set_title("可靠性图：越贴对角线越准")
    save(fig, "loss-calibration")


def main():
    print("=== 1. 概念图 ===")
    fig_regression(); fig_classification_gradient(); fig_focal(); fig_infonce()
    X, y, Xt, yt = data.load_mnist_standardized()
    case_mse_vs_ce(X, y, Xt, yt)
    case_focal_imbalance(X, y, Xt, yt)
    case_calibration(X, y, Xt, yt)


if __name__ == "__main__":
    main()
