"""案例（DL 03）：同一个 MNIST MLP，四种优化器 × 学习率扫描，再加 warmup——把曲线画出来。
https://arganzheng.life/optimizers-from-sgd-to-adamw.html

    python case_03_optimizers_mnist.py

复用 03_optimizers.py 的 dlf.optim；输出 out/case-03-*.svg。
"""
import numpy as np

from _plot import C, plt, save
from dlf import data, nn
from dlf.layers import make_deep_mlp
from dlf.optim import SGD, Adam, accuracy, lr_at, run


def smooth(x, k=20):
    x = np.asarray(x); return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    X, y = data.load_mnist("train"); Xt, yt = data.load_mnist("test")
    steps = 469                                                   # 1 个 epoch，batch 128
    grid = {"SGD": [0.03, 0.1, 0.3, 1.0, 3.0], "Momentum 0.9": [0.003, 0.01, 0.03, 0.1, 0.3],
            "Adam": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "AdamW (wd 0.1)": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]}
    colors = {"SGD": C["gray"], "Momentum 0.9": C["blue"], "Adam": C["green"], "AdamW (wd 0.1)": C["orange"]}

    def make(name, lr, net):
        return {"SGD": lambda: SGD(net.params(), lr), "Momentum 0.9": lambda: SGD(net.params(), lr, 0.9),
                "Adam": lambda: Adam(net.params(), lr), "AdamW (wd 0.1)": lambda: Adam(net.params(), lr, wd=0.1)}[name]()

    print("=== 1. 四种优化器 × 五个学习率，两层 MLP 784-256-10，1 个 epoch（469 步 × 128）===")
    acc = {}; curves = {}
    for name, lrs in grid.items():
        for lr in lrs:
            rng = np.random.default_rng(0); net = nn.MLP([784, 256, 10], rng)
            opt = make(name, lr, net)
            l, _, st = run(net, opt, X, y, steps, 128, rng)
            a = accuracy(net, Xt, yt) if st == "ok" else float("nan")
            acc[(name, lr)] = a; curves[(name, lr)] = l
        best_lr = max(lrs, key=lambda r: np.nan_to_num(acc[(name, r)], nan=-1))
        print(f"  {name:<16}" + "  ".join(f"lr {r:<6g} {acc[(name, r)] * 100:5.1f}%" if np.isfinite(acc[(name, r)]) else f"lr {r:<6g}  NaN " for r in lrs)
              + f"   → 最优 {best_lr}，状态 {opt.state_bytes() / 1024:.0f} KiB")
    best = {n: max(acc[(n, r)] for r in lrs if np.isfinite(acc[(n, r)])) for n, lrs in grid.items()}
    print(f"  解读：四种优化器调好学习率后成绩差不到 1 个点（{min(best.values()) * 100:.1f}%–{max(best.values()) * 100:.1f}%）——一个 epoch 的两层 MLP 上谁都能训。"
          "差别在两处：（一）学习率的量级——SGD 最优 1.0、Adam 最优 0.003，差 300 倍，因为 Adam 把每个参数的更新量归一到 ≈ lr，"
          "SGD 的更新量 = lr × 梯度，而这个网络的梯度很小；（二）代价——Momentum 多存一份参数（795 KiB），Adam 多存两份（1590 KiB），每参数 8 字节，"
          "这就是 Llama-3-8B 的优化器状态 64 GB 的来源。Adam 的优势不在最终成绩，在于最优学习率几乎不随模型变（都在 1e-3 量级），SGD 的最优学习率每个模型都要重找。")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    for name, lrs in grid.items():
        vals = [max(acc[(name, r)] * 100, 85.5) if np.isfinite(acc[(name, r)]) else 85.5 for r in lrs]
        ax.plot(lrs, vals, "o-", ms=4, c=colors[name], label=name)
        for r, v in zip(lrs, vals):
            if v == 85.5:
                ax.plot(r, 85.5, "x", c=colors[name], ms=9, mew=2)
                ax.text(r, 86.2, f"{acc[(name, r)] * 100:.0f}%" if np.isfinite(acc[(name, r)]) else "NaN", ha="center", fontsize=7, color=colors[name])
    ax.set_xscale("log"); ax.set_xlabel("学习率"); ax.set_ylabel("1 个 epoch 后的测试准确率 %"); ax.set_ylim(85, 98)
    ax.set_title("学习率扫描：× = 太大，训坏了"); ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    for name, lrs in grid.items():
        best_lr = max(lrs, key=lambda r: np.nan_to_num(acc[(name, r)], nan=-1))
        ax.plot(smooth(curves[(name, best_lr)]), c=colors[name], lw=1.2, label=f"{name}，lr {best_lr:g}")
    ax.set_xlabel("步（20 步滑动平均）"); ax.set_ylabel("训练 loss"); ax.set_ylim(0, 1.0); ax.set_title("各自最优学习率下的 loss 曲线"); ax.legend(fontsize=7)
    save(fig, "case-03-optimizers")

    # 2. warmup 在深网络上
    print("\n=== 2. warmup：64 层 Pre-Norm 网络，Adam β₂=0.95，峰值 lr 1e-3 / 3e-3 / 1e-2，400 步 ===")
    Xs, ys, Xts, yts = data.load_mnist_standardized()
    wu_curves = {}
    for lr in (1e-3, 3e-3, 1e-2):
        for wu in (0, 100):
            rng = np.random.default_rng(0); net = make_deep_mlp("prenorm", rng=rng)
            opt = Adam(net.params(), lr, b2=0.95)
            l, g, st = run(net, opt, Xs, ys, 400, 128, rng, warmup=wu)
            wu_curves[(lr, wu)] = (l, g)
            a = accuracy(net, Xts[:2000], yts[:2000]) if st == "ok" else float("nan")
            print(f"  峰值 lr {lr:<6g} warmup {wu:3d} 步：第 5–100 步最大 loss {max(l[5:100]):.2f}  @20 {l[19]:.3f}  @400 {l[-1]:.3f}  前 50 步最大梯度范数 {max(g[:50]):6.1f}  acc {a * 100:.1f}%")
    print("  解读：lr 1e-3 时有无 warmup 差别不大；lr 1e-2 不加 warmup 前 100 步 loss 冲到 4 以上——比随机初始化的 ln 10 = 2.30 还差，"
          "网络被打坏再慢慢爬回来；加 100 步 warmup 最高只到 1.3。Adam 第一步的更新量就是满的 lr（偏差修正后 = lr·sign(g)），此时二阶矩还没学到尺度，"
          "所有参数同时以最大步长乱跳——学习率越大伤得越重，而 LLM 训练总是想用尽量大的学习率，所以 warmup 几乎不能省。")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for (lr, wu), (l, g) in wu_curves.items():
        if lr == 3e-3:
            continue
        col = C["red"] if wu == 0 else C["green"]; ls = "-" if lr == 1e-2 else ":"
        axes[0].plot(smooth(l, 5), c=col, ls=ls, lw=1.1, label=f"峰值 lr {lr:g}，warmup {wu}")
        axes[1].plot(l[:100], c=col, ls=ls, lw=1.1)
    axes[0].axhline(np.log(10), c=C["gray"], lw=0.8, ls="--"); axes[0].text(150, np.log(10) + 0.1, "ln 10（随机初始化）", fontsize=7, color=C["gray"])
    axes[0].set_xlabel("步"); axes[0].set_ylabel("训练 loss（5 步滑动平均）"); axes[0].set_ylim(0, 4.8); axes[0].set_title("64 层 Pre-Norm 网络：有无 warmup"); axes[0].legend(fontsize=7)
    axes[1].axhline(np.log(10), c=C["gray"], lw=0.8, ls="--")
    axes[1].set_xlabel("步"); axes[1].set_ylabel("训练 loss（未平滑）"); axes[1].set_ylim(0, 5); axes[1].set_title("放大前 100 步：不加 warmup 冲到 4.59")
    save(fig, "case-03-warmup")

    # 3. 调度曲线本身
    fig, ax = plt.subplots(figsize=(7.6, 2.2))
    T = 1000; s = np.arange(T)
    ax.plot(s, [lr_at(i, 1.0, 100, T) for i in s], c=C["blue"], label="线性 warmup 100 步 + cosine 衰减到 10%")
    ax.plot(s, [1.0 if i < 800 else 1 - 0.9 * (i - 800) / 200 for i in s], c=C["orange"], label="WSD：warmup–稳定–最后 20% 线性衰减")
    ax.set_xlabel("步"); ax.set_ylabel("学习率 / 峰值"); ax.legend(fontsize=7); ax.set_title("两种常见的学习率调度")
    save(fig, "case-03-schedules")


if __name__ == "__main__":
    main()
