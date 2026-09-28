"""案例（DL 02）：把 MLP 加深到 64 层——七种接法哪些能训，把逐层统计和训练曲线画出来。
https://arganzheng.life/initialization-normalization-and-residual.html

    python case_02_deep_mlp.py

复用 02_init_norm_residual.py 的七种配置（dlf.layers.DEEP_CONFIGS）；输出 out/case-02-*.svg。
"""
import numpy as np

from _plot import C, plt, save
from dlf import data, nn
from dlf.layers import DEEP_CONFIGS, make_deep_mlp

COLORS = {"plain-naive": C["gray"], "plain-kaiming": C["purple"], "ln": C["teal"], "res-nonorm": C["red"],
          "res-nonorm-scaled": C["orange"], "prenorm": C["green"], "postnorm": C["blue"]}
LABEL = {"plain-naive": "plain，σ=1/√n（忘了 ReLU 的 2）", "plain-kaiming": "plain，Kaiming", "ln": "Linear → LN → ReLU",
         "res-nonorm": "残差，无归一化", "res-nonorm-scaled": "残差，无归一化，out-proj / √2L", "prenorm": "Pre-Norm 残差", "postnorm": "Post-Norm 残差"}


def per_block_stats(net, X, y):
    """每个块出口的激活 std（前向）与每个 Linear 权重的梯度范数（反向）。"""
    acts = []; h = X
    for l in net.layers:
        h = l.forward(h)
        if not isinstance(l, nn.ReLU):
            acts.append(h.std())
    loss, d, _ = nn.softmax_ce(h, y); net.backward(d)
    gn = [np.linalg.norm(dP) for P, dP in net.params() if P.ndim == 2]
    return loss, np.array(acts), np.array(gn)


def main():
    np.seterr(all="ignore")
    X, y, Xt, yt = data.load_mnist_standardized()
    print("模型：784 → 256 的输入投影 + 64 个 256 宽的块 + 256 → 10 的输出层；七种接法只差初始化 / 归一化 / 残差怎么放")

    # 1. 初始时刻逐层统计
    print("\n=== 1. 初始时刻：逐块激活 std 与逐层梯度范数 ===")
    stats = {}
    for c in DEEP_CONFIGS:
        rng = np.random.default_rng(0); net = make_deep_mlp(c, rng=rng)
        loss, acts, gn = per_block_stats(net, X[:256], y[:256])
        stats[c] = (acts, gn)
        print(f"  {c:18s} 初始 loss {loss:8.3f}  激活 std 块 1 → 64：{acts[1]:.2e} → {acts[-2]:.2e}   梯度范数 底层 → 顶层：{gn[1]:.2e} → {gn[-2]:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for c in DEEP_CONFIGS:
        acts, gn = stats[c]
        axes[0].plot(np.linspace(0, 64, len(acts)), acts, c=COLORS[c], lw=1.2)
        g = gn[1:-1:2] if len(gn) > 70 else gn[1:-1]                      # 残差块有两个 Linear，只画每块的第一个
        axes[1].plot(np.linspace(1, 64, len(g)), g, c=COLORS[c], lw=1.2, label=LABEL[c])
    axes[0].set_yscale("log"); axes[0].set_xlabel("深度（块）"); axes[0].set_ylabel("激活 std（对数）"); axes[0].set_title("前向：每层激活多大")
    axes[0].axhline(1, ls=":", c=C["gray"], lw=0.8)
    axes[1].set_yscale("log"); axes[1].set_xlabel("深度（块）"); axes[1].set_ylabel("权重梯度范数（对数）"); axes[1].set_title("反向：每层梯度多大")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), frameon=False)
    save(fig, "case-02-init-stats")

    # 2. 训练 300 步
    print("\n=== 2. SGD 300 步，batch 128，lr 0.05 与 0.005 ===")
    curves = {}
    for lr in (0.05, 0.005):
        for c in DEEP_CONFIGS:
            rng = np.random.default_rng(0); net = make_deep_mlp(c, rng=rng); losses = []
            for s in range(300):
                idx = rng.integers(0, X.shape[0], 128)
                loss, d, _ = nn.softmax_ce(net.forward(X[idx]), y[idx]); net.backward(d)
                if not np.isfinite(loss):
                    break
                for P, dP in net.params():
                    P -= lr * dP
                losses.append(loss)
            acc = (net.forward(Xt[:2000]).argmax(1) == yt[:2000]).mean() if len(losses) == 300 else float("nan")
            curves[(lr, c)] = losses
            tag = f"NaN @ 第 {len(losses) + 1} 步" if len(losses) < 300 else f"loss {losses[-1]:.3f}  acc {acc * 100:5.1f}%"
            print(f"  lr {lr:<6} {c:18s} {tag}")
    print("  解读：七种接法里只有 Pre-Norm 在两个学习率下都正常训——这就是它成为 LLM 默认结构的原因。")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, lr in zip(axes, (0.05, 0.005)):
        for c in DEEP_CONFIGS:
            l = curves[(lr, c)]
            if len(l) < 300:
                ax.plot(range(len(l)), np.minimum(l, 3.05), c=COLORS[c], lw=1.2, label=LABEL[c])
                ax.plot(len(l), 3.05, "x", c=COLORS[c], ms=8, mew=2)
            else:
                ax.plot(l, c=COLORS[c], lw=1.2, label=LABEL[c])
        ax.axhline(np.log(10), ls=":", c=C["gray"], lw=0.8)
        ax.set_xlabel("步"); ax.set_title(f"lr = {lr}（× = 发散成 NaN）"); ax.set_ylim(0, 3.2)
    axes[0].set_ylabel("训练 loss")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), frameon=False)
    save(fig, "case-02-training-curves")


if __name__ == "__main__":
    main()
