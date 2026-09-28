"""预训练（05）「先讲明白」的三张图（复用 training_recipe_lab.py 的字符级小 Transformer，CPU）：

  out/pretrain-05-lr-gone-wrong.svg   学习率合适 / 太小 / 太大 / 太大且没有 warmup 四条 loss 曲线——"调坏了长什么样"
  out/pretrain-05-spike-anatomy.svg   大学习率下 loss、attention logit 最大值、梯度范数三条曲线同步看，有无 QK-norm 对照
  out/pretrain-05-zloss.svg           z-loss 把 |log Z| 按在 0 附近
  out/pretrain-05-real-spike.svg      第一篇 MacBook 实训里那次真实的小 spike（step 833）放大看：loss / 梯度范数 / attention logit

    python recipe_intuition.py            # 约 6 分钟
    python recipe_intuition.py --quick    # 步数减半
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_e2e"))
import _plot  # noqa: E402
from _plot import C, plt  # noqa: E402
from scaling_law_fit import load_corpus  # noqa: E402
from training_recipe_lab import GPT, train  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_plot.OUT = os.path.join(HERE, "out")
save = _plot.save
QUICK = "--quick" in sys.argv
S = 0.5 if QUICK else 1.0


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def fig_lr_gone_wrong(data, val, vocab):
    steps = int(600 * S)
    runs = [("合适：lr 2e-3，warmup 30 步", 2e-3, 30, C["green"]),
            ("太小：lr 2e-4（1/10）", 2e-4, 30, C["blue"]),
            ("太大：lr 1e-1（50 倍），有 warmup", 1e-1, 30, C["red"]),
            ("太大且没有 warmup：lr 1e-1，第 1 步就是峰值", 1e-1, 1, C["orange"])]
    print(f"=== 图 1：学习率调坏了长什么样（d=64，2 层，batch 32，cosine，{steps} 步）===")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for name, lr, warmup, col in runs:
        torch.manual_seed(0)
        m = GPT(vocab)
        log = train(m, data, val, steps, 32, lr, warmup=warmup, log_every=5, eval_every=steps)
        xs = [l["step"] for l in log]; ls = smooth([l["loss"] for l in log])
        axes[0].plot(xs, ls, color=col, lw=1.2, label=name)
        axes[1].plot(xs[:12], [l["loss"] for l in log][:12], "o-", color=col, lw=1.2, ms=3)
        print(f"  {name:<34} 第 5 步 loss {log[1]['loss']:.2f}   第 25 步 {log[4]['loss']:.2f}   末 train loss {ls[-1]:.3f}   val {log[-1]['val']:.3f}   最大梯度范数 {max(l['gnorm'] for l in log):.1f}")
    axes[0].set(xlabel="step", ylabel="train loss（5 点平滑）", title="四种学习率设置，同样的模型与数据")
    axes[0].legend(fontsize=7)
    axes[1].axhline(log[0]["loss"], color=C["gray"], ls=":", lw=1); axes[1].text(2, log[0]["loss"] + 0.15, "初始 loss（ln V）", fontsize=7, color=C["gray"])
    axes[1].set(xlabel="step", title="前 60 步放大：没有 warmup 的那条第 5 步冲到初始 loss 之上")
    save(fig, "pretrain-05-lr-gone-wrong")


def fig_spike_anatomy(data, val, vocab):
    steps = int(300 * S)
    print(f"\n=== 图 2：大学习率（常数 1e-1，不裁剪）下三条曲线同步看，有无 QK-norm（d=128，4 层，{steps} 步）===")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for qk, col, name in [(False, C["red"], "无 QK-norm"), (True, C["green"], "QK-norm")]:
        torch.manual_seed(0)
        m = GPT(vocab, d=128, n_layers=4, n_heads=4, qk_norm=qk)
        log = train(m, data, val, steps, 32, 1e-1, sched="constant", warmup=10, clip=0, log_every=1)
        xs = [l["step"] for l in log]
        axes[0].plot(xs, [l["loss"] for l in log], color=col, lw=1, label=name)
        axes[1].plot(xs, [l["max_logit"] for l in log], color=col, lw=1)
        axes[2].plot(xs, [l["gnorm"] for l in log], color=col, lw=1)
        losses = [l["loss"] for l in log]
        jumps = sum(1 for a, b in zip(losses, losses[1:]) if b > a + 0.3)
        print(f"  {name:<10} 末 loss {losses[-1]:.2f}  attention logit 最大 {max(l['max_logit'] for l in log):,.0f}  梯度范数最大 {max(l['gnorm'] for l in log):.1f}  loss 单步上跳 > 0.3 的次数 {jumps}")
    axes[0].set(title="train loss", xlabel="step"); axes[0].legend(fontsize=7)
    axes[1].set(title="attention logit 最大值（对数）", xlabel="step", yscale="log")
    axes[2].set(title="梯度范数（裁剪前，对数）", xlabel="step", yscale="log")
    save(fig, "pretrain-05-spike-anatomy")


def fig_zloss(data, val, vocab):
    steps = int(600 * S)
    print(f"\n=== 图 3：z-loss 与 |log Z|（d=64，lr 4e-3，{steps} 步）===")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    for name, z, col in [("无 z-loss", 0.0, C["red"]), ("z-loss 1e-4（PaLM 的系数）", 1e-4, C["blue"]), ("z-loss 1e-2（夸张）", 1e-2, C["green"])]:
        torch.manual_seed(0)
        m = GPT(vocab)
        log = train(m, data, val, steps, 32, 4e-3, z_loss=z, log_every=10, eval_every=steps)
        ax.plot([l["step"] for l in log], [l["log_z"] for l in log], color=col, lw=1.2, label=f"{name}：末 val loss {log[-1]['val']:.3f}")
        print(f"  {name:<26} 末 val loss {log[-1]['val']:.3f}   |log Z| 末值 {log[-1]['log_z']:.2f}")
    ax.set(xlabel="step", ylabel="|log Z|（softmax 归一化常数的对数）", title="z-loss 把输出 logits 的整体漂移按住，loss 几乎不变")
    ax.legend(fontsize=7.5)
    save(fig, "pretrain-05-zloss")


def fig_real_spike():
    path = os.path.join(HERE, "pretrain_e2e", "data", "train_hist.json")
    if not os.path.exists(path):
        print("\n（跳过图 4：没有 pretrain_e2e/data/train_hist.json，先跑第一篇的 step7_train.py）"); return
    h = json.load(open(path))
    print("\n=== 图 4：第一篇实训里 step 833 的那次小 spike ===")
    i = max(range(len(h["step"])), key=lambda k: h["gnorm"][k])
    print(f"  梯度范数最大在 step {h['step'][i]}：|g| = {h['gnorm'][i]:.2f}（平时 0.4–0.8），train loss {h['train'][i]:.3f}（前一次记录 {h['train'][i-1]:.3f}），val {h['val'][i]:.3f} → 下一次 {h['val'][i+1]:.3f}，attention logit {h['attn'][i-1]:.1f} → {h['attn'][i]:.1f} → {h['attn'][i+1]:.1f}")
    lo, hi = max(0, i - 8), min(len(h["step"]), i + 9)
    sl = slice(lo, hi)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    axes[0].plot(h["step"][sl], h["train"][sl], "o-", color=C["blue"], lw=1, ms=3, label="train")
    axes[0].plot(h["step"][sl], h["val"][sl], "s-", color=C["orange"], lw=1, ms=3, label="val")
    axes[1].plot(h["step"][sl], h["gnorm"][sl], "o-", color=C["red"], lw=1, ms=3); axes[1].axhline(1.0, color=C["gray"], ls=":", lw=1)
    axes[1].text(h["step"][lo], 1.1, "裁剪阈值 1.0", fontsize=7, color=C["gray"])
    axes[2].plot(h["step"][sl], h["attn"][sl], "o-", color=C["purple"], lw=1, ms=3)
    for ax in axes:
        ax.axvline(h["step"][i], color=C["red"], ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("step")
    axes[0].set(title="train / val loss"); axes[0].legend(fontsize=7)
    axes[1].set(title="梯度范数（裁剪前）")
    axes[2].set(title="attention logit 最大值")
    save(fig, "pretrain-05-real-spike")


if __name__ == "__main__":
    data, val, vocab = load_corpus()
    fig_lr_gone_wrong(data, val, vocab)
    fig_spike_anatomy(data, val, vocab)
    fig_zloss(data, val, vocab)
    fig_real_spike()
