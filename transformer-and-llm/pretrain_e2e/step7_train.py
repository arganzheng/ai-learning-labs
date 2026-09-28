"""第 7 步：正式训练——按第 6 步的结论定尺寸（数据只有 10.9M token，是硬约束：跑 3 epoch，模型比同算力的
iso-FLOP 最优点大一号），定一份配方（AdamW、峰值 lr、warmup、cosine、batch），一路记录七条曲线。
输出 data/ckpt.pt、data/train_hist.json、out/e2e-7-curves.svg。

    python step7_train.py                                   # 4 层 × 192 宽，3 epoch，MPS 约 10 分钟
    python step7_train.py --layers 2 --embd 128 --lr 1.4e-3  # 文中的对照：同样 3 epoch 的小一号模型
"""
import argparse
import json
import math

import torch

from _plot import C, plt, save
from common import DATA, fmt
from trainer import BLOCK, DEV, Data, make_model, n_params, train

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--embd", type=int, default=192)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1.15e-3)      # 与 step6 同一条经验律：1e-3 × (256 / 宽)^0.5
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--warmup", type=float, default=0.03)
args = ap.parse_args()

if __name__ == "__main__":
    data = Data()
    model = make_model(args.layers, args.embd)
    N = n_params(model)
    tokens_per_step = args.bs * BLOCK
    steps = int(args.epochs * data.n_train_tokens / tokens_per_step)
    warmup = int(steps * args.warmup)
    print(f"\n配方：{args.layers} 层 × {args.embd} 宽，N = {N/1e6:.2f}M（含 embedding {n_params(model, False)/1e6:.2f}M）")
    print(f"      数据 {fmt(data.n_train_tokens)} token × {args.epochs} epoch = {fmt(steps * tokens_per_step)} token；batch {args.bs} × {BLOCK} = {fmt(tokens_per_step)} token/步；{steps} 步")
    print(f"      AdamW β=(0.9, 0.95) wd 0.1 clip 1.0；峰值 lr {args.lr:.2e}，warmup {warmup} 步（{args.warmup:.0%}），cosine 衰减到 10%")
    print(f"      算力 6ND ≈ {6 * N * steps * tokens_per_step:.1e} FLOPs；设备 {DEV}\n")
    hist, _ = train(model, data, steps, args.bs, args.lr, warmup, log_every=max(10, steps // 40))
    torch.save({"model": model.state_dict(), "config": {"n_layer": args.layers, "n_embd": args.embd}, "hist": hist}, f"{DATA}/ckpt.pt")
    json.dump(hist, open(f"{DATA}/train_hist.json", "w"))
    print(f"\n最终 train loss {hist['train'][-1]:.3f}，val loss {hist['val'][-1]:.3f}（PPL {math.exp(hist['val'][-1]):.1f}）；ln V = {math.log(4096):.2f}")

    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.2))
    s = hist["step"]
    panels = [("train / val loss", [("train", C["blue"], hist["train"]), ("val", C["orange"], hist["val"])]),
              ("学习率", [("lr", C["green"], hist["lr"])]),
              ("梯度范数（裁剪前）", [("|g|", C["red"], hist["gnorm"])]),
              ("参数范数", [("|θ|", C["purple"], hist["pnorm"])]),
              ("attention logit 最大值", [("max |q·k|/√d", C["red"], hist["attn"])]),
              ("吞吐（K token/s）", [("tok/s", C["gray"], [t / 1e3 for t in hist["tps"]])]),
              ("train − val（过拟合信号）", [("gap", C["orange"], [a - b for a, b in zip(hist["train"], hist["val"])])])]
    for ax, (title, series) in zip(axes.ravel(), panels):
        for name, col, ys in series:
            ax.plot(s, ys, color=col, lw=1.2, label=name)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        if len(series) > 1:
            ax.legend(fontsize=6.5)
    axes[0, 0].axhline(math.log(4096), color=C["gray"], ls=":", lw=1); axes[0, 0].text(s[-1], math.log(4096) + 0.1, "ln V", fontsize=7, ha="right", color=C["gray"])
    axes[1, 3].axis("off")
    axes[1, 3].text(0, 0.5, f"{args.layers} 层 × {args.embd} 宽\nN = {N/1e6:.2f}M\n{fmt(steps)} 步 × {fmt(tokens_per_step)} token\n{args.epochs} epoch\nval loss {hist['val'][-1]:.3f}\nPPL {math.exp(hist['val'][-1]):.1f}", fontsize=8, va="center")
    for ax in axes[1]:
        ax.set_xlabel("step", fontsize=8)
    save(fig, "e2e-7-curves")
