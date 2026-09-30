"""从 out/knobs.json 与 out/spectrum.json 画文章里的 SVG（字体不嵌入，浏览器用本地字体渲染）。
    python plot.py [输出目录]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = ["PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", "sans-serif"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DST = sys.argv[1] if len(sys.argv) > 1 else OUT

ORDER = ["full", "r4_all", "r16_all", "r64_all", "r16_attn", "r64_a32", "r64_rslora", "r16_lr1e-5", "r16_lr1e-3",
         "r16_loraplus", "r16_pissa", "r16_dora", "r16_qlora"]
SHORT = {"full": "全量 1e-5", "r4_all": "r=4", "r16_all": "r=16（默认）", "r64_all": "r=64 α=128", "r16_attn": "r=16 attn-only",
         "r64_a32": "r=64 α=32", "r64_rslora": "r=64 rsLoRA α=32", "r16_lr1e-5": "r=16 lr 1e-5", "r16_lr1e-3": "r=16 lr 1e-3",
         "r16_loraplus": "LoRA+ ×4", "r16_pissa": "PiSSA", "r16_dora": "DoRA", "r16_qlora": "QLoRA NF4"}


def knobs():
    R = json.load(open(os.path.join(OUT, "knobs.json")))
    names = [n for n in ORDER if n in R]
    base = R["_base"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), gridspec_kw={"wspace": 0.55})
    ys = range(len(names))[::-1]
    v = [R[n]["val_loss"] for n in names]
    f = [R[n]["text_loss"] - base["text_loss"] for n in names]
    colors = ["#c98a00" if n == "r16_all" else ("#4a6fa5" if n == "full" else "#8fa8c8") for n in names]
    axes[0].barh(list(ys), v, color=colors)
    axes[0].axvline(base["val_loss"], color="#999", ls="--", lw=1)
    axes[0].text(base["val_loss"], len(names) - 0.4, f"训练前 {base['val_loss']:.3f}", fontsize=8, color="#666", ha="right")
    axes[0].set_xlim(min(v) - 0.05, max(v) + 0.05)
    axes[0].set_yticks(list(ys)); axes[0].set_yticklabels([SHORT[n] for n in names], fontsize=8)
    axes[0].set_xlabel("验证回复 loss（越低越好）", fontsize=9)
    for y, x in zip(ys, v):
        axes[0].text(x, y, f" {x:.3f}", va="center", fontsize=7)
    axes[1].barh(list(ys), f, color=colors)
    axes[1].axvline(0, color="#999", lw=1)
    axes[1].set_yticks(list(ys)); axes[1].set_yticklabels([SHORT[n] for n in names], fontsize=8)
    axes[1].set_xlabel("普通文本 loss 变化（遗忘，越接近 0 越好）", fontsize=9)
    for y, x in zip(ys, f):
        axes[1].text(x, y, f" {x:+.3f}", va="center", fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Qwen2.5-0.5B · no_robots 800 条 · 80 步：全量与十二种 LoRA 配置", fontsize=10)
    fig.savefig(os.path.join(DST, "lora-hyperparameters-rank-targets-alpha-lr-and-variants-matrix.svg"), bbox_inches="tight")
    # lr 三条训练曲线
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for n, c in [("r16_lr1e-5", "#8fa8c8"), ("r16_all", "#c98a00"), ("r16_lr1e-3", "#b5443c"), ("full", "#4a6fa5")]:
        if n in R:
            xs, ys_ = zip(*R[n]["train_losses"])
            ax.plot(xs, ys_, marker="o", ms=3, color=c, label=SHORT[n])
    ax.set_xlabel("步", fontsize=9); ax.set_ylabel("训练 loss", fontsize=9); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    ax.set_title("学习率：LoRA r=16 全部线性层 的 1e-5 / 1e-4 / 1e-3 与全量 1e-5", fontsize=10)
    fig.savefig(os.path.join(DST, "lora-hyperparameters-rank-targets-alpha-lr-and-variants-lr.svg"), bbox_inches="tight")


def spectrum():
    p = os.path.join(OUT, "spectrum.json")
    if not os.path.exists(p):
        return
    S = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"wspace": 0.35})
    for k, e in S["cum_energy"].items():
        label = k.replace("model.layers.", "L").replace(".self_attn.", " ").replace(".mlp.", " ").replace(".weight", "")
        ls = "-" if "q_proj" in k else ("--" if "o_proj" in k else ":")
        axes[0].plot(range(1, len(e) + 1), e, ls=ls, lw=1.2, label=label)
    axes[0].set_xscale("log"); axes[0].axvline(16, color="#c98a00", lw=1); axes[0].text(16, 0.05, " r=16", fontsize=8, color="#c98a00")
    axes[0].set_xlabel("秩 r（对数）", fontsize=9); axes[0].set_ylabel("前 r 个奇异值占 ‖ΔW‖² 的比例", fontsize=9)
    axes[0].legend(fontsize=6.5, ncol=1); axes[0].set_title("全量微调 80 步的 ΔW：能量随秩的累积", fontsize=9)
    t = S["trunc"]; keys = [k for k in t["ranks"]]
    labels = ["训练前"] + [("全量 ΔW" if k == "None" else f"截到秩 {k}") for k in keys]
    vals = [t["base"][0]] + [t["ranks"][k][0] for k in keys]
    axes[1].bar(range(len(vals)), vals, color=["#999"] + ["#8fa8c8"] * (len(vals) - 2) + ["#4a6fa5"])
    axes[1].set_xticks(range(len(vals))); axes[1].set_xticklabels(labels, fontsize=8, rotation=20)
    axes[1].set_ylim(min(vals) - 0.03, max(vals) + 0.03); axes[1].set_ylabel("验证回复 loss", fontsize=9)
    for i, x in enumerate(vals):
        axes[1].text(i, x, f"{x:.4f}", ha="center", va="bottom", fontsize=7)
    axes[1].set_title("把 168 个 ΔW 各截到秩 r 装回模型", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(DST, "lora-low-rank-hypothesis-gradients-and-accounts-spectrum.svg"), bbox_inches="tight")


if __name__ == "__main__":
    knobs(); spectrum(); print("saved to", DST)
