"""算法工程师的数学（08）：多种子实验——同一个配方跑 5 次差多少？改一个超参的差异是不是噪声？
https://arganzheng.life/statistical-inference-and-fitting-scaling-laws.html

在莎士比亚字符数据上训一个 0.1M 参数的小 GPT（与 05 篇 lnv 段同一个模型），400 步：
  配方 A：lr 1e-3，weight decay 0.01；配方 B：weight decay 改成 0.1；配方 C：lr 改成 1.02e-3（+2%）；配方 D：lr 改成 1.2e-3（+20%）。各 5 个种子，比 val loss。

    python 08_multi_seed.py          # 20 次训练，CPU 约 5 分钟
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _plot import C, plt, save

HERE = os.path.dirname(os.path.abspath(__file__))
TLL = os.path.join(HERE, "..", "transformer-and-llm")
sys.path.insert(0, TLL)
from nanogpt_model import GPT, GPTConfig  # noqa: E402

DATA = os.path.join(TLL, "nanogpt", "data", "shakespeare_char")
train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")


def batch(data, bs, block, g):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=g)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x, y


def run(lr, seed, steps=400, wd=0.01):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = GPT(GPTConfig(block_size=64, vocab_size=65, n_layer=2, n_head=4, n_embd=64, dropout=0.0, bias=False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        x, y = batch(train, 32, 64, g)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    gv = torch.Generator().manual_seed(12345)                  # 所有 run 用同一批 val 数据
    with torch.no_grad():
        losses = [model(*batch(val, 64, 64, gv))[1].item() for _ in range(8)]
    return float(np.mean(losses))


if __name__ == "__main__":
    import contextlib, io
    seeds = range(5)
    res = {}
    cfgs = (("A: lr 1e-3, wd 0.01", 1e-3, 0.01), ("B: wd 0.1", 1e-3, 0.1), ("C: lr 1.02e-3", 1.02e-3, 0.01), ("D: lr 1.2e-3", 1.2e-3, 0.01))
    for name, lr, wd in cfgs:
        vals = []
        for sd in seeds:
            with contextlib.redirect_stdout(io.StringIO()):     # 吞掉 nanoGPT 的 "number of parameters"
                v = run(lr, sd, wd=wd)
            vals.append(v)
            print(f"  {name}  seed {sd}: val loss {v:.4f}")
        res[name] = np.array(vals)
        print(f"  {name}  均值 {res[name].mean():.4f} ± 标准差 {res[name].std(ddof=1):.4f}（5 个种子；标准误 {res[name].std(ddof=1)/math.sqrt(5):.4f}）\n")
    from scipy.stats import ttest_ind
    a = res["A: lr 1e-3, wd 0.01"]
    pvals = {}
    for name in ("B: wd 0.1", "C: lr 1.02e-3", "D: lr 1.2e-3"):
        b = res[name]
        diff = a.mean() - b.mean()
        se = math.sqrt(a.var(ddof=1) / 5 + b.var(ddof=1) / 5)
        t, p = ttest_ind(a, b, equal_var=False)
        pvals[name] = p
        single = np.array([[x - y for y in b] for x in a]).ravel()
        print(f"  A − {name[0]}：5 种子均值之差 {diff:+.4f}，差值的标准误 {se:.4f}，t = {diff/se:.2f}，Welch t 检验 p = {p:.3f} → {'显著' if p < 0.05 else '分不出来'}")
        print(f"     若各只跑 1 个种子再相减：25 种组合里差值从 {single.min():+.4f} 到 {single.max():+.4f}，{(np.sign(single) != np.sign(diff)).mean():.0%} 的组合符号是反的")
    print("  读法：一个改动是否「有用」，看 5 种子均值之差相对差值标准误有多大（t）；t 在 2 以下的差异用单种子对比会得到相反的结论；多少种子够？让差值的标准误降到差值的 1/2 以下。")
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    for i, (name, v) in enumerate(res.items()):
        ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, color=[C["blue"], C["orange"], C["green"], C["purple"]][i], zorder=3, label=f"{name}：{v.mean():.4f} ± {v.std(ddof=1):.4f}")
        ax.errorbar(i, v.mean(), yerr=1.96 * v.std(ddof=1) / math.sqrt(5), fmt="_", color="k", capsize=8, ms=20, lw=1.2)
    ax.set_xticks([0, 1, 2, 3], ["A：lr 1e-3，wd 0.01", f"B：wd 0.1\n vs A 的 p = {pvals['B: wd 0.1']:.3f}", f"C：lr 1.02e-3\n vs A 的 p = {pvals['C: lr 1.02e-3']:.2f}", f"D：lr 1.2e-3\n vs A 的 p = {pvals['D: lr 1.2e-3']:.3f}"])
    ax.set(ylabel="val loss（400 步）", title="同一配方 5 个种子的散布（点）与均值的 95% 区间（黑线）")
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, "08-multi-seed")
