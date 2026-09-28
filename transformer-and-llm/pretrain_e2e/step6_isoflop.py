"""第 6 步：选模型大小——固定算力预算 C = 6·N·D，扫 8 个模型尺寸（N 大则 D 小），看哪个 loss 最低。
这是 Chinchilla 「iso-FLOP」方法的迷你版。两个预算各扫一遍，看最优 N 随预算怎么移。
输出 out/e2e-6-isoflop.svg 与 data/isoflop.json。

    python step6_isoflop.py            # MPS 约 20 分钟
    python step6_isoflop.py --quick    # 预算缩小 4 倍
    python step6_isoflop.py --plot-only  # 只从 data/isoflop.json 重画图
"""
import json
import sys

import numpy as np

from _plot import C, plt, save
from common import DATA, fmt
from trainer import Data, eval_loss, make_model, n_params, train

SIZES = [(1, 32), (1, 48), (1, 64), (2, 96), (2, 128), (3, 160), (4, 192), (6, 256)]   # (层数, 宽度)，N（含 embedding）从 0.14M 到 5.8M
BUDGETS = [1e13, 3e13, 1e14]
BS = 64                                                                  # 所有尺寸同一个 batch，只让 N 与 D 变
QUICK = "--quick" in sys.argv

if __name__ == "__main__":
    import contextlib, io
    if "--plot-only" in sys.argv:
        results = json.load(open(f"{DATA}/isoflop.json"))
    else:
        data = Data()
        print(f"训练集 {fmt(data.n_train_tokens)} 个 token；预算 C = 6·N·D，N 取全部参数（含 embedding）；batch 64 × 256 = 16K token/步")
        results = []
        for Cb in BUDGETS:
            Cb = Cb / 4 if QUICK else Cb
            print(f"\n=== 预算 C = {Cb:.0e} FLOPs ===")
            for L, E in SIZES:
                with contextlib.redirect_stdout(io.StringIO()):
                    m = make_model(L, E)
                N = n_params(m, non_embedding=False)      # 小模型的 embedding / lm_head 占大头，算 N 时要计入（Chinchilla 也计入）
                D = int(Cb / (6 * N))
                bs = BS
                steps = max(10, D // (bs * 256))
                epochs = D / data.n_train_tokens
                data.pos = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    hist, _ = train(m, data, steps, bs, peak_lr=1e-3 * (256 / E) ** 0.5, warmup=max(5, steps // 20), log_every=10**9, verbose=False, eval_batches=16)
                vl = eval_loss(m, data, "val", 32)
                results.append({"C": Cb, "L": L, "E": E, "N": N, "D": D, "steps": steps, "val": vl})
                print(f"  {L} 层 × {E} 宽  N = {N/1e6:5.2f}M  D = {D/1e6:6.2f}M token（{epochs:.1f} epoch，{steps} 步 × bs {bs}）  val loss {vl:.3f}")
            best = min((r for r in results if r["C"] == Cb), key=lambda r: r["val"])
            print(f"  → 这个预算下最优：{best['L']} 层 × {best['E']} 宽（N = {best['N']/1e6:.2f}M，D/N = {best['D']/best['N']:.0f}）")
        json.dump(results, open(f"{DATA}/isoflop.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for Cb, col in zip(sorted(set(r["C"] for r in results)), (C["blue"], C["red"], C["purple"])):
        rs = [r for r in results if r["C"] == Cb]
        ax.plot([r["N"] for r in rs], [r["val"] for r in rs], "o-", color=col, label=f"C = {Cb:.0e} FLOPs")
        b = min(rs, key=lambda r: r["val"])
        ax.plot(b["N"], b["val"], "*", color=col, ms=14)
        for r in rs:
            ax.annotate(f"{r['D']/1e6:.1f}M tok", (r["N"], r["val"]), (0, 6), textcoords="offset points", fontsize=6.5, ha="center", color=col)
    Ns = sorted(set(r["N"] for r in results))
    ax.set(xscale="log", xlabel="模型大小 N（全部参数，对数刻度）", ylabel="val loss", title="同样的算力：模型太小学不动，太大没喂饱——中间有个最优点（★）")
    ax.set_xticks(Ns, [f"{n/1e6:.2f}M" if n < 1e6 else f"{n/1e6:.1f}M" for n in Ns]); ax.minorticks_off()
    ax.legend(fontsize=8)
    save(fig, "e2e-6-isoflop")
