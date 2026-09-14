"""科学计算栈（工具箱 01）：NumPy 手写 attention 并与 PyTorch 对数值；广播的陷阱；Pandas 错误分析；Matplotlib 多 seed 曲线。
https://arganzheng.life/numpy-pandas-matplotlib-for-algorithm-engineers.html

    python 01_numpy_pandas_matplotlib.py            # 全部：attention broadcast pandas plot
    python 01_numpy_pandas_matplotlib.py attention  # 只跑一个
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("out")


# ---------------- 1. NumPy 手写单头 causal attention ----------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # 减最大值：L0 第五篇的数值技巧
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_attention_np(Q, K, V):
    """Q, K, V: [B, T, d]。返回 [B, T, d]。"""
    B, T, d = Q.shape
    S = np.einsum("btd,bsd->bts", Q, K) / np.sqrt(d)  # [B, T, T]：每个 query 对每个 key 的内积
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # 上三角为 True：未来位置
    S = np.where(mask, -np.inf, S)                     # 广播：[T, T] 作用到 [B, T, T] 的每个 batch
    P = softmax(S, axis=-1)
    return np.einsum("bts,bsd->btd", P, V)


def exp_attention():
    import torch
    import torch.nn.functional as F

    print("=== 1. NumPy 手写 causal attention vs torch.scaled_dot_product_attention ===")
    rng = np.random.default_rng(0)
    B, T, d = 2, 8, 16
    Q, K, V = (rng.standard_normal((B, T, d)).astype(np.float32) for _ in range(3))
    out_np = causal_attention_np(Q, K, V)
    out_pt = F.scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V), is_causal=True).numpy()
    print(f"形状: Q {Q.shape} -> S {(B, T, T)} -> 输出 {out_np.shape}")
    print(f"最大绝对误差: {np.abs(out_np - out_pt).max():.2e}   (float32 下 1e-6 量级即为一致)")
    P = softmax(np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf,
                         np.einsum("btd,bsd->bts", Q, K) / np.sqrt(d)))
    print("第 0 个 batch 的 attention 权重（每行和为 1，上三角为 0）：")
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(P[0])
    print()


# ---------------- 2. 广播：合法、不合法、以及"能跑但错" ----------------
def exp_broadcast():
    print("=== 2. 广播的三条规则 ===")
    X = np.zeros((4, 3))
    for name, other in [("(3,)  bias 加到每一行", np.ones(3)),
                        ("(4,1) 每行一个标量", np.ones((4, 1))),
                        ("(4,)  行数对不上最后一维", np.ones(4))]:
        try:
            print(f"  (4,3) + {name:<28} -> {(X + other).shape}")
        except ValueError as e:
            print(f"  (4,3) + {name:<28} -> ValueError: {e}")
    a = np.arange(3)          # (3,)
    b = np.arange(3)[:, None]  # (3,1)
    print(f"  陷阱：(3,) + (3,1) 不报错，得到 {(a + b).shape} ——想做逐元素相加却得到外积形状的表")
    print(a + b)
    print()


# ---------------- 3. Pandas：评测结果的错误分析 ----------------
def make_eval(seed, acc_by_cat):
    """造一份评测结果：每题一行，字段 id / category / correct。"""
    rng = np.random.default_rng(seed)
    rows = []
    for cat, (n, acc) in acc_by_cat.items():
        for i in range(n):
            rows.append({"id": f"{cat}-{i}", "category": cat, "correct": bool(rng.random() < acc)})
    return rows


def exp_pandas():
    import pandas as pd

    print("=== 3. Pandas：按类别聚合、找退化的题 ===")
    cats_base = {"algebra": (200, 0.70), "geometry": (150, 0.55), "number_theory": (100, 0.60), "combinatorics": (80, 0.45)}
    base = pd.DataFrame(make_eval(1, cats_base))
    # 新模型在同一套题上：原来对的题 92% 仍对；原来错的题按类别有不同的改善率（number_theory 反而退化）
    improve = {"algebra": 0.35, "geometry": 0.30, "number_theory": 0.05, "combinatorics": 0.30}
    rng = np.random.default_rng(2)
    new = base.copy()
    keep = rng.random(len(base)) < 0.92
    gain = rng.random(len(base)) < base["category"].map(improve).to_numpy()
    new["correct"] = np.where(base["correct"], keep, gain)
    summary = new.groupby("category")["correct"].agg(["mean", "count"]).rename(columns={"mean": "acc_new"})
    summary["acc_base"] = base.groupby("category")["correct"].mean()
    summary["delta"] = summary["acc_new"] - summary["acc_base"]
    # 每类的 95% 区间半宽（L0 第八篇）：1.96 * sqrt(p(1-p)/n)
    summary["ci95"] = 1.96 * np.sqrt(summary["acc_new"] * (1 - summary["acc_new"]) / summary["count"])
    print(summary.round(3).to_string())
    print(f"总体: base {base['correct'].mean():.3f} -> new {new['correct'].mean():.3f}")
    merged = new.merge(base, on=["id", "category"], suffixes=("", "_base"))
    regress = merged.query("not correct and correct_base")
    improve = merged.query("correct and not correct_base")
    print(f"退化的题（base 对、new 错）: {len(regress)}；改善的题: {len(improve)}")
    print("退化最多的类别:", regress["category"].value_counts().to_dict())
    print()


# ---------------- 4. Matplotlib：多 seed 的 loss 曲线 ----------------
def exp_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== 4. Matplotlib：多 seed 画均值与阴影带，对数 x 轴看早期 ===")
    steps = np.arange(1, 1001)
    curves = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(steps)).cumsum() * 0.002
        curves.append(4.8 * np.exp(-steps / 150) + 1.9 + 0.3 / np.sqrt(steps) + noise)
    curves = np.array(curves)                       # [5, 1000]
    mean, std = curves.mean(0), curves.std(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, xscale in zip(axes, ["linear", "log"]):
        ax.plot(steps, mean, label="mean of 5 seeds")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.3, label="±1 std")
        ax.plot(steps, curves[0], lw=0.6, alpha=0.6, label="seed 0 alone")
        ax.set_xscale(xscale)
        ax.set_xlabel("step" + (" (log)" if xscale == "log" else ""))
        ax.set_ylabel("loss")
        ax.legend()
    OUT.mkdir(exist_ok=True)
    path = OUT / "loss_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"曲线形状 {curves.shape}；step 10 / 100 / 1000 的均值 loss: "
          f"{mean[9]:.2f} / {mean[99]:.2f} / {mean[999]:.2f}，seed 间标准差 {std[999]:.3f}")
    print(f"已保存 {path}")
    print()


EXPS = {"attention": exp_attention, "broadcast": exp_broadcast, "pandas": exp_pandas, "plot": exp_plot}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
