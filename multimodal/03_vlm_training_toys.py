"""多模态（03）VLM 训练：两个 toy——为什么先冻结 LLM 只训 connector；共现偏差怎么变成幻觉。

    python 03_vlm_training_toys.py            # 全部：freeze cooccur
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _plot import C, plt, save

torch.manual_seed(0)
rng = np.random.default_rng(0)


def make_task(n, centers, noise=1.6):
    K, d = centers.shape
    y = torch.randint(0, K, (n,))
    return centers[y] + noise * torch.randn(n, d), y


def acc(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def run_freeze():
    print("=== 1. 为什么先冻结「LLM」只训 connector：一个能在 CPU 上 10 秒跑完的缩小版 ===")
    d, K = 16, 10
    # 「LLM」：在「文本」任务上预训练好的两层网络（输入 16 维「文本 embedding」，6 类）
    Ct = torch.randn(K, d) * 2                                          # 6 个类在「文本空间」里的中心
    Xt, yt = make_task(3000, Ct); Xt_te, yt_te = make_task(1000, Ct)
    llm = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, K))
    opt = torch.optim.Adam(llm.parameters(), 1e-2)
    for _ in range(300):
        idx = torch.randint(0, 3000, (128,)); loss = F.cross_entropy(llm(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    base = acc(llm, Xt_te, yt_te)
    print(f"  预训练好的「LLM」在文本任务上准确率 {base:.3f}")

    # 「视觉」任务：另一个空间（24 维）里的同样 6 类；connector 是 24 → 16 的线性层，随机初始化
    dv = 24
    Cv = torch.randn(K, dv) * 2                                         # 同样 6 个类在「视觉空间」里的中心，与文本空间毫无关系
    Xv, yv = make_task(3000, Cv); Xv_te, yv_te = make_task(1000, Cv)
    state0 = {k: v.clone() for k, v in llm.state_dict().items()}
    results = {}
    for mode in ("冻结 LLM，只训 connector", "LLM 与 connector 一起训（同一 lr）", "两阶段：先冻结训 connector，再一起训"):
        llm.load_state_dict(state0)
        conn = torch.nn.Linear(dv, d)
        hist = {"vision": [], "text": []}
        def train(steps, params, lr):
            opt = torch.optim.Adam(params, lr)
            for _ in range(steps):
                idx = torch.randint(0, 3000, (128,))
                loss = F.cross_entropy(llm(conn(Xv[idx])), yv[idx]); opt.zero_grad(); loss.backward(); opt.step()
                hist["vision"].append(acc(lambda X: llm(conn(X)), Xv_te, yv_te)); hist["text"].append(acc(llm, Xt_te, yt_te))
        if mode.startswith("冻结"):
            train(400, conn.parameters(), 1e-2)
        elif mode.startswith("LLM 与"):
            train(400, list(conn.parameters()) + list(llm.parameters()), 1e-2)
        else:
            train(150, conn.parameters(), 1e-2); train(250, list(conn.parameters()) + list(llm.parameters()), 1e-3)
        results[mode] = hist
        print(f"  {mode:<28} 视觉任务 {hist['vision'][-1]:.3f}   文本任务 {base:.3f} → {hist['text'][-1]:.3f}")
    print("  一起训：随机 connector 的噪声梯度改坏了 LLM 的参数，文本能力掉了；先冻结让 connector 先学会说 LLM 听得懂的话，再解冻时 LLM 收到的已是有意义的输入")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharex=True)
    colors = [C["blue"], C["red"], C["green"]]
    for (mode, hist), col in zip(results.items(), colors):
        axes[0].plot(hist["vision"], color=col, lw=1, label=mode); axes[1].plot(hist["text"], color=col, lw=1, label=mode)
    axes[0].set_title("视觉任务的准确率"); axes[1].set_title("文本任务的准确率（预训练能力保住了没）")
    axes[1].axhline(base, color=C["gray"], ls="--", lw=0.8)
    for ax in axes:
        ax.set_xlabel("训练步")
    axes[0].set_ylim(0, 1.02); axes[1].set_ylim(0.8, 1.0)
    axes[1].legend(fontsize=7, loc="lower left")
    save(fig, "03-freeze-vs-joint")


def run_cooccur():
    print("=== 2. 共现偏差 → 幻觉：一个只有两个特征的分类器 ===")
    # 训练集：「厨房」场景里 90% 有冰箱。模型看两个特征：场景是不是厨房（文本先验，总是准确）；视觉证据「有没有看到冰箱」（编码器弱，只有 60% 可靠）
    n = 20000
    kitchen = rng.random(n) < 0.5
    fridge = np.where(kitchen, rng.random(n) < 0.9, rng.random(n) < 0.1)      # 真实标签：厨房里 90% 有冰箱
    seen = np.where(rng.random(n) < 0.6, fridge, rng.random(n) < 0.5)          # 视觉证据：60% 情况下如实，40% 随机
    X = np.c_[kitchen, seen].astype(float); y = fridge.astype(int)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(X, y)
    print(f"  学到的权重：场景=厨房 {clf.coef_[0,0]:+.2f}，视觉证据=看到冰箱 {clf.coef_[0,1]:+.2f}，偏置 {clf.intercept_[0]:+.2f}")
    for k, s in [(1, 0), (1, 1), (0, 1), (0, 0)]:
        p = clf.predict_proba([[k, s]])[0, 1]
        print(f"  场景{'是' if k else '非'}厨房、视觉{'看到' if s else '没看到'}冰箱 → 模型说「有冰箱」的概率 {p:.2f}")
    print("  厨房里明明没看到冰箱，模型仍以约 0.7 说「有」——语言先验（共现统计）压过了不可靠的视觉证据；编码器越弱、先验越强，幻觉越多")
    for rel in (0.6, 0.8, 0.95, 1.0):
        seen = np.where(rng.random(n) < rel, fridge, rng.random(n) < 0.5)
        c = LogisticRegression().fit(np.c_[kitchen, seen].astype(float), y)
        print(f"     视觉证据可靠度 {rel:.2f} → 「厨房、没看到」时说有的概率 {c.predict_proba([[1, 0]])[0,1]:.2f}")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["freeze", "cooccur"]):
        globals()[f"run_{w}"](); print()
