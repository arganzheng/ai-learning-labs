"""案例（经典 ML 05）：SVM 做 MNIST 手写数字——把 LeCun 等 1998 那张对比表的几行重跑一遍。
https://arganzheng.life/svm-and-kernel-methods.html

    python case_05_svm_mnist.py            # 全部（RBF-SVM 全量训练几分钟）
    python case_05_svm_mnist.py grid       # 只跑 C / γ 网格

MNIST 1998 年发布时就是为了比较分类器：线性 12%、KNN 5%、SVM 1.1%、LeNet-5 0.95%。
这里用同一份数据，在今天的笔记本上重跑线性 / KNN / RBF-SVM。图输出到 out/case-05-*.svg。
"""
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from _data import mnist
from _plot import C, plt, save

X, y = mnist("train"); Xt, yt = mnist("test")


def run(name, model, n_train, note=""):
    idx = np.random.default_rng(0).choice(len(X), n_train, replace=False) if n_train < len(X) else slice(None)
    t = time.time(); model.fit(X[idx], y[idx]); t_fit = time.time() - t
    t = time.time(); pred = model.predict(Xt); t_pred = time.time() - t
    err = np.mean(pred != yt)
    extra = f"  支持向量 {model.n_support_.sum():,} / {n_train:,}" if hasattr(model, "n_support_") else ""
    print(f"  {name:<34} 训练 {n_train:>6,} 张  错误率 {err:6.2%}  训练 {t_fit:6.1f}s  预测 10k 张 {t_pred:5.1f}s{extra}{note}")
    return err, pred


def exp_grid():
    print("=== 1. RBF-SVM 的两个超参：C（软间隔松紧）与 γ（核的宽度），10k 训练子集、2k 验证 ===")
    idx = np.random.default_rng(0).choice(len(X), 10000, replace=False)
    Xs, ys = X[idx], y[idx]; Xv, yv = Xt[:2000], yt[:2000]
    Cs, gs = [0.1, 1, 10, 100], [0.001, 0.01, 0.03, 0.1]
    E = np.zeros((len(Cs), len(gs))); NSV = np.zeros_like(E, dtype=int)
    for i, c in enumerate(Cs):
        for j, g in enumerate(gs):
            m = SVC(C=c, gamma=g).fit(Xs, ys)
            E[i, j] = 1 - m.score(Xv, yv); NSV[i, j] = m.n_support_.sum()
        print(f"  C = {c:<5}" + "  ".join(f"γ={g}: {E[i, j]:.2%} (SV {NSV[i, j]:,})" for j, g in enumerate(gs)))
    bi, bj = np.unravel_index(E.argmin(), E.shape)
    print(f"  最好：C = {Cs[bi]}, γ = {gs[bj]}，错误率 {E[bi, bj]:.2%}；scikit-learn 默认 gamma='scale' = 1/(784·Var) ≈ {1 / (784 * X.var()):.4f}")
    print("  解读：γ 太大（0.1）核太窄，每个训练点只认自己附近，几乎全部成为支持向量——过拟合；"
          "γ 太小核太宽退化成近似线性。C 大 = 少犯错但边界贴数据，在干净的 MNIST 上 C 大一点没坏处。")

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    from matplotlib.colors import LogNorm
    im = ax.imshow(E * 100, cmap="Blues_r", norm=LogNorm(vmin=3, vmax=70))
    for i in range(len(Cs)):
        for j in range(len(gs)):
            ax.text(j, i, f"{E[i, j]:.1%}", ha="center", va="center", fontsize=8, color="white" if E[i, j] < 0.15 else "black")
    ax.set_xticks(range(len(gs))); ax.set_xticklabels([f"γ={g}" for g in gs])
    ax.set_yticks(range(len(Cs))); ax.set_yticklabels([f"C={c}" for c in Cs])
    ax.set_title("RBF-SVM 验证错误率（10k 训练子集）")
    fig.colorbar(im, ax=ax, label="错误率 %（对数刻度）", shrink=0.8)
    save(fig, "case-05-svm-grid")


def exp_table():
    print("\n=== 2. 1998 年那张表，今天重跑（全部 60,000 张训练、10,000 张测试）===")
    rows = {}
    rows["线性分类器（逻辑回归）"] = run("线性分类器（softmax 回归）", LogisticRegression(max_iter=200, C=0.1), 60000)[0]
    rows["线性 SVM"] = run("线性 SVM（hinge + L2）", LinearSVC(C=0.01, max_iter=5000), 60000)[0]
    rows["KNN k=3"] = run("KNN，k=3，欧氏距离", KNeighborsClassifier(3, algorithm="brute", n_jobs=-1), 60000)[0]
    rows["RBF-SVM 10k"] = run("RBF-SVM，C=10, γ=0.03", SVC(C=10, gamma=0.03), 10000)[0]
    err, pred = run("RBF-SVM，C=10, γ=0.03", SVC(C=10, gamma=0.03, cache_size=2000), 60000)
    rows["RBF-SVM 60k"] = err
    print("""
  对照 LeCun, Bottou, Bengio, Haffner 1998《Gradient-Based Learning Applied to Document Recognition》表（原始像素，未去斜）：
    线性分类器 12.0%   K-NN（欧氏）5.0%   SVM（多项式核，Cortes & Vapnik 1995 的做法）1.1%   LeNet-5 0.95%
  今天的差别：线性 7–8% 而不是 12%——他们的"线性分类器"是无正则的单层网络、像素也没归一化；KNN 3% 而不是 5%——同样是预处理不同；
  RBF 核 1.4% vs 多项式核 1.1%——核不同、超参也没细调，量级一致。
  结论没变：线性 → 邻居 → 核 → 卷积网络，错误率一路降；SVM 与 LeNet-5 在 1998 年打平，
  之后十年 SVM 是默认选择——直到数据量和算力让卷积网络在 ImageNet 上把差距拉开（L3 第五篇）。""")

    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    names = ["线性\n（softmax 回归）", "线性 SVM", "KNN\nk=3", "RBF-SVM\n10k 训练", "RBF-SVM\n60k 训练"]
    vals = [rows[k] * 100 for k in ["线性分类器（逻辑回归）", "线性 SVM", "KNN k=3", "RBF-SVM 10k", "RBF-SVM 60k"]]
    ax.bar(range(5), vals, color=[C["gray"], C["gray"], C["blue"], C["green"], C["green"]])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=8)
    ax.axhline(0.95, ls="--", c=C["red"], lw=1); ax.text(2.5, 1.1, "LeNet-5（1998）0.95%", ha="center", fontsize=8, color=C["red"])
    ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_ylabel("测试错误率 %"); ax.set_ylim(0, max(vals) + 1.2)
    ax.set_title("MNIST 上从线性到核方法：错误率一路降")
    save(fig, "case-05-mnist-table")

    wrong = np.where(pred != yt)[0][:24]
    fig, ax = plt.subplots(figsize=(7.6, 2.6)); ax.axis("off")
    ax.set_title(f"RBF-SVM 全量训练错分的 {len(np.where(pred != yt)[0])} 张里的前 24 张（真实→预测）", fontsize=9)
    for i, j in enumerate(wrong):
        r, c = divmod(i, 12)
        ax.imshow(Xt[j].reshape(28, 28), cmap="gray_r", extent=(c, c + 0.9, -r * 1.35 - 0.9, -r * 1.35))
        ax.text(c + 0.45, -r * 1.35 - 1.0, f"{yt[j]}→{pred[j]}", ha="center", va="top", fontsize=6.5, color=C["red"])
    ax.set_xlim(0, 12); ax.set_ylim(-2.7, 0); ax.set_aspect("equal")
    save(fig, "case-05-svm-errors")


EXPS = {"grid": exp_grid, "table": exp_table}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
