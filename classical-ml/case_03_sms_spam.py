"""案例（经典 ML 03）：垃圾短信识别——TF-IDF + 逻辑回归。
https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html

    python case_03_sms_spam.py

数据：UCI SMS Spam Collection，5,574 条英文短信，747 条垃圾（13.4%）。
图输出到 out/case-03-*.svg。
"""
from itertools import pairwise

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from _data import sms_spam
from _plot import C, plt, save


def report(name, y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    acc = np.mean(y == pred)
    print(f"  {name:<34} 准确率 {acc:.3f}  精确率 {p:.3f}  召回率 {r:.3f}  F1 {f:.3f}")
    return acc, p, r, f


def main():
    df = sms_spam()
    y = (df.label == "spam").astype(int).values
    print(f"数据：{len(df)} 条短信，spam {y.sum()} 条（{y.mean():.1%}）；平均长度 ham {df.text[y == 0].str.len().mean():.0f} 字符，"
          f"spam {df.text[y == 1].str.len().mean():.0f} 字符")
    for lab in ["ham", "spam"]:
        print(f"  {lab} 例：{df.text[df.label == lab].iloc[3][:90]}")
    Xtr, Xte, ytr, yte = train_test_split(df.text, y, test_size=0.2, random_state=0, stratify=y)
    print(f"划分：训练 {len(Xtr)}，测试 {len(Xte)}（按类别比例分层）")

    print("\n=== 效果 ===")
    report("基线：全部判 ham", yte, np.zeros_like(yte))
    # 词袋 + 逻辑回归；ngram (1,2) 让 "free entry"、"call now" 这种词对也成为特征
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(C=10, max_iter=2000))
    model.fit(Xtr, ytr)
    vocab = model[0].get_feature_names_out()
    print(f"  词表：{len(vocab):,} 个 1-gram / 2-gram → 每条短信是一个 {len(vocab):,} 维的稀疏向量；"
          f"逻辑回归 = {len(vocab):,} 个权重 + 1 个偏置")
    prob = model.predict_proba(Xte)[:, 1]
    report("TF-IDF + 逻辑回归（阈值 0.5）", yte, (prob >= 0.5).astype(int))
    report("同上，阈值 0.3", yte, (prob >= 0.3).astype(int))
    report("同上，阈值 0.2", yte, (prob >= 0.2).astype(int))
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred)
    print(f"  混淆矩阵（阈值 0.5）：真 ham 判 ham {cm[0, 0]}，真 ham 判 spam {cm[0, 1]}（误杀），"
          f"真 spam 判 ham {cm[1, 0]}（漏网），真 spam 判 spam {cm[1, 1]}")

    # 权重最大 / 最小的词：模型学到了什么
    w = model[-1].coef_[0]
    top = np.argsort(w)
    print("\n=== 权重最大的 15 个词（推向 spam）===")
    print("  " + "  ".join(f"{vocab[i]}({w[i]:+.1f})" for i in top[::-1][:15]))
    print("=== 权重最小的 15 个词（推向 ham）===")
    print("  " + "  ".join(f"{vocab[i]}({w[i]:+.1f})" for i in top[:15]))

    # 看几条错的
    print("\n=== 错在哪 ===")
    Xte_l = Xte.reset_index(drop=True)
    for kind, mask in [("漏网的 spam", (yte == 1) & (pred == 0)), ("误杀的 ham", (yte == 0) & (pred == 1))]:
        idx = np.where(mask)[0][:3]
        print(f"  {kind}（共 {mask.sum()} 条）：")
        for i in idx:
            print(f"    p(spam)={prob[i]:.2f}  {Xte_l[i][:100]}")

    # 概率校准：预测概率分桶，对比每桶里实际的 spam 比例
    print("\n=== 概率校准：说 70% 的那些短信里真有 70% 是 spam 吗 ===")
    edges = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if m.sum() == 0:
            continue
        ece += m.sum() / len(yte) * abs(prob[m].mean() - yte[m].mean())
        print(f"  预测 p(spam) ∈ [{lo:.1f}, {hi:.1f}{')' if hi < 1 else ']'}：{m.sum():>4} 条，平均预测 {prob[m].mean():.3f}，实际 spam 比例 {yte[m].mean():.3f}")
    print(f"  ECE（各桶 |平均预测 − 实际比例| 按桶大小加权）= {ece:.3f}，Brier = {np.mean((prob - yte) ** 2):.4f}；"
          f"0.5 以上的三个桶里实际 spam 比例都是 1.0、高于平均预测——模型在高概率区偏保守，这与阈值 0.5 时精确率 1.000、召回率 {np.mean(pred[yte == 1]):.3f} 是同一件事的两种说法")
    print("  准确率说的是阈值后判断对了多少；校准说的是概率本身可不可信——两者可以一个好一个差")

    # 图 1：混淆矩阵
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["判 ham", "判 spam"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["真 ham", "真 spam"])
    ax.set_title("测试集 1,115 条的混淆矩阵（阈值 0.5）")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "case-03-confusion")

    # 图 2：阈值移动 → 精确率 / 召回率此消彼长
    pr, rc, th = precision_recall_curve(yte, prob)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(th, pr[:-1], c=C["blue"], label="精确率（判成 spam 的里面真是 spam 的比例）")
    ax.plot(th, rc[:-1], c=C["orange"], label="召回率（所有 spam 里被抓住的比例）")
    for t in [0.2, 0.5]:
        ax.axvline(t, ls="--", c=C["gray"], lw=0.8); ax.text(t + 0.01, 0.62, f"阈值 {t}", fontsize=8, color=C["gray"])
    ax.set_xlabel("判为 spam 的概率阈值"); ax.set_ylim(0.5, 1.02); ax.legend(loc="lower right")
    save(fig, "case-03-threshold")

    # 图 3：权重最大 / 最小的词
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    idx = np.r_[top[:12], top[::-1][:12][::-1]]
    ax.barh(range(len(idx)), w[idx], color=[C["green"] if w[i] < 0 else C["red"] for i in idx])
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([vocab[i] for i in idx], fontsize=8)
    ax.set_xlabel("逻辑回归权重（正 → spam，负 → ham）")
    ax.set_title("模型学到的词：下面是「人话」，上面是「广告话」")
    save(fig, "case-03-top-words")


if __name__ == "__main__":
    main()
