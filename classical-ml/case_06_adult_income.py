"""案例（经典 ML 06）：人口普查收入预测（年收入 > 5 万美元？）——逻辑回归 → 随机森林 → 梯度提升。
https://arganzheng.life/ensembles-random-forest-and-gradient-boosting.html

    python case_06_adult_income.py

数据：UCI Adult，1994 年美国人口普查 48,842 人，14 个特征（6 数值 + 8 类别），正类 23.9%。
表格数据的标准考题：类别特征多、有缺失、特征间有交互，树模型的主场。图输出到 out/case-06-*.svg。
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from _data import adult
from _plot import C, plt, save

NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


def main():
    df = adult()
    y = df.pop("class").values
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = X[c].astype(str).replace("nan", "缺失")
    print(f"数据：{len(df):,} 人，年收入 >50K 的 {y.mean():.1%}；{len(NUM)} 个数值特征 + {len(CAT)} 个类别特征"
          f"（类别数：{', '.join(f'{c} {X[c].nunique()}' for c in CAT)}）")
    print(f"缺失：workclass / occupation 各约 2,800、native-country 857——当成一个类别「缺失」，树模型不在乎")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    onehot = ColumnTransformer([("num", make_pipeline(SimpleImputer(), StandardScaler()), NUM),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    ordinal = ColumnTransformer([("num", "passthrough", NUM),
                                 ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)])
    models = [
        ("基线：全部判 ≤50K", None),
        ("逻辑回归（one-hot 后 100 列）", make_pipeline(onehot, LogisticRegression(C=1.0, max_iter=2000))),
        ("一棵决策树（不限深度）", make_pipeline(ordinal, DecisionTreeClassifier(random_state=0))),
        ("一棵决策树（max_depth=6）", make_pipeline(ordinal, DecisionTreeClassifier(max_depth=6, random_state=0))),
        ("随机森林（500 棵）", make_pipeline(ordinal, RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0))),
        ("梯度提升（HistGB，默认 100 轮）", make_pipeline(ordinal, HistGradientBoostingClassifier(random_state=0))),
        ("梯度提升（500 轮，lr 0.05，早停）", make_pipeline(ordinal, HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=0))),
    ]
    print(f"\n{'模型':<34}{'准确率':>7}{'AUC':>7}{'训练':>8}{'预测':>7}")
    res = {}
    for name, m in models:
        if m is None:
            acc, auc, dt, dp, prob = np.mean(yte == 0), 0.5, 0, 0, None
        else:
            t = time.time(); m.fit(Xtr, ytr); dt = time.time() - t
            t = time.time(); prob = m.predict_proba(Xte)[:, 1]; dp = time.time() - t
            acc, auc = np.mean((prob >= 0.5) == yte), roc_auc_score(yte, prob)
        res[name] = (acc, auc, prob)
        print(f"{name:<34}{acc:>7.3f}{auc:>7.3f}{dt:>7.1f}s{dp:>6.2f}s")
    gb = models[-1][1]
    print(f"  最后一个模型早停在第 {gb[-1].n_iter_} 轮")
    print("解读：一棵不限深度的树 82%——把训练集背下来了；500 棵这样的树平均（随机森林）86%，AUC 从 0.75 跳到 0.91——"
          "bagging 只降方差，每棵树都过拟合也没关系；梯度提升再高一截 87.4% / 0.93，"
          "它是逐轮修正上一轮的错。逻辑回归 85.2% 并不差——但它要 one-hot 出 100 列、要标准化，树模型直接吃原始列。")

    # 特征重要性（permutation，在测试集上）
    m = models[-1][1]
    t = time.time()
    pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
    order = np.argsort(pi.importances_mean)[::-1]
    print(f"\n特征重要性（打乱这一列后 AUC 掉多少，测试集，{time.time() - t:.0f}s）：")
    for i in order:
        print(f"  {X.columns[i]:<16}{pi.importances_mean[i]:+.4f}")
    print("解读：资本收益、年龄、教育年限、婚姻状况、家庭关系是前五——和常识一致；race、native-country、sex 几乎不起作用。"
          "这也是审计的一部分：如果 sex 排第一，模型就在学一种不该学的关系。")

    # 学习率 × 轮数
    print("\n梯度提升的两个旋钮：学习率与轮数（训练集内 10% 做验证）")
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.1, random_state=1, stratify=ytr)
    curves = {}
    for lr in [0.3, 0.1, 0.03]:
        g = make_pipeline(ordinal, HistGradientBoostingClassifier(max_iter=400, learning_rate=lr, early_stopping=False, random_state=0)).fit(Xa, ya)
        Xv_t = g[0].transform(Xv)
        aucs = [roc_auc_score(yv, p[:, 1]) for p in g[-1].staged_predict_proba(Xv_t)]
        curves[lr] = aucs
        print(f"  lr = {lr:<5} 最佳轮数 {int(np.argmax(aucs)) + 1:>4}，最佳验证 AUC {max(aucs):.4f}，第 400 轮 {aucs[-1]:.4f}")
    print("解读：学习率大收敛快但早过拟合（AUC 掉头），学习率小要更多轮但更稳——这是第五章「调参」的实物。")

    # 图 1：AUC / 准确率对比
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    names = [n for n, _ in models[1:]]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [res[n][0] for n in names], w, color=C["blue"], label="准确率")
    ax.bar(x + w / 2, [res[n][1] for n in names], w, color=C["orange"], label="AUC")
    for i, n in enumerate(names):
        ax.text(i - w / 2, res[n][0] + 0.005, f"{res[n][0]:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, res[n][1] + 0.005, f"{res[n][1]:.3f}", ha="center", fontsize=7)
    ax.axhline(res["基线：全部判 ≤50K"][0], ls="--", c=C["gray"], lw=0.8)
    ax.text(5.4, res["基线：全部判 ≤50K"][0] + 0.005, "基线准确率（全判 ≤50K）0.761", fontsize=7, color=C["gray"], ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["逻辑回归", "一棵树\n不限深度", "一棵树\n深度 6", "随机森林\n500 棵", "梯度提升\n100 轮", "梯度提升\n500 轮早停"], fontsize=8)
    ax.set_ylim(0.7, 0.96); ax.legend(loc="upper left", ncol=2)
    ax.set_title("收入预测：从一棵树到一片森林到逐轮修正")
    save(fig, "case-06-models")

    # 图 2：ROC
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), width_ratios=[1, 1.2])
    ax = axes[0]
    for n, col in [("逻辑回归（one-hot 后 100 列）", C["gray"]), ("一棵决策树（不限深度）", C["red"]), ("随机森林（500 棵）", C["blue"]), ("梯度提升（500 轮，lr 0.05，早停）", C["green"])]:
        fpr, tpr, _ = roc_curve(yte, res[n][2]); ax.plot(fpr, tpr, c=col, lw=1.2, label=f"{n.split('（')[0]} {res[n][1]:.3f}")
    ax.plot([0, 1], [0, 1], ls=":", c=C["light"]); ax.set_xlabel("假正率"); ax.set_ylabel("真正率"); ax.set_title("ROC（数字是 AUC）"); ax.legend(fontsize=7)
    ax = axes[1]
    ax.barh([X.columns[i] for i in order[::-1]], pi.importances_mean[order[::-1]], color=C["purple"])
    ax.set_xlabel("打乱这一列后 AUC 掉多少"); ax.set_title("特征重要性（梯度提升，permutation）")
    save(fig, "case-06-roc-importance")

    # 图 3：学习率 × 轮数
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    for (lr, aucs), col in zip(curves.items(), [C["red"], C["blue"], C["green"]]):
        ax.plot(range(1, len(aucs) + 1), aucs, c=col, label=f"学习率 {lr}（最佳第 {int(np.argmax(aucs)) + 1} 轮）")
    ax.set_xlabel("轮数（树的棵数）"); ax.set_ylabel("验证 AUC"); ax.set_ylim(0.90, 0.935); ax.legend()
    ax.set_title("梯度提升：学习率大 → 快但早过拟合；小 → 慢但稳")
    save(fig, "case-06-lr-rounds")


if __name__ == "__main__":
    main()
