"""案例（经典 ML 10）：银行电话营销——一个 11.7% 正类的问题，怎么评、怎么定阈值、怎么比两个模型。
https://arganzheng.life/evaluation-from-confusion-matrix-to-judge-agreement.html

    python case_10_bank_marketing.py

数据：UCI Bank Marketing，45,211 次电话，目标：客户是否订了定期存款。图输出到 out/case-10-*.svg。
"""
import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from _data import bank_marketing
from _plot import C, plt, save

NUM = ["age", "balance", "day", "campaign", "pdays", "previous"]
CAT = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def main():
    df = bank_marketing()
    y = df.pop("y").values
    print(f"数据：{len(df):,} 次电话，订了定期存款的 {y.sum():,}（{y.mean():.1%}）；特征 {len(NUM)} 数值 + {len(CAT)} 类别")
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.25, random_state=0, stratify=y)

    def pre(kind):
        if kind == "onehot":
            return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
        return ColumnTransformer([("num", "passthrough", NUM), ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)])

    # ---- 0. 一个泄漏特征 ----
    print("\n=== 0. 先排除一个泄漏：duration（通话时长）===")
    m_leak = make_pipeline(pre("ordinal"), HistGradientBoostingClassifier(random_state=0))
    cols_leak = NUM + ["duration"] + CAT
    pre_leak = ColumnTransformer([("num", "passthrough", NUM + ["duration"]), ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)])
    m_leak = make_pipeline(pre_leak, HistGradientBoostingClassifier(random_state=0)).fit(Xtr[cols_leak], ytr)
    auc_leak = roc_auc_score(yte, m_leak.predict_proba(Xte[cols_leak])[:, 1])
    print(f"  带 duration 的模型 AUC {auc_leak:.3f}——但通话时长要打完电话才知道，而模型的用途是决定'给谁打'。"
          f"UCI 的说明也写了：要做真实预测必须去掉它。下面全部不用。")

    # ---- 1. 三个模型 ----
    print("\n=== 1. 准确率没用：11.7% 的正类 ===")
    models = {
        "逻辑回归": make_pipeline(pre("onehot"), LogisticRegression(max_iter=3000)),
        "随机森林": make_pipeline(pre("ordinal"), RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0)),
        "梯度提升": make_pipeline(pre("ordinal"), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)),
    }
    probs = {}
    print(f"  {'模型':<10}{'准确率(0.5)':>11}{'AUC':>7}{'AP':>7}{'ECE':>7}")
    print(f"  {'全判 0':<10}{np.mean(yte == 0):>11.3f}{0.5:>7.3f}{y.mean():>7.3f}{'—':>7}")
    for name, m in models.items():
        p = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]; probs[name] = p
        print(f"  {name:<10}{np.mean((p >= 0.5) == yte):>11.3f}{roc_auc_score(yte, p):>7.3f}{average_precision_score(yte, p):>7.3f}{ece(yte, p):>7.3f}")
    print("  解读：三个模型的准确率都和'全判 0'的 88.3% 差不多——不平衡数据上准确率不说话；AUC 0.75–0.80、AP 0.4–0.5 才分得出高下。"
          "AP（PR 曲线下面积）的基线是正类比例 0.117，比 AUC 的基线 0.5 更能体现'从 12% 里挑人'有多难。")

    # ---- 2. 按业务成本定阈值 ----
    print("\n=== 2. 阈值不是 0.5：按成本定 ===")
    p = probs["梯度提升"]
    n_all = len(yte)
    ths = np.linspace(0.02, 0.9, 200)

    def profit_curve(cost_call, gain):
        return np.array([(gain * ((p >= t) & (yte == 1)).sum() - cost_call * (p >= t).sum()) for t in ths])

    scen = {}
    for cost_call, gain in [(5, 100), (25, 100)]:
        profit = profit_curve(cost_call, gain); best = ths[profit.argmax()]
        all_call = gain * yte.sum() - cost_call * n_all
        scen[(cost_call, gain)] = (profit, best, all_call)
        print(f"\n  假设：打一次电话成本 {cost_call} 欧，成交一单收益 {gain} 欧。测试集 {n_all:,} 人：")
        for t in sorted({0.05, 0.1, round(best, 2), 0.3, 0.5}):
            sel = p >= t; tp = (sel & (yte == 1)).sum()
            print(f"    阈值 {t:.2f}：打 {sel.sum():>6,} 个电话，成交 {tp:>5,}（召回 {tp / yte.sum():.0%}，精确率 {tp / max(sel.sum(), 1):.0%}），利润 {gain * tp - cost_call * sel.sum():>8,} 欧")
        print(f"    全打：打 {n_all:,} 个，成交 {yte.sum():,}，利润 {all_call:,} 欧")
        print(f"  → 最优阈值 {best:.2f}，只打 {(p >= best).mean():.0%} 的人，拿到 {((p >= best) & (yte == 1)).sum() / yte.sum():.0%} 的成交，利润 {profit.max():,} 欧" + (f" = 全打的 {profit.max() / all_call:.2f} 倍" if all_call > 0 else f"；全打亏 {-all_call:,} 欧"))
    print("  解读：两点。（一）最优阈值是 0.07 / 0.2 而不是 0.5——正类只有 12%，模型给大多数人的概率都很低，0.5 只挑出 425 人；"
          "（二）电话便宜（5 欧）时全打也不亏，模型只多赚 8%；电话贵（25 欧）时全打亏钱，模型的价值才显出来。"
          "阈值和'模型值多少钱'都由成本 / 收益决定，是业务参数，不是模型参数。")

    # ---- 3. 校准 ----
    print("\n=== 3. 概率准不准（校准）===")
    for name, pp in probs.items():
        frac, mean_p = calibration_curve(yte, pp, n_bins=10, strategy="quantile")
        print(f"  {name}：ECE {ece(yte, pp):.3f}；概率最高的一档平均预测 {mean_p[-1]:.2f}、实际成交 {frac[-1]:.2f}")
    print("  解读：三个模型的 ECE 都在 0.01 上下——预测 0.5 的那一档实际成交也是 5 成，概率可信。"
          "这是上一步按成本定阈值的前提：5 欧 / 100 欧的算术建在'p 就是成交概率'上，p 不准算出来的阈值就是错的。"
          "逻辑回归天生校准好；树的集成不一定（第五章），这里随机森林每叶至少 5 个样本、梯度提升用对数损失训，所以也还好。")

    # ---- 4. 两个模型谁更好：配对检验 ----
    print("\n=== 4. 随机森林 vs 梯度提升，差 0.001 AUC 是真的吗 ===")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    a_rf, a_gb = [], []
    for tr, va in skf.split(df, y):
        for name, lst in [("随机森林", a_rf), ("梯度提升", a_gb)]:
            m = models[name].fit(df.iloc[tr], y[tr])
            lst.append(roc_auc_score(y[va], m.predict_proba(df.iloc[va])[:, 1]))
    d = np.array(a_gb) - np.array(a_rf)
    t, pv = stats.ttest_rel(a_gb, a_rf)
    print(f"  5 折 AUC  随机森林 {' '.join(f'{x:.4f}' for x in a_rf)} → {np.mean(a_rf):.4f} ± {np.std(a_rf):.4f}")
    print(f"           梯度提升 {' '.join(f'{x:.4f}' for x in a_gb)} → {np.mean(a_gb):.4f} ± {np.std(a_gb):.4f}")
    print(f"  同一折上的差：{' '.join(f'{x:+.4f}' for x in d)} → {int((d > 0).sum())} 折为正，配对 t 检验 p = {pv:.3f}")
    print("  解读：测试集上梯度提升 0.796 vs 随机森林 0.795，看起来赢了；5 折配对一比，差值有正有负、p = 0.1——"
          "**这个差别不显著**，换一份测试集可能反过来。'谁更好'要看同一份数据上的成对差值及其波动，"
          "不是各报一个数比大小（第八章）。要分出高下，要么更多折、要么承认两者一样好、按别的标准选（训练时间、可解释性）。")

    # 图 1：ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for (name, pp), col in zip(probs.items(), [C["gray"], C["blue"], C["green"]]):
        fpr, tpr, _ = roc_curve(yte, pp); axes[0].plot(fpr, tpr, c=col, label=f"{name} AUC {roc_auc_score(yte, pp):.3f}")
        pr, rc, _ = precision_recall_curve(yte, pp); axes[1].plot(rc, pr, c=col, label=f"{name} AP {average_precision_score(yte, pp):.3f}")
    axes[0].plot([0, 1], [0, 1], ":", c=C["light"]); axes[0].set_xlabel("假正率"); axes[0].set_ylabel("真正率（召回）"); axes[0].set_title("ROC：基线 0.5"); axes[0].legend(fontsize=7)
    axes[1].axhline(y.mean(), ls=":", c=C["light"]); axes[1].set_xlabel("召回率"); axes[1].set_ylabel("精确率"); axes[1].set_title("PR：基线 = 正类比例 0.117"); axes[1].legend(fontsize=7)
    save(fig, "case-10-roc-pr")

    # 图 2：利润 vs 阈值 + 校准
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    for (cost_call, gain), col in zip(scen, [C["green"], C["orange"]]):
        profit, best, all_call = scen[(cost_call, gain)]
        ax.plot(ths, profit / 1000, c=col, label=f"电话 {cost_call} 欧 / 成交 {gain} 欧（最优阈值 {best:.2f}）")
        ax.axvline(best, ls="--", c=col, lw=0.8)
        ax.axhline(all_call / 1000, ls=":", c=col, lw=0.8)
    ax.axvline(0.5, ls=":", c=C["gray"], lw=0.8); ax.text(0.51, 60, "默认 0.5", fontsize=7, color=C["gray"])
    ax.text(0.55, 75.7 + 3, "全打（5 欧）", fontsize=7, color=C["green"]); ax.text(0.55, -150 + 6, "全打（25 欧）：亏", fontsize=7, color=C["orange"])
    ax.set_xlabel("判为「会成交」的概率阈值"); ax.set_ylabel("利润（千欧）"); ax.set_title("利润随阈值：阈值由成本决定"); ax.legend(fontsize=6.5, loc="center right")
    ax.set_ylim(-160, 100)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], ":", c=C["light"])
    for (name, pp), col in zip(probs.items(), [C["gray"], C["blue"], C["green"]]):
        frac, mean_p = calibration_curve(yte, pp, n_bins=10, strategy="quantile"); ax.plot(mean_p, frac, "o-", ms=3, c=col, label=f"{name} ECE {ece(yte, pp):.3f}")
    ax.set_xlabel("预测的成交概率"); ax.set_ylabel("实际成交比例"); ax.set_title("可靠性图：在对角线上才可信"); ax.legend(fontsize=7)
    save(fig, "case-10-profit-calibration")


if __name__ == "__main__":
    main()
