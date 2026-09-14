"""评估（经典 ML 06）：混淆矩阵、精确率/召回率随阈值的权衡、AUC、类别不平衡的准确率陷阱、校准与 ECE、交叉验证、配对检验、多重比较。
https://arganzheng.life/evaluation-from-confusion-matrix-to-judge-agreement.html

    python 06_evaluation.py            # 全部：threshold imbalance calibration cv paired multiple
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def data(n=6000, weights=(0.5, 0.5), seed=0):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=8, weights=list(weights), flip_y=0.03, class_sep=1.0, random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


# ---------------- 1. 阈值：精确率与召回率的权衡 ----------------
def exp_threshold():
    print("=== 1. 同一个分类器，不同阈值：把它当'有害内容过滤器'读 ===")
    Xtr, Xte, ytr, yte = data()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"  AUC {roc_auc_score(yte, p):.3f}（与阈值无关：随机取一正一负，正例分数更高的概率）")
    print(f"  {'阈值':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'精确率':>7} {'召回率':>7} {'F1':>6}   含义")
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        note = "宽松：漏放少、误杀多" if t <= 0.3 else ("严格：误杀少、漏放多" if t >= 0.7 else "")
        print(f"  {t:>5.1f} {tp:>5} {fp:>5} {fn:>5} {tn:>5} {prec:>7.3f} {rec:>7.3f} {f1:>6.3f}   {note}")
    prec, rec, thr = precision_recall_curve(yte, p)
    i = int(np.max(np.where(rec >= 0.95)[0]))          # 召回随阈值升高而下降：取满足召回 ≥ 95% 的最高阈值
    print(f"  要求召回 ≥ 95%（漏放 < 5%）时阈值约 {thr[min(i, len(thr)-1)]:.2f}，精确率掉到 {prec[i]:.3f}——这就是 FineWeb-Edu 选 3 分还是 2 分切的那种取舍")
    print()


# ---------------- 2. 类别不平衡：准确率会骗人 ----------------
def exp_imbalance():
    print("=== 2. 正例只占 3%：准确率 97% 的'全判负'分类器 ===")
    Xtr, Xte, ytr, yte = data(weights=(0.97, 0.03), seed=1)
    dummy_acc = (yte == 0).mean()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"  测试集 {len(yte)} 条，正例 {yte.sum()} 条；全判负的准确率 {dummy_acc:.3f}")
    print(f"  逻辑回归阈值 0.5：准确率 {(pred == yte).mean():.3f}，但召回率只有 {tp/(tp+fn):.3f}（漏掉 {fn}/{tp+fn} 个正例）；AUC {roc_auc_score(yte, p):.3f}")
    pred2 = (p >= 0.1).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred2).ravel()
    print(f"  阈值降到 0.1：召回 {tp/(tp+fn):.3f}、精确率 {tp/(tp+fp):.3f}、准确率 {(pred2 == yte).mean():.3f}——不平衡时看 P/R/AUC，不看准确率")
    print()


# ---------------- 3. 校准 ----------------
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def exp_calibration():
    print("=== 3. 校准：说 80% 的时候，是不是十次里八次对 ===")
    Xtr, Xte, ytr, yte = data(seed=2)
    models = {"逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "梯度提升": GradientBoostingClassifier(n_estimators=300, random_state=0),
              "SVM (概率)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))}
    for name, m in models.items():
        p = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        frac, mean_p = calibration_curve(yte, p, n_bins=10)
        print(f"  {name:<10} AUC {roc_auc_score(yte, p):.3f}  ECE {ece(yte, p):.3f}   预测 0.7–0.8 那一档的实际正例率 {frac[np.argmin(np.abs(mean_p - 0.75))]:.2f}")
    print("  逻辑回归天然校准好（loss 就是对数似然）；树模型 / SVM 的分数排序对但概率不准，要 Platt / isotonic 校准")
    print("  LLM 上：RLHF 之后模型对自己答案的置信度校准变差（学会了'自信'）；奖励模型的 σ(r_w − r_l) 应等于人类偏好比例")
    print()


# ---------------- 4. 交叉验证 ----------------
def exp_cv():
    print("=== 4. 交叉验证：数据少时用 5 折代替一次划分 ===")
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6, random_state=3)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    single = []
    for s in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s)
        single.append(clf.fit(Xtr, ytr).score(Xte, yte))
    cv = cross_val_score(clf, X, y, cv=5)
    print(f"  300 条数据。5 次不同的单次 70/30 划分: {np.round(single, 3).tolist()}（同一个模型，分数抖 {max(single) - min(single):.3f}）")
    print(f"  5 折交叉验证: 均值 {cv.mean():.3f} ± {cv.std():.3f}——每条数据都当过一次验证，估计更稳，代价是训 5 次")
    print()


# ---------------- 5. 配对检验：同一套题上比两个模型 ----------------
def exp_paired():
    print("=== 5. 配对比较：500 题上 A 与 B 差 3 个点，独立比较 vs 配对比较 ===")
    r = np.random.default_rng(0)
    n = 500
    # 两个模型高度相关：共享 85% 的'难度'，A 略强
    skill = r.random(n)
    a = (r.random(n) < 0.15 + 0.7 * (skill > 0.35)).astype(int)
    b = a.copy()
    flip = r.random(n) < 0.12
    b[flip] = (r.random(flip.sum()) < 0.35).astype(int)
    pa, pb = a.mean(), b.mean()
    se_ind = np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    z_mcnemar = (n01 - n10) / np.sqrt(n01 + n10)
    print(f"  A 正确率 {pa:.3f}，B 正确率 {pb:.3f}，差 {pa - pb:+.3f}")
    print(f"  独立比较：差值标准误 {se_ind:.3f}，z = {(pa - pb)/se_ind:.2f} → {'显著' if abs((pa - pb)/se_ind) > 1.96 else '不显著'}（|z| > 1.96 才显著）")
    print(f"  配对比较：A 对 B 错 {n01} 题、A 错 B 对 {n10} 题，其余一致；McNemar z = {z_mcnemar:.2f} → {'显著' if abs(z_mcnemar) > 1.96 else '不显著'}")
    print("  同一套题上两个模型答错的题高度重叠，差异集中在少数分歧题上——配对检验只看这些题，噪声更小")
    print()


# ---------------- 6. 多重比较 ----------------
def exp_multiple():
    print("=== 6. 多重比较：两个一模一样的模型在 20 个 benchmark 上比 ===")
    r = np.random.default_rng(0)
    n_bench, n_q, trials = 20, 200, 2000
    at_least_one = 0
    n_sig = []
    for _ in range(trials):
        a = r.binomial(n_q, 0.6, n_bench) / n_q
        b = r.binomial(n_q, 0.6, n_bench) / n_q
        se = np.sqrt(2 * 0.6 * 0.4 / n_q)
        sig = np.abs(a - b) / se > 1.96
        n_sig.append(sig.sum())
        at_least_one += sig.any()
    n_sig = np.array(n_sig)
    print(f"  {trials} 次模拟：两个完全相同的模型，5% 显著性水平下")
    print(f"    平均 {n_sig.mean():.2f} 个 benchmark 上'显著'不同（期望 20 × 5% = 1）；至少一个显著的概率 {at_least_one/trials*100:.0f}%")
    print(f"    '20 个里领先 12 个'：随机情况下期望领先 10 个，标准差 {np.sqrt(20*0.25):.1f}——12 个不算什么")
    print("  报多个 benchmark 时先问：随机情况下期望几个显著、几个领先")
    print()


EXPS = {"threshold": exp_threshold, "imbalance": exp_imbalance, "calibration": exp_calibration, "cv": exp_cv, "paired": exp_paired, "multiple": exp_multiple}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
