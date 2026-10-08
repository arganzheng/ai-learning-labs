"""评估（经典 ML 10）：手写混淆矩阵 / 精确率 / 召回率 / ROC-AUC / ECE，阈值扫描，类别不平衡，校准（Platt / isotonic），
Cohen's κ 与 judge 的位置偏差，交叉验证，bootstrap 置信区间，配对检验，多重比较。
https://arganzheng.life/evaluation-from-confusion-matrix-to-judge-agreement.html

    python 10_evaluation.py            # 全部：metrics threshold imbalance calibration kappa cv bootstrap paired multiple

图输出到 out/10-*.svg。
"""
import sys
import warnings
from itertools import pairwise

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from _plot import C, plt, save


def data(n=6000, weights=(0.5, 0.5), seed=0):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=8, weights=list(weights), flip_y=0.03, class_sep=1.0, random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


# ---------------- 手写指标 ----------------
def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())   # ① 四个格子就是四次计数
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def prf(y, pred):
    tp, fp, fn, _tn = confusion(y, pred)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)                              # ② 精确率：判为正里多少真正；召回：真正里抓到多少
    f1 = 2 * prec * rec / max(1e-9, prec + rec)                                          # ③ 调和平均
    return prec, rec, f1


def roc_auc(y, p):
    order = np.argsort(-p)                                                              # ④ 按分数从高到低扫阈值
    y = y[order]
    tpr = np.cumsum(y) / y.sum(); fpr = np.cumsum(1 - y) / (1 - y).sum()                 # ⑤ 每个阈值处的召回与假正率
    return float(np.trapezoid(tpr, fpr)), fpr, tpr                                      # ⑥ 曲线下面积


def exp_metrics():
    print("=== 0. 手写指标 vs sklearn ===")
    Xtr, Xte, ytr, yte = data()
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"  混淆矩阵 (TP, FP, FN, TN) 手写 {confusion(yte, pred)}；sklearn {tuple(int(v) for v in confusion_matrix(yte, pred).ravel()[[3, 1, 2, 0]])}")
    prec, rec, f1 = prf(yte, pred)
    print(f"  精确率 {prec:.4f} 召回率 {rec:.4f} F1 {f1:.4f}")
    auc, _, _ = roc_auc(yte, p)
    print(f"  AUC 手写 {auc:.4f}；sklearn {roc_auc_score(yte, p):.4f}")
    # AUC 的概率解释：随机取一正一负，正例分数更高的比例
    r = np.random.default_rng(0)
    pos, neg = p[yte == 1], p[yte == 0]
    pairs = (pos[r.integers(0, len(pos), 200000)] > neg[r.integers(0, len(neg), 200000)]).mean()
    print(f"  随机抽 20 万对（一正一负），正例分数更高的比例 {pairs:.4f}——这就是 AUC 的含义")
    print()


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
    ts = np.linspace(0.02, 0.98, 97)
    curves = np.array([[*prf(yte, (p >= t).astype(int)), ((p >= t) == yte).mean()] for t in ts])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    ax = axes[0]
    for j, (name, col) in enumerate((("精确率", C["blue"]), ("召回率", C["red"]), ("F1", C["green"]), ("准确率", C["gray"]))):
        ax.plot(ts, curves[:, j], color=col, label=name, lw=1.4 if j < 3 else 0.9, ls="-" if j < 3 else "--")
    ax.axvline(0.5, color=C["light"], lw=0.8); ax.set_xlabel("阈值"); ax.set_ylabel("指标值"); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("同一个分类器：阈值升高，精确率升、召回率降", fontsize=8.5)
    ax = axes[1]
    auc, fpr, tpr = roc_auc(yte, p)
    ax.plot(fpr, tpr, color=C["red"], label=f"ROC，AUC = {auc:.3f}"); ax.plot([0, 1], [0, 1], "--", color=C["gray"], lw=0.8, label="随机（AUC 0.5）")
    for t in (0.3, 0.5, 0.7):
        tp, fp, fn, tn = confusion(yte, (p >= t).astype(int)); ax.plot(fp / (fp + tn), tp / (tp + fn), "o", color=C["blue"])
    ax.set_xlabel("假正率 FP / (FP + TN)"); ax.set_ylabel("召回率 TP / (TP + FN)"); ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_title("ROC：把阈值从 1 扫到 0 连成的线", fontsize=8.5)
    save(fig, "10-threshold-sweep-and-roc")
    print()


# ---------------- 2. 类别不平衡：准确率会骗人 ----------------
def exp_imbalance():
    print("=== 2. 正例只占 3%：准确率 97% 的'全判负'分类器 ===")
    Xtr, Xte, ytr, yte = data(weights=(0.97, 0.03), seed=1)
    dummy_acc = (yte == 0).mean()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    _tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"  测试集 {len(yte)} 条，正例 {yte.sum()} 条；全判负的准确率 {dummy_acc:.3f}")
    print(f"  逻辑回归阈值 0.5：准确率 {(pred == yte).mean():.3f}，但召回率只有 {tp/(tp+fn):.3f}（漏掉 {fn}/{tp+fn} 个正例）；AUC {roc_auc_score(yte, p):.3f}")
    pred2 = (p >= 0.1).astype(int)
    _tn, fp, fn, tp = confusion_matrix(yte, pred2).ravel()
    print(f"  阈值降到 0.1：召回 {tp/(tp+fn):.3f}、精确率 {tp/(tp+fp):.3f}、准确率 {(pred2 == yte).mean():.3f}——不平衡时看 P/R/AUC，不看准确率")
    print()


# ---------------- 3. 校准 ----------------
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in pairwise(edges):
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
    # 一个校准明显不好的模型：极端化的分数（模拟 RLHF 后「学会自信」），再用 Platt / isotonic 校准
    from sklearn.isotonic import IsotonicRegression
    p_lr = models["逻辑回归"].predict_proba(Xte)[:, 1]
    logit = np.log(p_lr / (1 - p_lr)) * 3.0                                        # 把 logit 放大 3 倍：排序不变、概率过度自信
    p_over = 1 / (1 + np.exp(-logit))
    half = len(yte) // 2                                                            # 前一半当校准集、后一半评估
    platt = LogisticRegression().fit(logit[:half, None], yte[:half])                # Platt：在分数上再套一个逻辑回归
    p_platt = platt.predict_proba(logit[half:, None])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_over[:half], yte[:half])  # isotonic：单调的分段常数映射
    p_iso = iso.predict(p_over[half:])
    print(f"  {'过度自信的模型':<10} AUC {roc_auc_score(yte[half:], p_over[half:]):.3f}  ECE {ece(yte[half:], p_over[half:]):.3f}（AUC 一点没变，概率却不可信）")
    print(f"  {'  Platt 校准后':<10} AUC {roc_auc_score(yte[half:], p_platt):.3f}  ECE {ece(yte[half:], p_platt):.3f}")
    print(f"  {'  isotonic 校准后':<10} AUC {roc_auc_score(yte[half:], p_iso):.3f}  ECE {ece(yte[half:], p_iso):.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot([0, 1], [0, 1], "--", color=C["gray"], lw=0.8, label="完美校准")
    for pp, col, name in ((p_over[half:], C["red"], "过度自信"), (p_platt, C["blue"], "Platt 校准后"), (p_iso, C["green"], "isotonic 校准后")):
        frac, mean_p = calibration_curve(yte[half:], pp, n_bins=10)
        ax.plot(mean_p, frac, "o-", ms=3, color=col, label=f"{name}：ECE {ece(yte[half:], pp):.3f}")
    ax.set_xlabel("预测概率（分 10 档取均值）"); ax.set_ylabel("该档的实际正例率"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("可靠性图：过度自信的模型偏离对角线（低估小概率、高估大概率），校准后贴回去", fontsize=8.5)
    save(fig, "10-calibration-platt-isotonic")
    print("  逻辑回归天然校准好（loss 就是对数似然）；树模型 / SVM 的分数排序对但概率不准，要 Platt / isotonic 校准；AUC 不看概率，校准要单独查")
    print("  LLM 上：RLHF 之后模型对自己答案的置信度校准变差（学会了'自信'）；奖励模型的 σ(r_w − r_l) 应等于人类偏好比例")
    print()


# ---------------- 3b. Cohen's κ 与 judge 的位置偏差 ----------------
def cohen_kappa(a, b):
    po = (a == b).mean()                                                            # ① 观察到的一致率
    pe = sum((a == c).mean() * (b == c).mean() for c in np.unique(np.r_[a, b]))     # ② 随机一致的期望（两人各自的边际分布相乘）
    return (po - pe) / (1 - pe)                                                     # ③ 扣掉随机一致再归一化


def exp_kappa():
    print("=== 3b. judge 的一致性：Cohen's κ；位置偏差怎么量 ===")
    r = np.random.default_rng(0)
    n = 1000
    human = (r.random(n) < 0.5).astype(int)                                          # 人类标注：A 好 (1) 还是 B 好 (0)
    # 一个 judge：80% 跟人一致，其余随机
    judge = np.where(r.random(n) < 0.6, human, (r.random(n) < 0.5).astype(int))
    print(f"  二选一、人类偏好各半：judge 与人一致率 {(judge == human).mean():.3f}，κ = {cohen_kappa(human, judge):.3f}（随机猜也有 50% 一致，κ 把它扣掉）")
    # 类别不平衡时一致率会骗人
    human2 = (r.random(n) < 0.9).astype(int)
    lazy = np.ones(n, int)
    print(f"  若 90% 的题人类都选 A，一个永远选 A 的 judge：一致率 {(lazy == human2).mean():.3f}，κ = {cohen_kappa(human2, lazy):.3f}——一致率高、κ 为零")
    # 位置偏差：judge 有 20% 概率无脑选先出现的那个
    def biased_judge(first_is_A):
        base = np.where(r.random(n) < 0.6, human, (r.random(n) < 0.5).astype(int))
        lazy_first = r.random(n) < 0.2
        return np.where(lazy_first, first_is_A.astype(int), base)
    j_ab = biased_judge(np.ones(n, bool))                                            # A 先出现
    j_ba = biased_judge(np.zeros(n, bool))                                           # B 先出现
    print(f"  位置偏差：A 排前面时 judge 选 A 的比例 {j_ab.mean():.3f}，B 排前面时 {j_ba.mean():.3f}——同一批题，只换顺序，差 {j_ab.mean() - j_ba.mean():+.3f}")
    print(f"    两次判断不一致的题占 {(j_ab != j_ba).mean():.1%}；对换顺序各评一次、只信两次一致的（或取平均），位置偏差就消掉了：一致题上与人的一致率 {(j_ab[j_ab == j_ba] == human[j_ab == j_ba]).mean():.3f}")
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


# ---------------- 4b. 一个分数的置信区间：公式与 bootstrap ----------------
def exp_bootstrap():
    print("=== 4b. benchmark 分数的置信区间：n 道题、正确率 p，标准误 √(p(1−p)/n) ===")
    for name, n, p in (("MMLU", 14042, 0.70), ("GSM8K", 1319, 0.85), ("一个 100 题的私有集", 100, 0.70)):
        se = np.sqrt(p * (1 - p) / n)
        print(f"  {name:<14} n = {n:>6}, p = {p:.2f} → 标准误 {se*100:.2f} 个点，95% 区间 ±{1.96*se*100:.1f} 个点")
    r = np.random.default_rng(0)
    n = 500
    correct = (r.random(n) < 0.65).astype(int)
    boots = np.array([correct[r.integers(0, n, n)].mean() for _ in range(5000)])   # 有放回重抽 5000 次
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  bootstrap：500 题正确率 {correct.mean():.3f}，重抽 5000 次的 2.5%–97.5% 分位 [{lo:.3f}, {hi:.3f}]；公式 ±{1.96*np.sqrt(correct.mean()*(1-correct.mean())/n):.3f}——两者一致")
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.hist(boots, bins=40, color=C["blue"], alpha=0.8)
    ax.axvline(lo, color=C["red"], ls="--"); ax.axvline(hi, color=C["red"], ls="--", label=f"95% 区间 [{lo:.3f}, {hi:.3f}]")
    ax.axvline(correct.mean(), color="k", lw=1, label=f"观察到的正确率 {correct.mean():.3f}")
    ax.set_xlabel("重抽样本的正确率"); ax.set_ylabel("次数（共 5000 次）"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("bootstrap：把 500 题有放回地重抽 5000 次，看正确率抖多大", fontsize=8.5)
    save(fig, "10-bootstrap-ci")
    print("  两个模型差 2 个点、各自 ±4 个点的区间——差异在噪声里；先看区间宽度再谈领先")
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
    n_sig_arr = np.array(n_sig)
    print(f"  {trials} 次模拟：两个完全相同的模型，5% 显著性水平下")
    print(f"    平均 {n_sig_arr.mean():.2f} 个 benchmark 上'显著'不同（期望 20 × 5% = 1）；至少一个显著的概率 {at_least_one/trials*100:.0f}%")
    print(f"    '20 个里领先 12 个'：随机情况下期望领先 10 个，标准差 {np.sqrt(20*0.25):.1f}——12 个不算什么")
    print("  报多个 benchmark 时先问：随机情况下期望几个显著、几个领先")
    print()


# ---------------- 7. AUC 遇到并列分数 ----------------
def roc_auc_ties(y, p):
    """并列分数一起放行：按不同的分数值分组，每组整体算一个 ROC 点（与 sklearn 一致）。"""
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    last = np.r_[np.where(np.diff(p) != 0)[0], len(p) - 1]        # 每个不同分数的最后一个位置
    tpr = np.r_[0, np.cumsum(y)[last] / y.sum()]
    fpr = np.r_[0, np.cumsum(1 - y)[last] / (1 - y).sum()]
    return float(np.trapezoid(tpr, fpr))


def auc_pairs(y, p):
    """概率定义：正例分数 > 负例算 1，相等算 0.5。"""
    pos, neg = p[y == 1], p[y == 0]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq), float(gt), float(eq)


def exp_ties():
    print("=== 7. AUC 遇到并列分数：一正一负同分算半对；逐个放行的手写版会随输入顺序变 ===")
    y = np.array([1, 1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.8, 0.5, 0.5, 0.2])
    auc, gt, eq = auc_pairs(y, p)
    print(f"  y = {y.tolist()}, p = {p.tolist()}：3 正 × 3 负 = 9 对，正例分数更高 {round(gt * 9)} 对、同分 {round(eq * 9)} 对 → AUC = ({round(gt * 9)} + 0.5 × {round(eq * 9)}) / 9 = {auc:.4f}")
    rev = slice(None, None, -1)
    print(f"  手写逐个放行版：原顺序 {roc_auc(y, p)[0]:.4f}，把数组倒过来再算 {roc_auc(y[rev], p[rev])[0]:.4f}——同一组数据两个答案，因为同分的正负例谁先放行是任意的")
    print(f"  并列一起放行版 {roc_auc_ties(y, p):.4f}，倒序 {roc_auc_ties(y[rev], p[rev]):.4f}；sklearn roc_auc_score {roc_auc_score(y, p):.4f}")
    Xtr, Xte, ytr, yte = data(seed=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)[:, 1]
    for name, pp in (("连续概率", pr), ("四舍五入到 1 位小数", np.round(pr, 1)), ("四舍五入到 0.5", np.round(pr * 2) / 2)):
        n_unique = len(np.unique(pp))
        print(f"  {name:<12} 不同分数 {n_unique:>5} 个：手写逐个放行 {roc_auc(yte, pp)[0]:.4f} / 倒序 {roc_auc(yte[rev], pp[rev])[0]:.4f}，并列版 {roc_auc_ties(yte, pp):.4f}，sklearn {roc_auc_score(yte, pp):.4f}")
    print("  分数分档越粗并列越多，差异越大；LLM judge 给 1–5 分、分类器输出 0/1 时全是并列，必须用并列版（或等价的 Mann–Whitney U）")
    print()


# ---------------- 8. 一条完整流程：定阈值 → 校准 → 留出评估 → 成对比较 ----------------
def three_way_split(n=8000, seed=4):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=8, weights=[0.7, 0.3], flip_y=0.03, class_sep=0.9, random_state=seed)
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def expected_cost(y, pred, c_fn=5.0, c_fp=1.0):
    _tp, fp, fn, _tn = confusion(y, pred)
    return (c_fn * fn + c_fp * fp) / len(y)


def best_threshold(y, p, c_fn=5.0, c_fp=1.0):
    grid = np.arange(0.02, 0.99, 0.01)
    costs = [expected_cost(y, (p >= t).astype(int), c_fn, c_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def exp_protocol():
    print("=== 8. 完整流程：训练集训模型，验证集定阈值 + 拟合校准，留出测试集只评一次，再在测试集上成对比较 ===")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = three_way_split()
    print(f"  训练 {len(y_tr)} / 验证 {len(y_va)} / 测试 {len(y_te)}，正例率 {y_tr.mean():.2f}；业务代价：漏一个正例 5、误判一个负例 1")
    models = {"A 梯度提升": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0),
              "B 逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    scores = {}
    preds = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
        t_va, cost_va = best_threshold(y_va, p_va)                                # ① 阈值只在验证集上选
        t_te, cost_te_cheat = best_threshold(y_te, p_te)                          # 对照：偷看测试集选的阈值
        cost_te = expected_cost(y_te, (p_te >= t_va).astype(int))
        logit_va = np.log(p_va / (1 - p_va))
        logit_te = np.log(p_te / (1 - p_te))
        platt = LogisticRegression(C=1e6).fit(logit_va[:, None], y_va)         # ② 校准映射只在验证集上拟合
        p_cal = platt.predict_proba(logit_te[:, None])[:, 1]
        print(f"  {name}：验证集最优阈值 {t_va:.2f}（验证代价 {cost_va:.3f}/样本）→ 测试代价 {cost_te:.3f}；"
              f"若在测试集上挑阈值得 {t_te:.2f}、代价 {cost_te_cheat:.3f}——偷看测试集的数字更漂亮但不能报")
        print(f"      留出评估：AUC {roc_auc_score(y_te, p_te):.3f}，ECE 校准前 {ece(y_te, p_te):.3f} → Platt 后 {ece(y_te, p_cal):.3f}；阈值 0.5 的代价 {expected_cost(y_te, (p_te >= 0.5).astype(int)):.3f}（默认阈值不是业务阈值）")
        scores[name] = p_te
        preds[name] = (p_te >= t_va).astype(int)
    # 校准映射成立的条件：P(y | 分数) 在校准集与应用集相同。把测试集正例率从 30% 压到 10%，同一张映射就失效
    pA = scores["A 梯度提升"]
    rng = np.random.default_rng(0)
    pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    keep = np.r_[rng.choice(pos, int(len(neg) / 9), replace=False), neg]
    m = models["A 梯度提升"]
    logit_va = np.log(m.predict_proba(X_va)[:, 1] / (1 - m.predict_proba(X_va)[:, 1]))
    platt = LogisticRegression(C=1e6).fit(logit_va[:, None], y_va)
    p_shift = platt.predict_proba(np.log(pA[keep] / (1 - pA[keep]))[:, None])[:, 1]
    print(f"  校准的前提：把测试集正例率从 {y_te.mean():.2f} 抽到 {y_te[keep].mean():.2f}（分数分布不变、先验变了），验证集上拟合的 Platt 映射在新分布上 ECE {ece(y_te[keep], p_shift):.3f}，"
          f"AUC {roc_auc_score(y_te[keep], p_shift):.3f} 不受影响——校准修的是 P(y | 分数)，先验一变映射就得重拟合")
    # ④ 成对比较：同一测试集上，A 与 B 的决策与 AUC
    a, b = preds["A 梯度提升"], preds["B 逻辑回归"]
    n01 = int(((a == y_te) & (b != y_te)).sum())
    n10 = int(((a != y_te) & (b == y_te)).sum())
    z = (n01 - n10) / np.sqrt(max(1, n01 + n10))
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(y_te), len(y_te))
        if y_te[idx].min() == y_te[idx].max():
            continue
        boots.append(roc_auc_score(y_te[idx], pA[idx]) - roc_auc_score(y_te[idx], scores["B 逻辑回归"][idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    d_auc = roc_auc_score(y_te, pA) - roc_auc_score(y_te, scores["B 逻辑回归"])
    print(f"  成对比较（同一测试集）：各用自己的验证集阈值，A 对 B 错 {n01} 条、A 错 B 对 {n10} 条，McNemar z = {z:.2f}（|z| > 1.96 为显著）；"
          f"AUC 差 {d_auc:+.4f}，配对 bootstrap 95% 区间 [{lo:+.4f}, {hi:+.4f}]")
    print("  区间不含 0 才能说 A 更好；若含 0，只说明这份测试集分不出两者，不是证明两者相同——区间宽度告诉你能排除多大的差距")
    print()


# ---------------- 9. 交叉验证各折不是独立重复实验 ----------------
def exp_cvdep():
    print("=== 9. 交叉验证各折不是独立重复实验：5 折均值 ± 折间标准差/√5 低估了真实抖动 ===")
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6, random_state=3)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    k = 5
    print(f"  {k} 折里任意两折的训练集共享 {(k - 2) / (k - 1):.0%} 的样本（(k−2)/(k−1)），验证集互斥但模型高度相关")
    cv0 = cross_val_score(clf, X, y, cv=KFold(k, shuffle=True, random_state=0))
    naive_se = cv0.std(ddof=1) / np.sqrt(k)
    means = np.array([cross_val_score(clf, X, y, cv=KFold(k, shuffle=True, random_state=s)).mean() for s in range(200)])
    print(f"  一次 5 折：各折 {np.round(cv0, 3).tolist()}，均值 {cv0.mean():.3f}，若当 5 次独立实验：标准误 {naive_se:.3f}")
    print(f"  换 200 种划分重跑 5 折：均值的标准差 {means.std(ddof=1):.3f}，范围 [{means.min():.3f}, {means.max():.3f}]——同一份 300 条数据，仅换划分就抖这么多")
    big_X, big_y = make_classification(n_samples=60000, n_features=20, n_informative=6, random_state=3)
    rng = np.random.default_rng(0)
    fresh_list = []
    for _ in range(200):
        idx = rng.choice(len(big_y), 300, replace=False)
        fresh_list.append(cross_val_score(clf, big_X[idx], big_y[idx], cv=KFold(k, shuffle=True, random_state=0)).mean())
    fresh = np.array(fresh_list)
    print(f"  再从同一分布抽 200 份新的 300 条各做一次 5 折：均值的标准差 {fresh.std(ddof=1):.3f}——这才是「换一份数据会差多少」，是折间标准差/√5（{naive_se:.3f}）的 {fresh.std(ddof=1) / naive_se:.1f} 倍")
    print("  折间标准差只反映同一份数据内部的划分噪声；拿它做 t 检验比较两个模型，显著性会虚高。要么报区间宽度时说明口径，要么用重复 CV / 新数据")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    for ax, arr, title, col in ((axes[0], means, "同一份 300 条，换 200 种划分", C["blue"]), (axes[1], fresh, "换 200 份新的 300 条", C["orange"])):
        ax.hist(arr, bins=25, color=col, alpha=0.8)
        ax.axvline(cv0.mean() - 1.96 * naive_se, color=C["red"], ls="--", lw=0.8)
        ax.axvline(cv0.mean() + 1.96 * naive_se, color=C["red"], ls="--", lw=0.8, label=f"一次 5 折当独立实验的 ±1.96 SE（宽 {2 * 1.96 * naive_se:.3f}）")
        ax.set_title(f"{title}：标准差 {arr.std(ddof=1):.3f}", fontsize=8.5)
        ax.set_xlabel("5 折均值准确率")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("次数（共 200 次）")
    save(fig, "10-cv-folds-not-independent")
    print()


# ---------------- 10. 相关样本上的不确定性：同一模板的多道题不是独立的 ----------------
def exp_cluster():
    print("=== 10. 相关样本：50 个模板 × 10 道改写题 = 500 题，同模板的题一起对一起错，公式标准误过于乐观 ===")
    rng = np.random.default_rng(0)
    n_tmpl, per, sd_tmpl = 50, 10, 2.0
    n = n_tmpl * per
    shift = np.log(0.65 / 0.35)
    true_p = float((1 / (1 + np.exp(-(rng.normal(0, sd_tmpl, 1_000_000) + shift)))).mean())   # 总体正确率：对模板难度分布取期望
    sims, cover_naive, cover_cluster, se_naive_all, se_cluster_all = 500, 0, 0, [], []
    for _ in range(sims):
        q = 1 / (1 + np.exp(-(rng.normal(0, sd_tmpl, n_tmpl) + shift)))      # 每个模板一个难度 → 该模板下每题的答对概率
        correct = (rng.random((n_tmpl, per)) < q[:, None]).astype(int)
        p_hat = correct.mean()
        se_naive = np.sqrt(p_hat * (1 - p_hat) / n)
        tmpl_means = correct.mean(1)
        boots = [tmpl_means[rng.integers(0, n_tmpl, n_tmpl)].mean() for _ in range(300)]  # 按模板重抽：整簇一起进出
        se_cluster = float(np.std(boots, ddof=1))
        se_naive_all.append(se_naive)
        se_cluster_all.append(se_cluster)
        cover_naive += abs(p_hat - true_p) <= 1.96 * se_naive
        cover_cluster += abs(p_hat - true_p) <= 1.96 * se_cluster
    between = correct.mean(1).var(ddof=1)
    within = correct.var(axis=1, ddof=1).mean()
    icc = between / (between + within)
    print(f"  总体正确率 {true_p:.3f}（对模板难度分布取期望）；最后一次抽样的组内相关 ICC ≈ {icc:.2f}：设计效应 1 + (10 − 1) × ICC ≈ {1 + 9 * icc:.1f}，500 题只相当于约 {n / (1 + 9 * icc):.0f} 道独立题")
    print(f"  公式标准误 √(p(1−p)/n) 平均 {np.mean(se_naive_all):.3f}，按模板整簇重抽的 bootstrap 标准误平均 {np.mean(se_cluster_all):.3f}")
    print(f"  {sims} 次模拟里 95% 区间盖住真值的比例：公式 {cover_naive / sims:.0%}，簇 bootstrap {cover_cluster / sims:.0%}——公式区间名义 95%、实际 {cover_naive / sims:.0%}；簇 bootstrap 把名义值找回来")
    print("  同一来源的样本（同模板、同篇文档的多段、同一用户的多轮）要按来源重抽或用聚类标准误；独立性是公式的前提，不是默认成立的事实")
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.hist(se_naive_all, bins=30, color=C["gray"], alpha=0.8, label=f"公式 √(p(1−p)/n)，均值 {np.mean(se_naive_all):.3f}，95% 区间盖住真值 {cover_naive / sims:.0%}")
    ax.hist(se_cluster_all, bins=30, color=C["blue"], alpha=0.7, label=f"按模板整簇重抽的 bootstrap，均值 {np.mean(se_cluster_all):.3f}，盖住真值 {cover_cluster / sims:.0%}")
    ax.set_xlabel("500 题正确率的标准误（500 次模拟）")
    ax.set_ylabel("次数")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("50 个模板 × 10 道改写题：同模板的题一起对一起错，公式标准误偏小", fontsize=8.5)
    save(fig, "10-clustered-se")
    print()


# ---------------- 11. 位置对换消不掉所有 judge 偏差 ----------------
def exp_judgebias():
    print("=== 11. 位置对换：能抵消位置偏好，消不掉长度偏好——模拟一个偏爱靠前 + 偏爱更长答案的 judge ===")
    rng = np.random.default_rng(1)
    n = 2000
    human = rng.integers(0, 2, n)                                               # 人类偏好：0 = 答案 X，1 = 答案 Y
    longer = rng.integers(0, 2, n)                                              # 哪个答案更长，与人类偏好独立
    quality = np.where(human == 1, 1.0, -1.0)                                   # Y 比 X 好记 +1

    def judge(first_is_y, pos_bias, len_bias):
        """judge 看到两个答案（first_is_y 决定谁在前），返回 1 = 选 Y。"""
        s = 1.2 * quality + pos_bias * np.where(first_is_y == 1, 1.0, -1.0) + len_bias * np.where(longer == 1, 1.0, -1.0)
        return (1 / (1 + np.exp(-s)) > rng.random(n)).astype(int)

    for pos_bias, len_bias, label in ((1.5, 0.0, "只有位置偏好"), (1.5, 1.0, "位置 + 长度偏好")):
        single = judge(np.ones(n, dtype=int), pos_bias, len_bias)               # 单次：Y 总在前
        order1 = judge(np.ones(n, dtype=int), pos_bias, len_bias)
        order2 = judge(np.zeros(n, dtype=int), pos_bias, len_bias)
        consistent = order1 == order2
        swapped = order1[consistent]
        agree_single = (single == human).mean()
        agree_swap = (swapped == human[consistent]).mean()
        long_pref_single = (single == longer).mean()
        long_pref_swap = (swapped == longer[consistent]).mean()
        print(f"  {label}：单次（Y 在前）选 Y 的比例 {single.mean():.2f}、与人类一致 {agree_single:.3f}；"
              f"对换两次一致的 {consistent.mean():.0%}，一致部分与人类一致 {agree_swap:.3f}，其中选了更长答案的比例 {long_pref_swap:.2f}（单次 {long_pref_single:.2f}）")
    print("  对换后「Y 在前」的优势没了（选 Y 比例回到 0.5 附近），但更长答案仍被多选——长度、自我偏好、评分标准措辞这些与位置无关的偏差，对换一次也测不出来；"
          "要分别设计对照（等长答案、匿名模型、改写 rubric）才量得到")
    print()


EXPS = {"metrics": exp_metrics, "threshold": exp_threshold, "imbalance": exp_imbalance, "calibration": exp_calibration, "kappa": exp_kappa,
        "cv": exp_cv, "bootstrap": exp_bootstrap, "paired": exp_paired, "multiple": exp_multiple,
        "ties": exp_ties, "protocol": exp_protocol, "cvdep": exp_cvdep, "cluster": exp_cluster, "judgebias": exp_judgebias}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
