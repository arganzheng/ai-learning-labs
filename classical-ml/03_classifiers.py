"""分类器一家（经典 ML 03）：朴素贝叶斯、KNN、SVM、决策树、随机森林、梯度提升在同一份数据上比一比；数据质量分类器的算力账。
https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html

    python 03_classifiers.py            # 全部：compare importance budget
"""
import sys
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer, make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ---------------- 1. 六个分类器在表格数据上 ----------------
def exp_compare():
    print("=== 1. 六个分类器：合成表格数据（5000 样本、20 特征、其中 8 个有用、含非线性）===")
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, n_redundant=4, flip_y=0.05, class_sep=0.8, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    models = {
        "逻辑回归":     make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "朴素贝叶斯":   GaussianNB(),
        "KNN (k=15)":  make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
        "SVM (RBF)":   make_pipeline(StandardScaler(), SVC(C=1.0)),
        "决策树":       DecisionTreeClassifier(max_depth=None, random_state=0),
        "随机森林":     RandomForestClassifier(300, random_state=0, n_jobs=-1),
        "梯度提升":     GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0),
    }
    print(f"  {'模型':<14} {'训练准确率':>10} {'测试准确率':>10} {'训练耗时':>9}   一句话")
    notes = {"逻辑回归": "线性边界：有非线性就吃亏", "朴素贝叶斯": "假设特征独立：冗余特征让它更差", "KNN (k=15)": "不训练；预测时找 15 个邻居投票",
             "SVM (RBF)": "核把线性边界变弯", "决策树": "训练 100%、测试掉一截：过拟合的教科书样子", "随机森林": "很多棵树的 bagging：降方差",
             "梯度提升": "逐棵树拟合残差：表格数据的默认最强"}
    for name, m in models.items():
        t0 = time.time(); m.fit(Xtr, ytr); dt = time.time() - t0
        print(f"  {name:<14} {m.score(Xtr, ytr):>10.3f} {m.score(Xte, yte):>10.3f} {dt:>8.2f}s   {notes[name]}")
    print()


# ---------------- 2. 树模型的特征重要性 ----------------
def exp_importance():
    print("=== 2. 梯度提升的特征重要性（乳腺癌 30 个特征）===")
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(Xtr, ytr)
    print(f"  测试准确率 {gb.score(Xte, yte):.3f}")
    order = np.argsort(-gb.feature_importances_)[:6]
    for i in order:
        print(f"    {data.feature_names[i]:<26} {gb.feature_importances_[i]:.3f}")
    print(f"  前 6 个特征占重要性 {gb.feature_importances_[order].sum():.0%}；数据质量打分里这张表告诉你'困惑度、长度、符号比例'哪个在起作用")
    print()


# ---------------- 3. 文本分类器 + 给 15T token 打分的算力账 ----------------
def exp_budget():
    print("=== 3. 文本分类器（TF-IDF + 线性模型）与'给 15T token 打分'的算力账 ===")
    try:
        cats = ["sci.space", "rec.sport.hockey", "talk.politics.misc", "comp.graphics"]
        train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
        test = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        Xtr, Xte = vec.fit_transform(train.data), vec.transform(test.data)
        for name, m in [("MultinomialNB", MultinomialNB()), ("逻辑回归", LogisticRegression(max_iter=3000, C=5))]:
            t0 = time.time(); m.fit(Xtr, train.target)
            print(f"  20newsgroups 4 类 {Xtr.shape[0]} 篇 → {Xtr.shape[1]} 个 1/2-gram 特征；{name:<14} 测试准确率 {m.score(Xte, test.target):.3f}  训练 {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  （20newsgroups 需要联网下载，跳过：{type(e).__name__}）")
    print("  给 15T token 打分的账（L0 第一篇：一个 token 前向 ≈ 2N FLOP）：")
    D = 15e12
    train_8b = 6 * 8e9 * D
    for name, N in [("8B LLM 逐段打分", 8e9), ("1B 小 LLM", 1e9), ("BERT 级 1 亿参数", 1e8), ("fastText / 线性模型 ~1M", 1e6)]:
        score = 2 * N * D
        print(f"    {name:<22} {score:.1e} FLOP = 训练 8B 模型 ({train_8b:.1e}) 的 {score/train_8b*100:6.2f}%")
    print("  → 两级做法：大模型标几十万段，训小分类器，小分类器过全部。不是效果更好，是只有它跑得起")
    print()


EXPS = {"compare": exp_compare, "importance": exp_importance, "budget": exp_budget}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
