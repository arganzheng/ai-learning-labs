"""分类器一家（经典 ML 03）第三章：用 NumPy 手写 KNN、高斯朴素贝叶斯、决策树，与 scikit-learn 在同一份数据上对数。
https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html
"""
import numpy as np, time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=5000, n_features=20, n_informative=8, n_redundant=4, flip_y=0.05, class_sep=0.8, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

def knn_predict(Xtrain, ytrain, Xtest, k=15):
    d2 = ((Xtest[:, None, :] - Xtrain[None, :, :]) ** 2).sum(-1)   # [n_test, n_train] 每对样本的距离平方
    idx = np.argpartition(d2, k, axis=1)[:, :k]                    # 每行最近的 k 个下标
    votes = ytrain[idx]                                             # [n_test, k] 邻居的标签
    return (votes.mean(1) > 0.5).astype(int)                        # 多数票

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.array([(y == c).mean() for c in self.classes])            # P(c)
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])             # 每类每特征的均值
        self.var = np.array([X[y == c].var(0) + 1e-9 for c in self.classes])      # 方差
        return self
    def predict(self, X):
        # log P(c) + Σ_j log N(x_j; μ_cj, σ²_cj)：特征独立 → 对数概率相加
        ll = -0.5 * (np.log(2 * np.pi * self.var[:, None, :]) + (X[None] - self.mu[:, None, :]) ** 2 / self.var[:, None, :]).sum(-1)
        return self.classes[(np.log(self.prior)[:, None] + ll).argmax(0)]

def gini(y):
    p = y.mean(); return 2 * p * (1 - p)
def best_split(X, y):
    best = (gini(y), None, None)
    for j in range(X.shape[1]):
        for thr in np.percentile(X[:, j], np.arange(5, 100, 5)):
            left = X[:, j] <= thr
            if left.sum() == 0 or left.sum() == len(y): continue
            g = (left.mean() * gini(y[left]) + (1 - left.mean()) * gini(y[~left]))
            if g < best[0]: best = (g, j, thr)
    return best[1], best[2]
def build_tree(X, y, depth, max_depth):
    if depth == max_depth or len(np.unique(y)) == 1: return int(y.mean() > 0.5)
    j, thr = best_split(X, y)
    if j is None: return int(y.mean() > 0.5)
    left = X[:, j] <= thr
    return (j, thr, build_tree(X[left], y[left], depth + 1, max_depth), build_tree(X[~left], y[~left], depth + 1, max_depth))
def tree_predict(node, x):
    while isinstance(node, tuple):
        j, thr, l, r = node; node = l if x[j] <= thr else r
    return node

t0=time.time(); p=knn_predict(Xtr_s,ytr,Xte_s); print(f"KNN 手写 {(p==yte).mean():.3f}  {time.time()-t0:.2f}s  sklearn {KNeighborsClassifier(15).fit(Xtr_s,ytr).score(Xte_s,yte):.3f}")
nb=GaussianNaiveBayes().fit(Xtr,ytr); print(f"NB  手写 {(nb.predict(Xte)==yte).mean():.3f}  sklearn {GaussianNB().fit(Xtr,ytr).score(Xte,yte):.3f}")
for d in (3,6,None):
    t0=time.time(); tree=build_tree(Xtr,ytr,0,d if d else 100); pt=np.array([tree_predict(tree,x) for x in Xte]); ptr=np.array([tree_predict(tree,x) for x in Xtr])
    sk=DecisionTreeClassifier(max_depth=d,random_state=0).fit(Xtr,ytr)
    print(f"树 depth={d} 手写 训练 {(ptr==ytr).mean():.3f} 测试 {(pt==yte).mean():.3f} {time.time()-t0:.1f}s  sklearn 训练 {sk.score(Xtr,ytr):.3f} 测试 {sk.score(Xte,yte):.3f}")
