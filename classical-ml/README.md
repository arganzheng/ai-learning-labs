# LLM 时代的经典机器学习：只讲它在哪里重现 — 配套脚本

博客系列：[总纲](https://arganzheng.life/classical-machine-learning-in-the-llm-era.html)（算法地图 L2）。六篇文章各配一个脚本，文中引用的数字由这些脚本跑出来。全部 CPU、离线（03 的 20newsgroups 子实验需要联网，没网自动跳过）。

| 脚本 | 文章 | 依赖 | 运行时间 |
|---|---|---|---|
| `01_learning_and_generalization.py` | [什么是学习](https://arganzheng.life/what-is-learning-splits-generalization-and-bias-variance.html) | numpy、scikit-learn | 十几秒 |
| `02_linear_and_logistic_regression.py` | [线性回归与逻辑回归](https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html) | 同上 | 几秒 |
| `03_classifiers.py` | [分类器一家](https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html) | 同上（20newsgroups 需联网） | 十几秒 |
| `04_unsupervised.py` | [无监督](https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html) | 同上 + matplotlib | 十几秒；图存到 `out/digits_pca.png` |
| `05_minhash_lsh.py` | [去重：MinHash 与 LSH](https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html) | 纯 numpy | 十几秒 |
| `06_evaluation.py` | [评估](https://arganzheng.life/evaluation-from-confusion-matrix-to-judge-agreement.html) | numpy、scikit-learn | 十几秒 |

## 运行

```bash
pip install -r ../requirements.txt
python 01_learning_and_generalization.py          # 全部子实验
python 05_minhash_lsh.py scurve                   # 只跑一个（子实验名见每个脚本头部）
```

数据全部是 scikit-learn 自带的小数据集（乳腺癌、手写数字）或合成数据（`make_classification`、`make_blobs`、`make_moons`、自造的近重复文本），不用下载。`expected/` 是作者机器上的完整输出；不同版本下小数末位会有出入，结论应一致。
