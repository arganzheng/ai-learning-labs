# LLM 时代的经典机器学习：只讲它在哪里重现 — 配套脚本

博客系列：[总纲](https://arganzheng.life/classical-machine-learning-in-the-llm-era.html)（算法地图 L2）。十篇正文各配一个脚本，**文中每一个数字、每一张图都由这些脚本跑出来**（图输出到 `out/*.svg`，博客里的 `img/in-post/classical-ml-*.svg` 就是它们）。全部 CPU、离线；第 07 / 08 篇的句向量与权重谱用本地缓存的 Qwen2.5-0.5B（`HF_HUB_OFFLINE=1`，没有缓存时第一次运行会下载约 1 GB）。

| 脚本 | 文章 | 子实验 | 运行时间 |
|---|---|---|---|
| `01_learning_and_generalization.py` | [什么是学习](https://arganzheng.life/what-is-learning-splits-generalization-and-bias-variance.html) | fit learncurve split groups contamination leak biasvar | 一分钟 |
| `02_linear_regression.py` | [线性回归](https://arganzheng.life/linear-regression-least-squares-ridge-and-lasso.html) | line surface gd scaling collinear paths geometry wd smooth robust | 十几秒 |
| `03_logistic_regression_and_reward_model.py` | [逻辑回归与奖励模型](https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html) | sigmoid gradient boundary cancer softmax reward noise | 十几秒 |
| `04_naive_bayes_knn_and_trees.py` | [三个基础分类器](https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html) | compare grid bayes nb knn curse tree depth | 一分钟 |
| `05_svm_and_kernels.py` | [SVM 与核方法](https://arganzheng.life/svm-and-kernel-methods.html) | margin hinge primal softc kernel rbf attention scale | 半分钟 |
| `06_ensembles_and_gradient_boosting.py` | [集成](https://arganzheng.life/ensembles-random-forest-and-gradient-boosting.html) | bagging forest boost_steps boost_hand lr importance tabular budget | 一两分钟 |
| `07_clustering.py` | [聚类](https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html) | iterate kmeans choose_k init dbscan hierarchical corpus | 一分钟（corpus 首次加载模型） |
| `08_dimensionality_reduction.py` | [降维](https://arganzheng.life/dimensionality-reduction-pca-svd-tsne-and-umap.html) | geometry pca reconstruct svd tsne anisotropy spectrum | 一分钟 |
| `09_minhash_lsh.py` | [去重：MinHash 与 LSH](https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html) | tiny estimate error scurve dedup semantic | 一分钟 |
| `10_evaluation.py` | [评估](https://arganzheng.life/evaluation-from-confusion-matrix-to-judge-agreement.html) | metrics threshold imbalance calibration kappa cv bootstrap paired multiple | 半分钟 |

公共文件：`_plot.py`（中文字体、博客列宽、SVG 输出）、`_sentences.py`（07 / 08 / 09 共用的 78 句小语料与 Qwen2.5-0.5B 句向量，缓存在 `out/sentence_embeddings.npz`）。

## 运行

```bash
pip install -r ../requirements.txt
python 01_learning_and_generalization.py          # 全部子实验
python 09_minhash_lsh.py scurve                   # 只跑一个（子实验名见上表或脚本头部）
PLOT_PNG=/tmp/plots python 02_linear_regression.py   # 另存一份 png 方便自己看
```

数据全部是 scikit-learn 自带的小数据集（乳腺癌、手写数字）或合成数据（`make_classification`、`make_blobs`、`make_moons`、`make_circles`、自造的近重复文本），不用下载。`expected/` 是作者机器上的完整输出；不同版本下小数末位与耗时会有出入，结论应一致。
