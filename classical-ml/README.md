# LLM 时代的经典机器学习：只讲它在哪里重现 — 配套脚本

博客系列：[总纲](https://arganzheng.life/classical-machine-learning-in-the-llm-era.html)（算法地图 L2）。十篇正文各配一个脚本，**文中每一个数字、每一张图都由这些脚本跑出来**（图输出到 `out/*.svg`，博客里的 `img/in-post/classical-ml-*.svg` 就是它们）。全部 CPU、离线；第 07 / 08 篇的句向量与权重谱用本地缓存的 Qwen2.5-0.5B（`HF_HUB_OFFLINE=1`，没有缓存时第一次运行会下载约 1 GB）。

| 脚本 | 文章 | 子实验 | 运行时间 |
|---|---|---|---|
| `01_learning_and_generalization.py` | [什么是学习](https://arganzheng.life/what-is-learning-splits-generalization-and-bias-variance.html) | fit learncurve split groups contamination leak biasvar cv pipeline | 一分钟 |
| `02_linear_regression.py` | [线性回归](https://arganzheng.life/linear-regression-least-squares-ridge-and-lasso.html) | line surface gd scaling collinear paths geometry wd smooth robust step solvers soft align | 十几秒 |
| `03_logistic_regression_and_reward_model.py` | [逻辑回归与奖励模型](https://arganzheng.life/linear-and-logistic-regression-the-skeleton-of-reward-models.html) | sigmoid gradient boundary cancer softmax reward noise | 十几秒 |
| `04_naive_bayes_knn_and_trees.py` | [三个基础分类器](https://arganzheng.life/a-family-of-classifiers-from-naive-bayes-to-gradient-boosting.html) | compare grid bayes nb knn curse tree depth | 一分钟 |
| `05_svm_and_kernels.py` | [SVM 与核方法](https://arganzheng.life/svm-and-kernel-methods.html) | margin hinge primal softc kernel rbf attention scale dual cscale cost analogy | 一分钟 |
| `06_ensembles_and_gradient_boosting.py` | [集成](https://arganzheng.life/ensembles-random-forest-and-gradient-boosting.html) | bagging forest boost_steps boost_hand lr importance tabular budget variance logit early forest_cost | 四五分钟（variance 要训 50 份 × 1000 棵） |
| `07_clustering.py` | [聚类](https://arganzheng.life/unsupervised-learning-kmeans-pca-and-embedding-clusters.html) | iterate kmeans choose_k init dbscan hierarchical corpus edge expand linkage | 一分钟（corpus 首次加载模型） |
| `08_dimensionality_reduction.py` | [降维](https://arganzheng.life/dimensionality-reduction-pca-svd-tsne-and-umap.html) | geometry pca reconstruct svd tsne anisotropy spectrum scaling neighbors faces | 一分钟 |
| `09_minhash_lsh.py` | [去重：MinHash 与 LSH](https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html) | tiny estimate error scurve dedup semantic trace family retain | 一分钟（semantic 首次加载模型） |
| `10_evaluation.py` | [评估](https://arganzheng.life/evaluation-from-confusion-matrix-to-judge-agreement.html) | metrics threshold imbalance calibration kappa cv bootstrap paired multiple ties protocol cvdep cluster judgebias | 一两分钟 |

## 案例脚本（每篇一个真实数据集的端到端案例）

| 脚本 | 文章 | 案例 | 数据（首次运行自动下载到 `data/`） |
|---|---|---|---|
| `case_01_housing_split.py` | 01 什么是学习 | 同一个 KNN，随机划分 vs 按地区划分，RMSE 差 65% | 加州房价 openml #43939（1 MB） |
| `case_02_housing_regression.py` | 02 线性回归 | 从猜均值到 Ridge / Lasso 十步，每步误差降多少 | 同上 |
| `case_03_sms_spam.py` | 03 逻辑回归 | 垃圾短信识别：TF-IDF + 逻辑回归，阈值与权重 | UCI SMS Spam Collection（200 KB） |
| `case_04_nb_knn_tree.py` | 04 三个分类器 | `nb` 朴素贝叶斯做垃圾短信；`knn` KNN 做 MNIST（全量 2.95%）；`tree` 决策树做泰坦尼克（整棵画出 + boat 泄漏陷阱） | SMS Spam；MNIST（复用 ../deep-learning-foundations/data，12 MB）；Titanic openml #40945 |
| `case_05_svm_mnist.py` | 05 SVM | 重跑 LeCun 1998 的表：线性 / KNN / RBF-SVM（60k 全量 1.43%，约 4 分钟）；`grid` C×γ 网格；`cost` 训练集 2k → 20k 时支持向量与预测时间 | MNIST |
| `case_06_adult_income.py` | 06 集成 | 人口普查收入预测：逻辑回归 → 树 → 随机森林 → 梯度提升；permutation 重要性；学习率 × 轮数；早停的验证集 vs 测试集；森林棵数的收益与代价 | Adult openml #1590（4 MB） |
| `case_07_customer_segments.py` | 07 聚类 | `rfm` 54 万行交易 → RFM → K-Means 4 群画像；`colors` 照片颜色量化 | UCI Online Retail（23 MB xlsx，需 `openpyxl`）；sklearn 自带 china.jpg |
| `case_08_eigenfaces.py` | 08 降维 | Eigenfaces：平均脸 / 特征脸 / 重建 / PCA + 分类器认人 | Olivetti_Faces openml #41083（4 MB） |
| `case_09_wikitext_dedup.py` | 09 去重 | wikitext-2 14,813 段 + 注入 500 近重复：MinHash-LSH vs 暴力；重复对 → 连通分量 / 贪心删除清单 | HF `Salesforce/wikitext`（4 MB，需 `datasets`） |
| `case_10_bank_marketing.py` | 10 评估 | 银行营销：泄漏特征、AUC/AP、按成本定阈值、校准、5 折配对检验；训练 / 验证 / 测试三份走完定阈值 → 校准 → 留出评估 → 配对 bootstrap | bank-marketing openml #1461（1 MB） |

公共文件：`_data.py`（数据下载与缓存：openml / UCI），`_plot.py`（中文字体、博客列宽、SVG 输出）、`_sentences.py`（07 / 08 / 09 共用的 78 句小语料与 Qwen2.5-0.5B 句向量，缓存在 `out/sentence_embeddings.npz`）。

## 运行

```bash
pip install -r ../requirements.txt
python 01_learning_and_generalization.py          # 全部子实验
python 09_minhash_lsh.py scurve                   # 只跑一个（子实验名见上表或脚本头部）
PLOT_PNG=/tmp/plots python 02_linear_regression.py   # 另存一份 png 方便自己看
```

数据全部是 scikit-learn 自带的小数据集（乳腺癌、手写数字）或合成数据（`make_classification`、`make_blobs`、`make_moons`、`make_circles`、自造的近重复文本），不用下载。`expected/` 是作者机器上的完整输出；不同版本下小数末位与耗时会有出入，结论应一致。

## 07 / 08 深化实验

- 07：`edge` 验证空簇、提前停止与最终 labels/centers/inertia 一致性；`expand` 跟踪 DBSCAN 队列并逐点对照 sklearn；`linkage` 给五点合并记录与两组形状对照。
- 08：`scaling` 对照中心化、标准化、样本协方差白化（并与 sklearn 对数）；`neighbors` 控制同一输入、随机种子下的邻域参数；`faces` 固定人脸数据划分与维数，只切换白化。
- UMAP 可选：`pip install umap-learn==0.5.7`。未安装会明确跳过 UMAP；其他子实验仍可用。
- 新增记录由 NumPy 2.2.6、scikit-learn 1.7.2、umap-learn 0.5.7 实跑产生。旧 `expected/` 区段与文章执行输出保留；不同 BLAS/平台下 PCA 的符号对齐误差和 t-SNE 末位可能不同，不应为迎合旧日志手改输出。
- 邻域实验的二维 KNN 是先做全数据嵌入后的聚集度诊断，不是对未见样本的泛化评估。图每类固定抽 10 点，指标使用全部 1797 点。
- 07/08 全部子实验及 RFM、颜色量化、Eigenfaces 案例可在 CPU 运行；首次模型/数据下载及 UMAP 编译缓存不包含在原表的粗略时间里。

## 09 / 10 深化实验

- 09：`trace` 用 6 段文本、词级 2-gram、6 个真随机排列分 3 band × 2 行，逐步打印 shingle、签名矩阵、band 键、候选对与精确复核，并解释漏掉的对；`family` 对照独立随机排列的理论候选概率、仿射 hash `(a·x + b) mod p` 与只平移的强相关 hash（图 `09-lsh-theory-vs-hash-family.svg`）；`retain` 展示 A ≈ B ≈ C 而 A ≉ C 时连通分量与贪心两种保留策略的差别。`case_09` 末尾新增第 4 节：把判重对连成分量并对照两种删除策略。
- 10：`ties` 给并列分数下 AUC 的手算例子，对照逐个放行的旧手写版、并列一起放行版与 sklearn；`protocol` 走完训练 / 验证 / 测试三份数据的「定阈值 → 校准 → 留出评估 → 成对比较」，含偷看测试集选阈值的对照与先验变化后校准映射失效的例子；`cvdep` 用 200 次换划分与 200 份新数据对照折间标准差/√5；`cluster` 模拟 50 模板 × 10 改写题的簇相关，比较公式标准误与按簇重抽的覆盖率；`judgebias` 模拟同时有位置偏好和长度偏好的 judge，说明位置对换只抵消前者。`case_10` 末尾新增第 5 节：验证集定阈值与校准、测试集只评一次、配对 bootstrap 比较两个模型。
- 新增记录由 NumPy 2.2.6、scikit-learn 1.7.2、CPU 实跑产生，追加在 `expected/` 对应文件末尾；旧区段原样保留。`case_10` 旧区段的随机森林 / 梯度提升末位、按成本扫出的阈值与 5 折配对 p 值在不同平台 / 线程数下会有出入（本次运行 p = 0.030，旧记录 0.096），`case_09` 的耗时与机器相关，都不应为迎合旧日志手改输出。
- 09 的 `semantic` 子实验需要 `torch` + `transformers`（首次加载 Qwen2.5-0.5B，约 1 GB）；没有也可以只跑其余子实验。
